"""
Translation quality metrics module.

Design decisions:
- Wraps sacrebleu for BLEU/ChrF: industry-standard, reproducible tokenization
- Custom LengthRatio & BigramRepetition: lightweight, no external dependency
- Optional semantic metric (BERTScore) for meaning-level similarity
- All metrics follow the same interface: (predictions, references) → float
  → Easy to plug into evaluation pipeline and grid search

Why these metrics:
1. BLEU: Standard MT metric; captures n-gram precision + brevity penalty
2. ChrF: Character-level F-score; more robust to morphological variation
3. Length Ratio: Directly measures under/over-translation tendency
4. Bigram Repetition: Detects degenerate looping in beam search output
5. BERTScore: Semantic similarity using pretrained representations
"""

import logging
from typing import Callable, Dict, List, Optional

import sacrebleu
import torch

logger = logging.getLogger(__name__)

# 说明：统一指标接口，便于评估与搜索复用。
# Default metric set used when callers do not specify a list.
# Keep BERTScore opt-in to avoid heavy downloads in default runs.
DEFAULT_METRICS = ["bleu", "chrf", "length_ratio", "bigram_repetition"]


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """
    Corpus-level BLEU score using sacrebleu.

    Why sacrebleu over nltk.bleu_score:
    - Standardized tokenization (no manual tokenization variance)
    - Reproducible: same inputs → same score across environments
    - Handles edge cases (empty translations, single references)
    """
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return round(bleu.score / 100, 4)  # Normalize to [0, 1]


def compute_chrf(predictions: List[str], references: List[str]) -> float:
    """
    Corpus-level ChrF score.

    Why ChrF alongside BLEU:
    - BLEU is purely precision-based with brevity penalty
    - ChrF is character-level F-score: captures partial word matches
    - More forgiving for paraphrases and morphological variants
    - Better correlates with human judgment for morphologically rich languages
    """
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    return round(chrf.score, 2)


def compute_length_ratio(predictions: List[str], references: List[str]) -> float:
    """
    Average length ratio: mean(len(pred) / len(ref)) over all sentence pairs.

    Why this metric:
    - Directly measures under-translation (ratio << 1) vs. over-generation (ratio >> 1)
    - Complements BLEU's brevity penalty which operates at corpus level
    - Essential for diagnosing the "collapse to short output" failure mode

    Note: Uses word-level length (split on whitespace).
    """
    ratios = []
    for pred, ref in zip(predictions, references):
        pred_len = len(pred.split())
        ref_len = len(ref.split())
        if ref_len > 0:
            ratios.append(pred_len / ref_len)
        else:
            ratios.append(1.0 if pred_len == 0 else float("inf"))

    avg = sum(ratios) / len(ratios) if ratios else 0.0
    return round(avg, 4)


def compute_bigram_repetition(predictions: List[str], references: List[str]) -> float:
    """
    Average bigram repetition rate across predictions.

    Definition: For each prediction, count repeated bigrams / total bigrams.

    Why this metric:
    - Detects degenerate beam search behavior (looping, stuttering)
    - Low values expected for healthy translations
    - High values indicate the model is stuck in a repetitive pattern
    """
    total_rate = 0.0
    count = 0

    for pred in predictions:
        words = pred.split()
        if len(words) < 2:
            total_rate += 0.0
            count += 1
            continue

        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        unique_bigrams = set(bigrams)
        n_total = len(bigrams)
        n_repeated = n_total - len(unique_bigrams)

        rate = n_repeated / n_total if n_total > 0 else 0.0
        total_rate += rate
        count += 1

    avg = total_rate / count if count > 0 else 0.0
    return round(avg, 6)


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "en",
    model_type: Optional[str] = None,
    num_layers: Optional[int] = None,
    idf: bool = False,
    batch_size: int = 64,
    device: Optional[str] = None,
    rescale_with_baseline: bool = True,
) -> float:
    """
    Corpus-level BERTScore (F1) average.

    Notes:
    - Requires the optional 'bert-score' dependency.
    - Downloads a pretrained model on first use.
    - Returns a value in [0, 1] when rescale_with_baseline=True.
    """
    try:
        from bert_score import score as bertscore_score
    except ImportError as exc:
        raise ImportError(
            "BERTScore requires the 'bert-score' package. "
            "Install it with: pip install bert-score"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, f1 = bertscore_score(
        predictions,
        references,
        lang=lang,
        model_type=model_type,
        num_layers=num_layers,
        idf=idf,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=rescale_with_baseline,
    )
    return round(f1.mean().item(), 4)


# Registry for easy access from config
METRIC_REGISTRY: Dict[str, Callable] = {
    "bleu": compute_bleu,
    "chrf": compute_chrf,
    "length_ratio": compute_length_ratio,
    "bigram_repetition": compute_bigram_repetition,
    "bertscore": compute_bertscore,
}


def compute_all_metrics(
    predictions: List[str],
    references: List[str],
    metric_names: Optional[List[str]] = None,
    metric_kwargs: Optional[Dict[str, Dict]] = None,
) -> Dict[str, float]:
    """
    Compute multiple metrics in one pass.

    Args:
        predictions: Model translations.
        references: Gold reference translations.
        metric_names: Which metrics to compute (default: all).

    Returns:
        Dict of metric_name → score.
    """
    if metric_names is None:
        metric_names = DEFAULT_METRICS

    if metric_kwargs is None:
        metric_kwargs = {}

    # 说明：通过注册表按名称调用指标，配置驱动更直观。
    results = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            logger.warning(f"Unknown metric: {name}, skipping")
            continue
        kwargs = metric_kwargs.get(name, {})
        if kwargs:
            results[name] = METRIC_REGISTRY[name](predictions, references, **kwargs)
        else:
            results[name] = METRIC_REGISTRY[name](predictions, references)
        logger.info(f"  {name}: {results[name]}")

    return results

