"""
Attention coverage diagnosis module.

Core idea:
- Extract decoder cross-attention weights from the last decoder layer
- For each source token, compute max attention received across all decoder steps
- Tokens with max attention < threshold are "low-coverage" → likely under-translated
- Aggregate low-coverage fraction per sentence as a diagnostic signal

Design decisions:
- Uses last decoder layer attention (most directly influences output distribution)
- Threshold is configurable to tune sensitivity
- Per-sentence fractions enable both aggregation and per-example debugging
- Optional task prefix for T5-style models in single-sentence mode
"""

import argparse
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .error_taxonomy import ErrorType
from .inference import DecodingConfig, load_model, translate_single
from .metrics import compute_bigram_repetition

logger = logging.getLogger(__name__)

# 说明：诊断模块支持覆盖率统计与单句级可视化报告。

# Heuristic thresholds for single-sentence diagnosis.
DEFAULT_LOW_COVERAGE_THRESHOLD = 0.2
OMISSION_RATIO_THRESHOLD = 0.8
HALLUCINATION_RATIO_THRESHOLD = 1.3
REPETITION_RATE_THRESHOLD = 0.2


def _slugify(text: str, max_len: int = 32) -> str:
    """Convert free text into a filename-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return slug[:max_len] if slug else "case"


def _pretty_token(token: str) -> str:
    """Simplify tokenizer artifacts for display."""
    return token.replace("▁", "").replace("Ġ", "")


def _filter_special_tokens(
    token_ids: List[int],
    tokens: List[str],
    special_ids: set,
) -> Tuple[List[int], List[str], List[bool]]:
    """Remove special tokens and return a mask for indexing attention."""
    keep_mask = [tid not in special_ids for tid in token_ids]
    filtered_ids = [tid for tid, keep in zip(token_ids, keep_mask) if keep]
    filtered_tokens = [_pretty_token(tok) for tok, keep in zip(tokens, keep_mask) if keep]
    return filtered_ids, filtered_tokens, keep_mask


@torch.no_grad()
def compute_attention_coverage(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    sources: List[str],
    targets: List[str],
    threshold: float = 0.2,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute low-coverage source token fractions for each sentence pair.

    Algorithm:
    1. Encode source, force-decode target to get cross-attention weights
    2. For the last decoder layer, average across heads: (n_heads, tgt_len, src_len) → (tgt_len, src_len)
    3. For each source token j, take max over all decoder steps i: coverage_j = max_i attn[i, j]
    4. Fraction of source tokens with coverage_j < threshold = low-coverage fraction

    Args:
        model: Encoder-decoder model with eager attention.
        tokenizer: Corresponding tokenizer.
        sources: Source language texts.
        targets: Target language texts (predictions, not references).
        threshold: Attention threshold for "low coverage" (default 0.2).
        batch_size: Processing batch size.

    Returns:
        np.ndarray of shape (n_sentences,) with low-coverage fractions.
    """
    # 说明：计算每个源词的最大注意力并统计低覆盖比例。
    device = next(model.parameters()).device
    model.eval()

    low_frac_list = []

    for start in tqdm(range(0, len(sources), batch_size), desc="Coverage analysis"):
        batch_src = sources[start : start + batch_size]
        batch_tgt = targets[start : start + batch_size]

        # Tokenize source
        src_enc = tokenizer(
            batch_src, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Tokenize target (for forced decoding)
        tgt_enc = tokenizer(
            text_target=batch_tgt, return_tensors="pt", padding=True, truncation=True
        ).to(device)

        # Forward pass with attention output
        outputs = model(
            input_ids=src_enc["input_ids"],
            attention_mask=src_enc["attention_mask"],
            decoder_input_ids=tgt_enc["input_ids"],
            output_attentions=True,
        )

        # Extract last decoder layer cross-attention
        # Shape: (batch, n_heads, tgt_len, src_len)
        cross_attn = outputs.cross_attentions[-1]

        # Average over heads → (batch, tgt_len, src_len)
        avg_attn = cross_attn.mean(dim=1)

        # Source attention mask for valid tokens
        src_mask = src_enc["attention_mask"]  # (batch, src_len)

        for i in range(avg_attn.size(0)):
            attn_matrix = avg_attn[i]  # (tgt_len, src_len)
            mask = src_mask[i].bool()  # (src_len,)

            # Max attention each source token receives across all decoder steps
            max_coverage = attn_matrix[:, mask].max(dim=0).values  # (n_valid_src,)

            n_valid = mask.sum().item()
            n_low = (max_coverage < threshold).sum().item()
            frac = n_low / n_valid if n_valid > 0 else 0.0
            low_frac_list.append(frac)

    return np.array(low_frac_list)


@torch.no_grad()
def compute_cross_attention_matrix(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    source: str,
    target: str,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute average cross-attention matrix for a single source-target pair.

    Returns:
        (attn_matrix, source_tokens, target_tokens)
    """
    device = next(model.parameters()).device
    model.eval()

    src_enc = tokenizer(source, return_tensors="pt", truncation=True).to(device)
    tgt_enc = tokenizer(text_target=target, return_tensors="pt", truncation=True).to(device)

    outputs = model(
        input_ids=src_enc["input_ids"],
        attention_mask=src_enc["attention_mask"],
        decoder_input_ids=tgt_enc["input_ids"],
        output_attentions=True,
    )

    cross_attn = outputs.cross_attentions[-1]  # (batch, n_heads, tgt_len, src_len)
    avg_attn = cross_attn.mean(dim=1)[0]  # (tgt_len, src_len)

    src_ids = src_enc["input_ids"][0].tolist()
    tgt_ids = tgt_enc["input_ids"][0].tolist()
    src_tokens = tokenizer.convert_ids_to_tokens(src_ids)
    tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_ids)

    special_ids = set(tokenizer.all_special_ids)
    _, src_tokens, src_keep = _filter_special_tokens(src_ids, src_tokens, special_ids)
    _, tgt_tokens, tgt_keep = _filter_special_tokens(tgt_ids, tgt_tokens, special_ids)

    src_keep_mask = torch.tensor(src_keep, device=avg_attn.device)
    tgt_keep_mask = torch.tensor(tgt_keep, device=avg_attn.device)
    filtered_attn = avg_attn[tgt_keep_mask][:, src_keep_mask].cpu().numpy()

    return filtered_attn, src_tokens, tgt_tokens


def save_attention_heatmap(
    attn_matrix: np.ndarray,
    source_tokens: List[str],
    target_tokens: List[str],
    output_path: str,
    title: str,
):
    """Save cross-attention heatmap to disk."""
    if attn_matrix.size == 0:
        logger.warning("Empty attention matrix; skipping heatmap.")
        return

    width = min(20, max(6, 0.35 * len(source_tokens)))
    height = min(20, max(4, 0.35 * len(target_tokens)))
    fig, ax = plt.subplots(figsize=(width, height))

    sns.heatmap(
        attn_matrix,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        cmap="viridis",
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Source tokens", fontsize=11)
    ax.set_ylabel("Target tokens", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Heatmap saved to {output_path}")


def _classify_error_type(
    low_coverage_frac: float,
    length_ratio: Optional[float],
    repetition_rate: float,
    prediction: str,
    has_reference: bool,
    low_coverage_threshold: float,
) -> Tuple[Optional[ErrorType], str]:
    """
    Heuristic error classifier for single-sentence diagnosis.

    Returns:
        (error_type or None, reason string)
    """
    if repetition_rate >= REPETITION_RATE_THRESHOLD:
        return ErrorType.HALLUCINATION, "High bigram repetition suggests degenerate output."

    if has_reference and length_ratio is not None:
        if length_ratio < OMISSION_RATIO_THRESHOLD:
            if not prediction.strip().endswith((".", "!", "?")):
                return ErrorType.TRUNCATION, "Short output without terminal punctuation."
            return ErrorType.OMISSION, "Prediction is much shorter than reference."
        if length_ratio > HALLUCINATION_RATIO_THRESHOLD:
            return ErrorType.HALLUCINATION, "Prediction is much longer than reference."

    if low_coverage_frac > low_coverage_threshold:
        return ErrorType.OMISSION, "Many source tokens have low attention coverage."

    return None, "No strong heuristic triggers; treat as success."


def _highlight_tokens(tokens: List[str], low_mask: List[bool]) -> str:
    """Build a human-readable string with low-coverage tokens marked."""
    highlighted = []
    for tok, is_low in zip(tokens, low_mask):
        if is_low:
            highlighted.append(f"[{tok}]")
        else:
            highlighted.append(tok)
    return " ".join([t for t in highlighted if t])


def run_single_sentence_diagnosis(
    model_path: str,
    source: str,
    reference: Optional[str],
    output_dir: str,
    case_id: Optional[str] = None,
    threshold: float = DEFAULT_LOW_COVERAGE_THRESHOLD,
    decoding_config: Optional[DecodingConfig] = None,
    source_prefix: Optional[str] = None,
) -> Dict:
    """
    Diagnose a single sentence: heatmap + low-coverage highlight + auto label.

    Args:
        source_prefix: Optional task prefix for models like T5.

    Returns:
        Dict containing artifacts and summary statistics.
    """
    model, tokenizer = load_model(model_path)

    if decoding_config is None:
        decoding_config = DecodingConfig()

    # 说明：先生成译文，再基于注意力矩阵做诊断。
    prediction = translate_single(
        model,
        tokenizer,
        source,
        decoding_config,
        source_prefix=source_prefix,
    )

    attn_matrix, src_tokens, tgt_tokens = compute_cross_attention_matrix(
        model, tokenizer, source, prediction
    )

    coverage = attn_matrix.max(axis=0) if attn_matrix.size else np.array([])
    low_mask = (coverage < threshold).tolist() if coverage.size else []
    low_tokens = [
        {"token": tok, "coverage": float(cov)}
        for tok, cov in zip(src_tokens, coverage)
        if cov < threshold
    ]
    low_fraction = float(np.mean(coverage < threshold)) if coverage.size else 0.0

    length_ratio = None
    if reference:
        ref_len = len(reference.split())
        pred_len = len(prediction.split())
        length_ratio = pred_len / ref_len if ref_len > 0 else None

    repetition_rate = compute_bigram_repetition([prediction], [reference or ""])
    error_type, reason = _classify_error_type(
        low_fraction,
        length_ratio,
        repetition_rate,
        prediction,
        has_reference=bool(reference),
        low_coverage_threshold=threshold,
    )

    case_slug = case_id or _slugify(source)
    heatmap_path = os.path.join(output_dir, f"heatmap_{case_slug}.png")
    report_json_path = os.path.join(output_dir, f"report_{case_slug}.json")
    report_md_path = os.path.join(output_dir, f"report_{case_slug}.md")

    save_attention_heatmap(
        attn_matrix,
        src_tokens,
        tgt_tokens,
        heatmap_path,
        title=f"Cross-attention heatmap ({case_slug})",
    )

    report = {
        "case_id": case_slug,
        "model_path": model_path,
        "source_prefix": source_prefix,
        "source": source,
        "prediction": prediction,
        "reference": reference,
        "tokens": {
            "source": src_tokens,
            "target": tgt_tokens,
            "low_coverage_source": low_tokens,
            "highlighted_source": _highlight_tokens(src_tokens, low_mask),
        },
        "metrics": {
            "low_coverage_fraction": round(low_fraction, 4),
            "length_ratio": round(length_ratio, 4) if length_ratio is not None else None,
            "bigram_repetition": repetition_rate,
        },
        "auto_label": error_type.value if error_type else "success",
        "auto_reason": reason,
        "artifacts": {
            "heatmap": heatmap_path,
            "report_json": report_json_path,
            "report_md": report_md_path,
        },
        "notes": "Auto labels are heuristic and for quick triage only.",
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Single-Sentence Diagnosis Report\n\n")
        f.write(f"- Case ID: `{case_slug}`\n")
        f.write(f"- Model: `{model_path}`\n")
        if source_prefix:
            f.write(f"- Source prefix: `{source_prefix}`\n")
        f.write(f"- Auto label: `{report['auto_label']}`\n")
        f.write(f"- Reason: {reason}\n\n")
        f.write("## Text\n\n")
        f.write(f"- Source: {source}\n")
        f.write(f"- Prediction: {prediction}\n")
        if reference:
            f.write(f"- Reference: {reference}\n")
        f.write("\n## Diagnostics\n\n")
        f.write(f"- Low-coverage fraction: {report['metrics']['low_coverage_fraction']}\n")
        if length_ratio is not None:
            f.write(f"- Length ratio (pred/ref): {report['metrics']['length_ratio']}\n")
        f.write(f"- Bigram repetition: {report['metrics']['bigram_repetition']}\n")
        f.write(f"- Highlighted source: {report['tokens']['highlighted_source']}\n")
        f.write(f"- Heatmap: `{heatmap_path}`\n")

    logger.info(f"Single-sentence report saved to {report_json_path}")
    return report


def plot_coverage_distribution(
    low_fracs: np.ndarray,
    model_name: str,
    split_name: str,
    output_path: Optional[str] = None,
    threshold: float = 0.2,
):
    """
    Plot histogram of low-coverage fractions.

    Args:
        low_fracs: Array of per-sentence low-coverage fractions.
        model_name: For title/filename.
        split_name: For title/filename.
        output_path: If provided, save figure to this path.
        threshold: The threshold used (for labeling).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(low_fracs, bins=20, alpha=0.7, edgecolor="black", color="#4C72B0")
    ax.axvline(
        low_fracs.mean(),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"mean={low_fracs.mean():.3f}",
    )

    ax.set_xlabel("Low-coverage fraction per sentence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Low-coverage source token fraction (thr={threshold}) - {model_name} {split_name}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.text(
        0.95,
        0.95,
        split_name,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Coverage plot saved to {output_path}")
    plt.close(fig)


def run_diagnosis(
    model_path: str,
    dataset_name: str = "EdinburghNLP/europarl-de-en-mini",
    split: str = "ood",
    threshold: float = DEFAULT_LOW_COVERAGE_THRESHOLD,
    output_dir: str = "outputs/diagnosis",
    source_prefix: Optional[str] = None,
):
    """
    Run attention coverage diagnosis for a model on a split.

    Args:
        source_prefix: Optional task prefix for models like T5.
    """
    from .data import load_parallel_corpus, get_source_and_references
    from .inference import load_model, translate_batch

    split_map = {"iid": "validation", "ood": "gen_val"}
    split_str = split_map.get(split, split)

    # Load data
    datasets = load_parallel_corpus(dataset_name, {split: split_str})
    ds = datasets[split]
    sources, references = get_source_and_references(ds)

    # Generate translations
    model, tokenizer = load_model(model_path)
    predictions = translate_batch(
        model,
        tokenizer,
        sources,
        source_prefix=source_prefix,
    )

    # Compute coverage
    low_fracs = compute_attention_coverage(
        model, tokenizer, sources, predictions, threshold=threshold
    )

    # Plot
    model_name = os.path.basename(model_path)
    plot_path = os.path.join(output_dir, f"coverage_{model_name}_{split}.png")
    plot_coverage_distribution(low_fracs, model_name, split, plot_path, threshold)

    # Summary statistics
    stats = {
        "model": model_path,
        "split": split,
        "threshold": threshold,
        "mean_low_coverage": float(low_fracs.mean()),
        "median_low_coverage": float(np.median(low_fracs)),
        "std_low_coverage": float(low_fracs.std()),
        "max_low_coverage": float(low_fracs.max()),
        "n_sentences": len(low_fracs),
    }

    logger.info(f"Coverage stats: {stats}")
    return stats, low_fracs


def main():
    parser = argparse.ArgumentParser(description="Attention coverage diagnosis")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="ood", choices=["iid", "ood"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_LOW_COVERAGE_THRESHOLD)
    parser.add_argument("--output-dir", type=str, default="outputs/diagnosis")
    parser.add_argument("--source", type=str, default=None, help="Single sentence source text")
    parser.add_argument("--reference", type=str, default=None, help="Optional reference translation")
    parser.add_argument("--case-id", type=str, default=None, help="Identifier for report filenames")
    parser.add_argument("--source-prefix", type=str, default=None, help="Optional task prefix")
    parser.add_argument("--max-length", type=int, default=20, help="Max generation length")
    parser.add_argument("--num-beams", type=int, default=4, help="Beam size")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Length penalty")
    parser.add_argument(
        "--no-early-stopping",
        action="store_false",
        dest="early_stopping",
        help="Disable early stopping",
    )
    parser.set_defaults(early_stopping=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.source:
        dc = DecodingConfig(
            max_length=args.max_length,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
        )
        run_single_sentence_diagnosis(
            model_path=args.model_path,
            source=args.source,
            reference=args.reference,
            output_dir=args.output_dir,
            case_id=args.case_id,
            threshold=args.threshold,
            decoding_config=dc,
            source_prefix=args.source_prefix,
        )
    else:
        run_diagnosis(
            args.model_path,
            split=args.split,
            threshold=args.threshold,
            output_dir=args.output_dir,
            source_prefix=args.source_prefix,
        )


if __name__ == "__main__":
    main()

