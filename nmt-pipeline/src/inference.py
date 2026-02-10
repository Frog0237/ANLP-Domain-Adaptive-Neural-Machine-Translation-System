"""
Inference engine for NMT model with configurable decoding strategies.

Design decisions:
- Batched inference for GPU efficiency (vs. single-sentence which wastes compute)
- Decoding config is a plain dict → easy to serialize for experiment tracking
- Supports beam search, greedy, and length penalty configurations
- Returns both translations and raw token IDs for downstream analysis
- Optional task prefix for T5-style models
- Optional ALiBi patch for Marian models
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel

from .alibi import apply_alibi_to_marian
logger = logging.getLogger(__name__)

# 说明：推理阶段支持批量生成与解码参数配置。


@dataclass
class DecodingConfig:
    """
    Decoding strategy configuration.

    Why expose these parameters:
    - max_length: Controls output truncation; must match training regime
    - num_beams: Beam search width; 4 is standard balance of quality vs. speed
    - length_penalty: >1.0 favors longer outputs (critical for under-translation)
    - early_stopping: Stop when all beams produce EOS
    """

    max_length: int = 20
    num_beams: int = 4
    length_penalty: float = 1.0
    early_stopping: bool = True

    def to_dict(self) -> Dict:
        return {
            "max_length": self.max_length,
            "num_beams": self.num_beams,
            "length_penalty": self.length_penalty,
            "early_stopping": self.early_stopping,
        }


def load_model(
    model_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, AutoTokenizer]:
    """Load model and tokenizer for inference."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, attn_implementation="eager"
    )
    if getattr(model.config, "use_alibi", False):
        if getattr(model.config, "model_type", "") != "marian":
            raise ValueError("ALiBi patching is only supported for Marian models.")
        # 说明：根据模型配置启用 ALiBi。
        model = apply_alibi_to_marian(
            model,
            use_on_encoder=getattr(model.config, "alibi_on_encoder", True),
            use_on_decoder=getattr(model.config, "alibi_on_decoder", True),
        )
    model = model.to(device)
    model.eval()

    # Tokenizer: use base model name if local path doesn't have tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    logger.info(f"Loaded model from {model_path} on {device}")
    return model, tokenizer


@torch.no_grad()
def translate_batch(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    sources: List[str],
    decoding_config: Optional[DecodingConfig] = None,
    batch_size: int = 128,
    return_token_ids: bool = False,
    source_prefix: Optional[str] = None,
) -> List[str]:
    """
    Translate a list of source sentences in batches.

    Why batched inference:
    - GPU utilization: single-sentence inference underutilizes GPU capacity
    - Consistent with training regime (same padding/batching behavior)

    Args:
        model: Loaded seq2seq model.
        tokenizer: Corresponding tokenizer.
        sources: List of source language strings.
        decoding_config: Decoding parameters (default: beam=4, len_penalty=1.0).
        batch_size: Inference batch size.
        return_token_ids: If True, also return raw token IDs.
        source_prefix: Optional task prefix for models like T5.

    Returns:
        List of translated strings (and optionally token IDs).
    """
    if decoding_config is None:
        decoding_config = DecodingConfig()

    device = next(model.parameters()).device
    gen_kwargs = decoding_config.to_dict()

    all_translations = []
    all_token_ids = []

    for start in tqdm(range(0, len(sources), batch_size), desc="Translating"):
        batch_sources = sources[start : start + batch_size]
        if source_prefix:
            batch_sources = [f"{source_prefix}{text}" for text in batch_sources]

        # 说明：先编码再 generate，最后解码为文本。
        inputs = tokenizer(
            batch_sources,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        generated = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_translations.extend(translations)

        if return_token_ids:
            all_token_ids.extend(generated.cpu().tolist())

    if return_token_ids:
        return all_translations, all_token_ids
    return all_translations


def translate_single(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    source: str,
    decoding_config: Optional[DecodingConfig] = None,
    source_prefix: Optional[str] = None,
) -> str:
    """Convenience wrapper for single-sentence translation."""
    results = translate_batch(
        model,
        tokenizer,
        [source],
        decoding_config,
        batch_size=1,
        source_prefix=source_prefix,
    )
    return results[0]

