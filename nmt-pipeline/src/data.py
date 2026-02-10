"""
Data loading, preprocessing, and tokenization for NMT pipeline.

Design decisions:
- Uses HuggingFace datasets for reproducible data loading with caching
- Tokenization is a closure to avoid global state (testable, composable)
- Supports configurable source/target language pairs for extensibility
- Optional task prefix for T5-style models
"""

import logging
from typing import Dict, Optional, Tuple

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# 说明：负责数据加载与分词预处理，返回可直接训练的数据集对象。


def load_parallel_corpus(
    dataset_name: str,
    splits: Dict[str, str],
    source_lang: str = "de",
    target_lang: str = "en",
) -> Dict[str, Dataset]:
    """
    Load parallel corpus with multiple splits.

    Args:
        dataset_name: HuggingFace dataset identifier.
        splits: Mapping of split_name -> HF split string.
        source_lang: Source language code.
        target_lang: Target language code.

    Returns:
        Dictionary mapping split names to Dataset objects.
    """
    loaded = {}
    for name, split_str in splits.items():
        logger.info(f"Loading split '{name}' from {dataset_name} (split={split_str})")
        ds = load_dataset(dataset_name, split=split_str)
        loaded[name] = ds
        logger.info(f"  → {len(ds)} examples loaded")
    return loaded


def create_preprocess_fn(
    tokenizer: PreTrainedTokenizer,
    source_lang: str = "de",
    target_lang: str = "en",
    max_source_length: Optional[int] = None,
    max_target_length: Optional[int] = None,
    source_prefix: Optional[str] = None,
):
    """
    Create a preprocessing function as a closure over tokenizer and config.

    Why closure instead of global state:
    - Avoids mutable global variables (hard to test, thread-unsafe)
    - Each call to create_preprocess_fn produces an independent function
    - Enables different tokenizer configs for different data splits if needed

    Args:
        tokenizer: HuggingFace tokenizer instance.
        source_lang: Source language key in translation dict.
        target_lang: Target language key in translation dict.
        max_source_length: Optional max tokens for source (None = no limit).
        max_target_length: Optional max tokens for target (None = no limit).
        source_prefix: Optional task prefix (useful for T5-style models).

    Returns:
        A preprocessing function compatible with Dataset.map().
    """

    def preprocess_function(examples):
        # 说明：整理源/目标文本为批次，交给 tokenizer 生成模型输入。
        inputs = [ex[source_lang] for ex in examples["translation"]]
        if source_prefix:
            inputs = [f"{source_prefix}{text}" for text in inputs]
        targets = [ex[target_lang] for ex in examples["translation"]]

        tokenizer_kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        if max_source_length is not None:
            tokenizer_kwargs["max_length"] = max_source_length
            tokenizer_kwargs["truncation"] = True

        model_inputs = tokenizer(
            inputs, text_target=targets, **tokenizer_kwargs
        )
        return model_inputs

    return preprocess_function


def prepare_datasets(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    splits: Dict[str, str],
    source_lang: str = "de",
    target_lang: str = "en",
    max_source_length: Optional[int] = None,
    max_target_length: Optional[int] = None,
    source_prefix: Optional[str] = None,
) -> Dict[str, Dataset]:
    """
    End-to-end: load + tokenize all splits.

    Args:
        dataset_name: HuggingFace dataset identifier.
        tokenizer: Tokenizer instance.
        splits: Split name -> HF split string mapping.
        source_lang: Source language code.
        target_lang: Target language code.
        source_prefix: Optional task prefix (useful for T5-style models).

    Returns:
        Dictionary of tokenized datasets.
    """
    raw_datasets = load_parallel_corpus(dataset_name, splits, source_lang, target_lang)

    preprocess_fn = create_preprocess_fn(
        tokenizer,
        source_lang=source_lang,
        target_lang=target_lang,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        source_prefix=source_prefix,
    )

    tokenized = {}
    for name, ds in raw_datasets.items():
        logger.info(f"Tokenizing split '{name}'...")
        tokenized[name] = ds.map(preprocess_fn, batched=True)

    return tokenized


def get_source_and_references(
    dataset: Dataset,
    source_lang: str = "de",
    target_lang: str = "en",
) -> Tuple[list, list]:
    """
    Extract raw source texts and reference translations from a dataset.

    Returns:
        Tuple of (sources, references) as string lists.
    """
    sources = [ex[source_lang] for ex in dataset["translation"]]
    references = [ex[target_lang] for ex in dataset["translation"]]
    return sources, references

