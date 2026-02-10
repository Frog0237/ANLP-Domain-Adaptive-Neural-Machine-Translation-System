"""
Multi-model, multi-split evaluation pipeline.

Design decisions:
- Decoupled from training: can evaluate any checkpoint independently
- Saves translations + metrics to disk for reproducibility
- Supports batch comparison across models and data splits
- Outputs structured JSON for programmatic consumption
- Allows per-model task prefixes for T5-style models
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

from .data import load_parallel_corpus, get_source_and_references
from .inference import DecodingConfig, load_model, translate_batch
from .metrics import compute_all_metrics, DEFAULT_METRICS

logger = logging.getLogger(__name__)

# 说明：评估流程按模型与数据分割循环执行，输出指标与译文。


def evaluate_model_on_split(
    model_path: str,
    dataset_name: str,
    split_name: str,
    split_str: str,
    source_lang: str,
    target_lang: str,
    decoding_config: DecodingConfig,
    metric_names: List[str],
    metric_kwargs: Dict[str, Dict],
    source_prefix: Optional[str],
    batch_size: int = 128,
) -> Dict:
    """
    Evaluate a single model on a single data split.

    Args:
        metric_kwargs: Optional per-metric kwargs (e.g., BERTScore settings).
        source_prefix: Optional task prefix for models like T5.

    Returns:
        Dict with keys: model, split, metrics, translations (list of dicts)
    """
    # Load data
    datasets = load_parallel_corpus(
        dataset_name, {split_name: split_str}, source_lang, target_lang
    )
    ds = datasets[split_name]
    sources, references = get_source_and_references(ds, source_lang, target_lang)

    # Load model & translate
    model, tokenizer = load_model(model_path)
    predictions = translate_batch(
        model,
        tokenizer,
        sources,
        decoding_config,
        batch_size,
        source_prefix=source_prefix,
    )

    # Compute metrics
    logger.info(f"Computing metrics for {model_path} on {split_name}...")
    metrics = compute_all_metrics(predictions, references, metric_names, metric_kwargs)

    # Build per-example results
    translations = [
        {
            "source": src,
            "prediction": pred,
            "reference": ref,
        }
        for src, pred, ref in zip(sources, predictions, references)
    ]

    return {
        "model": model_path,
        "split": split_name,
        "decoding_config": decoding_config.to_dict(),
        "metrics": metrics,
        "translations": translations,
    }


def run_evaluation(config_path: str):
    """
    Run full evaluation from config: all models × all splits.

    Outputs:
    - Per-model-split JSON with translations and metrics
    - Summary CSV table comparing all conditions
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg["output"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    decoding_config = DecodingConfig(
        max_length=cfg["inference"].get("max_length", 20),
        num_beams=cfg["inference"].get("num_beams", 4),
        length_penalty=cfg["inference"].get("length_penalty", 1.0),
        early_stopping=cfg["inference"].get("early_stopping", True),
    )

    metric_names = cfg.get("metrics", DEFAULT_METRICS)
    metric_kwargs = cfg.get("metric_kwargs", {})
    batch_size = cfg["inference"].get("batch_size", 128)
    # Allow per-model prefix overrides (useful for T5-style models).
    default_prefix = cfg["data"].get("source_prefix")

    all_results = []
    summary_rows = []

    for model_key, model_info in cfg["models"].items():
        # 说明：逐模型逐分割推理并汇总指标。
        model_prefix = model_info.get("source_prefix", default_prefix)
        for split_name, split_str in cfg["data"]["splits"].items():
            logger.info(f"Evaluating {model_key} on {split_name}...")

            result = evaluate_model_on_split(
                model_path=model_info["path"],
                dataset_name=cfg["data"]["dataset_name"],
                split_name=split_name,
                split_str=split_str,
                source_lang=cfg["data"]["source_lang"],
                target_lang=cfg["data"]["target_lang"],
                decoding_config=decoding_config,
                metric_names=metric_names,
                metric_kwargs=metric_kwargs,
                source_prefix=model_prefix,
                batch_size=batch_size,
            )

            # Save individual results
            if cfg["output"].get("save_translations", True):
                result_path = os.path.join(output_dir, f"{model_key}_{split_name}.json")
                with open(result_path, "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                logger.info(f"  Saved to {result_path}")

            # Summary row
            row = {
                "model": model_info.get("label", model_key),
                "split": split_name,
                **result["metrics"],
            }
            summary_rows.append(row)
            all_results.append(result)

    # Save summary table
    if cfg["output"].get("save_metrics_table", True):
        df = pd.DataFrame(summary_rows)
        csv_path = os.path.join(output_dir, "metrics_summary.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"\nMetrics summary saved to {csv_path}")
        print("\n" + df.to_string(index=False))

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate NMT models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_config.yaml",
        help="Path to evaluation config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_evaluation(args.config)


if __name__ == "__main__":
    main()

