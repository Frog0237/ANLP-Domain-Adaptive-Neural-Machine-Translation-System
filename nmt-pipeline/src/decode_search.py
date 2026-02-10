"""
Decoding strategy grid search: beam size × length penalty.

Design decisions:
- Grid search over discrete (beam_size, length_penalty) combinations
- Evaluates all metrics on all splits for each configuration
- Outputs a comparison table for easy selection
- Optional task prefix for T5-style models

Why methodical search (vs. manual tuning):
- Manual tuning introduces human bias and is hard to reproduce
- Grid search with recorded configs enables fair comparison
- The search space is small enough to be exhaustive
"""

import argparse
import json
import logging
import os
from itertools import product
from typing import Dict, List

import pandas as pd
import yaml

from .data import load_parallel_corpus, get_source_and_references
from .inference import DecodingConfig, load_model, translate_batch
from .metrics import compute_all_metrics, DEFAULT_METRICS

logger = logging.getLogger(__name__)

# 说明：对解码参数做网格搜索，便于比较不同组合。


def run_decode_search(config_path: str):
    """
    Run grid search over decoding strategies.

    For each (beam_size, length_penalty) combination:
    1. Translate all splits
    2. Compute all metrics
    3. Record results

    Outputs:
    - CSV with all configurations × splits × metrics
    - JSON with full details
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["model_path"]
    output_dir = cfg["output"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Load model once
    model, tokenizer = load_model(model_path)

    # Load data splits
    split_map = {}
    for s in cfg["splits"]:
        split_map[s] = s

    datasets = load_parallel_corpus(
        cfg["data"]["dataset_name"],
        split_map,
        cfg["data"]["source_lang"],
        cfg["data"]["target_lang"],
    )

    metric_names = cfg.get("metrics", DEFAULT_METRICS)
    metric_kwargs = cfg.get("metric_kwargs", {})
    # Optional task prefix for T5-style models.
    source_prefix = cfg["data"].get("source_prefix")
    max_length = cfg["search_space"].get("max_length", 20)

    # Grid search
    beam_sizes = cfg["search_space"]["beam_sizes"]
    length_penalties = cfg["search_space"]["length_penalties"]

    all_rows = []

    for beam, penalty in product(beam_sizes, length_penalties):
        # 说明：逐组合逐分割计算指标，形成对照表。
        dc = DecodingConfig(
            max_length=max_length,
            num_beams=beam,
            length_penalty=penalty,
            early_stopping=True,
        )

        for split_name, ds in datasets.items():
            sources, references = get_source_and_references(
                ds,
                cfg["data"]["source_lang"],
                cfg["data"]["target_lang"],
            )

            logger.info(f"beam={beam}, penalty={penalty}, split={split_name}")
            predictions = translate_batch(
                model,
                tokenizer,
                sources,
                dc,
                source_prefix=source_prefix,
            )
            metrics = compute_all_metrics(
                predictions,
                references,
                metric_names=metric_names,
                metric_kwargs=metric_kwargs,
            )

            row = {
                "beam_size": beam,
                "length_penalty": penalty,
                "split": split_name,
                **metrics,
            }
            all_rows.append(row)

    # Save results
    df = pd.DataFrame(all_rows)

    if cfg["output"].get("save_csv", True):
        csv_path = os.path.join(output_dir, "decode_search_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")

    print("\n" + df.to_string(index=False))
    return df


def main():
    parser = argparse.ArgumentParser(description="Decoding strategy grid search")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/decode_search.yaml",
        help="Path to decode search config YAML",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_decode_search(args.config)


if __name__ == "__main__":
    main()

