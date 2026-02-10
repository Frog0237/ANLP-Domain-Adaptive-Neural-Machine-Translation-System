"""
Generate single-sentence diagnosis artifacts for a few curated examples.

This script is designed to populate README visuals with reproducible cases.
"""

import argparse
import os
import sys
from typing import List, Dict

# Ensure the project root is on sys.path for direct script execution.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "assets", "diagnosis"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.diagnose import run_single_sentence_diagnosis


DEFAULT_CASES: List[Dict[str, str]] = [
    {
        "id": "case_short",
        "source": "Wir danken dem Ausschuss für seine Arbeit.",
        "reference": "We thank the committee for its work.",
    },
    {
        "id": "case_medium",
        "source": "Die Präsidentin erklärte, dass die Debatte morgen fortgesetzt wird.",
        "reference": "The President stated that the debate will continue tomorrow.",
    },
    {
        "id": "case_long",
        "source": (
            "Trotz der schwierigen Verhandlungen stimmte das Parlament dem Abkommen "
            "mit großer Mehrheit zu."
        ),
        "reference": (
            "Despite the difficult negotiations, Parliament approved the agreement "
            "by a large majority."
        ),
    },
]


# 说明：批量生成单句诊断样例与热力图文件。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate README case study artifacts")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Helsinki-NLP/opus-mt-de-en",
        help="Model checkpoint or HF model name",
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default=None,
        help="Optional task prefix for models like T5",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store heatmaps and reports",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for case in DEFAULT_CASES:
        run_single_sentence_diagnosis(
            model_path=args.model_path,
            source=case["source"],
            reference=case["reference"],
            output_dir=args.output_dir,
            case_id=case["id"],
            source_prefix=args.source_prefix,
        )


if __name__ == "__main__":
    main()

