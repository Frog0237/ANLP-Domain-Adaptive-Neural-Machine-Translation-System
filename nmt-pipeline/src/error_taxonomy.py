"""
Translation error taxonomy and annotation framework.

Defines a 6-class error classification standard for NMT output analysis:
1. Omission     - Source content missing from translation
2. Hallucination - Generated content not present in source
3. Grammar       - Syntactic errors or unnatural constructions
4. Truncation    - Translation cut off before source content is fully rendered
5. Entity        - Named entity mistranslation or corruption
6. Register      - Inappropriate formality/style level

Design decisions:
- Enum-based classification for type safety and serialization
- Each error has severity (minor/major/critical) for weighted scoring
- Supports multi-label annotation (one sentence can have multiple errors)
- Annotation output is JSON → easily consumed by analysis scripts

Why this taxonomy (vs. MQM or DQF):
- MQM (Multidimensional Quality Metrics) has 100+ subcategories → overkill for this scale
- Our 6 classes cover the dominant failure modes observed in enc-dec NMT:
  - Omission + Truncation: under-translation in long or complex inputs
  - Hallucination: over-generation / unfaithful content
  - Grammar + Register: fluency issues even when content is correct
  - Entity: critical for parliamentary domain (names, institutions)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 说明：定义错误类型与标注结构，便于统计与可视化。


class ErrorType(str, Enum):
    """Translation error types."""
    OMISSION = "omission"
    HALLUCINATION = "hallucination"
    GRAMMAR = "grammar"
    TRUNCATION = "truncation"
    ENTITY = "entity"
    REGISTER = "register"


class Severity(str, Enum):
    """Error severity levels."""
    MINOR = "minor"       # Noticeable but doesn't impede understanding
    MAJOR = "major"       # Significantly impacts meaning or readability
    CRITICAL = "critical" # Completely wrong or misleading


@dataclass
class ErrorAnnotation:
    """Single error annotation on a translation."""
    error_type: ErrorType
    severity: Severity
    span_start: Optional[int] = None  # Character offset in prediction
    span_end: Optional[int] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "error_type": self.error_type.value,
            "severity": self.severity.value,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "description": self.description,
        }


@dataclass
class AnnotatedExample:
    """A translation pair with error annotations."""
    source: str
    prediction: str
    reference: str
    errors: List[ErrorAnnotation] = field(default_factory=list)
    is_success: bool = False  # True if translation is semantically correct
    is_difficult: bool = False  # True if error boundaries are unclear
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "prediction": self.prediction,
            "reference": self.reference,
            "errors": [e.to_dict() for e in self.errors],
            "is_success": self.is_success,
            "is_difficult": self.is_difficult,
            "notes": self.notes,
        }


def save_annotations(annotations: List[AnnotatedExample], output_path: str):
    """Save annotations to JSON file."""
    data = [a.to_dict() for a in annotations]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(annotations)} annotations to {output_path}")


def load_annotations(input_path: str) -> List[Dict]:
    """Load annotations from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_error_distribution(annotations: List[AnnotatedExample]) -> Dict[str, int]:
    """
    Count occurrences of each error type across all annotations.

    Returns:
        Dict mapping error type name to count.
    """
    counts = {e.value: 0 for e in ErrorType}
    counts["success"] = 0
    counts["difficult"] = 0

    for ann in annotations:
        if ann.is_success:
            counts["success"] += 1
        if ann.is_difficult:
            counts["difficult"] += 1
        for error in ann.errors:
            counts[error.error_type.value] += 1

    return counts


def plot_error_distribution(
    counts: Dict[str, int],
    title: str = "Error Distribution",
    output_path: Optional[str] = None,
):
    """Plot bar chart of error type distribution."""
    import matplotlib.pyplot as plt

    labels = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color="#4C72B0", edgecolor="black", alpha=0.8)

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(val),
                ha="center",
                fontsize=10,
            )

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Error distribution plot saved to {output_path}")
    plt.close(fig)

