"""
Evaluation utilities for the Stage 1 salience classifier.

Returns a dict with:
    macro_f1         float
    accuracy         float
    per_class_f1     list[float]   — [f1_class0, f1_class1]
    confusion_matrix np.ndarray   — shape (2, 2)
    krippendorff_alpha  float     — nominal level, binary
"""

from typing import Sequence

import krippendorff
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics_dict(
    predictions: Sequence[int],
    gold: Sequence[int],
) -> dict:
    """Compute evaluation metrics given integer predictions and gold labels.

    Args:
        predictions: Model predictions (0 or 1).
        gold:        Gold labels (0 or 1).

    Returns:
        Dict with macro_f1, accuracy, per_class_f1, confusion_matrix,
        krippendorff_alpha.
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"Length mismatch: predictions={len(predictions)}, gold={len(gold)}"
        )

    preds = np.array(predictions)
    labels = np.array(gold)

    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    accuracy = float(accuracy_score(labels, preds))
    per_class_f1 = f1_score(
        labels, preds, average=None, labels=[0, 1], zero_division=0
    ).tolist()
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # krippendorff alpha — nominal, binary
    # reliability_data shape: (2, n_items) — annotator 0 = gold, annotator 1 = pred
    reliability_data = np.array([labels.tolist(), preds.tolist()], dtype=float)
    alpha = float(
        krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement="nominal",
        )
    )

    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "krippendorff_alpha": alpha,
    }
