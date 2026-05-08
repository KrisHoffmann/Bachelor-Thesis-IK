"""
Evaluation utilities for Stage 1 (salience) and Stage 2 (CLT dimensions).

Stage 1 — compute_metrics_dict:
    macro_f1, accuracy, per_class_f1, confusion_matrix, krippendorff_alpha (nominal)

Stage 2 — evaluate_stage2:
    Per dimension: macro_f1, per_class_f1, confusion_matrix, krippendorff_alpha (ordinal)
    Aggregate: mean_macro_f1 across all four dimensions
    Collapse flag: True when model predicts only one class for a dimension
"""

from typing import Sequence

import krippendorff
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

DIMS = ("temporal", "spatial", "social", "hypothetical")
# Raw label domain for Stage 2 ordinal Krippendorff measurement
STAGE2_VALUE_DOMAIN = [-1, 0, 1]


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


def _evaluate_one_dim(
    predictions: Sequence[int],
    gold: Sequence[int],
    dim_name: str,
) -> dict:
    """Evaluate a single Stage 2 dimension. Values are in {-1, 0, 1}.

    Returns macro_f1, per_class_f1 (for classes -1/0/1), confusion_matrix (3×3),
    krippendorff_alpha (ordinal), and a 'collapsed' bool flag.
    """
    preds = np.array(predictions)
    labels = np.array(gold)

    unique_preds = set(predictions)
    collapsed = len(unique_preds) == 1

    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    per_class_f1 = f1_score(
        labels, preds, average=None, labels=[-1, 0, 1], zero_division=0
    ).tolist()
    cm = confusion_matrix(labels, preds, labels=[-1, 0, 1])

    # Krippendorff alpha — ordinal level for ordered {-1, 0, 1}
    reliability_data = np.array([labels.tolist(), preds.tolist()], dtype=float)
    try:
        alpha = float(
            krippendorff.alpha(
                reliability_data=reliability_data,
                level_of_measurement="ordinal",
            )
        )
    except Exception:
        # degenerate case (all predictions identical) → alpha undefined
        alpha = float("nan")

    return {
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,  # [f1_{-1}, f1_0, f1_1]
        "confusion_matrix": cm,
        "krippendorff_alpha": alpha,
        "collapsed": collapsed,
    }


def evaluate_stage2(
    predictions_dict: dict[str, list[int]],
    gold_dict: dict[str, list[int]],
) -> dict:
    """Evaluate Stage 2 CLT dimension predictions.

    Args:
        predictions_dict: {dim_name: list of int} — values in {-1, 0, 1} (decoded).
        gold_dict:        {dim_name: list of int} — values in {-1, 0, 1} (decoded).

    Returns:
        Dict with one sub-dict per dimension (keys: macro_f1, per_class_f1,
        confusion_matrix, krippendorff_alpha, collapsed) plus 'mean_macro_f1'
        across all four dimensions.

        'collapsed' is True when the model predicts only one class for a
        dimension — expected for Temporal and Spatial due to severe N/A imbalance.
    """
    result = {}
    macro_f1s = []

    for dim in DIMS:
        if dim not in predictions_dict:
            raise KeyError(f"predictions_dict missing key '{dim}'")
        if dim not in gold_dict:
            raise KeyError(f"gold_dict missing key '{dim}'")
        if len(predictions_dict[dim]) != len(gold_dict[dim]):
            raise ValueError(
                f"Length mismatch for '{dim}': "
                f"predictions={len(predictions_dict[dim])}, gold={len(gold_dict[dim])}"
            )
        dim_metrics = _evaluate_one_dim(predictions_dict[dim], gold_dict[dim], dim)
        result[dim] = dim_metrics
        macro_f1s.append(dim_metrics["macro_f1"])

    result["mean_macro_f1"] = float(np.mean(macro_f1s))
    return result
