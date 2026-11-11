"""Metrics, in the same shape the paper reports them.

Overall accuracy and weighted F1, then one-vs-rest precision / recall / F1 per
class (Tables 5 and 6), plus one-vs-rest ROC and precision-recall curves with
micro and prevalence-weighted averages (Figures 3-6).

The averaging follows the paper's figures: "micro" pools every one-vs-rest
decision, and "weighted" weights the per-class curves by class prevalence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

from .config import CLASS_NAMES


@dataclass
class Predictions:
    """Parallel arrays over the evaluated images."""

    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray  # (n_images, n_classes) softmax scores

    def __len__(self) -> int:
        return len(self.y_true)


def overall(pred: Predictions) -> dict[str, float]:
    """Accuracy and weighted F1, as percentages."""
    return {
        "accuracy": 100.0 * accuracy_score(pred.y_true, pred.y_pred),
        "f1_weighted": 100.0 * f1_score(pred.y_true, pred.y_pred, average="weighted"),
    }


def per_class(pred: Predictions, step: int) -> dict[str, dict[str, float]]:
    """One-vs-rest precision / recall / F1 / support for every class."""
    names = CLASS_NAMES[step]
    precision, recall, f1, support = precision_recall_fscore_support(
        pred.y_true,
        pred.y_pred,
        labels=list(range(len(names))),
        average=None,
        zero_division=0,
    )
    return {
        name: {
            "precision": 100.0 * precision[i],
            "recall": 100.0 * recall[i],
            "f1": 100.0 * f1[i],
            "support": int(support[i]),
        }
        for i, name in enumerate(names)
    }


def confusion(pred: Predictions, step: int) -> np.ndarray:
    n = len(CLASS_NAMES[step])
    return confusion_matrix(pred.y_true, pred.y_pred, labels=list(range(n)))


def _weighted_mean_curve(x_by_class, y_by_class, counts) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate every class curve onto the union grid, average by prevalence."""
    grid = np.unique(np.concatenate(list(x_by_class.values())))
    total = sum(counts)
    mean = np.zeros_like(grid, dtype=float)
    for label, x in x_by_class.items():
        interp = interp1d(x, y_by_class[label], bounds_error=False)
        mean += np.nan_to_num(interp(grid)) * counts[label] / total
    return grid, mean


def roc_curves(pred: Predictions, step: int) -> dict:
    """One-vs-rest ROC curves and AUCs, keyed by class name plus micro/weighted."""
    names = CLASS_NAMES[step]
    counts = [int((pred.y_true == i).sum()) for i in range(len(names))]

    fpr, tpr, areas = {}, {}, {}
    pooled_true, pooled_score = [], []
    for i, name in enumerate(names):
        true_bin = (pred.y_true == i).astype(int)
        score = pred.probabilities[:, i]
        pooled_true.append(true_bin)
        pooled_score.append(score)
        fpr[i], tpr[i], _ = roc_curve(true_bin, score)
        areas[name] = auc(fpr[i], tpr[i])

    micro_fpr, micro_tpr, _ = roc_curve(
        np.concatenate(pooled_true), np.concatenate(pooled_score)
    )
    areas["micro"] = auc(micro_fpr, micro_tpr)
    grid, mean = _weighted_mean_curve(fpr, tpr, counts)
    areas["weighted"] = sum(
        areas[name] * counts[i] / sum(counts) for i, name in enumerate(names)
    )
    return {
        "areas": areas,
        "per_class": {names[i]: (fpr[i], tpr[i]) for i in range(len(names))},
        "micro": (micro_fpr, micro_tpr),
        "weighted": (grid, mean),
    }


def pr_curves(pred: Predictions, step: int) -> dict:
    """One-vs-rest precision-recall curves and average precisions (AUPRC)."""
    names = CLASS_NAMES[step]
    counts = [int((pred.y_true == i).sum()) for i in range(len(names))]

    precision, recall, areas = {}, {}, {}
    pooled_true, pooled_score = [], []
    for i, name in enumerate(names):
        true_bin = (pred.y_true == i).astype(int)
        score = pred.probabilities[:, i]
        pooled_true.append(true_bin)
        pooled_score.append(score)
        precision[i], recall[i], _ = precision_recall_curve(true_bin, score)
        areas[name] = average_precision_score(true_bin, score)

    pooled_true_arr = np.concatenate(pooled_true)
    pooled_score_arr = np.concatenate(pooled_score)
    micro_precision, micro_recall, _ = precision_recall_curve(
        pooled_true_arr, pooled_score_arr
    )
    areas["micro"] = average_precision_score(pooled_true_arr, pooled_score_arr)
    # Curves are plotted recall on x, precision on y.
    grid, mean = _weighted_mean_curve(recall, precision, counts)
    areas["weighted"] = sum(
        areas[name] * counts[i] / sum(counts) for i, name in enumerate(names)
    )
    return {
        "areas": areas,
        "per_class": {names[i]: (recall[i], precision[i]) for i in range(len(names))},
        "micro": (micro_recall, micro_precision),
        "weighted": (grid, mean),
    }


def full_report(pred: Predictions, step: int) -> dict:
    """Everything the paper's tables need, as plain JSON-serialisable data."""
    roc = roc_curves(pred, step)
    pr = pr_curves(pred, step)
    return {
        "n_images": len(pred),
        "overall": overall(pred),
        "per_class": per_class(pred, step),
        "confusion_matrix": confusion(pred, step).tolist(),
        "auc": {k: float(v) for k, v in roc["areas"].items()},
        "auprc": {k: float(v) for k, v in pr["areas"].items()},
    }


def write_predictions_csv(path, records, pred: Predictions, step: int) -> None:
    """One row per test image: what it is, what the model said, how sure it was.

    Saved next to every checkpoint so any later analysis -- per-patient
    aggregation, error review, a different averaging scheme -- can be done
    without re-running inference. `records` must be in loader order, which it
    is: the test loader never shuffles.
    """
    import csv

    names = CLASS_NAMES[step]
    if len(records) != len(pred.y_true):
        raise ValueError(
            f"{len(records)} records but {len(pred.y_true)} predictions; "
            "the loader order and the record list have diverged"
        )

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "patient", "source_stem", "true_label", "true_class",
             "predicted_label", "predicted_class", "correct"]
            + [f"prob_{name}" for name in names]
        )
        for i, record in enumerate(records):
            true_label = int(pred.y_true[i])
            predicted_label = int(pred.y_pred[i])
            writer.writerow(
                [
                    record.path.name,
                    record.patient,
                    record.source_stem,
                    true_label,
                    names[true_label],
                    predicted_label,
                    names[predicted_label],
                    int(true_label == predicted_label),
                ]
                + [f"{value:.6f}" for value in pred.probabilities[i]]
            )
