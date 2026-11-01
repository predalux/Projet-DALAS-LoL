"""Simple metrics utility: precision, recall, F1

This module provides a small function `precision_recall_f1` that returns
precision, recall and f1-score for given true and predicted labels.
It will use scikit-learn if available, otherwise falls back to a pure-Python
implementation that supports 'macro', 'micro', 'weighted' and 'binary' averages.
"""
from collections import Counter
from typing import Iterable, Dict, Any, Optional


def _safe_div(n, d, zero_division: float):
    return zero_division if d == 0 else n / d


def precision_recall_f1(y_true: Iterable, y_pred: Iterable, *, average: str = "macro", zero_division: float = 0.0, pos_label: Optional[Any] = None) -> Dict[str, float]:
    """Compute precision, recall and F1.

    Args:
        y_true: iterable of true labels
        y_pred: iterable of predicted labels
        average: one of 'macro', 'micro', 'weighted', 'binary'
        zero_division: value to use when division by zero occurs
        pos_label: label to consider positive when average=='binary'. If None
            and there are exactly two labels, the second label in sorted order
            is used as positive.

    Returns:
        dict with keys 'precision', 'recall', 'f1'
    """
    try:
        from sklearn.metrics import precision_recall_fscore_support

        p, r, f1, _ = precision_recall_fscore_support(
            list(y_true), list(y_pred), average=average, zero_division=zero_division, pos_label=pos_label
        )
        return {"precision": float(p), "recall": float(r), "f1": float(f1)}
    except Exception:
        # fallback implementation
        y_t = list(y_true)
        y_p = list(y_pred)
        labels = sorted(set(y_t) | set(y_p), key=lambda x: str(x))

        if average == "binary":
            if pos_label is None:
                if len(labels) == 2:
                    pos_label = labels[1]
                else:
                    raise ValueError("pos_label must be specified for binary average when labels != 2")
            labels = [pos_label]

        # per-class counts
        per_tp = Counter()
        per_fp = Counter()
        per_fn = Counter()
        support = Counter()

        for t, p in zip(y_t, y_p):
            support[t] += 1
            if t == p:
                per_tp[t] += 1
            else:
                per_fp[p] += 1
                per_fn[t] += 1

        # per-class precision/recall/f1
        precisions = {}
        recalls = {}
        f1s = {}
        total_support = sum(support.values()) or 1

        for lbl in labels:
            tp = per_tp.get(lbl, 0)
            fp = per_fp.get(lbl, 0)
            fn = per_fn.get(lbl, 0)

            prec = _safe_div(tp, tp + fp, zero_division)
            rec = _safe_div(tp, tp + fn, zero_division)
            f1 = _safe_div(2 * prec * rec, prec + rec, zero_division) if (prec + rec) != 0 else zero_division

            precisions[lbl] = prec
            recalls[lbl] = rec
            f1s[lbl] = f1

        if average == "micro":
            sum_tp = sum(per_tp.values())
            sum_fp = sum(per_fp.values())
            sum_fn = sum(per_fn.values())
            prec = _safe_div(sum_tp, sum_tp + sum_fp, zero_division)
            rec = _safe_div(sum_tp, sum_tp + sum_fn, zero_division)
            f1 = _safe_div(2 * prec * rec, prec + rec, zero_division) if (prec + rec) != 0 else zero_division
            return {"precision": prec, "recall": rec, "f1": f1}

        if average == "weighted":
            prec = sum(precisions.get(lbl, 0) * support.get(lbl, 0) for lbl in labels) / total_support
            rec = sum(recalls.get(lbl, 0) * support.get(lbl, 0) for lbl in labels) / total_support
            f1 = sum(f1s.get(lbl, 0) * support.get(lbl, 0) for lbl in labels) / total_support
            return {"precision": prec, "recall": rec, "f1": f1}

        # default: macro
        prec = sum(precisions.get(lbl, 0) for lbl in labels) / len(labels)
        rec = sum(recalls.get(lbl, 0) for lbl in labels) / len(labels)
        f1 = sum(f1s.get(lbl, 0) for lbl in labels) / len(labels)
        return {"precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    # quick demonstration
    y_true = [0, 1, 2, 2, 1, 0]
    y_pred = [0, 2, 2, 2, 1, 0]
    print(precision_recall_f1(y_true, y_pred, average="macro"))
