"""Per-run metrics (labels passed explicitly so a missing class never changes shapes)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_metrics(y_true, y_pred, label_names) -> dict:
    labels = list(range(len(label_names)))
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, n in enumerate(label_names)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "label_names": list(label_names),
    }


def macro_f1(y_true, y_pred, n_labels) -> float:
    return float(precision_recall_fscore_support(y_true, y_pred, labels=list(range(n_labels)),
                                                 average="macro", zero_division=0)[2])
