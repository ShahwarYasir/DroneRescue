"""Evaluation helpers: the same numbers for every model, on Val, all of Test and the unexposed part of Test."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import config as C


def summarize(y_true, y_pred):
    labels = list(range(len(C.CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "macro_f1": float(f.mean()),
        "min_class_recall": float(r.min()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(C.CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def evaluate_all(y_val, p_val, y_test, p_test, exposed_test):
    """Return metrics for Val, Test (all) and Test without look-alikes in Train/Val ('unexposed')."""
    exposed_test = np.asarray(exposed_test, dtype=bool)
    out = {"val": summarize(y_val, p_val), "test_all": summarize(y_test, p_test)}
    if (~exposed_test).sum() > 0:
        out["test_unexposed"] = summarize(np.asarray(y_test)[~exposed_test], np.asarray(p_test)[~exposed_test])
    return out


def plot_confusion(ax, cm, title):
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    names = [C.DISPLAY_NAMES[c].split(" /")[0] for c in C.CLASSES]
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{100 * norm[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
