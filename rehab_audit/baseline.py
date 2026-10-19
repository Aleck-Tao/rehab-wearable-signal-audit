"""A small NumPy ridge classifier, with preprocessing fitted on training data only."""

import math

import numpy as np


def _features(values):
    values = np.asarray(values)
    if values.ndim != 2 or not all(values.shape) or values.dtype.kind not in "fiu":
        raise ValueError("features must be a nonempty two-dimensional numeric array")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("features must be finite")
    return values


def _labels(values, expected_size=None, name="labels"):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty one-dimensional integer array")
    if expected_size is not None and values.size != expected_size:
        raise ValueError(f"{name} length must match the number of examples")
    return values


class RidgeClassifier:
    """Least-squares one-hot classifier; default fixed alpha is 1.0.

    Population mean/std are fitted only in fit(). Constant columns use scale 1.
    The intercept is not penalized. Exact score ties select the smallest label.
    This is an independent baseline, not the upstream paper's neural model.
    """

    def __init__(self, alpha=1.0):
        if not isinstance(alpha, (int, float, np.integer, np.floating)) or isinstance(alpha, bool):
            raise ValueError("alpha must be a finite positive number")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be a finite positive number")
        self.alpha = float(alpha)
        self._fitted = False

    def fit(self, features, labels):
        self._fitted = False
        features = _features(features)
        labels = _labels(labels, features.shape[0])
        self.classes_ = np.unique(labels)
        with np.errstate(over="ignore", invalid="ignore"):
            self.mean_ = features.mean(axis=0)
            self.scale_ = features.std(axis=0)
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.scale_).all():
            raise ValueError("feature magnitudes overflow training mean/std")
        self.scale_[self.scale_ == 0] = 1.0
        normalized = (features - self.mean_) / self.scale_
        design = np.column_stack((normalized, np.ones(features.shape[0])))
        targets = (labels[:, None] == self.classes_[None, :]).astype(np.float64)
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[-1, -1] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
        if not np.isfinite(coefficients).all():
            raise ValueError("ridge solution is nonfinite")
        self.coef_ = coefficients[:-1]
        self.intercept_ = coefficients[-1]
        self._fitted = True
        return self

    def decision_function(self, features):
        if not self._fitted:
            raise RuntimeError("fit the classifier before predicting")
        features = _features(features)
        if features.shape[1] != self.mean_.size:
            raise ValueError("prediction feature count differs from training")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = ((features - self.mean_) / self.scale_) @ self.coef_ + self.intercept_
        if not np.isfinite(scores).all():
            raise ValueError("prediction scores are nonfinite")
        return scores

    def predict(self, features):
        scores = self.decision_function(features)
        return self.classes_[np.argmax(scores, axis=1)]


def classification_metrics(y_true, y_pred, classes=None):
    """JSON-safe metrics; rows=true and columns=predicted in confusion_matrix.

    Explicit classes are sorted and retained even when support is zero. Recall,
    precision and F1 use zero for zero denominators; macro-F1 and balanced
    accuracy average across *all* retained classes, including zero-support ones.
    With classes omitted, the union of observed true/predicted labels is used.
    Empty evaluation sets and labels outside the declared classes are rejected.
    """
    y_true = _labels(y_true, name="y_true")
    y_pred = _labels(y_pred, y_true.size, name="y_pred")
    if classes is None:
        classes = np.unique(np.concatenate((y_true, y_pred)))
    else:
        classes = _labels(classes, name="classes")
        if np.unique(classes).size != classes.size:
            raise ValueError("classes must not contain duplicates")
        classes = np.sort(classes)
    if not np.isin(y_true, classes).all() or not np.isin(y_pred, classes).all():
        raise ValueError("true and predicted labels must belong to declared classes")
    confusion = np.zeros((classes.size, classes.size), dtype=np.int64)
    np.add.at(confusion, (np.searchsorted(classes, y_true),
                          np.searchsorted(classes, y_pred)), 1)
    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    tp = np.diag(confusion).astype(np.float64)
    precision = np.divide(tp, predicted_support, out=np.zeros_like(tp), where=predicted_support > 0)
    recall = np.divide(tp, true_support, out=np.zeros_like(tp), where=true_support > 0)
    f1 = np.divide(2 * tp, true_support + predicted_support,
                   out=np.zeros_like(tp), where=(true_support + predicted_support) > 0)
    return {
        "accuracy": float(tp.sum() / y_true.size),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall.mean()),
        "classes": classes.tolist(),
        "confusion_matrix": confusion.tolist(),
        "zero_support_policy": "retain declared classes; undefined precision, recall and F1 are zero",
        "per_class": [
            {"label": int(label), "support": int(true_support[index]),
             "predicted_count": int(predicted_support[index]),
             "precision": float(precision[index]), "recall": float(recall[index]),
             "f1": float(f1[index])}
            for index, label in enumerate(classes)
        ],
    }
