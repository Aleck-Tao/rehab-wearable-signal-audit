"""Signal summaries preserve record identity and do not assume patient metadata."""
from __future__ import annotations

import numpy as np


FEATURE_NAMES = ("mean", "std", "range", "rms", "mean_abs_difference")


def constant_tail_start(record: np.ndarray, atol: float = 1e-10) -> int:
    """Index of an all-channel constant suffix, a proxy rather than known padding."""
    values = np.asarray(record, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("record must be a finite time-by-channel matrix with >=2 samples")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    equal_to_last = np.all(np.abs(values - values[-1]) <= atol, axis=1)
    different = np.flatnonzero(~equal_to_last)
    return int(different[-1] + 1) if len(different) else 0


def constant_tail_length(record: np.ndarray, atol: float = 1e-10) -> int:
    """Count a constant suffix only when at least two samples support it."""
    length = len(record) - constant_tail_start(record, atol)
    return length if length >= 2 else 0


def window_features(window: np.ndarray) -> np.ndarray:
    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("window must be a finite nonempty time-by-channel matrix")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = np.concatenate((values.mean(axis=0), values.std(axis=0),
                                     np.ptp(values, axis=0), np.sqrt(np.mean(values * values, axis=0)),
                                     np.mean(np.abs(np.diff(values, axis=0)), axis=0)))
    except FloatingPointError as exc:
        raise ValueError("feature computation overflowed; check signal scale") from exc
    if not np.isfinite(result).all():
        raise ValueError("non-finite window features")
    return result


def window_starts(length: int, size: int, stride: int) -> list[int]:
    if any(type(x) is not int or x <= 0 for x in (length, size, stride)):
        raise ValueError("length, size and stride must be positive integers")
    return list(range(0, length - size + 1, stride))


def interval_overlap_audit(record_ids, starts, train, test, window_size: int) -> dict:
    """Measure overlap in sample coordinates within each published source row."""
    ids = np.asarray(record_ids)
    offsets = np.asarray(starts)
    if (ids.ndim != 1 or offsets.ndim != 1 or len(ids) != len(offsets)
            or type(window_size) is not int or window_size <= 0
            or (offsets.size and (offsets.dtype.kind not in "iu" or np.any(offsets < 0)))):
        raise ValueError("invalid window metadata")
    selections = []
    for selection in (train, test):
        indices = np.asarray(selection)
        if (indices.ndim != 1 or (indices.size and
                (indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= len(ids))))):
            raise ValueError("split indices must be one-dimensional valid integers")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("split indices must not be duplicated")
        selections.append(indices.astype(np.int64))
    train, test = selections
    if np.intersect1d(train, test).size:
        raise ValueError("train and test indices must be disjoint")
    by_record: dict[str, list[int]] = {}
    for index in train:
        by_record.setdefault(str(ids[index]), []).append(int(offsets[index]))
    overlapping, maximum_fractions = 0, []
    for index in test:
        same_source = by_record.get(str(ids[index]), [])
        max_shared = max((max(0, window_size - abs(int(offsets[index]) - start))
                          for start in same_source), default=0)
        overlapping += int(max_shared > 0)
        maximum_fractions.append(max_shared / window_size)
    return {
        "test_windows_overlapping_training": overlapping,
        "test_window_overlap_fraction": overlapping / len(test) if len(test) else None,
        "mean_max_pairwise_sample_overlap": float(np.mean(maximum_fractions)) if len(test) else None,
    }
