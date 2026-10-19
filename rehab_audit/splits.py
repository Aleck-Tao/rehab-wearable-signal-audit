"""Deterministic window and record splits; record identity is not patient identity."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Split:
    train_indices: np.ndarray
    test_indices: np.ndarray


def _labels_and_records(labels, record_ids, *, require_single_label=True):
    labels = np.asarray(labels)
    record_ids = np.asarray(record_ids)
    if labels.ndim != 1 or labels.size == 0 or labels.dtype.kind not in "iu":
        raise ValueError("labels must be a nonempty one-dimensional integer array")
    if record_ids.ndim != 1 or record_ids.shape != labels.shape:
        raise ValueError("record_ids must be one-dimensional and aligned with labels")
    if record_ids.dtype.kind not in "US" and not (
        record_ids.dtype.kind == "O"
        and all(isinstance(value, str) for value in record_ids)
    ):
        raise ValueError("record_ids must contain strings")
    record_ids = record_ids.astype(str)
    # One expected-O(n) pass, rather than one full-array mask per record.
    record_labels = {}
    for record, label in zip(record_ids, labels):
        if not record.strip():
            raise ValueError("record_ids must not be empty or whitespace")
        if require_single_label and record_labels.setdefault(record, label) != label:
            raise ValueError(f"record {record!r} occurs under more than one label")
    return labels, record_ids


def _test_count(count, test_fraction):
    if not isinstance(test_fraction, (int, float, np.integer, np.floating)) or isinstance(test_fraction, bool):
        raise ValueError("test_fraction must be finite and strictly between zero and one")
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be finite and strictly between zero and one")
    if count < 2:
        raise ValueError("each class needs at least two independent split units")
    # Round upward, then reserve at least one training unit for small classes.
    return min(count - 1, max(1, math.ceil(count * test_fraction)))


def stratified_window_split(labels, record_ids, test_fraction=0.2, seed=0):
    """Split individual windows per class, allowing shared source records.

    Each class contributes ceil(n * test_fraction) test windows, capped to
    preserve one training window. This deliberately leakage-prone comparator
    must not be described as independent-record or independent-patient testing.
    """
    labels, record_ids = _labels_and_records(labels, record_ids)
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        count = _test_count(len(indices), test_fraction)
        indices = rng.permutation(indices)
        test.extend(indices[:count])
        train.extend(indices[count:])
    return Split(np.sort(np.asarray(train, dtype=np.int64)),
                 np.sort(np.asarray(test, dtype=np.int64)))


def stratified_record_split(labels, record_ids, test_fraction=0.2, seed=0):
    """Split unique original records per class, keeping all their windows together.

    The fraction applies to record counts, not window counts. Records can have
    different lengths. No patient-level claim is possible without patient IDs.
    """
    labels, record_ids = _labels_and_records(labels, record_ids)
    rng = np.random.default_rng(seed)
    # unique() sorts IDs, retaining the original per-class lexical ordering.
    # Integer record codes avoid repeated string membership tests over all
    # windows; indexing test_records[inverse] expands the final selection once.
    records, first_indices, inverse = np.unique(
        record_ids, return_index=True, return_inverse=True)
    record_labels = labels[first_indices]
    test_records = np.zeros(records.size, dtype=bool)
    for label in np.unique(labels):
        record_indices = np.flatnonzero(record_labels == label)
        count = _test_count(record_indices.size, test_fraction)
        record_indices = rng.permutation(record_indices)
        test_records[record_indices[:count]] = True
    test_mask = test_records[inverse]
    return Split(np.flatnonzero(~test_mask).astype(np.int64),
                 np.flatnonzero(test_mask).astype(np.int64))


def content_group_disjoint_split(labels, content_ids, test_fraction=0.2, seed=0):
    """Globally split unique content hashes; cross-label duplicate groups stay together.

    This is not stratified and not a patient split. The fraction applies to
    unique content groups. A fixed draw is rejected, not retried, if any class
    lacks training or test examples; therefore no split is selected by scores.
    """
    labels, content_ids = _labels_and_records(labels, content_ids, require_single_label=False)
    groups, inverse = np.unique(content_ids, return_inverse=True)
    count = _test_count(len(groups), test_fraction)
    order = np.random.default_rng(seed).permutation(len(groups))
    test_groups = np.zeros(len(groups), dtype=bool)
    test_groups[order[:count]] = True
    test_mask = test_groups[inverse]
    classes = np.unique(labels)
    if (not np.array_equal(np.unique(labels[test_mask]), classes)
            or not np.array_equal(np.unique(labels[~test_mask]), classes)):
        raise ValueError("fixed global content-group split leaves a class without training or test support; no redraw performed")
    return Split(np.flatnonzero(~test_mask).astype(np.int64),
                 np.flatnonzero(test_mask).astype(np.int64))


def split_leakage(split, record_ids):
    """Report record sharing, not sample overlap, patient leakage or performance bias."""
    record_ids = np.asarray(record_ids)
    # Reuse record validation while giving every record the same dummy label.
    _, record_ids = _labels_and_records(np.zeros(record_ids.size, dtype=int), record_ids)
    parts = []
    for name, values in (("train", split.train_indices), ("test", split.test_indices)):
        indices = np.asarray(values)
        if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
            raise ValueError(f"{name} indices must be nonempty one-dimensional integers")
        if np.any(indices < 0) or np.any(indices >= record_ids.size):
            raise ValueError(f"{name} index out of bounds")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"{name} indices contain duplicates")
        parts.append(indices)
    train, test = parts
    if np.intersect1d(train, test).size:
        raise ValueError("train and test indices must be disjoint")
    if train.size + test.size != record_ids.size:
        raise ValueError("split must partition all supplied windows")
    records, inverse = np.unique(record_ids, return_inverse=True)
    in_train = np.zeros(records.size, dtype=bool)
    in_test = np.zeros(records.size, dtype=bool)
    in_train[inverse[train]] = True
    in_test[inverse[test]] = True
    shared = records[in_train & in_test]
    contaminated_count = int(in_train[inverse[test]].sum())
    return {
        "train_window_count": int(train.size),
        "test_window_count": int(test.size),
        "train_record_count": int(in_train.sum()),
        "test_record_count": int(in_test.sum()),
        "shared_record_count": int(shared.size),
        "shared_record_ids": shared.tolist(),
        "test_windows_with_training_record": contaminated_count,
        "test_window_shared_record_fraction": contaminated_count / int(test.size),
    }
