"""Strict numeric NPY ingestion: quarantine malformed files, never repair in place."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .acquire import digest
from .signals import constant_tail_length, constant_tail_start, window_features, window_starts


def load_numeric(path: Path, length: int = 880, channels: int = 6) -> np.ndarray:
    if any(type(value) is not int or value <= 0 for value in (length, channels)):
        raise ValueError("expected dimensions must be positive integers")
    with path.open("rb") as stream:
        magic = stream.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"invalid NPY magic: {magic.hex()}")
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.dtype.kind not in "fiu" or data.ndim != 3 or data.shape[1:] != (length, channels) or data.shape[0] == 0:
        raise ValueError(f"expected numeric (records,{length},{channels}); got {data.shape}/{data.dtype}")
    expected_bytes = data.offset + data.size * data.dtype.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"NPY payload length mismatch: expected {expected_bytes}, got {path.stat().st_size}")
    if not np.isfinite(data).all():
        raise ValueError("non-finite signal values")
    return data


def inspect_inputs(cache: Path, manifest_path: Path, settings: dict) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaded, files = {}, []
    expected_names = {f"{label:03}_{sensor}.npy" for label in range(settings["classes"]) for sensor in (1, 2)}
    actual_names = [e["fileName"] for e in manifest["files"]]
    if len(actual_names) != len(set(actual_names)) or set(actual_names) != expected_names:
        raise ValueError("source manifest must cover exactly two files for every declared class")
    for entry in manifest["files"]:
        name = entry["fileName"]
        path = cache / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing source {name}; run fetch first")
        sha256, md5 = digest(path), digest(path, "md5")
        if path.stat().st_size != entry["size"] or md5 != entry["md5"].lower():
            raise ValueError(f"Download checksum/size mismatch for {name}; do not analyze an unverified file")
        record = {"file": name, "bytes": path.stat().st_size, "sha256": sha256,
                  "publisher_md5": md5, "publisher_checksum_verified": True,
                  "url": "https://china.scidb.cn/download?fileId=" + entry["id"]}
        try:
            array = load_numeric(path, settings["samples_per_record"], settings["channels_per_sensor_file"])
            loaded[name] = array
            record.update({"status": "usable", "shape": list(array.shape), "dtype": str(array.dtype),
                           "min": float(array.min()), "max": float(array.max())})
        except (ValueError, OSError, EOFError) as exc:
            record.update({"status": "quarantined", "reason": str(exc)})
        files.append(record)
    pairs, excluded = {}, []
    for label in range(settings["classes"]):
        names = [f"{label:03}_{s}.npy" for s in (1, 2)]
        if not all(name in loaded for name in names):
            excluded.append({"class_id": label, "reason": "at least one paired source file is quarantined"})
        elif loaded[names[0]].shape != loaded[names[1]].shape:
            excluded.append({"class_id": label, "reason": "paired sensor array shapes differ"})
        else:
            pairs[label] = (loaded[names[0]], loaded[names[1]])
    return {"files": files, "usable_classes": sorted(pairs), "excluded_classes": excluded,
            "pairing_rule": "same movement ID and row index; matching shape is necessary, not proof of original synchronization",
            "patient_mapping": "not present in the downloaded exercise arrays"}, pairs


def prepare_features(pairs: dict, settings: dict) -> tuple[dict, list[dict]]:
    features, labels, ids, starts, content_ids, tail_fractions, records = [], [], [], [], [], [], []
    limb_content_ids, glove_content_ids = [], []
    size, stride = settings["window_samples"], settings["stride_samples"]
    for label, arrays in sorted(pairs.items()):
        if (len(arrays) != 2 or arrays[0].ndim != 3 or arrays[0].shape != arrays[1].shape
                or arrays[0].shape[2] == 0):
            raise ValueError("paired sensors must have matching record/sample/channel shapes")
        sensor_channels = arrays[0].shape[2]
        for row_index in range(len(arrays[0])):
            signal = np.concatenate((arrays[0][row_index], arrays[1][row_index]), axis=1).astype(np.float64)
            record_id = f"{label:03}/{row_index:04}"
            content_id = hashlib.sha256(np.ascontiguousarray(signal, dtype="<f8").tobytes()).hexdigest()
            limb_content_id = hashlib.sha256(np.ascontiguousarray(signal[:, :sensor_channels], dtype="<f8").tobytes()).hexdigest()
            glove_content_id = hashlib.sha256(np.ascontiguousarray(signal[:, sensor_channels:], dtype="<f8").tobytes()).hexdigest()
            tail_start = constant_tail_start(signal, settings["constant_tail_atol"])
            # A one-sample suffix is tautological, not evidence of a constant tail.
            tail_length = constant_tail_length(signal, settings["constant_tail_atol"])
            offsets = window_starts(len(signal), size, stride)
            records.append({"record_id": record_id, "class_id": label, "row_index": row_index,
                            "signal_sha256": content_id, "limb_signal_sha256": limb_content_id,
                            "glove_signal_sha256": glove_content_id, "window_count": len(offsets),
                            "constant_tail_samples": tail_length,
                            "limb_constant_tail_samples": constant_tail_length(signal[:, :sensor_channels], settings["constant_tail_atol"]),
                            "glove_constant_tail_samples": constant_tail_length(signal[:, sensor_channels:], settings["constant_tail_atol"]),
                            "constant_channels": int(np.count_nonzero(np.ptp(signal, axis=0) <= settings["constant_tail_atol"]))})
            for start in offsets:
                features.append(window_features(signal[start:start + size]))
                labels.append(label)
                ids.append(record_id)
                starts.append(start)
                content_ids.append(content_id)
                limb_content_ids.append(limb_content_id)
                glove_content_ids.append(glove_content_id)
                tail_fractions.append(max(0, start + size - max(start, tail_start)) / size if tail_length else 0.)
    if not features:
        raise ValueError("no usable paired records")
    return {"features": np.asarray(features), "labels": np.asarray(labels, dtype=np.int64),
            "record_ids": np.asarray(ids), "starts": np.asarray(starts, dtype=np.int64),
            "content_ids": np.asarray(content_ids), "limb_content_ids": np.asarray(limb_content_ids),
            "glove_content_ids": np.asarray(glove_content_ids), "tail_fractions": np.asarray(tail_fractions)}, records
