"""Fixed, record-aware comparisons with input, split and output provenance."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import platform
import tempfile

import numpy as np

from .acquire import digest
from .baseline import RidgeClassifier, classification_metrics
from .dataset import inspect_inputs, prepare_features
from .signals import FEATURE_NAMES, interval_overlap_audit
from .splits import (content_group_disjoint_split, stratified_record_split,
                     stratified_window_split, split_leakage)


RESULT_FILES = frozenset({"inventory.json", "summary.json", "per_class.csv", "study.md", "scores.svg"})
PRIVATE_RESULT_FILES = frozenset({"record_audit.csv", "split_assignments.csv.gz", "predictions.csv.gz"})
UPSTREAM_CODE_COMMIT = "1073d6cb147e502cbfccd86c3011e7c487cbdb70"
COMPARISONS = (("window_random", "paired"), ("record_disjoint", "paired"),
               ("content_group_disjoint", "paired"), ("content_group_disjoint", "limb"),
               ("content_group_disjoint", "glove"))


def validate_settings(settings: dict) -> None:
    if not isinstance(settings, dict):
        raise ValueError("study configuration must be a JSON object")
    required = {"dataset_doi", "dataset_version", "classes", "samples_per_record",
                "channels_per_sensor_file", "window_samples", "stride_samples",
                "test_fraction", "seeds", "ridge_alpha", "constant_tail_atol", "features"}
    missing = required - settings.keys()
    if missing:
        raise ValueError(f"study configuration missing fields: {sorted(missing)}")
    for name in ("classes", "samples_per_record", "channels_per_sensor_file", "window_samples", "stride_samples"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if settings["classes"] < 2 or not 2 <= settings["window_samples"] <= settings["samples_per_record"]:
        raise ValueError("study needs at least two classes and a window of 2..samples_per_record samples")
    for name in ("test_fraction", "ridge_alpha", "constant_tail_atol"):
        if type(settings[name]) not in (int, float) or not np.isfinite(settings[name]):
            raise ValueError(f"{name} must be a finite number")
    if not 0 < settings["test_fraction"] < 1 or settings["ridge_alpha"] <= 0 or settings["constant_tail_atol"] < 0:
        raise ValueError("invalid split fraction, ridge alpha or constant-tail tolerance")
    seeds = settings["seeds"]
    if (not isinstance(seeds, list) or not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be a nonempty list of distinct nonnegative integers")
    if settings["features"] != list(FEATURE_NAMES):
        raise ValueError("features must match the implemented fixed feature order")
    for name in ("dataset_doi", "dataset_version"):
        if not isinstance(settings[name], str) or not settings[name].strip():
            raise ValueError(f"{name} must be a nonempty string")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write a headerless empty table")
    if path.suffix == ".gz":
        with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    else:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def source_hashes(root: Path) -> dict:
    paths = [*root.glob("rehab_audit/*.py"), *root.glob("tests/*.py"),
             *root.glob("config/*.json"), *root.glob(".github/workflows/*.yml"), root / "pyproject.toml"]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(paths) if p.is_file()}


def feature_columns(modality: str, channels_per_sensor: int = 6) -> list[int]:
    if type(channels_per_sensor) is not int or channels_per_sensor <= 0:
        raise ValueError("channels_per_sensor must be a positive integer")
    total_channels = 2 * channels_per_sensor
    if modality == "paired":
        return list(range(len(FEATURE_NAMES) * total_channels))
    if modality not in ("limb", "glove"):
        raise ValueError("unknown modality")
    offset = 0 if modality == "limb" else channels_per_sensor
    return [block * total_channels + channel for block in range(len(FEATURE_NAMES))
            for channel in range(offset, offset + channels_per_sensor)]


def content_sharing(content_ids, record_ids, train, test, *, basis: str) -> dict:
    """Audit full-record exact content at the actually selected signal scope."""
    train_groups = np.unique(content_ids[train])
    test_groups = np.unique(content_ids[test])
    seen = np.isin(content_ids[test], train_groups)
    test_record_count = int(len(np.unique(record_ids[test])))
    seen_record_count = int(len(np.unique(record_ids[test][seen])))
    return {"basis": basis, "train_group_count": int(len(train_groups)),
            "test_group_count": int(len(test_groups)),
            "shared_group_count": int(len(np.intersect1d(train_groups, test_groups))),
            "test_window_count": int(len(test)), "test_windows_with_training_content": int(seen.sum()),
            "test_window_seen_fraction": float(seen.mean()), "test_record_count": test_record_count,
            "test_records_with_training_content": seen_record_count,
            "test_record_seen_fraction": seen_record_count / test_record_count}


def evaluate(data: dict, settings: dict) -> tuple[list, list, list, list]:
    validate_settings(settings)
    labels, ids, features = data["labels"], data["record_ids"], data["features"]
    expected_columns = 2 * settings["channels_per_sensor_file"] * len(FEATURE_NAMES)
    if features.ndim != 2 or features.shape != (len(labels), expected_columns):
        raise ValueError("feature matrix does not match the declared channels and feature order")
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError("at least two usable movement classes are required")
    runs, assignments, predictions, per_class = [], [], [], []
    for seed in settings["seeds"]:
        for protocol, splitter, groups in (("window_random", stratified_window_split, ids),
                                            ("record_disjoint", stratified_record_split, ids),
                                            ("content_group_disjoint", content_group_disjoint_split, data["content_ids"])):
            split = splitter(labels, groups, test_fraction=settings["test_fraction"], seed=seed)
            train, test = split.train_indices, split.test_indices
            sharing = split_leakage(split, ids)
            sharing.pop("shared_record_ids")
            sharing.update(interval_overlap_audit(ids, data["starts"], train, test, settings["window_samples"]))
            train_content = np.unique(data["content_ids"][train])
            test_content = np.unique(data["content_ids"][test])
            sharing["train_content_group_count"] = int(len(train_content))
            sharing["test_content_group_count"] = int(len(test_content))
            sharing["shared_content_group_count"] = int(len(np.intersect1d(train_content, test_content)))
            sharing["test_window_exact_record_content_seen_fraction"] = float(np.isin(data["content_ids"][test], train_content).mean())
            # Existing flat content fields refer to the paired signal hash.
            # Keep this scope explicit beside the modality-specific audit.
            sharing["paired_content_sharing"] = content_sharing(
                data["content_ids"], ids, train, test, basis="paired full record: limb and glove channels")
            if protocol == "content_group_disjoint" and sharing["shared_content_group_count"]:
                raise ValueError("content-group split unexpectedly shares complete signal contents")
            assignment = np.zeros(len(labels), dtype=np.uint8)
            assignment[test] = 1
            for index in range(len(labels)):
                assignments.append({"seed": seed, "protocol": protocol, "window_index": index,
                    "record_id": ids[index], "content_id": data["content_ids"][index],
                    "limb_content_id": data["limb_content_ids"][index],
                    "glove_content_id": data["glove_content_ids"][index],
                    "start_sample": int(data["starts"][index]),
                    "class_id": int(labels[index]), "partition": "test" if assignment[index] else "train"})
            modalities = ("paired", "limb", "glove") if protocol == "content_group_disjoint" else ("paired",)
            for modality in modalities:
                selected_content = data["content_ids" if modality == "paired" else f"{modality}_content_ids"]
                # A new dict per run prevents the last modality overwriting
                # the diagnostics of earlier runs that share the same split.
                run_sharing = {**sharing, "selected_modality_content_sharing": content_sharing(
                    selected_content, ids, train, test, basis=f"{modality} full-record input channels")}
                selected = features[:, feature_columns(modality, settings["channels_per_sensor_file"])]
                model = RidgeClassifier(alpha=settings["ridge_alpha"]).fit(selected[train], labels[train])
                predicted = model.predict(selected[test])
                metrics = classification_metrics(labels[test], predicted, classes=classes)
                run_id = f"{protocol}-{modality}-{seed}"
                for index, label in zip(test, predicted, strict=True):
                    predictions.append({"run_id": run_id, "window_index": int(index), "record_id": ids[index],
                                        "content_id": data["content_ids"][index],
                                        "selected_modality_content_id": selected_content[index],
                                        "start_sample": int(data["starts"][index]),
                                        "true_class": int(labels[index]), "predicted_class": int(label)})
                for row in metrics["per_class"]:
                    per_class.append({"run_id": run_id, **row})
                by_record = {}
                for index, prediction in zip(test, predicted, strict=True):
                    item = by_record.setdefault(str(ids[index]), {"true": int(labels[index]), "predictions": []})
                    item["predictions"].append(int(prediction))
                record_true, record_predicted = [], []
                for item in by_record.values():
                    votes = np.array([item["predictions"].count(int(label)) for label in classes])
                    record_true.append(item["true"])
                    record_predicted.append(int(classes[np.argmax(votes)]))
                record_metrics = classification_metrics(np.asarray(record_true), np.asarray(record_predicted), classes=classes)
                runs.append({"run_id": run_id, "seed": seed, "protocol": protocol, "modality": modality,
                             "window_metrics": metrics, "record_vote_metrics": record_metrics, "split_audit": run_sharing,
                             "training_class_support": {str(int(label)): int(np.count_nonzero(labels[train] == label)) for label in classes},
                             "split_assignment_sha256": hashlib.sha256(assignment.tobytes()).hexdigest(),
                             "training_feature_mean": model.mean_.tolist(), "training_feature_scale": model.scale_.tolist()})
                print(f"{run_id}: macro-F1={metrics['macro_f1']:.6f}, overlap={sharing['test_window_overlap_fraction']:.6f}", flush=True)
    return runs, assignments, predictions, per_class


def report_files(output: Path, summary: dict) -> None:
    groups = summary["comparisons"]
    settings = summary["settings"]
    adjacent_overlap = max(0.0, 1 - settings["stride_samples"] / settings["window_samples"])
    lines = ["# REHAB record-aware evaluation", "",
             f"Analyzed {summary['record_count']:,} paired records and {summary['window_count']:,} windows across {len(summary['classes'])} usable movement classes. "
             f"Published files checksum-checked: {summary['source_file_count']}; quarantined files: {summary['quarantined_file_count']}.", "",
             "This is an independent fixed ridge baseline, not a reproduction of the authors' GRU/LSTM scores. "
             "Record rows are not patient IDs; no comparison establishes performance on unseen patients.", "",
             "| Split | Signals | Mean macro-F1 | Seed range | Mean balanced accuracy | Test windows overlapping train | Paired full-record content seen in train | Selected-modality full-record content seen in train |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for row in groups:
        lines.append(f"| {row['protocol']} | {row['modality']} | {row['macro_f1_mean']:.4f} | {row['macro_f1_min']:.4f}–{row['macro_f1_max']:.4f} | {row['balanced_accuracy_mean']:.4f} | {row['overlap_fraction_mean']:.2%} | {row['paired_content_seen_fraction_mean']:.2%} | {row['selected_modality_content_seen_fraction_mean']:.2%} |")
    lines.extend(["", "![Split and signal comparisons](scores.svg)", "",
                  "Seeds vary the split, not the data acquisition. Seed ranges are descriptive, not confidence intervals. "
                  f"Windows overlap by {adjacent_overlap:.0%} at adjacent starts; repeated windows are not independent participants. "
                  "The window-random comparator is class-stratified and seeded, rather than a bit-for-bit replay of the upstream unseeded split.", "",
                  "The content-group holdout globally shuffles unique full-record signal hashes, without class stratification. "
                  "All copies of a content hash, including copies assigned different labels, stay on one side. "
                  "A seed that removes training or test support for any class is rejected without redraw. "
                  "The protocols have different held-out examples and class supports; score differences do not isolate a causal effect of leakage. "
                  "Exact-content separation also does not rule out near duplicates, shared participants or shared sessions.", "",
                  "Content groups are defined by the concatenated limb-and-glove record. Two distinct paired records can still have "
                  "identical limb-only or glove-only contents. The single-modality comparisons retain the same paired-content split for "
                  "comparability; they are not guaranteed to exclude duplicates of the selected input block. The last two table columns "
                  "separately report paired-content and selected-modality sharing as fractions of test windows. Per-run audits also give "
                  "record-level counts and fractions. Source-row hashes for both blocks are saved only in the locally generated record audit.", "",
                  "## Input audit", ""])
    for row in summary["excluded_classes"]:
        lines.append(f"- Movement class {row['class_id']:03d}: excluded from all paired and single-modality comparisons — {row['reason']}.")
    lines.extend(["", f"{summary['records_with_constant_tail_200']} records have an all-channel constant suffix of at least 200 samples. "
                  "After preprocessing, a constant suffix is only a candidate-padding indicator; it may also be stationary data. "
                  "No inferred suffix is silently removed or relabeled as known padding.", "",
                  f"Exact repeated full-record contents beyond their first occurrence: {summary['duplicate_record_contents']}. "
                  f"Repeated-content groups: {summary['duplicate_content_group_count']}; "
                  f"groups spanning more than one movement label: {summary['cross_class_content_group_count']}. "
                  "No repeated or flat record is removed. Per-run content-sharing statistics are retained separately from row-ID sharing; "
                  "locally generated record audit and split tables map the content hashes back to every source row. "
                  "These row-level hashes, assignments and predictions are not included in the public results.", "",
                  "## Reproduce and inspect", "",
                  "`rehab-audit fetch` downloads the versioned public files. `rehab-audit run` regenerates these outputs. "
                  "`rehab-audit verify` checks source and result hashes; `rehab-audit check` additionally revalidates cached inputs, "
                  "reruns the study, and compares every public result file (complete JSON structures with 1e-10 float tolerance; other files exactly). "
                  "When a local record-level directory is present, check also requires all three private tables and compares them exactly after gzip decoding.", "",
                  "[Input inventory](inventory.json) · [Per-class metrics](per_class.csv) · "
                  "[Full summary and confusion matrices](summary.json) · [Provenance](manifest.json)", "",
                  "Record audit, split assignments and held-out predictions are generated under the local `record-level/` subdirectory. "
                  "They are excluded from the public manifest and publication; the public summary retains aggregate duplicate-group counts only.", "",
                  "The limb/glove/paired comparisons use exactly the same content-group split, usable classes and windows. "
                  "Training-set-only standardization precedes fixed-alpha ridge fitting; no hyperparameter is selected on the test scores. "
                  "Features are per-channel mean, standard deviation, range, RMS and mean absolute first difference; angle wrapping is not corrected. "
                  "The limb/glove ablation tests information available in each signal block, not robustness to unexpected sensor loss.", ""])
    (output / "study.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    height = 138 + 48 * len(groups)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
           f'<rect width="900" height="{height}" fill="#f8fafc"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#172033;font-size:13px}</style>',
           '<text x="24" y="29" font-size="19">Fixed ridge baseline: split and signal-block comparisons</text>',
           f'<text x="24" y="52">Bar: mean macro-F1; dots: {len(settings["seeds"])} split seeds. Same usable classes throughout.</text>']
    for index, row in enumerate(groups):
        y = 90 + index * 48
        label = f"{row['protocol']} / {row['modality']}"
        color = "#d97706" if row["protocol"] == "window_random" else "#2563eb"
        svg.extend([f'<text x="24" y="{y + 15}">{label}</text>',
                    f'<rect x="350" y="{y}" width="400" height="22" fill="#e2e8f0"/>',
                    f'<rect x="350" y="{y}" width="{400 * row["macro_f1_mean"]:.3f}" height="22" fill="{color}"/>',
                    f'<text x="800" y="{y + 15}">{row["macro_f1_mean"]:.4f}</text>'])
        for score in row["macro_f1_values"]:
            svg.append(f'<circle cx="{350 + 400 * score:.3f}" cy="{y + 11}" r="4" fill="#fff" stroke="#0f172a"/>')
    svg.extend([f'<text x="350" y="{height - 28}">0</text><text x="746" y="{height - 28}">1</text>', '</svg>'])
    (output / "scores.svg").write_text("\n".join(svg) + "\n", encoding="utf-8", newline="\n")


def run_study(root: Path, cache: Path, output: Path) -> dict:
    settings = json.loads((root / "config/study.json").read_text(encoding="utf-8"))
    validate_settings(settings)
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {p.name for p in output.iterdir() if p.is_file()} - RESULT_FILES - {"manifest.json"}
    if unexpected:
        raise ValueError(f"output contains unrelated files; choose a dedicated output directory: {sorted(unexpected)}")
    private_output = output / "record-level"
    private_output.mkdir(exist_ok=True)
    unexpected_private = {p.name for p in private_output.iterdir() if p.is_file()} - PRIVATE_RESULT_FILES
    if unexpected_private:
        raise ValueError(f"record-level output contains unrelated files: {sorted(unexpected_private)}")
    inventory, pairs = inspect_inputs(cache, root / "config/source_files.json", settings)
    data, records = prepare_features(pairs, settings)
    print(f"Prepared {len(records)} records / {len(data['labels'])} windows", flush=True)
    runs, assignments, predictions, per_class = evaluate(data, settings)
    comparisons = []
    for protocol, modality in COMPARISONS:
        subset = [r for r in runs if r["protocol"] == protocol and r["modality"] == modality]
        f1 = [r["window_metrics"]["macro_f1"] for r in subset]
        comparisons.append({"protocol": protocol, "modality": modality, "macro_f1_values": f1,
                            "macro_f1_mean": float(np.mean(f1)), "macro_f1_min": min(f1), "macro_f1_max": max(f1),
                            "balanced_accuracy_mean": float(np.mean([r["window_metrics"]["balanced_accuracy"] for r in subset])),
                            "overlap_fraction_mean": float(np.mean([r["split_audit"]["test_window_overlap_fraction"] for r in subset])),
                            "exact_content_seen_fraction_mean": float(np.mean([r["split_audit"]["test_window_exact_record_content_seen_fraction"] for r in subset])),
                            "paired_content_seen_fraction_mean": float(np.mean([r["split_audit"]["paired_content_sharing"]["test_window_seen_fraction"] for r in subset])),
                            "selected_modality_content_seen_fraction_mean": float(np.mean([r["split_audit"]["selected_modality_content_sharing"]["test_window_seen_fraction"] for r in subset]))})
    content_groups = {}
    for record in records:
        group = content_groups.setdefault(record["signal_sha256"], {"record_count": 0, "class_ids": set()})
        group["record_count"] += 1
        group["class_ids"].add(record["class_id"])
    summary = {"schema_version": 1, "settings": settings, "record_count": len(records),
               "window_count": len(data["labels"]), "classes": inventory["usable_classes"],
               "source_file_count": len(inventory["files"]),
               "quarantined_file_count": sum(r["status"] == "quarantined" for r in inventory["files"]),
               "excluded_classes": inventory["excluded_classes"],
               "records_with_constant_tail_200": sum(r["constant_tail_samples"] >= 200 for r in records),
               "duplicate_record_contents": len(records) - len({r["signal_sha256"] for r in records}),
               "duplicate_limb_record_contents": len(records) - len({r["limb_signal_sha256"] for r in records}),
               "duplicate_glove_record_contents": len(records) - len({r["glove_signal_sha256"] for r in records}),
               "content_group_basis": "SHA-256 of full paired limb-and-glove record, canonical little-endian float64",
               "content_group_count": len(content_groups),
               "duplicate_content_group_count": sum(group["record_count"] > 1 for group in content_groups.values()),
               "cross_class_content_group_count": sum(len(group["class_ids"]) > 1 for group in content_groups.values()),
               "comparisons": comparisons, "runs": runs}
    write_json(output / "inventory.json", inventory)
    write_json(output / "summary.json", summary)
    write_csv(private_output / "record_audit.csv", records)
    write_csv(output / "per_class.csv", per_class)
    write_csv(private_output / "split_assignments.csv.gz", assignments)
    write_csv(private_output / "predictions.csv.gz", predictions)
    report_files(output, summary)
    results = {name: digest(output / name) for name in sorted(RESULT_FILES)}
    write_json(output / "manifest.json", {"schema_version": 1, "source_hashes_lf": source_hashes(root),
               "result_hashes": results, "environment": {"python": platform.python_version(), "numpy": np.__version__},
               "dataset_doi": settings["dataset_doi"], "dataset_version": settings["dataset_version"],
               "upstream_code_commit": UPSTREAM_CODE_COMMIT,
               "reproduce_command": "rehab-audit fetch && rehab-audit run",
               "scope": "independent implementation; downloaded source arrays unchanged; no patient mapping inferred"})
    return summary


def verify(root: Path, output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    settings = json.loads((root / "config/study.json").read_text(encoding="utf-8"))
    validate_settings(settings)
    if (manifest["schema_version"] != 1 or manifest["dataset_doi"] != settings["dataset_doi"]
            or manifest["dataset_version"] != settings["dataset_version"]
            or manifest["upstream_code_commit"] != UPSTREAM_CODE_COMMIT):
        raise ValueError("manifest schema or dataset/upstream identity changed")
    if source_hashes(root) != manifest["source_hashes_lf"]:
        raise ValueError("study source/config/test/workflow hashes changed")
    actual = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    if set(actual) != RESULT_FILES or actual != manifest["result_hashes"]:
        raise ValueError("result hashes or file coverage changed")


def _compare_json(current, original, location: str) -> None:
    """Compare all semantic fields; allow only small floating-point replay drift."""
    if type(current) is not type(original):
        raise ValueError(f"replay changed type at {location}")
    if isinstance(current, dict):
        if current.keys() != original.keys():
            raise ValueError(f"replay changed fields at {location}")
        for key in current:
            _compare_json(current[key], original[key], f"{location}.{key}")
    elif isinstance(current, list):
        if len(current) != len(original):
            raise ValueError(f"replay changed length at {location}")
        for index, (a, b) in enumerate(zip(current, original, strict=True)):
            _compare_json(a, b, f"{location}[{index}]")
    elif isinstance(current, float):
        if (not np.isfinite(current) or not np.isfinite(original)
                or not np.isclose(current, original, atol=1e-10, rtol=1e-10)):
            raise ValueError(f"replay changed float at {location}")
    elif current != original:
        raise ValueError(f"replay changed value at {location}")


def check(root: Path, cache: Path, output: Path) -> None:
    verify(root, output)
    relative_files = [Path(name) for name in sorted(RESULT_FILES)]
    private_output = output / "record-level"
    if private_output.exists():
        if not private_output.is_dir() or {p.name for p in private_output.iterdir()} != PRIVATE_RESULT_FILES:
            raise ValueError("local record-level file coverage changed")
        relative_files.extend(Path("record-level") / name for name in sorted(PRIVATE_RESULT_FILES))
    with tempfile.TemporaryDirectory(prefix="rehab-replay-") as temporary:
        replay = Path(temporary)
        run_study(root, cache, replay)
        for filename in relative_files:
            current, original = replay / filename, output / filename
            if current.suffix == ".json":
                _compare_json(json.loads(current.read_text(encoding="utf-8")),
                              json.loads(original.read_text(encoding="utf-8")), filename.as_posix())
            elif current.suffix == ".gz":
                with gzip.open(current, "rb") as a, gzip.open(original, "rb") as b:
                    if a.read() != b.read():
                        raise ValueError(f"replay changed {filename}")
            elif current.read_bytes() != original.read_bytes():
                raise ValueError(f"replay changed {filename}")
