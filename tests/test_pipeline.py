"""Small real-NPY end-to-end studies; no network or public data required."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import gzip
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from rehab_audit.acquire import digest
from rehab_audit.cli import main
from rehab_audit.dataset import inspect_inputs, prepare_features
from rehab_audit.pipeline import (COMPARISONS, PRIVATE_RESULT_FILES, RESULT_FILES, check, evaluate, feature_columns,
                                   run_study, validate_settings, verify, write_json)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rehab-pipeline-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cache, self.output = self.root / "cache", self.root / "results"
        self.cache.mkdir()
        (self.root / "config").mkdir()
        (self.root / "rehab_audit").mkdir()
        code = Path(__file__).resolve().parents[1] / "rehab_audit"
        for path in code.glob("*.py"):
            shutil.copyfile(path, self.root / "rehab_audit" / path.name)
        (self.root / "pyproject.toml").write_text("# temporary test fixture\n", encoding="utf-8")
        self.settings = {
            "dataset_doi": "10.0000/synthetic-test-fixture", "dataset_version": "test-only",
            "classes": 3, "samples_per_record": 24, "channels_per_sensor_file": 6,
            "window_samples": 8, "stride_samples": 2, "test_fraction": 0.2,
            "seeds": [101, 307, 809], "ridge_alpha": 1.0, "constant_tail_atol": 1e-10,
            "features": ["mean", "std", "range", "rms", "mean_abs_difference"],
        }
        write_json(self.root / "config/study.json", self.settings)
        rng = np.random.default_rng(42)
        source = {}
        for label in range(3):
            for sensor in (1, 2):
                source[label, sensor] = rng.normal(size=(20, 24, 6)) + 2 * label + sensor / 10
        # One duplicate within class 0 and one shared across classes 0 and 1.
        # The latter must stay on one side without pretending labels are unique.
        for sensor in (1, 2):
            source[0, sensor][1] = source[0, sensor][0]
            source[1, sensor][0] = source[0, sensor][0]
        entries = []
        for label in range(3):
            for sensor in (1, 2):
                path = self.cache / f"{label:03}_{sensor}.npy"
                if label == 2 and sensor == 1:
                    path.write_bytes(b"published-but-malformed-array")
                else:
                    np.save(path, source[label, sensor], allow_pickle=False)
                entries.append({"fileName": path.name, "id": f"fixture{label}{sensor}",
                                "size": path.stat().st_size, "md5": digest(path, "md5")})
        write_json(self.root / "config/source_files.json", {"files": entries})
        with redirect_stdout(io.StringIO()):
            self.summary = run_study(self.root, self.cache, self.output)

    def refresh_result_digest(self, filename):
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["result_hashes"][filename] = digest(self.output / filename)
        write_json(manifest_path, manifest)

    def run_check(self):
        with redirect_stdout(io.StringIO()):
            check(self.root, self.cache, self.output)

    def test_full_replay_covers_all_outputs_and_quarantined_input(self):
        self.assertEqual(self.summary["record_count"], 40)
        self.assertEqual(self.summary["window_count"], 360)
        self.assertEqual(self.summary["classes"], [0, 1])
        self.assertEqual(self.summary["quarantined_file_count"], 1)
        self.assertEqual(self.summary["excluded_classes"][0]["class_id"], 2)
        self.assertEqual(len(self.summary["runs"]), 15)
        self.assertEqual(len(self.summary["comparisons"]), len(COMPARISONS))
        self.assertEqual(self.summary["duplicate_record_contents"], 2)
        self.assertEqual(self.summary["cross_class_content_group_count"], 1)
        self.assertNotIn("cross_class_content_groups", self.summary)
        self.assertEqual({p.name for p in self.output.iterdir()}, RESULT_FILES | {"manifest.json", "record-level"})
        self.assertEqual({p.name for p in (self.output / "record-level").iterdir()}, PRIVATE_RESULT_FILES)
        manifest = json.loads((self.output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["result_hashes"]), RESULT_FILES)
        verify(self.root, self.output)
        self.run_check()

    def test_modality_runs_use_identical_content_splits_and_training_only_statistics(self):
        _, pairs = inspect_inputs(self.cache, self.root / "config/source_files.json", self.settings)
        data, _ = prepare_features(pairs, self.settings)
        with gzip.open(self.output / "record-level/split_assignments.csv.gz", "rt", encoding="utf-8", newline="") as stream:
            assignments = list(csv.DictReader(stream))
        for seed in self.settings["seeds"]:
            runs = [run for run in self.summary["runs"]
                    if run["seed"] == seed and run["protocol"] == "content_group_disjoint"]
            self.assertEqual({run["modality"] for run in runs}, {"paired", "limb", "glove"})
            self.assertEqual(len({run["split_assignment_sha256"] for run in runs}), 1)
            rows = [row for row in assignments if row["seed"] == str(seed)
                    and row["protocol"] == "content_group_disjoint"]
            train = np.array([int(row["window_index"]) for row in rows if row["partition"] == "train"])
            test = np.array([int(row["window_index"]) for row in rows if row["partition"] == "test"])
            train_groups = {row["content_id"] for row in rows if row["partition"] == "train"}
            test_groups = {row["content_id"] for row in rows if row["partition"] == "test"}
            self.assertFalse(train_groups & test_groups)
            self.assertEqual(len(rows), 360)
            for run in runs:
                selected = data["features"][:, feature_columns(run["modality"])]
                np.testing.assert_allclose(run["training_feature_mean"], selected[train].mean(axis=0))
                scale = selected[train].std(axis=0)
                scale[scale == 0] = 1
                np.testing.assert_allclose(run["training_feature_scale"], scale)
                self.assertEqual(run["split_audit"]["shared_record_count"], 0)
                self.assertEqual(run["split_audit"]["test_window_exact_record_content_seen_fraction"], 0)
                self.assertEqual(sum(item["support"] for item in run["window_metrics"]["per_class"]), len(test))
                self.assertEqual(sum(run["training_class_support"].values()), len(train))

    def test_source_change_and_new_source_file_are_rejected(self):
        path = self.root / "rehab_audit" / "added_source.py"
        path.write_text("# new unrecorded source\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source/config/test/workflow"):
            verify(self.root, self.output)

    def test_single_modality_duplicates_are_reported_despite_distinct_paired_records(self):
        rng = np.random.default_rng(53)
        limb = np.ones((20, 24, 6))
        pairs = {label: (limb.copy(), rng.normal(size=(20, 24, 6)) + label)
                 for label in (0, 1)}
        data, records = prepare_features(pairs, self.settings)
        self.assertEqual(len({row["signal_sha256"] for row in records}), 40)
        self.assertEqual(len({row["limb_signal_sha256"] for row in records}), 1)
        self.assertEqual(len({row["glove_signal_sha256"] for row in records}), 40)
        self.assertEqual(np.unique(data["content_ids"]).size, 40)
        self.assertEqual(np.unique(data["limb_content_ids"]).size, 1)
        self.assertEqual(np.unique(data["glove_content_ids"]).size, 40)
        with redirect_stdout(io.StringIO()):
            runs, assignments, predictions, _ = evaluate(data, self.settings)
        for seed in self.settings["seeds"]:
            selected_runs = {run["modality"]: run for run in runs
                             if run["protocol"] == "content_group_disjoint" and run["seed"] == seed}
            self.assertEqual(len({run["split_assignment_sha256"] for run in selected_runs.values()}), 1)
            for modality, run in selected_runs.items():
                audit = run["split_audit"]
                self.assertEqual(audit["paired_content_sharing"]["test_window_seen_fraction"], 0)
                selected = audit["selected_modality_content_sharing"]
                expected = 1 if modality == "limb" else 0
                self.assertEqual(selected["test_window_seen_fraction"], expected)
                self.assertEqual(selected["test_record_seen_fraction"], expected)
                self.assertEqual(selected["shared_group_count"], expected)
        self.assertIn("limb_content_id", assignments[0])
        self.assertIn("glove_content_id", assignments[0])
        self.assertIn("selected_modality_content_id", predictions[0])

    def test_result_corruption_is_rejected_before_replay(self):
        path = self.output / "per_class.csv"
        path.write_bytes(path.read_bytes() + b"corrupted\n")
        with self.assertRaisesRegex(ValueError, "result hashes"):
            self.run_check()

    def test_cached_input_corruption_is_rejected_by_replay(self):
        path = self.cache / "000_1.npy"
        changed = bytearray(path.read_bytes())
        changed[-1] ^= 1
        path.write_bytes(changed)
        verify(self.root, self.output)  # Static hashes intentionally do not claim to re-read cached inputs.
        with self.assertRaisesRegex(ValueError, "checksum/size mismatch"):
            self.run_check()

    def test_public_only_checkout_replays_without_private_tables(self):
        (self.output / "record-level").rename(self.root / "saved-private-tables")
        verify(self.root, self.output)
        self.run_check()
        self.assertFalse((self.output / "record-level").exists())

    def test_local_private_table_tamper_is_rejected_by_replay(self):
        path = self.output / "record-level/record_audit.csv"
        path.write_bytes(path.read_bytes() + b"tampered-private-table\n")
        verify(self.root, self.output)  # Public verification does not attest to private tables.
        with self.assertRaisesRegex(ValueError, "record_audit.csv"):
            self.run_check()

    def test_partial_private_directory_is_not_treated_as_public_only(self):
        (self.output / "record-level/predictions.csv.gz").rename(self.root / "saved-predictions.csv.gz")
        with self.assertRaisesRegex(ValueError, "record-level file coverage"):
            self.run_check()

    def test_nested_unchecked_before_metric_is_now_replayed(self):
        path = self.output / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["runs"][0]["record_vote_metrics"]["confusion_matrix"][0][0] += 1
        write_json(path, summary)
        self.refresh_result_digest("summary.json")
        verify(self.root, self.output)
        with self.assertRaisesRegex(ValueError, "record_vote_metrics.confusion_matrix"):
            self.run_check()

    def test_input_inventory_is_replayed_even_if_result_hash_is_recomputed(self):
        path = self.output / "inventory.json"
        inventory = json.loads(path.read_text(encoding="utf-8"))
        inventory["files"][0]["min"] += 1.0
        write_json(path, inventory)
        self.refresh_result_digest("inventory.json")
        with self.assertRaisesRegex(ValueError, "inventory.json"):
            self.run_check()

    def test_derived_report_is_replayed_even_if_result_hash_is_recomputed(self):
        path = self.output / "study.md"
        path.write_text(path.read_text(encoding="utf-8") + "Unsupported conclusion.\n", encoding="utf-8")
        self.refresh_result_digest("study.md")
        with self.assertRaisesRegex(ValueError, "study.md"):
            self.run_check()

    def test_unrelated_output_file_is_preserved_and_not_silently_manifested(self):
        path = self.output / "personal_note.txt"
        path.write_text("keep this", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unrelated files"):
            run_study(self.root, self.cache, self.output)
        self.assertEqual(path.read_text(encoding="utf-8"), "keep this")

    def test_manifest_dataset_identity_is_checked(self):
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["dataset_version"] = "a different dataset version"
        write_json(path, manifest)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            verify(self.root, self.output)

    def test_cli_check_succeeds_and_corruption_exits_two(self):
        args = ["check", "--root", str(self.root), "--cache", str(self.cache), "--output", str(self.output)]
        with redirect_stdout(io.StringIO()) as stdout:
            self.assertEqual(main(args), 0)
        self.assertIn("matched all result files", stdout.getvalue())
        (self.output / "scores.svg").write_text("broken", encoding="utf-8")
        with redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as caught:
                main(args)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn("Audit stopped:", stderr.getvalue())

    def test_invalid_configuration_is_rejected_before_outputs_are_written(self):
        for key, value in (("seeds", []), ("seeds", [1, 1]), ("seeds", [True]),
                           ("ridge_alpha", float("nan")), ("test_fraction", 1),
                           ("features", ["mean"]), ("window_samples", 25),
                           ("channels_per_sensor_file", 0), ("dataset_doi", "")):
            settings = {**self.settings, key: value}
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    validate_settings(settings)


if __name__ == "__main__":
    unittest.main()
