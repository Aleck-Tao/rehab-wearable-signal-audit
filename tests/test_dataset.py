import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from rehab_audit.acquire import digest, fetch_files
from rehab_audit.dataset import inspect_inputs, load_numeric, prepare_features


class IngestionTests(unittest.TestCase):
    def test_rejects_corrupt_magic_extra_payload_object_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npy"
            path.write_bytes(b"\n\xef\xbf\xbdNUMPY")
            with self.assertRaisesRegex(ValueError, "magic"):
                load_numeric(path, 4, 2)
            np.save(path, np.zeros((2, 4, 2)))
            with path.open("ab") as stream:
                stream.write(b"extra")
            with self.assertRaisesRegex(ValueError, "payload"):
                load_numeric(path, 4, 2)
            np.save(path, np.array([{"execute": "nothing"}], dtype=object))
            with self.assertRaises(ValueError):
                load_numeric(path, 4, 2)
            np.save(path, np.full((2, 4, 2), np.nan))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_numeric(path, 4, 2)

    def test_publisher_matching_corrupt_file_is_quarantined_not_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for sensor in (1, 2):
                path = root / f"000_{sensor}.npy"
                if sensor == 1:
                    path.write_bytes(b"released malformed file")
                else:
                    np.save(path, np.ones((2, 4, 2)))
                entries.append({"fileName": path.name, "id": str(sensor), "size": path.stat().st_size, "md5": digest(path, "md5")})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": entries}))
            audit, pairs = inspect_inputs(root, manifest, {"classes": 1, "samples_per_record": 4, "channels_per_sensor_file": 2})
            self.assertEqual(pairs, {})
            self.assertEqual(audit["files"][0]["status"], "quarantined")
            self.assertTrue(audit["files"][0]["publisher_checksum_verified"])
            self.assertEqual(audit["excluded_classes"][0]["class_id"], 0)
            self.assertEqual((root / "000_1.npy").read_bytes(), b"released malformed file")

    def test_truncated_payload_and_invalid_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npy"
            np.save(path, np.zeros((2, 4, 2)))
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises((ValueError, OSError)):
                load_numeric(path, 4, 2)
            for array in (np.zeros((2, 4, 3)), np.zeros((0, 4, 2)), np.ones((2, 4, 2), dtype=complex)):
                np.save(path, array)
                with self.subTest(shape=array.shape, dtype=array.dtype), self.assertRaises(ValueError):
                    load_numeric(path, 4, 2)

    def test_valid_integer_array_and_read_only_memmap(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npy"
            np.save(path, np.arange(16, dtype=np.int16).reshape(2, 4, 2))
            data = load_numeric(path, 4, 2)
            self.assertEqual(data.shape, (2, 4, 2))
            self.assertFalse(data.flags.writeable)
            del data

    def test_checksum_mismatch_aborts_instead_of_quarantining(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries = []
            for sensor in (1, 2):
                path = root / f"000_{sensor}.npy"
                np.save(path, np.ones((2, 4, 2)))
                entries.append({"fileName": path.name, "id": str(sensor), "size": path.stat().st_size, "md5": "0" * 32})
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": entries}))
            with self.assertRaisesRegex(ValueError, "checksum/size"):
                inspect_inputs(root, manifest, {"classes": 1, "samples_per_record": 4, "channels_per_sensor_file": 2})

    def test_record_identity_survives_windowing(self):
        pair = (np.arange(2 * 8 * 6).reshape(2, 8, 6), np.ones((2, 8, 6)))
        data, records = prepare_features({0: pair}, {"window_samples": 4, "stride_samples": 2, "constant_tail_atol": 1e-10})
        self.assertEqual(data["features"].shape, (6, 60))
        self.assertEqual(list(data["record_ids"]), ["000/0000"] * 3 + ["000/0001"] * 3)
        self.assertEqual(len(records), 2)

    def test_tail_tolerance_is_applied_equally_to_both_sensor_groups(self):
        signal = np.repeat(np.array([0., 1., 2., 2.01, 2.02, 2.03])[:, None], 6, axis=1)
        pair = (signal[None], signal[None])
        data, records = prepare_features({0: pair}, {"window_samples": 4, "stride_samples": 2, "constant_tail_atol": .1})
        self.assertEqual(records[0]["constant_tail_samples"], 4)
        self.assertEqual(records[0]["limb_constant_tail_samples"], 4)
        self.assertEqual(records[0]["glove_constant_tail_samples"], 4)
        np.testing.assert_allclose(data["tail_fractions"], [.5, 1.])


class DownloadTests(unittest.TestCase):
    def make_manifest(self, root, payload=b"numeric source", **changes):
        import hashlib
        entry = {"fileName": "000_1.npy", "id": "abc123", "size": len(payload),
                 "md5": hashlib.md5(payload).hexdigest()}
        entry.update(changes)
        manifest = root / "source.json"
        manifest.write_text(json.dumps({"files": [entry]}))
        return manifest

    def test_verified_cache_does_not_retrieve_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_manifest(root)
            destination = root / "cache"
            with patch("rehab_audit.acquire.urlopen", return_value=io.BytesIO(b"numeric source")) as retrieve:
                first = fetch_files(manifest, destination)
                second = fetch_files(manifest, destination)
                self.assertEqual(first, second)
                self.assertEqual(retrieve.call_count, 1)
            self.assertEqual([path.name for path in destination.iterdir()], ["000_1.npy"])

    def test_bad_payload_never_replaces_existing_file_and_cleans_temporary(self):
        for payload in (b"oversized numeric source", b"short"):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self.make_manifest(root)
                destination = root / "cache"
                destination.mkdir()
                target = destination / "000_1.npy"
                target.write_bytes(b"prior cached content")
                with patch("rehab_audit.acquire.urlopen", return_value=io.BytesIO(payload)), self.assertRaises(ValueError):
                    fetch_files(manifest, destination)
                self.assertEqual(target.read_bytes(), b"prior cached content")
                self.assertEqual([path.name for path in destination.iterdir()], ["000_1.npy"])

    def test_unsafe_manifest_is_rejected_before_network(self):
        for changes in ({"fileName": "../escape.npy"}, {"fileName": "C:escape.npy"},
                        {"md5": "bad"}, {"size": True}, {"size": -1}, {"id": "x&extra=1"}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = self.make_manifest(root, **changes)
                with patch("rehab_audit.acquire.urlopen") as retrieve, self.assertRaises(ValueError):
                    fetch_files(manifest, root / "cache")
                retrieve.assert_not_called()
