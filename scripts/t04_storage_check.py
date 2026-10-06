"""CPU storage regression checks; no checkpoint, replay, GPU or environment.

All files are tiny, synthetic, invalid for training, and confined to an owned
temporary directory. Disk-full failures are injected, never produced by
filling a disk. Real resume equivalence still requires T04 verify.
"""

import argparse
import contextlib
import errno
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import goal_bc as bc
import t00_baseline as baseline


class StorageChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="t04_storage_probe_")
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def test_storage_size_includes_views_without_duplication(self):
        value = torch.arange(100, dtype=torch.float32)
        self.assertEqual(bc.tensor_storage_bytes([value, value[:1], value]), 400)

    def test_low_space_rejects_before_write(self):
        path = self.directory / "latest.pt"
        path.write_bytes(b"existing checkpoint")
        with patch.object(bc.shutil, "disk_usage", return_value=SimpleNamespace(free=0)), \
                patch.object(bc.torch, "save") as writer:
            with self.assertRaises(OSError) as caught:
                bc.save_atomic({"value": torch.ones(4)}, path, overwrite=True)
            self.assertEqual(caught.exception.errno, errno.ENOSPC)
            writer.assert_not_called()
        self.assertEqual(path.read_bytes(), b"existing checkpoint")
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_partial_pytorch_failure_cleans_own_temporary(self):
        path = self.directory / "latest.pt"
        path.write_bytes(b"existing checkpoint")

        def fail(_payload, stream):
            stream.write(b"partial new checkpoint")
            raise RuntimeError("PytorchStreamWriter failed writing file")

        with patch.object(bc, "require_disk_space"), patch.object(bc.torch, "save", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "PytorchStreamWriter"):
                bc.save_atomic({"value": torch.ones(4)}, path, overwrite=True)
        self.assertEqual(path.read_bytes(), b"existing checkpoint")
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_replace_failure_preserves_previous_checkpoint(self):
        path = self.directory / "latest.pt"
        path.write_bytes(b"existing checkpoint")
        with patch.object(bc, "require_disk_space"), \
                patch.object(Path, "replace", side_effect=OSError(errno.ENOSPC, "injected replace failure")):
            with self.assertRaises(OSError):
                bc.save_atomic({"value": torch.ones(4)}, path, overwrite=True)
        self.assertEqual(path.read_bytes(), b"existing checkpoint")
        self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_preexisting_temporary_is_never_removed(self):
        path = self.directory / "latest.pt"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(b"not owned by this call")
        with self.assertRaises(ValueError):
            bc.save_atomic({"value": torch.ones(4)}, path, overwrite=True)
        self.assertEqual(temporary.read_bytes(), b"not owned by this call")

    def test_legacy_and_shared_dependency_roundtrip_and_guards(self):
        # This is a storage fixture, not a GoalLibrary/WM. Only bundle semantic
        # validation is stubbed; path, file hashing and storage guards are real.
        bundle = {"bundle_id": "invalid_for_training_storage_probe", "probe": torch.arange(32)}
        dependency = self.directory / "cache" / "frozen_bundle.pt"
        bc.save_atomic(bundle, dependency)
        reference = {"source_path": dependency, "sha256": bc.file_hash(dependency), "bundle_id": bundle["bundle_id"]}
        payload = {"checkpoint_format": bc.CHECKPOINT_FORMAT, "frozen_bundle": bundle,
                   "worker": {"probe": torch.arange(5)}, "candidate": {"probe": torch.arange(3)},
                   "optimizers": {"worker": {"exp_avg": torch.ones(5), "step": 7}},
                   "counters": {"step": 7, "new_env_steps": 0},
                   "rng_state": {"torch": torch.get_rng_state()}, "sampler_state": {"probe": 13},
                   "selection": {"step": 7}, "verification_artifact": True}
        legacy = self.directory / "legacy.pt"
        compact = self.directory / "train" / "best_worker.pt"
        bc.save_atomic(payload, legacy)
        bc.save_checkpoint(payload, compact, reference)
        raw = bc._torch_load(compact)
        self.assertNotIn("frozen_bundle", raw)
        self.assertEqual(raw["checkpoint_storage"], bc.CHECKPOINT_STORAGE)
        self.assertFalse(Path(raw["frozen_bundle_ref"]["path"]).is_absolute())
        with patch.object(bc, "validate_bundle") as validator:
            loaded = bc.torch_load(compact)
            validator.assert_called_once()
        old = bc.torch_load(legacy)
        for name in ("worker", "candidate"):
            self.assertTrue(torch.equal(old[name]["probe"], loaded[name]["probe"]))
        self.assertTrue(torch.equal(old["optimizers"]["worker"]["exp_avg"], loaded["optimizers"]["worker"]["exp_avg"]))
        self.assertTrue(torch.equal(old["rng_state"]["torch"], loaded["rng_state"]["torch"]))
        for name in ("counters", "sampler_state", "selection", "verification_artifact"):
            self.assertEqual(old[name], loaded[name])
        self.assertTrue(torch.equal(loaded["frozen_bundle"]["probe"], bundle["probe"]))
        wrong_id = dict(raw, frozen_bundle_ref=dict(raw["frozen_bundle_ref"], bundle_id="wrong_content_id"))
        bad_path = self.directory / "train" / "bad_id.pt"
        bc.save_atomic(wrong_id, bad_path)
        with patch.object(bc, "validate_bundle"), self.assertRaisesRegex(ValueError, "内容 ID"):
            bc.torch_load(bad_path)
        ambiguous = dict(raw, frozen_bundle=bundle)
        ambiguous_path = self.directory / "train" / "ambiguous.pt"
        bc.save_atomic(ambiguous, ambiguous_path)
        with self.assertRaisesRegex(ValueError, "重复声明"):
            bc.torch_load(ambiguous_path)
        with dependency.open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            bc.torch_load(compact)
        dependency.unlink()  # only the fixture created by this test
        with self.assertRaisesRegex(FileNotFoundError, "frozen_bundle.pt"):
            bc.torch_load(compact)

    def test_failed_json_write_preserves_last_report(self):
        path = self.directory / "report.json"
        baseline.write_json(path, {"previous": True})
        before = path.read_bytes()
        with patch.object(baseline.os, "fsync", side_effect=OSError(errno.ENOSPC, "injected disk full")):
            with self.assertRaises(OSError):
                baseline.write_json(path, {"new": True})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.directory.glob("*.tmp")), [])

    def test_report_and_error_log_failure_does_not_cascade(self):
        report = baseline.Report(self.directory, argparse.Namespace(command="inspect"))
        report.check("source_inputs_unchanged", "PASS", "storage fixture only")
        before = (self.directory / "report.json").read_bytes()
        console = io.StringIO()
        with patch.object(baseline, "write_text_atomic", side_effect=OSError(errno.ENOSPC, "injected disk full")), \
                contextlib.redirect_stderr(console), contextlib.redirect_stdout(console):
            try:
                raise RuntimeError("original failure")
            except RuntimeError as error:
                baseline.record_exception(report, error)
            code = report.finish()
        self.assertEqual(code, 2)
        self.assertEqual(report.data["status"], "failed")
        self.assertIn("original failure", console.getvalue())
        self.assertIn("report_output", console.getvalue())
        self.assertEqual(len([check for check in report.data["checks"] if check["name"] == "report_output"]), 1)
        source = next(check for check in report.data["checks"] if check["name"] == "source_inputs_unchanged")
        self.assertEqual(source["level"], "PASS")
        self.assertEqual((self.directory / "report.json").read_bytes(), before)
        self.assertEqual(json.loads(before)["checks"][0]["level"], "PASS")


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StorageChecks))
    print(f"[{'PASS' if result.wasSuccessful() else 'FAIL'}] T04_STORAGE_CHECKS; synthetic_io_only=1; new_env_steps=0", flush=True)
    sys.exit(0 if result.wasSuccessful() else 2)
