"""Isolated provenance and copy-boundary tests; no real runs, GPU, or torch.

Run with ``python -m unittest experiments.gated_asl_classification.test_migrate``.
All data, including 65 runtime sources and 20 six-file fits, lives in a fresh
temporary directory. The synthetic checkpoint/prediction bytes are deliberate:
migration verifies provenance and copying; statistical/model audits are separate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.gated_asl_classification import migrate as migration


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    return path


def read(path):
    return json.loads(Path(path).read_text())


def tree_identity(root):
    """Ignore read-induced atime but detect byte, inode, or write-time changes."""
    result = {}
    for path in sorted(Path(root).rglob("*")):
        info = path.lstat()
        result[str(path.relative_to(root))] = (
            info.st_mode, info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns,
            digest(path) if path.is_file() and not path.is_symlink() else None,
            os.readlink(path) if path.is_symlink() else None,
        )
    return result


class CampaignFixture:
    """The same fit/source counts as the interrupted campaign, with tiny data."""

    MODELS = ["AlphaRAD", "RadZero", "CARZero", "KAD", "BiomedCLIP", "BioViL-T", "MaCo"]
    METHODS = ["pre_bce", "gated_bce", "pre_asl", "gated_asl"]
    FILES = ("result.json", "training.json", "progress.json", "adapter.pt",
             "test_predictions.npz", "validation_predictions.npz")

    def __init__(self, root):
        self.root = Path(root)
        self.origin = self.root / "old_run"
        self.target = self.root / "independent_run"
        self.recovery = self.origin / "recovery" / "snapshot"
        self.origin.mkdir()
        self.recovery.mkdir(parents=True)
        self.runtime = self.root / "runtime"
        self.sources = {}
        archives = {}
        for i in range(65):
            rel = Path("package") / ("train.py" if i == 0 else f"source_{i:02d}.py")
            source = put(self.runtime / rel, f"# synthetic source {i}\n".encode())
            archive = put(self.origin / "source" / rel, source.read_bytes())
            self.sources[str(source)] = digest(source)
            archives[str(source)] = str(archive)

        references = []
        reference_keys = []
        for model in self.MODELS:
            for dataset in ("openi", "chexpert", "chestxray14"):
                reference_keys.append((model, dataset, 42, "original"))
                reference_keys.extend((model, dataset, seed, "pre_mlp") for seed in (42, 43, 44))
                if model in ("AlphaRAD", "KAD", "MaCo"):
                    reference_keys.extend((model, dataset, seed, "post_mlp") for seed in (42, 43, 44))
                if model in ("BiomedCLIP", "BioViL-T", "MaCo"):
                    reference_keys.append((model, dataset, 42, "aggregation_only"))
        for i, (model, dataset, seed, method) in enumerate(reference_keys):
            folder = self.root / "historical" / f"reference_{i:03d}"
            artifact = put(folder / "result.json", {"historical_index": i})
            references.append(dict(model=model, dataset=dataset,
                                   seed=seed, method=method, source=str(folder),
                                   files_sha256={"result.json": digest(artifact)}))
        datasets = {}
        for name in ("openi", "chexpert", "chestxray14"):
            samples = put(self.root / "datasets" / name / "samples.jsonl", b''.join(
                (json.dumps(dict(id=i, split=split, labels=[1])) + "\n").encode()
                for i, split in enumerate(("train", "val", "test"))))
            dataset = put(samples.parent / "dataset.json", dict(samples_file=str(samples),
                          samples_sha256=digest(samples), class_names=["condition"]))
            datasets[name] = dict(path=str(dataset), sha256=digest(dataset),
                                  samples_sha256=digest(samples), classes=["condition"],
                                  counts=dict(train=1, val=1, test=1))
        caches = {}
        for model in self.MODELS:
            for name in datasets:
                array = put(self.root / "caches" / model / name / "tokens.npy", b"tiny frozen array\n")
                cached = put(array.parent / "cache.json", dict(status="complete"))
                caches[f"{model}/{name}"] = [dict(path=str(cached), sha256=digest(cached),
                    files={str(array): dict(bytes=array.stat().st_size, mtime_ns=array.stat().st_mtime_ns)})]
        parent = put(self.root / "parent" / "PROTOCOL.json", dict(datasets=datasets,
                     references=references, caches=caches))
        self.protocol = dict(status="frozen", campaign="gated_asl_classification",
                             models=self.MODELS, methods=self.METHODS, datasets=datasets,
                             seeds=[42, 43, 44], new_training_runs=252,
                             all_methods_fresh=True, reused_training_runs=0,
                             references=references, caches=caches, allowed_physical_gpus=[4, 5, 6, 7],
                             train=dict(lr=0.0001, max_epochs=20),
                             method_specs={name: dict(fresh_fit=True) for name in self.METHODS},
                             parent_protocol=dict(path=str(parent), sha256=digest(parent)))
        put(self.origin / "PROTOCOL.json", self.protocol)
        self.manifest = dict(sources_sha256=self.sources, archives=archives,
                             protocol_sha256=digest(self.origin / "PROTOCOL.json"),
                             all_methods_fresh=True, new_fits=252)
        put(self.origin / "source_manifest.json", self.manifest)
        self.fit_dirs = []
        self.audit_rows = []
        completed = {}
        for model in ("AlphaRAD", "RadZero"):
            for method in self.METHODS:
                for seed in (42, 43, 44):
                    if method == "gated_asl" and (model == "RadZero" or seed == 44):
                        continue
                    relative = Path("results") / model / "chestxray14" / f"seed{seed}" / method
                    folder = self.origin / relative
                    self.fit_dirs.append(relative)
                    artifact_hashes = {}
                    for name in ("adapter.pt", "test_predictions.npz", "validation_predictions.npz"):
                        artifact_hashes[name] = digest(put(folder / name, f"{relative}/{name}\n".encode()))
                    record = dict(status="complete", model=model, dataset="chestxray14", seed=seed,
                                  method=method, fresh_fit=True, historical_reference_used_for_training=False,
                                  protocol_sha256=digest(self.origin / "PROTOCOL.json"),
                                  source_manifest_sha256=digest(self.origin / "source_manifest.json"),
                                  sources_sha256=self.sources, adapter_sha256=artifact_hashes["adapter.pt"],
                                  test_predictions_sha256=artifact_hashes["test_predictions.npz"],
                                  validation_predictions_sha256=artifact_hashes["validation_predictions.npz"],
                                  config=self.protocol["train"], method_spec=self.protocol["method_specs"][method],
                                  dataset_sha256=datasets["chestxray14"]["sha256"],
                                  samples_sha256=datasets["chestxray14"]["samples_sha256"],
                                  class_names=["condition"], counts=datasets["chestxray14"]["counts"],
                                  frozen_backbone_verified=True, test_selection=False,
                                  matched_initialization_verified=True, matched_overlapping_epoch_orders_verified=True,
                                  selected_val=dict(macro_auroc=0.81, macro_auprc=0.4),
                                  test=dict(macro_auroc=0.82, macro_auprc=0.42))
                    put(folder / "result.json", record)
                    put(folder / "training.json", dict(record, history=[dict(epoch=1)]))
                    put(folder / "progress.json", dict(status="complete", method=method, epoch=1))
                    hashes = {name: digest(folder / name) for name in self.FILES}
                    completed.update({str(relative / name): value for name, value in hashes.items()})
                    self.audit_rows.append(dict(key=[model, "chestxray14", seed, method],
                                                source=str(folder), fresh_fit=True,
                                                files_sha256={name: value for name, value in hashes.items()
                                                              if name != "progress.json"},
                                                result_current_sha256=hashes["result.json"],
                                                recomputed_val=record["selected_val"], recomputed_test=record["test"]))
        assert len(self.fit_dirs) == 20 and len(completed) == 120
        self.partial_dirs = [Path("results/AlphaRAD/chestxray14/seed44/gated_asl"),
                             Path("results/RadZero/chestxray14/seed42/gated_asl")]
        for relative in self.partial_dirs:
            put(self.origin / relative / "training.json", dict(status="running", epoch=0))
            put(self.origin / relative / "progress.json", dict(status="running", epoch=0))
            put(self.origin / relative / "adapter.pt", b"unfinished checkpoint\n")
        for model in ("AlphaRAD", "RadZero"):
            put(self.origin / "validation" / model / "chestxray14" / "frozen_path.json", {"status": "passed"})
        self.audit = dict(status="partial", expected_new_fits=252, completed_new_fits=20,
                          expected_reference_evaluations=120, completed_reference_evaluations=120,
                          old_fits_counted_as_new=0, protocol_sha256=digest(self.origin / "PROTOCOL.json"),
                          errors=[], runs=self.audit_rows,
                          historical_references=[dict(entry, new_fit=False) for entry in references])
        put(self.origin / "AUDIT.json", self.audit)
        put(self.origin / "PREFLIGHT.json", dict(status="passed", protocol_sha256=digest(self.origin / "PROTOCOL.json")))
        self.recovery_record = dict(status="snapshot_preserved_waiting_for_exclusive_locks",
                                    run_dir=str(self.origin), completed_fits_preserved=20,
                                    runtime_sources_verified=65, completed_files_sha256=completed,
                                    protocol_sha256=digest(self.origin / "PROTOCOL.json"),
                                    source_manifest_sha256=digest(self.origin / "source_manifest.json"), archived_files={})
        for name in ("PROTOCOL.json", "source_manifest.json", "AUDIT.json"):
            copied = put(self.recovery / name, (self.origin / name).read_bytes())
            self.recovery_record["archived_files"][name] = digest(copied)
        for relative in self.partial_dirs:
            for filename in ("training.json", "progress.json", "adapter.pt"):
                archived = Path("interrupted_fits") / relative.relative_to("results") / filename
                copied = put(self.recovery / archived, (self.origin / relative / filename).read_bytes())
                self.recovery_record["archived_files"][str(archived)] = digest(copied)
        put(self.recovery / "recovery.json", self.recovery_record)
        for name in ("controller.lock", "supervision.lock"):
            put(self.origin / name, b"")
        for name in ("training_matrix.json", "supervision.json"):
            put(self.origin / name, {"status": "stale", "must_not_copy_or_edit": True})

    def run(self, check_only=False):
        return migration.migrate(self.origin, self.target, self.recovery, check_only=check_only)

    def replace_fit_record(self, changes):
        """Update evidence too, so semantic tests cannot pass only on hash drift."""
        relative = self.fit_dirs[0]
        folder = self.origin / relative
        for name in ("result.json", "training.json"):
            record = read(folder / name)
            record.update(changes)
            put(folder / name, record)
        hashes = {name: digest(folder / name) for name in self.FILES}
        self.audit_rows[0]["files_sha256"] = {k: v for k, v in hashes.items() if k != "progress.json"}
        self.audit_rows[0]["result_current_sha256"] = hashes["result.json"]
        self.recovery_record["completed_files_sha256"].update(
            {str(relative / name): value for name, value in hashes.items()})
        put(self.origin / "AUDIT.json", self.audit)
        put(self.recovery / "AUDIT.json", (self.origin / "AUDIT.json").read_bytes())
        self.recovery_record["archived_files"]["AUDIT.json"] = digest(self.recovery / "AUDIT.json")
        put(self.recovery / "recovery.json", self.recovery_record)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gated-migration-test-")
        self.addCleanup(self.temp.cleanup)
        self.fixture = CampaignFixture(self.temp.name)

    def assert_rejected(self, operation):
        with self.assertRaises((ValueError, RuntimeError, FileNotFoundError, FileExistsError, AssertionError)):
            operation()

    def test_absent_migration_leaves_normal_run_unhandled(self):
        before = tree_identity(self.fixture.root)
        self.assertIsNone(migration.verify_migration(self.fixture.origin))
        self.assertEqual(before, tree_identity(self.fixture.root))

    def test_check_only_has_no_filesystem_writes(self):
        before = tree_identity(self.fixture.root)
        plan = self.fixture.run(check_only=True)
        self.assertIsInstance(plan, dict)
        self.assertFalse(self.fixture.target.exists())
        self.assertEqual(before, tree_identity(self.fixture.root))

    def test_cli_check_only_creates_nothing(self):
        fixture = self.fixture
        before = tree_identity(fixture.root)
        completed = subprocess.run([sys.executable, str(Path(migration.__file__)),
            "--source-run", str(fixture.origin), "--destination-run", str(fixture.target),
            "--recovery-dir", str(fixture.recovery), "--check-only"],
            capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        evidence = json.loads(completed.stdout)
        self.assertFalse(evidence["writes_performed"])
        self.assertEqual(evidence["copied_fits"], 20)
        self.assertEqual(evidence["complete_fit_files"], 120)
        self.assertEqual(before, tree_identity(fixture.root))

    def test_copy_preserves_bytes_and_has_independent_inodes(self):
        fixture = self.fixture
        before = tree_identity(fixture.origin)
        fixture.run()
        verified = migration.verify_migration(fixture.target, runtime=fixture.manifest)
        self.assertIsInstance(verified, dict)
        self.assertEqual(before, tree_identity(fixture.origin))
        for relative in [Path("PROTOCOL.json"), Path("source_manifest.json")]:
            self.assertEqual((fixture.origin / relative).read_bytes(), (fixture.target / relative).read_bytes())
        for relative in fixture.fit_dirs:
            for name in fixture.FILES:
                old, new = fixture.origin / relative / name, fixture.target / relative / name
                self.assertEqual(old.read_bytes(), new.read_bytes())
                self.assertFalse(os.path.samefile(old, new))
                self.assertEqual(new.stat().st_nlink, 1)
        for source, old_name in fixture.manifest["archives"].items():
            relative = Path(old_name).relative_to(fixture.origin)
            new = fixture.target / relative
            self.assertEqual(digest(new), fixture.sources[source])
            self.assertFalse(os.path.samefile(old_name, new))
        for relative in fixture.partial_dirs:
            self.assertFalse((fixture.target / relative).exists())
        for name in ("controller.lock", "supervision.lock", "training_matrix.json", "supervision.json"):
            self.assertFalse((fixture.target / name).exists())

    def test_rejects_nonempty_target_without_touching_either_tree(self):
        fixture = self.fixture
        put(fixture.target / "keep.txt", b"pre-existing content\n")
        before = tree_identity(fixture.root)
        self.assert_rejected(lambda: fixture.run())
        self.assertEqual(before, tree_identity(fixture.root))

    def test_rejects_target_inside_origin(self):
        fixture = self.fixture
        fixture.target = fixture.origin / "new_run"
        before = tree_identity(fixture.root)
        self.assert_rejected(lambda: fixture.run())
        self.assertEqual(before, tree_identity(fixture.root))

    def test_rejects_source_changes_before_copy(self):
        fixture = self.fixture
        path = Path(next(iter(fixture.manifest["archives"].values())))
        path.write_bytes(path.read_bytes() + b"changed\n")
        before = tree_identity(fixture.root)
        self.assert_rejected(lambda: fixture.run())
        self.assertFalse(fixture.target.exists())
        self.assertEqual(before, tree_identity(fixture.root))

    def test_rejects_current_runtime_changes_before_copy(self):
        fixture = self.fixture
        path = Path(next(iter(fixture.sources)))
        path.write_bytes(path.read_bytes() + b"changed\n")
        self.assert_rejected(lambda: fixture.run())
        self.assertFalse(fixture.target.exists())

    def test_rejects_missing_file_in_claimed_complete_fit(self):
        fixture = self.fixture
        (fixture.origin / fixture.fit_dirs[0] / "adapter.pt").unlink()
        self.assert_rejected(lambda: fixture.run())
        self.assertFalse(fixture.target.exists())

    def test_rejects_historical_fit_even_with_consistent_record_hashes(self):
        fixture = self.fixture
        fixture.replace_fit_record(dict(fresh_fit=False, historical_reference_used_for_training=True))
        self.assert_rejected(lambda: fixture.run())
        self.assertFalse(fixture.target.exists())

    def test_rejects_incomplete_fit_even_with_consistent_record_hashes(self):
        fixture = self.fixture
        fixture.replace_fit_record(dict(status="running"))
        self.assert_rejected(lambda: fixture.run())
        self.assertFalse(fixture.target.exists())

    def test_rejects_source_archive_symlink_even_with_identical_bytes(self):
        fixture = self.fixture
        source, name = next(iter(fixture.manifest["archives"].items()))
        archive = Path(name)
        archive.unlink()
        archive.symlink_to(source)
        self.assert_rejected(lambda: fixture.run())
        self.assertFalse(fixture.target.exists())

    def test_rejects_source_fit_hardlink_even_with_identical_bytes(self):
        fixture = self.fixture
        artifact = fixture.origin / fixture.fit_dirs[0] / "adapter.pt"
        os.link(artifact, fixture.root / "another_link.pt")
        self.assert_rejected(lambda: fixture.run())
        self.assertFalse(fixture.target.exists())

    def test_rejects_target_parent_symlink(self):
        fixture = self.fixture
        real = fixture.root / "another_location"
        real.mkdir()
        alias = fixture.root / "alias"
        alias.symlink_to(real, target_is_directory=True)
        fixture.target = alias / "new_run"
        before = tree_identity(fixture.root)
        self.assert_rejected(lambda: fixture.run())
        self.assertEqual(before, tree_identity(fixture.root))

    def test_destination_archive_tampering_is_rejected(self):
        fixture = self.fixture
        fixture.run()
        old = Path(next(iter(fixture.manifest["archives"].values())))
        new = fixture.target / old.relative_to(fixture.origin)
        new.write_bytes(new.read_bytes() + b"tampered\n")
        self.assert_rejected(lambda: migration.verify_migration(fixture.target))

    def test_destination_manifest_tampering_is_rejected(self):
        fixture = self.fixture
        fixture.run()
        manifest_path = fixture.target / "source_manifest.json"
        changed = read(manifest_path)
        changed["new_fits"] = 251
        put(manifest_path, changed)
        self.assert_rejected(lambda: migration.verify_migration(fixture.target))

    def test_origin_archive_changes_after_copy_do_not_change_independent_snapshot(self):
        fixture = self.fixture
        fixture.run()
        path = Path(next(iter(fixture.manifest["archives"].values())))
        path.write_bytes(path.read_bytes() + b"changed after copy\n")
        self.assertIsInstance(migration.verify_migration(fixture.target), dict)

    def test_origin_fit_changes_after_copy_do_not_change_independent_snapshot(self):
        fixture = self.fixture
        fixture.run()
        path = fixture.origin / fixture.fit_dirs[0] / "adapter.pt"
        path.write_bytes(path.read_bytes() + b"changed after copy\n")
        self.assertIsInstance(migration.verify_migration(fixture.target), dict)

    def test_origin_can_be_offline_after_verified_copy(self):
        fixture = self.fixture
        fixture.run()
        fixture.origin.rename(fixture.root / "offline_old_run")
        self.assertIsInstance(migration.verify_migration(fixture.target), dict)

    def test_old_diagnostics_and_partial_progress_can_change_after_copy(self):
        fixture = self.fixture
        fixture.run()
        put(fixture.recovery / "lock_observations.jsonl", b'{"later_check":"still_locked"}\n')
        put(fixture.origin / fixture.partial_dirs[0] / "training.json", dict(status="interrupted", epoch=16))
        self.assertIsInstance(migration.verify_migration(fixture.target), dict)

    def test_new_validation_progress_and_fit_do_not_mutate_imported_set(self):
        fixture = self.fixture
        fixture.run()
        put(fixture.target / "validation/AlphaRAD/chestxray14/frozen_path.json", {"status": "passed", "new_worker": True})
        source = fixture.target / "results/AlphaRAD/chestxray14/seed43/gated_asl"
        target = fixture.target / fixture.partial_dirs[0]
        for name in fixture.FILES:
            if name in ("training.json", "result.json"):
                record = read(source / name)
                record["seed"] = 44
                put(target / name, record)
            else:
                put(target / name, (source / name).read_bytes())
        verified = migration.verify_migration(fixture.target)
        self.assertEqual(verified["evidence"]["copied_fits"], 20)
        self.assertEqual(verified["evidence"]["copied_files"], 120)
        self.assertEqual(len(list((fixture.target / "results").glob("*/*/seed*/*/result.json"))), 21)

    def test_current_runtime_changes_after_copy_are_rejected(self):
        fixture = self.fixture
        fixture.run()
        path = Path(next(iter(fixture.sources)))
        path.write_bytes(path.read_bytes() + b"changed runtime\n")
        self.assert_rejected(lambda: migration.verify_migration(fixture.target))

    def test_copied_origin_evidence_tampering_is_rejected(self):
        fixture = self.fixture
        fixture.run()
        path = fixture.target / "migration/origin/AUDIT.json"
        audit = read(path)
        audit["runs"][0]["recomputed_test"]["macro_auroc"] = 0.99
        put(path, audit)
        self.assert_rejected(lambda: migration.verify_migration(fixture.target))

    def test_destination_fit_changes_after_copy_are_rejected(self):
        fixture = self.fixture
        fixture.run()
        path = fixture.target / fixture.fit_dirs[0] / "test_predictions.npz"
        path.write_bytes(path.read_bytes() + b"changed after copy\n")
        self.assert_rejected(lambda: migration.verify_migration(fixture.target))

    def test_wrong_runtime_manifest_is_rejected(self):
        fixture = self.fixture
        fixture.run()
        wrong = copy.deepcopy(fixture.manifest)
        wrong["new_fits"] = 251
        self.assert_rejected(lambda: migration.verify_migration(fixture.target, runtime=wrong))

    def test_destination_file_alias_to_origin_is_rejected(self):
        fixture = self.fixture
        fixture.run()
        relative = fixture.fit_dirs[0] / "adapter.pt"
        new = fixture.target / relative
        new.unlink()
        new.symlink_to(fixture.origin / relative)
        self.assert_rejected(lambda: migration.verify_migration(fixture.target))

    def test_destination_hardlink_to_origin_is_rejected(self):
        fixture = self.fixture
        fixture.run()
        relative = fixture.fit_dirs[0] / "adapter.pt"
        new = fixture.target / relative
        new.unlink()
        os.link(fixture.origin / relative, new)
        self.assert_rejected(lambda: migration.verify_migration(fixture.target))

    def test_migration_archive_path_cannot_escape_destination(self):
        fixture = self.fixture
        fixture.run()
        path = fixture.target / "migration.json"
        manifest = read(path)
        source = next(iter(manifest["source_archives"]))
        manifest["source_archives"][source] = fixture.manifest["archives"][source]
        put(path, manifest)
        self.assert_rejected(lambda: migration.verify_migration(fixture.target))

    def test_migration_immutable_path_cannot_escape_destination(self):
        fixture = self.fixture
        fixture.run()
        path = fixture.target / "migration.json"
        manifest = read(path)
        outsider = put(fixture.root / "outside.txt", b"same bytes outside\n")
        manifest["immutable_files_sha256"]["../outside.txt"] = digest(outsider)
        manifest["copy_sources"]["../outside.txt"] = str(outsider)
        put(path, manifest)
        self.assert_rejected(lambda: migration.verify_migration(fixture.target))


if __name__ == "__main__":
    unittest.main(verbosity=2)
