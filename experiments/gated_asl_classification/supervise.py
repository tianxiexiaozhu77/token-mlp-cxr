"""Watch fresh-fit evidence and refresh audited tables; never control training.

Only new complete three-seed method groups trigger partial reports. The final
strict report/export requires all 252 observed results and all 21 controller
jobs complete. This watcher never seals artifacts, restarts workers, or kills
training. --check-only performs reads only, including no lock/state creation.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.gated_asl_classification.migrate import verify_migration

LAUNCHER = ROOT / "AlphaRAD/local_setup/python_local.sh"
METHODS = ("pre_bce", "gated_bce", "pre_asl", "gated_asl")
MODELS = ("AlphaRAD", "RadZero", "CARZero", "KAD", "BiomedCLIP", "BioViL-T", "MaCo")
DATASETS = ("openi", "chexpert", "chestxray14")
SEEDS = (42, 43, 44)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process_identity(pid):
    """Read Linux field 22 after the final ')', even for spaced process names."""
    folder = Path("/proc") / str(pid)
    try:
        fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        command = [part.decode(errors="replace") for part in (folder / "cmdline").read_bytes().split(b"\0") if part]
        # Recheck starttime to rule out PID reuse between reads.
        repeated = (folder / "stat").read_text().rsplit(")", 1)[1].split()
        require(fields[19] == repeated[19], "PID changed identity during /proc inspection")
        alive = repeated[0] not in ("Z", "X", "x")
        return dict(pid=pid, starttime=int(fields[19]), state=repeated[0], alive=alive,
                    cmdline=command, cwd=str((folder / "cwd").resolve()) if alive else None,
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    except (FileNotFoundError, ProcessLookupError):
        return dict(pid=pid, alive=False)


def controller_matches(process, run):
    if not process["alive"]:
        return False
    command, cwd = process["cmdline"], Path(process["cwd"])
    scripts = [(cwd / part).resolve() for part in command[1:] if part.endswith(".py")]
    if HERE / "run.py" not in scripts or command.count("--run-dir") != 1:
        return False
    index = command.index("--run-dir") + 1
    return index < len(command) and (cwd / command[index]).resolve() == run


def group_keys(keys):
    grouped = defaultdict(set)
    for model, dataset, seed, method in keys:
        grouped[model, dataset, method].add(seed)
    return {key for key, seeds in grouped.items() if seeds == set(SEEDS)}


class Watcher:
    def __init__(self, run, pid, poll_seconds):
        self.run, self.pid, self.poll_seconds = run, pid, poll_seconds
        protocol = read(run / "PROTOCOL.json")
        self.protocol_hash = sha256(run / "PROTOCOL.json")
        self.manifest_hash = sha256(run / "source_manifest.json")
        self.runtime = read(run / "source_manifest.json")
        self.migration = verify_migration(run, self.runtime)
        self.archives = self.migration["archives"] if self.migration else self.runtime["archives"]
        require(protocol["status"] == "frozen" and protocol["campaign"] == "gated_asl_classification"
                and protocol["models"] == list(MODELS) and protocol["methods"] == list(METHODS)
                and set(protocol["datasets"]) == set(DATASETS) and protocol["seeds"] == list(SEEDS)
                and protocol["new_training_runs"] == 252 and protocol["all_methods_fresh"] is True
                and protocol["reused_training_runs"] == 0, "Wrong frozen 252-fit campaign")
        require(self.runtime["protocol_sha256"] == self.protocol_hash and len(self.runtime["sources_sha256"]) == 65,
                "Expected the matching 65-file runtime snapshot")
        self.expected = {(model, dataset, seed, method) for model in MODELS for dataset in DATASETS
                         for seed in SEEDS for method in METHODS}
        self.controller_identity = None
        self.state = None
        self.protected = {str(HERE / name): sha256(HERE / name)
                          for name in ("supervise.py", "report.py", "export_tables.py", "migrate.py")}
        self.protected[str(LAUNCHER)] = sha256(LAUNCHER)
        initial_audit = read(run / "AUDIT.json")
        require(initial_audit.get("migration") == (self.migration["evidence"] if self.migration else None),
                "Initial audit migration provenance differs")
        for source, digest in initial_audit["reporter_sources_sha256"].items():
            require(sha256(source) == digest, "Reporter changed since existing audit: " + source)
            self.protected[source] = digest
        previous_path = run / "supervision.json"
        self.previous = read(previous_path) if previous_path.exists() else None
        if self.previous:
            require(self.previous.get("migration") == (self.migration["evidence"] if self.migration else None),
                    "Migration differs from previous supervision")
            require(self.previous["protocol_sha256"] == self.protocol_hash
                    and self.previous["source_manifest_sha256"] == self.manifest_hash,
                    "Existing supervision belongs to a different frozen run")
            require(self.previous["status"] != "failed", "Existing supervisor failed; inspect its state/logs before a manual restart")
            require(self.previous["status"] in ("watching", "ready_for_final_review"),
                    "Previous supervisor stopped during a postprocessing command; inspect its state/logs before restarting")
            require(self.previous["postprocessing_sources_sha256"] == self.protected,
                    "Postprocessing sources changed since previous supervision")
            self.controller_identity = self.previous["controller_identity"]
            require(self.controller_identity["pid"] == pid, "Controller PID differs from existing supervision")

    def verify_sources(self):
        require((self.run / "migration.json").exists() == (self.migration is not None),
                "Migration presence changed during supervision")
        if self.migration:
            require(sha256(self.run / "migration.json") == self.migration["evidence"]["manifest_sha256"],
                    "Migration manifest changed during supervision")
        require(sha256(self.run / "PROTOCOL.json") == self.protocol_hash
                and sha256(self.run / "source_manifest.json") == self.manifest_hash,
                "Frozen protocol/runtime manifest changed")
        for source, digest in self.runtime["sources_sha256"].items():
            require(sha256(source) == sha256(self.archives[source]) == digest,
                    "Runtime source/archive changed: " + source)
        for source, digest in self.protected.items():
            require(sha256(source) == digest, "Postprocessing source changed: " + source)

    def inspect_matrix(self):
        matrix = read(self.run / "training_matrix.json")
        jobs = matrix["jobs"]
        require(matrix["protocol_sha256"] == self.protocol_hash and matrix["new_fits"] == 252
                and matrix["methods"] == list(METHODS) and matrix["all_methods_fresh"] is True,
                "Controller matrix belongs to a different campaign")
        require(len(jobs) == 21 and {(job["model"], job["dataset"]) for job in jobs}
                == {(model, dataset) for model in MODELS for dataset in DATASETS}
                and all(job["expected_fits"] == 12 for job in jobs), "Controller 21-job matrix is incomplete/duplicated")
        require(matrix["status"] in ("running", "complete") and not any(job["status"] == "failed" for job in jobs),
                "Controller/worker reports failure; training was left untouched")
        complete = all(job["status"].startswith("complete") for job in jobs)
        require((matrix["status"] == "complete") == complete, "Controller completion flags disagree")
        return matrix

    def observe(self):
        self.verify_sources()
        matrix = self.inspect_matrix()
        process = process_identity(self.pid)
        matches = controller_matches(process, self.run)
        if self.controller_identity is not None:
            matches = matches and all(process.get(key) == self.controller_identity.get(key)
                                      for key in ("pid", "starttime", "boot_id", "cmdline"))
        if not matches and matrix["status"] != "complete":
            # The controller can publish complete and exit between these reads.
            matrix = self.inspect_matrix()
        require(matches or matrix["status"] == "complete",
                "Controller is absent, exited, or changed PID identity while matrix remains unfinished; no automatic restart")
        if self.controller_identity is None:
            self.controller_identity = ({key: process[key] for key in ("pid", "starttime", "boot_id", "cmdline")}
                                        if matches else dict(pid=self.pid, exited_before_supervision=True))
        observed = set()
        for model, dataset, seed, method in self.expected:
            path = self.run / "results" / model / dataset / f"seed{seed}" / method / "result.json"
            if not path.exists():
                continue
            result = read(path)
            require(result.get("status") == "complete" and result.get("protocol_sha256") == self.protocol_hash
                    and (result.get("model"), result.get("dataset"), result.get("seed"), result.get("method"))
                    == (model, dataset, seed, method) and result.get("fresh_fit") is True
                    and result.get("historical_reference_used_for_training") is False,
                    "Invalid observed fresh-fit record: " + str(path))
            observed.add((model, dataset, seed, method))
        require(matrix["status"] != "complete" or observed == self.expected, "Complete controller matrix lacks 252 fresh result records")
        return dict(matrix=matrix, controller=dict(**process, matches_recorded_controller=matches), observed=observed)

    def inspect_audit(self, observed, strict=False):
        audit = read(self.run / "AUDIT.json")
        require(audit.get("migration") == (self.migration["evidence"] if self.migration else None),
                "Audit migration provenance differs from pinned supervision")
        require(audit["protocol_sha256"] == self.protocol_hash and audit["status"] in ("partial", "passed")
                and not audit["errors"] and audit["expected_new_fits"] == 252
                and audit["completed_reference_evaluations"] == audit["expected_reference_evaluations"] == 120
                and audit["old_fits_counted_as_new"] == 0, "Audit failed or belongs to a different/incomplete reference campaign")
        keys = [tuple(item["key"]) for item in audit["runs"]]
        require(len(keys) == len(set(keys)) == audit["completed_new_fits"] and set(keys) <= observed,
                "Audited fresh-fit count/identity differs from current observations")
        for record in audit["runs"]:
            model, dataset, seed, method = record["key"]
            folder = self.run / "results" / model / dataset / f"seed{seed}" / method
            require(Path(record["source"]).resolve() == folder and record["fresh_fit"] is True
                    and sha256(folder / "result.json") == record["files_sha256"]["result.json"],
                    "Previously audited result changed: " + str(folder))
        for source, digest in audit["reporter_sources_sha256"].items():
            require(self.protected.get(source) == digest, "Audit reporter source differs from pinned supervision source")
        if strict:
            require(audit["status"] == "passed" and not audit["pending"] and set(keys) == self.expected,
                    "Final report did not pass all 252 fits")
        return audit, set(keys)

    def inspect_export(self, audit, strict=False):
        path = self.run / "gated_asl_tables_metadata.json"
        metadata = read(path)
        require(metadata["status"] == ("complete" if audit["status"] == "passed" else "partial_preview")
                and metadata["completed_new_fits"] == audit["completed_new_fits"]
                and metadata["expected_new_fits"] == 252 and metadata["completed_reference_evaluations"] == 120
                and metadata["pdf_build"]["returncode"] == 0, "Export is failed/stale or omits its compiled PDF")
        require(not strict or metadata["status"] == "complete", "Final export is still partial")
        for name in ("PROTOCOL.json", "AUDIT.json", "summary.csv"):
            require(metadata["source_sha256"][name] == sha256(self.run / name), "Export source hash differs: " + name)
        require(metadata["exporter_sha256"] == self.protected[str(HERE / "export_tables.py")], "Exporter hash differs")
        require("gated_asl_tables.pdf" in metadata["outputs_sha256"], "Compiled PDF missing from output manifest")
        for name, digest in metadata["outputs_sha256"].items():
            require(Path(name).name == name and sha256(self.run / name) == digest, "Export output changed: " + name)
        return sha256(path)

    def plan(self, observation, audited):
        if observation["matrix"]["status"] == "complete" and observation["observed"] == self.expected:
            return "strict_report_then_strict_export"
        if group_keys(observation["observed"]) - group_keys(audited):
            return "partial_report_then_partial_export"
        return "wait_for_new_three_seed_group"

    def save_state(self, **changes):
        self.state.update(changes, updated_utc=stamp())
        atomic_json(self.run / "supervision.json", self.state)

    def record_observation(self, observation, audit, audited):
        self.save_state(observed_new_fits=len(observation["observed"]),
            observed_complete_groups=[list(key) for key in sorted(group_keys(observation["observed"]))],
            audited_new_fits=len(audited), audited_complete_groups=[list(key) for key in sorted(group_keys(audited))],
            completed_reference_evaluations=audit["completed_reference_evaluations"], audit_sha256=sha256(self.run / "AUDIT.json"),
            matrix_status=observation["matrix"]["status"],
            complete_worker_jobs=sum(job["status"].startswith("complete") for job in observation["matrix"]["jobs"]),
            controller_observation=observation["controller"], next_action=self.plan(observation, audited))

    def command(self, tool, strict):
        self.verify_sources()
        stage = ("strict_" if strict else "partial_") + tool
        command = ["bash", str(LAUNCHER), "-u", str(HERE / (tool + ".py")), "--run-dir", str(self.run)]
        if not strict:
            command.append("--allow-partial")
        if tool == "export_tables":
            command.append("--compile")
        log_dir = self.run / "supervision_logs"
        log_dir.mkdir(exist_ok=True)
        number = len(self.state["commands"]) + 1
        logfile = log_dir / f"{number:04d}_{stage}_{time.time_ns()}.log"
        temporary = logfile.with_suffix(".log.running")
        entry = dict(stage=stage, command=command, started_utc=stamp(), log=str(logfile.relative_to(self.run)), status="running")
        self.state["commands"].append(entry)
        self.save_state(status="reporting" if tool == "report" else "exporting", active_command=number)
        print(stage, "log:", logfile, flush=True)
        try:
            with temporary.open("x") as stream:
                process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
                stream.flush()
                os.fsync(stream.fileno())
            entry.update(returncode=process.returncode, status="passed" if process.returncode == 0 else "failed")
        except BaseException as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if temporary.exists():
                temporary.replace(logfile)
                entry["log_sha256"] = sha256(logfile)
            entry["ended_utc"] = stamp()
            self.save_state(active_command=None)
        require(process.returncode == 0, f"{stage} failed with exit {process.returncode}; inspect {logfile}; no automatic retry")

    def watch(self):
        observation = self.observe()
        audit, audited = self.inspect_audit(observation["observed"])
        if self.previous and self.previous["status"] == "ready_for_final_review":
            self.inspect_audit(observation["observed"], strict=True)
            require(observation["matrix"]["status"] == "complete", "Ready supervision has unfinished controller")
            require(self.inspect_export(audit, strict=True) == self.previous["export_metadata_sha256"]
                    and sha256(self.run / "AUDIT.json") == self.previous["audit_sha256"], "Existing ready supervision is stale")
            print("Already ready_for_final_review; no report/export repeated", flush=True)
            return
        self.state = dict(version=1, status="watching", run_dir=str(self.run), created_utc=stamp(),
            protocol_sha256=self.protocol_hash, source_manifest_sha256=self.manifest_hash,
            migration=self.migration["evidence"] if self.migration else None,
            postprocessing_sources_sha256=self.protected, runtime_sources_checked=65,
            supervisor_identity=process_identity(os.getpid()), controller_identity=self.controller_identity,
            poll_seconds=self.poll_seconds, expected_new_fits=252, expected_reference_evaluations=120,
            expected_worker_jobs=21, expected_three_seed_groups=84, commands=[],
            policy="Observe only; no training launch/restart/kill, timeout, automatic retry, seal, or goal update")
        if self.previous:
            self.state.update(created_utc=self.previous["created_utc"], commands=self.previous["commands"],
                              previous_supervisor_identity=self.previous["supervisor_identity"])
        if (self.run / "gated_asl_tables_metadata.json").exists():
            self.state["export_metadata_sha256"] = self.inspect_export(audit)
        self.record_observation(observation, audit, audited)
        print(f"Watching controller {self.pid}: observed {len(observation['observed'])}/252, "
              f"audited {len(audited)}/252, audited three-seed groups {len(group_keys(audited))}/84", flush=True)
        try:
            while True:
                action = self.plan(observation, audited)
                if action != "wait_for_new_three_seed_group":
                    strict = action.startswith("strict_")
                    trigger_groups = group_keys(observation["observed"]) - group_keys(audited)
                    self.save_state(trigger_groups=[list(key) for key in sorted(trigger_groups)])
                    self.command("report", strict)
                    observation = self.observe()
                    audit, audited = self.inspect_audit(observation["observed"], strict=strict)
                    require(trigger_groups <= group_keys(audited), "Successful report omitted a triggering complete group")
                    self.record_observation(observation, audit, audited)
                    self.command("export_tables", strict)
                    export_hash = self.inspect_export(audit, strict=strict)
                    self.save_state(export_metadata_sha256=export_hash, status="watching", active_command=None)
                    if strict:
                        self.verify_sources()
                        require(self.inspect_matrix()["status"] == "complete", "Controller changed after final export")
                        self.save_state(status="ready_for_final_review", ended_utc=stamp(), exit_code=0, next_action="manual_final_review")
                        print("ready_for_final_review: audited 252 fresh fits / 120 references; final PDF compiled; no seal executed", flush=True)
                        return
                time.sleep(self.poll_seconds)
                observation = self.observe()
                # A second owner must not rewrite an audit between supervised reports.
                require(sha256(self.run / "AUDIT.json") == self.state["audit_sha256"], "AUDIT changed outside this supervisor")
                audit, audited = self.inspect_audit(observation["observed"])
                self.record_observation(observation, audit, audited)
        except BaseException as error:
            self.save_state(status="failed", ended_utc=stamp(), exit_code=1, error=f"{type(error).__name__}: {error}",
                            training_left_untouched=True, automatic_retry=False)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--check-only", action="store_true", help="Read-only validation and planned action; no lock/state/log creation")
    args = parser.parse_args()
    if args.controller_pid <= 0 or not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("controller-pid and poll-seconds must be positive and finite")
    run = args.run_dir.resolve()
    lock = None
    try:
        if not args.check_only:
            # Separate from controller.lock. Keep this lock throughout report/export.
            lock = (run / "supervision.lock").open("a+")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        watcher = Watcher(run, args.controller_pid, args.poll_seconds)
        if args.check_only:
            observation = watcher.observe()
            audit, audited = watcher.inspect_audit(observation["observed"])
            export_hash = (watcher.inspect_export(audit) if (run / "gated_asl_tables_metadata.json").exists() else None)
            print(json.dumps(dict(status="check_passed", read_only=True, run_dir=str(run),
                protocol_sha256=watcher.protocol_hash, source_manifest_sha256=watcher.manifest_hash,
                runtime_sources_verified=65, controller=observation["controller"],
                controller_identity=watcher.controller_identity, matrix_status=observation["matrix"]["status"],
                observed_new_fits=len(observation["observed"]), audited_new_fits=len(audited),
                completed_reference_evaluations=audit["completed_reference_evaluations"],
                current_export_metadata_sha256=export_hash, current_export_matches_audit=export_hash is not None,
                newly_observed_three_seed_groups=[list(key) for key in sorted(group_keys(observation["observed"]) - group_keys(audited))],
                planned_action=watcher.plan(observation, audited)), indent=2))
        else:
            watcher.watch()
    except (OSError, ValueError, KeyError, TypeError, IndexError, KeyboardInterrupt) as error:
        print(f"Supervision stopped: {error}; training left untouched", file=sys.stderr)
        return 1
    finally:
        if lock is not None:
            lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
