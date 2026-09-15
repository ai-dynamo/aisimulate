# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute documented collector CLI filters on node-local storage.

Case identity, timing, classification and parquet finalization remain owned by
collect.py. This wrapper snapshots quiescent results and cleans up only stuck
child workers AFTER the official sidecar is committed and all IDs accounted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


def checkpoint_state(data):
    details = {}
    done = set()
    failed = set()
    for path in (Path(data) / "checkpoint/vllm").glob("*.json"):
        document = json.loads(path.read_text())
        passed_ids = set(document.get("done", []))
        failed_ids = set(document.get("failed", []))
        done.update(passed_ids)
        failed.update(failed_ids)
        details[path.name] = {"done": len(passed_ids), "failed": len(failed_ids)}
    if done & failed:
        raise ValueError("A checkpoint ID cannot be both passed and failed")
    return details, done, failed


def finalized_all_ids(data, planned_tasks):
    data = Path(data)
    _, done, failed = checkpoint_state(data)
    pending = (
        ".perf-finalization.transaction.json",
        ".collection_meta.transaction.json",
        ".collection_meta.pending.yaml",
    )
    return (
        len(done | failed) == planned_tasks
        and (data / "collection_meta.yaml").is_file()
        and not any((data / name).exists() for name in pending)
    )


def reap_finished_workers(parent_pid, *, proc_root=Path("/proc")):
    """Never match process names globally: require exact parent and own UID."""
    killed = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = dict(line.split(":", 1) for line in (entry / "status").read_text().splitlines() if ":" in line)
            command = (entry / "cmdline").read_bytes()
            if (
                int(fields["PPid"].strip()) == parent_pid
                and int(fields["Uid"].split()[0]) == os.getuid()
                and b"multiprocessing.spawn" in command
            ):
                pid = int(entry.name)
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
        except (FileNotFoundError, ProcessLookupError, PermissionError, KeyError):
            continue
    return killed


def save(path, document):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_id")
    parser.add_argument("--root", type=Path, default=Path("/campaign"))
    parser.add_argument("--plan")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.shard_id) or args.shard_id in {".", ".."}:
        parser.error("shard_id must be a single safe path component")
    root = args.root
    candidates = [
        root / name
        for name in ("gemm-shards.json", "ops-shards.json", "mla-shards.json", "final-shards.json")
        if (root / name).exists()
    ]
    if args.plan:
        plan_path = root / args.plan
    elif len(candidates) == 1:
        plan_path = candidates[0]
    else:
        parser.error("Provide exactly one campaign plan")
    manifest = json.loads(plan_path.read_text())
    spec = next(s for s in manifest["smokes"] + manifest["shards"] if s["id"] == args.shard_id)
    base = root / "results" / args.shard_id
    job = base / ("job-" + os.environ["SLURM_JOB_ID"])
    job.mkdir(parents=True, exist_ok=True)
    data = Path(manifest["local_root"]) / args.shard_id / "data"
    data.mkdir(parents=True, exist_ok=True)
    previous = base / "latest_snapshot.json"
    if previous.exists():
        snapshot = root / json.loads(previous.read_text())["relative_path"]
        shutil.copytree(snapshot, data, dirs_exist_ok=True)
    status = {
        "spec": spec,
        "source_commit": manifest["source_commit"],
        "clock_policy": manifest["clock_policy"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_unix": time.time(),
        "status": "starting",
    }
    command = [
        sys.executable,
        "-u",
        str(root / "source/python/aisimulate/collector/collect.py"),
        "--backend",
        "vllm",
        "--ops",
        spec["op"],
        "--sm",
        "100",
        "--resume",
        "--checkpoint-dir",
        str(data / "checkpoint"),
    ]
    if spec.get("model_cases_full", True):
        command.append("--model-cases-full")
    for fragment in spec["filters"]:
        command.extend(["--case-filter", fragment])
    status["command"] = command
    status_path = job / "status.json"
    save(status_path, status)
    review = manifest.get("reviewed_failure_continuation", {})
    reviewed = spec["op"] in review
    status["failure_review"] = review.get(spec["op"])
    finalized_since = None
    with (job / "collector.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=data, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status.update(status="running", collector_pid=proc.pid)
        save(status_path, status)
        while proc.poll() is None:
            time.sleep(15)
            details, done, failed = checkpoint_state(data)
            status.update(checkpoints=details, done=len(done), failed=len(failed))
            accounted = len(done | failed)
            if finalized_all_ids(data, spec["planned_tasks"]):
                finalized_since = finalized_since or time.monotonic()
                if time.monotonic() - finalized_since > 60:
                    killed = reap_finished_workers(proc.pid)
                    if killed:
                        status.setdefault("completed_worker_cleanup", []).append({"time": time.time(), "pids": killed})
            should_stop = False
            if not reviewed and accounted >= 100 and len(failed) / accounted >= 1 / 3:
                status.update(
                    status="stopped_systemic_failure",
                    stop_reason="Unreviewed systemic failures; all observations retained",
                )
                should_stop = True
            if time.time() - status["started_unix"] > 6600:
                status["stop_reason"] = "walltime_checkpoint_yield"
                should_stop = True
            if should_stop:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            save(status_path, status)
    status["collector_exit_code"] = proc.wait()
    details, done, failed = checkpoint_state(data)
    status.update(checkpoints=details, done=len(done), failed=len(failed))
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    snapshot = job / "data"
    shutil.copytree(data, snapshot, dirs_exist_ok=True)
    save(
        previous,
        {
            "relative_path": str(snapshot.relative_to(root)),
            "job": os.environ["SLURM_JOB_ID"],
            "collector_exit_code": status["collector_exit_code"],
        },
    )
    status.update(
        snapshot=str(snapshot.relative_to(root)),
        finished_unix=time.time(),
        parquet_files=[p.name for p in data.glob("*_perf.parquet")],
        sidecar_present=(data / "collection_meta.yaml").is_file(),
    )
    complete = (
        len(done | failed) == spec["planned_tasks"]
        and status["collector_exit_code"] == 0
        and bool(done)
        and status["sidecar_present"]
    )
    if status["status"] != "stopped_systemic_failure":
        status["status"] = "complete_with_failures" if complete and failed else "complete" if complete else "incomplete"
    save(status_path, status)
    print(json.dumps(status, indent=2), flush=True)
    return 0 if complete and (spec["mode"] != "smoke" or not failed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
