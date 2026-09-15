# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observe, validate and resume explicitly declared Slurm collection shards.

This hook never changes collector case IDs, failure ledgers, timing methods or
finalization. A missing family, failed case or unvalidated output prevents full
completion. Infrastructure retries are bounded per runner revision; systemic
case failures await a code fix while unrelated ready shards keep progressing.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path

import yaml


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, document):
    """Replace a hook report; collector transactions remain untouched."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n")
    temporary.replace(path)


def case_hash(cases):
    return hashlib.sha256("\n".join(sorted(cases)).encode()).hexdigest()


def validate_snapshot(job_dir, spec, *, table_reader=None):
    job_dir = Path(job_dir)
    status = read_json(job_dir / "status.json")
    if status.get("spec") != spec:
        raise ValueError("Snapshot spec differs from the immutable shard plan")
    if status.get("status") != "complete" or status.get("failed", 0):
        raise ValueError("Shard is incomplete or has failed cases")
    if status.get("done") != spec["planned_tasks"] or status.get("collector_exit_code") != 0:
        raise ValueError("Successful collector exit and full task accounting are required")
    data = job_dir / "data"
    checkpoints = [read_json(p) for p in (data / "checkpoint" / "vllm").glob("*.json")]
    done = []
    for checkpoint in checkpoints:
        if checkpoint.get("framework_version") != "0.25.0" or checkpoint.get("sm_version") != 100:
            raise ValueError("Checkpoint belongs to another runtime/platform")
        if checkpoint.get("failed"):
            raise ValueError("Failed cases remain in the original checkpoint")
        done.extend(checkpoint.get("done", []))
    if len(done) != len(set(done)) or len(done) != spec["planned_tasks"]:
        raise ValueError("Missing or duplicate checkpoint case IDs")
    if spec.get("expected_case_ids_sha256") and case_hash(done) != spec["expected_case_ids_sha256"]:
        raise ValueError("Checkpoint case set differs from the frozen plan")
    if spec["op"] == "gemm":
        prefix = "vllm.gemm:run_gemm:"
        if not all(case.startswith(prefix) for case in done):
            raise ValueError("Unexpected GEMM producer identity")
        physical = {case[len(prefix) :] for case in done}
        if spec["mode"] == "smoke" and physical != set(spec["filters"]):
            raise ValueError("Smoke did not execute its exact planned cases")
        if spec.get("case_set_sha256") and case_hash(physical) != spec["case_set_sha256"]:
            raise ValueError("GEMM shard coverage hash differs from plan")
    meta = yaml.safe_load((data / "collection_meta.yaml").read_text())
    if meta["runtime"]["version"] != "0.25.0" or meta["runtime"]["framework"] != "vllm":
        raise ValueError("Performance provenance belongs to another framework/version")
    if table_reader is None:
        import pyarrow.parquet as pq

        table_reader = lambda path: pq.read_table(path).to_pylist()
    files = {}
    row_count = 0
    for path in sorted(data.glob("*_perf.parquet")):
        rows = table_reader(path)
        entry = meta["tables"][path.stem]
        if entry.get("status") != "complete" or entry["rows"] != len(rows):
            raise ValueError("Parquet row count/status does not match its provenance")
        for row in rows:
            if row["version"] != "0.25.0" or not math.isfinite(row["latency"]):
                raise ValueError("Wrong-version or non-finite performance row")
            if row["latency"] < 0 or (row["latency"] == 0 and spec["op"] != "compute_scale"):
                raise ValueError("Invalid performance latency")
        if spec["op"] == "gemm":
            keys = {str([r["gemm_dtype"], r["m"], r["n"], r["k"]]) for r in rows}
            if len(keys) != len(rows) or keys != physical:
                raise ValueError("GEMM output keys differ from successful task keys")
        files[path.name] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        row_count += len(rows)
    if not files or not row_count:
        raise ValueError("No finalized performance rows")
    return {"done": len(done), "rows": row_count, "files": files, "job_dir": str(job_dir)}


def completion(expected_tasks, validated_by_op, unresolved_scopes):
    """GEMM-only or empty plans cannot complete an all-model campaign."""
    return (
        bool(expected_tasks)
        and not unresolved_scopes
        and set(validated_by_op) <= set(expected_tasks)
        and all(isinstance(count, int) and count > 0 for count in expected_tasks.values())
        and all(validated_by_op.get(op, 0) == count for op, count in expected_tasks.items())
    )


def job_records(root):
    records = []
    for name in ("submitted.jsonl", "full-release.jsonl", "hook-submitted.jsonl"):
        path = root / name
        if path.exists():
            records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    retry_file = root / "arrow-retries.tsv"
    if retry_file.exists():
        for line in retry_file.read_text().splitlines():
            shard, job = line.split()
            records.append({"shard": shard, "job": job, "mode": "smoke", "runner_revision": "arrow-retry"})
    return records


def latest_status(root, shard):
    paths = list((root / "results" / shard).glob("job-*/status.json"))
    return max(paths, key=lambda p: int(p.parent.name.split("-")[1])) if paths else None


def submission_environment():
    """Do not export the CPU watcher's resource limits into GPU job steps."""
    resource_variables = {
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_MEM_PER_GPU",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_PER_GPU",
        "SLURM_NTASKS",
        "SLURM_NTASKS_PER_NODE",
        "SLURM_NNODES",
        "SLURM_TRES_PER_TASK",
        "SBATCH_MEM_PER_NODE",
        "SBATCH_MEM_PER_CPU",
        "SBATCH_CPUS_PER_TASK",
        "SRUN_MEM_PER_NODE",
        "SRUN_CPUS_PER_TASK",
    }
    return {key: value for key, value in os.environ.items() if key not in resource_variables}


def submit(config, campaign, spec):
    root = Path(campaign["root"])
    args = campaign["slurm"]
    qos = "interactive" if spec["mode"] == "smoke" else "short"
    memory = args.get("memory", "128G")
    step = [
        "srun",
        f"--account={args['account']}",
        f"--qos={qos}",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:8",
        f"--mem={memory}",
        "--cpus-per-task=32",
        "--gpu-freq=1965",
        f"--container-image={args['image']}",
        f"--container-mounts={root}:/campaign",
        "--container-workdir=/campaign",
        "bash",
        "/campaign/run.sh",
        spec["id"],
    ]
    command = [
        "sbatch",
        "--parsable",
        f"--account={args['account']}",
        f"--partition={args['partition']}",
        f"--qos={qos}",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:8",
        "--exclusive",
        "--cpus-per-task=32",
        f"--mem={memory}",
        "--time=" + ("00:15:00" if spec["mode"] == "smoke" else "02:00:00"),
        f"--output={root}/logs/hook-{spec['id']}-%j.out",
        "--wrap=" + shlex.join(step),
    ]
    job = subprocess.check_output(command, text=True, env=submission_environment()).strip().split(";")[0]
    record = {
        "shard": spec["id"],
        "job": job,
        "mode": spec["mode"],
        "planned_tasks": spec["planned_tasks"],
        "runner_revision": campaign["runner_revision"],
        "submitted_unix": time.time(),
        "command": command,
    }
    with (root / "hook-submitted.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
        output.flush()
        os.fsync(output.fileno())
    return record


def tick(config, *, apply=False):
    queue = subprocess.check_output(["squeue", "-h", "-u", config["user"], "-o", "%i|%T"], text=True)
    active = {line.split("|")[0] for line in queue.splitlines() if "|" in line}
    report = {"checked_unix": time.time(), "shards": [], "validated_by_op": {}, "submitted": []}
    launch_slots = max(0, config["max_active_jobs"] - len(active))
    for campaign in config["campaigns"]:
        root = Path(campaign["root"])
        plan = read_json(root / campaign["plan"])
        specs = plan["smokes"] + plan["shards"]
        records = job_records(root)
        smoke_validated = set()
        for spec in specs:
            state = {"campaign": str(root), "id": spec["id"], "op": spec["op"], "mode": spec["mode"]}
            report["shards"].append(state)
            relevant = [r for r in records if r["shard"] == spec["id"]]
            path = latest_status(root, spec["id"])
            status = read_json(path) if path else {}
            if any(str(r["job"]) in active for r in relevant):
                state.update(
                    state="active",
                    active_jobs=[str(r["job"]) for r in relevant if str(r["job"]) in active],
                    observed_done=status.get("done", 0),
                    observed_failed=status.get("failed", 0),
                )
                continue
            if status.get("status") == "complete":
                try:
                    verified = validate_snapshot(path.parent, spec)
                    state.update(state="validated", validation=verified)
                    if spec["mode"] == "smoke":
                        smoke_validated.add(spec["id"])
                    else:
                        totals = report["validated_by_op"]
                        totals[spec["op"]] = totals.get(spec["op"], 0) + verified["done"]
                    continue
                except (OSError, ValueError, KeyError) as error:
                    state.update(state="validation_failed", error=str(error))
                    continue
            if status.get("failed", 0):
                state.update(state="needs_case_fix", failed=status["failed"], done=status.get("done", 0))
                continue
            if spec.get("smoke_dependency") and spec["smoke_dependency"] not in smoke_validated:
                state["state"] = "waiting_for_smoke"
                continue
            if path and status.get("status") == "running" and time.time() - path.stat().st_mtime < 120:
                state["state"] = "finishing_or_queue_transition"
                continue
            attempts = sum(r.get("runner_revision") == campaign["runner_revision"] for r in relevant)
            if attempts >= config["max_infra_attempts_per_revision"]:
                state["state"] = "needs_infrastructure_fix"
                continue
            state["state"] = "ready_to_resume" if path else "ready_to_start"
            if apply and launch_slots:
                record = submit(config, campaign, spec)
                report["submitted"].append(record)
                state.update(state="submitted", job=record["job"])
                launch_slots -= 1
    report["missing_or_incomplete_ops"] = {
        op: {"expected": count, "validated": report["validated_by_op"].get(op, 0)}
        for op, count in config["expected_tasks"].items()
        if report["validated_by_op"].get(op, 0) != count
    }
    report["unresolved_scopes"] = config.get("unresolved_scopes", [])
    report["all_data_complete"] = completion(
        config["expected_tasks"], report["validated_by_op"], report["unresolved_scopes"]
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Submit ready/resumable task-owned shards")
    parser.add_argument("--watch", action="store_true", help="Keep watching until the entire declared scope validates")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    root = args.config.resolve().parent
    with (root / "collection-hook.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            config = read_json(args.config)
            try:
                report = tick(config, apply=args.apply)
            except Exception as error:
                report = {"checked_unix": time.time(), "all_data_complete": False, "hook_error": repr(error)}
                if not args.watch:
                    atomic_json(root / "hook_status.json", report)
                    raise
            atomic_json(root / "hook_status.json", report)
            print(json.dumps(report), flush=True)
            if not args.watch or report.get("all_data_complete"):
                return 0 if report.get("all_data_complete") else 2
            time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
