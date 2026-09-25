# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit cleanup-only public API. Never render, launch or rerun native work.

Call reconcile(request, fresh_output) only after operational authorization.
This source candidate has not performed any real cleanup or native raw read.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import shutil
import stat
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from . import cleanup_reconciliation as c
else:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_cleanup_executor_contract", Path(__file__).with_name("cleanup_reconciliation.py")
    )
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def checked(root, relative):
    """Canonical root aliases are explicit; descendants cannot be symlinks."""
    root = Path(root).resolve(strict=True)
    path = root
    for part in c.relative(relative).parts:
        path = path / part
        c.require(not path.is_symlink(), "symlink inside original storage")
    c.require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "regular original required")
    return path


def load_original(request):
    c.require(
        set(request)
        in (
            {"task_root", "attempt_directory", "references", "host_source"},
            {"task_root", "attempt_directory", "references", "host_source", "routing_mode"},
        ),
        "recovery request fields differ",
    )
    docs = {}
    for key, ref in request["references"].items():
        c.file_ref(ref)
        raw = checked(request["task_root"], ref["path"]).read_bytes()
        c.require(c.digest(raw) == ref["sha256"] and len(raw) == ref["bytes"], "original request member changed")
        docs[key] = c.embedded(raw)
    return {**request, "documents": docs}


def storage(identity):
    lexical = Path(identity["cell_directory"])
    canonical = lexical.resolve(strict=True)
    c.require(canonical.is_dir(), "original cell missing")
    c.require(Path(identity["entry"]["artifact_dir"]).resolve(strict=True) == canonical, "checkpoint storage differs")
    # Reject internal symlink components even when the task-root ancestors are
    # the pre-existing /lustre aliases. No path in original metadata is changed.
    task = Path(identity["task_root"]).resolve(strict=True)
    relative = lexical.relative_to(Path(identity["task_root"]))
    point = task
    for part in relative.parts:
        point /= part
        c.require(not point.is_symlink(), "symlink inside original cell path")
    s = canonical.stat()
    root_stat = task.stat()
    return {
        "lexical_cell": str(lexical),
        "recorded_cell": identity["entry"]["artifact_dir"],
        "canonical_task_root": str(task),
        "task_device": root_stat.st_dev,
        "task_inode": root_stat.st_ino,
        "canonical_cell": str(canonical),
        "device": s.st_dev,
        "inode": s.st_ino,
        "verified_before_and_after": True,
    }


def inventory(identity):
    task = Path(identity["task_root"]).resolve(strict=True)
    directory = task / c.relative(identity["attempt_directory"])
    c.require(directory.is_dir() and not directory.is_symlink(), "original attempt directory missing")
    files = {}
    for parent, directories, names in os.walk(directory, followlinks=False):
        for name in directories + names:
            path = Path(parent) / name
            c.require(not path.is_symlink(), "symlink inside original attempt")
        for name in names:
            path = Path(parent) / name
            c.require(stat.S_ISREG(path.stat().st_mode), "nonregular original member")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            files[str(path.relative_to(task))] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    return files


def cleanup_original(identity, output):
    """Reuse the pinned existing cleanup algorithm with fresh diagnostic sink."""
    from collector.fpm_forward.runner import _run_command
    from collector.fpm_forward.slurm import SlurmCellRunner

    c.require(
        c.digest(Path(inspect.getfile(SlurmCellRunner)).read_bytes()) == c.SLURM_SOURCE,
        "unreviewed Slurm cleanup implementation",
    )
    canonical = Path(identity["cell_directory"]).resolve(strict=True)
    step = "fpm-" + c.digest(str(canonical).encode())[:20]
    c.require(identity["owner"] == {"job_id": identity["job"], "step_name": step}, "original owned step differs")
    records = []
    mode = identity["routing_mode"]
    local_checks = []

    def file_identity(path):
        path = Path(path)
        resolved = path.resolve(strict=True)
        c.require(path.is_absolute() and resolved.is_file(), "local routing file unavailable")
        return {"path": str(path), "resolved_path": str(resolved), "sha256": c.digest(resolved.read_bytes())}

    def context():
        # Hash only routing/executable-affecting environment; do not publish
        # values (including any authentication material) in the receipt.
        environment = {
            name: value
            for name, value in os.environ.items()
            if name.startswith(("SLURM_", "SCONTROL_", "SQUEUE_", "SCANCEL_"))
            or name in {"PATH", "LD_LIBRARY_PATH", "LD_PRELOAD"}
        }
        executables = {}
        for name in ("scontrol", "squeue", "scancel"):
            path = shutil.which(name)
            c.require(path is not None, "local routing executable unavailable: " + name)
            executables[name] = file_identity(path)
        return {"environment_sha256": c.digest(c.canonical(environment)), "executables": executables}

    def command(args, label):
        # Reject ambient target/federation and filtering options: an empty
        # filtered response cannot establish that the original steps are gone.
        c.require(
            not any(
                value
                for name, value in os.environ.items()
                if name == "SLURM_CLUSTERS" or name.startswith(("SQUEUE_", "SCANCEL_", "SCONTROL_"))
            ),
            "inherited Slurm targeting/filter options must be unset",
        )
        try:
            result = _run_command(args, timeout=60, check=True)
        except BaseException as error:

            def text(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else value or ""

            write(
                output / (label + "-failure.json"),
                {
                    "argv": list(args),
                    "type": type(error).__name__,
                    "error": str(error),
                    "returncode": getattr(error, "returncode", None),
                    "stdout": text(getattr(error, "stdout", None)),
                    "stderr": text(getattr(error, "stderr", None)),
                },
            )
            raise
        row = {"argv": list(args), "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        write(output / (label + ".json"), row)
        return result, row

    initial_context = context() if mode == "verified-local" else None
    _, cluster_command = command(["scontrol", "--local", "show", "config"], "cluster-command")
    c.validate_cluster_command(cluster_command, identity["cluster"])

    def observation(label, *, prior_command=None, prior_context=None):
        before = context() if prior_context is None else prior_context
        if prior_command is None:
            _, record = command(["scontrol", "--local", "show", "config"], label)
        else:
            record = prior_command
        configuration = c.local_configuration(record, identity["cluster"])
        client = file_identity(os.environ.get("SLURM_CONF") or configuration["SLURM_CONF"])
        after = context()
        c.require(before == after, "local scheduler environment/executables changed during config query")
        result = {"command": record, "state": {**after, "configuration": configuration, "client_config": client}}
        c.validate_local_observation(result, identity["cluster"])
        write(output / (label + "-observation.json"), result)
        return result

    baseline = (
        observation("initial-local-route", prior_command=cluster_command, prior_context=initial_context)
        if mode == "verified-local"
        else None
    )

    class CleanupOnly(SlurmCellRunner):
        def __init__(self):
            # This is an original-resource descriptor, not a fake current
            # allocation. Only cleanup uses it; no SLURM_JOB_ID is assigned.
            self.cell_dir = canonical
            self.job_id = identity["job"]
            self.step_name = step
            self.owner_path = canonical.parent / ".slurm-owners" / (step + ".json")
            expected_owner = checked(identity["task_root"], identity["owner_reference_path"])
            c.require(
                expected_owner == self.owner_path
                and c.digest(expected_owner.read_bytes()) == identity["original_owner_sha256"],
                "live owner path/hash changed",
            )
            c.require(json.loads(self.owner_path.read_bytes()) == identity["owner"], "live owner changed")

        def _command(self, args, *, timeout=60, check=True):
            c.require(check is True and timeout == 60, "changed cleanup command policy")
            query = ["squeue", "--steps", "--me", "--noheader", "--format=%i|%j"]
            allowed = args == query
            if not allowed and len(args) == 2 and args[0] == "scancel":
                # The original algorithm selects its original job/name. Check
                # again before execution and reject cancellation of any other
                # job even if a future implementation changes that algorithm.
                candidate = args[1]
                allowed = (
                    candidate.startswith(self.job_id + ".")
                    and candidate[len(self.job_id) + 1 :].isdecimal()
                    and any(candidate + "|" + step in row["stdout"].splitlines() for row in records)
                )
            c.require(allowed, "non-owned cleanup command")
            c.require(
                c.digest(self.owner_path.read_bytes()) == identity["original_owner_sha256"],
                "original owned step changed before cleanup command",
            )
            before = None
            if mode == "verified-local":
                before = baseline if not records else observation(f"local-before-{len(records):03d}")
                c.require(before["state"] == baseline["state"], "local scheduler configuration/environment changed")
                current = context()
                c.require(
                    all(current[k] == baseline["state"][k] for k in current)
                    and file_identity(baseline["state"]["client_config"]["path"]) == baseline["state"]["client_config"],
                    "local routing inputs changed immediately before operation",
                )
            actual_args = (
                c.queue_argv(identity["cluster"], mode)
                if args == query
                else c.cancel_argv(identity["cluster"], args[1], mode)
            )
            result, row = command(actual_args, f"cleanup-command-{len(records):03d}")
            if mode == "verified-local":
                after = observation(f"local-after-{len(records):03d}")
                c.require(after["state"] == baseline["state"], "local scheduler configuration/environment changed")
                local_checks.append({"before": before, "after": after})
            records.append(row)
            if args == query:
                for line in result.stdout.splitlines():
                    value, separator, name = line.strip().partition("|")
                    c.require(separator and value and name, "malformed original ownership query")
            return result

        def apply(self):
            raise AssertionError("cleanup-only reconciliation cannot apply a workload")

        stage = execute = prepare_attempt = apply

    CleanupOnly().cleanup()
    return {
        "canonical_cell_directory": str(canonical),
        "owner_sha256": identity["original_owner_sha256"],
        "job_id": identity["job"],
        "step_name": step,
        "commands": records,
        "cluster_command": cluster_command,
        "routing_mode": mode,
        "local_checks": local_checks,
        "outcome": "OWNED_STEPS_ABSENT",
    }


def native_original(identity, output):
    """Read unchanged original data through public complete-cell aggregation."""
    from collector.fpm_forward.database import aggregate_cell
    from collector.fpm_forward.glm53flash_validation import installed_consumer_identity
    from collector.fpm_forward.planner import BackendPolicy, FPMCell
    from collector.fpm_forward.types import ParallelTopology

    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS

    plan = identity["plan"]
    data = plan["cells"][0]
    fields = {f.name for f in dataclasses.fields(FPMCell)}
    kw = {k: v for k, v in data.items() if k in fields}
    kw.update({k: v for k, v in data["resolved_dtypes"].items() if k in fields})
    kw.update(
        topology=ParallelTopology(**data["topology"]),
        backend_policy=BackendPolicy(**data["backend_policy"]),
        execution_identity=tuple(data["execution_identity"][k] for k in EXECUTION_COLUMNS),
    )
    cell = FPMCell(**kw)
    c.require(cell.to_dict() == data, "typed original cell changed")
    view = SimpleNamespace(
        **{k: plan[k] for k in ("backend", "model_path", "system", "sha256", "aic_revision")},
        capability=SimpleNamespace(**plan["capability"]),
        options=SimpleNamespace(**plan["options"]),
    )
    consumer = installed_consumer_identity()
    rows = aggregate_cell(view, cell, Path(identity["cell_directory"]), expected_attempt_id=identity["attempt_id"])
    write(output / "aggregated-rows.json", rows)
    return {
        "method": "PUBLIC_COMPLETE_EXACT_ATTEMPT_AGGREGATE_CELL",
        "attempt_id": identity["attempt_id"],
        "plan_sha256": identity["plan_sha256"],
        "raw_root": identity["raw_root"],
        "rows_sha256": c.digest(c.canonical(rows)),
        "point_count": len(rows),
        "consumer_identity": consumer,
        "consumer_identity_sha256": c.digest(c.canonical(consumer)),
    }


def _reconcile(original, output, *, cleanup, native, scan):
    """Injection seam for TEST_ONLY tests; public reconcile wires real APIs."""
    output = Path(output)
    c.require(not output.exists(), "fresh reconciliation output required")
    identity = c.original_identity(original)
    identity["original_owner_sha256"] = original["references"]["owner"]["sha256"]
    root = Path(identity["task_root"]).resolve(strict=True)
    candidate = output.resolve()
    original_dir = root / identity["attempt_directory"]
    c.require(
        not candidate.is_relative_to(original_dir) and not original_dir.is_relative_to(candidate),
        "diagnostics overlap original attempt",
    )
    output.mkdir(parents=True)
    write(output / "original-inputs.json", original)
    try:
        request = {key: value for key, value in original.items() if key != "documents"}
        c.require(load_original(request) == original, "original metadata changed before cleanup")
        before_storage = storage(identity)
        observed = cleanup(identity, output)
        c.validate_cleanup(observed, identity)
        write(output / "cleanup.json", observed)
        # Do not read giant raw files before verified disposal.
        before = scan(identity)
        c.require(
            {p: v for p, v in before.items() if p.endswith(".log")} == identity["review"]["logs"],
            "reviewed native log closure changed",
        )
        result = native(identity, output)
        after = scan(identity)
        c.require(after == before, "original artifacts changed during strict read")
        c.require(storage(identity) == before_storage, "original storage retargeted")
        c.require(load_original(request) == original, "original metadata changed during reconciliation")
        proof = {
            "contract": c.CONTRACT,
            "status": "COMPLETE_ORIGINAL_ATTEMPT_RECONCILED",
            "original": original,
            "cleanup": observed,
            "storage_binding": before_storage,
            "artifact_inventory": before,
            "native_validation": result,
            "source": {
                "eligibility_sha256": c.digest(Path(c.__file__).read_bytes()),
                "executor_sha256": c.digest(Path(__file__).read_bytes()),
            },
        }
        c.verify(proof)
        write(output / "receipt.json", proof)
        return proof
    except BaseException as error:
        write(
            output / "failure.json",
            {
                "contract": c.CONTRACT,
                "status": "RECONCILIATION_FAILED_PRESERVED",
                "error_type": type(error).__name__,
                "error": str(error),
                "historical_status_unchanged": True,
            },
        )
        raise


def reconcile(request, output):
    """Explicit observed reconciliation; never ordinary resume or native retry."""
    original = load_original(request)
    return _reconcile(original, output, cleanup=cleanup_original, native=native_original, scan=inventory)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reconcile(json.loads(args.request.read_bytes()), args.output)


if __name__ == "__main__":
    main()
