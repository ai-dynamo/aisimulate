# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit eligibility of one unchanged cleanup-only original attempt.

This module performs no Slurm operation or native execution. Historical failed
states remain historical. A successful reconciliation is a separate proof.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path

CONTRACT = "fpm_cleanup_reconciliation_v1"
SELECTION_SCHEMA = "fpm_complete_child_selection_reconciled_v2"
LEGACY_SELECTION_SCHEMA = "fpm_complete_child_selection_v1"
REVIEW_CONTRACT = "fpm_original_terminal_failure_review_v1"
SLURM_SOURCE = "432dc0b578a8d738bf0b691de5009f2df2bb274b75ecb74363f7fc92c2d9e01b"
HOST_SOURCES = {
    "0469efad05c6395420993a6b3fe24ac72e4d050a": "d69825d13e5f492eff167f8e25126e41916dec101a3cbf0228ede794358bad2c",
    "642b23b79f7254664483f334928935e965809efb": "208e03e8fe22177f987da088e78677822826bc91c5ce5c4fa51515ae67120686",
}


def require(ok, message):
    if not ok:
        raise ValueError("cleanup reconciliation: " + message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def relative(value):
    require(isinstance(value, str) and value, "empty relative path")
    p = Path(value)
    require(not p.is_absolute() and value != "." and ".." not in p.parts and str(p) == value, "unsafe relative path")
    return p


def sha(value):
    require(isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef"), "invalid SHA256")
    return value


def file_ref(value):
    require(isinstance(value, dict) and set(value) == {"path", "sha256", "bytes"}, "invalid file reference")
    relative(value["path"])
    sha(value["sha256"])
    require(type(value["bytes"]) is int and value["bytes"] >= 0, "invalid file size")
    return value


def embedded(raw):
    return {"sha256": digest(raw), "bytes": len(raw), "base64": base64.b64encode(raw).decode()}


def decode(record):
    require(isinstance(record, dict) and set(record) == {"sha256", "bytes", "base64"}, "invalid original bytes")
    raw = base64.b64decode(record["base64"], validate=True)
    require(digest(raw) == record["sha256"] and len(raw) == record["bytes"], "original bytes changed")
    return json.loads(raw)


def original_identity(original):
    """Validate exact original metadata and the reviewed post-collection branch."""
    require(
        set(original) == {"task_root", "attempt_directory", "references", "documents", "host_source"},
        "original fields differ",
    )
    task = Path(original["task_root"])
    require(task.is_absolute() and ".." not in task.parts, "invalid original task root")
    directory = relative(original["attempt_directory"])
    refs, docs = original["references"], original["documents"]
    require(
        set(refs) == set(docs) == {"started", "final", "checkpoint", "plan", "owner", "failure_review"},
        "missing original inputs",
    )
    values = {}
    for key in refs:
        ref = file_ref(refs[key])
        record = docs[key]
        require(all(record[k] == ref[k] for k in ("sha256", "bytes")), "original reference differs: " + key)
        values[key] = decode(record)
    start, final, checkpoint, plan, owner, review = (
        values[k] for k in ("started", "final", "checkpoint", "plan", "owner", "failure_review")
    )
    job = str(start["job"])
    require(job.isdecimal() and str(final["job"]) == job and directory.name == job, "original job differs")
    backend = plan["backend"]
    require(backend in {"vllm", "sglang"}, "unsupported backend")
    child = start["child"] if backend == "vllm" else start["selected"]["child_identity"]
    other = final["child"] if backend == "vllm" else final["selected"]["child_identity"]
    require(child == other, "original child changed")
    cid, plan_sha = child["child_cell_id"], sha(child["child_plan_sha256"])
    require(plan["sha256"] == checkpoint["plan_sha256"] == plan_sha, "plan/checkpoint differs")
    require(
        len(plan["cells"]) == 1 and plan["cells"][0]["cell_id"] == cid and set(checkpoint["cells"]) == {cid},
        "not one original whole child",
    )
    entry = checkpoint["cells"][cid]
    require(
        final["state"] == "COLLECTION_FAILED_PRESERVED" and entry["status"] == "cleanup_failed",
        "not historical cleanup failure",
    )
    require(
        entry.get("attempt_id")
        and not entry.get("error")
        and not entry.get("error_type")
        and not final.get("exception"),
        "primary failure is ineligible",
    )
    errors = final.get("errors")
    require(
        isinstance(errors, list)
        and errors
        and all(e.get("classification") == "resource_cleanup_failed" and e.get("cell_id") == cid for e in errors),
        "not solely resource cleanup failure",
    )
    require(isinstance(entry.get("cleanup_error"), str) and entry["cleanup_error"], "original cleanup error missing")
    marks = entry.get("collector_phase_seconds", {})
    require(
        all(
            type(marks.get(k)) in (int, float) and math.isfinite(marks[k]) and marks[k] >= 0
            for k in ("render_s", "schedule_s", "stage_s", "execute_wall_s", "collect_s")
        ),
        "post-collection order not observed",
    )
    host = original["host_source"]
    require(
        set(host) == {"commit", "runner_sha256", "slurm_sha256"}
        and HOST_SOURCES.get(host["commit"]) == host["runner_sha256"]
        and host["slurm_sha256"] == SLURM_SOURCE,
        "unreviewed original host source",
    )
    observed_host = final.get("host_source_commit", start.get("host_source_commit", start.get("source_commit")))
    require(observed_host == host["commit"], "actual host source differs")
    expected = {
        "started": directory / "started.json",
        "final": directory / "result.json",
        "checkpoint": directory / "checkpoint/fpm_forward.json",
    }
    require(all(refs[k]["path"] == str(p) for k, p in expected.items()), "original metadata path changed")
    cell_dir = task / directory / "artifacts" / plan_sha[:16] / "cells" / cid
    require(
        Path(entry["artifact_dir"]).is_absolute() and ".." not in Path(entry["artifact_dir"]).parts,
        "invalid recorded original cell directory",
    )
    require(
        review.get("contract") == REVIEW_CONTRACT
        and review.get("job") == job
        and review.get("attempt_id") == entry["attempt_id"],
        "terminal review identifies another attempt",
    )
    require(
        review.get("outcome") == "SOLE_POST_COLLECTION_CLEANUP_FAILURE"
        and review.get("native_failures") == []
        and review.get("unresolved_findings") == [],
        "native/watchdog/unknown failure remains",
    )
    require(
        review.get("original_final_sha256") == refs["final"]["sha256"]
        and review.get("original_checkpoint_sha256") == refs["checkpoint"]["sha256"],
        "terminal review not bound to original failure",
    )
    require(isinstance(review.get("logs"), dict) and review["logs"], "terminal review lacks original logs")
    for path, info in review["logs"].items():
        require(relative(path).is_relative_to(directory), "terminal review log crosses original attempt")
        require(set(info) == {"sha256", "bytes"}, "invalid reviewed log identity")
        sha(info["sha256"])
        require(type(info["bytes"]) is int and info["bytes"] >= 0, "invalid reviewed log bytes")
    return {
        "backend": backend,
        "job": job,
        "cell_id": cid,
        "plan_sha256": plan_sha,
        "attempt_id": entry["attempt_id"],
        "task_root": str(task),
        "attempt_directory": str(directory),
        "cell_directory": str(cell_dir),
        "raw_root": str(cell_dir / "raw/node0000"),
        "owner": owner,
        "owner_reference_path": refs["owner"]["path"],
        "review": review,
        "plan": plan,
        "entry": entry,
    }


def validate_cleanup(cleanup, identity):
    require(
        set(cleanup) == {"canonical_cell_directory", "owner_sha256", "job_id", "step_name", "commands", "outcome"},
        "cleanup fields differ",
    )
    canonical_cell = Path(cleanup["canonical_cell_directory"])
    require(canonical_cell.is_absolute() and ".." not in canonical_cell.parts, "invalid canonical cell")
    # A live samefile binding is produced by the executor; offline this is its
    # immutable descriptor. It never rewrites original lexical metadata.
    require(cleanup["job_id"] == identity["job"], "cleanup job differs")
    step = "fpm-" + digest(str(canonical_cell).encode())[:20]
    require(
        cleanup["step_name"] == step and identity["owner"] == {"job_id": identity["job"], "step_name": step},
        "exact original owner differs",
    )
    require(
        identity["owner_reference_path"]
        == str(
            Path(identity["attempt_directory"])
            / "artifacts"
            / identity["plan_sha256"][:16]
            / "cells/.slurm-owners"
            / (step + ".json")
        ),
        "owner receipt path differs",
    )
    commands = cleanup["commands"]
    require(
        isinstance(commands, list) and len(commands) >= 2 and cleanup["outcome"] == "OWNED_STEPS_ABSENT",
        "teardown not observed",
    )
    require(
        commands[0]["argv"] == ["squeue", "--steps", "--me", "--noheader", "--format=%i|%j"],
        "missing initial ownership query",
    )
    seen = set()
    queries = 0
    for index, command in enumerate(commands):
        require(
            set(command) == {"argv", "returncode", "stdout", "stderr"} and command["returncode"] == 0,
            "cleanup command failed or incomplete",
        )
        require(
            isinstance(command["stdout"], str) and isinstance(command["stderr"], str), "cleanup raw streams missing"
        )
        argv = command["argv"]
        if argv == ["squeue", "--steps", "--me", "--noheader", "--format=%i|%j"]:
            current = set()
            for line in command["stdout"].splitlines():
                value, sep, name = line.strip().partition("|")
                require(sep and value and name, "malformed step query")
                if (
                    name == step
                    and value.startswith(identity["job"] + ".")
                    and value[len(identity["job"]) + 1 :].isdecimal()
                ):
                    current.add(value)
            if queries == 0:
                seen = current
            queries += 1
            if index == len(commands) - 1:
                require(not current, "owned steps remain")
        else:
            require(
                isinstance(argv, list) and len(argv) == 2 and argv[0] == "scancel" and argv[1] in seen,
                "attempt to cancel unrelated/unobserved step",
            )
            seen.remove(argv[1])
    require(queries >= 2 and commands[-1]["argv"][0] == "squeue", "missing final verified query")


def verify(proof, *, expected_original_refs=None):
    require(
        isinstance(proof, dict)
        and proof.get("contract") == CONTRACT
        and proof.get("status") == "COMPLETE_ORIGINAL_ATTEMPT_RECONCILED",
        "successful reconciliation missing",
    )
    require(
        set(proof)
        == {
            "contract",
            "status",
            "original",
            "cleanup",
            "storage_binding",
            "artifact_inventory",
            "native_validation",
            "source",
        },
        "reconciliation fields differ",
    )
    identity = original_identity(proof["original"])
    if expected_original_refs is not None:
        for key, ref in expected_original_refs.items():
            require(proof["original"]["references"].get(key) == ref, "another original reference: " + key)
    require(
        proof["cleanup"]["owner_sha256"] == proof["original"]["references"]["owner"]["sha256"],
        "owner receipt hash changed",
    )
    validate_cleanup(proof["cleanup"], identity)
    storage = proof["storage_binding"]
    require(
        set(storage)
        == {
            "lexical_cell",
            "recorded_cell",
            "canonical_cell",
            "canonical_task_root",
            "task_device",
            "task_inode",
            "device",
            "inode",
            "verified_before_and_after",
        }
        and storage["lexical_cell"] == identity["cell_directory"]
        and storage["canonical_cell"] == proof["cleanup"]["canonical_cell_directory"]
        and storage["verified_before_and_after"] is True,
        "live original storage binding missing",
    )
    require(
        storage["recorded_cell"] == identity["entry"]["artifact_dir"]
        and storage["recorded_cell"] in {storage["lexical_cell"], storage["canonical_cell"]},
        "recorded original storage differs",
    )
    canonical_root = Path(storage["canonical_task_root"])
    suffix = Path(identity["cell_directory"]).relative_to(identity["task_root"])
    require(
        canonical_root.is_absolute()
        and ".." not in canonical_root.parts
        and str(canonical_root / suffix) == storage["canonical_cell"],
        "canonical task storage differs",
    )
    require(
        type(storage["task_device"]) is int and type(storage["task_inode"]) is int and storage["task_inode"] > 0,
        "task storage identity missing",
    )
    require(
        type(storage["device"]) is int and type(storage["inode"]) is int and storage["inode"] > 0,
        "invalid original storage identity",
    )
    inventory = proof["artifact_inventory"]
    require(isinstance(inventory, dict) and inventory, "complete immutable original inventory missing")
    directory = Path(identity["attempt_directory"])
    for name, info in inventory.items():
        require(
            relative(name).is_relative_to(directory) and set(info) == {"sha256", "bytes"},
            "inventory crosses original attempt",
        )
        sha(info["sha256"])
        require(type(info["bytes"]) is int and info["bytes"] >= 0, "invalid inventory bytes")
    logs = {p: v for p, v in inventory.items() if p.endswith(".log")}
    require(logs == identity["review"]["logs"], "terminal review does not cover exact original logs")
    for key in ("started", "final", "checkpoint"):
        ref = proof["original"]["references"][key]
        require(
            inventory.get(ref["path"]) == {k: ref[k] for k in ("sha256", "bytes")},
            "inventory omits original failure metadata",
        )
    native = proof["native_validation"]
    require(
        set(native)
        == {
            "method",
            "attempt_id",
            "plan_sha256",
            "raw_root",
            "rows_sha256",
            "point_count",
            "consumer_identity",
            "consumer_identity_sha256",
        }
        and native["method"] == "PUBLIC_COMPLETE_EXACT_ATTEMPT_AGGREGATE_CELL",
        "strict original native read missing",
    )
    require(
        all(native[k] == identity[k] for k in ("attempt_id", "plan_sha256", "raw_root")),
        "native reader changed original attempt",
    )
    sha(native["rows_sha256"])
    require(
        native["consumer_identity"]
        and digest(canonical(native["consumer_identity"])) == sha(native["consumer_identity_sha256"]),
        "actual reader identity missing",
    )
    points = identity["plan"]["options"]["benchmark_points"]["payload"]
    phase = identity["plan"]["cells"][0]["workload_kind"]
    require(
        type(native["point_count"]) is int
        and native["point_count"] == len(points[phase])
        and native["point_count"] > 0,
        "incomplete original whole child",
    )
    require(
        set(proof["source"]) == {"eligibility_sha256", "executor_sha256"}
        and proof["source"]["eligibility_sha256"] == digest(Path(__file__).read_bytes())
        and proof["source"]["executor_sha256"] == digest(Path(__file__).with_name("cleanup_executor.py").read_bytes()),
        "reconciliation source closure differs",
    )
    return identity


def verify_archive_inventory(proof, records, *, campaign_root):
    identity = verify(proof)
    root = relative(campaign_root)
    expected = proof["artifact_inventory"]
    attempt = Path(identity["attempt_directory"]).relative_to(root)
    actual = {
        str(root / p): {"sha256": row["sha256"], "bytes": row["stat"]["size"]}
        for p, row in records.items()
        if row["kind"] == "file" and Path(p).is_relative_to(attempt)
    }
    require(actual == expected, "archived original attempt differs from reconciliation inventory")
    return identity


def selection_schema(ledger):
    require(ledger.get("schema") in {LEGACY_SELECTION_SCHEMA, SELECTION_SCHEMA}, "unknown whole-child selection schema")
    if ledger["schema"] == LEGACY_SELECTION_SCHEMA:
        require(
            all("reconciliation" not in choice for choice in ledger["selections"].values()),
            "legacy selection cannot carry reconciliation",
        )


def selected_proof(ledger, choice, attempt, load):
    """Validate an explicit whole-child choice; never modify original statuses."""
    selection_schema(ledger)
    if "reconciliation" not in choice:
        require(attempt["terminal_state"] == "COLLECTION_PASSED", "historical failure needs explicit reconciliation")
        return None
    require(ledger["schema"] == SELECTION_SCHEMA, "reconciliation requires explicit v2 ledger")
    require(
        attempt["terminal_state"] == "COLLECTION_FAILED_PRESERVED",
        "reconciliation is not for successful/unknown original state",
    )
    ref = file_ref(choice["reconciliation"])
    proof = load(ref)
    identity = verify(proof, expected_original_refs={key: attempt[key] for key in ("started", "final", "checkpoint")})
    require(
        identity["job"] == choice["job"] == attempt["job"]
        and identity["cell_id"] == attempt["cell_id"]
        and identity["attempt_directory"] == attempt["original_attempt_directory"],
        "reconciliation selected another original attempt",
    )
    return proof
