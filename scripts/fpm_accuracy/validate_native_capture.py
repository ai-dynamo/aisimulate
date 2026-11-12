#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check local native FPM trace envelopes without changing or exporting records.

Original utility, independent of the Gym evaluator. Supply one worker lifetime
per identity per invocation; validate known restarts separately. Directories are
scanned in argument order, files in lexical order. Roles are caller declarations,
checked against native worker_type when present, never inferred from batch mix.
Use closed, stable files. Reading gzip to EOF cannot prove that its writer stopped
or that upstream publication was lossless. Exit 0: valid; 1: invalid; 2: review
required (or command-line error). Counter gaps alone are warnings.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sqlite3
import zlib
from collections import Counter
from contextlib import closing
from pathlib import Path

FIELDS = (
    "num_prefill_requests",
    "sum_prefill_tokens",
    "sum_prefill_kv_tokens",
    "num_decode_requests",
    "sum_decode_kv_tokens",
)
IDENTITY = ("namespace", "component", "producer_id", "worker_id", "dp_rank")
ROLES = ("aggregated", "prefill", "decode")


def nonnegative_integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a nonnegative integer")
    if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
        raise ValueError("expected a finite integer")
    if value < 0:
        raise ValueError("negative integer")
    return int(value)


def identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("expected a nonempty string or integer identity")
    return str(value)


def expected_workers(specifications):
    expected = {}
    for role, namespace, component, producer, worker, rank in specifications:
        if role not in ROLES:
            raise ValueError(f"unknown serving role: {role}")
        key = tuple(identifier(v) for v in (namespace, component, producer, worker)) + (nonnegative_integer(int(rank)),)
        if key in expected:
            raise ValueError("each expected identity must appear exactly once")
        expected[key] = role
    if not expected:
        raise ValueError("at least one expected worker is required")
    return expected


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def native_record(raw):
    event = raw["event"]
    if event["schema"] != "dynamo.fpm.trace.v1" or event["capture_mode"] != "full":
        raise ValueError("expected dynamo.fpm.trace.v1 with capture_mode=full")
    fpm, source = event["fpm"], event["source"]
    if type(fpm["version"]) is not int or fpm["version"] != 1:
        raise ValueError("expected native FPM version 1")
    key = tuple(identifier(source[name]) for name in IDENTITY[:3]) + (
        identifier(fpm["worker_id"]),
        nonnegative_integer(fpm["dp_rank"]),
    )
    counter = nonnegative_integer(fpm["counter_id"])
    nonnegative_integer(raw["timestamp"])
    nonnegative_integer(event["observed_at_unix_ms"])
    p, tokens, cached, d, context = (nonnegative_integer(fpm["scheduled_requests"][name]) for name in FIELDS)
    if (not p and (tokens or cached)) or tokens < p or (not d and context) or context < d:
        raise ValueError("inconsistent scheduled request/token counts")
    duration = fpm["wall_time"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration < 0
        or ((p or d) and duration <= 0)
    ):
        raise ValueError("wall_time must be finite seconds, nonnegative when idle and positive when busy")
    kind = "mixed" if p and d else "prefill" if p else "decode" if d else "idle"
    roles = [obj["worker_type"] for obj in (raw, event, source, fpm) if "worker_type" in obj]
    payload = json.dumps(fpm, sort_keys=True, allow_nan=False).encode()
    return key, counter, kind, roles, hashlib.sha256(payload).hexdigest()


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture_files(directories):
    files = []
    for directory in directories:
        if not directory.is_dir():
            raise ValueError(f"not a trace directory: {directory}")
        files.extend(
            sorted(
                p.resolve() for p in directory.rglob("*") if p.is_file() and p.name.endswith((".jsonl", ".jsonl.gz"))
            )
        )
    if len(files) != len(set(files)):
        raise ValueError("overlapping trace directories or duplicate input files")
    return files


def validate_capture(directories, expected):
    files = capture_files(directories)
    counts = Counter(dict.fromkeys(("raw", "valid_records", "busy", "idle", "prefill", "decode", "mixed"), 0))
    issues, examples, workers, inventories = Counter(), [], {}, []
    severity_counts = Counter()

    def issue(severity, code, path=None, line=None, detail=None):
        issues[code] += 1
        severity_counts[severity] += 1
        if len(examples) < 50:
            examples.append(
                dict(severity=severity, code=code, path=str(path) if path else None, line=line, detail=detail)
            )

    if not files:
        issue("error", "no_trace_files")
    # Temporary on-disk duplicate index keeps memory bounded for long captures.
    with closing(sqlite3.connect("")) as index:
        index.execute("CREATE TABLE seen (worker TEXT, counter TEXT, payload TEXT, PRIMARY KEY(worker, counter))")
        for path in files:
            inventory = dict(path=str(path), bytes=None, sha256=None, complete_read=False, raw=0)
            inventories.append(inventory)
            try:
                before = path.stat()
                inventory.update(bytes=before.st_size, sha256=file_hash(path))
                if path.name.endswith(".gz") and not before.st_size:
                    issue("error", "empty_gzip", path, detail="No complete gzip member")
                    continue
                opener = gzip.open if path.name.endswith(".gz") else Path.open
                with opener(path, "rt", encoding="utf-8") as stream:
                    for number, line in enumerate(stream, 1):
                        if not line.strip():
                            counts["blank_lines"] += 1
                            continue
                        counts["raw"] += 1
                        inventory["raw"] += 1
                        try:
                            raw = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                            key, counter, kind, native_roles, payload = native_record(raw)
                        except (ValueError, KeyError, TypeError, OverflowError) as error:
                            issue("error", "invalid_record", path, number, str(error))
                            continue
                        counts["valid_records"] += 1
                        counts[kind] += 1
                        if kind != "idle":
                            counts["busy"] += 1
                        if key not in workers:
                            workers[key] = dict(
                                identity=dict(zip(IDENTITY, key, strict=True)),
                                declared_role=expected.get(key),
                                counts=Counter(),
                                last_counter=None,
                            )
                            if key not in expected:
                                issue("error", "unexpected_worker", path, number, workers[key]["identity"])
                        worker = workers[key]
                        worker["counts"][kind] += 1
                        worker["counts"]["raw"] += 1
                        if kind != "idle":
                            worker["counts"]["busy"] += 1
                        if any(role != expected.get(key) for role in native_roles):
                            issue("error", "role_conflict", path, number, native_roles)
                        stream_id = json.dumps(key)
                        prior = index.execute(
                            "SELECT payload FROM seen WHERE worker=? AND counter=?", (stream_id, str(counter))
                        ).fetchone()
                        if prior:
                            code = "duplicate_counter" if prior[0] == payload else "reused_counter_different_payload"
                            issue("review", code, path, number, counter)
                        else:
                            index.execute("INSERT INTO seen VALUES (?, ?, ?)", (stream_id, str(counter), payload))
                        previous = worker["last_counter"]
                        if previous is not None and counter < previous:
                            issue(
                                "review",
                                "counter_backwards",
                                path,
                                number,
                                "Possible restart or file-order ambiguity; separate worker lifetimes",
                            )
                        elif previous is not None and counter > previous + 1:
                            absent = counter - previous - 1
                            counts["counter_jump_slots"] += absent
                            issue(
                                "warning",
                                "counter_gap",
                                path,
                                number,
                                f"Forward jump over {absent} slots in scan order; not a lost-forward count",
                            )
                        worker["last_counter"] = counter
                inventory["complete_read"] = True
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or file_hash(
                    path
                ) != inventory["sha256"]:
                    issue("error", "file_changed", path)
            except (OSError, EOFError, UnicodeError, zlib.error) as error:
                issue("error", "file_read_error", path, detail=str(error))
        if files != capture_files(directories):
            issue("error", "inventory_changed")
    for key in expected:
        if key not in workers:
            issue("error", "missing_worker", detail=dict(zip(IDENTITY, key, strict=True)))
        elif not workers[key]["counts"]["busy"]:
            issue("review", "worker_has_no_busy_records", detail=workers[key]["identity"])
    status = "invalid" if severity_counts["error"] else "review_required" if severity_counts["review"] else "valid"
    return dict(
        status=status,
        label_unit="seconds",
        counts=dict(counts),
        workers=list(workers.values()),
        files=inventories,
        issues=dict(issues),
        severity_counts=dict(severity_counts),
        issue_examples=examples,
        expected_workers=[
            dict(identity=dict(zip(IDENTITY, key, strict=True)), declared_role=role) for key, role in expected.items()
        ],
        publisher_losslessness="not_proven",
        model_readiness="not_assessed",
        limitations=[
            "Source bytes are unchanged; counts include duplicates and no rows are relabeled or exported.",
            "Roles are supplied by the caller; native worker_type is checked only when present.",
            "File integrity does not prove writer shutdown, workload completion, timing alignment "
            "or publisher losslessness.",
            "Counter gaps may include idle heartbeats; repeated/backward counters require session/order review.",
            "No training coverage, model readiness or prediction accuracy is qualified.",
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        action="append",
        required=True,
        help="closed native JSONL(.gz) directory; repeat for separate workers",
    )
    parser.add_argument(
        "--expect",
        nargs=6,
        action="append",
        required=True,
        metavar=("ROLE", "NAMESPACE", "COMPONENT", "PRODUCER_ID", "WORKER_ID", "DP_RANK"),
    )
    parser.add_argument("--report", type=Path, required=True, help="new JSON report outside the trace directories")
    args = parser.parse_args(argv)
    try:
        directories = [path.resolve() for path in args.trace_dir]
        report_path = args.report.resolve()
        if report_path.exists() or any(report_path.is_relative_to(path) for path in directories):
            raise ValueError("report must be a new path outside every trace directory")
        report = validate_capture(directories, expected_workers(args.expect))
        with report_path.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(dict(status=report["status"], counts=report["counts"], report=str(report_path))))
    return {"valid": 0, "invalid": 1, "review_required": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
