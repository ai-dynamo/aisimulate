#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage reviewed native rows for offline training or later online feedback.

Reads ROLE.selection.json and receipts/ROLE.capture.json under --run-dir. Each
selection entry supplies path (relative to the run), sha256, first_line and
last_line (inclusive, one range per file, in reviewed segment order). The capture
report must be valid for this role. The user selects the workload population;
this helper cannot identify smoke, warmup or drain from a busy record.

Writes selected/ROLE.{train,test}.jsonl and receipts/ROLE.selection.json. Busy
native lines remain byte-for-byte unchanged; idle rows are excluded. Training
gets the first, third, etc. observation of each work kind. Outputs must be fresh.
Failure retains .partial files; require a completed receipt before training.

--mode online instead reads ROLE.online.selection.json and
receipts/ROLE.online.capture.json, then writes selected/ROLE.online.jsonl and
receipts/ROLE.online.selection.json. It requires the original offline inputs,
receipt and train/test outputs, and rejects reuse of their capture sources or
rows. Feedback uses one expected DP=0 worker stream, preserving reviewed file
and row order without alternating. Idle and role-incompatible work is excluded.
Observed timestamps must be nondecreasing and strictly later than every offline
row; ties at that boundary require a later capture. Counters must increase and
must exceed the original capture's counter when continuing the same stream.
Worker restarts and merged streams are unsupported. These conservative checks
do not prove timing-label accuracy or matching hardware/model/runtime identity;
the operator must review that the deployment is unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

if __package__:
    from .validate_native_capture import IDENTITY, ROLES, file_hash, native_record, reject_constant, unique_object
else:
    from validate_native_capture import IDENTITY, ROLES, file_hash, native_record, reject_constant, unique_object


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    data = path.read_bytes()
    return json.loads(data, object_pairs_hook=unique_object, parse_constant=reject_constant), hashlib.sha256(
        data
    ).hexdigest()


def file_stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def offline_inputs(run, role):
    """Verify the completed split and derive exclusion bounds in one row scan."""
    selection_path = run / f"{role}.selection.json"
    report_path = run / f"receipts/{role}.capture.json"
    receipt_path = run / f"receipts/{role}.selection.json"
    _, selection_hash = read_json(selection_path)
    report, report_hash = read_json(report_path)
    receipt, receipt_hash = read_json(receipt_path)
    require(
        receipt["role"] == role and receipt["split"] == "alternating_per_work_kind",
        "require the completed offline selection receipt for this role",
    )
    require(
        receipt["selection_sha256"] == selection_hash and receipt["capture_report_sha256"] == report_hash,
        "offline selection or capture report differs from its completed receipt",
    )
    require(report["status"] == "valid", "require a valid offline capture report")
    expected = report["expected_workers"]
    require(expected and all(worker["declared_role"] == role for worker in expected), "offline role mismatch")
    identities = {tuple(worker["identity"][name] for name in IDENTITY) for worker in expected}
    forbidden_hashes = {item["sha256"] for item in report["files"]}
    forbidden_paths = {Path(item["path"]).resolve() for item in report["files"]}
    last_counters = {
        tuple(worker["identity"][name] for name in IDENTITY): worker["last_counter"] for worker in report["workers"]
    }
    stamps, latest_observed = {}, -1
    for part in ("train", "test"):
        path = run / f"selected/{role}.{part}.jsonl"
        output = receipt["outputs"][part]
        require(output["path"] == str(path.relative_to(run)), "offline output path differs from its receipt")
        stamps[path] = file_stamp(path)
        digest, rows = hashlib.sha256(), 0
        with path.open("rb") as source:
            for line in source:
                digest.update(line)
                raw = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                identity, counter, kind, native_roles, _ = native_record(raw)
                require(
                    identity in identities and all(value == role for value in native_roles) and kind != "idle",
                    "offline output conflicts with its capture identity/role or contains idle rows",
                )
                latest_observed = max(latest_observed, raw["event"]["observed_at_unix_ms"])
                last_counters[identity] = max(last_counters.get(identity, -1), counter)
                rows += 1
        require(rows > 0 and digest.hexdigest() == output["sha256"], "offline output hash differs from its receipt")
        require(file_stamp(path) == stamps[path], "offline output changed during verification")
        forbidden_hashes.add(output["sha256"])
        forbidden_paths.add(path.resolve())
    return {
        "hashes": {selection_path: selection_hash, report_path: report_hash, receipt_path: receipt_hash},
        "stamps": stamps,
        "forbidden_hashes": forbidden_hashes,
        "forbidden_paths": forbidden_paths,
        "last_counters": last_counters,
        "latest_observed": latest_observed,
        "receipt_hash": receipt_hash,
    }


def prepare(run, role, mode="train-test"):
    run = Path(run).resolve()
    require(role in ROLES, "unknown serving role")
    require(mode in ("train-test", "online"), "unknown preparation mode")
    online = mode == "online"
    suffix = ".online" if online else ""
    selection_path = run / f"{role}{suffix}.selection.json"
    report_path = run / f"receipts/{role}{suffix}.capture.json"
    selection, selection_hash = read_json(selection_path)
    report, report_hash = read_json(report_path)
    require(report["status"] == "valid", "resolve capture validation before selecting observations")
    expected = report["expected_workers"]
    require(
        expected and all(worker["declared_role"] == role for worker in expected),
        "capture report must bind every expected worker to the selected role",
    )
    if online:
        require(
            len(expected) == 1 and expected[0]["identity"]["dp_rank"] == 0,
            "online feedback requires exactly one expected DP=0 worker stream; separate restarts and streams",
        )
    offline = offline_inputs(run, role) if online else None
    identities = {tuple(worker["identity"][name] for name in IDENTITY) for worker in expected}
    validated = {item["sha256"] for item in report["files"] if item["complete_read"] is True}
    require(isinstance(selection, list) and selection, "selection must be a nonempty list")
    sources, seen = [], set()
    for entry in selection:
        relative = Path(entry["path"])
        path = (run / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(run), "source must be relative and inside this run")
        digest = file_hash(path)
        require(digest == entry["sha256"] and digest in validated, "source hash is not in the valid capture report")
        if online:
            require(
                digest not in offline["forbidden_hashes"] and path not in offline["forbidden_paths"],
                "online feedback cannot reuse an offline capture source or train/test output",
            )
        require(digest not in seen, "source repeated in selection; use one reviewed range per file")
        seen.add(digest)
        first, last = entry["first_line"], entry["last_line"]
        require(type(first) is int and type(last) is int and 1 <= first <= last, "invalid inclusive line range")
        sources.append((path, digest, first, last))

    targets = {part: run / f"selected/{role}.{part}.jsonl" for part in (("online",) if online else ("train", "test"))}
    receipt_path = run / f"receipts/{role}{suffix}.selection.json"
    for path in (*targets.values(), receipt_path):
        require(path.resolve().is_relative_to(run), "output must stay inside the run directory")
        require(not path.exists() and not Path(str(path) + ".partial").exists(), "choose fresh outputs and receipt")
    (run / "selected").mkdir(exist_ok=True)
    counts, totals, excluded = Counter(), Counter(), Counter()
    last_observed = offline["latest_observed"] if online else None
    with ExitStack() as stack:
        outputs = {part: stack.enter_context(Path(str(path) + ".partial").open("xb")) for part, path in targets.items()}
        for path, digest, first, last in sources:
            opener = gzip.open if path.suffix == ".gz" else Path.open
            number = 0
            with opener(path, "rb") as source:
                for number, line in enumerate(source, 1):
                    if not first <= number <= last:
                        excluded["outside_selected_ranges"] += 1
                        continue
                    if not line.strip():
                        excluded["blank_in_selected_ranges"] += 1
                        continue
                    raw = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
                    identity, counter, kind, native_roles, _ = native_record(raw)
                    require(
                        identity in identities and all(value == role for value in native_roles),
                        "selected native row conflicts with the capture report identity/role",
                    )
                    if online:
                        observed = raw["event"]["observed_at_unix_ms"]
                        require(
                            observed > offline["latest_observed"],
                            "online observations must be strictly later than offline train/test rows; "
                            "boundary ties fail",
                        )
                        require(observed >= last_observed, "online observed timestamps go backwards in reviewed order")
                        require(
                            counter > offline["last_counters"].get(identity, -1),
                            "online counters must increase; reused rows, reordered segments and worker restarts fail",
                        )
                        last_observed = observed
                        offline["last_counters"][identity] = counter
                    if kind == "idle":
                        excluded["idle_in_selected_ranges"] += 1
                        continue
                    if online and role != "aggregated" and kind != role:
                        excluded["role_incompatible_in_selected_ranges"] += 1
                        continue
                    counts[kind] += 1
                    part = "online" if online else "train" if counts[kind] % 2 else "test"
                    require(line.endswith(b"\n"), "expected complete native JSONL lines")
                    outputs[part].write(line)
                    totals[f"{part}:{kind}"] += 1
            require(number >= last, "selected range extends beyond the source")
            require(file_hash(path) == digest, "source changed during selection")
        require(
            all(output.tell() > 0 for output in outputs.values()),
            "empty role-compatible online feedback" if online else "empty training or test split",
        )
        require(
            file_hash(selection_path) == selection_hash and file_hash(report_path) == report_hash,
            "selection or capture report changed during staging",
        )
        if online:
            require(
                all(file_hash(path) == digest for path, digest in offline["hashes"].items())
                and all(file_stamp(path) == stamp for path, stamp in offline["stamps"].items()),
                "offline inputs changed during online staging",
            )

    receipt = {
        "selection_sha256": selection_hash,
        "capture_report_sha256": report_hash,
        "role": role,
        "split": "chronological_feedback" if online else "alternating_per_work_kind",
        "counts": dict(totals),
        "excluded": dict(excluded),
        "outputs": {
            part: {"path": str(path.relative_to(run)), "sha256": file_hash(Path(str(path) + ".partial"))}
            for part, path in targets.items()
        },
    }
    if online:
        receipt["offline_selection_receipt_sha256"] = offline["receipt_hash"]
        receipt["offline_latest_observed_at_unix_ms"] = offline["latest_observed"]
    receipt_partial = Path(str(receipt_path) + ".partial")
    with receipt_partial.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write("\n")
    # Exclusive hard-link publication never replaces a concurrently created file.
    # Publish the receipt last; its presence marks a completed staging operation.
    for path in (*targets.values(), receipt_path):
        partial = Path(str(path) + ".partial")
        os.link(partial, path)
        partial.unlink()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="run directory containing selection and capture report"
    )
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument(
        "--mode",
        choices=("train-test", "online"),
        default="train-test",
        help="offline split or later single-stream feedback",
    )
    args = parser.parse_args(argv)
    try:
        receipt = prepare(args.run_dir, args.role, args.mode)
    except (OSError, EOFError, ValueError, KeyError, TypeError, OverflowError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
