#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage reviewed native rows into an alternating within-workload train/test split.

Reads ROLE.selection.json and receipts/ROLE.capture.json under --run-dir. Each
selection entry supplies path (relative to the run), sha256, first_line and
last_line (inclusive, one range per file, in reviewed segment order). The capture
report must be valid for this role. The user selects the workload population;
this helper cannot identify smoke, warmup or drain from a busy record.

Writes selected/ROLE.{train,test}.jsonl and receipts/ROLE.selection.json. Busy
native lines remain byte-for-byte unchanged; idle rows are excluded. Training
gets the first, third, etc. observation of each work kind. Outputs must be fresh.
Failure retains .partial files; require a completed receipt before training.
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


def prepare(run, role):
    run = Path(run).resolve()
    require(role in ROLES, "unknown serving role")
    selection_path = run / f"{role}.selection.json"
    report_path = run / f"receipts/{role}.capture.json"
    selection, selection_hash = read_json(selection_path)
    report, report_hash = read_json(report_path)
    require(report["status"] == "valid", "resolve capture validation before selecting training rows")
    expected = report["expected_workers"]
    require(
        expected and all(worker["declared_role"] == role for worker in expected),
        "capture report must bind every expected worker to the selected role",
    )
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
        require(digest not in seen, "source repeated in selection; use one reviewed range per file")
        seen.add(digest)
        first, last = entry["first_line"], entry["last_line"]
        require(type(first) is int and type(last) is int and 1 <= first <= last, "invalid inclusive line range")
        sources.append((path, digest, first, last))

    targets = {part: run / f"selected/{role}.{part}.jsonl" for part in ("train", "test")}
    receipt_path = run / f"receipts/{role}.selection.json"
    for path in (*targets.values(), receipt_path):
        require(path.resolve().is_relative_to(run), "output must stay inside the run directory")
        require(not path.exists() and not Path(str(path) + ".partial").exists(), "choose fresh outputs and receipt")
    (run / "selected").mkdir(exist_ok=True)
    counts, totals, excluded = Counter(), Counter(), Counter()
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
                    identity, _, kind, native_roles, _ = native_record(raw)
                    require(
                        identity in identities and all(value == role for value in native_roles),
                        "selected native row conflicts with the capture report identity/role",
                    )
                    if kind == "idle":
                        excluded["idle_in_selected_ranges"] += 1
                        continue
                    counts[kind] += 1
                    part = "train" if counts[kind] % 2 else "test"
                    require(line.endswith(b"\n"), "expected complete native JSONL lines")
                    outputs[part].write(line)
                    totals[f"{part}:{kind}"] += 1
            require(number >= last, "selected range extends beyond the source")
            require(file_hash(path) == digest, "source changed during selection")
        require(all(output.tell() > 0 for output in outputs.values()), "empty training or test split")
        require(
            file_hash(selection_path) == selection_hash and file_hash(report_path) == report_hash,
            "selection or capture report changed during staging",
        )

    receipt = {
        "selection_sha256": selection_hash,
        "capture_report_sha256": report_hash,
        "role": role,
        "split": "alternating_per_work_kind",
        "counts": dict(totals),
        "excluded": dict(excluded),
        "outputs": {
            part: {"path": str(path.relative_to(run)), "sha256": file_hash(Path(str(path) + ".partial"))}
            for part, path in targets.items()
        },
    }
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
    args = parser.parse_args(argv)
    try:
        receipt = prepare(args.run_dir, args.role)
    except (OSError, EOFError, ValueError, KeyError, TypeError, OverflowError) as error:
        parser.error(str(error))
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
