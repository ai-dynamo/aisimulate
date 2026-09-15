#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the packaged-power invariant gate against this checkout."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("python/aisimulate/src/aiconfigurator_core/systems/data")
COMMAND = (
    "python scripts/power_qualification_data.py --expected-revision <candidate-sha>"
)


def checkout_revision(root: Path, expected: str) -> str:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != root.resolve():
        raise ValueError("evidence root must be the checked-out repository")
    revision = git("rev-parse", "HEAD")
    if revision != expected:
        raise ValueError("evidence revision does not match checked-out HEAD")
    if git("diff", "HEAD", "--name-only") or git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "scripts",
        "tests",
        "python",
        "crates",
    ):
        raise ValueError("evidence requires a clean source checkout")
    return revision


def scan_details(root: Path) -> dict:
    # Resolve from the repository, never from an importable installed package.
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_root = root / DATA_ROOT
    if not data_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("data root escapes checkout")
    paths = sorted(data_root.rglob("*.parquet"))
    if not paths:
        raise ValueError("no parquet files discovered in the checked-out data tree")
    files = []
    missing = []
    anomalies = []
    digest = hashlib.sha256()

    def counts(column: str, values, sentinel) -> dict:
        invalid = "negative_count" if column == "power" else "non_positive_count"
        return {
            "unit": "W",
            "nan_count": int(values.isna().sum()),
            "positive_infinity_count": int((values == float("inf")).sum()),
            "negative_infinity_count": int((values == float("-inf")).sum()),
            invalid: int(
                (
                    (values < 0) if column == "power" else ((values <= 0) & ~sentinel)
                ).sum()
            ),
        }

    totals = {
        column: {
            "unit": "W",
            "nan_count": 0,
            "positive_infinity_count": 0,
            "negative_infinity_count": 0,
            "negative_count" if column == "power" else "non_positive_count": 0,
        }
        for column in ("power", "power_limit")
    }
    without_power = 0
    for path in paths:
        if not path.resolve().is_relative_to(data_root.resolve()):
            raise ValueError(f"data path escapes checkout: {path}")
        relative = path.relative_to(data_root).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(f"{relative}\0{file_digest}\n".encode())
        schema = pq.read_schema(path)
        present = [name for name in ("power", "power_limit") if name in schema.names]
        if not present:
            without_power += 1
            continue
        if len(present) != 2:
            missing.append(relative)
            continue
        if any(not pa.types.is_float64(schema.field(name).type) for name in present):
            anomalies.append(f"{relative}: power and power_limit must be double")
            continue
        frame = pq.read_table(path, columns=present).to_pandas()
        sentinel = (frame.power == 0) & (frame.power_limit == 0)
        mismatched = (frame.power == 0) != (frame.power_limit == 0)
        if mismatched.any():
            anomalies.append(
                f"{relative}: {int(mismatched.sum())} unpaired zero sentinels"
            )
        result = {
            "path": relative,
            "sha256": file_digest,
            "rows_scanned": len(frame),
            "power_columns_present": present,
            "pairing_result": "pass",
            "key_columns": [
                name
                for name in schema.names
                if name not in {"power", "power_limit", "latency", "latency_ms"}
            ],
            "unavailable_sentinel_rows": int(sentinel.sum()),
        }
        for column in present:
            result[column] = counts(column, frame[column], sentinel)
            for name, value in result[column].items():
                if name != "unit":
                    totals[column][name] += value
        files.append(result)
    if not files:
        anomalies.append("no paired power-carrying parquet files discovered")
    finite = all(
        value == 0
        for summary in totals.values()
        for name, value in summary.items()
        if name != "unit"
    )
    if not finite:
        anomalies.append("non-finite or invalid power values")
    if missing:
        anomalies.append("power and power_limit columns are not paired")
    return {
        "discovery_root": DATA_ROOT.as_posix(),
        "total_discovered_parquet_count": len(paths),
        "files_scanned": len(files),
        "files_without_power_columns": without_power,
        "skipped_file_anomalies": anomalies,
        "rows_scanned": sum(item["rows_scanned"] for item in files),
        "checks": {
            "data_tree_sha256": digest.hexdigest(),
            "finite_nonnegative_values": {
                "result": "pass" if finite else "fail",
                **totals,
            },
            "power_and_limit_columns_paired": {
                "result": "fail" if missing else "pass",
                "files_missing_pair": missing,
            },
        },
        "files": files,
    }


def build_report(root: Path, gate: dict, revision: str) -> dict:
    details = scan_details(root)
    anomalies = details["skipped_file_anomalies"]
    return {
        "schema_version": "1.0",
        "gate_id": "power-data-invariants",
        "source_revision": revision,
        "matrix": copy.deepcopy(gate["matrix"]),
        "assertion_results": [
            {**copy.deepcopy(assertion), "result": "fail" if anomalies else "pass"}
            for assertion in gate["assertions"]
        ],
        "units": {"power": "W", "power_limit": "W"},
        "anomalies": anomalies,
        "details": details,
    }


def verify_automated_evidence(root: Path, document: dict, expected: str) -> None:
    revision = checkout_revision(root, expected)
    for gate in document["gates"]:
        execution = gate["execution"]
        if execution["kind"] != "automated" or execution["status"] != "passed":
            continue
        if gate["id"] != "power-data-invariants":
            raise ValueError(
                f"gate {gate['id']} has no execution verifier; it cannot qualify a release"
            )
        if execution["command"] != COMMAND:
            raise ValueError(
                "power-data-invariants command does not name the evidence producer"
            )
        actual = build_report(root, gate, revision)
        if actual["anomalies"]:
            raise ValueError(f"checked-out power data failed: {actual['anomalies']}")
        for evidence in execution["evidence"]:
            artifact = (root / evidence["artifact"]).resolve()
            if not artifact.is_relative_to(root.resolve()):
                raise ValueError("evidence artifact escapes checkout")
            if (
                evidence["source_revision"] != revision
                or json.loads(artifact.read_text()) != actual
            ):
                raise ValueError(
                    "automated evidence does not reproduce against the checked-out tree"
                )


def output_directory(root: Path, requested: Path) -> Path:
    """Only create new evidence files in the dedicated artifacts subtree."""
    if requested.is_absolute() or ".." in requested.parts:
        raise ValueError("output must stay under artifacts/power-qualification")
    allowed = root / "artifacts/power-qualification"
    output = root / requested
    if not output.resolve().is_relative_to(allowed.resolve()):
        raise ValueError("output must stay under artifacts/power-qualification")
    for component in (output, *output.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError("output must not traverse symlinks")
    for name in ("power-data-invariants.json", "qualification-matrix.json"):
        target = output / name
        if target.exists() or target.is_symlink():
            raise ValueError(f"output already exists: {target}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/power-qualification")
    )
    args = parser.parse_args(argv)
    revision = checkout_revision(ROOT, args.expected_revision)
    output = output_directory(ROOT, args.output_dir)
    document = json.loads((ROOT / "docs/power/qualification-matrix.json").read_text())
    gate = next(
        item for item in document["gates"] if item["id"] == "power-data-invariants"
    )
    report = build_report(ROOT, gate, revision)
    # Recheck after scanning so a concurrent tracked source edit cannot be attested.
    checkout_revision(ROOT, revision)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "power-data-invariants.json"
    with artifact.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    gate["execution"].update(
        {
            "status": "failed" if report["anomalies"] else "passed",
            "command": COMMAND,
            "evidence": [
                {
                    "result": "fail" if report["anomalies"] else "pass",
                    "artifact": artifact.relative_to(ROOT).as_posix(),
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "source_revision": revision,
                    "recorded_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                }
            ],
        }
    )
    document["candidate_revision"] = revision
    with (output / "qualification-matrix.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")
    print(f"power-data-invariants: {gate['execution']['status']} at {revision}")
    return int(bool(report["anomalies"]))


if __name__ == "__main__":
    raise SystemExit(main())
