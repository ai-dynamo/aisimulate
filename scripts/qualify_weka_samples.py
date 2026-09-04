#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify AISimulate Weka ingestion against two public AgentX trace rows."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


DATASET = "semianalysisai/cc-traces-weka-062126-256k"
DATASET_REVISION = "8fecd2fc56694469f758f0afbbb6335ad3043740"
EXPECTED_IDS = (
    "002001296e8a8c38ad9d7cc436d691afc602",
    "006c98de37d819e95b0840e25426bb7ca99d",
)
EXPECTED_SOURCE_DIGEST = (
    "fc88c786e1dab775b550f9387d9c53358123df78b2af7cb344235bcfafe33ea6"
)
EXPECTED_GRAPH_DIGEST = (
    "363718b2a1325136638d0be05a358551e982f8e767fecdeb66e10ac979e55d58"
)


def model_request_count(trace: dict[str, Any]) -> int:
    total = 0
    for entry in trace["requests"]:
        if entry["type"] == "subagent":
            total += len(entry["requests"])
        else:
            total += 1
    return total


def fetch_two_rows() -> list[dict[str, Any]]:
    metadata = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--max-time",
            "120",
            f"https://huggingface.co/api/datasets/{DATASET}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = json.loads(metadata.stdout).get("sha")
    if revision != DATASET_REVISION:
        raise RuntimeError(
            f"dataset revision changed: expected {DATASET_REVISION}, found {revision}"
        )

    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset={DATASET.replace('/', '%2F')}&config=default&split=train&offset=0&length=2"
    )
    completed = subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--max-time",
            "120",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    rows = [item["row"] for item in payload["rows"]]
    ids = tuple(row["id"] for row in rows)
    if ids != EXPECTED_IDS:
        raise RuntimeError(
            f"dataset sample changed: expected {EXPECTED_IDS}, found {ids}"
        )
    return rows


def qualify(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="aisimulate-weka-samples-") as directory:
        path = Path(directory) / "traces.jsonl"
        with path.open("w", encoding="utf-8") as destination:
            for row in rows:
                json.dump(row, destination, separators=(",", ":"))
                destination.write("\n")
        completed = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "--locked",
                "-p",
                "aisimulate-core",
                "--example",
                "qualify_weka",
                "--",
                str(path),
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    result = json.loads(completed.stdout)
    expected = {
        "block_size": 64,
        "files": 1,
        "plays": 2,
        "requests": sum(model_request_count(row) for row in rows),
        "source_digest": EXPECTED_SOURCE_DIGEST,
        "graph_digest": EXPECTED_GRAPH_DIGEST,
    }
    mismatches = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Weka sample qualification mismatch: {mismatches}")
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report = qualify(root, fetch_two_rows())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
