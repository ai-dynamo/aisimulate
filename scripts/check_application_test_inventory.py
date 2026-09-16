#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Account for actual collected application cases, including explicit exceptions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "python/aisimulate"
EXCEPTIONS = ROOT / ".github/application-test-inventory.json"


def assignment(path: str, markers: set[str], manual_suites: dict[str, str]) -> tuple[str, str]:
    """Map the collected case to the selectors declared in Full CI."""
    for prefix, shard in (
        ("tests/unit/", "unit"),
        ("tests/golden/", "unit"),
        ("tests/integration/", "integration"),
        ("tests/cross_package/", "contracts"),
    ):
        if path.startswith(prefix):
            return shard, "entire directory"
    for prefix, shard in (
        ("tests/e2e/cli/", "cli-build"),
        ("tests/e2e/support_matrix/", "support-matrix"),
        ("tests/e2e/tools/", "tools-build"),
    ):
        if path.startswith(prefix) and "build" in markers:
            return shard, "build marker"
    if reason := manual_suites.get(path):
        return "manual", reason
    raise ValueError(f"unassigned collected test: {path}; markers={sorted(markers)}")


class Inventory:
    def __init__(self, exceptions: dict[str, dict[str, str]]) -> None:
        self.exceptions = exceptions
        self.cases: list[dict[str, object]] = []
        self.collection_skips: list[dict[str, str]] = []
        self.errors: list[str] = []

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.errors.append(f"collection failed: {report.nodeid}: {report.longrepr}")
        elif report.skipped:
            reason = str(report.longrepr)
            path = report.nodeid.split("::", 1)[0]
            expected = self.exceptions["optional_collection_skips"].get(path)
            if not expected or expected not in reason:
                self.errors.append(f"unexpected collection skip: {path}: {reason}")
            self.collection_skips.append({"path": path, "reason": reason})

    def pytest_collection_finish(self, session) -> None:
        for item in session.items:
            path = item.path.relative_to(APP).as_posix()
            markers = {marker.name for marker in item.iter_markers()}
            try:
                shard, reason = assignment(path, markers, self.exceptions["manual_suites"])
            except ValueError as exc:
                self.errors.append(f"{item.nodeid}: {exc}")
                shard, reason = "unassigned", str(exc)
            self.cases.append({"nodeid": item.nodeid, "path": path, "shard": shard, "reason": reason})


def main() -> int:
    import pytest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exceptions = json.loads(EXCEPTIONS.read_text(encoding="utf-8"))
    for entries in exceptions.values():
        for path, reason in entries.items():
            if not (APP / path).is_file() or not reason.strip():
                parser.error(f"stale or unexplained inventory exception: {path}")
    inventory = Inventory(exceptions)
    result = pytest.main(
        [
            "--collect-only",
            "-c",
            str(APP / "pytest.ini"),
            str(APP / "tests"),
            "-o",
            "addopts=",
            "-o",
            "timeout=0",
            "-p",
            "no:terminal",
            "-p",
            "no:cacheprovider",
        ],
        plugins=[inventory],
    )
    if not inventory.cases:
        inventory.errors.append("no application cases collected")
    counts = dict(sorted(Counter(str(case["shard"]) for case in inventory.cases).items()))
    args.output.write_text(
        json.dumps(
            {
                "counts": counts,
                "cases": inventory.cases,
                "collection_skips": inventory.collection_skips,
                "errors": inventory.errors,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"counts": counts, "collection_skips": inventory.collection_skips, "errors": inventory.errors}, indent=2
        )
    )
    return int(result != pytest.ExitCode.OK or bool(inventory.errors))


if __name__ == "__main__":
    raise SystemExit(main())
