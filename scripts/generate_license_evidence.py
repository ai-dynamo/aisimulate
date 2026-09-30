#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine dependency inventories into the CSV evidence used by OSRB."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path

FIELDS = ["dependency_type", "name", "version", "spdx_license"]
DIFF_FIELDS = ["change", *FIELDS, "prior_version", "prior_spdx_license"]


def identity(row: dict[str, str]) -> tuple[str, str, str]:
    return row["dependency_type"], row["name"], row["version"]


def inventory(directory: Path) -> list[dict[str, str]]:
    rows = []
    for filename in sorted((directory / "python").glob("*.csv")):
        with filename.open(newline="") as stream:
            for row in csv.DictReader(stream):
                rows.append(
                    {
                        "dependency_type": "python",
                        "name": row["Name"],
                        "version": row["Version"],
                        "spdx_license": row["License"],
                    }
                )
    if not rows:
        raise ValueError("missing Python dependency inventories")
    with (directory / "cargo-metadata.json").open() as stream:
        metadata = json.load(stream)
    workspace = set(metadata.get("workspace_members") or [])
    for package in metadata["packages"]:
        if package.get("id") not in workspace:
            rows.append(
                {
                    "dependency_type": "crate",
                    "name": package["name"],
                    "version": package["version"],
                    "spdx_license": package.get("license") or "UNKNOWN",
                }
            )
    unique = {}
    for row in rows:
        key = identity(row)
        if key in unique and unique[key] != row:
            raise ValueError("conflicting license metadata for one dependency version")
        unique[key] = row
    return [unique[key] for key in sorted(unique)]


def previous_nightly() -> list[dict[str, str]]:
    """Keep nightlies compared with scheduled main runs, never manual releases."""
    token = os.environ["GH_API_TOKEN"]
    repository = os.environ["GITHUB_REPOSITORY"]
    current = os.environ["GITHUB_RUN_ID"]

    def get(url: str):
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
        )
        return urllib.request.urlopen(request, timeout=30)

    with get(
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        "nightly-ci.yml/runs?branch=main&event=schedule&status=success&per_page=10"
    ) as response:
        runs = json.load(response)
    for run in runs.get("workflow_runs", []):
        if str(run["id"]) == current:
            continue
        with get(run["artifacts_url"]) as response:
            artifacts = json.load(response)
        for artifact in artifacts.get("artifacts", []):
            if artifact["name"] != "license-artifacts" or artifact.get("expired"):
                continue
            with get(artifact["archive_download_url"]) as response:
                data = response.read()
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for name in archive.namelist():
                    if Path(name).name == "deps.csv":
                        with io.TextIOWrapper(archive.open(name)) as stream:
                            return list(csv.DictReader(stream))
    return []


def difference(rows: list[dict[str, str]], prior: list[dict[str, str]]) -> list[dict[str, str]]:
    current = {identity(row): row for row in rows}
    baseline = {identity(row): row for row in prior}
    result = []
    for key in sorted(current.keys() | baseline.keys()):
        new, old = current.get(key), baseline.get(key)
        if new and not old:
            result.append({"change": "added", **new, "prior_version": "", "prior_spdx_license": ""})
        elif old and not new:
            result.append(
                {
                    "change": "removed",
                    "dependency_type": old["dependency_type"],
                    "name": old["name"],
                    "version": "",
                    "spdx_license": "",
                    "prior_version": old["version"],
                    "prior_spdx_license": old["spdx_license"],
                }
            )
        elif new and old and new["spdx_license"] != old["spdx_license"]:
            result.append(
                {"change": "changed", **new, "prior_version": old["version"], "prior_spdx_license": old["spdx_license"]}
            )
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing python/*.csv and cargo-metadata.json")
    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--base-dir", type=Path, help="Inventories collected from the exact comparison commit")
    baseline.add_argument("--previous-nightly", action="store_true")
    parser.add_argument("--source-sha", default="")
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--source-ref", default="")
    args = parser.parse_args(argv)
    rows = inventory(args.directory)
    prior = []
    baseline_status = "not_requested"
    if args.base_dir:
        # Exact-commit CI must fail if its baseline inputs are missing; it must
        # never present an all-added report as a successful comparison.
        prior = inventory(args.base_dir)
        baseline_status = "exact_commit"
    elif args.previous_nightly:
        baseline_status = "previous_scheduled_nightly"
        try:
            prior = previous_nightly()
        except Exception as error:
            print(f"::warning::prior-artifact lookup failed ({error}); deps-diff.csv lists every dependency as added")
            baseline_status = "lookup_failed"
        if not prior and baseline_status != "lookup_failed":
            baseline_status = "unavailable"
    changes = difference(rows, prior)
    write_csv(args.directory / "deps.csv", FIELDS, rows)
    write_csv(args.directory / "deps-diff.csv", DIFF_FIELDS, changes)
    (args.directory / "evidence.json").write_text(
        json.dumps(
            {
                "source_sha": args.source_sha,
                "base_sha": args.base_sha,
                "source_ref": args.source_ref,
                "baseline_status": baseline_status,
                "dependency_count": len(rows),
                "change_count": len(changes),
                "python_resolution": "runtime requirements resolved at inventory collection time",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"deps.csv: {len(rows)} rows; deps-diff.csv: {len(changes)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
