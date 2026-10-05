#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract data from a checksum-pinned public InferenceX PostgreSQL dump.

pg_restore writes COPY text; no SQL from the downloaded dump is executed.
Raw measurements stay in the runner's temporary directory.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

TABLES = {"configs", "benchmark_results", "workflow_runs"}
RELEASE_ROOT = "https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/"
POLICY = "latest-complete-config-run-v1"


def validate_manifest(manifest: dict) -> None:
    fields = {"schema_version", "release_tag", "selection_policy", "max_age_days", "minimum_free_bytes", "parts"}
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise ValueError("invalid measurement manifest fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ValueError("unsupported measurement manifest schema_version")
    tag = manifest["release_tag"]
    if not isinstance(tag, str) or not re.fullmatch(r"db-dump/\d{4}-\d{2}-\d{2}", tag):
        raise ValueError("invalid pinned measurement release")
    if manifest["selection_policy"] != POLICY:
        raise ValueError("unknown cohort selection policy")
    for field, minimum in (("max_age_days", 0), ("minimum_free_bytes", 1)):
        if type(manifest[field]) is not int or manifest[field] < minimum:
            raise ValueError(f"invalid measurement manifest {field}")
    parts = manifest["parts"]
    if not isinstance(parts, list) or not parts:
        raise ValueError("measurement manifest requires dump parts")
    for index, part in enumerate(parts):
        if not isinstance(part, dict) or set(part) != {"name", "sha256", "size"}:
            raise ValueError("invalid pinned dump part fields")
        if (
            part["name"] != f"inferencex-{tag.split('/')[1]}.dump.zst.part{index:02d}"
            or not isinstance(part["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", part["sha256"])
            or type(part["size"]) is not int
            or part["size"] <= 0
        ):
            raise ValueError("invalid pinned dump part")


def release_items(path: str) -> list[dict]:
    """Read every page of release metadata from the fixed upstream repository."""
    items = []
    for page in range(1, 101):
        headers = {"Accept": "application/vnd.github+json", "User-Agent": "aisimulate-accuracy"}
        if token := os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(
            f"https://api.github.com/repos/SemiAnalysisAI/InferenceX-app/{path}?per_page=100&page={page}",
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            batch = json.load(response)
        if not isinstance(batch, list):
            raise ValueError("invalid upstream release metadata")
        items.extend(batch)
        if len(batch) < 100:
            return items
    raise ValueError("upstream release pagination limit exceeded")


def resolve_latest_manifest(template: dict) -> dict:
    """Freeze the newest dated dump and its published checksums for one run."""
    validate_manifest(template)
    releases = [
        release
        for release in release_items("releases")
        if not release.get("draft")
        and not release.get("prerelease")
        and re.fullmatch(r"db-dump/\d{4}-\d{2}-\d{2}", release.get("tag_name", ""))
    ]
    if not releases:
        raise ValueError("no published database dump release")
    release = max(releases, key=lambda item: item["tag_name"])
    tag = release["tag_name"]
    assets = release_items(f"releases/{int(release['id'])}/assets")
    by_name = {asset["name"]: asset for asset in assets}
    if len(by_name) != len(assets):
        raise ValueError("duplicate release assets")
    checksum_asset = by_name.get("SHA256SUMS", {})
    if checksum_asset.get("state") != "uploaded":
        raise ValueError("latest dump has no uploaded SHA256SUMS")
    with urllib.request.urlopen(RELEASE_ROOT + tag + "/SHA256SUMS", timeout=120) as response:
        checksums = response.read(1024 * 1024 + 1)
    if (
        len(checksums) > 1024 * 1024
        or len(checksums) != checksum_asset.get("size")
        or "sha256:" + hashlib.sha256(checksums).hexdigest() != checksum_asset.get("digest")
    ):
        raise ValueError("release checksum file digest or size mismatch")
    parts = []
    for line in checksums.decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](\S+)", line)
        if not match:
            raise ValueError("invalid release checksum entry")
        checksum, name = match.groups()
        asset = by_name.get(name, {})
        if asset.get("state") != "uploaded" or asset.get("digest") != "sha256:" + checksum:
            raise ValueError("dump asset missing or checksum mismatch")
        parts.append({"name": name, "sha256": checksum, "size": asset["size"]})
    parts.sort(key=lambda part: part["name"])
    if {part["name"] for part in parts} != {name for name in by_name if ".dump.zst.part" in name}:
        raise ValueError("dump assets and checksum entries differ")
    manifest = {**template, "release_tag": tag, "parts": parts}
    validate_manifest(manifest)
    # Leave room for extracted COPY data and tables.json in addition to the dump.
    manifest["minimum_free_bytes"] = max(template["minimum_free_bytes"], sum(p["size"] for p in parts) + 10_000_000_000)
    return manifest


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def copy_value(value: str):
    if value == r"\N":
        return None
    escapes = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}

    def unescape(match):
        text = match[1]
        if text.startswith("x"):
            return chr(int(text[1:], 16))
        if text[0].isdigit():
            return chr(int(text, 8))
        return escapes.get(text, text)

    return re.sub(r"\\(x[0-9a-fA-F]{1,2}|[0-7]{1,3}|.)", unescape, value)


def copy_columns(value: str) -> list[str]:
    # pg_restore quotes reserved identifiers such as "precision". Accept the
    # pinned schema's simple names and reject unfamiliar identifier syntax.
    columns = []
    for name in value.split(", "):
        match = re.fullmatch(r'(?:"([a-z_][a-z0-9_]*)"|([a-z_][a-z0-9_]*))', name)
        if not match:
            raise ValueError("unsupported COPY column identifier")
        columns.append(match[1] or match[2])
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate COPY column")
    return columns


def read_copy(stream):
    """Read only the three allowlisted tables, rejecting incomplete COPY data."""
    tables = {}
    current = None
    columns = []
    for line in stream:
        if current is None:
            match = re.fullmatch(r"COPY public\.([a-z_]+) \(([^)]+)\) FROM stdin;\n?", line)
            if not match:
                continue
            current = match[1]
            if current not in TABLES or current in tables:
                raise ValueError("unexpected or duplicate COPY table")
            columns = copy_columns(match[2])
            tables[current] = []
        elif line.rstrip("\n") == r"\.":
            current = None
        else:
            values = line.rstrip("\n").split("\t")
            if len(values) != len(columns):
                raise ValueError("COPY row width does not match columns")
            row = dict(zip(columns, map(copy_value, values), strict=True))
            # Use explicit schema conversions; numeric-looking text stays text.
            for name in (
                "id",
                "workflow_run_id",
                "config_id",
                "isl",
                "osl",
                "conc",
                "run_attempt",
                "prefill_tp",
                "prefill_ep",
                "prefill_num_workers",
                "decode_tp",
                "decode_ep",
                "decode_num_workers",
                "num_prefill_gpu",
                "num_decode_gpu",
            ):
                if row.get(name) is not None:
                    row[name] = int(row[name])
            for name in ("disagg", "is_multinode", "prefill_dp_attention", "decode_dp_attention"):
                if name in row:
                    if row[name] not in ("t", "f"):
                        raise ValueError("invalid COPY boolean")
                    row[name] = row[name] == "t"
            if "metrics" in row:
                row["metrics"] = json.loads(row["metrics"])
            tables[current].append(row)
    if current is not None or set(tables) != TABLES or any(not rows for rows in tables.values()):
        raise ValueError("incomplete measurement tables")
    return tables


def download_part(url: str, part: dict, target) -> None:
    """Retry a failed part without retaining corrupt bytes or redownloading prior parts."""
    offset = target.tell()
    for attempt in range(1, 4):
        target.seek(offset)
        target.truncate()
        sha = hashlib.sha256()
        size = 0
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                while block := response.read(1024 * 1024):
                    sha.update(block)
                    size += len(block)
                    if size > part["size"]:
                        raise ValueError("dump part exceeds pinned size")
                    target.write(block)
            if size != part["size"] or sha.hexdigest() != part["sha256"]:
                raise ValueError(
                    f"measurement part checksum or size mismatch: {part['name']}; "
                    f"received {size}/{part['size']} bytes, "
                    f"SHA256 {sha.hexdigest()}, expected {part['sha256']}"
                )
            return
        except (OSError, http.client.HTTPException, ValueError) as error:
            target.seek(offset)
            target.truncate()
            if attempt == 3:
                raise
            print(f"Retrying {part['name']} after attempt {attempt}/3: {error}", flush=True)
            time.sleep(5 * attempt)


def fetch(manifest: dict, output: Path) -> Path:
    validate_manifest(manifest)
    tag = manifest["release_tag"]
    output.mkdir(parents=True, exist_ok=True)
    compressed = output / "measurements.dump.zst"
    # Avoid filling small runner disks. The pinned release is ~25 GB compressed.
    if shutil.disk_usage(output).free < manifest["minimum_free_bytes"]:
        raise ValueError("insufficient disk for the pinned measurement dump")
    with compressed.open("wb") as target:
        for index, part in enumerate(manifest["parts"]):
            download_part(RELEASE_ROOT + tag + "/" + part["name"], part, target)
            print(f"Verified measurement part {index + 1}/{len(manifest['parts'])}", flush=True)
    sql = output / "measurements.copy"
    # Serial pg_restore reads custom-format archives from stdin. Stream past
    # large server-log tables without storing an expanded dump on disk.
    with subprocess.Popen(["zstd", "-dc", str(compressed)], stdout=subprocess.PIPE) as decompressor:
        try:
            subprocess.run(
                [
                    "pg_restore",
                    "--data-only",
                    "--no-owner",
                    "--no-privileges",
                    *[flag for table in sorted(TABLES) for flag in ("--table", table)],
                    "--file",
                    str(sql),
                ],
                stdin=decompressor.stdout,
                check=True,
            )
            # pg_restore may finish after its last selected table. Drain the
            # remaining stream so zstd validates the frame and cannot SIGPIPE.
            while decompressor.stdout.read(1024 * 1024):
                pass
        finally:
            decompressor.stdout.close()
        if decompressor.wait() != 0:
            raise ValueError("measurement decompression failed")
    compressed.unlink()
    with sql.open() as stream:
        tables = read_copy(stream)
    destination = output / "tables.json"
    destination.write_text(json.dumps(tables, sort_keys=True, allow_nan=False) + "\n")
    sql.unlink()
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resolve-latest", action="store_true", help="Write a resolved manifest without downloading the dump"
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.resolve_latest:
        resolved = resolve_latest_manifest(manifest)
        args.output.write_text(json.dumps(resolved, sort_keys=True) + "\n")
        print(f"Resolved {resolved['release_tag']} with {len(resolved['parts'])} verified part checksums")
    else:
        fetch(manifest, args.output)
