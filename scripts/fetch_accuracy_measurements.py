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
import json
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path

TABLES = {"configs", "benchmark_results", "workflow_runs"}
RELEASE_ROOT = "https://github.com/SemiAnalysisAI/InferenceX-app/releases/download/"


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


def fetch(manifest: dict, output: Path) -> Path:
    tag = manifest["release_tag"]
    if not re.fullmatch(r"db-dump/\d{4}-\d{2}-\d{2}", tag):
        raise ValueError("invalid pinned measurement release")
    output.mkdir(parents=True, exist_ok=True)
    compressed = output / "measurements.dump.zst"
    # Avoid filling small runner disks. The pinned release is ~25 GB compressed.
    if shutil.disk_usage(output).free < manifest["minimum_free_bytes"]:
        raise ValueError("insufficient disk for the pinned measurement dump")
    with compressed.open("wb") as target:
        for index, part in enumerate(manifest["parts"]):
            expected_name = f"inferencex-{tag.split('/')[1]}.dump.zst.part{index:02d}"
            if part["name"] != expected_name or not re.fullmatch(r"[0-9a-f]{64}", part["sha256"]):
                raise ValueError("invalid pinned dump part")
            sha = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(RELEASE_ROOT + tag + "/" + part["name"], timeout=120) as response:
                while block := response.read(1024 * 1024):
                    sha.update(block)
                    size += len(block)
                    if size > part["size"]:
                        raise ValueError("dump part exceeds pinned size")
                    target.write(block)
            if size != part["size"] or sha.hexdigest() != part["sha256"]:
                raise ValueError("measurement part checksum or size mismatch")
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
    args = parser.parse_args()
    fetch(json.loads(args.manifest.read_text()), args.output)
