#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve original host-cost workers in a verified deterministic archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import shutil
import tarfile
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw-backup", type=Path, required=True)
    args = parser.parse_args()
    root, backup = args.root.resolve(), args.raw_backup.resolve()
    if backup.exists():
        raise ValueError("The original-worker backup destination must be new")
    paths = sorted(
        path
        for path in root.glob("*/*")
        if path.is_file() and path.name != "summary.json"
    )
    if not paths or any(path.suffix not in {".json", ".log"} for path in paths):
        raise ValueError("Expected only raw worker JSON and logs before compaction")
    shutil.copytree(root, backup)
    inventory = []
    metrics = []
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            if (backup / relative).read_bytes() != data:
                raise ValueError(f"Backup changed bytes: {relative}")
            inventory.append(
                {"path": relative, "bytes": len(data), "sha256": digest(data)}
            )
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            member.mode = 0o644
            member.mtime = 0
            archive.addfile(member, io.BytesIO(data))
            if path.suffix == ".json":
                value = json.loads(data)
                identity = value.pop("identity")
                value["identity"] = {
                    key: identity[key]
                    for key in (
                        "checkout",
                        "revision",
                        "tracked_diff_sha256",
                        "code_sha256",
                        "harness_sha256",
                        "native_sha256",
                        "dependency_freeze_sha256",
                    )
                }
                # These exact metrics are retained; only repeated full source
                # and dependency inventories move into the original archive.
                value["raw_path"] = relative
                value["raw_sha256"] = digest(data)
                metrics.append(value)
    output = root / "raw-workers.tar.xz"
    output.write_bytes(lzma.compress(stream.getvalue(), preset=6))
    with tarfile.open(output, "r:xz") as archive:
        if sorted(archive.getnames()) != sorted(item["path"] for item in inventory):
            raise ValueError("Archive member set differs from original workers")
        for item in inventory:
            data = archive.extractfile(item["path"]).read()
            if len(data) != item["bytes"] or digest(data) != item["sha256"]:
                raise ValueError(f"Archive changed bytes: {item['path']}")
    write_json(root / "trial-metrics.json", metrics)
    write_json(
        root / "raw-archive.json",
        {
            "archive": output.name,
            "archive_sha256": digest(output.read_bytes()),
            "archive_bytes": output.stat().st_size,
            "original_private_backup": str(backup),
            "member_count": len(inventory),
            "members": inventory,
            "verification": "Every archive member and private backup is byte-identical to its original worker.",
            "summaries": "Unchanged summary.json files retain original worker paths and SHA-256 values.",
        },
    )
    for path in paths:
        path.unlink()
    print(
        json.dumps(
            {
                "members": len(inventory),
                "archive_bytes": output.stat().st_size,
                "backup": str(backup),
            }
        )
    )


if __name__ == "__main__":
    main()
