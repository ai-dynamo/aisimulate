#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Materialize the reviewed native SGLang correction into a fresh CPU-only artifact.

Original orchestration; no upstream module is embedded. The separately attributed
patch, immutable upstream source and license are identified in provenance.json.
Outputs are read-only and suitable for explicit runtime bind mounts. This checks
bytes and provenance only; it does not qualify GPU execution or collected labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
REVISION = "e087e662ba1ac4ef7747537e2a9141085efd4561"
UPSTREAM_PATH = "python/sglang/srt/managers/scheduler_components/metrics_reporter.py"
RUNTIME_PATH = "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/metrics_reporter.py"
PATCH_NAME = "sglang-native-five-fields.patch"
RECEIPT_NAME = "materialization.json"
SOURCE_LIMIT = 2 * 1024 * 1024


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def contract(directory: Path = ROOT) -> dict:
    path = directory / "provenance.json"
    meta = json.loads(path.read_text())
    if (
        meta.get("schema") != "sglang-native-five-fields-patch-v1"
        or meta.get("upstream_repository") != "https://github.com/sgl-project/sglang"
        or meta.get("upstream_revision") != REVISION
        or meta.get("upstream_path") != UPSTREAM_PATH
        or meta.get("license") != "Apache-2.0"
        or meta.get("modified") is not True
    ):
        raise ValueError("Provenance differs from the selected native correction identity")
    for name in (
        "upstream_file_sha256",
        "patched_file_sha256",
        "patch_sha256",
        "license_sha256",
    ):
        if not isinstance(meta.get(name), str) or not re.fullmatch("[0-9a-f]{64}", meta[name]):
            raise ValueError(f"Invalid provenance hash: {name}")
    license_path = directory / meta["license_path"]
    if not license_path.resolve(strict=True).is_relative_to(directory.resolve(strict=True)):
        raise ValueError("License path escapes patch source")
    if sha256(license_path) != meta["license_sha256"] or sha256(directory / PATCH_NAME) != meta["patch_sha256"]:
        raise ValueError("Patch or license hash differs from reviewed provenance")
    return meta


def apply_exact_patch(original: bytes, patch_bytes: bytes, relative_path: str = UPSTREAM_PATH) -> bytes:
    """Apply one unified diff at its stated line numbers, rejecting offsets/fuzz."""
    lines = patch_bytes.splitlines(keepends=True)
    header = next((i for i, line in enumerate(lines) if line.startswith(b"--- ")), None)
    if header is None or lines[header] != f"--- a/{relative_path}\n".encode():
        raise ValueError("Unexpected patch old-file header")
    if header + 1 >= len(lines) or lines[header + 1] != f"+++ b/{relative_path}\n".encode():
        raise ValueError("Unexpected patch new-file header")
    base = original.splitlines(keepends=True)
    output = []
    old_cursor = 0
    position = header + 2
    hunks = 0
    while position < len(lines):
        match = re.fullmatch(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@[^\n]*\n", lines[position])
        if match is None:
            raise ValueError("Unsupported unified patch hunk/header")
        old_start, old_count, new_start, new_count = [int(v) if v is not None else 1 for v in match.groups()]
        old_index = old_start - 1 if old_count else old_start
        new_index = new_start - 1 if new_count else new_start
        if old_index < old_cursor or old_index > len(base):
            raise ValueError("Patch hunk location is outside the unmodified base")
        output.extend(base[old_cursor:old_index])
        if len(output) != new_index:
            raise ValueError("Patch new-file line position differs; offsets are forbidden")
        cursor = old_index
        consumed = produced = 0
        position += 1
        while position < len(lines) and not lines[position].startswith(b"@@ "):
            line = lines[position]
            operation, payload = line[:1], line[1:]
            if operation not in (b" ", b"-", b"+"):
                raise ValueError("Unsupported patch body; no fuzz or secondary files allowed")
            if operation in (b" ", b"-"):
                if cursor >= len(base) or base[cursor] != payload:
                    raise ValueError("Patch context does not exactly match the base at its stated line")
                cursor += 1
                consumed += 1
            if operation in (b" ", b"+"):
                output.append(payload)
                produced += 1
            position += 1
        if (consumed, produced) != (old_count, new_count):
            raise ValueError("Patch hunk lengths differ from declared lengths")
        old_cursor = cursor
        hunks += 1
    if not hunks:
        raise ValueError("Patch has no hunks")
    output.extend(base[old_cursor:])
    return b"".join(output)


def materialize(output: Path, upstream_file: Path | None = None, patch_source: Path = ROOT) -> dict:
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Materialization output must be a fresh directory")
    meta = contract(patch_source)
    url = f"https://raw.githubusercontent.com/sgl-project/sglang/{REVISION}/{UPSTREAM_PATH}"
    if upstream_file is not None:
        original = upstream_file.read_bytes()
        acquisition = {
            "method": "supplied_file",
            "path": str(upstream_file.resolve()),
            "immutable_source_url": url,
        }
    else:
        with urlopen(url, timeout=60) as response:
            original = response.read(SOURCE_LIMIT + 1)
        acquisition = {"method": "immutable_public_source", "url": url}
    if len(original) > SOURCE_LIMIT or digest_bytes(original) != meta["upstream_file_sha256"]:
        raise ValueError("Upstream source size/hash differs from the exact reviewed base")
    patch_bytes = (patch_source / PATCH_NAME).read_bytes()
    patched = apply_exact_patch(original, patch_bytes)
    if digest_bytes(patched) != meta["patched_file_sha256"]:
        raise ValueError("Patched full-file hash differs from reviewed provenance")
    output.mkdir(parents=True, exist_ok=False)
    files = {
        "metrics_reporter.py": patched,
        PATCH_NAME: patch_bytes,
        "provenance.json": (patch_source / "provenance.json").read_bytes(),
        meta["license_path"]: (patch_source / meta["license_path"]).read_bytes(),
    }
    inventory = []
    for name, contents in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        inventory.append(
            {
                "path": name,
                "size_bytes": len(contents),
                "sha256": digest_bytes(contents),
            }
        )
    receipt = {
        "schema": "sglang-native-five-fields-materialization-v1",
        "status": "materialized_bytes_verified",
        "created_at": datetime.now(UTC).isoformat(),
        "output_directory": str(output.resolve()),
        "runtime_path": RUNTIME_PATH,
        "upstream_acquisition": acquisition,
        "upstream_revision": REVISION,
        "upstream_file_sha256": meta["upstream_file_sha256"],
        "patched_file_sha256": meta["patched_file_sha256"],
        "patch_sha256": meta["patch_sha256"],
        "provenance_sha256": sha256(patch_source / "provenance.json"),
        "license_sha256": meta["license_sha256"],
        "materializer_sha256": sha256(Path(__file__)),
        "files": inventory,
        "patch_application": "Exact declared line positions and every context/deletion byte; no offsets or fuzz",
        "modified": True,
        "serving_qualified": False,
        "label_semantics_gpu_qualified": False,
        "capture_identity_policy": "Corrected native SGLang capture; stock captures retain their original identity",
    }
    (output / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    for file in output.rglob("*"):
        if file.is_file():
            file.chmod(0o444)
    for directory in sorted(
        (p for p in output.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    output.chmod(0o555)
    verify_materialization(output, patch_source)
    return receipt


def verify_materialization(directory: Path, patch_source: Path = ROOT, mounted_file: Path | None = None) -> dict:
    meta = contract(patch_source)
    root = directory.resolve(strict=True)
    receipt_path = root / RECEIPT_NAME
    if receipt_path.is_symlink():
        raise ValueError("Materialization receipt must be a regular file")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema"),
        receipt.get("status"),
        receipt.get("runtime_path"),
        receipt.get("upstream_revision"),
    ) != (
        "sglang-native-five-fields-materialization-v1",
        "materialized_bytes_verified",
        RUNTIME_PATH,
        REVISION,
    ):
        raise ValueError("Materialization receipt identity differs")
    expected = {
        "metrics_reporter.py": meta["patched_file_sha256"],
        PATCH_NAME: meta["patch_sha256"],
        "provenance.json": sha256(patch_source / "provenance.json"),
        meta["license_path"]: meta["license_sha256"],
    }
    rows = receipt.get("files", [])
    if len(rows) != len(expected) or {row.get("path") for row in rows} != set(expected):
        raise ValueError("Materialization file inventory differs")
    for row in rows:
        name = row["path"]
        file = root / name
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
            raise ValueError(f"Invalid materialized file: {name}")
        if (
            sha256(file) != expected[name]
            or row.get("sha256") != expected[name]
            or row.get("size_bytes") != file.stat().st_size
        ):
            raise ValueError(f"Materialized file hash/size mismatch: {name}")
        if file.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(f"Materialized file is not read-only: {name}")
    for key in (
        "upstream_file_sha256",
        "patched_file_sha256",
        "patch_sha256",
        "license_sha256",
    ):
        if receipt.get(key) != meta[key]:
            raise ValueError(f"Materialization receipt hash differs: {key}")
    if receipt.get("provenance_sha256") != expected["provenance.json"]:
        raise ValueError("Materialization provenance receipt differs")
    result = {
        "status": "materialized_bytes_verified",
        "directory": str(root),
        "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "provenance_sha256": expected["provenance.json"],
        "patched_file_sha256": expected["metrics_reporter.py"],
        "runtime_path": RUNTIME_PATH,
        "patch_sha256": meta["patch_sha256"],
        "upstream_revision": REVISION,
        "serving_qualified": False,
        "label_semantics_gpu_qualified": False,
    }
    if mounted_file is not None:
        if mounted_file.resolve() != Path(RUNTIME_PATH) or sha256(mounted_file) != expected["metrics_reporter.py"]:
            raise ValueError("Mounted runtime file path/hash differs from selected correction")
        result["mounted_file_sha256"] = sha256(mounted_file)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--output",
        type=Path,
        help="Fresh run-private directory for the corrected module and provenance",
    )
    mode.add_argument(
        "--verify-dir",
        type=Path,
        help="Verify an existing materialization; no downloads or writes",
    )
    parser.add_argument(
        "--upstream-file",
        type=Path,
        help="Use exact upstream bytes already available locally",
    )
    args = parser.parse_args()
    if args.verify_dir:
        if args.upstream_file:
            parser.error("--upstream-file applies only with --output")
        result = verify_materialization(args.verify_dir)
    else:
        result = materialize(args.output, args.upstream_file)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
