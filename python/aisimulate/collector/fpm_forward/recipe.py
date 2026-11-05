# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialize and explicitly run an immutable, separately audited FPM recipe.

Recipes contain executable collector source. Hash verification establishes its
identity, not its safety, GPU qualification, or measurement acceptance. This
entry point preserves that source instead of maintaining another implementation
of its native runtime, startup, sampling, or shutdown protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

SCHEMA = "aisimulate.fpm-source-recipe.v1"
MAX_FILES = 20_000
MAX_BYTES = 512 * 1024**2
MAX_MANIFEST_BYTES = 1024**2


def _relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("recipe member must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError("recipe member escapes or aliases the source directory")
    return value


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024**2):
            value.update(block)
    return value.hexdigest()


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def read_pin(path: Path) -> dict:
    pin = json.loads(path.read_text())
    return _validate_pin(pin)


def _validate_pin(pin: dict) -> dict:
    if not isinstance(pin, dict) or set(pin) != {"repo_id", "revision", "filename", "sha256"}:
        raise ValueError("recipe pin requires repo_id, revision, filename and sha256")
    if not isinstance(pin["repo_id"], str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", pin["repo_id"]):
        raise ValueError("recipe pin requires an explicit Hugging Face dataset repository")
    if not _hex(pin["revision"], 40) or not _hex(pin["sha256"], 64):
        raise ValueError("recipe requires immutable HF commit and archive SHA256")
    _relative(pin["filename"])
    return pin


def _manifest(value: dict) -> dict:
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported FPM source recipe schema")
    files = value.get("files")
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise ValueError("recipe requires a bounded, nonempty source manifest")
    total = 0
    for name, facts in files.items():
        _relative(name)
        if name == "recipe.json" or not isinstance(facts, dict) or set(facts) != {"sha256", "bytes"}:
            raise ValueError("invalid source manifest member")
        if not _hex(facts["sha256"], 64) or type(facts["bytes"]) is not int or facts["bytes"] < 0:
            raise ValueError("invalid source member size or digest")
        total += facts["bytes"]
    if total > MAX_BYTES:
        raise ValueError("recipe source exceeds the materialization limit")
    entry = value.get("entrypoint")
    if not isinstance(entry, dict) or set(entry) != {"path", "interpreter"}:
        raise ValueError("recipe requires one explicit entry point")
    if entry["interpreter"] not in ("python", "bash") or entry["path"] not in files:
        raise ValueError("entry point is not bound to the source manifest")
    notices = value.get("licenses")
    if not isinstance(notices, list) or not notices or any(name not in files for name in notices):
        raise ValueError("recipe must retain its applicable license and attribution files")
    if value.get("readme") not in files:
        raise ValueError("recipe must include its runtime and usage instructions")
    return value


def materialize(pin: dict, destination: Path, *, archive: Path | None = None) -> Path:
    """Verify every source byte before publishing a new source directory.

    Existing destinations are never replaced. Extraction deliberately supports
    only regular files: no tar links, device nodes, ownership, or mode restoration.
    """
    _validate_pin(pin)
    if archive is None:
        result = subprocess.run(
            [
                "hf",
                "download",
                pin["repo_id"],
                pin["filename"],
                "--repo-type",
                "dataset",
                "--revision",
                pin["revision"],
                "--quiet",
            ],
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=600,
        )
        archive = Path(result.stdout.strip())
    destination = destination.absolute()
    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    receipt = {"pin": pin, "state": "materializing", "started_ns": time.time_ns()}
    receipt_path = destination / "materialization.json"
    try:
        with tempfile.TemporaryDirectory(prefix=".source-", dir=destination) as temporary:
            staging = Path(temporary)
            # Copy once into private staging; do not reopen a changing HF cache
            # file after hashing it. Archive and member identities are independent.
            copied = staging / "archive.tar"
            with archive.open("rb") as source, copied.open("xb") as target:
                remaining = MAX_BYTES
                while block := source.read(min(1024**2, remaining + 1)):
                    remaining -= len(block)
                    if remaining < 0:
                        raise ValueError("recipe archive exceeds the materialization limit")
                    target.write(block)
            if copied.stat().st_size > MAX_BYTES or _digest(copied) != pin["sha256"]:
                raise ValueError("downloaded recipe archive differs from its immutable pin")
            source_root = staging / "source"
            source_root.mkdir()
            with tarfile.open(copied, "r:*") as source:
                members = []
                total = 0
                for member in source:
                    total += member.size
                    if len(members) >= MAX_FILES + 1 or total > MAX_BYTES + MAX_MANIFEST_BYTES:
                        raise ValueError("recipe archive inventory exceeds its limit")
                    if not member.isfile():
                        raise ValueError("recipe contains a link or non-file entry")
                    _relative(member.name)
                    members.append(member)
                names = [_relative(member.name) for member in members]
                if len(set(names)) != len(names) or any(not member.isfile() for member in members):
                    raise ValueError("recipe contains duplicate names, links or non-file entries")
                metadata = source.getmember("recipe.json")
                if not 0 < metadata.size <= MAX_MANIFEST_BYTES:
                    raise ValueError("recipe manifest exceeds its size limit")
                raw = source.extractfile(metadata).read()
                recipe = _manifest(json.loads(raw))
                if set(names) != {"recipe.json", *recipe["files"]}:
                    raise ValueError("archive and source manifest have different members")
                for member in members:
                    facts = recipe["files"].get(member.name)
                    if facts is not None and member.size != facts["bytes"]:
                        raise ValueError("archive source member size differs: " + member.name)
                    target = source_root / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as stream, target.open("xb") as output:
                        shutil.copyfileobj(stream, output, length=1024**2)
                    if facts is not None and _digest(target) != facts["sha256"]:
                        raise ValueError("archive source member digest differs: " + member.name)
                    target.chmod(0o400)
            copied.chmod(0o400)
            copied.rename(destination / "archive.tar")
            source_root.rename(destination / "source")
            receipt.update(
                state="source_verified_not_measurement_admission",
                recipe_sha256=hashlib.sha256(raw).hexdigest(),
                source_members=len(recipe["files"]),
            )
    except BaseException as error:
        receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["finished_ns"] = time.time_ns()
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return destination / "source"


def verify_materialized(destination: Path, pin: dict) -> dict:
    _validate_pin(pin)
    receipt = json.loads((destination / "materialization.json").read_text())
    root = destination / "source"
    if (
        receipt.get("state") != "source_verified_not_measurement_admission"
        or receipt.get("pin") != pin
        or root.is_symlink()
    ):
        raise ValueError("recipe source materialization did not complete")
    archive = destination / "archive.tar"
    if archive.is_symlink() or _digest(archive) != pin["sha256"]:
        raise ValueError("retained recipe archive differs from the supplied pin")
    with tarfile.open(archive, "r:*") as source:
        member = source.getmember("recipe.json")
        if not member.isfile() or not 0 < member.size <= MAX_MANIFEST_BYTES:
            raise ValueError("invalid pinned recipe manifest")
        raw = source.extractfile(member).read()
    if hashlib.sha256(raw).hexdigest() != receipt["recipe_sha256"] or (root / "recipe.json").read_bytes() != raw:
        raise ValueError("materialized recipe manifest changed")
    recipe = _manifest(json.loads((root / "recipe.json").read_text()))
    current = list(root.rglob("*"))
    if any(path.is_symlink() for path in current):
        raise ValueError("materialized recipe contains a link")
    if {str(path.relative_to(root)) for path in current if path.is_file()} != {"recipe.json", *recipe["files"]}:
        raise ValueError("materialized recipe member inventory changed")
    for name, facts in recipe["files"].items():
        path = root / name
        if path.stat().st_size != facts["bytes"] or _digest(path) != facts["sha256"]:
            raise ValueError("materialized source changed: " + name)
    return recipe


def run_recipe(destination: Path, pin: dict, arguments: list[str]) -> int:
    """Run only the pinned entry point; its native acceptance remains authoritative."""
    destination = destination.absolute()
    recipe = verify_materialized(destination, pin)
    entry = recipe["entrypoint"]
    interpreter = sys.executable if entry["interpreter"] == "python" else "/bin/bash"
    argv = [
        interpreter,
        *(["-B"] if entry["interpreter"] == "python" else []),
        str(destination / "source" / entry["path"]),
        *arguments,
    ]
    record = {
        "state": "starting",
        "argv": argv,
        "started_ns": time.time_ns(),
        "recipe_sha256": _digest(destination / "source/recipe.json"),
        "measurement_admission": "owned by the archived native recipe",
    }
    # No shell interpolation, automatic sbatch, retries, source substitution, or
    # conversion of the recipe's nonzero status into successful collection.
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        record["exit_code"] = subprocess.run(argv, cwd=destination / "source", env=environment, check=False).returncode
        record["state"] = "process_exited"
        return record["exit_code"]
    except BaseException as error:
        record.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_ns"] = time.time_ns()
        (destination / f"execution-{uuid.uuid4()}.json").write_text(json.dumps(record, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded archive with the same SHA256.")
    parser.add_argument("--run", action="store_true", help="Explicitly execute the pinned collector recipe.")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    pin = read_pin(args.pin)
    materialize(pin, args.destination, archive=args.archive)
    if args.run:
        return run_recipe(args.destination, pin, args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
    print(json.dumps({"source": str(args.destination / "source"), "measurement_admission": "NOT_EVALUATED"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
