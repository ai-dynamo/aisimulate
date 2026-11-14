# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore the exact original source archives, manifests, reports and harnesses."""

import argparse
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = Path("benchmarks/evidence/afd-review585-round3")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def target(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes output directory: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_base(index, members, repo):
    names = sorted(
        {
            name
            for snapshot in index["snapshots"]
            for name, expected in json.loads(members[snapshot["manifest_member"]])["source_files"].items()
            if "objects/" + expected not in members
        }
    )
    request = "".join(index["base_commit"] + ":" + name + "\n" for name in names)
    raw = subprocess.check_output(["git", "-C", str(repo), "cat-file", "--batch"], input=request.encode())
    offset = 0
    for name in names:
        end = raw.index(b"\n", offset)
        header, offset = raw[offset:end].split(), end + 1
        if len(header) != 3 or header[1] != b"blob":
            raise ValueError(f"pinned Git base unavailable for {name}; use --repo with the recorded main commit")
        size = int(header[2])
        data = raw[offset : offset + size]
        offset += size + 1
        members["objects/" + digest(data)] = data


def restore(output, repo):
    if output.exists():
        raise ValueError("output must be a new directory; existing files will not be overwritten")
    validation = json.loads((HERE / "validation.json").read_text())
    for relative, expected in validation["evidence_files"].items():
        if digest((HERE / relative).read_bytes()) != expected:
            raise ValueError(f"evidence checksum mismatch: {relative}")
    with tarfile.open(HERE / "sources.tar.gz", "r:gz") as archive:
        members = {}
        for entry in archive:
            if not entry.isfile() or entry.name in members:
                raise ValueError("invalid or duplicate source package entry")
            members[entry.name] = archive.extractfile(entry).read()
    index = json.loads(members.pop("index.json"))
    if set(members) != set(index["member_sha256"]):
        raise ValueError("source package inventory mismatch")
    for name, data in members.items():
        if digest(data) != index["member_sha256"][name]:
            raise ValueError(f"source package checksum mismatch: {name}")
    load_base(index, members, repo)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="afd-restore-", dir=output.parent) as temporary:
        staged = Path(temporary) / "restored"
        evidence = staged / EVIDENCE
        for snapshot in index["snapshots"]:
            raw = members[snapshot["manifest_member"]]
            if digest(raw) != snapshot["manifest_sha256"]:
                raise ValueError("original manifest checksum mismatch")
            manifest = json.loads(raw)
            inventory = json.dumps(manifest["source_files"], sort_keys=True).encode()
            if digest(inventory) != manifest["source_inventory_sha256"]:
                raise ValueError("original source inventory mismatch")
            manifest_path = target(evidence, snapshot["output_manifest"])
            manifest_path.write_bytes(raw)
            archive_path = target(manifest_path.parent, manifest["source_snapshot"])
            with (
                archive_path.open("wb") as stream,
                gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as zipped,
                tarfile.open(fileobj=zipped, mode="w") as archive,
            ):
                for relative, expected in manifest["source_files"].items():
                    data = members["objects/" + expected]
                    if digest(data) != expected:
                        raise ValueError(f"original source checksum mismatch: {relative}")
                    entry = tarfile.TarInfo(relative)
                    entry.size, entry.mode = len(data), 0o644
                    archive.addfile(entry, io.BytesIO(data))
            if digest(archive_path.read_bytes()) != manifest["source_snapshot_sha256"]:
                raise ValueError("restored source archive differs from its original digest")
        for relative, member in index["exports"].items():
            target(evidence, relative).write_bytes(members[member])
        for relative, report in validation["compressed_reports"].items():
            raw = gzip.decompress((HERE / report["compressed"]).read_bytes())
            if digest(raw) != report["uncompressed_sha256"]:
                raise ValueError(f"original report checksum mismatch: {relative}")
            target(evidence, relative).write_bytes(raw)
        for relative in ("memory.json", "forward-gate/raw-results.json.gz", "forward-gate/comparison.json.gz"):
            target(evidence, relative).write_bytes((HERE / relative).read_bytes())
        followup = validation["follow_up_prefill_batched_token_fix"]
        for filename, key in (("validation.json", "validation"), ("memory-deltas.json", "memory_deltas")):
            target(evidence, "prefill-batched-token-fix/" + filename).write_text(
                json.dumps(followup[key], indent=2, sort_keys=True) + "\n"
            )
        staged.rename(output)
    print(f"Restored {len(index['snapshots'])} source archives with their original SHA256 identities: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the evidence package")
    parser.add_argument("--repo", type=Path, default=HERE.parents[2], help="Git repository containing pinned main")
    args = parser.parse_args()
    restore(args.output.resolve(), args.repo.resolve())
