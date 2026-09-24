# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only source capture and streaming verification of raw campaign evidence.

Python 3.12+, Linux/ARM64 or x86_64, standard library only. Inputs must be
quiescent. This provides verified bytes and membership, not a filesystem
snapshot or native inference/accuracy qualification. Failed outputs are kept.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

CHUNK = 1024 * 1024
SCHEMA = "glm53flash_raw_archive_v1"
INVENTORY = "source-inventory.jsonl"
INPUT_MANIFEST = "input-manifest.json"
ARCHIVE = "campaign.tar.gz"
FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
    "st_uid",
    "st_gid",
    "st_nlink",
)


class ArchiveError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ArchiveError(message)


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def absolute_safe(path, *, must_exist=True):
    path = Path(path).expanduser().absolute()
    require(".." not in path.parts, "parent traversal in path")
    for ancestor in reversed((path, *path.parents)):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            require(not must_exist and ancestor == path, "missing parent path")
            continue
        require(not stat.S_ISLNK(info.st_mode), "symlink path is forbidden")
    return path


def relative_parts(value):
    require(
        isinstance(value, str) and "\\" not in value and "\x00" not in value,
        "unsafe member path",
    )
    if value == "":
        return ()
    parts = value.split("/")
    require(
        not PurePosixPath(value).is_absolute() and all(part not in ("", ".", "..") for part in parts),
        "unsafe member path",
    )
    return tuple(parts)


def stat_identity(info):
    return {field.removeprefix("st_"): getattr(info, field) for field in STAT_FIELDS}


def assert_stat(info, expected, path):
    require(stat_identity(info) == expected, f"source stat changed: {path}")


@contextlib.contextmanager
def open_at(root_fd, parts, *, directory=False):
    fd = os.dup(root_fd)
    try:
        for index, component in enumerate(parts):
            flags = FLAGS | (os.O_DIRECTORY if directory or index < len(parts) - 1 else 0)
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        yield fd
    finally:
        os.close(fd)


def walk_source(root_fd, parts=()):
    """Memory scales with siblings in one directory, never payload bytes."""
    with open_at(root_fd, parts, directory=True) as directory:
        before = stat_identity(os.fstat(directory))
        yield {"path": "/".join(parts), "kind": "directory", "stat": before}
        with os.scandir(directory) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            relative_parts(name)
            require("/" not in name, "invalid directory entry")
            child = parts + (name,)
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                yield from walk_source(root_fd, child)
            else:
                require(
                    stat.S_ISREG(info.st_mode),
                    f"non-regular/symlink source: {'/'.join(child)}",
                )
                yield {
                    "path": "/".join(child),
                    "kind": "file",
                    "stat": stat_identity(info),
                }
        assert_stat(os.fstat(directory), before, "/".join(parts))


def hash_stream(stream):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def hash_source_file(root_fd, record):
    with open_at(root_fd, relative_parts(record["path"])) as fd:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "source file changed type")
        assert_stat(before, record["stat"], record["path"])
        with os.fdopen(os.dup(fd), "rb") as stream:
            digest, size = hash_stream(stream)
        assert_stat(os.fstat(fd), record["stat"], record["path"])
        require(size == record["stat"]["size"], "source size changed during read")
        return digest


def build_inventory(root_fd, path):
    counts = {"files": 0, "directories": 0, "logical_bytes": 0}
    with path.open("x", encoding="utf-8") as stream:
        for record in walk_source(root_fd):
            if record["kind"] == "file":
                record["sha256"] = hash_source_file(root_fd, record)
                counts["files"] += 1
                counts["logical_bytes"] += record["stat"]["size"]
            else:
                counts["directories"] += 1
            stream.write(canonical(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return counts


def inventory_records(path):
    stack = []
    with path.open("r", encoding="utf-8") as stream:
        while line := stream.readline(CHUNK):
            require(line.endswith("\n"), "oversized or incomplete inventory record")
            record = json.loads(line)
            parts = relative_parts(record["path"])
            require(record["kind"] in ("file", "directory"), "invalid inventory kind")
            if not stack:
                require(
                    parts == () and record["kind"] == "directory",
                    "inventory must start with source root",
                )
                stack.append([(), None])
            else:
                require(parts, "duplicate root inventory record")
                parent = parts[:-1]
                while stack and stack[-1][0] != parent:
                    stack.pop()
                require(
                    stack,
                    "inventory parent is absent or traversal returned to a closed directory",
                )
                require(
                    stack[-1][1] is None or parts[-1] > stack[-1][1],
                    "duplicate or unordered inventory member",
                )
                stack[-1][1] = parts[-1]
                if record["kind"] == "directory":
                    stack.append([parts, None])
            yield record


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        require(0 <= size <= CHUNK, "unbounded archive read requested")
        block = self.stream.read(size)
        self.digest.update(block)
        self.size += len(block)
        return block


def tar_info(record):
    name = "payload" + ("/" + record["path"] if record["path"] else "")
    result = tarfile.TarInfo(name)
    info = record["stat"]
    result.mode = stat.S_IMODE(info["mode"])
    result.uid, result.gid = info["uid"], info["gid"]
    seconds, nanos = divmod(info["mtime_ns"], 1_000_000_000)
    result.mtime = seconds
    result.pax_headers = {"mtime": f"{seconds}.{nanos:09d}"}
    result.type = tarfile.DIRTYPE if record["kind"] == "directory" else tarfile.REGTYPE
    result.size = 0 if record["kind"] == "directory" else info["size"]
    return result


def write_archive(root_fd, output):
    with (output / ARCHIVE).open("xb") as raw:
        with (
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed,
            tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive,
        ):
            for record in inventory_records(output / INVENTORY):
                if record["kind"] == "directory":
                    with open_at(root_fd, relative_parts(record["path"]), directory=True) as fd:
                        assert_stat(os.fstat(fd), record["stat"], record["path"])
                    archive.addfile(tar_info(record))
                    continue
                with open_at(root_fd, relative_parts(record["path"])) as fd:
                    assert_stat(os.fstat(fd), record["stat"], record["path"])
                    with os.fdopen(os.dup(fd), "rb") as stream:
                        reader = HashingReader(stream)
                        archive.addfile(tar_info(record), reader)
                    assert_stat(os.fstat(fd), record["stat"], record["path"])
                    require(
                        reader.digest.hexdigest() == record["sha256"] and reader.size == record["stat"]["size"],
                        f"source bytes changed while archiving: {record['path']}",
                    )
            for name in (INVENTORY, INPUT_MANIFEST):
                info = tarfile.TarInfo(name)
                info.size = (output / name).stat().st_size
                info.mode = 0o600
                with (output / name).open("rb") as stream:
                    archive.addfile(info, stream)
        raw.flush()
        os.fsync(raw.fileno())


def verify_archive(archive_path, inventory_path, input_manifest_path):
    """Extract each member into a bounded hash stream, never onto the filesystem."""
    inventory_path, input_manifest_path = (
        Path(inventory_path),
        Path(input_manifest_path),
    )
    expected = iter(inventory_records(inventory_path))
    current = next(expected, None)
    files, directories, size = 0, 0, 0
    metadata = iter((inventory_path, input_manifest_path))
    with (
        Path(archive_path).open("rb") as raw,
        gzip.GzipFile(fileobj=raw, mode="rb") as compressed,
        tarfile.open(fileobj=compressed, mode="r|") as archive,
    ):
        for member in archive:
            relative_parts(member.name)
            require(
                member.isfile() or member.isdir(),
                "archive contains link or special member",
            )
            if current is not None:
                info = tar_info(current)
                require(
                    member.name == info.name and member.type == info.type and member.size == info.size,
                    "archive membership, order, type or size differs from source inventory",
                )
                require(
                    (
                        member.mode,
                        member.uid,
                        member.gid,
                        member.pax_headers.get("mtime"),
                    )
                    == (info.mode, info.uid, info.gid, info.pax_headers["mtime"]),
                    "archive source metadata mismatch",
                )
                if member.isfile():
                    digest, count = hash_stream(archive.extractfile(member))
                    require(
                        digest == current["sha256"] and count == current["stat"]["size"],
                        f"archive file SHA256 mismatch: {member.name}",
                    )
                    files += 1
                    size += count
                else:
                    directories += 1
                current = next(expected, None)
            else:
                path = next(metadata, None)
                require(
                    path is not None and member.isfile() and member.name == path.name,
                    "archive contains unexpected or duplicate members",
                )
                digest, count = hash_stream(archive.extractfile(member))
                require(
                    digest == sha_file(path) and count == path.stat().st_size,
                    "embedded manifest differs",
                )
        require(
            current is None and next(metadata, None) is None,
            "archive is incomplete",
        )
        # tar iteration stops at a zero header. Drain buffered bytes too,
        # rejecting appended nonzero tar data and checking gzip trailer CRC.
        while block := archive.fileobj.read(CHUNK):
            require(not any(block), "nonzero trailing tar data")
    return {
        "status": "PASS",
        "files": files,
        "directories": directories,
        "logical_bytes": size,
        "method": "streaming_tar_extractfile_sha256_and_exact_ordered_membership",
    }


def verify_source(root_fd, inventory_path, *, hash_files):
    expected = iter(inventory_records(inventory_path))
    for current in walk_source(root_fd):
        record = next(expected, None)
        require(
            record is not None and current["path"] == record["path"] and current["kind"] == record["kind"],
            "source membership changed",
        )
        require(current["stat"] == record["stat"], f"source stat changed: {current['path']}")
        if hash_files and current["kind"] == "file":
            require(
                hash_source_file(root_fd, current) == record["sha256"],
                "source content changed after archive",
            )
    require(next(expected, None) is None, "source files disappeared")


def validate_identity(uri, labels):
    url = urlsplit(uri)
    require(
        url.scheme in ("https", "s3", "ssh")
        and url.netloc
        and url.path
        and not url.username
        and not url.password
        and not url.query
        and not url.fragment,
        "external URI must be stable and credential-free",
    )
    require(
        isinstance(labels, list) and labels,
        "nonempty explicit phase/role labels required",
    )
    seen = set()
    for label in labels:
        require(
            set(label) == {"backend", "weight_quantization", "tp", "phase", "role"},
            "invalid label keys",
        )
        require(
            label["backend"] in ("vllm", "sglang")
            and label["weight_quantization"] in ("fp8", "nvfp4")
            and type(label["tp"]) is int
            and label["tp"] in (2, 4)
            and label["phase"] in ("prefill", "decode")
            and label["role"] in ("calibration", "holdout"),
            "label is outside the required eight-configuration matrix",
        )
        require(canonical(label) not in seen, "duplicate external evidence label")
        seen.add(canonical(label))


def create_archive(source, output, uri, labels):
    source = absolute_safe(source)
    output = absolute_safe(output, must_exist=False)
    require(source.is_dir(), "source must be a directory")
    require(not output.is_relative_to(source), "output must not be inside source")
    validate_identity(uri, labels)
    os.mkdir(output, mode=0o700)
    root_fd = None
    try:
        root_fd = os.open(source, FLAGS | os.O_DIRECTORY)
        root_stat = stat_identity(os.fstat(root_fd))
        counts = build_inventory(root_fd, output / INVENTORY)
        inventory = {
            "path": INVENTORY,
            "sha256": sha_file(output / INVENTORY),
            "bytes": (output / INVENTORY).stat().st_size,
            **counts,
        }
        write_json(
            output / INPUT_MANIFEST,
            {
                "schema": SCHEMA,
                "source_path": str(source),
                "source_root_stat": root_stat,
                "source_inventory": inventory,
                "external_uri": uri,
                "labels": labels,
                "native_or_accuracy_acceptance": "NOT_EVALUATED",
            },
        )
        write_archive(root_fd, output)
        verification = verify_archive(output / ARCHIVE, output / INVENTORY, output / INPUT_MANIFEST)
        require(
            all(verification[key] == counts[key] for key in counts),
            "archive verification counts differ",
        )
        verify_source(root_fd, output / INVENTORY, hash_files=True)
        verify_source(root_fd, output / INVENTORY, hash_files=False)
        assert_stat(source.lstat(), root_stat, "source root")
        receipt = {
            "schema": SCHEMA,
            "status": "PASS",
            "archive": {
                "path": ARCHIVE,
                "sha256": sha_file(output / ARCHIVE),
                "bytes": (output / ARCHIVE).stat().st_size,
            },
            "source_inventory": inventory,
            "input_manifest": {
                "path": INPUT_MANIFEST,
                "sha256": sha_file(output / INPUT_MANIFEST),
                "bytes": (output / INPUT_MANIFEST).stat().st_size,
            },
            "verification": verification,
            "source_recheck": "STAT_AND_SHA256_PASS",
            "external_uri": uri,
            "external_uri_verification": "NOT_CHECKED",
            "labels": labels,
            "native_or_accuracy_acceptance": "NOT_EVALUATED",
        }
        write_json(
            output / "external-raw-evidence.json",
            [
                dict(
                    label,
                    uri=uri,
                    sha256=receipt["archive"]["sha256"],
                    bytes=receipt["archive"]["bytes"],
                    archive_verification="PASS",
                    external_uri_verification="NOT_CHECKED",
                )
                for label in labels
            ],
        )
        write_json(output / "receipt.json", receipt)
        return receipt
    except Exception as error:
        write_json(
            output / "failure.json",
            {
                "schema": SCHEMA,
                "status": "FAILED",
                "verification": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "source_actions": "READ_ONLY_NO_DELETION",
                "partial_output_preserved": True,
                "native_or_accuracy_acceptance": "NOT_EVALUATED",
            },
        )
        raise
    finally:
        if root_fd is not None:
            os.close(root_fd)


def verify_bundle(output):
    output = absolute_safe(output)
    require(not (output / "failure.json").exists(), "bundle contains a failure receipt")
    receipt = json.loads((output / "receipt.json").read_text())
    require(
        receipt.get("schema") == SCHEMA and receipt.get("status") == "PASS",
        "bundle has no passed receipt",
    )
    for key, name in (
        ("archive", ARCHIVE),
        ("source_inventory", INVENTORY),
        ("input_manifest", INPUT_MANIFEST),
    ):
        item = receipt[key]
        require(item["path"] == name, "unexpected receipt filename")
        path = absolute_safe(output / name)
        require(sha_file(path) == item["sha256"], "bundle receipt SHA256 mismatch")
        require(path.stat().st_size == item["bytes"], "bundle receipt byte count mismatch")
    manifest = json.loads((output / INPUT_MANIFEST).read_text())
    require(
        manifest["source_inventory"] == receipt["source_inventory"]
        and manifest["labels"] == receipt["labels"]
        and manifest["external_uri"] == receipt["external_uri"],
        "input manifest disagrees with receipt",
    )
    validate_identity(receipt["external_uri"], receipt["labels"])
    external = json.loads(absolute_safe(output / "external-raw-evidence.json").read_text())
    expected_external = [
        dict(
            label,
            uri=receipt["external_uri"],
            sha256=receipt["archive"]["sha256"],
            bytes=receipt["archive"]["bytes"],
            archive_verification="PASS",
            external_uri_verification="NOT_CHECKED",
        )
        for label in receipt["labels"]
    ]
    require(
        external == expected_external,
        "external evidence receipts disagree with verified archive",
    )
    actual = verify_archive(output / ARCHIVE, output / INVENTORY, output / INPUT_MANIFEST)
    require(
        actual == receipt["verification"],
        "verification receipt differs from archive contents",
    )
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--uri", required=True)
    create.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="JSON array of explicit backend/precision/TP/phase/role labels",
    )
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        labels_path = absolute_safe(args.labels)
        with os.fdopen(os.open(labels_path, FLAGS), "r", encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            require(stat.S_ISREG(info.st_mode) and info.st_size <= CHUNK, "labels must be a small regular JSON file")
            labels = json.load(stream)
        result = create_archive(args.source, args.output, args.uri, labels)
    else:
        result = verify_bundle(args.bundle)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
