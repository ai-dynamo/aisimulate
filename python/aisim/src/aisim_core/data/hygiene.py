# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Candidate hygiene for public dataset candidates (DEP #435, "CI and public evaluation").

A contributor publishes a candidate as a Hugging Face PR revision of the public
dataset repository. Before a maintainer starts the trusted validation workflow,
these pure checks decide whether the candidate may be fetched at all:

* who may create candidates (:func:`check_creator`),
* file-count and size limits plus allowed paths and suffixes
  (:func:`check_listing`),
* content checks on the leading and trailing bytes of each file
  (:func:`check_content`).

Nothing here touches the network or the filesystem. Callers pass the listing and
byte samples they obtained, so every rule is unit-testable with fixtures. A
finding is final: a candidate with any error finding must not reach a workflow.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

PARQUET_MAGIC = b"PAR1"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
# Executable and archive magics that never belong in a data candidate.
_FORBIDDEN_MAGIC: dict[str, bytes] = {
    "ELF executable": b"\x7fELF",
    "script": b"#!",
    "zip archive": b"PK\x03\x04",
    "gzip archive": b"\x1f\x8b",
    "Windows executable": b"MZ",
}
_FORBIDDEN_SUFFIXES = frozenset({".pkl", ".pickle", ".pt", ".pth", ".bin", ".so", ".py", ".sh", ".exe", ".zip", ".gz"})
_RESERVED_NAMES = frozenset({".gitattributes", ".gitignore", ".gitmodules"})


@dataclass(frozen=True)
class CandidatePolicy:
    """Limits and allow-lists. Defaults are conservative; the repository owner tunes them."""

    # HF user names allowed to open candidate PRs. Empty means nobody (fail closed).
    allowed_creators: frozenset[str] = frozenset()
    max_files: int = 200
    max_total_bytes: int = 512 * 1024 * 1024
    max_file_bytes: int = 256 * 1024 * 1024
    max_json_bytes: int = 4 * 1024 * 1024
    allowed_roots: tuple[str, ...] = ("data/", "catalog/")
    allowed_suffixes: frozenset[str] = frozenset({".parquet", ".json"})
    max_path_length: int = 240


@dataclass(frozen=True)
class CandidateFile:
    path: str
    size: int
    is_symlink: bool = False
    is_lfs: bool = False


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    path: str | None = None
    severity: Literal["error", "warning"] = "error"


@dataclass
class HygieneReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def extend(self, items: Iterable[Finding]) -> HygieneReport:
        self.findings.extend(items)
        return self


def check_creator(policy: CandidatePolicy, creator: str | None) -> list[Finding]:
    """Only listed HF users may create candidates; an unknown author fails closed."""
    if not creator:
        return [Finding("creator-unknown", "candidate creator could not be determined")]
    if creator.casefold() not in {c.casefold() for c in policy.allowed_creators}:
        return [Finding("creator-not-allowed", f"{creator!r} may not create candidates")]
    return []


def _path_findings(policy: CandidatePolicy, path: str) -> list[Finding]:
    bad: list[Finding] = []

    def err(code: str, msg: str) -> None:
        bad.append(Finding(code, msg, path))

    if not path or len(path) > policy.max_path_length:
        err("path-length", "path is empty or too long")
        return bad
    if path.startswith("/") or "\\" in path or "\x00" in path or any(ord(c) < 32 for c in path):
        err("path-unsafe", "absolute path, backslash or control character")
    parts = path.split("/")
    if ".." in parts or "." in parts or "" in parts or posixpath.normpath(path) != path:
        err("path-traversal", "path is not normalized or escapes the repository")
    if any(p in _RESERVED_NAMES for p in parts):
        err("path-reserved", "repository control files may not be changed by a candidate")
    if not any(path.startswith(root) for root in policy.allowed_roots):
        err("path-root", f"path is outside {list(policy.allowed_roots)}")
    suffix = posixpath.splitext(parts[-1])[1].lower()
    if suffix in _FORBIDDEN_SUFFIXES or suffix not in policy.allowed_suffixes:
        err("suffix", f"suffix {suffix or '(none)'!r} is not allowed")
    return bad


def check_listing(policy: CandidatePolicy, files: Iterable[CandidateFile]) -> list[Finding]:
    """Count, size, path and symlink rules over the whole candidate file list."""
    listing = list(files)
    out: list[Finding] = []
    if not listing:
        out.append(Finding("empty", "candidate has no files"))
    if len(listing) > policy.max_files:
        out.append(Finding("too-many-files", f"{len(listing)} files exceed the limit of {policy.max_files}"))
    total = sum(f.size for f in listing)
    if total > policy.max_total_bytes:
        out.append(Finding("too-large", f"{total} bytes exceed the limit of {policy.max_total_bytes}"))
    seen: dict[str, str] = {}
    for f in listing:
        out.extend(_path_findings(policy, f.path))
        folded = f.path.casefold()
        if folded in seen:
            out.append(Finding("duplicate-path", f"collides with {seen[folded]!r} ignoring case", f.path))
        seen[folded] = f.path
        if f.size < 0 or f.size > policy.max_file_bytes:
            out.append(Finding("file-too-large", f"{f.size} bytes exceed {policy.max_file_bytes}", f.path))
        if f.is_symlink:
            out.append(Finding("symlink", "symbolic links are not allowed", f.path))
        if f.is_lfs:
            out.append(Finding("lfs-pointer", "LFS pointers are not allowed; upload the content", f.path))
    return out


def check_content(policy: CandidatePolicy, file: CandidateFile, head: bytes, tail: bytes = b"") -> list[Finding]:
    """Check magic bytes and JSON shape from the first and last bytes of a file.

    ``head`` is the first bytes (at least 16 when available), ``tail`` the last
    four bytes. JSON files must be complete in ``head`` (small enough to read).
    """
    path = file.path
    suffix = posixpath.splitext(path)[1].lower()
    out: list[Finding] = []
    if head.startswith(LFS_POINTER_PREFIX):
        out.append(Finding("lfs-pointer", "file is a Git LFS pointer", path))
    for label, magic in _FORBIDDEN_MAGIC.items():
        if head.startswith(magic):
            out.append(Finding("forbidden-content", f"content looks like a {label}", path))
    if suffix == ".parquet":
        if head[:4] != PARQUET_MAGIC or tail[-4:] != PARQUET_MAGIC:
            out.append(Finding("parquet-magic", "not a Parquet file (missing PAR1 magic)", path))
    elif suffix == ".json":
        if file.size > policy.max_json_bytes:
            out.append(Finding("json-too-large", f"JSON over {policy.max_json_bytes} bytes", path))
        elif len(head) < file.size:
            out.append(Finding("json-incomplete", "full JSON content is needed for the check", path))
        else:
            try:
                json.loads(head.decode("utf-8"), parse_constant=_reject_constant)
            except (UnicodeDecodeError, ValueError) as exc:
                out.append(Finding("json-invalid", f"not strict UTF-8 JSON: {exc}", path))
    return out


def _reject_constant(name: str):
    raise ValueError(f"non-finite constant {name}")


def check_candidate(
    policy: CandidatePolicy,
    creator: str | None,
    files: Iterable[CandidateFile],
    samples: Mapping[str, tuple[bytes, bytes]],
) -> HygieneReport:
    """Run every check. ``samples`` maps path -> (head, tail); missing samples are errors."""
    listing = list(files)
    report = HygieneReport()
    report.extend(check_creator(policy, creator))
    report.extend(check_listing(policy, listing))
    for f in listing:
        sample = samples.get(f.path)
        if sample is None:
            report.findings.append(Finding("no-sample", "content was not inspected", f.path))
            continue
        report.extend(check_content(policy, f, *sample))
    return report
