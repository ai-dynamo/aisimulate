# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve, plan and fetch dataset files from a pinned Hugging Face revision.

This is the internal library behind ``aisim data fetch`` (DEP #435,
"Selective access and download size"). It has four guarantees:

* A branch or tag is resolved to a full commit SHA exactly once per call; every
  later read uses that SHA, so a moving ref can never mix revisions.
* A selection is required. It must name configuration IDs or a hardware filter
  (plus a model for fpm/e2e/eval) and must match at least one catalog entry;
  an empty selection is an error, never a whole-repository download.
* Files are verified (size, SHA-256, Parquet schema ID and row count) before
  they enter the cache. The cache is keyed by repo + commit + path and entries
  are renamed into place atomically, so an interrupted transfer leaves nothing
  at the final path.
* ``offline=True`` never touches the transport. Missing files are reported.

Network access goes through the injectable :class:`Transport`; tests use fakes.
:class:`HfTransport` is the real implementation and imports ``huggingface_hub``
lazily. Credentials are taken from the caller (``token``) or the standard
Hugging Face environment and are never stored or printed here.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq
from pydantic import ValidationError

from aisim_core.data import (
    Artifact,
    Category,
    CategoryIndex,
    DatasetIndex,
    Entry,
    Manifest,
    _relative,
)
from aisim_core.data.parquet import read_parquet_schema_id

_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_CHUNK = 1 << 20
DATASET_INDEX_PATH = "catalog/dataset.json"
_MODEL_REQUIRED = {"fpm", "e2e", "eval"}


# --------------------------------------------------------------------------- errors


class DataFetchError(Exception):
    """Base class; ``to_dict`` gives agents a typed, JSON-serializable error."""

    code = "data-fetch-error"

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": str(self)}


class SelectionError(DataFetchError):
    code = "invalid-selection"


class RevisionError(DataFetchError):
    code = "revision-unresolved"


class IntegrityError(DataFetchError):
    code = "integrity"

    def __init__(self, path: str, reason: str):
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "path": self.path, "reason": self.reason}


class TransferError(DataFetchError):
    code = "transfer-failed"


class BudgetExceededError(DataFetchError):
    code = "over-budget"

    def __init__(self, plan: FetchPlan, max_bytes: int):
        super().__init__(
            f"plan needs {plan.uncached_bytes} uncached bytes ({plan.total_bytes} total) which exceeds "
            f"max_bytes={max_bytes}; pass allow_over_budget=True to proceed"
        )
        self.plan = plan
        self.max_bytes = max_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "max_bytes": self.max_bytes,
            "total_bytes": self.plan.total_bytes,
            "uncached_bytes": self.plan.uncached_bytes,
        }


class OfflineMissingError(DataFetchError):
    code = "offline-missing"

    def __init__(self, missing: Sequence[str]):
        super().__init__(f"offline: {len(missing)} file(s) not in the cache: {', '.join(missing)}")
        self.missing = list(missing)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "missing": self.missing}


# --------------------------------------------------------------------------- transport


class Transport(Protocol):
    """The only network surface. Both calls may be slow and may raise."""

    def resolve_revision(self, repo: str, revision: str) -> str:
        """Return the full 40-hex commit SHA for a branch, tag or SHA."""

    def open_file(self, repo: str, sha: str, path: str) -> Iterator[bytes]:
        """Yield the content of ``path`` at commit ``sha`` in chunks."""


class HfTransport:
    """Hugging Face dataset transport (``huggingface_hub`` is imported lazily)."""

    def __init__(self, token: str | None = None):
        self._token = token

    def resolve_revision(self, repo: str, revision: str) -> str:
        from huggingface_hub import HfApi

        return HfApi(token=self._token).repo_info(repo, repo_type="dataset", revision=revision).sha

    def open_file(self, repo: str, sha: str, path: str) -> Iterator[bytes]:
        from huggingface_hub import hf_hub_download

        # hf_hub_download handles auth, retries and resumable transfer; the
        # temporary copy is streamed into our own cache and removed.
        with tempfile.TemporaryDirectory(prefix="aisim-hf-") as scratch:
            local = hf_hub_download(
                repo_id=repo,
                filename=path,
                repo_type="dataset",
                revision=sha,
                token=self._token,
                local_dir=scratch,
            )
            with open(local, "rb") as stream:
                while chunk := stream.read(_CHUNK):
                    yield chunk


def resolve_revision(transport: Transport | None, repo: str, revision: str, *, offline: bool = False) -> str:
    """Resolve ``revision`` to a full commit SHA. A full SHA needs no network."""
    _check_repo(repo)
    if _FULL_SHA.fullmatch(revision):
        return revision
    if offline or transport is None:
        raise RevisionError(
            f"offline: {repo}@{revision} is not a full commit SHA and cannot be resolved without network"
        )
    sha = transport.resolve_revision(repo, revision)
    if not isinstance(sha, str) or not _FULL_SHA.fullmatch(sha):
        raise RevisionError(f"{repo}@{revision} did not resolve to a full commit SHA")
    return sha


# --------------------------------------------------------------------------- selection and plan


@dataclass(frozen=True)
class Selection:
    """What to fetch. Selectors are required; see the module docstring."""

    category: Category
    configuration_ids: tuple[str, ...] = ()
    model: str | None = None
    hardware: str | None = None
    framework: str | None = None
    framework_version: str | None = None
    operation_family: str | None = None
    purpose: str | None = None
    # Artifact roles to leave out of the plan (for example ``fpm-provenance``,
    # 75 % of the FPM bytes, which no evaluation reads). Manifests are still read.
    exclude_roles: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.configuration_ids:
            if self.hardware is None:
                raise SelectionError("selection needs configuration_ids or a hardware filter")
            if self.category in _MODEL_REQUIRED and self.model is None:
                raise SelectionError(f"{self.category} selection needs a model or configuration_ids")

    def matches(self, entry: Entry) -> bool:
        if self.configuration_ids and entry.configuration_id not in self.configuration_ids:
            return False
        if self.purpose is not None and entry.purpose != self.purpose:
            return False
        identity = entry.identity
        for want, have in (
            (self.model, identity.model),
            (self.hardware, identity.hardware),
            (self.framework, identity.framework),
            (self.framework_version, identity.framework_version),
            (self.operation_family, identity.operation_family),
        ):
            if want is not None and want != have:
                return False
        return True


@dataclass
class PlannedFile:
    path: str
    bytes: int | None  # None for catalog files whose size is not declared
    sha256: str | None
    role: str  # catalog | index | manifest | artifact
    cached: bool
    format: str | None = None
    schema: str | None = None
    rows: int | None = None


@dataclass
class FetchPlan:
    repo: str
    revision: str  # full commit SHA
    requested_revision: str
    offline: bool
    files: list[PlannedFile] = field(default_factory=list)
    # Offline only: files that are needed but absent, including files whose
    # content (a manifest) could not be read so the closure is incomplete.
    missing: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(f.bytes or 0 for f in self.files)

    @property
    def uncached_bytes(self) -> int:
        return sum(f.bytes or 0 for f in self.files if not f.cached)

    @property
    def uncached(self) -> list[PlannedFile]:
        return [f for f in self.files if not f.cached]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "requested_revision": self.requested_revision,
            "offline": self.offline,
            "files": [asdict(f) for f in self.files],
            "missing": list(self.missing),
            "total_bytes": self.total_bytes,
            "uncached_bytes": self.uncached_bytes,
        }


@dataclass
class FetchResult:
    plan: FetchPlan
    root: Path  # cache root for this repo + commit
    paths: dict[str, Path]
    downloaded_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "root": str(self.root),
            "paths": {k: str(v) for k, v in self.paths.items()},
            "downloaded_bytes": self.downloaded_bytes,
        }


# --------------------------------------------------------------------------- cache


def _check_repo(repo: str) -> None:
    if not _REPO.fullmatch(repo):
        raise SelectionError(f"invalid repository id: {repo!r}")


def default_cache_dir() -> Path:
    override = os.environ.get("AISIM_DATA_CACHE")
    if override:
        return Path(override)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "aisim" / "data"


class Cache:
    """Files live at ``<root>/<owner>__<name>/<sha>/<path>``; writes are atomic."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_cache_dir()

    def commit_root(self, repo: str, sha: str) -> Path:
        _check_repo(repo)
        if not _FULL_SHA.fullmatch(sha):
            raise RevisionError("cache is keyed by a full commit SHA")
        return self.root / repo.replace("/", "__") / sha

    def path_for(self, repo: str, sha: str, path: str) -> Path:
        _relative(path)
        return self.commit_root(repo, sha).joinpath(*path.split("/"))

    def valid(self, repo: str, sha: str, path: str, expected: PlannedFile | None) -> bool:
        """True if the file exists and (when declared) matches size and digest."""
        target = self.path_for(repo, sha, path)
        if target.is_symlink() or not target.is_file():
            return False
        if expected is None or expected.sha256 is None:
            return True
        try:
            return _digest(target) == (expected.sha256, expected.bytes)
        except OSError:
            return False

    def store(
        self, repo: str, sha: str, path: str, chunks: Iterable[bytes], expected: PlannedFile | None
    ) -> tuple[Path, int]:
        """Stream ``chunks`` to a temp file, verify, then rename into place."""
        target = self.path_for(repo, sha, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".part")
        tmp = Path(tmp_name)
        written = 0
        try:
            with os.fdopen(fd, "wb") as out:
                for chunk in chunks:
                    written += len(chunk)
                    if expected is not None and expected.bytes is not None and written > expected.bytes:
                        raise IntegrityError(path, f"transfer exceeds declared size {expected.bytes}")
                    out.write(chunk)
            if expected is not None:
                _verify_file(tmp, path, expected)
            os.replace(tmp, target)
        except DataFetchError:
            tmp.unlink(missing_ok=True)
            raise
        except BaseException as exc:
            tmp.unlink(missing_ok=True)
            if isinstance(exc, Exception):
                raise TransferError(f"{path}: transfer interrupted: {exc}") from exc
            raise
        return target, written

    def discard(self, repo: str, sha: str, path: str) -> None:
        self.path_for(repo, sha, path).unlink(missing_ok=True)


def _digest(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest(), path.stat().st_size


def _verify_file(local: Path, path: str, expected: PlannedFile) -> None:
    digest, size = _digest(local)
    if expected.bytes is not None and size != expected.bytes:
        raise IntegrityError(path, f"size {size} != declared {expected.bytes}")
    if expected.sha256 is not None and digest != expected.sha256:
        raise IntegrityError(path, "SHA-256 mismatch")
    if expected.format == "parquet":
        try:
            schema_id = read_parquet_schema_id(local)
            rows = pq.ParquetFile(local).metadata.num_rows
        except Exception as exc:
            raise IntegrityError(path, f"unreadable Parquet: {exc}") from exc
        # A file published before schema stamping (the byte-identical FPM libraries) has no embedded ID;
        # it can only declare an external schema, never an aisim-* one.
        unstamped_external = (
            schema_id is None and expected.schema is not None and not expected.schema.startswith("aisim-")
        )
        if expected.schema is not None and schema_id != expected.schema and not unstamped_external:
            raise IntegrityError(path, f"schema ID {schema_id!r} != declared {expected.schema!r}")
        if expected.rows is not None and rows != expected.rows:
            raise IntegrityError(path, f"row count {rows} != declared {expected.rows}")


# --------------------------------------------------------------------------- plan


def _ref(artifact: Artifact, role: str) -> PlannedFile:
    return PlannedFile(
        path=artifact.path,
        bytes=artifact.bytes,
        sha256=artifact.sha256,
        role=role,
        cached=False,
        format=artifact.format,
        schema=artifact.artifact_schema,
        rows=artifact.rows,
    )


class _Planner:
    def __init__(self, transport: Transport | None, cache: Cache, plan: FetchPlan):
        self.transport, self.cache, self.plan = transport, cache, plan
        self._seen: dict[str, PlannedFile] = {}

    def _add(self, ref: PlannedFile) -> None:
        prior = self._seen.get(ref.path)
        if prior is not None:
            if (prior.sha256, prior.bytes) != (ref.sha256, ref.bytes):
                raise IntegrityError(ref.path, "declared with conflicting digests")
            return
        ref.cached = self.cache.valid(self.plan.repo, self.plan.revision, ref.path, ref)
        self._seen[ref.path] = ref
        self.plan.files.append(ref)

    def read(self, ref: PlannedFile) -> bytes | None:
        """Bytes of a metadata file (catalog/index/manifest), fetching if online.

        Metadata is small and needed to expand the closure, so it is downloaded
        during planning (and counted as cached afterwards). Offline, a missing
        file is recorded and None is returned.
        """
        plan = self.plan
        if not self.cache.valid(plan.repo, plan.revision, ref.path, ref):
            if plan.offline or self.transport is None:
                self._add(ref)
                plan.missing.append(ref.path)
                return None
            self.cache.discard(plan.repo, plan.revision, ref.path)
            chunks = self.transport.open_file(plan.repo, plan.revision, ref.path)
            self.cache.store(plan.repo, plan.revision, ref.path, chunks, ref)
        self._add(ref)
        return self.cache.path_for(plan.repo, plan.revision, ref.path).read_bytes()


def plan(
    selection: Selection,
    *,
    repo: str,
    revision: str,
    transport: Transport | None = None,
    cache: Cache | None = None,
    offline: bool = False,
    _resolved: str | None = None,
) -> FetchPlan:
    """List the files a selection needs: catalog, index, manifests and artifacts.

    ``revision`` is resolved to a SHA once. Online, small metadata files are
    downloaded to expand the dependency closure; artifacts are never downloaded
    here. Offline, nothing is downloaded and absent files go in ``missing``.
    """
    selection.validate()
    cache = cache or Cache()
    sha = _resolved or resolve_revision(transport, repo, revision, offline=offline)
    out = FetchPlan(repo=repo, revision=sha, requested_revision=revision, offline=offline)
    planner = _Planner(transport, cache, out)

    def parse(model, raw: bytes, path: str):
        try:
            return model.model_validate_json(raw)
        except ValidationError as exc:
            raise IntegrityError(path, f"invalid {model.__name__}: {exc.error_count()} error(s)") from exc

    raw = planner.read(PlannedFile(DATASET_INDEX_PATH, None, None, "catalog", False))
    if raw is None:
        return out
    dataset = parse(DatasetIndex, raw, DATASET_INDEX_PATH)
    reference = dataset.categories.get(selection.category)
    if reference is None:
        raise SelectionError(f"dataset has no {selection.category!r} category")
    raw = planner.read(_ref(reference.index, "index"))
    if raw is None:
        return out
    index = parse(CategoryIndex, raw, reference.index.path)
    entries = [e for e in index.entries if selection.matches(e)]
    if not entries:
        raise SelectionError("selection matched no catalog entries")
    for entry in entries:
        raw = planner.read(_ref(entry.manifest, "manifest"))
        if raw is None:
            continue
        manifest = parse(Manifest, raw, entry.manifest.path)
        if manifest.configuration_id != entry.configuration_id:
            raise IntegrityError(entry.manifest.path, "manifest configuration ID differs from the catalog entry")
        for artifact in manifest.artifacts:
            if artifact.role not in selection.exclude_roles:
                planner._add(_ref(artifact, "artifact"))
    out.files.sort(key=lambda f: f.path)
    if offline:
        out.missing = sorted({*out.missing, *(f.path for f in out.files if not f.cached)})
    return out


# --------------------------------------------------------------------------- fetch


def fetch(
    selection: Selection,
    *,
    repo: str,
    revision: str = "main",
    transport: Transport | None = None,
    cache: Cache | None = None,
    max_bytes: int | None = None,
    allow_over_budget: bool = False,
    offline: bool = False,
) -> FetchResult:
    """Resolve once, plan, enforce the budget, then download and verify.

    ``max_bytes`` bounds the bytes still to download (cached files cost
    nothing); a larger plan raises :class:`BudgetExceededError` before any
    artifact transfer unless ``allow_over_budget`` is set. Offline with any
    file missing raises :class:`OfflineMissingError` listing every such file.
    """
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    cache = cache or Cache()
    sha = resolve_revision(transport, repo, revision, offline=offline)
    result_plan = plan(
        selection, repo=repo, revision=revision, transport=transport, cache=cache, offline=offline, _resolved=sha
    )
    if offline:
        if result_plan.missing:
            raise OfflineMissingError(result_plan.missing)
    elif max_bytes is not None and result_plan.uncached_bytes > max_bytes and not allow_over_budget:
        raise BudgetExceededError(result_plan, max_bytes)

    downloaded = 0
    for f in result_plan.files:
        if f.cached:
            continue
        assert transport is not None  # online: plan() already needed it
        cache.discard(repo, sha, f.path)
        _, size = cache.store(repo, sha, f.path, transport.open_file(repo, sha, f.path), f)
        downloaded += size
        f.cached = True
    paths = {f.path: cache.path_for(repo, sha, f.path) for f in result_plan.files}
    return FetchResult(plan=result_plan, root=cache.commit_root(repo, sha), paths=paths, downloaded_bytes=downloaded)


def load_category_index(
    category: Category,
    *,
    repo: str,
    revision: str,
    transport: Transport | None = None,
    cache: Cache | None = None,
    offline: bool = False,
) -> tuple[str, CategoryIndex, DatasetIndex]:
    """Resolve ``revision`` once and read the dataset and category indexes (a few KB).

    Returns ``(full_sha, category_index, dataset_index)``. Use it to choose
    configuration IDs before any artifact is planned or downloaded.
    """
    cache = cache or Cache()
    sha = resolve_revision(transport, repo, revision, offline=offline)
    out = FetchPlan(repo=repo, revision=sha, requested_revision=revision, offline=offline)
    planner = _Planner(transport, cache, out)

    def parse(model, raw: bytes | None, path: str):
        if raw is None:
            raise OfflineMissingError([path])
        try:
            return model.model_validate_json(raw)
        except ValidationError as exc:
            raise IntegrityError(path, f"invalid {model.__name__}: {exc.error_count()} error(s)") from exc

    dataset = parse(
        DatasetIndex,
        planner.read(PlannedFile(DATASET_INDEX_PATH, None, None, "catalog", False)),
        DATASET_INDEX_PATH,
    )
    reference = dataset.categories.get(category)
    if reference is None:
        raise SelectionError(f"dataset has no {category!r} category")
    index = parse(CategoryIndex, planner.read(_ref(reference.index, "index")), reference.index.path)
    return sha, index, dataset


def clear_cache(cache: Cache, repo: str, sha: str) -> None:
    """Remove one commit's cached files."""
    shutil.rmtree(cache.commit_root(repo, sha), ignore_errors=True)
