# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The default ``systems/`` root: packaged specs plus op-latency data from the pinned dataset (AISIM-2063).

Operation tables are no longer packaged. The default systems root is composed on first use from the
``ai-dynamo/aisim-data`` revision named in ``op_latency_pin.json`` (a full commit SHA that ships in the wheel):

1. fetch every ``op-latency`` entry of that revision into the data cache (``AISIM_DATA_CACHE``; a warm cache
   makes no network call, a cold one downloads and hash-verifies),
2. write one ``systems/`` root next to it with :func:`~aisim_core.data.op_tree.load_op_latency_tree`, which
   also copies the packaged specs, profiles and version tables,
3. return that root; later processes reuse it.

Overrides, all explicit and never silently replaced by other data:

* ``systems_paths`` / ``AICONFIGURATOR_SYSTEMS_PATH`` name a root directly and skip this module.
* ``AISIM_OP_LATENCY_DATASET=<repo>@<full commit SHA>`` selects another revision.
* ``AISIM_OP_LATENCY_LOCAL=<dir>`` uses a local dataset-layout directory (no network).
* ``AISIM_DATA_OFFLINE=1`` forbids any download; ``AISIM_DATA_CACHE`` and ``AISIM_DATA_MAX_BYTES`` apply as for
  :func:`~aisim_core.data.ensure.pin_from_environment`.

Missing, unverifiable or offline-uncached data raises :class:`~aisim_core.data.ensure.MissingDataError`;
there is no packaged fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from aisim_core.data.ensure import DEFAULT_MAX_BYTES, DataTooLargeError, MissingDataError
from aisim_core.data.fetch import (
    BudgetExceededError,
    Cache,
    DataFetchError,
    OfflineMissingError,
    PlannedFile,
    RevisionError,
    Selection,
    SelectionError,
    Transport,
    fetch,
    load_category_index,
    plan,
    resolve_revision,
)
from aisim_core.data.op_schema import DATA_PREFIX, QUERY_VERSIONS_NAME, REUSE_MANIFEST_NAME
from aisim_core.data.op_tree import load_op_latency_tree
from aisim_core.data.systems import PACKAGED_SYSTEMS_DIR

PIN_FILE = Path(__file__).with_name("op_latency_pin.json")
DATASET_ENV = "AISIM_OP_LATENCY_DATASET"
LOCAL_ENV = "AISIM_OP_LATENCY_LOCAL"
MARKER = ".aisim-default-systems.json"
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_LOOKUP_FILES = tuple(f"{DATA_PREFIX}/{name}.parquet" for name in (REUSE_MANIFEST_NAME, QUERY_VERSIONS_NAME))
_WORKERS = 16
_resolved: dict[tuple, str] = {}


def op_latency_pin() -> tuple[str, str]:
    """``(repo, full commit SHA)`` this release reads op-latency data from (``op_latency_pin.json``)."""
    pin = json.loads(PIN_FILE.read_text(encoding="utf-8"))
    repo, revision = pin["repo_id"], pin["revision"]
    if not _FULL_SHA.fullmatch(revision):
        raise ValueError(f"{PIN_FILE.name} must pin a full commit SHA, got {revision!r}")
    return repo, revision


def _settings(env: Mapping[str, str]) -> tuple:
    local = env.get(LOCAL_ENV)
    repo, revision = op_latency_pin()
    spec = env.get(DATASET_ENV)
    if spec:
        repo, sep, revision = spec.partition("@")
        if not sep or not repo or not revision:
            raise ValueError(f"{DATASET_ENV} must be <owner>/<name>@<revision>")
    return (
        repo,
        revision,
        str(Path(local).absolute()) if local else None,
        env.get("AISIM_DATA_OFFLINE", "").lower() in {"1", "true", "yes"},
        env.get("AISIM_DATA_CACHE") or None,
        int(env.get("AISIM_DATA_MAX_BYTES") or DEFAULT_MAX_BYTES),
    )


def default_systems_path(*, transport: Transport | None = None) -> str:
    """The composed default ``systems/`` root (built on first use, then reused). Raises, never falls back."""
    settings = _settings(os.environ)
    key = (*settings, id(transport) if transport is not None else None)
    if key not in _resolved:
        _resolved[key] = str(_compose(*settings, transport=transport))
    return _resolved[key]


def resolved_default_systems_path() -> str | None:
    """The default root if already composed in this process (never builds or downloads)."""
    try:
        return _resolved.get((*_settings(os.environ), None))
    except (ValueError, OSError):
        return None


def clear_default_systems_cache() -> None:
    """Forget composed roots held by this process (tests); the on-disk cache is untouched."""
    _resolved.clear()


def _specs_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGED_SYSTEMS_DIR.rglob("*")):
        if path.is_file() and path.relative_to(PACKAGED_SYSTEMS_DIR).parts[0] != "data":
            stat = path.stat()
            digest.update(f"{path.relative_to(PACKAGED_SYSTEMS_DIR)}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    return digest.hexdigest()[:12]


def _compose(
    repo: str,
    revision: str,
    local: str | None,
    offline: bool,
    cache_dir: str | None,
    max_bytes: int,
    *,
    transport: Transport | None,
) -> Path:
    cache = Cache(cache_dir)
    if local is not None:
        tree = Path(local)
        index = tree / "catalog" / "dataset.json"
        if not index.is_file():
            raise MissingDataError(f"{LOCAL_ENV}={local} is not a dataset-layout directory (no catalog/dataset.json)")
        name = "local-" + hashlib.sha256(index.read_bytes()).hexdigest()[:16]
        source = {"source": "local", "root": local}
    else:
        tree, sha = _ensure_tree(repo, revision, cache, offline, max_bytes, transport)
        name = f"{repo.replace('/', '__')}-{sha}"
        source = {"source": "hf", "repo": repo, "revision": sha}
    destination = cache.root / "systems" / f"{name}-{_specs_fingerprint()}"
    if (destination / MARKER).is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".compose-")) / "systems"
    try:
        load_op_latency_tree(tree, staging)
        (staging / MARKER).write_text(json.dumps(source, sort_keys=True) + "\n")
        try:
            os.rename(staging, destination)  # atomic; a concurrent process may have won
        except OSError:
            if not (destination / MARKER).is_file():
                raise
    except DataFetchError:
        raise
    except Exception as exc:
        raise MissingDataError(f"op-latency data at {tree} failed verification: {exc}") from exc
    finally:
        shutil.rmtree(staging.parent, ignore_errors=True)
    return destination


def _ensure_tree(
    repo: str, revision: str, cache: Cache, offline: bool, max_bytes: int, transport: Transport | None
) -> tuple[Path, str]:
    where = f"{repo}@{revision}"
    try:
        return _fetch_all(repo, revision, cache, offline=True, max_bytes=max_bytes, transport=None)
    except (OfflineMissingError, RevisionError) as exc:
        if offline:
            raise MissingDataError(
                f"offline: op-latency data {where} is not fully cached ({exc}); there is no fallback to other "
                f"data. Unset AISIM_DATA_OFFLINE once to download it, or set {LOCAL_ENV}."
            ) from exc
    if transport is None:
        from aisim_core.data.fetch import HfTransport

        transport = HfTransport()
    try:
        return _fetch_all(repo, revision, cache, offline=False, max_bytes=max_bytes, transport=transport)
    except BudgetExceededError as exc:
        raise DataTooLargeError(
            exc.plan.uncached_bytes, max_bytes, f"aisim data fetch --category op-latency --repo {repo}"
        ) from exc
    except (DataFetchError, SelectionError, OSError) as exc:
        raise MissingDataError(f"cannot obtain op-latency data {where}: {exc}", source=where) from exc


def _download(transport: Transport, cache: Cache, repo: str, sha: str, f: PlannedFile) -> None:
    cache.discard(repo, sha, f.path)
    cache.store(repo, sha, f.path, transport.open_file(repo, sha, f.path), f)


def _fetch_all(
    repo: str, revision: str, cache: Cache, *, offline: bool, max_bytes: int, transport: Transport | None
) -> tuple[Path, str]:
    sha = resolve_revision(transport, repo, revision, offline=offline)
    _, index, _dataset = load_category_index(
        "op-latency", repo=repo, revision=sha, transport=transport, cache=cache, offline=offline
    )
    selections = [
        Selection(category="op-latency", hardware=hardware)
        for hardware in sorted({e.identity.hardware for e in index.entries})
    ]
    common = {"repo": repo, "revision": sha, "transport": transport, "cache": cache}
    if not offline:
        # Thousands of small files: plan every hardware, enforce one budget, then download concurrently.
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            plans = list(pool.map(lambda sel: plan(sel, **common), selections))
            todo = {f.path: f for p in plans for f in p.uncached}
            if sum(f.bytes or 0 for f in todo.values()) > max_bytes:
                raise BudgetExceededError(plans[0], max_bytes)
            list(pool.map(lambda f: _download(transport, cache, repo, sha, f), todo.values()))
    for selection in selections:  # verifies every cached file again; offline, so no transfer can happen here
        fetch(selection, repo=repo, revision=sha, transport=None, cache=cache, offline=True)
    for path in _LOOKUP_FILES:
        if cache.valid(repo, sha, path, None):
            continue
        if offline or transport is None:
            raise OfflineMissingError([path])
        cache.store(
            repo, sha, path, transport.open_file(repo, sha, path), PlannedFile(path, None, None, "catalog", False)
        )
    return cache.commit_root(repo, sha), sha
