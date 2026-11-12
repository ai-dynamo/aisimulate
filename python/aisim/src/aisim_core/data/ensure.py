# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make the dataset a resolved configuration needs available before a predictor is built.

:func:`ensure_data` is the hidden loading step of DEP #435 ("Loading is hidden
behind predict/estimate"). For a resolved ``best_available`` configuration it
derives the catalog selection (``system``, ``backend``, ``backend_version``,
``model``), then resolves, plans, fetches, verifies and loads the data through
the category loader, and returns the configuration with the loaded inputs wired
in plus the dataset revision that was used.

Source order, per category named by the :class:`DataPin`:

1. local override (``DataPin.local_root``, a dataset-layout directory; no network),
2. cached pinned Hugging Face revision (a warm cache makes zero network calls;
   a cold cache downloads when the uncached size is within ``max_bytes``),
3. packaged data: only for categories the pin does not name. A category the pin
   names is never served from packaged data; absent or invalid data is an error.

With no pin the whole step is a no-op and behavior is unchanged. Credentials are
the standard Hugging Face ones (``HF_TOKEN`` / ``huggingface-cli login``); this
module accepts, stores and prints no token.

Environment (read by :func:`pin_from_environment`): ``AISIM_DATASET=<repo>@<revision>``
(use a full commit SHA for offline reuse), ``AISIM_DATA_LOCAL``, ``AISIM_DATA_CATEGORIES``
(default ``fpm``), ``AISIM_DATA_CONFIGURATION``, ``AISIM_DATA_MAX_BYTES``,
``AISIM_DATA_OFFLINE`` and ``AISIM_DATA_CACHE``.
"""

from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aisim_core.data import CategoryIndex, DatasetIndex, Entry, _file, _verify
from aisim_core.data.fetch import (
    DATASET_INDEX_PATH,
    BudgetExceededError,
    Cache,
    DataFetchError,
    FetchResult,
    OfflineMissingError,
    RevisionError,
    Selection,
    SelectionError,
    Transport,
    fetch,
)
from aisim_core.data.loaders import LOADERS, FpmData, LoaderError, OpLatencyData
from aisim_core.data.repos import resolve_ref

DEFAULT_MAX_BYTES = 256 * 1024 * 1024
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_ensuring: contextvars.ContextVar[bool] = contextvars.ContextVar("aisim_ensuring_data", default=False)


# --------------------------------------------------------------------------- errors


class MissingDataError(DataFetchError):
    """The pinned source has no usable data for this configuration."""

    code = "missing-data"

    def __init__(self, message: str, *, selection: Selection | None = None, source: str | None = None):
        super().__init__(message)
        self.selection = selection
        self.source = source

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "source": self.source,
            "selection": None if self.selection is None else {k: v for k, v in vars(self.selection).items() if v},
        }


class DataTooLargeError(DataFetchError):
    """Auto-download is refused above the size threshold."""

    code = "data-too-large"

    def __init__(self, uncached_bytes: int, max_bytes: int, command: str):
        super().__init__(
            f"dataset needs {uncached_bytes} uncached bytes, above the auto-download threshold of {max_bytes} "
            f"bytes (AISIM_DATA_MAX_BYTES). Nothing was downloaded. To proceed, run: {command}"
        )
        self.uncached_bytes, self.max_bytes, self.command = uncached_bytes, max_bytes, command

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "uncached_bytes": self.uncached_bytes,
            "max_bytes": self.max_bytes,
            "command": self.command,
        }


class DataConflictError(DataFetchError):
    """Dataset content contradicts an explicit user setting or is ambiguous."""

    code = "data-conflict"


# --------------------------------------------------------------------------- pin and result


@dataclass(frozen=True)
class DataPin:
    """Where the data comes from. No token field: auth is the standard HF environment."""

    repo: str | None = None
    revision: str | None = None  # full commit SHA recommended: it resolves with no network
    local_root: Path | None = None
    categories: tuple[str, ...] = ("fpm",)
    configuration_id: str | None = None  # disambiguates several catalog matches
    max_bytes: int = DEFAULT_MAX_BYTES
    offline: bool = False
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.local_root is None and not (self.repo and self.revision):
            raise ValueError("a data pin needs local_root or both repo and revision")
        unknown = [c for c in self.categories if c not in LOADERS or c == "e2e"]
        if unknown:
            raise ValueError(f"categories not loadable into a predictor: {unknown}")


@dataclass
class EnsuredData:
    """The configuration with dataset inputs wired in, and where they came from."""

    config: dict[str, Any]
    datasets: dict[str, dict[str, Any]] = field(default_factory=dict)  # category -> provenance

    def to_dict(self) -> dict[str, Any]:
        return {"datasets": self.datasets}

    @property
    def revisions(self) -> dict[str, str]:
        return {c: d["revision"] for c, d in self.datasets.items()}


def pin_from_environment(env: Mapping[str, str] | None = None) -> DataPin | None:
    """Build a pin from ``AISIM_DATA*`` variables; ``None`` when none is set (the default)."""
    env = os.environ if env is None else env
    spec, local = env.get("AISIM_DATASET"), env.get("AISIM_DATA_LOCAL")
    if not spec and not local:
        return None
    repo = revision = None
    if spec:
        repo, sep, revision = spec.partition("@")
        if not sep or not repo or not revision:
            raise ValueError("AISIM_DATASET must be <owner>/<name>@<revision>")
        # A symbolic ref on the retired FPM repository follows its successor; a commit SHA never moves.
        resolved = resolve_ref(repo, revision)
        repo, revision = resolved.repo, resolved.revision
    kwargs: dict[str, Any] = {}
    if env.get("AISIM_DATA_CATEGORIES"):
        kwargs["categories"] = tuple(c.strip() for c in env["AISIM_DATA_CATEGORIES"].split(",") if c.strip())
    if env.get("AISIM_DATA_MAX_BYTES"):
        kwargs["max_bytes"] = int(env["AISIM_DATA_MAX_BYTES"])
    if env.get("AISIM_DATA_CACHE"):
        kwargs["cache_dir"] = Path(env["AISIM_DATA_CACHE"])
    return DataPin(
        repo=repo,
        revision=revision,
        local_root=Path(local) if local else None,
        configuration_id=env.get("AISIM_DATA_CONFIGURATION") or None,
        offline=env.get("AISIM_DATA_OFFLINE", "").lower() in {"1", "true", "yes"},
        **kwargs,
    )


@contextmanager
def _guard():
    token = _ensuring.set(True)
    try:
        yield
    finally:
        _ensuring.reset(token)


def ensuring() -> bool:
    """True while :func:`ensure_data` is running (its own predictor checks must not recurse)."""
    return _ensuring.get()


# --------------------------------------------------------------------------- catalog lookup


def _matching_entries(root: Path, selection: Selection) -> list[Entry]:
    """Entries of the category index under ``root`` that match ``selection`` (files already verified)."""
    try:
        dataset = DatasetIndex.model_validate_json(_file(root, DATASET_INDEX_PATH).read_bytes())
        reference = dataset.categories.get(selection.category)
        if reference is None:
            return []
        index = CategoryIndex.model_validate_json(_verify(root, reference.index).read_bytes())
    except (ValueError, ValidationError) as exc:
        raise MissingDataError(f"dataset catalog under {root} is unusable: {exc}", selection=selection) from exc
    return [e for e in index.entries if selection.matches(e)]


def _describe(selection: Selection) -> str:
    parts = {
        "system": selection.hardware,
        "backend": selection.framework,
        "backend_version": selection.framework_version,
        "model": selection.model,
    }
    return ", ".join(f"{k}={v}" for k, v in parts.items() if v is not None)


def _command(pin: DataPin, selection: Selection) -> str:
    flags = [f"--category {selection.category}"]
    for flag, value in (
        ("--model", selection.model),
        ("--hardware", selection.hardware),
        ("--framework", selection.framework),
        ("--framework-version", selection.framework_version),
        ("--repo", pin.repo),
        ("--revision", pin.revision),
    ):
        if value:
            flags.append(f"{flag} {value}")
    return "aisim data fetch " + " ".join(flags)


# --------------------------------------------------------------------------- ensure


def ensure_data(
    config: Mapping[str, Any],
    pin: DataPin | None,
    *,
    transport: Transport | None = None,
) -> EnsuredData:
    """Return ``config`` with verified dataset inputs wired in; no-op without a pin.

    Raises :class:`MissingDataError`, :class:`DataTooLargeError`,
    :class:`DataConflictError` or another :class:`DataFetchError`; never falls
    back to other data for a category the pin names.
    """
    out = EnsuredData(config=copy.deepcopy(dict(config)))
    if pin is None:
        return out
    if transport is None and pin.local_root is None and not pin.offline:
        from aisim_core.data.fetch import HfTransport

        transport = HfTransport()
    with _guard():
        for category in pin.categories:
            _ensure_category(out, category, pin, transport)
    return out


def _ensure_category(out: EnsuredData, category: str, pin: DataPin, transport: Transport | None) -> None:
    loader = LOADERS[category]()
    try:
        selection = loader.select(out.config)
    except SelectionError as exc:
        raise MissingDataError(f"cannot derive a {category} selection from the configuration: {exc}") from exc
    if pin.configuration_id:
        selection = replace(selection, configuration_ids=(pin.configuration_id,))

    if pin.local_root is not None:
        root = Path(pin.local_root).absolute()
        source, revision, downloaded = "local", f"local:{root}", 0
        source_key = "local-" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    else:
        result = _fetch_pinned(selection, pin, transport)
        root = result.root
        source = "hf-download" if result.downloaded_bytes else "hf-cache"
        revision, downloaded = result.plan.revision, result.downloaded_bytes
        source_key = f"{pin.repo.replace('/', '__')}-{revision}"

    entries = _matching_entries(root, selection)
    where = f"{pin.repo}@{revision}" if pin.local_root is None else str(root)
    if not entries:
        raise MissingDataError(
            f"no {category} data for {_describe(selection)} in {where}; there is no fallback to other data",
            selection=selection,
            source=where,
        )
    if len(entries) > 1:
        ids = ", ".join(sorted(e.configuration_id for e in entries))
        raise DataConflictError(
            f"{len(entries)} {category} entries match {_describe(selection)} in {where} ({ids}); "
            "set AISIM_DATA_CONFIGURATION (DataPin.configuration_id) to choose one"
        )
    entry = entries[0]
    cache = Cache(pin.cache_dir)
    output_dir = cache.root / "materialized" / source_key / entry.configuration_id
    try:
        loaded = _load(loader, category, root, entry, out.config, output_dir)
    except (LoaderError, ValueError) as exc:
        raise MissingDataError(
            f"{category} data {entry.configuration_id} in {where} failed verification: {exc}",
            selection=selection,
            source=where,
        ) from exc
    _wire(out.config, category, loaded, entry)
    out.datasets[category] = {
        "repo": pin.repo,
        "revision": revision,
        "source": source,
        "configuration_id": entry.configuration_id,
        "downloaded_bytes": downloaded,
    }


def _fetch_pinned(selection: Selection, pin: DataPin, transport: Transport | None) -> FetchResult:
    cache = Cache(pin.cache_dir)
    common = {"repo": pin.repo, "revision": pin.revision, "cache": cache}
    where = f"{pin.repo}@{pin.revision}"
    # Warm cache first: a full-SHA pin needs no network at all.
    try:
        return fetch(selection, transport=None, offline=True, **common)
    except OfflineMissingError as exc:
        if pin.offline:
            raise MissingDataError(
                f"offline: {where} is not fully cached for {_describe(selection)}; missing: " + ", ".join(exc.missing),
                selection=selection,
                source=where,
            ) from exc
    except RevisionError as exc:
        if pin.offline:
            raise MissingDataError(
                f"offline: {where} is not a full commit SHA, so it cannot be resolved without network: {exc}",
                selection=selection,
                source=where,
            ) from exc
    except SelectionError as exc:
        # Cached catalog exists but matches nothing: the answer is final for this revision.
        raise MissingDataError(f"{exc} ({_describe(selection)} in {where})", selection=selection, source=where) from exc
    try:
        return fetch(selection, transport=transport, max_bytes=pin.max_bytes, **common)
    except BudgetExceededError as exc:
        raise DataTooLargeError(exc.plan.uncached_bytes, pin.max_bytes, _command(pin, selection)) from exc
    except SelectionError as exc:
        raise MissingDataError(f"{exc} ({_describe(selection)} in {where})", selection=selection, source=where) from exc


def _load(loader, category: str, root: Path, entry: Entry, config: Mapping[str, Any], output_dir: Path):
    manifest_path = entry.manifest.path
    if category == "fpm":
        cached = output_dir / "config.json"
        if cached.is_file():  # materialized by an earlier verified load; config.json is written last
            return FpmData(loader.read_manifest(root, manifest_path, config), config=json.loads(cached.read_text()))
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        data = loader.load(root, manifest_path, output_dir=output_dir, config=config)
        if data.config is None:
            raise LoaderError("an fpm-profile artifact cannot be built into a best_available configuration")
        return data
    existing = output_dir / "systems"
    if existing.is_dir():
        manifest = loader.read_manifest(root, manifest_path, config)
        files = tuple(sorted(a.runtime_path for a in manifest.artifacts if a.runtime_path))
        return OpLatencyData(existing, manifest, files)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return loader.load(root, manifest_path, output_dir=output_dir, config=config)


def _wire(config: dict[str, Any], category: str, loaded: Any, entry: Entry) -> None:
    """Merge loaded inputs into ``config``, preserving explicit user choices."""
    if category == "op-latency":
        paths = [str(loaded.systems_root), *(str(p) for p in config.get("systems_paths") or ())]
        config["systems_paths"] = list(dict.fromkeys(paths))
        return
    loaded_config = loaded.config
    paths = [*(loaded_config.get("systems_paths") or ()), *(config.get("systems_paths") or ())]
    config["systems_paths"] = list(dict.fromkeys(str(p) for p in paths))
    wanted = loaded_config["estimator_config"]["op_level"]["execution_profile"]
    estimator = config.setdefault("estimator_config", {})
    op_level = estimator.setdefault("op_level", {})
    user = op_level.get("execution_profile")
    if user is None:
        op_level["execution_profile"] = wanted
    elif user.get("sha256") not in (None, wanted.get("sha256")):
        raise DataConflictError(
            f"configuration pins execution profile {user.get('sha256')} but dataset entry "
            f"{entry.configuration_id} provides {wanted.get('sha256')}"
        )
    else:
        op_level["execution_profile"] = {**user, **wanted}
