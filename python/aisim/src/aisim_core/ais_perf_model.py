# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loader for the unified ``ais_perf_model`` input.

One value describes the performance model of one worker type, or a list of
them. ``load`` turns a value (inline JSON, a file, or parsed data) into plain
data with absolute paths. ``expand_template`` merges a ``from: <file>``
template. ``resolve`` returns the mapping that
``RustForwardPassPerfModel.best_available`` takes.

This module is pure Python and imports only the standard library at import
time. It checks shape, paths, ``from``, merging and fill. It does not know the
canonical field names, their values or their defaults: Rust validates the
fields when the model is built.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import json
import math
import os
import pathlib
import shutil
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEPRECATION_PREFIX",
    "ESTIMATOR_KEYS",
    "WORKER_TYPES",
    "AisPerfModelDeprecationWarning",
    "AisPerfModelDocument",
    "AisPerfModelError",
    "expand_template",
    "load",
    "resolve",
    "warn_deprecated",
]

WORKER_TYPES = ("prefill", "decode", "aggregated")
ESTIMATOR_KEYS = (
    "estimation_mode",
    "fallback_policy",
    "estimator_config",
    "systems_paths",
    "database_mode",
    "transfer_policy",
    "fpm_profile",
)
DEPRECATION_PREFIX = "ais-perf-model: "

# Path fields under ``estimator_config``. Relative values start at the base folder.
_ESTIMATOR_PATH_FIELDS = (
    ("state_snapshot", "path"),
    ("fpm_learned", "artifact_path"),
    ("fpm_learned", "checkpoint_path"),
    ("temporary_mixed_composition", "original_a", "manifest_path"),
    ("op_level", "execution_profile", "path"),
    ("fpm_interpolation", "fpm_parquet_path"),
)
# An empty value of these keys means the key is not written.
_EMPTY_IS_ABSENT = ("systems_paths", "estimator_config")
# A saved identity fact the producer did not know. A caller's fill is not compared with it.
_UNKNOWN = "unknown"
# The one key of a state snapshot that ``from.state_snapshot`` may set.
_ON_TUNE = "estimator_config.state_snapshot.on_tune"
_VALUE = "ais_perf_model"
_ENTRY = "ais_perf_model entry"
_MAX_DEPTH = 64
# A config holds far fewer values. The limit stops YAML aliases that expand to millions of copies.
_MAX_VALUES = 1_000_000
_BOM = "\ufeff"
# No system takes a longer path. Longer text (inline JSON with a mistake, YAML given as the value)
# is not a path, and following its folders one by one costs the square of its length.
_LONGEST_PATH = 32_768


class AisPerfModelError(ValueError):
    """The ``ais_perf_model`` value is wrong. The message says what to fix."""

    def __init__(self, *args: object) -> None:
        # A path can hold text that cannot be encoded. Escape it so the message can always be printed.
        super().__init__(
            *(arg.encode("utf-8", "backslashreplace").decode("utf-8") if isinstance(arg, str) else arg for arg in args)
        )


class AisPerfModelDeprecationWarning(FutureWarning):
    """An old performance-model field was used. Callers can filter on this category."""


def warn_deprecated(old: str | Sequence[str], new: str, *, stacklevel: int = 2) -> None:
    """Warn that ``old`` is deprecated and name its replacement ``new``.

    ``old`` is one field name or several; one warning names all of them.

    This is the only place that emits ``AisPerfModelDeprecationWarning``.
    ``stacklevel`` counts as it does for ``warnings.warn`` called in place of
    this function: 1 is the line that calls ``warn_deprecated``, 2 is the
    caller of that function.
    """
    names = [old] if isinstance(old, str) else list(old)
    if not names:
        raise ValueError("warn_deprecated needs at least one old field name")
    subject = f"{', '.join(names)} {'is' if len(names) == 1 else 'are'}"
    warnings.warn(
        f"{DEPRECATION_PREFIX}{subject} deprecated and will be removed in the next release; use {new}",
        AisPerfModelDeprecationWarning,
        stacklevel=stacklevel + 1,
    )


@dataclass(frozen=True)
class AisPerfModelDocument:
    """A loaded ``ais_perf_model`` value.

    ``entries`` holds the configs as written: relative paths are absolute and
    null keys are dropped, but ``from`` is not followed. ``bare`` is true when
    the value was one config and not a list. ``source`` is the file path, or
    ``None`` for inline JSON and parsed data.
    """

    entries: tuple[dict[str, Any], ...]
    bare: bool
    source: str | None = None

    @property
    def worker_types(self) -> tuple[str, ...]:
        """Worker types the document names, in document order."""
        return tuple(entry["worker_type"] for entry in self.entries if "worker_type" in entry)

    def select(self, worker_type: str) -> dict[str, Any]:
        """Return a deep copy of the entry for ``worker_type``.

        A bare config without ``worker_type`` serves every worker type and
        gets ``worker_type`` set in the copy.
        """
        _require_worker_type(worker_type, "the requested worker type")
        for entry in self.entries:
            written = entry.get("worker_type")
            if written == worker_type or (written is None and self.bare):
                return {"worker_type": worker_type, **copy.deepcopy(entry)}
        raise AisPerfModelError(
            f"{self.source or _VALUE}: no entry for worker type {worker_type!r}; it has: {', '.join(self.worker_types)}"
        )

    def to_data(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Return the value in the shape it was written, with absolute paths."""
        data = copy.deepcopy(list(self.entries))
        return data[0] if self.bare else data


def load(
    value: str | os.PathLike[str] | Mapping[str, Any] | Sequence[Any],
    *,
    base_dir: str | os.PathLike[str] | None = None,
) -> AisPerfModelDocument:
    """Load an ``ais_perf_model`` value.

    A string that starts with ``{`` or ``[`` is inline JSON. Any other string
    or path is a file: ``.json`` is read as JSON, everything else as YAML. To
    read a file whose name starts with ``{`` or ``[``, write ``./NAME`` or pass
    a path object. A mapping or a list is parsed data. Relative paths inside a
    file start at the file's folder. For inline JSON and parsed data they
    start at ``base_dir``, which defaults to the current directory. A relative
    file path also starts at ``base_dir``. A UTF-8 byte order mark at the
    start of a file or of inline JSON is ignored.

    Symbolic links in the folders of a path are followed before ``..`` is
    applied, as the system does when it opens the path. A file reached through
    a linked folder therefore starts its relative paths at the real folder.
    The last name of a path is kept as written, so a file that is itself a
    link starts its relative paths at the folder of the link. ``base_dir`` is
    a folder: a link at its end is followed too.
    """
    base = _directory(base_dir)
    if isinstance(value, str):
        text = value.removeprefix(_BOM)
        if text.lstrip()[:1] in ("{", "["):
            label = f"{_VALUE} (inline JSON)"
            try:
                data = _parse_json(text, label)
            except AisPerfModelError as exc:
                path = _absolute(value, base)
                if not os.path.isfile(path):
                    raise
                raise AisPerfModelError(
                    f"{exc}; a value that starts with {{ or [ is read as inline JSON. "
                    f"To read the file {path}, write ./{value}"
                ) from exc.__cause__
            return _document(data, base, label, None)
    if isinstance(value, (str, os.PathLike)):
        name = os.fsdecode(value)
        if not name.strip():
            raise AisPerfModelError(f"{_VALUE}: the value is empty; write a file path or inline JSON")
        source = _absolute(name, base)
        return _document(_read_file(source), os.path.dirname(source), source, source)
    return _document(value, base, _VALUE, None)


def expand_template(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Merge the template file named by ``from: <file>`` into ``entry``.

    The file holds one config, or a list from which the config with the same
    ``worker_type`` is taken. One config without ``worker_type`` serves every
    worker type. One config that writes ``worker_type`` is taken only by an
    entry of that worker type; an entry without ``worker_type`` takes the
    template's. Keys of the entry win over keys of the template.
    ``estimator_config`` is merged mapping by mapping; lists and scalars are
    replaced. The result has no ``from``. An entry without ``from``, or with a
    dataset or state snapshot ``from``, is returned unchanged.
    """
    return _expand(_config(entry, os.getcwd(), _ENTRY))


def resolve(
    entry: Mapping[str, Any],
    *,
    worker_type: str | None = None,
    fill: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
    workspace: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the canonical mapping for one entry, ready for ``best_available``.

    The result has no ``from``, no null keys at the top level and only
    absolute paths. Pass the entry as written (from
    ``AisPerfModelDocument.select``), not an already resolved mapping.

    ``worker_type`` is set when the entry does not write one and must match
    when it does; after that it counts as written by the entry. ``fill`` holds
    values the caller knows: a missing key is added and a written key must be
    equal. ``defaults`` only add what is still missing.

    A dataset or a state snapshot is sealed: the result is its saved
    ``config.json``. Nothing is added to it, written keys (``worker_type``
    too) and ``fill`` are only compared, and ``defaults`` are ignored.
    Relative paths in the saved ``config.json`` start at its folder, and
    ``from`` in it is an error. A dataset is prepared once under ``workspace``
    and reused after that. ``from.state_snapshot.on_tune`` sets
    ``estimator_config.state_snapshot.on_tune``; without it the saved value
    (``reject``) stays, and the restored model cannot be tuned.

    A null below the top level (inside ``estimator_config``) is a value, not a
    missing key: it is kept, compared, and it stops ``defaults`` from filling
    that key. Rust gives some nulls a meaning and rejects others. ``fill`` and
    ``defaults`` match keys by their exact spelling. Use the canonical names
    (``tp``, not ``tp_size``): Rust rejects a config that has both spellings.
    """
    base = os.getcwd()
    config = _config(entry, base, _ENTRY)
    if worker_type is not None:
        _require_worker_type(worker_type, "the requested worker type")
        # From here on the requested worker type counts as written by the entry.
        written = config.setdefault("worker_type", worker_type)
        if written != worker_type:
            raise AisPerfModelError(
                f"{_ENTRY}: worker_type is {written!r} but worker type {worker_type!r} was requested"
            )
    who = _who(config)
    known = _fragment(fill, base, f"{who}: fill")
    fallback = _fragment(defaults, base, f"{who}: defaults")
    if worker_type is not None and known.get("worker_type", worker_type) != worker_type:
        raise AisPerfModelError(
            f"{who}: fill has worker_type {_show(known['worker_type'])} but worker type {worker_type!r} was requested"
        )
    source = config.get("from")
    if isinstance(source, dict):
        return _resolve_sealed(config, source, known, workspace, who)
    result = _expand(config)
    for key, value in known.items():
        if key not in result:
            result[key] = value
        elif not _equal(result[key], value):
            raise AisPerfModelError(
                f"{who}: {key} is {_show(result[key])} but the caller requires {_show(value)}; "
                f"write the same value or remove {key}"
            )
    for key, value in fallback.items():
        if key not in result:
            result[key] = value
        elif key == "estimator_config" and isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _merge(value, result[key])
    return result


# --- Reading and parsing ---------------------------------------------------------------------


def _real(path: str) -> str:
    """Make ``path`` absolute and follow its links the way the system does when it opens the path."""
    plain = os.path.abspath(path)
    if len(path) > _LONGEST_PATH:
        return plain
    try:
        real = os.path.realpath(path)
    except (OSError, ValueError, RecursionError):
        # The system cannot take the path (a NUL character, text that cannot be encoded) or
        # its links cannot be followed. Reading the path reports that.
        return plain
    # realpath drops one of two leading slashes. A path without a link stays as abspath writes it.
    return plain if real.lstrip(os.sep) == plain.lstrip(os.sep) else real


def _directory(value: str | os.PathLike[str] | None) -> str:
    if value is None:
        return os.getcwd()
    return _real(os.path.expanduser(os.fsdecode(value)))


def _absolute(path: str, base: str) -> str:
    """Join ``path`` to ``base``. Links in its folders are followed; its last name stays as written.

    So ``link/..`` is the parent of the folder the link names, as it is for the
    system, and a file that is a link keeps the folder of the link.
    """
    joined = os.path.join(base, os.path.expanduser(path))
    folder, name = os.path.split(joined)
    while name in ("", os.curdir) and folder != joined:
        # The path ends with a separator or ".": its last name is the one before.
        joined = folder
        folder, name = os.path.split(joined)
    if name in ("", os.curdir, os.pardir):
        return _real(joined)
    return os.path.join(_real(folder), name)


def _read_text(path: str) -> str:
    try:
        # utf-8-sig drops a byte order mark, which some editors put at the start of a file.
        with open(path, encoding="utf-8-sig") as handle:
            return handle.read()
    except (OSError, ValueError) as exc:
        # ValueError covers bytes that are not UTF-8 and a path the system cannot take (a NUL character).
        raise AisPerfModelError(f"cannot read {path}: {exc}") from exc


def _read_file(path: str) -> Any:
    text = _read_text(path)
    if path.lower().endswith(".json"):
        return _parse_json(text, path)
    return _parse_yaml(text, path)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a finite number")


def _parse_json(text: str, label: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as exc:
        raise AisPerfModelError(f"{label}: invalid JSON: {exc}") from exc


@functools.cache
def _yaml() -> tuple[Any, type]:
    """Import yaml and build the loader on first use, so importing this module does not import yaml."""
    import yaml

    class StrictLoader(yaml.SafeLoader):
        """A safe loader that rejects a key written twice in one mapping."""

        def compose_mapping_node(self, anchor: Any) -> Any:
            # Check the keys as written, before merge keys (<<) are expanded.
            node = super().compose_mapping_node(anchor)
            seen = set()
            for key_node, _ in node.value:
                if not isinstance(key_node, yaml.ScalarNode) or key_node.tag == "tag:yaml.org,2002:merge":
                    continue
                key = (key_node.tag, key_node.value)
                if key in seen:
                    raise yaml.MarkedYAMLError(
                        "while reading a mapping",
                        node.start_mark,
                        f"found duplicate key {key_node.value!r}",
                        key_node.start_mark,
                    )
                seen.add(key)
            return node

    return yaml, StrictLoader


def _parse_yaml(text: str, path: str) -> Any:
    try:
        yaml, strict_loader = _yaml()
    except ImportError as exc:
        raise AisPerfModelError(
            f"{path}: reading a YAML file needs the pyyaml package ({exc}); install pyyaml or use a .json file"
        ) from exc
    loader = None
    try:
        loader = strict_loader(text)
        loader.name = path
        return loader.get_single_data()
    except (yaml.YAMLError, ValueError, RecursionError) as exc:
        raise AisPerfModelError(f"{path}: invalid YAML: {exc}") from exc
    finally:
        if loader is not None:
            loader.dispose()


# --- Shape, nulls and paths ------------------------------------------------------------------


def _is_list(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "a mapping"
    if _is_list(value):
        return "a list"
    return f"{type(value).__name__} {_show(value)}"


def _show(value: Any) -> str:
    """Render a value the way a config author writes it."""
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _who(config: Mapping[str, Any]) -> str:
    worker_type = config.get("worker_type")
    return f"{_ENTRY} for worker type {worker_type!r}" if worker_type is not None else _ENTRY


def _require_worker_type(value: Any, name: str) -> None:
    if not isinstance(value, str) or value not in WORKER_TYPES:
        raise AisPerfModelError(f"{name} is {_show(value)}; use one of: {', '.join(WORKER_TYPES)}")


class _Budget:
    """Counts the values copied for one config. A YAML alias is counted every time it is used."""

    def __init__(self, where: str) -> None:
        self.where = where
        self.left = _MAX_VALUES

    def spend(self) -> None:
        self.left -= 1
        if self.left < 0:
            raise AisPerfModelError(
                f"{self.where}: the config has more than {_MAX_VALUES} values; "
                "write the values out and do not nest YAML aliases"
            )


def _plain(value: Any, where: str, budget: _Budget, depth: int = 0) -> Any:
    """Copy parsed data into dicts and lists. Reject what JSON cannot carry."""
    budget.spend()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AisPerfModelError(f"{where} is {value}; only finite numbers are allowed")
        return value
    if depth > _MAX_DEPTH:
        raise AisPerfModelError(f"{where} is nested more than {_MAX_DEPTH} levels deep")
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise AisPerfModelError(f"{where}: key {key!r} is not a string; quote it")
            result[key] = _plain(item, f"{where}.{key}", budget, depth + 1)
        return result
    if _is_list(value):
        return [_plain(item, f"{where}[{index}]", budget, depth + 1) for index, item in enumerate(value)]
    if isinstance(value, os.PathLike):
        return os.fsdecode(value)
    raise AisPerfModelError(
        f"{where} has a value of type {type(value).__name__}; "
        "use a string, a number, true or false, a list or a mapping"
    )


def _plain_keys(raw: Any, where: str) -> dict[str, Any]:
    """Copy the top level of a config and drop its null keys."""
    if not isinstance(raw, Mapping):
        raise AisPerfModelError(f"{where}: expected a mapping, got {_kind(raw)}")
    budget = _Budget(where)
    result = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise AisPerfModelError(f"{where}: key {key!r} is not a string; quote it")
        if value is not None:
            result[key] = _plain(value, f"{where}: {key}", budget)
    return result


def _config(raw: Any, base: str, where: str) -> dict[str, Any]:
    """Return one config as written: plain data, no null keys, absolute paths."""
    config = _plain_keys(raw, where)
    if "worker_type" in config:
        _require_worker_type(config["worker_type"], f"{where}: worker_type")
    if "from" in config:
        config["from"] = _source(config["from"], base, f"{where}: from")
    _anchor(config, base, where)
    if not config:
        raise AisPerfModelError(f"{where}: the config is empty; write at least one key")
    return config


def _fragment(raw: Mapping[str, Any] | None, base: str, where: str) -> dict[str, Any]:
    """Normalize ``fill`` or ``defaults`` the same way as a config. It may be empty."""
    if raw is None:
        return {}
    fragment = _plain_keys(raw, where)
    if "from" in fragment:
        raise AisPerfModelError(f"{where} may not contain from; only an entry can name a source")
    _anchor(fragment, base, where)
    return fragment


def _anchor(config: dict[str, Any], base: str, where: str) -> None:
    """Make the path fields absolute and drop an empty systems_paths or estimator_config."""
    if "systems_paths" in config:
        paths = config["systems_paths"]
        if not isinstance(paths, list):
            raise AisPerfModelError(
                f"{where}: systems_paths must be a list of directories (or the word default), got {_kind(paths)}"
            )
        if paths:
            config["systems_paths"] = [_systems_path(item, base) for item in paths]
        else:
            del config["systems_paths"]
    estimator = config.get("estimator_config")
    if not isinstance(estimator, dict):
        return
    if not estimator:
        del config["estimator_config"]
        return
    for *parents, name in _ESTIMATOR_PATH_FIELDS:
        holder: Any = estimator
        for parent in parents:
            holder = holder.get(parent) if isinstance(holder, dict) else None
        if isinstance(holder, dict) and isinstance(holder.get(name), str) and holder[name].strip():
            holder[name] = _absolute(holder[name], base)


def _systems_path(item: Any, base: str) -> Any:
    # The word default names the packaged data. Rust rejects blank and non-string items.
    if not isinstance(item, str) or not item.strip() or item.lower() == "default":
        return item
    return _absolute(item, base)


def _source(value: Any, base: str, name: str) -> str | dict[str, Any]:
    """Check the shape of ``from`` and make its paths absolute."""
    shapes = "a template file path, {dataset: {root, manifest}} or {state_snapshot: {path, sha256, on_tune}}"
    if isinstance(value, str):
        if not value.strip():
            raise AisPerfModelError(f"{name} is empty; write {shapes}")
        return _absolute(value, base)
    if not isinstance(value, dict):
        raise AisPerfModelError(f"{name} must be {shapes}; got {_kind(value)}")
    written = {key: item for key, item in value.items() if item is not None}
    if len(written) != 1 or next(iter(written)) not in ("dataset", "state_snapshot"):
        raise AisPerfModelError(f"{name} must be {shapes}; got a mapping with keys: {', '.join(written) or '(none)'}")
    kind, body = next(iter(written.items()))
    name = f"{name}.{kind}"
    if not isinstance(body, dict):
        raise AisPerfModelError(f"{name} must be a mapping, got {_kind(body)}")
    body = {key: item for key, item in body.items() if item is not None}
    if kind == "dataset":
        _require_text_keys(body, ("root", "manifest"), (), name)
        if os.path.isabs(body["manifest"]):
            raise AisPerfModelError(f"{name}.manifest must be a path relative to root, got {body['manifest']!r}")
        return {"dataset": {"root": _absolute(body["root"], base), "manifest": body["manifest"]}}
    # on_tune is passed on as written. Rust checks its value when the model is built.
    _require_text_keys(body, ("path",), ("sha256", "on_tune"), name)
    return {"state_snapshot": {**body, "path": _absolute(body["path"], base)}}


def _require_text_keys(body: dict[str, Any], required: tuple[str, ...], optional: tuple[str, ...], name: str) -> None:
    allowed = (*required, *optional)
    extra = [key for key in body if key not in allowed]
    if extra:
        raise AisPerfModelError(f"{name} accepts only {', '.join(allowed)}; remove: {', '.join(extra)}")
    for key in required:
        if key not in body:
            raise AisPerfModelError(f"{name}.{key} is missing")
    for key, item in body.items():
        if not isinstance(item, str) or not item.strip():
            raise AisPerfModelError(f"{name}.{key} must be a non-empty string, got {_kind(item)}")


def _document(data: Any, base: str, label: str, source: str | None) -> AisPerfModelDocument:
    if isinstance(data, Mapping):
        return AisPerfModelDocument((_config(data, base, label),), True, source)
    if not _is_list(data):
        raise AisPerfModelError(f"{label}: expected one config (a mapping) or a list of configs, got {_kind(data)}")
    if not data:
        raise AisPerfModelError(f"{label}: the list is empty; write at least one config")
    entries: list[dict[str, Any]] = []
    positions: dict[str, int] = {}
    for position, item in enumerate(data, start=1):
        where = f"{label}: entry {position}"
        entry = _config(item, base, where)
        worker_type = entry.get("worker_type")
        if worker_type is None:
            raise AisPerfModelError(
                f"{where}: worker_type is missing; every config in a list needs one of: {', '.join(WORKER_TYPES)}"
            )
        if worker_type in positions:
            raise AisPerfModelError(
                f"{label}: entries {positions[worker_type]} and {position} both have worker_type {worker_type!r}; "
                "a worker type may appear once"
            )
        positions[worker_type] = position
        entries.append(entry)
    return AisPerfModelDocument(tuple(entries), False, source)


# --- Merging and comparing -------------------------------------------------------------------


def _merge(low: dict[str, Any], high: dict[str, Any]) -> dict[str, Any]:
    """Merge two mappings. ``high`` wins. Only mappings merge; lists and scalars are replaced."""
    result = dict(low)
    for key, value in high.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(map(_equal, left, right))
    # JSON keeps true and 1 apart. Python does not.
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return bool(left == right)


def _stated(key: str, value: Any) -> bool:
    return value is not None and not (key in _EMPTY_IS_ABSENT and value in ([], {}))


def _expand(config: dict[str, Any]) -> dict[str, Any]:
    path = config.get("from")
    if not isinstance(path, str):
        return config
    who = _who(config)
    try:
        document = load(path)
    except AisPerfModelError as exc:
        raise AisPerfModelError(f"{who}: cannot use the template named by from: {exc}") from exc
    worker_type = config.get("worker_type")
    if document.bare and (worker_type is None or "worker_type" not in document.entries[0]):
        # One config that names no worker type serves every worker type.
        template = document.entries[0]
    elif worker_type is None:
        raise AisPerfModelError(
            f"{who}: the template {path} is a list; "
            f"set worker_type on the entry to pick one of: {', '.join(document.worker_types)}"
        )
    else:
        # A template that names its worker type is taken only by an entry of that worker type.
        template = next((item for item in document.entries if item["worker_type"] == worker_type), None)
        if template is None:
            raise AisPerfModelError(
                f"{who}: the template {path} has no entry for worker type {worker_type!r}; "
                f"it has: {', '.join(document.worker_types)}"
            )
    if "from" in template:
        raise AisPerfModelError(f"{who}: the template {path} contains from; a template may not name another source")
    merged = copy.deepcopy(template)
    for key, value in config.items():
        if key == "from":
            continue
        if key == "estimator_config" and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# --- Sealed sources: dataset and state snapshot ----------------------------------------------


def _resolve_sealed(
    config: dict[str, Any],
    source: dict[str, Any],
    known: dict[str, Any],
    workspace: str | os.PathLike[str] | None,
    who: str,
) -> dict[str, Any]:
    if "dataset" in source:
        sealed, origin = _dataset_config(source["dataset"], workspace, who)
        locations: tuple[str, ...] = ()
    else:
        sealed, origin = _snapshot_config(source["state_snapshot"], who)
        # Where the performance data lives is a location, not part of the saved identity.
        locations = ("systems_paths",)
    for key, value in config.items():
        if key == "from":
            continue
        if key in locations:
            sealed[key] = value
            continue
        _require_sealed(key, value, sealed.get(key), who, origin, written=True)
    for key, value in known.items():
        if key not in locations:
            _require_sealed(key, value, sealed.get(key), who, origin, written=False)
    return sealed


def _require_sealed(field: str, value: Any, saved: Any, who: str, origin: str, *, written: bool) -> None:
    """Compare one value with the sealed config.

    A written key must be stated by the sealed config and be equal. A fill key
    is compared only when the sealed config states a known value.
    ``estimator_config`` is compared key by key, so a partial mapping is fine.
    """
    top = field.partition(".")[0]
    if value is None and saved is None:
        # Inside estimator_config a null is a value. Null equals null and a missing key.
        return
    if not _stated(field, saved) or (not written and saved == _UNKNOWN):
        if not written:
            return
        if field == "worker_type":
            raise AisPerfModelError(
                f"{who}: {origin} does not state a worker type; it cannot be used for worker type {_show(value)}"
            )
        raise AisPerfModelError(
            f"{who}: {field} is written but {origin} does not state it; "
            f"a config taken from a dataset or a state snapshot cannot be extended. Remove {field}"
        )
    if top == "estimator_config" and isinstance(value, dict) and isinstance(saved, dict):
        for key, item in value.items():
            if key in saved or written:
                _require_sealed(f"{field}.{key}", item, saved.get(key), who, origin, written=written)
        return
    if _equal(value, saved):
        return
    if field == "worker_type":
        # The worker type often comes from select() or the caller, so do not say it was written.
        raise AisPerfModelError(f"{who}: {origin} is for worker type {_show(saved)}, not {_show(value)}")
    if field == _ON_TUNE:
        wanted = f"{field} is {_show(value)}" if written else f"the caller requires {field} {_show(value)}"
        raise AisPerfModelError(
            f"{who}: {wanted} but {origin} is loaded with on_tune {_show(saved)}; "
            "set from.state_snapshot.on_tune to change it"
        )
    if written:
        raise AisPerfModelError(
            f"{who}: {field} is {_show(value)} but {origin} has {_show(saved)}; "
            f"a config taken from a dataset or a state snapshot cannot be changed. "
            f"Remove {field} or write the same value"
        )
    raise AisPerfModelError(f"{who}: {origin} has {field} {_show(saved)} but the caller requires {_show(value)}")


def _saved_config(path: str, who: str, form: str) -> dict[str, Any]:
    """Read the ``config.json`` of a sealed source like a config: no null keys, absolute paths.

    Relative paths in it start at its folder. Data that is not a mapping gives
    an empty config; the caller reports what is missing.
    """
    try:
        raw = _parse_json(_read_text(path), path)
        config = _plain_keys(raw, path) if isinstance(raw, dict) else {}
    except AisPerfModelError as exc:
        raise AisPerfModelError(f"{who}: {form}: {exc}") from exc
    if "from" in config:
        raise AisPerfModelError(f"{who}: {path} contains from; a saved config may not name another source")
    _anchor(config, os.path.dirname(path), f"{who}: {path}")
    return config


def _snapshot_config(source: dict[str, Any], who: str) -> tuple[dict[str, Any], str]:
    """Read the config a state snapshot saved and point it at the snapshot's folder."""
    directory = source["path"]
    path = os.path.join(directory, "config.json")
    config = _saved_config(path, who, "from.state_snapshot")
    estimator = config.get("estimator_config")
    saved = estimator.get("state_snapshot") if isinstance(estimator, dict) else None
    if not isinstance(saved, dict):
        raise AisPerfModelError(
            f"{who}: {path} is not the config of a state snapshot (it has no estimator_config.state_snapshot); "
            "from.state_snapshot.path must be a folder written by save_state_snapshot"
        )
    saved["path"] = directory
    if "on_tune" in source:
        # Whether the restored model may be tuned is the user's choice, not part of the saved identity.
        saved["on_tune"] = source["on_tune"]
    wanted = source.get("sha256")
    recorded = saved.get("sha256")
    if wanted is not None and (not isinstance(recorded, str) or recorded.lower() != wanted.lower()):
        raise AisPerfModelError(
            f"{who}: from.state_snapshot.sha256 is {_show(wanted)} but {path} records {_show(recorded)}"
        )
    return config, f"the state snapshot {path}"


def _dataset_config(
    source: dict[str, Any], workspace: str | os.PathLike[str] | None, who: str
) -> tuple[dict[str, Any], str]:
    """Prepare a dataset once under ``workspace`` and return its config."""
    root, manifest = source["root"], source["manifest"]
    manifest_file = os.path.join(root, manifest)
    if workspace is None or not os.fsdecode(workspace).strip():
        # An empty workspace would mean the current directory. Nobody means that by an unset value.
        raise AisPerfModelError(
            f"{who}: from.dataset needs a workspace folder to prepare the dataset {manifest_file} in; "
            "the caller did not give one (resolve(..., workspace=DIR))"
        )
    digest = hashlib.sha256()
    try:
        with open(manifest_file, "rb") as handle:
            manifest_bytes = handle.read()
        for part in (os.fsencode(root), os.fsencode(manifest), manifest_bytes):
            digest.update(part)
            digest.update(b"\0")
    except (OSError, ValueError) as exc:
        # ValueError covers a path the system cannot take: a NUL character or text that cannot be encoded.
        raise AisPerfModelError(f"{who}: from.dataset: cannot read the manifest {manifest_file}: {exc}") from exc
    workspace = _directory(workspace)
    target = os.path.join(workspace, digest.hexdigest()[:16])
    path = os.path.join(target, "config.json")
    if not os.path.isfile(path):
        _prepare_dataset(root, manifest, workspace, target, who)
    config = _saved_config(path, who, "from.dataset")
    estimator = config.get("estimator_config")
    op_level = estimator.get("op_level") if isinstance(estimator, dict) else None
    profile = op_level.get("execution_profile") if isinstance(op_level, dict) else None
    if not isinstance(profile, dict):
        raise AisPerfModelError(
            f"{who}: {path} is not the config of a prepared dataset "
            f"(it has no estimator_config.op_level.execution_profile); remove {target} and try again"
        )
    # config.json keeps the paths of the folder it was prepared in. Point them at the final folder.
    config["systems_paths"] = [os.path.join(target, "systems")]
    profile["path"] = os.path.join(target, "profile.json")
    return config, f"the dataset {manifest_file}"


def _prepare_dataset(root: str, manifest: str, workspace: str, target: str, who: str) -> None:
    """Prepare into a fresh sibling folder, then move it to ``target`` in one step."""
    # prepare_execution_profile needs an output folder that does not exist yet.
    scratch = os.path.join(workspace, f".tmp-{os.path.basename(target)}-{uuid.uuid4().hex}")
    try:
        os.makedirs(workspace, exist_ok=True)
        # Imported here: this module loads pyarrow, and preparing loads the native library.
        from aisim_core import datasets

        datasets.prepare_execution_profile(pathlib.Path(root), manifest, pathlib.Path(scratch))
    except BaseException as exc:
        # Also on Ctrl-C: a half-prepared folder is never reused, so do not leave it behind.
        _discard(scratch)
        if not isinstance(exc, Exception):
            raise
        raise AisPerfModelError(
            f"{who}: from.dataset: cannot prepare the dataset {os.path.join(root, manifest)} in {workspace}: {exc}"
        ) from exc
    try:
        os.replace(scratch, target)
    except OSError as exc:
        # Another process prepared the same dataset first. Use its folder.
        _discard(scratch)
        if not os.path.isfile(os.path.join(target, "config.json")):
            raise AisPerfModelError(
                f"{who}: from.dataset: cannot move the prepared dataset to {target}: {exc}; "
                f"remove {target} and try again"
            ) from exc


def _discard(folder: str) -> None:
    """Remove a folder if it is there. Never raise."""
    # rmtree raises ValueError for a path with a NUL character, even with ignore_errors.
    with contextlib.suppress(OSError, ValueError):
        shutil.rmtree(folder, ignore_errors=True)
