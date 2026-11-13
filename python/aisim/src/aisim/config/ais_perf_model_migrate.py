# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Move the old estimator fields of a config file into ``ais_perf_model``.

``migrate_mapping`` takes a predict, recommend or search config as read from
its file and returns the same config with every old estimator field replaced
by ``engine.ais_perf_model`` or ``search_space.ais_perf_model``.

The conversion is all or nothing. When one old field cannot be moved, nothing
is moved and ``CannotConvert`` lists every such field. Before a result is
returned, the original and the converted config are both read through the
config model and the estimator settings of every worker type are compared. A
difference is a ``MigrationCheckError``, never a config that quietly means
something else.

Which fields count as old, which configs the key cannot describe and how the
key is read back are the rules of ``aisim.config.ais_perf_model`` and
``aisim.sweeper.ais_perf_model``. This module adds where the two config
families keep those fields and what the legacy spellings stand for.

Importing this module imports the rules module, with it the loader, and the
standard library only. The config models are imported inside the functions
that need them.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from .ais_perf_model import (
    ESTIMATOR_KEYS,
    KEY,
    ROLE_ESTIMATOR_KEYS,
    active_worker_types,
    anchor_paths,
    is_snapshot_entry,
    same_setting,
    uncovered_reason,
    written_old_keys,
)

__all__ = ["CannotConvert", "MigrationCheckError", "migrate_mapping"]

# What the legacy ``forward_model`` field of a worker selects.
_MODE_BY_FORWARD_MODEL = {"op_level": "op_level", "fpm": "fpm_interpolation"}
# The search space calls the aggregated worker ``agg`` in its field names.
_SEARCH_ROLES = {"aggregated": "agg", "prefill": "prefill", "decode": "decode"}
_PARQUET_TARGET = "estimator_config.fpm_interpolation.fpm_parquet_path"
# A config holds far fewer values. The limit stops YAML aliases of aliases, which a copy writes out at every use.
_MAX_VALUES = 1_000_000


class CannotConvert(ValueError):
    """Some old estimator fields cannot be moved, so nothing was converted.

    ``problems`` holds one ``cannot convert: ...`` line for each reason.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("\n".join(self.problems))


class MigrationCheckError(RuntimeError):
    """The converted config does not mean what the original means.

    The conversion checks its own result before returning it. This error is a
    bug in the conversion, not a problem with the config.
    """


class _OldKey(NamedTuple):
    """One old estimator field the config states.

    ``path`` is the dotted path from the top of the config and ``location``
    is the keys to follow from the section that holds the field.
    ``worker_type`` is None for a field that holds for the whole config.
    ``name`` is the field name without a role prefix, so the search field
    ``agg_forward_model`` is ``forward_model``.
    """

    path: str
    worker_type: str | None
    name: str
    location: tuple[str, ...]
    value: Any


@dataclass
class _Report:
    """What the conversion tells the caller: notes, and the fields it cannot move."""

    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def cannot_convert(self, path: str, reason: str) -> None:
        self.problems.append(f"cannot convert: {path}: {reason}")


# --- Where each config family keeps its estimator fields -------------------------------------


class _EngineLayout:
    """The estimator fields of the ``engine`` section of a predict or recommend config."""

    section = "engine"
    # ``engine.workers.<role>`` stays in the file even when moving a field leaves it empty.
    kept_depth = 2

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def model(self) -> Any:
        from .engine import EnginePredictionConfig, EngineRecommendationConfig

        return EngineRecommendationConfig if self.kind == "recommend" else EnginePredictionConfig

    def uncovered_reason(self, section: Mapping[str, Any]) -> str | None:
        return uncovered_reason(section, self.kind)

    def worker_types(self, section: Mapping[str, Any]) -> tuple[str, ...]:
        return active_worker_types(section, self.kind)

    def role_location(self, worker_type: str) -> tuple[str, ...]:
        return ("workers", worker_type, "timing")

    def old_key(self, path: str, section: Mapping[str, Any]) -> _OldKey:
        """Read ``engine.<name>`` or ``engine.workers.<role>.timing.<name>``."""
        location = tuple(path.split(".")[1:])
        worker_type = location[1] if len(location) > 1 else None
        return _OldKey(path, worker_type, location[-1], location, _value_at(section, location))

    def role_overrides(self, role_fields: Mapping[str, Any]) -> dict[str, Any]:
        """A worker's timing replaces the engine value with every value that is not null."""
        return {name: role_fields[name] for name in ROLE_ESTIMATOR_KEYS if role_fields.get(name) is not None}

    def forward_model_settings(self, mode: str) -> dict[str, Any]:
        """The engine model reads a worker's ``forward_model`` as a mode that never falls back."""
        return {"estimation_mode": mode, "fallback_policy": "deny"}

    def settings(self, engine: Any, worker_type: str) -> dict[str, Any]:
        """Return the estimator settings a validated engine gives one worker type."""
        timing = getattr(engine.workers, worker_type).timing
        estimator_config = timing.estimator_config if timing.estimator_config is not None else engine.estimator_config
        return {
            "estimation_mode": timing.estimation_mode or engine.estimation_mode,
            "fallback_policy": timing.fallback_policy or engine.fallback_policy,
            "estimator_config": _with_parquet_path(estimator_config, timing.fpm_parquet_path),
            "systems_paths": timing.systems_paths or engine.systems_paths,
            "database_mode": timing.database_mode or engine.database_mode,
            "transfer_policy": timing.transfer_policy if timing.transfer_policy is not None else engine.transfer_policy,
            "fpm_profile": None if engine.fpm_profile is None else engine.fpm_profile.model_dump(mode="json"),
        }


class _SearchLayout:
    """The estimator fields of the ``search_space`` section of a search config."""

    section = "search_space"
    kind = "search"
    # An emptied ``role_estimator_controls.<role>`` and an emptied ``role_estimator_controls`` are removed.
    kept_depth = 0

    def model(self) -> Any:
        from ..sweeper.config import SearchSpace

        return SearchSpace

    def uncovered_reason(self, section: Mapping[str, Any]) -> str | None:
        from ..sweeper.ais_perf_model import uncovered_reason_search

        return uncovered_reason_search(section)

    def worker_types(self, section: Mapping[str, Any]) -> tuple[str, ...]:
        from ..sweeper.ais_perf_model import active_worker_types_search

        return active_worker_types_search(section)

    def role_location(self, worker_type: str) -> tuple[str, ...]:
        return ("role_estimator_controls", _SEARCH_ROLES[worker_type])

    def old_key(self, path: str, section: Mapping[str, Any]) -> _OldKey:
        """Read a top-level field, a ``role_estimator_controls`` key or a ``<role>_...`` legacy field."""
        location = tuple(path.split(".")[1:])
        value = _value_at(section, location)
        for worker_type, role in _SEARCH_ROLES.items():
            if location[:-1] == self.role_location(worker_type):
                return _OldKey(path, worker_type, location[-1], location, value)
            for name in ("forward_model", "fpm_parquet_path"):
                if location == (f"{role}_{name}",):
                    return _OldKey(path, worker_type, name, location, value)
        return _OldKey(path, None, location[-1], location, value)

    def role_overrides(self, role_fields: Mapping[str, Any]) -> dict[str, Any]:
        """A role's settings replace the top-level value with every key they have, a null one included."""
        return {name: role_fields[name] for name in ROLE_ESTIMATOR_KEYS if name in role_fields}

    def forward_model_settings(self, mode: str) -> dict[str, Any]:
        """The search space reads ``<role>_forward_model`` as the mode alone."""
        return {"estimation_mode": mode}

    def settings(self, space: Any, worker_type: str) -> dict[str, Any]:
        """Return the estimator settings a validated search space gives one worker type."""
        role = _SEARCH_ROLES[worker_type]
        controls = space.role_estimator_controls.get(role, {})
        own = {
            name: controls.get(name, getattr(space, name))
            for name in ("estimation_mode", "fallback_policy", "estimator_config", "database_mode", "transfer_policy")
        }
        return {
            **own,
            "estimator_config": _with_parquet_path(own["estimator_config"], getattr(space, f"{role}_fpm_parquet_path")),
            "systems_paths": space.systems_paths_for(role),
            "fpm_profile": space.fpm_profile,
        }


_Layout = _EngineLayout | _SearchLayout


def _layout(kind: str) -> _Layout:
    if kind == "search":
        return _SearchLayout()
    if kind in ("predict", "recommend"):
        return _EngineLayout(kind)
    raise ValueError(f"unknown config kind {kind!r}; use predict, recommend or search")


# --- Small helpers ---------------------------------------------------------------------------


def _show(value: Any) -> str:
    """Render a value the way a config author writes it."""
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _is_list(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


class _Budget:
    """Counts the values of one copy: every mapping, list, scalar and set member, each time the copy reaches it."""

    def __init__(self) -> None:
        self.left = _MAX_VALUES

    def spend(self) -> None:
        self.left -= 1
        if self.left < 0:
            raise ValueError(
                f"the config has more than {_MAX_VALUES} values; a value that a YAML alias names counts "
                "every time it is used. Write the values out and do not nest YAML aliases"
            )


def _plain_copy(value: Any) -> Any:
    """Copy mappings and lists into dicts and lists that share nothing.

    A YAML file can name one mapping in two places (an anchor and an alias).
    The copy has two separate mappings, so a field can be removed from each,
    and a dump of the copy needs no anchors.

    Aliases of aliases multiply: a few lines can name more values than a
    copy can hold. The copy counts its values and stops at ``_MAX_VALUES``
    with a ``ValueError``.

    An alias can also name the mapping or list it is written in. Such a value
    contains itself and has no copy; it is a ``ValueError`` that names the
    place.
    """
    return _copied(value, _Budget(), (), frozenset())


def _copied(value: Any, budget: _Budget, path: tuple[Any, ...], inside: frozenset[int]) -> Any:
    """Copy ``value``; ``path`` holds the keys that lead to it and ``inside`` the mappings and lists around it."""
    budget.spend()
    if not isinstance(value, Mapping) and not _is_list(value):
        if isinstance(value, (set, frozenset)):
            # A YAML set is copied whole, so each of its members counts.
            for _ in value:
                budget.spend()
        return copy.deepcopy(value)
    if id(value) in inside:
        where = ".".join(str(key) for key in path)
        raise ValueError(
            f"{where} contains itself: the value is the mapping or list it is written in, as a YAML alias can say. "
            "A config cannot be circular; write the value out"
        )
    inside = inside | {id(value)}
    if isinstance(value, Mapping):
        return {key: _copied(item, budget, (*path, key), inside) for key, item in value.items()}
    return [_copied(item, budget, (*path, index), inside) for index, item in enumerate(value)]


def _mapping_at(section: Mapping[str, Any], location: Sequence[str]) -> Mapping[str, Any]:
    """Return the mapping at ``location``; anything else found on the way reads as an empty mapping."""
    found: Any = section
    for key in location:
        found = found.get(key) if isinstance(found, Mapping) else None
    return found if isinstance(found, Mapping) else {}


def _value_at(section: Mapping[str, Any], location: Sequence[str]) -> Any:
    return _mapping_at(section, location[:-1])[location[-1]]


def _is_relative(path: str) -> bool:
    return not os.path.isabs(path) and not path.startswith("~")


def _real_path(path: str, cwd: str) -> str:
    return os.path.realpath(os.path.join(cwd, os.path.expanduser(path)))


def _in_key_order(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return estimator settings in the one order every entry is written in."""
    return {name: settings[name] for name in ESTIMATOR_KEYS if name in settings}


def _with_parquet_path(estimator_config: Mapping[str, Any], parquet_path: str | None) -> dict[str, Any]:
    """Return a copy of ``estimator_config`` that names a worker's legacy ``fpm_parquet_path`` file."""
    result = copy.deepcopy(dict(estimator_config))
    if parquet_path is not None:
        result.setdefault("fpm_interpolation", {})["fpm_parquet_path"] = parquet_path
    return result


# --- Building the new block ------------------------------------------------------------------


def _fold_systems_path(
    layout: _Layout, stated: Mapping[str, _OldKey], settings: dict[str, Any], report: _Report
) -> None:
    """Write the legacy ``systems_path`` as a one-item ``systems_paths``."""
    legacy = stated["systems_path"]
    directory = legacy.value
    if isinstance(directory, str) and directory.lower() == "default":
        # systems_path always names a directory. In systems_paths the bare word means the packaged data.
        directory = os.path.join(os.curdir, directory)
    plural = stated.get("systems_paths")
    if plural is None:
        settings["systems_paths"] = [directory]
        report.notes.append(f"{legacy.path} written as systems_paths: {_show([directory])}")
    elif same_setting(layout.model(), "systems_paths", plural.value, [directory]):
        report.notes.append(f"{legacy.path} dropped: {plural.path} names the same directory")
    else:
        report.cannot_convert(legacy.path, f"{plural.path} names other directories; keep one of the two fields")


def _shared_settings(layout: _Layout, old_keys: Sequence[_OldKey], report: _Report) -> dict[str, Any]:
    """Return the settings the old config-wide fields give every worker type."""
    stated = {old.name: old for old in old_keys if old.worker_type is None}
    settings = {name: _plain_copy(old.value) for name, old in stated.items() if name != "systems_path"}
    if "systems_path" in stated:
        _fold_systems_path(layout, stated, settings, report)
    return settings


def _forward_model_settings(
    layout: _Layout, old: _OldKey | None, role_fields: Mapping[str, Any], report: _Report
) -> dict[str, Any]:
    """Return what a worker's legacy ``forward_model`` field adds to the worker's settings.

    The config models decide this by the keys written next to the field: a
    written ``estimation_mode`` makes them ignore ``forward_model``, and a
    written ``fallback_policy`` is kept.
    """
    if old is None:
        return {}
    if "estimation_mode" in role_fields:
        report.notes.append(
            f"{old.path} dropped: estimation_mode is written next to it, and that is the field the config uses today"
        )
        return {}
    mode = _MODE_BY_FORWARD_MODEL.get(old.value) if isinstance(old.value, str) else None
    if mode is None:
        # The config model rejects any other value, and reading the original config reports it.
        return {}
    added = {name: value for name, value in layout.forward_model_settings(mode).items() if name not in role_fields}
    written = " and ".join(f"{name}: {value}" for name, value in added.items())
    report.notes.append(f"{old.path}: {old.value} written as {written}")
    return added


def _fold_parquet_path(settings: dict[str, Any], old: _OldKey, report: _Report) -> None:
    """Write a worker's legacy ``fpm_parquet_path`` where ``estimator_config`` names that file."""
    estimator_config = settings.setdefault("estimator_config", {})
    interpolation = estimator_config.setdefault("fpm_interpolation", {}) if isinstance(estimator_config, dict) else None
    if not isinstance(interpolation, dict):
        report.cannot_convert(old.path, f"the estimator_config of this worker has no mapping to hold {_PARQUET_TARGET}")
    elif "fpm_parquet_path" in interpolation and interpolation["fpm_parquet_path"] != old.value:
        other = _show(interpolation["fpm_parquet_path"])
        report.cannot_convert(old.path, f"{_PARQUET_TARGET} already names another file, {other}")
    else:
        interpolation["fpm_parquet_path"] = old.value
        report.notes.append(f"{old.path} written as {_PARQUET_TARGET}")


def _worker_settings(
    layout: _Layout,
    section: Mapping[str, Any],
    worker_type: str,
    shared: Mapping[str, Any],
    old_keys: Sequence[_OldKey],
    report: _Report,
) -> dict[str, Any]:
    """Return the estimator settings one worker type has today, as the keys of its entry."""
    own = {old.name: old for old in old_keys if old.worker_type == worker_type}
    role_fields = _mapping_at(section, layout.role_location(worker_type))
    overrides = layout.role_overrides(role_fields)
    overrides.update(_forward_model_settings(layout, own.get("forward_model"), role_fields, report))
    settings = _plain_copy(shared)
    for name, value in overrides.items():
        if value is None:
            # A null of this worker blanks the config-wide value; an entry says that by leaving the key out.
            settings.pop(name, None)
        else:
            settings[name] = _plain_copy(value)
    if "fpm_parquet_path" in own:
        _fold_parquet_path(settings, own["fpm_parquet_path"], report)
    if settings.get("estimator_config") == {}:
        # An empty mapping of this worker blanks the config-wide value in the same way.
        del settings["estimator_config"]
    return settings


def _new_block(
    layout: _Layout, section: Mapping[str, Any], old_keys: Sequence[_OldKey], report: _Report
) -> dict[str, Any] | list[dict[str, Any]]:
    """Return the ``ais_perf_model`` value that stands for the old fields.

    One config without ``worker_type`` when every old field holds for the
    whole config and no worker type differs from it. Otherwise a list with
    one entry per worker type the config runs.
    """
    worker_types = layout.worker_types(section)
    for old in old_keys:
        if old.worker_type is not None and old.worker_type not in worker_types:
            report.cannot_convert(
                old.path,
                f"this config runs no {old.worker_type} worker, so {KEY} has no entry for the field; "
                "remove the field from the file",
            )
    shared = _shared_settings(layout, old_keys, report)
    by_worker_type = {
        worker_type: _worker_settings(layout, section, worker_type, shared, old_keys, report)
        for worker_type in worker_types
    }
    config_wide = all(old.worker_type is None for old in old_keys)
    if config_wide and all(settings == shared for settings in by_worker_type.values()):
        return _in_key_order(shared)
    return [{"worker_type": worker_type, **_in_key_order(settings)} for worker_type, settings in by_worker_type.items()]


# --- Relative paths --------------------------------------------------------------------------


def _rewrite_paths(block: Any, name: str, cwd: str, config_dirs: Sequence[str], notes: list[str]) -> Any:
    """Return ``block`` with its relative paths rewritten for the folders the config file has a name in.

    The old fields start a relative path at the working directory;
    ``ais_perf_model`` starts it at the folder of the config file. The loader
    knows which values are paths: they are the strings it makes absolute.

    The loader also refuses values the old fields accept, a YAML date for
    example. The config states such a value, so it is reported as a field
    that cannot be converted. ``name`` is the dotted name of the new key.
    """
    from aisim_core.ais_perf_model import AisPerfModelError, load

    try:
        anchored = load(block, base_dir=cwd).to_data()
    except AisPerfModelError as exc:
        reason = str(exc).removeprefix(f"{KEY}: ")
        raise CannotConvert([f"cannot convert: {name} cannot hold a value of the old fields: {reason}"]) from exc
    return _rewritten(block, anchored, cwd, config_dirs, notes)


def _rewritten(written: Any, anchored: Any, cwd: str, config_dirs: Sequence[str], notes: list[str]) -> Any:
    if isinstance(written, dict) and isinstance(anchored, dict):
        # The loader drops keys that say nothing (an empty list of paths); those stay as written.
        return {
            key: _rewritten(value, anchored.get(key, value), cwd, config_dirs, notes) for key, value in written.items()
        }
    if isinstance(written, list) and isinstance(anchored, list) and len(written) == len(anchored):
        return [
            _rewritten(value, other, cwd, config_dirs, notes) for value, other in zip(written, anchored, strict=True)
        ]
    if isinstance(written, str) and isinstance(anchored, str) and written != anchored and _is_relative(written):
        return _for_config_file(written, cwd, config_dirs, notes)
    return written


def _for_config_file(path: str, cwd: str, config_dirs: Sequence[str], notes: list[str]) -> str:
    """Return a relative path of the old fields as the converted config writes it, and note the change."""
    rewritten_as = f"path {path!r} (relative to the working directory) rewritten as"
    if len(config_dirs) > 1:
        # A relative path starts at the folder of the name the config is read by, so none is right for every name.
        rewritten = _real_path(path, cwd)
        folders = f"{', '.join(config_dirs[:-1])} and {config_dirs[-1]}"
        note = (
            f"{rewritten_as} {rewritten!r} (an absolute path: the converted config has names in {folders}, "
            "and a relative path would start at a different folder for each name)"
        )
    else:
        rewritten = os.path.relpath(_real_path(path, cwd), config_dirs[0])
        if rewritten.lower() == "default":
            # In systems_paths the bare word names the packaged data, not a folder next to the config file.
            rewritten = os.path.join(os.curdir, rewritten)
        note = f"{rewritten_as} {rewritten!r} (relative to the folder of the converted config, {config_dirs[0]})"
    # A config-wide path is copied into every entry; it is reported once.
    if rewritten != path and note not in notes:
        notes.append(note)
    return rewritten


# --- Checking the result ---------------------------------------------------------------------


def _systems_paths_at(paths: Any, cwd: str) -> Any:
    if not _is_list(paths):
        return paths
    return [os.path.join(cwd, item) if _is_relative_directory(item) else item for item in paths]


def _is_relative_directory(item: Any) -> bool:
    """Whether a ``systems_paths`` item is a relative directory. An empty string and ``default`` name none."""
    return isinstance(item, str) and bool(item) and item.lower() != "default" and _is_relative(item)


def _old_paths_at(layout: _Layout, section: Mapping[str, Any], worker_types: Sequence[str], cwd: str) -> dict[str, Any]:
    """Return a copy of a section whose old relative systems paths start at ``cwd``.

    The config models start such a path at the working directory of this
    process, which is not always the directory the caller converts for.
    """
    result = _plain_copy(section)
    holders = [result, *(_mapping_at(result, layout.role_location(worker_type)) for worker_type in worker_types)]
    for holder in holders:
        if "systems_paths" in holder:
            holder["systems_paths"] = _systems_paths_at(holder["systems_paths"], cwd)
    legacy = result.get("systems_path")
    # An empty string names no directory; it stays as written for the config model to report.
    if isinstance(legacy, str) and legacy and _is_relative(legacy):
        result["systems_path"] = os.path.join(cwd, legacy)
    return result


def _settings_by_worker_type(
    layout: _Layout, section: Mapping[str, Any], worker_types: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Read a section through its config model and return every worker type's estimator settings."""
    from aisim_core.sdk.common import resolve_transfer_policy

    try:
        validated = layout.model().model_validate(section)
    except TypeError as exc:
        # A config model can stop on a value of the wrong type with a TypeError; the callers report a ValueError.
        raise ValueError(str(exc)) from exc
    settings = {worker_type: layout.settings(validated, worker_type) for worker_type in worker_types}
    for found in settings.values():
        # A preset name and the list of kinds it stands for are the same policy.
        found["transfer_policy"] = sorted(kind.value for kind in resolve_transfer_policy(found["transfer_policy"]))
    return settings


def _same(left: Any, right: Any, cwd: str) -> bool:
    """Whether two validated values are equal. Two strings also match when they name the same file."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_same(left[key], right[key], cwd) for key in left)
    if _is_list(left) and _is_list(right):
        return len(left) == len(right) and all(_same(a, b, cwd) for a, b in zip(left, right, strict=True))
    if isinstance(left, str) and isinstance(right, str):
        return left == right or _real_path(left, cwd) == _real_path(right, cwd)
    return bool(left == right)


def _check_same_settings(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]], cwd: str
) -> None:
    for worker_type, settings in before.items():
        for name, was in settings.items():
            now = after[worker_type][name]
            if not _same(was, now, cwd):
                raise MigrationCheckError(
                    f"the converted config gives the {worker_type} worker {name}={_show(now)}, "
                    f"but the original config gives {_show(was)}"
                )


# --- The conversion --------------------------------------------------------------------------


def _section_of(raw: Any, layout: _Layout) -> Mapping[str, Any]:
    section = raw.get(layout.section) if isinstance(raw, Mapping) else None
    if not isinstance(section, Mapping):
        raise ValueError(
            f"a {layout.kind} config needs a top-level {layout.section} mapping, and this config has none; "
            "check the config kind"
        )
    return section


def _converted(raw: Mapping[str, Any], layout: _Layout, old_keys: Sequence[_OldKey], block: Any) -> dict[str, Any]:
    """Return a copy of the config with the old fields removed and the new block added."""
    result = _plain_copy(raw)
    section = result[layout.section]
    for old in old_keys:
        _remove(section, old.location, layout.kept_depth)
    section[KEY] = block
    return result


def _remove(section: dict[str, Any], location: Sequence[str], kept_depth: int) -> None:
    """Delete the field at ``location``, then each parent mapping below ``kept_depth`` keys that this empties."""
    holders = [section]
    for key in location[:-1]:
        holders.append(holders[-1][key])
    del holders[-1][location[-1]]
    for depth in range(len(location) - 1, kept_depth, -1):
        if holders[depth]:
            return
        del holders[depth - 1][location[depth - 1]]


def migrate_mapping(
    raw: Mapping[str, Any],
    kind: str,
    *,
    config_dir: str | os.PathLike[str],
    cwd: str | os.PathLike[str],
    other_config_dirs: Sequence[str | os.PathLike[str]] = (),
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Move the old estimator fields of a config into ``ais_perf_model``.

    ``raw`` is a whole config mapping as read from its file and ``kind`` is
    ``predict``, ``recommend`` or ``search``. ``config_dir`` is the folder of
    the file the converted config will be read from: the folder of the file
    the result is written to, which is the original file's folder when the
    result replaces it. ``cwd`` is the working directory the config is run
    from. The old fields start a relative path at ``cwd`` and the new key
    starts it at ``config_dir``, so relative paths are rewritten, and a note
    names the folder each rewritten path starts at.

    ``other_config_dirs`` are the other folders in which that file has a
    name, as a file that is written through a symbolic link in another
    folder has. The new key starts a relative path at the folder of the name
    the file is read by, so no relative path is right for every name. Each
    relative path is then written as an absolute path, and its note says so.
    A folder that is ``config_dir`` itself changes nothing.

    Returns ``(new_raw, moved_paths, notes)``. ``new_raw`` is a copy of the
    config with the old fields removed and ``ais_perf_model`` added;
    ``moved_paths`` are the dotted paths of the removed fields; ``notes`` say
    what was rewritten on the way. A config without old fields comes back as
    a copy with nothing moved. ``raw`` is never changed.

    Raises ``CannotConvert`` when a field cannot be moved, when the key cannot
    hold a value of an old field or cannot describe the config, ``ValueError``
    when the config cannot be read, contains itself or has more than a
    million values once its YAML aliases are written out, and
    ``MigrationCheckError`` when the converted config would not mean the same.
    """
    layout = _layout(kind)
    # Every step below reads this copy. A config that contains itself, or that YAML aliases make too large
    # to write out, is refused here, before a step can loop on it.
    raw = _plain_copy(raw)
    section = _section_of(raw, layout)
    reason = layout.uncovered_reason(section)
    if reason is not None:
        raise CannotConvert([f"cannot convert: {reason}"])
    old_keys = [layout.old_key(path, section) for path in written_old_keys(raw, kind)]
    if not old_keys:
        return raw, [], []
    name = f"{layout.section}.{KEY}"
    if section.get(KEY) is not None:
        merge = f"{name} is already set, and old fields are not merged into it; remove the field or the key"
        raise CannotConvert([f"cannot convert: {old.path}: {merge}" for old in old_keys])
    report = _Report()
    block = _new_block(layout, section, old_keys, report)
    if report.problems:
        raise CannotConvert(report.problems)

    cwd = os.path.abspath(cwd)
    worker_types = layout.worker_types(section)
    try:
        before = _settings_by_worker_type(layout, _old_paths_at(layout, section, worker_types, cwd), worker_types)
    except ValueError as exc:
        raise ValueError(f"the {layout.section} section is not a valid {kind} config: {exc}") from exc
    if not _is_list(block) and len(worker_types) > 1 and is_snapshot_entry(block):
        # The config is a valid one; the key takes a state snapshot for one worker type only.
        raise CannotConvert(
            [
                f"cannot convert: {layout.section}.estimator_config.state_snapshot: the field names a state snapshot "
                f"for the whole config, and {name} takes a state snapshot for one worker type. "
                f"This config runs {', '.join(worker_types)}; write {name} as a list with one entry per worker_type"
            ]
        )

    # The config file's folder is read the way the config loaders read it: with symbolic links resolved.
    config_dirs = list(dict.fromkeys(os.path.realpath(folder) for folder in (config_dir, *other_config_dirs)))
    block = _rewrite_paths(block, name, cwd, config_dirs, report.notes)
    try:
        new_raw = _converted(raw, layout, old_keys, block)
        # The converted config is read from every folder it has a name in, and each reading is compared.
        readings = [
            _settings_by_worker_type(layout, anchor_paths(new_raw, folder, kind)[layout.section], worker_types)
            for folder in config_dirs
        ]
    except ValueError as exc:
        raise MigrationCheckError(f"the converted config is not valid: {exc}") from exc
    for after in readings:
        _check_same_settings(before, after, cwd)
    return new_raw, [old.path for old in old_keys], report.notes
