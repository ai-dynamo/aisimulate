# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ais_perf_model`` inside AISimulate config files.

``engine.ais_perf_model`` (predict and recommend configs) and
``search_space.ais_perf_model`` (search configs) are another spelling of the
estimator fields those configs already have. Lowering rewrites the raw mapping
so the existing fields carry the meaning; everything downstream keeps reading
the existing fields, and a dump never contains the key.

One entry describes one worker type. Its keys are of three kinds:

- structural: ``worker_type`` and ``from``;
- estimator keys: written into the existing estimator fields;
- identity keys (every other key): never written, only compared with what the
  config already states. A predict entry may use any field of the canonical
  perf model config; a key that is none of them is rejected when the config
  is validated.

The first half of this module holds the rules both config families share. The
second half is the engine side: ``lower_engine`` for the ``engine`` block, the
identity check the compiler runs, and the helpers that work on a whole config
mapping (``anchor_paths``, ``written_old_keys``, ``warn_old_keys``).

This module imports only the standard library when it is imported. The engine
models import it, and it sits on the import path of the command line parent,
so the loader and the config models are imported inside the functions that
need them.
"""

from __future__ import annotations

import copy
import json
import os
import shlex
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

__all__ = [
    "ESTIMATOR_KEYS",
    "KEY",
    "PERF_MODEL_FIELD_NAMES",
    "ROLE_ESTIMATOR_KEYS",
    "STRUCTURAL_KEYS",
    "TRANSFER_KINDS",
    "TRANSFER_PRESETS",
    "WORKER_TYPE_ORDER",
    "active_worker_types",
    "anchor_paths",
    "check_entry_identity",
    "identity_keys",
    "is_snapshot_entry",
    "is_written",
    "lower_engine",
    "normalize_entries",
    "old_fields_conflict",
    "plan_estimator_keys",
    "require_stated_value",
    "same_setting",
    "uncovered_config",
    "uncovered_reason",
    "warn_old_keys",
    "written_old_keys",
]

KEY = "ais_perf_model"
# Entries, active worker types and messages always use this order.
WORKER_TYPE_ORDER = ("aggregated", "prefill", "decode")
# Estimator keys that each worker type may set on its own.
ROLE_ESTIMATOR_KEYS = (
    "estimation_mode",
    "fallback_policy",
    "estimator_config",
    "systems_paths",
    "database_mode",
    "transfer_policy",
)
# ``fpm_profile`` describes the model, so a config has one value for all worker types.
ESTIMATOR_KEYS = (*ROLE_ESTIMATOR_KEYS, "fpm_profile")
STRUCTURAL_KEYS = ("worker_type", "from")
# The fields of the canonical perf model config, by name only. An entry key is checked where the
# config is validated, and that code must not import the perf model, so the names are repeated
# here. A test compares them with the canonical type.
PERF_MODEL_FIELD_NAMES = (
    "model",
    "system",
    "backend",
    "worker_type",
    "backend_version",
    "fpm_profile",
    "tp",
    "pp",
    "attention_dp",
    "dcp",
    "moe_tp_size",
    "moe_ep_size",
    "cp_size",
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "fpm_fmha_quant_mode",
    "kvcache_quant_mode",
    "comm_quant_mode",
    "nextn",
    "speculation",
    "kv_block_size",
    "decoder_replay",
    "estimation_mode",
    "database_mode",
    "transfer_policy",
    "systems_paths",
    "fallback_policy",
    "estimator_config",
    "attention_backend",
    "moe_kernel_source",
    "enable_shared_layer",
    "strict_provenance",
    "moe_backend",
    "enable_eplb",
    "wideep_num_slots",
    "encoder_parallel",
    "deployment_limits",
    "afd_routing_location",
    "afd_pool_role",
)
# The perf model fields that describe an AFD pool. An entry cannot set them yet.
_AFD_FIELD_NAMES = ("afd_routing_location", "afd_pool_role")
# The transfer kinds of the perf model and the kinds each preset stands for, by name only. An old
# ``transfer_policy`` is compared with the key where the config is validated, and that code must
# not import the perf model, so the names are repeated here. A test compares both tables, and the
# reading of a policy that uses them, with the perf model.
TRANSFER_KINDS = ("xshape", "xquant", "xprofile", "xop")
TRANSFER_PRESETS = {
    "off": (),
    "conservative": ("xshape",),
    "balanced": ("xshape", "xquant"),
    "aggressive": TRANSFER_KINDS,
}

_ENGINE_KEY = f"engine.{KEY}"
_WORKER_TYPES_BY_MODE = {"aggregated": ("aggregated",), "disaggregated": ("prefill", "decode")}
_OLD_ENGINE_KEYS = (*ROLE_ESTIMATOR_KEYS, "systems_path", "fpm_profile")
_OLD_TIMING_KEYS = (*ROLE_ESTIMATOR_KEYS, "forward_model", "fpm_parquet_path")
# Recommend states these three in fields of their own.
_RECOMMEND_HOMED_KEYS = ("model", "system", "backend")


# --- Small helpers ---------------------------------------------------------------------------


def _error(message: str) -> ValueError:
    """Build the loader's error type. The loader is imported here, not when this module is imported."""
    from aisimulate_core.ais_perf_model import AisPerfModelError

    return AisPerfModelError(message)


def _show(value: Any) -> str:
    """Render a value the way a config author writes it."""
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _is_list(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _kind_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, Mapping):
        return "a mapping"
    if _is_list(value):
        return "a list"
    if isinstance(value, (str, int, float)):
        return f"{type(value).__name__} {_show(value)}"
    return f"an object of type {type(value).__name__}"


def _same_value(key: str, left: Any, right: Any) -> bool:
    """Plain equality. Only ``database_mode`` ignores case, as the field itself does."""
    if key == "database_mode" and isinstance(left, str) and isinstance(right, str):
        return left.upper() == right.upper()
    return bool(left == right)


# --- Rules shared by engine and search configs -----------------------------------------------


def normalize_entries(value: Any, worker_types: Sequence[str], *, name: str) -> list[dict[str, Any]]:
    """Return one entry for each of ``worker_types``, in that order.

    ``value`` is the ``ais_perf_model`` value as written. ``name`` is how error
    messages call it, for example ``engine.ais_perf_model``. Each returned
    entry has its ``from: <file>`` template merged, ``worker_type`` set and
    null keys dropped.

    One config without ``worker_type`` serves every worker type. A list, or
    one config that names its worker type, must have exactly the worker types
    the config runs.
    """
    from aisimulate_core.ais_perf_model import expand_template, load

    if isinstance(value, str):
        raise _error(f"{name} cannot be a string; write the mapping or 'from: <file>'")
    if not isinstance(value, Mapping) and not _is_list(value):
        raise _error(f"{name} must be a mapping or a list of mappings, got {_kind_of(value)}")
    document = load(value)
    serves_all = document.bare and not document.worker_types
    if not serves_all:
        _require_same_worker_types(document.worker_types, worker_types, name)
    entries = [expand_template(document.select(worker_type)) for worker_type in worker_types]
    if serves_all and len(worker_types) > 1 and any(is_snapshot_entry(entry) for entry in entries):
        raise _error(
            f"{name} names a state snapshot, and a state snapshot belongs to one worker type. "
            f"This config runs {', '.join(worker_types)}; write a list with one entry per worker_type"
        )
    return entries


def _require_same_worker_types(present: Sequence[str], active: Sequence[str], name: str) -> None:
    for worker_type in present:
        if worker_type not in active:
            raise _error(
                f"{name} has an entry for worker type {worker_type}, which this config does not run; "
                f"it runs: {', '.join(active)}"
            )
    missing = [worker_type for worker_type in active if worker_type not in present]
    if missing:
        raise _error(f"{name} has no entry for worker type {', '.join(missing)}; it has: {', '.join(present)}")


def is_snapshot_entry(entry: Mapping[str, Any]) -> bool:
    """Whether the entry is timed by a state snapshot.

    That is ``from: {state_snapshot: ...}``, or an ``estimator_config`` that
    has a ``state_snapshot`` key.
    """
    source = entry.get("from")
    if isinstance(source, Mapping) and "state_snapshot" in source:
        return True
    estimator_config = entry.get("estimator_config")
    return isinstance(estimator_config, Mapping) and "state_snapshot" in estimator_config


def identity_keys(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return the keys of an entry that are compared and never written."""
    return {key: value for key, value in entry.items() if key not in ESTIMATOR_KEYS and key not in STRUCTURAL_KEYS}


def require_stated_value(who: str, key: str, value: Any, *, home: str, raw: Any, stated: Any, one: str = "it") -> None:
    """Fail unless the config states one value for the identity key ``key`` and the entry repeats it.

    ``who`` names the entry in the error. ``home`` is the dotted path of the
    field that states the value, ``raw`` is what is written there and
    ``stated`` is the one value read from it. Anything but a string means the
    field does not state one value. ``one`` is what the error asks to state.
    """
    if not isinstance(stated, str):
        raise _error(
            f"{who}: {key}={_show(value)} needs one stated value at {home} (found {_show(raw)}); "
            f"state {one} there or drop the key"
        )
    if value != stated:
        raise _error(f"{who}: {key}={_show(value)} differs from {home}={_show(stated)}")


def plan_estimator_keys(
    entries: Sequence[Mapping[str, Any]], *, name: str, config_kinds: str
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Decide where the estimator keys of the entries are written.

    Returns ``(shared, per_worker)``. ``shared`` holds the values written once
    for the whole config. ``per_worker`` maps a worker type to the values
    written for that worker type only; a worker type with nothing to write is
    left out.

    A key is shared when every entry gives it with the same value. Otherwise
    each entry that gives it writes it for its own worker type. So entries
    that agree lower to the config-wide spelling, and entries that differ
    lower to the per-worker spelling.

    A state snapshot belongs to one worker type. When any entry is a snapshot
    entry, nothing but ``fpm_profile`` is shared.

    ``entries`` come from ``normalize_entries``. ``config_kinds`` names the
    configs in the dataset error, for example ``search configs``.
    """
    values = {entry["worker_type"]: _estimator_values(entry, name=name, config_kinds=config_kinds) for entry in entries}
    shared: dict[str, Any] = {}
    profile = _agreed_fpm_profile(values, name)
    if profile is not None:
        shared["fpm_profile"] = profile
    keep_per_worker = any(is_snapshot_entry(entry) for entry in entries)
    per_worker: dict[str, dict[str, Any]] = {worker_type: {} for worker_type in values}
    for key in ROLE_ESTIMATOR_KEYS:
        given = {worker_type: found[key] for worker_type, found in values.items() if key in found}
        if not given:
            continue
        first = next(iter(given.values()))
        agreed = len(given) == len(values) and all(_same_value(key, first, other) for other in given.values())
        if agreed and not keep_per_worker:
            shared[key] = first
            continue
        for worker_type, value in given.items():
            per_worker[worker_type][key] = value
    return shared, {worker_type: found for worker_type, found in per_worker.items() if found}


def _estimator_values(entry: Mapping[str, Any], *, name: str, config_kinds: str) -> dict[str, Any]:
    """Return a copy of the estimator keys one entry gives, with a snapshot source folded in."""
    values = {key: copy.deepcopy(entry[key]) for key in ESTIMATOR_KEYS if key in entry}
    source = entry.get("from")
    if not isinstance(source, Mapping):
        return values
    if "dataset" in source:
        raise _error(
            f"dataset entries are not supported in {config_kinds} yet; "
            "use 'aisimulate perf-model check' or a Dynamo component"
        )
    # The existing fields name a snapshot inside estimator_config, next to the other estimator settings.
    who = f"{name} entry for {entry['worker_type']}"
    written = values.get("estimator_config", {})
    if not isinstance(written, Mapping):
        raise _error(f"{who}: estimator_config must be a mapping next to from.state_snapshot, got {_kind_of(written)}")
    if "state_snapshot" in written:
        raise _error(
            f"{who}: the state snapshot is written twice, as from.state_snapshot and as "
            "estimator_config.state_snapshot; keep one"
        )
    values["estimator_config"] = {**written, "state_snapshot": copy.deepcopy(source["state_snapshot"])}
    return values


def _agreed_fpm_profile(values: Mapping[str, dict[str, Any]], name: str) -> Any:
    """Take ``fpm_profile`` out of every worker type's values and return the one value they share."""
    profiles = {worker_type: found.pop("fpm_profile", None) for worker_type, found in values.items()}
    (first, profile), *others = profiles.items()
    for worker_type, other in others:
        if other == profile:
            continue
        if profile is not None and other is not None:
            problem = f"the entries for {first} and {worker_type} give different values"
        else:
            # An entry that leaves the profile out is the usual slip, and it is not a different value.
            gives, leaves_out = (first, worker_type) if other is None else (worker_type, first)
            problem = (
                f"the entry for {gives} gives it and the entry for {leaves_out} does not; "
                "every entry must give the same value"
            )
        raise _error(f"{name}: fpm_profile is one value for the whole config, but {problem}")
    return profile


def is_written(model_cls: Any, name: str, mapping: Mapping[str, Any]) -> bool:
    """Whether a raw mapping states the existing field ``name`` of the pydantic model ``model_cls``.

    A field is stated when its key is present, its value is not null, and the
    value differs from the default the model declares. A dumped config lists
    every default, and a default in a dump is not a choice the author made.

    Two legacy fields are never dumped (role-level ``forward_model`` and the
    search ``{role}_forward_model``); callers count those as stated whenever
    they are present and not null.
    """
    value = mapping.get(name)
    if value is None:
        return False
    default = model_cls.model_fields[name].get_default(call_default_factory=True)
    return not _same_value(name, value, default)


def same_setting(model_cls: Any, name: str, left: Any, right: Any) -> bool:
    """Whether two raw values mean the same thing to field ``name`` of ``model_cls``.

    Both values are read through the type the field declares, so two spellings
    the field stores the same way are equal. ``transfer_policy`` is compared by
    the transfer kinds it names: the field keeps a preset, a list and a
    comma-separated string as written, and they can name the same kinds. A
    policy that cannot be read as kinds is the same setting only as a policy
    written identically. A name that is not a field of the model is never the
    same setting.
    """
    if name not in model_cls.model_fields:
        return False
    if name == "transfer_policy":
        return bool(_transfer_kinds(left) == _transfer_kinds(right))
    return _same_value(name, _as_stored(model_cls, name, left), _as_stored(model_cls, name, right))


def _transfer_kinds(value: Any) -> Any:
    """Return the transfer kinds a ``transfer_policy`` value names, or the value as written when it cannot be read.

    The value is read without importing the perf model (see ``TRANSFER_KINDS``).
    A null allows every kind. Text is a preset, a kind, or several of them
    separated by commas, in any letter case. A list must hold kind names
    exactly as the perf model spells them. The comparison reads a list as exact
    kind names at every level; on their own, the timing of a recommend worker
    and a role of a search space read a list more loosely.

    A value that cannot be read this way comes back as written, so it is the
    same setting only as a value written the same way, never as one that works.
    A tuple comes back as a list, which is what the loader makes of a tuple in
    the key.
    """
    if value is None:
        return frozenset(TRANSFER_KINDS)
    kinds = None
    if isinstance(value, str):
        kinds = _kinds_of_text(value)
    elif isinstance(value, (list, tuple)):
        kinds = _kinds_of_list(value)
    if kinds is not None:
        return kinds
    return list(value) if isinstance(value, tuple) else value


def _kinds_of_text(text: str) -> frozenset[str] | None:
    """Read a policy written as text: a preset, a kind, or several of them separated by commas."""
    text = text.strip().lower()
    if text in TRANSFER_PRESETS:
        return frozenset(TRANSFER_PRESETS[text])
    # Only a text with a comma is split, and its empty parts are skipped: "," names no kind, "" is not a policy.
    words = [word for word in (part.strip() for part in text.split(",")) if word] if "," in text else [text]
    kinds: set[str] = set()
    for word in words:
        if word in TRANSFER_PRESETS:
            kinds.update(TRANSFER_PRESETS[word])
        elif word in TRANSFER_KINDS:
            kinds.add(word)
        else:
            return None
    return frozenset(kinds)


def _kinds_of_list(items: Sequence[Any]) -> frozenset[str] | None:
    """Read a policy written as a list: kind names only, with no presets and no change of letter case."""
    if all(isinstance(item, str) and item in TRANSFER_KINDS for item in items):
        return frozenset(items)
    return None


def _as_stored(model_cls: Any, name: str, value: Any) -> Any:
    from pydantic import TypeAdapter

    try:
        adapter = TypeAdapter(model_cls.model_fields[name].rebuild_annotation())
        return adapter.dump_python(adapter.validate_python(value), mode="json")
    except Exception:
        # The model reports a value its field cannot read. Compare such a value as written.
        return value


def old_fields_conflict(name: str, paths: Sequence[str]) -> ValueError:
    """Build the error for existing estimator fields that contradict ``name``.

    ``paths`` are the dotted paths of those fields.
    """
    return _error(
        f"{name} cannot be combined with old estimator fields that say something else: "
        f"{', '.join(paths)}. Remove the old fields."
    )


def uncovered_config(name: str, reason: str) -> ValueError:
    """Build the error for a config whose estimator settings ``name`` cannot describe yet."""
    return _error(f"{name} cannot be used in this config: {reason}. Keep the existing estimator fields.")


# --- Engine block of predict and recommend configs -------------------------------------------


def _engine_class(kind: str) -> Any:
    from aisimulate.config.engine import EnginePredictionConfig, EngineRecommendationConfig

    classes = {"predict": EnginePredictionConfig, "recommend": EngineRecommendationConfig}
    if kind not in classes:
        raise ValueError(f"unknown config kind {kind!r}; use predict or recommend")
    return classes[kind]


def _plain_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """Return the mapping written at ``where``; nothing written reads as an empty mapping."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise _error(f"{_ENGINE_KEY} needs {where} written as plain data (a mapping), got {_kind_of(value)}")


def _worker(raw_engine: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    workers = _plain_mapping(raw_engine.get("workers"), "engine.workers")
    return _plain_mapping(workers.get(role), f"engine.workers.{role}")


def _timing(raw_engine: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    return _plain_mapping(_worker(raw_engine, role).get("timing"), f"engine.workers.{role}.timing")


def _modes(raw_engine: Mapping[str, Any], kind: str) -> tuple[str, ...]:
    """Return the engine modes the raw mapping asks for."""
    engine_cls = _engine_class(kind)
    mode = raw_engine.get("mode")
    if mode is None:
        default = engine_cls.model_fields["mode"].get_default(call_default_factory=True)
        return (default,) if isinstance(default, str) else tuple(default.choices)
    if isinstance(mode, str):
        names = [mode]
    elif kind == "recommend" and isinstance(mode, Mapping) and _is_list(mode.get("choices")) and mode["choices"]:
        names = list(mode["choices"])
    else:
        shapes = "one mode name" if kind == "predict" else "one mode name or {choices: [<mode name>, ...]}"
        raise _error(f"{_ENGINE_KEY} needs engine.mode written as {shapes}, got {_kind_of(mode)}")
    for name in names:
        if not isinstance(name, str) or name not in _WORKER_TYPES_BY_MODE:
            raise _error(
                f"{_ENGINE_KEY} cannot be used with engine.mode {_show(name)}; "
                f"the modes are: {', '.join(_WORKER_TYPES_BY_MODE)}"
            )
    return tuple(names)


def active_worker_types(raw_engine: Mapping[str, Any], kind: str) -> tuple[str, ...]:
    """Return the worker types a raw engine mapping runs, in ``WORKER_TYPE_ORDER``.

    ``kind`` is ``predict`` or ``recommend``. Aggregated mode runs
    ``aggregated``; disaggregated mode runs ``prefill`` and ``decode``. A
    recommend config that searches several modes runs the worker types of all
    of them. An engine mode that cannot be read is an ``AisPerfModelError``.
    """
    active = {worker_type for mode in _modes(raw_engine, kind) for worker_type in _WORKER_TYPES_BY_MODE[mode]}
    return tuple(worker_type for worker_type in WORKER_TYPE_ORDER if worker_type in active)


def _afd_worker_role(raw_engine: Mapping[str, Any]) -> str | None:
    """Return the first role in ``WORKER_TYPE_ORDER`` whose worker is written as an AFD worker, or None.

    The engine model reads a worker as AFD from ``type: afd``, in any of the
    three roles, whatever the mode. This reads the raw mapping the same way
    and never fails: a shape it cannot read is not an AFD worker.
    """
    workers = raw_engine.get("workers")
    if not isinstance(workers, Mapping):
        return None
    for role in WORKER_TYPE_ORDER:
        worker = workers.get(role)
        if isinstance(worker, Mapping) and worker.get("type") == "afd":
            return role
    return None


def uncovered_reason(raw_engine: Mapping[str, Any], kind: str) -> str | None:
    """Say why ``engine.ais_perf_model`` cannot describe this engine mapping, or return None.

    Not covered yet: an AFD worker, an encoder pool, and a worker whose
    timing is not the default type. Such a config keeps the existing
    estimator fields, and those fields are not reported as old. One AFD
    worker is enough: the ordinary worker next to it keeps its fields too. An
    engine mapping that cannot be read is an ``AisPerfModelError``.
    """
    afd_role = _afd_worker_role(raw_engine)
    if afd_role is not None:
        return f"engine.workers.{afd_role}.type is afd, and {KEY} does not cover AFD yet"
    worker_types = active_worker_types(raw_engine, kind)
    workers = _plain_mapping(raw_engine.get("workers"), "engine.workers")
    if workers.get("encoder") is not None:
        return f"engine.workers.encoder is set, and {KEY} does not cover an encoder pool yet"
    for role in worker_types:
        timing_type = _timing(raw_engine, role).get("type")
        if timing_type is not None and timing_type != "default":
            return f"engine.workers.{role}.timing.type is {_show(timing_type)}, and {KEY} covers only default timing"
    return None


def lower_engine(raw_engine: Any, kind: str) -> Any:
    """Rewrite ``engine.ais_perf_model`` into the existing estimator fields.

    ``raw_engine`` is the engine mapping as written and ``kind`` is
    ``predict`` or ``recommend``. Input that is not a mapping, or has no
    ``ais_perf_model`` (absent or null), is returned as it is. Otherwise the
    result is a new mapping and the input is not changed:

    - the estimator keys of the entries are written to ``engine.<key>`` or to
      ``engine.workers.<role>.timing.<key>`` (see ``plan_estimator_keys``);
    - ``ais_perf_model`` becomes the list of normalized entries, one per
      active worker type, so the identity check can read it later.

    The identity keys are judged here as far as the raw mapping allows. A
    recommend entry is compared with what the engine section states. A
    predict entry is only checked for keys that no perf model field has; the
    compiler compares the values (``check_entry_identity``).

    Lowering its own result gives the same result. The only file read is a
    ``from: <file>`` template. A problem is an ``AisPerfModelError``.
    """
    if not isinstance(raw_engine, Mapping) or raw_engine.get(KEY) is None:
        return raw_engine
    reason = uncovered_reason(raw_engine, kind)
    if reason is not None:
        raise uncovered_config(_ENGINE_KEY, reason)
    worker_types = active_worker_types(raw_engine, kind)
    entries = normalize_entries(raw_engine[KEY], worker_types, name=_ENGINE_KEY)
    for entry in entries:
        if kind == "recommend":
            _check_recommend_identity(entry, raw_engine)
        else:
            _require_perf_model_fields(entry)
            _reject_afd_fields(entry)
    shared, per_worker = plan_estimator_keys(entries, name=_ENGINE_KEY, config_kinds="predict and recommend configs")
    conflicts = [
        old.path for old in _written_old(raw_engine, kind, worker_types) if not _restates(old, kind, shared, per_worker)
    ]
    if conflicts:
        raise old_fields_conflict(_ENGINE_KEY, conflicts)
    lowered = {**raw_engine, **shared, KEY: entries}
    if per_worker:
        lowered["workers"] = _with_worker_timing(raw_engine, per_worker)
    return lowered


def _with_worker_timing(raw_engine: Mapping[str, Any], per_worker: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Return a copy of ``engine.workers`` with the per-worker values added to each worker's timing."""
    workers = dict(_plain_mapping(raw_engine.get("workers"), "engine.workers"))
    for role, values in per_worker.items():
        if workers.get(role) is None:
            # Never invent a worker. The engine model reports the missing worker in its own words.
            continue
        workers[role] = {**_worker(raw_engine, role), "timing": {**_timing(raw_engine, role), **values}}
    return workers


class _OldKey(NamedTuple):
    """One existing estimator field the caller stated."""

    path: str
    role: str | None
    name: str
    value: Any


def _written_old(raw_engine: Mapping[str, Any], kind: str, worker_types: Sequence[str]) -> list[_OldKey]:
    """Return the existing estimator fields the engine mapping states, engine level first."""
    from aisimulate.config.engine import TimingConfig

    engine_cls = _engine_class(kind)
    found = [
        _OldKey(f"engine.{name}", None, name, raw_engine[name])
        for name in _OLD_ENGINE_KEYS
        if is_written(engine_cls, name, raw_engine)
    ]
    for role in worker_types:
        timing = _timing(raw_engine, role)
        found.extend(
            _OldKey(f"engine.workers.{role}.timing.{name}", role, name, timing[name])
            for name in _OLD_TIMING_KEYS
            # forward_model is never dumped, and op_level there pins the estimator, so any value is a choice.
            if (timing.get(name) is not None if name == "forward_model" else is_written(TimingConfig, name, timing))
        )
    return found


def _restates(old: _OldKey, kind: str, shared: Mapping[str, Any], per_worker: Mapping[str, dict[str, Any]]) -> bool:
    """Whether a stated old field only repeats what lowering writes to that same field.

    That is the case for a dumped config with the unified key added back, and
    for a mapping that was already lowered.
    """
    from aisimulate.config.engine import TimingConfig

    if old.role is None:
        return old.name in shared and same_setting(_engine_class(kind), old.name, old.value, shared[old.name])
    values = per_worker.get(old.role, {})
    return old.name in values and same_setting(TimingConfig, old.name, old.value, values[old.name])


# --- Identity keys ---------------------------------------------------------------------------


def _single_name(value: Any) -> Any:
    """Return the name inside ``{choices: [<one name>]}``; any other value is returned as it is."""
    choices = value.get("choices") if isinstance(value, Mapping) else None
    return choices[0] if _is_list(choices) and len(choices) == 1 else value


def _check_recommend_identity(entry: Mapping[str, Any], raw_engine: Mapping[str, Any]) -> None:
    """Compare the identity keys of a recommend entry with what the engine mapping states.

    Recommend chooses most identity values for each candidate, so an entry
    can only repeat what the engine block states once for all of them. This
    runs when the config is validated and must not import the SDK.
    """
    from aisimulate.config.common import ENGINE_MODEL_CONTROL_FIELDS

    keys = identity_keys(entry)
    who = f"{_ENGINE_KEY} entry for {entry['worker_type']}"
    stated_once = (*ENGINE_MODEL_CONTROL_FIELDS, "backend_version", "nextn")
    unchecked = [key for key in keys if key not in _RECOMMEND_HOMED_KEYS and key not in stated_once]
    if unchecked:
        raise _error(
            f"{who}: {', '.join(unchecked)} cannot be checked in a recommend config. "
            "The recommend engine section does not state them (recommend chooses tp and similar values "
            "for each candidate). Drop these keys from the entry, or from the template file when "
            "'from: <file>' brings them in."
        )
    homes = _recommend_homes(raw_engine, entry["worker_type"])
    for key in _RECOMMEND_HOMED_KEYS:
        if key not in keys:
            continue
        home, raw, stated = homes[key]
        if key == "system" and stated == "auto":
            # ``auto`` leaves the hardware to recommend, so it states no system.
            stated = None
        require_stated_value(who, key, keys[key], home=home, raw=raw, stated=stated)
    for key in stated_once:
        if key in keys:
            _check_stated_once(key, keys[key], raw_engine, who, snapshot=is_snapshot_entry(entry))


def _recommend_homes(raw_engine: Mapping[str, Any], role: str) -> dict[str, tuple[str, Any, Any]]:
    """Return, for each key with a field of its own, ``(where it is stated, the raw value, the one value)``."""
    hardware = _worker(raw_engine, role).get("hardware")
    backend = raw_engine.get("backend")
    return {
        "model": ("engine.model", raw_engine.get("model"), raw_engine.get("model")),
        "system": (f"engine.workers.{role}.hardware", hardware, hardware)
        if hardware is not None
        else ("engine.hardware", raw_engine.get("hardware"), raw_engine.get("hardware")),
        "backend": ("engine.backend", backend, _single_name(backend)),
    }


def _check_stated_once(key: str, value: Any, raw_engine: Mapping[str, Any], who: str, *, snapshot: bool) -> None:
    """Compare one identity key with the engine field recommend applies to every candidate."""
    raw = raw_engine.get(key)
    stated = raw
    if key == "backend_version" and isinstance(raw, Mapping):
        # One version per backend: read the version of the one backend the config names.
        backend = _single_name(raw_engine.get("backend"))
        if not isinstance(backend, str):
            raise _error(
                f"{who}: {key}={_show(value)} needs one stated value at engine.{key} (found {_show(raw)}); "
                "state one backend at engine.backend or drop the key"
            )
        stated = raw.get(backend)
    if stated is not None:
        if value != stated:
            raise _error(f"{who}: {key}={_show(value)} differs from engine.{key}={_show(stated)}")
        return
    if snapshot:
        raise _error(
            f"{who}: {key}={_show(value)} cannot be checked because engine.{key} is not set, and a worker "
            "timed by a state snapshot takes an unset value from the snapshot; set it in the engine section "
            "or drop the key"
        )
    default = _engine_class("recommend").model_fields[key].get_default(call_default_factory=True)
    if value != default:
        raise _error(
            f"{who}: {key}={_show(value)} differs from engine.{key}, which is not set "
            f"(its default is {_show(default)}); set it in the engine section or drop the key"
        )


def _require_perf_model_fields(entry: Mapping[str, Any]) -> None:
    """Fail when a predict entry has a key that no perf model field has.

    Such a key is most often a misspelled one. It is reported when the config
    is validated, before a run starts and before any output is touched. The
    values of the keys that are fields are compared later, by the compiler.
    """
    unknown = [key for key in identity_keys(entry) if key not in PERF_MODEL_FIELD_NAMES]
    if not unknown:
        return
    problems = "; ".join(f"{key}: not a perf model field{_nearest_key_hint(key)}" for key in unknown)
    raise _error(
        f"{_ENGINE_KEY} entry for {entry['worker_type']}: {problems}. "
        "Correct the key or drop it from the entry, or from the template file when 'from: <file>' brings it in."
    )


def _reject_afd_fields(entry: Mapping[str, Any]) -> None:
    """Fail when a predict entry sets a perf model field that describes an AFD pool.

    Only the pools of an AFD worker have these fields, and a config with an
    AFD worker cannot use the key yet. So no worker an entry describes can
    match a value written here.
    """
    for key in _AFD_FIELD_NAMES:
        if key in entry:
            raise _error(
                f"{_ENGINE_KEY} entry for {entry['worker_type']}: {key} is set, and {KEY} does not cover AFD yet. "
                "Drop the key from the entry, or from the template file when 'from: <file>' brings it in."
            )


def _nearest_key_hint(key: str) -> str:
    """Return `` (did you mean <key>?)`` when ``key`` is nearly a key an entry may have, else nothing."""
    import difflib

    known = [*STRUCTURAL_KEYS, *ESTIMATOR_KEYS, *PERF_MODEL_FIELD_NAMES]
    # Only a near-identical spelling is offered: a looser match names fields the author never meant.
    nearest = difflib.get_close_matches(key, known, n=1, cutoff=0.9)
    return f" (did you mean {nearest[0]}?)" if nearest else ""


def check_entry_identity(
    entries: Sequence[Mapping[str, Any]] | None, role: str, timing_config: Mapping[str, Any]
) -> None:
    """Fail when an identity key of the ``role`` entry differs from the compiled perf model config.

    ``entries`` is ``engine.ais_perf_model`` of a validated predict config and
    ``timing_config`` is the canonical perf model config the compiler built
    for ``role`` from the engine section. An identity key the compiled config
    leaves out is compared with the default the canonical type declares.
    Estimator keys are not compared: lowering already wrote them. Without
    entries there is nothing to check.
    """
    entry = next((item for item in entries or () if isinstance(item, Mapping) and item.get("worker_type") == role), {})
    keys = identity_keys(entry)
    if not keys:
        return
    defaults = _perf_model_defaults()
    problems = []
    for key, value in keys.items():
        if key not in defaults:
            problems.append(f"{key}: not a perf model field")
            continue
        expected = timing_config[key] if key in timing_config else defaults[key]
        if value == expected:
            continue
        theirs = "does not set it" if expected is None else f"gives {_show(expected)}"
        problems.append(f"{key}: entry has {_show(value)}, the engine section {theirs}")
    if problems:
        raise _error(
            f"{_ENGINE_KEY} entry for {role} does not match the engine section: {'; '.join(problems)}. "
            "Fix the engine section or drop the key from the entry."
        )


def _perf_model_defaults() -> dict[str, Any]:
    """Return every field of the canonical perf model config with the default it declares.

    The defaults are read from the type, so there is no second copy to keep
    in step. A field without a default maps to None.
    """
    import dataclasses

    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    defaults: dict[str, Any] = {}
    for field in dataclasses.fields(ForwardPassPerfModelConfig):
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            defaults[field.name] = field.default_factory()
        else:
            defaults[field.name] = None
    return defaults


# --- Whole config mappings: paths and the deprecation warning --------------------------------


def _section(kind: str) -> str:
    """Return the top-level key of the config mapping that holds ``ais_perf_model``."""
    sections = {"predict": "engine", "recommend": "engine", "search": "search_space"}
    if kind not in sections:
        raise ValueError(f"unknown config kind {kind!r}; use predict, recommend or search")
    return sections[kind]


def anchor_paths(raw: Any, base_dir: str | os.PathLike[str], kind: str) -> Any:
    """Make the relative paths inside ``ais_perf_model`` absolute against ``base_dir``.

    ``raw`` is a whole config mapping as read from a file, ``base_dir`` is the
    folder of that file and ``kind`` is ``predict``, ``recommend`` or
    ``search``. Call this right after reading the file, before any override
    is applied: a path written in a config file starts at the file's folder,
    while a path given later starts at the working directory.

    Returns a copy with the value replaced when it is a mapping or a list.
    Any other value, a missing key and a missing section are left alone, so
    validation reports them the same way for every input. A value the loader
    rejects is an ``AisPerfModelError``.
    """
    from aisimulate_core.ais_perf_model import load

    name = _section(kind)
    section = raw.get(name) if isinstance(raw, Mapping) else None
    value = section.get(KEY) if isinstance(section, Mapping) else None
    if not isinstance(value, Mapping) and not _is_list(value):
        return raw
    return {**raw, name: {**section, KEY: load(value, base_dir=base_dir).to_data()}}


def written_old_keys(raw: Any, kind: str) -> list[str]:
    """Return the dotted paths of the old estimator fields a config mapping states.

    ``raw`` is a whole config mapping and ``kind`` is ``predict``,
    ``recommend`` or ``search``. A field counts when ``is_written`` says so.
    The result is empty for a config ``ais_perf_model`` cannot describe yet,
    and for a mapping that cannot be read: validation reports those.
    """
    from aisimulate_core.ais_perf_model import AisPerfModelError

    section = raw.get(_section(kind)) if isinstance(raw, Mapping) else None
    if not isinstance(section, Mapping):
        return []
    if kind == "search":
        # The search rules live with the search space model, which the engine configs must not import.
        from aisimulate.sweeper.ais_perf_model import written_old_keys_search

        return [f"search_space.{name}" for name in written_old_keys_search(section)]
    try:
        if uncovered_reason(section, kind) is not None:
            return []
        return [old.path for old in _written_old(section, kind, active_worker_types(section, kind))]
    except AisPerfModelError:
        return []


def _sets_key(raw: Any, kind: str) -> bool:
    """Whether a config mapping sets ``ais_perf_model``; a null does not set it."""
    section = raw.get(_section(kind)) if isinstance(raw, Mapping) else None
    return isinstance(section, Mapping) and section.get(KEY) is not None


def warn_old_keys(
    raw: Any,
    kind: str,
    *,
    source: str | os.PathLike[str] | None = None,
    file_raw: Any = None,
    stacklevel: int = 2,
) -> None:
    """Warn once when a config mapping states old estimator fields.

    ``raw`` is the config mapping that was validated. ``source`` is the config
    file, named in the hint. ``file_raw`` is the mapping as the file had it
    when overrides were applied afterwards; it defaults to ``raw``. Old fields
    in the file are fixed by converting the file; old fields that only an
    override added are fixed by overriding the new key. Old fields next to a
    key that is already set, in the run or in the file, are fixed by removing
    them.

    Call this outside validation, and only where a person wrote the input: a
    model validator must never warn. ``stacklevel`` counts as it does for
    ``warnings.warn`` called in place of this function, so the default points
    at the caller of the function that calls ``warn_old_keys``.
    """
    from aisimulate_core.ais_perf_model import warn_deprecated

    old = written_old_keys(raw, kind)
    if not old:
        return
    in_file = old if file_raw is None else written_old_keys(file_raw, kind)
    name = f"{_section(kind)}.{KEY}"
    if _sets_key(raw, kind):
        # Validation accepts old fields next to the key only when they repeat it, and migrate does not merge them.
        hint = "it is already set, so remove the old fields"
    elif in_file and _sets_key(file_raw, kind):
        # An override removed the key for this run only. migrate reads the file, and it does not convert a file
        # that sets the key.
        hint = "the file already sets it, so remove the old fields from the file"
    elif in_file:
        # Quoted for a shell, so that the command also works as written for a path with spaces.
        target = "<config file>" if source is None else shlex.quote(os.fspath(source))
        hint = f"run 'aisimulate perf-model migrate {target}' to convert the file"
    else:
        hint = f"set {name}.<key> with --set instead"
    warn_deprecated(old, f"{name} ({hint})", stacklevel=stacklevel + 1)
