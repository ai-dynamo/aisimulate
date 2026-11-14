# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``ais_perf_model`` inside a search space.

``search_space.ais_perf_model`` is another spelling of the estimator fields a
search space already has. ``lower_search_space`` rewrites the raw mapping so
those fields carry the meaning: the search keeps reading the existing fields,
and a dump never contains the key.

The rules shared with predict and recommend configs live in
``aisim.config.ais_perf_model``. This module adds what only a search
space knows:

- which worker types the space runs (``deployment_mode``);
- where an estimator key is written: the top-level field when every worker
  type agrees, else ``role_estimator_controls[agg|prefill|decode]``;
- which keys of an entry can be compared with the space (model, system and
  backend);
- which existing estimator fields count as stated.

The search space model is imported inside the functions that need it, because
that model imports this module.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, NamedTuple

from aisim_core.ais_perf_model import AisPerfModelError

from ..config.ais_perf_model import (
    KEY,
    ROLE_ESTIMATOR_KEYS,
    WORKER_TYPE_ORDER,
    identity_keys,
    is_snapshot_entry,
    is_written,
    normalize_entries,
    old_fields_conflict,
    plan_estimator_keys,
    require_stated_value,
    same_setting,
    uncovered_config,
)

__all__ = [
    "active_worker_types_search",
    "lower_search_space",
    "uncovered_reason_search",
    "written_old_keys_search",
]

_SECTION = "search_space"
_SPACE_KEY = f"{_SECTION}.{KEY}"
_WORKER_TYPES_BY_MODE = {"agg": ("aggregated",), "disagg": ("prefill", "decode")}
# The search space calls the aggregated worker ``agg`` in its field names.
_ROLE_BY_WORKER_TYPE = {"aggregated": "agg", "prefill": "prefill", "decode": "decode"}
_ROLES = tuple(_ROLE_BY_WORKER_TYPE[worker_type] for worker_type in WORKER_TYPE_ORDER)
_OLD_TOP_LEVEL_KEYS = (*ROLE_ESTIMATOR_KEYS, "systems_path", "fpm_profile")
# The identity keys a search space states once. The search chooses every other one per candidate.
_COMPARED_KEYS = ("model", "system", "backend")
# What the search space stores for a role timed by a state snapshot, next to the snapshot reference.
_SNAPSHOT_PINNED_KEYS = ("estimation_mode", "fallback_policy")


# --- Reading the raw space mapping -----------------------------------------------------------


def _show(value: Any) -> str:
    """Render a value the way a config author writes it."""
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _modes(space: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the deployment modes the raw space searches. An unstated value is the field's default."""
    from .config import SEARCH_CHOICES, SearchSpace

    modes = space.get("deployment_mode")
    if modes is None:
        return tuple(SearchSpace.model_fields["deployment_mode"].get_default(call_default_factory=True))
    if not isinstance(modes, (list, tuple)) or not modes:
        raise AisPerfModelError(
            f"{_SPACE_KEY} needs {_SECTION}.deployment_mode written as a list with at least one mode, "
            f"got {_show(modes)}"
        )
    for mode in modes:
        if not isinstance(mode, str) or mode not in _WORKER_TYPES_BY_MODE:
            raise AisPerfModelError(
                f"{_SPACE_KEY} cannot be used with {_SECTION}.deployment_mode {_show(mode)}; "
                f"the modes are: {', '.join(SEARCH_CHOICES['deployment_mode'])}"
            )
    return tuple(modes)


def _plain_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """Return the mapping written at ``where``; nothing written reads as an empty mapping."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    raise AisPerfModelError(f"{_SPACE_KEY} needs {_SECTION}.{where} written as a mapping, got {_show(value)}")


def _role_controls(space: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    """Return the estimator settings the raw space states for one role."""
    controls = _plain_mapping(space.get("role_estimator_controls"), "role_estimator_controls")
    return _plain_mapping(controls.get(role), f"role_estimator_controls.{role}")


def active_worker_types_search(space: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the worker types a raw search space runs, in ``WORKER_TYPE_ORDER``.

    ``agg`` runs ``aggregated``; ``disagg`` runs ``prefill`` and ``decode``.
    A space that searches several modes runs the worker types of all of them,
    and a space that does not state ``deployment_mode`` searches both. A
    ``deployment_mode`` that cannot be read is an ``AisPerfModelError``.
    """
    active = {worker_type for mode in _modes(space) for worker_type in _WORKER_TYPES_BY_MODE[mode]}
    return tuple(worker_type for worker_type in WORKER_TYPE_ORDER if worker_type in active)


def uncovered_reason_search(space: Mapping[str, Any]) -> str | None:
    """Say why ``search_space.ais_perf_model`` cannot describe this search space, or return None.

    Not covered yet: a space with AFD roles, an encoder pool, and a worker
    with a timing model of its own. Such a space keeps the existing estimator
    fields, and those fields are not reported as old. One AFD role is enough:
    the ordinary role next to it keeps its fields too. A ``deployment_mode``
    that cannot be read is an ``AisPerfModelError``.
    """
    afd_roles = space.get("afd_roles")
    # An empty mapping searches no AFD role. A value that is not a mapping is left to validation.
    if isinstance(afd_roles, Mapping) and afd_roles:
        return f"{_SECTION}.afd_roles is set, and {KEY} does not cover AFD yet"
    worker_types = active_worker_types_search(space)
    if space.get("encoder") is not None:
        return f"{_SECTION}.encoder is set, and {KEY} does not cover an encoder pool yet"
    for worker_type in worker_types:
        field = f"{_ROLE_BY_WORKER_TYPE[worker_type]}_timing_model"
        if space.get(field) is not None:
            return f"{_SECTION}.{field} is set, and {KEY} covers only default timing"
    return None


# --- Existing estimator fields ---------------------------------------------------------------


class _OldKey(NamedTuple):
    """One existing estimator field the caller stated.

    ``path`` is relative to the space mapping. ``role`` is None for a
    top-level field and names the role of a ``role_estimator_controls`` key.
    """

    path: str
    role: str | None
    name: str
    value: Any


def _written_old(space: Mapping[str, Any]) -> list[_OldKey]:
    """Return the existing estimator fields the raw space states, top level first."""
    from .config import SearchSpace

    # ``systems_path`` is never dumped, so ``is_written`` counts any value of it.
    found = [
        _OldKey(name, None, name, space[name]) for name in _OLD_TOP_LEVEL_KEYS if is_written(SearchSpace, name, space)
    ]
    for role in _ROLES:
        found.extend(
            _OldKey(f"role_estimator_controls.{role}.{name}", role, name, value)
            for name, value in _role_controls(space, role).items()
            if _is_role_choice(name, value)
        )
    for role in _ROLES:
        # ``{role}_forward_model`` is not dumped for the spaces this key covers, and op_level
        # there pins the role's estimator. The parquet path has no default. Any value is a choice.
        for name in (f"{role}_forward_model", f"{role}_fpm_parquet_path"):
            if space.get(name) is not None:
                found.append(_OldKey(name, None, name, space[name]))
    return found


def _is_role_choice(name: str, value: Any) -> bool:
    """Whether a key of ``role_estimator_controls.<role>`` is a setting the author chose.

    A role key is an override without a default of its own, so any value is a
    choice and a null says nothing. ``transfer_policy`` is the exception: the
    search reads a role's null in place of the top-level policy, so there the
    key counts whenever it is present.
    """
    if name not in ROLE_ESTIMATOR_KEYS:
        return False
    return value is not None or name == "transfer_policy"


def written_old_keys_search(space: Any) -> list[str]:
    """Return the old estimator fields a raw search space mapping states.

    The names are relative to the space mapping, for example
    ``estimation_mode`` or ``role_estimator_controls.agg.estimation_mode``.
    A null counts for ``role_estimator_controls.<role>.transfer_policy`` only.
    The result is empty for a space ``ais_perf_model`` cannot describe yet and
    for a mapping that cannot be read: validation reports those.
    """
    if not isinstance(space, Mapping):
        return []
    try:
        if uncovered_reason_search(space) is not None:
            return []
        return [old.path for old in _written_old(space)]
    except AisPerfModelError:
        return []


def _restates(old: _OldKey, shared: Mapping[str, Any], per_role: Mapping[str, dict[str, Any]]) -> bool:
    """Whether a stated old field only repeats what the space stores for that field anyway.

    That is the case for a dumped space with the unified key added back, and
    for a mapping that was already lowered.
    """
    if old.role is None:
        return old.name in shared and _same_space_setting(old.name, old.value, shared[old.name])
    values = per_role.get(old.role, {})
    if old.name in values:
        return _same_role_setting(old.name, old.value, values[old.name])
    # A dumped snapshot role lists the mode and the fallback policy its snapshot pins. The
    # validators that read the snapshot reject any other value, so the key cannot contradict it.
    return old.name in _SNAPSHOT_PINNED_KEYS and is_snapshot_entry(values)


def _same_space_setting(name: str, left: Any, right: Any) -> bool:
    """Whether two raw values say the same thing in the top-level field ``name``."""
    from .config import SearchSpace

    if name == "fpm_profile":
        # The field is typed as a plain mapping, but the space stores the profile with its defaults filled in.
        return bool(_stored_profile(left) == _stored_profile(right))
    return same_setting(SearchSpace, name, left, right)


def _same_role_setting(name: str, left: Any, right: Any) -> bool:
    """Whether two raw values are stored the same way as the setting ``name`` of one role."""
    from ..config.engine import TimingConfig

    # The search space reads the settings of a role through the worker timing model. A null transfer
    # policy of a role names every kind, which is how same_setting compares that field.
    return same_setting(TimingConfig, name, left, right)


def _stored_profile(value: Any) -> Any:
    from aisim.fpm_profile import load_fpm_profile

    try:
        return load_fpm_profile(value).model_dump(mode="json")
    except (ValueError, RecursionError):
        # The search space reports a profile it cannot read. Compare such a value as written. The
        # reader says so with a ValueError, and with a RecursionError for JSON text nested too deep.
        return value


# --- Identity keys ---------------------------------------------------------------------------


def _stated_identity(space: Mapping[str, Any], role: str) -> dict[str, tuple[str, Any, Any]]:
    """Return, for each compared key, ``(where the space states it, the raw value, the one value)``."""
    hardware_field = f"{role}_hardware_sku" if space.get(f"{role}_hardware_sku") is not None else "hardware_sku"
    backend = space.get("backend")
    # A search over several backends, or over the default list, states no single backend.
    one_backend = backend[0] if isinstance(backend, (list, tuple)) and len(backend) == 1 else None
    return {
        "model": (f"{_SECTION}.model_name", space.get("model_name"), space.get("model_name")),
        "system": (f"{_SECTION}.{hardware_field}", space.get(hardware_field), space.get(hardware_field)),
        "backend": (f"{_SECTION}.backend", backend, one_backend),
    }


def _check_identity(entry: Mapping[str, Any], space: Mapping[str, Any]) -> None:
    """Compare the identity keys of an entry with what the search space states.

    The search chooses most identity values for each candidate, so an entry
    can only repeat the model, the system and the backend the space states
    once for all of them.
    """
    keys = identity_keys(entry)
    who = f"{_SPACE_KEY} entry for {entry['worker_type']}"
    unchecked = [key for key in keys if key not in _COMPARED_KEYS]
    if unchecked:
        raise AisPerfModelError(
            f"{who}: {', '.join(unchecked)} cannot be checked in a search config. "
            "The search space does not state them (the search chooses tp and similar values "
            "for each candidate). Drop these keys from the entry, or from the template file when "
            "'from: <file>' brings them in."
        )
    stated_by_key = _stated_identity(space, _ROLE_BY_WORKER_TYPE[entry["worker_type"]])
    for key, value in keys.items():
        home, raw, stated = stated_by_key[key]
        one = "one backend" if key == "backend" else "it"
        require_stated_value(who, key, value, home=home, raw=raw, stated=stated, one=one)


# --- Lowering --------------------------------------------------------------------------------


def lower_search_space(raw: Any) -> Any:
    """Rewrite ``ais_perf_model`` of a raw search space into the existing estimator fields.

    Input that is not a mapping, or has no ``ais_perf_model`` (absent or
    null), is returned as it is. Otherwise the result is a new dict and the
    input is not changed:

    - the estimator keys of the entries are written to the top-level field
      or to ``role_estimator_controls[agg|prefill|decode]`` (see
      ``plan_estimator_keys``);
    - ``ais_perf_model`` becomes the list of normalized entries, one per
      active worker type.

    Lowering its own result gives the same result. The only file read is a
    ``from: <file>`` template. A problem is an ``AisPerfModelError``.
    """
    if not isinstance(raw, Mapping) or raw.get(KEY) is None:
        return raw
    reason = uncovered_reason_search(raw)
    if reason is not None:
        raise uncovered_config(_SPACE_KEY, reason)
    entries = normalize_entries(raw[KEY], active_worker_types_search(raw), name=_SPACE_KEY)
    for entry in entries:
        _check_identity(entry, raw)
    shared, per_worker = plan_estimator_keys(entries, name=_SPACE_KEY, config_kinds="search configs")
    per_role = {_ROLE_BY_WORKER_TYPE[worker_type]: values for worker_type, values in per_worker.items()}
    conflicts = [f"{_SECTION}.{old.path}" for old in _written_old(raw) if not _restates(old, shared, per_role)]
    if conflicts:
        raise old_fields_conflict(_SPACE_KEY, conflicts)
    lowered = {**raw, **shared, KEY: entries}
    if per_role:
        lowered["role_estimator_controls"] = _with_role_values(raw, per_role)
    return lowered


def _with_role_values(space: Mapping[str, Any], per_role: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    """Return a copy of ``role_estimator_controls`` with the per-role values added to each role's settings.

    An empty mapping some callers write for a role is filled in the same way.
    """
    controls = dict(_plain_mapping(space.get("role_estimator_controls"), "role_estimator_controls"))
    for role, values in per_role.items():
        controls[role] = {**_role_controls(space, role), **values}
    return controls
