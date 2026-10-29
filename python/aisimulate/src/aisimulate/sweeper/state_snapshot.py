# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sweeper roles timed by a saved forward-pass model state snapshot.

A role whose estimator controls name only ``estimator_config.state_snapshot``
is timed by that calibrated model. A snapshot restores exactly one engine
identity, so the role is pinned to it: the search space takes the backend,
backend version, the role's parallel shape and KV block size, and the
engine-wide model controls and MTP depth from the snapshot instead of from
the caller. Values the caller states explicitly must agree. Replicas,
scheduler and memory knobs and the other roles are still searched. Core's
restoration check compares every candidate with the complete saved identity.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

from ..config.common import ENGINE_MODEL_CONTROL_FIELDS, is_active_engine_model_control

_WORKER_TYPES = {"agg": "aggregated", "prefill": "prefill", "decode": "decode"}
_PUBLIC_ROLES = {"agg": "aggregated", "prefill": "prefill", "decode": "decode"}
# Correction stores replay applies for each worker role; replay times each rank
# alone, so cross-rank stores never apply.
_REPLAY_QUERIES = {"agg": {"prefill", "decode", "mixed"}, "prefill": {"prefill"}, "decode": {"decode"}}
_REPLAY_MODES = {"op_level", "fpm_interpolation"}
_PARALLEL = (
    ("tp", "tp"),
    ("pp", "pp"),
    ("attention_dp", "attention_dp"),
    ("moe_tp", "moe_tp_size"),
    ("moe_ep", "moe_ep_size"),
)
# Estimator controls the snapshot pins and the sweeper never searches.
SNAPSHOT_CONTROL_FIELDS = (
    "estimation_mode",
    "database_mode",
    "transfer_policy",
    "estimator_config",
    "decoder_replay",
    "enable_shared_layer",
    "strict_provenance",
    "fpm_fmha_quant_mode",
)


def state_snapshot_reference(controls: Any) -> dict[str, Any] | None:
    """Return a role's ``estimator_config.state_snapshot`` entry, if it names one."""

    estimator_config = controls.get("estimator_config") if isinstance(controls, dict) else None
    reference = estimator_config.get("state_snapshot") if isinstance(estimator_config, dict) else None
    return reference


@dataclass(frozen=True)
class SnapshotRole:
    """One role's validated snapshot: its manifest summary and restoring configuration."""

    role: str
    path: str
    summary: dict[str, Any]
    config: dict[str, Any]

    @property
    def identity(self) -> dict[str, Any]:
        return self.summary["identity"]

    @property
    def packed_root(self) -> str | None:
        """The snapshot's own systems root when it was saved with ``include_perf_db``."""

        root = os.path.join(self.path, "perf_db", "systems")
        return root if self.identity.get("systems_paths") == [root] else None

    def estimator_controls(self) -> dict[str, Any]:
        """Controls a candidate request takes from the snapshot instead of the search space."""

        controls = {name: copy.deepcopy(self.config.get(name)) for name in SNAPSHOT_CONTROL_FIELDS}
        controls["fallback_policy"] = "deny"
        return controls

    def describe(self) -> str:
        return f"the {self.role} role's state snapshot {self.path}"

    def saved_spellings(self, request: dict[str, Any]) -> dict[str, Any]:
        """Spell request fields the sweeper leaves at their defaults as the snapshot saved them.

        A role pinned to its snapshot asks for the saved engine, but the
        sweeper writes an unset context-parallel degree, an unsharded MoE and
        inactive model controls one way and a saved configuration may write
        them another (``1``, ``1``/``1`` or ``"default"``). Equivalent values
        take the saved spelling; any other difference stays for restoration
        to reject.
        """

        identity, result = self.identity, {}
        for name in ("dcp", "cp_size"):
            if (request.get(name) or 1) == (identity.get(name) or 1):
                result[name] = identity.get(name)
        saved_moe = (identity.get("moe_tp_size") or 1, identity.get("moe_ep_size") or 1)
        if (request.get("moe_tp_size") or 1, request.get("moe_ep_size") or 1) == saved_moe:
            result["moe_tp_size"], result["moe_ep_size"] = identity.get("moe_tp_size"), identity.get("moe_ep_size")
        for name in ENGINE_MODEL_CONTROL_FIELDS:
            saved = identity.get(name)
            if not is_active_engine_model_control(name, request.get(name)) and not is_active_engine_model_control(
                name, saved
            ):
                result[name] = saved
        return result


def snapshot_pinned_controls(estimator_config: Any) -> dict[str, Any]:
    """Controls a saved snapshot pins that no search or prediction field sets; empty without a snapshot.

    The configuration that restores a snapshot must repeat these, so callers
    that construct it from public fields take them from the snapshot.
    """

    from aisimulate_core.sdk.rust_engine_step import _snapshot_config

    reference = state_snapshot_reference({"estimator_config": estimator_config})
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        return {}
    config = _snapshot_config(reference["path"])
    return {
        name: config.get(name)
        for name in ("decoder_replay", "enable_shared_layer", "strict_provenance", "fpm_fmha_quant_mode")
    }


def load_snapshot_role(role: str, reference: Any) -> SnapshotRole:
    """Read and validate one role's snapshot reference; raise ``ValueError`` when replay cannot use it."""

    from aisimulate_core.sdk import RustForwardPassPerfModel
    from aisimulate_core.sdk.rust_engine_step import _snapshot_config

    where = f"engine.workers.{_PUBLIC_ROLES[role]}.timing.estimator_config.state_snapshot"
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str) or not reference["path"]:
        raise ValueError(f"{where} must be a mapping with a nonempty path")
    unknown = set(reference) - {"path", "sha256"}
    if unknown:
        raise ValueError(f"{where} accepts only path and sha256; got {sorted(unknown)}")
    path = os.path.abspath(os.path.expanduser(reference["path"]))
    try:
        summary = RustForwardPassPerfModel.inspect_state_snapshot(path)
        config = _snapshot_config(path)
    except OSError as exc:
        raise ValueError(f"{where}: cannot read the snapshot at {path}: {exc}") from exc
    pinned = config["estimator_config"]["state_snapshot"].get("sha256")
    expected = reference.get("sha256")
    if expected is not None and (not isinstance(expected, str) or expected.lower() != str(pinned).lower()):
        raise ValueError(f"{where}.sha256 does not match the snapshot manifest at {path}")
    snapshot = SnapshotRole(role=role, path=path, summary=summary, config=config)
    _validate_replayable(snapshot)
    return snapshot


def _validate_replayable(snapshot: SnapshotRole) -> None:
    summary, role = snapshot.summary, snapshot.role
    kind = summary["kind"]
    if kind == "regression":
        raise ValueError(
            f"{snapshot.describe()} holds an fpm_regression model, which engine replay cannot run: replay "
            "prices phases with a native op_level or fpm_interpolation engine. The Dynamo planner saves such a "
            "snapshot when it has no ais_perf_model role configuration (system 'unprofiled', fpm_regression); "
            "configure the planner role with estimation_mode op_level or fpm_interpolation to calibrate a model "
            "the sweeper can use"
        )
    if kind != "native" or summary["selected_estimation_mode"] not in _REPLAY_MODES:
        raise ValueError(
            f"{snapshot.describe()} holds a {kind} {summary['selected_estimation_mode']} model; engine replay "
            "needs complete forward-pass metrics for learned models and can only run op_level or "
            "fpm_interpolation snapshots"
        )
    if summary["worker_type"] != _WORKER_TYPES[role]:
        raise ValueError(
            f"{snapshot.describe()} was calibrated for a {summary['worker_type']} worker, not {_WORKER_TYPES[role]}"
        )
    identity = snapshot.identity
    if identity.get("kv_block_size") is None:
        raise ValueError(
            f"{snapshot.describe()} records no kv_block_size, so it cannot pin the role's KV block size; save it "
            "from a model configured with the deployed kv_block_size"
        )
    queries = _REPLAY_QUERIES[role]
    ready = sorted(
        store["query_type"]
        for store in summary["readiness"]["corrections"]
        if store["query_type"] in queries and store["enabled"] and store["ready"] and store["ready_buckets"] > 0
    )
    if not ready:
        raise ValueError(
            f"{snapshot.describe()} has no enabled, ready correction store for the {sorted(queries)} iterations "
            "replay prices for this role, so it would only reproduce the uncalibrated estimator; tune the model "
            "with observed forward passes before saving it"
        )
    if identity.get("pp") not in (None, 1):
        raise ValueError(
            f"{snapshot.describe()} records pp={identity['pp']}; engine replay has no pipeline-stage model and "
            "only runs pipeline=1"
        )
    for name, default in (("dcp", None), ("cp_size", None)):
        if identity.get(name) not in (default, 1):
            raise ValueError(
                f"{snapshot.describe()} records {name}={identity[name]}, which the sweeper does not search"
            )
    if identity.get("fpm_profile") is not None:
        raise ValueError(f"{snapshot.describe()} uses an FPM profile, which snapshot roles do not support")
    if identity.get("speculation") is not None:
        raise ValueError(
            f"{snapshot.describe()} records speculation, which snapshot roles do not support (MTP depth, nextn, "
            "is supported)"
        )


def _conflict(snapshot: SnapshotRole, setting: str, value: Any, saved: Any) -> ValueError:
    return ValueError(
        f"{setting}={value!r} conflicts with {snapshot.describe()}, which was saved with {saved!r}; "
        "remove the setting to take the snapshot's value"
    )


def _modes(value: dict[str, Any]) -> list[Any]:
    modes = value.get("deployment_mode", ["disagg", "agg"])
    return list(modes) if isinstance(modes, (list, tuple)) else [modes]


def _active_roles(value: dict[str, Any]) -> set[str]:
    modes = _modes(value)
    return ({"agg"} if "agg" in modes else set()) | ({"prefill", "decode"} if "disagg" in modes else set())


def reject_engine_wide_snapshot(value: dict[str, Any]) -> None:
    estimator_config = value.get("estimator_config")
    if isinstance(estimator_config, dict) and estimator_config.get("state_snapshot") is not None:
        raise ValueError(
            "estimator_config.state_snapshot applies to one worker role; set it under "
            "engine.workers.<role>.timing.estimator_config instead of engine.estimator_config"
        )


def derive_snapshot_roles(value: dict[str, Any]) -> dict[str, Any]:
    """Pin a raw search-space mapping to the snapshots its role controls name.

    Runs before field validation, so a key the caller omitted takes the
    snapshot's value and an explicit value must agree with it. The result is
    a fixed point: validating the derived mapping again changes nothing.
    """

    reject_engine_wide_snapshot(value)
    controls = value.get("role_estimator_controls") or {}
    active = _active_roles(value)
    snapshots: dict[str, SnapshotRole] = {}
    for role, settings in controls.items():
        reference = state_snapshot_reference(settings)
        if reference is None or role not in active:
            continue
        snapshot = load_snapshot_role(role, reference)
        snapshots[role] = snapshot
        controls[role] = _role_controls(snapshot, settings, reference)
    if not snapshots:
        return value
    for role, snapshot in snapshots.items():
        _check_role_identity(value, snapshot)
        _pin_block_size(value, snapshot)
        _pin_parallel_shape(value, snapshot)
    _pin_engine_identity(value, list(snapshots.values()))
    return value


def _role_controls(snapshot: SnapshotRole, settings: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    where = f"engine.workers.{_PUBLIC_ROLES[snapshot.role]}.timing"
    if set(settings["estimator_config"]) != {"state_snapshot"}:
        raise ValueError(
            f"{where}.estimator_config may only name state_snapshot: {snapshot.describe()} pins every estimator control"
        )
    for name in ("database_mode", "transfer_policy"):
        if name in settings:
            raise ValueError(f"{where}.{name} conflicts with {snapshot.describe()}, which pins it; remove the setting")
    mode = snapshot.summary["selected_estimation_mode"]
    if settings.get("estimation_mode", mode) != mode:
        raise _conflict(snapshot, f"{where}.estimation_mode", settings["estimation_mode"], mode)
    if settings.get("fallback_policy", "deny") != "deny":
        raise _conflict(snapshot, f"{where}.fallback_policy", settings["fallback_policy"], "deny")
    if "systems_paths" in settings and snapshot.packed_root is not None:
        raise ValueError(
            f"{where}.systems_paths does not apply: {snapshot.describe()} carries its own performance data"
        )
    return {
        **settings,
        "estimation_mode": mode,
        "fallback_policy": "deny",
        "estimator_config": {"state_snapshot": {**reference, "path": snapshot.path}},
    }


def _check_role_identity(value: dict[str, Any], snapshot: SnapshotRole) -> None:
    identity, role = snapshot.identity, snapshot.role
    if value.get("model_name") != identity["model"]:
        raise _conflict(snapshot, "engine.model", value.get("model_name"), identity["model"])
    hardware = value.get(f"{role}_hardware_sku") or value.get("hardware_sku")
    if hardware != identity["system"]:
        raise _conflict(snapshot, f"the {role} role's hardware", hardware, identity["system"])
    if value.get("fpm_profile") is not None:
        raise ValueError(f"engine.fpm_profile cannot be combined with {snapshot.describe()}")
    if value.get(f"{role}_fpm_parquet_path") is not None:
        raise ValueError(f"{role}_fpm_parquet_path cannot be combined with {snapshot.describe()}")
    vision = value.get(f"{role}_vision") or {}
    if vision.get("encoder_parallel") != identity.get("encoder_parallel"):
        raise _conflict(
            snapshot, "the vision encoder_parallel", vision.get("encoder_parallel"), identity.get("encoder_parallel")
        )


def _pin_block_size(value: dict[str, Any], snapshot: SnapshotRole) -> None:
    name = f"{snapshot.role}_block_size"
    saved = int(snapshot.identity["kv_block_size"])
    current = value.get(name)
    ranges = value.get("engine_integer_log_ranges") or {}
    if name in ranges:
        low, high = ranges[name]
        if not low <= saved <= high:
            raise _conflict(snapshot, f"the {snapshot.role} block_size range", ranges[name], saved)
        ranges = {key: bounds for key, bounds in ranges.items() if key != name}
        value["engine_integer_log_ranges"] = ranges
    elif isinstance(current, list):
        if saved not in current:
            raise _conflict(snapshot, f"the {snapshot.role} block_size choices", current, saved)
    elif current is not None and current != saved:
        raise _conflict(snapshot, f"the {snapshot.role} block_size", current, saved)
    value[name] = saved


def _saved_shape(snapshot: SnapshotRole) -> dict[str, int]:
    identity = snapshot.identity
    return {knob: int(identity.get(field) or 1) for knob, field in _PARALLEL}


def _pin_parallel_shape(value: dict[str, Any], snapshot: SnapshotRole) -> None:
    """Restrict the role's parallel domain to the saved shape; replicas stay free."""

    role, shape = snapshot.role, _saved_shape(snapshot)
    mode = "agg" if role == "agg" else "disagg"

    def flat_match(entry: dict[str, Any]) -> bool:
        return all(int(entry.get(knob, 1)) == saved for knob, saved in shape.items())

    def matches(entry: dict[str, Any]) -> bool:
        # A disaggregated preset pairs the two roles' shapes.
        return flat_match(entry if mode == "agg" else entry.get(role, {}))

    if value.get("parallel_configs") and _modes(value) == [mode]:
        value["parallel_configs"] = [entry for entry in value["parallel_configs"] if matches(entry)]
        if not value["parallel_configs"]:
            raise _conflict(snapshot, f"every {mode} parallel preset", "a different shape", shape)
        return
    by_mode = value.get("parallel_configs_by_mode") or {}
    if by_mode.get(mode):
        kept = [entry for entry in by_mode[mode] if matches(entry)]
        if not kept:
            raise _conflict(snapshot, f"every {mode} parallel preset", "a different shape", shape)
        value["parallel_configs_by_mode"] = {**by_mode, mode: kept}
        return
    custom = value.get("parallel_custom_configs_by_mode") or {}
    if custom.get(mode, {}).get(role):
        kept = [entry for entry in custom[mode][role] if flat_match(entry)]
        if not kept:
            raise _conflict(snapshot, f"every {role} parallel preset", "a different shape", shape)
        value["parallel_custom_configs_by_mode"] = {**custom, mode: {**custom[mode], role: kept}}
        return
    independent = dict(value.get("parallel_independent_by_mode") or {})
    domains = dict(independent.get(mode) or {})
    log_ranges = dict((value.get("parallel_independent_log_ranges_by_mode") or {}).get(mode) or {})
    prefix = "" if mode == "agg" else f"{role}_"
    domains.setdefault(f"{prefix}replicas", None)
    for knob, saved in shape.items():
        name = f"{prefix}{knob}"
        if name in log_ranges:
            low, high = log_ranges.pop(name)
            if not low <= saved <= high:
                raise _conflict(snapshot, f"the {role} {knob} range", [low, high], saved)
        elif domains.get(name) is not None and saved not in domains[name]:
            raise _conflict(snapshot, f"the {role} {knob} choices", domains[name], saved)
        domains[name] = [saved]
    independent[mode] = domains
    value["parallel_independent_by_mode"] = independent
    all_log_ranges = dict(value.get("parallel_independent_log_ranges_by_mode") or {})
    if log_ranges:
        all_log_ranges[mode] = log_ranges
    else:
        all_log_ranges.pop(mode, None)
    value["parallel_independent_log_ranges_by_mode"] = all_log_ranges


def _agreed(snapshots: list[SnapshotRole], read) -> tuple[Any, SnapshotRole]:
    first = snapshots[0]
    for other in snapshots[1:]:
        if read(other) != read(first):
            raise ValueError(
                f"{first.describe()} and {other.describe()} disagree on engine-wide settings "
                f"({read(first)!r} vs {read(other)!r}); both roles run one engine"
            )
    return read(first), first


def _pin_engine_identity(value: dict[str, Any], snapshots: list[SnapshotRole]) -> None:
    backend, owner = _agreed(snapshots, lambda snapshot: snapshot.identity["backend"])
    if "backend" in value:
        configured = value["backend"] if isinstance(value["backend"], list) else [value["backend"]]
        if backend not in configured:
            raise _conflict(owner, "engine.backend", configured, backend)
    value["backend"] = [backend]

    version, owner = _agreed(snapshots, lambda snapshot: snapshot.identity["backend_version"])
    current = value.get("backend_version")
    if isinstance(current, dict):
        current = current.get(backend)
    if current is not None and str(current).strip() != version:
        raise _conflict(owner, "engine.backend_version", current, version)
    value["backend_version"] = version

    for name in ENGINE_MODEL_CONTROL_FIELDS:
        saved, owner = _agreed(
            snapshots,
            lambda snapshot, name=name: (
                snapshot.identity.get(name)
                if is_active_engine_model_control(name, snapshot.identity.get(name))
                else None
            ),
        )
        current = value.get(name)
        if is_active_engine_model_control(name, current) and current != saved:
            raise _conflict(owner, f"engine.{name}", current, saved)
        if saved is not None:
            value[name] = saved

    nextn, owner = _agreed(snapshots, lambda snapshot: int(snapshot.identity.get("nextn") or 0))
    current = value.get("aic_nextn")
    if (current or 0) != nextn and current is not None:
        raise _conflict(owner, "engine.nextn", current, nextn)
    if value.get("speculation") is not None:
        raise ValueError(f"engine.speculation conflicts with {owner.describe()}, which records no speculation")
    if nextn:
        if value.get("nextn_accepted") is None:
            raise ValueError(
                f"{owner.describe()} was saved with nextn={nextn}; set engine.nextn={nextn} and "
                "engine.nextn_accepted, the accepted draft tokens per step, which a snapshot does not record"
            )
        value["aic_nextn"] = nextn
