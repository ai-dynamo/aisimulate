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

A regression or learned snapshot, such as a Dynamo planner's, may record
identity facts as ``"unknown"``: an unknown hardware takes the engine's, an
unknown parallel shape the role's explicit single shape, and other unknown
facts stay unknown in the role's estimator identity only. Its recorded
``deployment_limits`` pin the role's KV capacity and cap its scheduler limits.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any

from aisimulate_core.sdk.rust_engine_step import UNKNOWN_IDENTITY as UNKNOWN

from ..config.common import ENGINE_MODEL_CONTROL_FIELDS, is_active_engine_model_control

_WORKER_TYPES = {"agg": "aggregated", "prefill": "prefill", "decode": "decode"}
_PUBLIC_ROLES = {"agg": "aggregated", "prefill": "prefill", "decode": "decode"}
# Correction stores replay applies for each worker role; replay times each rank
# alone, so cross-rank stores never apply.
_REPLAY_QUERIES = {"agg": {"prefill", "decode", "mixed"}, "prefill": {"prefill"}, "decode": {"decode"}}
_REPLAY_MODES = {"op_level", "fpm_interpolation"}
# Regression and learned snapshots price every replay iteration themselves.
_FITTED_KINDS = {"regression", "learned_frozen", "learned_online", "causal_selector"}
# Workload stores such a snapshot needs for the iterations replay schedules for
# each role. An aggregated role also needs the locally mixed store on a backend
# whose scheduler computes prefill and decode in one rank pass, and the
# cross-rank store with attention DP.
_FITTED_STORES = {
    "agg": {"pure_prefill", "pure_decode"},
    "prefill": {"pure_prefill"},
    "decode": {"pure_decode"},
}
# vLLM, and TensorRT-LLM on the shared vLLM core, schedule chunked prefill beside
# running decodes; SGLang runs a prefill batch alone (mixed chunk is not modeled).
_LOCALLY_MIXED_BACKENDS = {"vllm", "trtllm"}
_PARALLEL = (
    ("tp", "tp"),
    ("pp", "pp"),
    ("attention_dp", "attention_dp"),
    ("moe_tp", "moe_tp_size"),
    ("moe_ep", "moe_ep_size"),
)
_PUBLIC_PARALLEL = {
    "tp": "tensor",
    "pp": "pipeline",
    "attention_dp": "attention_data",
    "moe_tp": "moe_tensor",
    "moe_ep": "moe_expert",
}
# Identity facts a regression or learned snapshot may record as unknown.
_UNKNOWN_CAPABLE = (
    "system",
    "backend_version",
    "tp",
    "pp",
    "attention_dp",
    "gemm_quant_mode",
    "moe_quant_mode",
    "fmha_quant_mode",
    "kvcache_quant_mode",
    "comm_quant_mode",
    "attention_backend",
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
    def fitted(self) -> bool:
        """Whether a regression or learned model, which reads no performance data, is saved."""

        return self.summary["kind"] in _FITTED_KINDS

    def limit(self, name: str) -> int | None:
        """A known ``deployment_limits`` value; ``None`` when it is unknown or not recorded."""

        value = (self.identity.get("deployment_limits") or {}).get(name, UNKNOWN)
        return None if value == UNKNOWN else int(value)

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
        return spell_as_saved(self.identity, request)


def spell_as_saved(saved: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Spell request fields left at their defaults as the snapshot's saved configuration does.

    A role pinned to its snapshot asks for the saved engine, but the sweeper
    and the prediction compiler write an unset context-parallel degree, an
    unsharded MoE and inactive model controls one way and a saved
    configuration may write them another (``1``, ``1``/``1`` or
    ``"default"``). Equivalent values take the saved spelling; any other
    difference stays for restoration to reject.
    """

    result = {}
    for name in ("dcp", "cp_size"):
        if (request.get(name) or 1) == (saved.get(name) or 1):
            result[name] = saved.get(name)
    moe = ("moe_tp_size", "moe_ep_size")
    if UNKNOWN in (saved.get(name) for name in moe):
        # An unknown width is spelled so; a known one keeps its saved spelling
        # when equivalent and is otherwise left for restoration to reject.
        for name in moe:
            if saved.get(name) == UNKNOWN or (request.get(name) or 1) == (saved.get(name) or 1):
                result[name] = saved.get(name)
    elif tuple(request.get(name) or 1 for name in moe) == tuple(saved.get(name) or 1 for name in moe):
        result["moe_tp_size"], result["moe_ep_size"] = saved.get("moe_tp_size"), saved.get("moe_ep_size")
    for name in ENGINE_MODEL_CONTROL_FIELDS:
        value = saved.get(name)
        if not is_active_engine_model_control(name, request.get(name)) and not is_active_engine_model_control(
            name, value
        ):
            result[name] = value
    # A fact the snapshot does not know stays unknown in the role's estimator
    # identity, whatever engine the role runs; its limits are identity too.
    for name in _UNKNOWN_CAPABLE:
        if saved.get(name) == UNKNOWN:
            result[name] = UNKNOWN
    if saved.get("deployment_limits") is not None:
        result["deployment_limits"] = copy.deepcopy(saved["deployment_limits"])
    return result


def saved_snapshot_config(estimator_config: Any) -> dict[str, Any]:
    """The configuration a referenced state snapshot was saved with; empty without a snapshot."""

    from aisimulate_core.sdk.rust_engine_step import _snapshot_config

    reference = state_snapshot_reference({"estimator_config": estimator_config})
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        return {}
    return _snapshot_config(reference["path"])


def snapshot_pinned_controls(saved: dict[str, Any]) -> dict[str, Any]:
    """Controls a saved snapshot pins that no search or prediction field sets; empty without a snapshot.

    The configuration that restores a snapshot must repeat these, so callers
    that construct it from public fields take them from the snapshot.
    """

    if not saved:
        return {}
    return {
        name: saved.get(name)
        for name in ("decoder_replay", "enable_shared_layer", "strict_provenance", "fpm_fmha_quant_mode")
    }


def packed_systems_root(estimator_config: Any) -> str | None:
    """The systems root a referenced snapshot carries itself; ``None`` without one.

    It moves with the snapshot and restoration reads it from there, so a
    saved configuration names the snapshot instead of this root.
    """

    from aisimulate_core.sdk import RustForwardPassPerfModel

    reference = state_snapshot_reference({"estimator_config": estimator_config})
    if not isinstance(reference, dict) or not isinstance(reference.get("path"), str):
        return None
    path = os.path.abspath(os.path.expanduser(reference["path"]))
    root = os.path.join(path, "perf_db", "systems")
    identity = RustForwardPassPerfModel.inspect_state_snapshot(path)["identity"]
    return root if identity.get("systems_paths") == [root] else None


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
    kind, mode = summary["kind"], summary["selected_estimation_mode"]
    if kind == "learned_without_identity":
        raise ValueError(
            f"{snapshot.describe()} holds a from_learned model, which records no deployment identity to pin the "
            "role to; save a snapshot of a model constructed with best_available instead"
        )
    if not snapshot.fitted and (kind != "native" or mode not in _REPLAY_MODES):
        raise ValueError(
            f"{snapshot.describe()} holds a {kind} {mode} model; snapshot roles run op_level, fpm_interpolation, "
            "fpm_regression or fpm_learned models"
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
    if snapshot.fitted:
        _validate_fitted(snapshot)
    else:
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
    if identity.get("pp") not in (None, 1, UNKNOWN):
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


def _validate_fitted(snapshot: SnapshotRole) -> None:
    """A regression or learned snapshot prices every replay iteration and bounds its deployment."""

    summary, role, identity = snapshot.summary, snapshot.role, snapshot.identity
    mode = summary["selected_estimation_mode"]
    ready = {store["workload_kind"] for store in summary["readiness"]["stores"] if store["ready"]}
    required = set(_FITTED_STORES[role])
    if role == "agg" and identity["backend"] in _LOCALLY_MIXED_BACKENDS:
        required.add("contains_locally_mixed")
    missing = sorted(required - ready)
    if missing:
        raise ValueError(
            f"{snapshot.describe()} holds an {mode} model without a ready {', '.join(missing)} workload store; "
            f"replay prices every {_WORKER_TYPES[role]} iteration with it, so tune it with observed forward passes "
            "of that work before saving it"
        )
    if int(identity.get("nextn") or 0) > 0:
        raise ValueError(
            f"{snapshot.describe()} records nextn={identity['nextn']}; an {mode} model prices no speculative decoding"
        )
    if identity.get("backend_version") is None:
        raise ValueError(
            f"{snapshot.describe()} records no backend_version; save it with the deployed version or 'unknown'"
        )
    for name, purpose in (
        ("max_kv_tokens", "the role's KV capacity"),
        ("max_num_batched_tokens", "the role's scheduler token budget"),
        ("max_num_seqs", "the role's scheduler sequence limit"),
    ):
        if snapshot.limit(name) is None:
            raise ValueError(
                f"{snapshot.describe()} does not record deployment_limits.{name}, which bounds {purpose}; save "
                "it with the calibrated deployment's limits"
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
        shape = _pin_parallel_shape(value, snapshot)
        if snapshot.fitted:
            _check_fitted_shape(snapshot, shape)
            _pin_deployment_limits(value, snapshot)
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
    # An unknown system takes the engine's hardware.
    if identity["system"] != UNKNOWN and hardware != identity["system"]:
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


def _saved_shape(snapshot: SnapshotRole) -> dict[str, int | None]:
    """The saved parallel shape; ``None`` for a width the snapshot does not know."""

    identity = snapshot.identity
    return {
        knob: None if identity.get(field) == UNKNOWN else int(identity.get(field) or 1) for knob, field in _PARALLEL
    }


def _pin_parallel_shape(value: dict[str, Any], snapshot: SnapshotRole) -> dict[str, int]:
    """Restrict the role's parallel domain to the saved shape and return it; replicas stay free.

    A width the snapshot does not know must be one explicit value in the
    role's domain, which the returned shape takes.
    """

    role, shape = snapshot.role, _saved_shape(snapshot)
    mode = "agg" if role == "agg" else "disagg"
    known = {knob: saved for knob, saved in shape.items() if saved is not None}
    unknown = [knob for knob, saved in shape.items() if saved is None]

    def flat_match(entry: dict[str, Any]) -> bool:
        return all(int(entry.get(knob, 1)) == saved for knob, saved in known.items())

    def role_entry(entry: dict[str, Any]) -> dict[str, Any]:
        # A disaggregated preset pairs the two roles' shapes.
        return entry if mode == "agg" else entry.get(role, {})

    def single(entries: list[dict[str, Any]]) -> dict[str, int]:
        widths = {tuple(int(entry.get(knob, 1)) for knob in unknown) for entry in entries}
        if len(widths) != 1:
            raise _unknown_shape(snapshot, unknown, f"its presets give {sorted(widths)}")
        return {**known, **dict(zip(unknown, widths.pop(), strict=True))}

    if value.get("parallel_configs") and _modes(value) == [mode]:
        value["parallel_configs"] = [entry for entry in value["parallel_configs"] if flat_match(role_entry(entry))]
        if not value["parallel_configs"]:
            raise _conflict(snapshot, f"every {mode} parallel preset", "a different shape", known)
        return single([role_entry(entry) for entry in value["parallel_configs"]])
    by_mode = value.get("parallel_configs_by_mode") or {}
    if by_mode.get(mode):
        kept = [entry for entry in by_mode[mode] if flat_match(role_entry(entry))]
        if not kept:
            raise _conflict(snapshot, f"every {mode} parallel preset", "a different shape", known)
        value["parallel_configs_by_mode"] = {**by_mode, mode: kept}
        return single([role_entry(entry) for entry in kept])
    custom = value.get("parallel_custom_configs_by_mode") or {}
    if custom.get(mode, {}).get(role):
        kept = [entry for entry in custom[mode][role] if flat_match(entry)]
        if not kept:
            raise _conflict(snapshot, f"every {role} parallel preset", "a different shape", known)
        value["parallel_custom_configs_by_mode"] = {**custom, mode: {**custom[mode], role: kept}}
        return single(kept)
    independent = dict(value.get("parallel_independent_by_mode") or {})
    domains = dict(independent.get(mode) or {})
    log_ranges = dict((value.get("parallel_independent_log_ranges_by_mode") or {}).get(mode) or {})
    prefix = "" if mode == "agg" else f"{role}_"
    domains.setdefault(f"{prefix}replicas", None)
    pinned = {}
    for knob, saved in shape.items():
        name = f"{prefix}{knob}"
        if saved is None:
            choices = domains.get(name)
            if name in log_ranges or choices is None or len(choices) != 1:
                given = log_ranges.get(name) or choices
                raise _unknown_shape(snapshot, unknown, f"its {knob} domain is {given!r}")
            saved = int(choices[0])
        elif name in log_ranges:
            low, high = log_ranges.pop(name)
            if not low <= saved <= high:
                raise _conflict(snapshot, f"the {role} {knob} range", [low, high], saved)
        elif domains.get(name) is not None and saved not in domains[name]:
            raise _conflict(snapshot, f"the {role} {knob} choices", domains[name], saved)
        domains[name] = [saved]
        pinned[knob] = saved
    independent[mode] = domains
    value["parallel_independent_by_mode"] = independent
    all_log_ranges = dict(value.get("parallel_independent_log_ranges_by_mode") or {})
    if log_ranges:
        all_log_ranges[mode] = log_ranges
    else:
        all_log_ranges.pop(mode, None)
    value["parallel_independent_log_ranges_by_mode"] = all_log_ranges
    return pinned


def _unknown_shape(snapshot: SnapshotRole, unknown: list[str], given: str) -> ValueError:
    knobs = ", ".join(f"parallelism.{_PUBLIC_PARALLEL[knob]}" for knob in unknown)
    return ValueError(
        f"{snapshot.describe()} does not know the deployment's {', '.join(unknown)}, so the "
        f"{_PUBLIC_ROLES[snapshot.role]} role needs one explicit shape; set engine.workers."
        f"{_PUBLIC_ROLES[snapshot.role]}.{knobs} to single values ({given})"
    )


def _check_fitted_shape(snapshot: SnapshotRole, shape: dict[str, int]) -> None:
    """The role's shape must use the calibrated deployment's GPUs and stores."""

    gpus = shape["tp"] * shape["pp"] * shape["attention_dp"]
    recorded = snapshot.limit("num_gpus")
    if recorded is not None and gpus != recorded:
        raise ValueError(
            f"the {_PUBLIC_ROLES[snapshot.role]} role's shape {shape} uses {gpus} GPUs per replica, but "
            f"{snapshot.describe()} was saved with deployment_limits.num_gpus={recorded}"
        )
    ready = {store["workload_kind"] for store in snapshot.summary["readiness"]["stores"] if store["ready"]}
    if snapshot.role == "agg" and shape["attention_dp"] > 1 and "cross_rank_aggregated" not in ready:
        raise ValueError(
            f"{snapshot.describe()} has no ready cross_rank_aggregated workload store, which prices iterations "
            f"whose attention-DP ranks compute different work; the role runs attention_dp={shape['attention_dp']}"
        )


def _pin_deployment_limits(value: dict[str, Any], snapshot: SnapshotRole) -> None:
    """Fix the role's KV capacity to the recorded one and cap its scheduler limits."""

    from .config import SearchSpace

    role = snapshot.role
    max_kv_tokens, block_size = snapshot.limit("max_kv_tokens"), int(value[f"{role}_block_size"])
    blocks = max_kv_tokens // block_size
    if blocks <= 0:
        raise ValueError(
            f"{snapshot.describe()} records deployment_limits.max_kv_tokens={max_kv_tokens}, less than one "
            f"{block_size}-token block"
        )
    name = f"{role}_num_gpu_blocks"
    if value.get(name) is not None and value[name] != blocks:
        raise _conflict(snapshot, f"the {role} KV capacity blocks", value[name], blocks)
    value[name] = blocks
    memory = f"{role}_gpu_memory_utilization"
    fractions = value.get(memory)
    if memory in (value.get("engine_float_ranges") or {}) or (isinstance(fractions, list) and len(fractions) > 1):
        raise ValueError(
            f"{snapshot.describe()} pins the {role} KV capacity to deployment_limits.max_kv_tokens={max_kv_tokens}, "
            "so KV memory is not searched; remove the role's kv_cache.capacity memory_fraction search"
        )
    for limit, knob in (("max_num_batched_tokens", "max_batched_tokens"), ("max_num_seqs", "max_sequences")):
        cap, name = snapshot.limit(limit), f"{role}_{limit}"
        ranges = dict(value.get("engine_integer_log_ranges") or {})
        setting = f"the {role} scheduler {knob}"
        if name in ranges:
            low, high = ranges.pop(name)
            if low > cap:
                raise _exceeds(snapshot, setting, [low, high], limit, cap)
            if min(high, cap) > low:
                ranges[name] = [low, min(high, cap)]
            else:
                # A range the cap collapses to one value is that value, not
                # the knob's uncapped default domain.
                value[name] = [low]
            value["engine_integer_log_ranges"] = ranges
            continue
        domain = value.get(name, SearchSpace.model_fields[name].default)
        domain = list(domain) if isinstance(domain, (list, tuple)) else [domain]
        kept = [choice for choice in domain if int(choice) <= cap]
        if not kept:
            raise _exceeds(snapshot, setting, domain, limit, cap)
        value[name] = kept


def _exceeds(snapshot: SnapshotRole, setting: str, value: Any, limit: str, cap: int) -> ValueError:
    return ValueError(
        f"{setting}={value!r} exceeds {snapshot.describe()}, which was saved with deployment_limits.{limit}={cap}; "
        f"give values up to {cap}"
    )


def _agreed(snapshots: list[SnapshotRole], read) -> tuple[Any, SnapshotRole]:
    """The value every snapshot reads, ignoring those that do not know it; ``UNKNOWN`` when none does."""

    known = [snapshot for snapshot in snapshots if read(snapshot) != UNKNOWN]
    if not known:
        return UNKNOWN, snapshots[0]
    snapshots = known
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

    # A version or model control no snapshot knows stays the caller's for the
    # engine; each such role's estimator identity spells it unknown.
    version, owner = _agreed(snapshots, lambda snapshot: snapshot.identity["backend_version"])
    current = value.get("backend_version")
    if isinstance(current, dict):
        current = current.get(backend)
    if version != UNKNOWN:
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
        if saved == UNKNOWN:
            continue
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
