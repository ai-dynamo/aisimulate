# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and execute the public recommendation configuration."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .aic import resolve_model_context_length
from .public_config import PredictionConfig, RecommendationConfig, TrafficConfig
from .sweeper.config import SmartSearchConfig
from .sweeper.replay import ReplaySpec, RunnerFactory


def run_recommendation(
    config: RecommendationConfig,
    *,
    stack: str,
    runner_factory: RunnerFactory,
    providers: Mapping[str, Any] | None = None,
    show_progress: bool = True,
):
    """Run a public recommendation through the existing Sweeper core."""

    from .sweeper.search import Sweeper

    smart = recommendation_to_sweeper(config, stack=stack)
    sweeper = Sweeper(
        runner_factory=runner_factory,
        providers=providers,
        show_progress=show_progress,
        prediction_config_factory=lambda sample, spec: _candidate_prediction(
            config, sample, spec
        ).model_dump(mode="python", exclude_none=True),
    )
    return sweeper.run(smart)


def recommendation_to_sweeper(
    config: RecommendationConfig, *, stack: str = "engine"
) -> SmartSearchConfig:
    engine = deepcopy(config.engine)
    optimization = config.optimization
    mode_values = _choices(
        engine.get("mode"), default=["aggregated", "disaggregated"]
    )
    modes = [_legacy_mode(str(value)) for value in mode_values]
    backend_values = _choices(engine.get("backend"), default=["vllm", "sglang"])
    model = engine.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("engine.model is required and must be concrete")
    hardware = engine.get("hardware")
    if hardware == "auto":
        hardware = optimization.hardware
    if not isinstance(hardware, str) or not hardware:
        raise ValueError("engine.hardware must resolve to one concrete identifier")
    context = engine.get("context_length", "max")
    if context != "max" and (
        not isinstance(context, int) or isinstance(context, bool) or context <= 0
    ):
        raise ValueError("engine.context_length must be 'max' or a positive integer")

    workers = engine.get("workers")
    if not isinstance(workers, dict):
        raise ValueError("engine.workers is required")
    search_space: dict[str, Any] = {
        "deployment_mode": modes,
        "backend": [str(value) for value in backend_values],
        "model_name": model,
        "hardware_sku": hardware,
        "gpu_budget": optimization.constraints.max_candidate_gpus,
        "min_gpu_budget": optimization.constraints.min_candidate_gpus,
        "context_length": (
            resolve_model_context_length(model) if context == "max" else context
        ),
    }
    search_space.update(_role_search_space(workers, modes))
    transfer = engine.get("kv_transfer")
    if isinstance(transfer, dict):
        bytes_per_token = transfer.get("bytes_per_token", "auto")
        search_space["kv_transfer_bytes_per_token"] = bytes_per_token
        search_space["kv_transfer_bandwidth"] = transfer.get(
            "bandwidth_gb_per_second"
        )
        search_space["kv_transfer_timing_mode"] = transfer.get(
            "timing_mode", "destination_missing"
        )
    pinned_parallel, flat_modes, independent_parallel = _parallel_config_choices(
        workers, modes
    )
    if pinned_parallel:
        search_space["parallel_configs_by_mode"] = pinned_parallel
    if flat_modes:
        search_space["flat_parallel_modes"] = flat_modes
    if independent_parallel:
        search_space["parallel_independent_by_mode"] = independent_parallel

    workload = _recommendation_workload(config.traffic)
    goal = _goal(config)
    adapters: dict[str, Any] = {}
    if config.router is not None:
        adapters[f"{stack}.router"] = {
            "search_space": _router_search_space(config.router)
        }
    if config.planner is not None:
        adapters[f"{stack}.planner"] = {
            "search_space": _planner_search_space(config.planner)
        }

    parallelism = config.optimizer.parallelism
    # New exact-global controls are carried alongside the legacy fields. The
    # Sweeper consumes them directly; max_rounds remains one for old callers.
    sweep = {
        "max_rounds": max(
            1,
            math.ceil(
                config.optimizer.max_trials
                / max(1, min(parallelism, config.optimizer.max_trials))
            ),
        ),
        "parallel_evals": parallelism,
        "candidates_per_round": min(parallelism, config.optimizer.max_trials),
        "max_eval_seconds": config.optimizer.candidate_timeout_seconds,
        "max_trials": config.optimizer.max_trials,
        "algorithm": config.optimizer.algorithm,
        "seed": config.optimizer.seed,
    }
    return SmartSearchConfig.model_validate(
        {
            "search_space": search_space,
            "adapters": adapters,
            "workload": workload,
            "goal": goal,
            "sweep": sweep,
        }
    )


def _choices(value: Any, *, default: list[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, dict) and set(value) == {"choices"}:
        return list(value["choices"])
    if isinstance(value, dict) and set(value) == {"range"}:
        raw = value["range"]
        step = raw.get("step")
        if raw.get("scale", "linear") != "linear" or step is None:
            raise ValueError("this engine field requires choices or a stepped linear range")
        values: list[Any] = []
        current = raw["min"]
        while current <= raw["max"]:
            values.append(current)
            current += step
        return values
    return [value]


def _legacy_mode(mode: str) -> str:
    return {"aggregated": "agg", "disaggregated": "disagg"}.get(mode, mode)


def _role_search_space(workers: dict[str, Any], modes: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "engine_float_ranges": {},
        "engine_log_ranges": [],
    }
    specifications = (
        ("aggregated", "agg", "agg" in modes),
        ("prefill", "prefill", "disagg" in modes),
        ("decode", "decode", "disagg" in modes),
    )
    for public_role, legacy_role, required in specifications:
        raw = workers.get(public_role)
        if not required:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"engine.workers.{public_role} is required")
        scheduler = raw.get("scheduler") or {}
        if not isinstance(scheduler, dict):
            raise ValueError(f"engine.workers.{public_role}.scheduler must be a mapping")
        tokens_default = [8192] if legacy_role == "decode" else [8192, 16384, 32768]
        sequences_default = (
            [1, 2, 4, 8, 16, 32, 64, 128, 256]
            if legacy_role == "prefill"
            else [256, 512, 1024]
        )
        result[f"{legacy_role}_max_num_batched_tokens"] = _choices(
            scheduler.get("max_batched_tokens"), default=tokens_default
        )
        result[f"{legacy_role}_max_num_seqs"] = _choices(
            scheduler.get("max_sequences"), default=sequences_default
        )
        cache = raw.get("kv_cache") or {}
        capacity = cache.get("capacity") or {}
        block_value = cache.get("block_size")
        result[f"{legacy_role}_block_size"] = (
            None
            if block_value is None
            else _choices(block_value, default=[])
            if isinstance(block_value, dict)
            else block_value
        )
        memory_value = capacity.get("memory_fraction")
        memory_name = f"{legacy_role}_gpu_memory_utilization"
        if isinstance(memory_value, dict) and "range" in memory_value:
            bounds = memory_value["range"]
            if bounds.get("scale", "linear") == "linear" and bounds.get("step") is not None:
                result[memory_name] = _choices(memory_value, default=[])
            else:
                result[memory_name] = bounds["min"]
                result["engine_float_ranges"][memory_name] = [
                    bounds["min"],
                    bounds["max"],
                ]
                if bounds.get("scale", "linear") == "log":
                    result["engine_log_ranges"].append(memory_name)
        elif isinstance(memory_value, dict):
            result[memory_name] = _choices(memory_value, default=[])
        else:
            result[memory_name] = memory_value
        result[f"{legacy_role}_enable_prefix_caching"] = cache.get(
            "prefix_caching", True
        )
        capacity_type = capacity.get("type", "default")
        result[f"{legacy_role}_num_gpu_blocks"] = (
            capacity.get("blocks") if capacity_type == "fixed" else None
        )
        timing = raw.get("timing") or {}
        timing_type = timing.get("type", "default")
        if timing_type == "fixed":
            result[f"{legacy_role}_timing_model"] = {
                "type": "fixed",
                "prefill_ms": timing.get("prefill_ms"),
                "decode_ms": timing.get("decode_ms"),
            }
        elif timing_type == "polynomial":
            result[f"{legacy_role}_timing_model"] = {"type": "polynomial"}
        else:
            result[f"{legacy_role}_timing_model"] = None
        result[f"{legacy_role}_startup_time"] = raw.get("startup_seconds", 0)
    # Remove empty internal maps so legacy serialization remains concise.
    if not result["engine_float_ranges"]:
        result.pop("engine_float_ranges")
    if not result["engine_log_ranges"]:
        result.pop("engine_log_ranges")
    return result


_PARALLEL_KEYS = (
    "replicas",
    "tensor",
    "pipeline",
    "attention_data",
    "moe_tensor",
    "moe_expert",
)


def _parallel_entries(role: str, raw: dict[str, Any]) -> tuple[str, Any]:
    parallel = raw.get("parallelism") or {}
    if not isinstance(parallel, dict):
        raise ValueError(f"engine.workers.{role}.parallelism must be a mapping")
    preset = parallel.get("preset", "default")
    independent = [key for key in _PARALLEL_KEYS if key in parallel]
    if preset not in (False, {}) and independent:
        raise ValueError(
            f"engine.workers.{role}.parallelism cannot combine preset with "
            f"independent knobs {independent}"
        )
    if preset == "default":
        return "default", None
    if isinstance(preset, list):
        return "flat", [
            _parallel_mapping(
                entry, f"engine.workers.{role}.parallelism.preset"
            )
            for entry in preset
        ]
    if preset not in (False, {}):
        raise ValueError(f"invalid parallelism preset for role {role}")
    return "independent", {
        key: _choices(parallel[key], default=[])
        if key in parallel
        else None
        for key in _PARALLEL_KEYS
    }


def _parallel_mapping(value: Any, path: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} entries must be mappings")
    missing = set(_PARALLEL_KEYS) - set(value)
    unknown = set(value) - set(_PARALLEL_KEYS)
    if missing or unknown:
        raise ValueError(
            f"{path} entry must cover exactly {_PARALLEL_KEYS}; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return {key: int(value[key]) for key in _PARALLEL_KEYS}


def _legacy_parallel(entry: dict[str, int]) -> dict[str, int]:
    return {
        "replicas": entry["replicas"],
        "tp": entry["tensor"],
        "pp": entry["pipeline"],
        "attention_dp": entry["attention_data"],
        "moe_tp": entry["moe_tensor"],
        "moe_ep": entry["moe_expert"],
    }


def _parallel_config_choices(
    workers: dict[str, Any], modes: list[str]
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[str],
    dict[str, dict[str, list[int] | None]],
]:
    role_specs: dict[str, tuple[str, Any]] = {}
    if "agg" in modes:
        role_specs["agg"] = _parallel_entries("aggregated", workers["aggregated"])
    if "disagg" in modes:
        role_specs["prefill"] = _parallel_entries("prefill", workers["prefill"])
        role_specs["decode"] = _parallel_entries("decode", workers["decode"])
    pinned: dict[str, list[dict[str, Any]]] = {}
    flat_modes: list[str] = []
    independent: dict[str, dict[str, list[int] | None]] = {}
    if "agg" in modes:
        kind, value = role_specs["agg"]
        if kind == "flat":
            pinned["agg"] = [_legacy_parallel(entry) for entry in value]
            flat_modes.append("agg")
        elif kind == "independent":
            independent["agg"] = {
                {
                    "replicas": "replicas",
                    "tensor": "tp",
                    "pipeline": "pp",
                    "attention_data": "attention_dp",
                    "moe_tensor": "moe_tp",
                    "moe_expert": "moe_ep",
                }[name]: choices
                for name, choices in value.items()
            }
    if "disagg" in modes:
        prefill_kind, prefill = role_specs["prefill"]
        decode_kind, decode = role_specs["decode"]
        if prefill_kind != decode_kind:
            raise ValueError(
                "disaggregated parallelism roles must use the same preset mode"
            )
        if prefill_kind == "flat":
            pinned["disagg"] = [
                {"prefill": _legacy_parallel(p), "decode": _legacy_parallel(d)}
                for p, d in itertools.product(prefill, decode)
            ]
            flat_modes.append("disagg")
        elif prefill_kind == "independent":
            combined: dict[str, list[int] | None] = {}
            mapping = {
                "replicas": "replicas",
                "tensor": "tp",
                "pipeline": "pp",
                "attention_data": "attention_dp",
                "moe_tensor": "moe_tp",
                "moe_expert": "moe_ep",
            }
            for role, values in (("prefill", prefill), ("decode", decode)):
                for name, choices in values.items():
                    combined[f"{role}_{mapping[name]}"] = choices
            independent["disagg"] = combined
    return pinned, flat_modes, independent


def _recommendation_workload(raw: dict[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {
            "isl": 1024,
            "osl": 128,
            "concurrency": 10,
            "num_request_ratio": 10,
        }
    source = raw.get("source")
    load = raw.get("load")
    stop = raw.get("stop")
    if not isinstance(source, dict) or not isinstance(load, dict):
        raise ValueError("traffic.source and traffic.load are required")
    source_type = source.get("type")
    load_type = load.get("type")
    result: dict[str, Any] = {
        "source_type": source_type,
        "load_type": load_type,
    }
    if source_type == "trace":
        paths = source.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError("trace traffic requires paths")
        result.update(
            trace_path=paths[0],
            trace_paths=paths,
            trace_format=source.get("format", "mooncake"),
            trace_block_size=source.get("block_size"),
        )
        if load_type == "concurrency":
            _configure_load_domain(
                result,
                load.get("concurrency"),
                field="replay_concurrency",
                integer=True,
            )
        else:
            _configure_load_domain(
                result,
                load.get("speedup", 1.0),
                field="arrival_speedup_ratio",
                integer=False,
            )
        if isinstance(stop, dict) and stop.get("max_virtual_time_seconds") is not None:
            result["max_sim_time_ms"] = 1_000.0 * float(
                stop["max_virtual_time_seconds"]
            )
        return result
    if source_type == "synthetic":
        result.update(isl=source.get("input_tokens", 1024), osl=source.get("output_tokens", 128))
        count = stop.get("requests") if isinstance(stop, dict) else None
        ratio = stop.get("requests_per_load_unit") if isinstance(stop, dict) else None
    elif source_type == "synthetic-session":
        session = source.get("session") or {}
        result.update(
            isl=source.get("new_input_tokens_per_turn", 1024),
            osl=source.get("output_tokens_per_turn", 128),
            turns_per_session=session.get("turns", 4),
            shared_prefix_ratio=session.get("shared_prefix_ratio", 0.0),
            num_prefix_groups=session.get("prefix_groups", 0),
            inter_turn_delay_ms=session.get("inter_turn_delay_ms", 0.0),
        )
        count = stop.get("sessions") if isinstance(stop, dict) else None
        ratio = stop.get("sessions_per_load_unit") if isinstance(stop, dict) else None
    else:
        raise ValueError(f"unsupported traffic source type {source_type!r}")
    if load_type == "concurrency":
        _configure_load_domain(
            result,
            load.get("concurrency"),
            field="concurrency",
            integer=True,
        )
    elif load_type in {"poisson", "constant_rate"}:
        value = load.get("requests_per_second", load.get("sessions_per_second"))
        _configure_load_domain(
            result,
            value,
            field="request_rate",
            integer=False,
        )
        if load_type == "poisson":
            result["arrival_seed"] = load.get("seed", 42)
    elif load_type == "kv_capacity_fraction":
        value = load.get("fraction")
        if isinstance(value, dict) and "range" in value:
            bounds = value["range"]
            result["kv_load_ratio"] = [bounds["min"], bounds["max"]]
        else:
            result["kv_load_ratio"] = _single_value(value)
    else:
        raise ValueError(f"unsupported synthetic load type {load_type!r}")
    if count is not None:
        result["request_count"] = count
    else:
        result["num_request_ratio"] = ratio
    return result


def _single_value(value: Any) -> Any:
    values = _choices(value, default=[])
    if len(values) != 1:
        raise ValueError("this traffic domain is not yet representable as one Sweeper load")
    return values[0]


def _configure_load_domain(
    result: dict[str, Any],
    value: Any,
    *,
    field: str,
    integer: bool,
) -> None:
    if isinstance(value, dict) and set(value) == {"choices"}:
        choices = list(value["choices"])
        result[field] = choices[0]
        result["load_search_field"] = field
        result["load_choices"] = choices
        result["load_integer"] = integer
        return
    if isinstance(value, dict) and set(value) == {"range"}:
        bounds = value["range"]
        step = bounds.get("step")
        if bounds.get("scale", "linear") == "linear" and step is not None:
            choices = _choices(value, default=[])
            result[field] = choices[0]
            result["load_search_field"] = field
            result["load_choices"] = choices
        else:
            if integer and bounds.get("scale", "linear") == "linear":
                raise ValueError("integer linear traffic ranges require step")
            result[field] = bounds["min"]
            result["load_search_field"] = field
            result["load_range"] = [bounds["min"], bounds["max"]]
            result["load_log_scale"] = bounds.get("scale", "linear") == "log"
        result["load_integer"] = integer
        return
    result[field] = value


def _goal(config: RecommendationConfig) -> dict[str, Any]:
    target = config.optimization.target
    payload: dict[str, Any] = {"target": target}
    sla = config.evaluation.sla
    if sla is not None:
        payload["sla"] = sla.model_dump(mode="json", exclude_none=True)
    return payload


def _router_search_space(raw: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(raw)
    result.setdefault(
        "policy", {"choices": ["round_robin", "kv_router"]}
    )
    return result


def _planner_search_space(raw: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(raw)
    result.setdefault("policy", {"choices": ["disabled", "enabled"]})
    return result


def _candidate_prediction(
    source: RecommendationConfig,
    sample: dict[str, Any],
    replay_spec: ReplaySpec,
) -> PredictionConfig:
    deployment = replay_spec.backend_deployment
    engine: dict[str, Any] = {
        "mode": "aggregated" if deployment.deployment_mode == "agg" else "disaggregated",
        "model": sample["model_name"],
        "hardware": sample["hardware_sku"],
        "backend": sample["backend"],
        "backend_version": sample.get("backend_version") or None,
        "context_length": sample.get("context_length") or "max",
        "workers": {},
    }
    roles = ("agg",) if deployment.deployment_mode == "agg" else ("prefill", "decode")
    for role in roles:
        prefix = "" if role == "agg" else f"{role}_"
        public_role = "aggregated" if role == "agg" else role
        raw_worker = (
            source.engine.get("workers", {}).get(public_role, {})
            if isinstance(source.engine.get("workers"), dict)
            else {}
        )
        block_size = sample[f"{role}_block_size"]
        if block_size is None:
            block_size = {"vllm": 64, "sglang": 1, "trtllm": 32}[
                sample["backend"]
            ]
        memory_fraction = sample[f"{role}_gpu_memory_utilization"]
        if memory_fraction is None:
            memory_fraction = 0.88 if sample["backend"] == "sglang" else 0.9
        num_blocks = sample.get(f"{role}_num_gpu_blocks")
        capacity = (
            {"type": "fixed", "blocks": num_blocks}
            if num_blocks is not None
            else {"type": "default", "memory_fraction": memory_fraction}
        )
        timing_model = sample.get(f"{role}_timing_model")
        if isinstance(timing_model, dict):
            timing = deepcopy(timing_model)
        else:
            timing = {"type": "default"}
        engine["workers"][public_role] = {
            "parallelism": {
                "replicas": sample[f"{prefix}replicas"],
                "tensor": sample[f"{prefix}tp"],
                "pipeline": sample[f"{prefix}pp"],
                "attention_data": sample[f"{prefix}attention_dp"],
                "moe_tensor": sample[f"{prefix}moe_tp"],
                "moe_expert": sample[f"{prefix}moe_ep"],
            },
            "scheduler": {
                "max_batched_tokens": sample[f"{role}_max_num_batched_tokens"],
                "max_sequences": sample[f"{role}_max_num_seqs"],
            },
            "kv_cache": {
                "block_size": block_size,
                "prefix_caching": sample[f"{role}_enable_prefix_caching"],
                "capacity": capacity,
            },
            "timing": timing,
            "startup_seconds": sample.get(f"{role}_startup_time")
            if sample.get(f"{role}_startup_time") is not None
            else raw_worker.get("startup_seconds", 0),
        }
    raw_engine = source.engine
    if deployment.deployment_mode == "disagg" and raw_engine.get("kv_transfer") is not None:
        engine["kv_transfer"] = deepcopy(raw_engine["kv_transfer"])

    traffic = _candidate_traffic(source.traffic, sample)
    router: dict[str, Any] = {
        "policy": "round_robin",
        "prefill_load_model": {"type": "none"},
    }
    planner: dict[str, Any] = {"policy": "disabled"}
    for name, adapter in replay_spec.adapters.items():
        if name.endswith(".router"):
            router = deepcopy(adapter.config)
            router.setdefault("policy", router.pop("mode", "round_robin"))
            router.setdefault("prefill_load_model", {"type": "none"})
        elif name.endswith(".planner"):
            planner = deepcopy(adapter.config)
            planner.setdefault("policy", "enabled" if adapter.runtime_hooks else "disabled")
    return PredictionConfig.model_validate(
        {
            "traffic": traffic,
            "engine": engine,
            "router": router,
            "planner": planner,
            "evaluation": source.evaluation.model_dump(mode="python", exclude_none=True),
        }
    )


def _candidate_traffic(
    raw: dict[str, Any] | None, sample: dict[str, Any]
) -> dict[str, Any]:
    if raw is None:
        return TrafficConfig.default().model_dump(mode="python", exclude_none=True)
    traffic = deepcopy(raw)
    load = traffic["load"]
    for name in (
        "concurrency",
        "requests_per_second",
        "sessions_per_second",
        "fraction",
        "speedup",
    ):
        value = load.get(name)
        if isinstance(value, dict):
            internal = {
                "concurrency": "concurrency",
                "requests_per_second": "request_rate",
                "sessions_per_second": "request_rate",
                "speedup": "arrival_speedup_ratio",
            }.get(name)
            if internal is not None and sample.get(internal) is not None:
                load[name] = sample[internal]
            elif name == "fraction" and sample.get("kv_load_ratio") is not None:
                # Recommendation output must be directly replayable and therefore
                # materializes capacity-relative load as concrete concurrency.
                load.clear()
                load.update(type="concurrency", concurrency=sample["concurrency"])
            else:
                raise ValueError(f"candidate did not materialize traffic.load.{name}")
    return traffic
