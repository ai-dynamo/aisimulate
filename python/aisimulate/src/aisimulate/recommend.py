# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile and execute the public recommendation configuration."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .capacity import resolve_model_context_length
from .config.cli import CorePredictionConfig, CoreRecommendationConfig
from .config.traffic import TrafficPredictionConfig
from .config_adapter import (
    CompiledSweepProvider,
    RecommendationAdapterContext,
    SimulationConfigAdapter,
)
from .resources import GuardedRunnerFactory, discover_host, resolve_budget
from .sweeper.afd_perfmodel import AFDPerformanceModel
from .sweeper.config import SmartSearchConfig
from .sweeper.provider import InfeasibleCandidate, SweepContext
from .sweeper.replay import ReplaySpec, RunnerFactory
from .sweeper.result import SweepResult


def run_recommendation(
    config: CoreRecommendationConfig,
    *,
    adapter_configs: Mapping[str, Mapping[str, Any]] | None = None,
    stack: str,
    runner_factory: RunnerFactory,
    providers: Mapping[str, SimulationConfigAdapter] | None = None,
    afd_performance_model: AFDPerformanceModel | None = None,
    show_progress: bool = True,
) -> SweepResult:
    """Run a public recommendation through the existing Sweeper core."""

    from .supervision import in_supervised_process, supervised_recommendation

    kwargs = dict(
        adapter_configs=adapter_configs,
        stack=stack,
        runner_factory=runner_factory,
        providers=providers,
        afd_performance_model=afd_performance_model,
        show_progress=show_progress,
    )
    if not in_supervised_process():
        return supervised_recommendation(config, kwargs)
    return _run_recommendation(config, **kwargs)


def _run_recommendation(
    config: CoreRecommendationConfig,
    *,
    adapter_configs: Mapping[str, Mapping[str, Any]] | None = None,
    stack: str,
    runner_factory: RunnerFactory,
    providers: Mapping[str, SimulationConfigAdapter] | None = None,
    afd_performance_model: AFDPerformanceModel | None = None,
    show_progress: bool = True,
) -> SweepResult:
    from .supervision import checkpoint

    checkpoint("requested_config", config.model_dump(mode="json"))
    budget = resolve_budget(config.execution.resources, discover_host())
    from .sweeper.search import Sweeper

    runner_factory = GuardedRunnerFactory(runner_factory, stack, config.execution.resources)
    if config.engine.workers.encoder is not None and (stack != "engine" or adapter_configs):
        raise ValueError("analytical EPD requires --stack engine without adapters")
    if config.engine.speculation is not None and (stack != "engine" or adapter_configs):
        raise ValueError("ngram speculation requires --stack engine without adapters")
    smart = recommendation_to_sweeper(config, adapter_configs=adapter_configs, stack=stack)
    smart.sweep.parallel_evals = min(config.optimizer.parallelism, budget["cpu_limit"])
    sweep_context = SweepContext(
        core_search_space=smart.search_space.model_dump(mode="json"),
        workload=smart.workload.model_dump(mode="json"),
        goal=smart.goal.model_dump(mode="json"),
        show_progress=show_progress,
    )
    compiled_providers: dict[str, CompiledSweepProvider] = {}
    for section, raw in (adapter_configs or {}).items():
        name = f"{stack}.{section}"
        adapter = (providers or {})[name]
        context = RecommendationAdapterContext(
            engine=config.engine.model_dump(mode="json", exclude_none=True),
            traffic=(config.traffic.model_dump(mode="json", exclude_none=True) if config.traffic is not None else {}),
            evaluation=config.evaluation.model_dump(mode="json", exclude_none=True),
            optimization=config.optimization.model_dump(mode="json", exclude_none=True),
            sweep=sweep_context,
        )
        compiled_providers[name] = CompiledSweepProvider(
            adapter=adapter,
            plan=adapter.compile_recommendation(raw, context),
        )
    adapter_sections = {name: provider.section for name, provider in (providers or {}).items()}
    sweeper = Sweeper(
        runner_factory=runner_factory,
        providers=compiled_providers,
        show_progress=show_progress,
        prediction_config_factory=lambda sample, spec: _candidate_prediction(
            config, sample, spec, adapter_sections=adapter_sections
        ),
        afd_performance_model=afd_performance_model,
    )
    return sweeper.run(smart, top_n=None)


def recommendation_to_sweeper(
    config: CoreRecommendationConfig,
    *,
    adapter_configs: Mapping[str, Mapping[str, Any]] | None = None,
    stack: str = "engine",
) -> SmartSearchConfig:
    engine = config.engine.model_dump(mode="python", exclude_none=True)
    optimization = config.optimization
    mode_values = _choices(engine.get("mode"), default=["aggregated", "disaggregated"])
    modes = [_legacy_mode(str(value)) for value in mode_values]
    afd = engine.get("afd")
    if modes == ["afd"]:
        if not isinstance(afd, dict):
            raise ValueError("engine.mode='afd' requires engine.afd")
        modes = ["afd+pd" if afd["combined_with_pd"] else "afd"]
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
    if context != "max" and (not isinstance(context, int) or isinstance(context, bool) or context <= 0):
        raise ValueError("engine.context_length must be 'max' or a positive integer")

    workers = engine.get("workers")
    if not isinstance(workers, dict):
        raise ValueError("engine.workers is required")
    search_space: dict[str, Any] = {
        "deployment_mode": modes,
        "backend": [str(value) for value in backend_values],
        "backend_version": engine.get("backend_version"),
        "model_name": model,
        "hardware_sku": hardware,
        "gpu_budget": optimization.constraints.max_candidate_gpus,
        "min_gpu_budget": optimization.constraints.min_candidate_gpus,
        "context_length": (resolve_model_context_length(model) if context == "max" else context),
    }
    for name in (
        "database_mode",
        "transfer_policy",
        "systems_paths",
        "estimation_mode",
        "fallback_policy",
        "estimator_config",
    ):
        if name in engine:
            search_space[name] = deepcopy(engine[name])
    search_space["role_estimator_controls"] = {
        ("agg" if role == "aggregated" else role): {
            name: deepcopy(raw.get("timing", {})[name])
            for name in (
                "estimation_mode",
                "fallback_policy",
                "estimator_config",
                "systems_paths",
                "database_mode",
                "transfer_policy",
            )
            if name in raw.get("timing", {})
        }
        for role, raw in workers.items()
        if role in {"aggregated", "prefill", "decode"}
        and not set(modes) & {"afd", "afd+pd"}
        and workers.get("encoder") is None
    }
    if engine.get("speculation") is not None:
        search_space["speculation"] = deepcopy(engine["speculation"])
    for role in ("prefill", "decode"):
        if workers.get(role, {}).get("hardware") is not None:
            search_space[f"{role}_hardware_sku"] = workers[role]["hardware"]
    if isinstance(afd, dict):
        search_space.update(_afd_search_space(afd))
        if modes == ["afd+pd"]:
            companion_role = "decode" if afd["phase"] == "prefill" else "prefill"
            workers.setdefault(companion_role, {})
            kind, value = _parallel_entries(companion_role, workers[companion_role])
            if kind == "flat":
                search_space["afd_companion_parallel_configs"] = [_legacy_parallel(entry) for entry in value]
            elif kind == "independent":
                raise ValueError(
                    f"engine.workers.{companion_role}.parallelism supports preset=default or a concrete "
                    "preset list for AFD recommendations"
                )
    search_space.update(
        _role_search_space(
            workers,
            modes,
            afd_phase=afd.get("phase") if isinstance(afd, dict) else None,
        )
    )
    if engine.get("workers", {}).get("encoder") is not None:
        encoder = workers["encoder"]
        search_space["encoder"] = {
            "hardware_sku": encoder.get("hardware"),
            "backend_version": encoder.get("backend_version"),
            "tp": _choices(encoder["tensor"], default=[1]),
            "workers": _choices(encoder["replicas"], default=[1]),
            "batch_size": _choices(encoder["batch_size"], default=[1]),
            "latency_correction": encoder["latency_correction"],
            "rate_degradation": encoder["rate_degradation"],
        }
    transfer = engine.get("kv_transfer")
    if isinstance(transfer, dict):
        search_space["kv_transfer_bytes_per_token"] = transfer.get("bytes_per_token")
        search_space["kv_transfer_bandwidth"] = transfer.get("bandwidth_gb_per_second")
        search_space["kv_transfer_timing_mode"] = transfer.get("timing_mode", "destination_missing")
    (
        pinned_parallel,
        flat_modes,
        independent_parallel,
        independent_parallel_log_ranges,
        custom_parallel,
    ) = _parallel_config_choices(workers, modes)
    if pinned_parallel:
        search_space["parallel_configs_by_mode"] = pinned_parallel
    if flat_modes:
        search_space["flat_parallel_modes"] = flat_modes
    if independent_parallel:
        search_space["parallel_independent_by_mode"] = independent_parallel
    if independent_parallel_log_ranges:
        search_space["parallel_independent_log_ranges_by_mode"] = independent_parallel_log_ranges
    if custom_parallel:
        search_space["parallel_custom_configs_by_mode"] = custom_parallel

    workload = _recommendation_workload(
        config.traffic.model_dump(mode="python", exclude_none=True) if config.traffic is not None else None
    )
    goal = _goal(config)
    adapters = {
        f"{stack}.{section}": {"search_space": deepcopy(dict(search_spec))}
        for section, search_spec in (adapter_configs or {}).items()
    }

    parallelism = config.optimizer.parallelism
    # New exact-global controls are carried alongside the legacy fields. The
    # Sweeper consumes them directly; max_rounds remains one for old callers.
    sweep = {
        "max_rounds": max(
            1,
            math.ceil(config.optimizer.max_trials / max(1, min(parallelism, config.optimizer.max_trials))),
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
        if raw.get("scale", "linear") == "log":
            raise ValueError("integer log ranges must be lowered as compact bounds")
        if step is None:
            raise ValueError("integer linear engine ranges require step")
        values: list[Any] = []
        current = raw["min"]
        while current <= raw["max"]:
            values.append(current)
            current += step
        return values
    return [value]


def _integer_domain(value: Any, *, default: list[int]) -> tuple[list[int], list[int] | None]:
    """Lower an integer domain without eagerly expanding a log-scale interval."""

    if isinstance(value, dict) and set(value) == {"range"}:
        raw = value["range"]
        if raw.get("scale", "linear") == "log":
            minimum, maximum = int(raw["min"]), int(raw["max"])
            if minimum == maximum:
                return [minimum], None
            return [minimum], [minimum, maximum]
    return [int(choice) for choice in _choices(value, default=default)], None


def _legacy_mode(mode: str) -> str:
    return {"aggregated": "agg", "disaggregated": "disagg"}.get(mode, mode)


def _afd_search_space(afd: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "afd_phase": afd["phase"],
        "afd_batch_size_candidates": [int(value) for value in _choices(afd["a_batch_size"], default=[])],
        "afd_microbatch_candidates": [int(value) for value in _choices(afd.get("num_microbatches"), default=[2, 3, 4])],
        "afd_pipeline_model_candidates": [
            str(value)
            for value in _choices(
                afd.get("pipeline_model"),
                default=["optimistic", "conservative"],
            )
        ],
        "afd_comm_overhead_factor": afd.get("comm_overhead_factor", 1.0),
        "afd_boundary_on_attn": afd.get("boundary_on_attn", True),
        "afd_max_af_ratio": afd.get("max_af_ratio", 4.0),
        "afd_max_candidates": afd.get("max_candidates", 10_000),
    }
    if afd.get("tp_a") is not None:
        result["afd_tp_a_candidates"] = [int(value) for value in _choices(afd["tp_a"], default=[])]
    if afd.get("f_moe_ep_size") is not None:
        result["afd_f_moe_ep_size_candidates"] = _choices(afd["f_moe_ep_size"], default=[])
    return result


def _role_search_space(
    workers: dict[str, Any],
    modes: list[str],
    *,
    afd_phase: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "engine_float_ranges": {},
        "engine_log_ranges": [],
        "engine_log_discrete": [],
        "engine_integer_log_ranges": {},
    }
    specifications = (
        ("aggregated", "agg", "agg" in modes),
        (
            "prefill",
            "prefill",
            "disagg" in modes or ("afd+pd" in modes and afd_phase == "decode"),
        ),
        (
            "decode",
            "decode",
            "disagg" in modes or ("afd+pd" in modes and afd_phase == "prefill"),
        ),
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
        sequences_default = [1, 2, 4, 8, 16, 32, 64, 128, 256] if legacy_role == "prefill" else [256, 512, 1024]
        tokens_name = f"{legacy_role}_max_num_batched_tokens"
        sequences_name = f"{legacy_role}_max_num_seqs"
        tokens, tokens_log_range = _integer_domain(scheduler.get("max_batched_tokens"), default=tokens_default)
        sequences, sequences_log_range = _integer_domain(scheduler.get("max_sequences"), default=sequences_default)
        result[tokens_name] = tokens
        result[sequences_name] = sequences
        if tokens_log_range is not None:
            result["engine_integer_log_ranges"][tokens_name] = tokens_log_range
        if sequences_log_range is not None:
            result["engine_integer_log_ranges"][sequences_name] = sequences_log_range
        cache = raw.get("kv_cache") or {}
        capacity = cache.get("capacity") or {}
        block_value = cache.get("block_size")
        block_name = f"{legacy_role}_block_size"
        if block_value is None:
            result[block_name] = None
        elif isinstance(block_value, dict):
            block_choices, block_log_range = _integer_domain(block_value, default=[])
            result[block_name] = block_choices
            if block_log_range is not None:
                result["engine_integer_log_ranges"][block_name] = block_log_range
        else:
            result[block_name] = block_value
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
        result[f"{legacy_role}_enable_prefix_caching"] = cache.get("prefix_caching", True)
        result[f"{legacy_role}_kv_bytes_per_token"] = cache.get("bytes_per_token", "auto")
        result[f"{legacy_role}_native_host_offload"] = deepcopy(cache.get("host_offload"))
        capacity_type = capacity.get("type", "default")
        result[f"{legacy_role}_num_gpu_blocks"] = capacity.get("blocks") if capacity_type == "fixed" else None
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
        if timing.get("estimation_mode") is not None:
            result[f"{legacy_role}_forward_model"] = (
                "fpm" if timing["estimation_mode"] == "fpm_interpolation" else "op_level"
            )
        result[f"{legacy_role}_startup_time"] = raw.get("startup_seconds", 0)
    # Remove empty internal maps so legacy serialization remains concise.
    if not result["engine_float_ranges"]:
        result.pop("engine_float_ranges")
    if not result["engine_log_ranges"]:
        result.pop("engine_log_ranges")
    if not result["engine_log_discrete"]:
        result.pop("engine_log_discrete")
    if not result["engine_integer_log_ranges"]:
        result.pop("engine_integer_log_ranges")
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
            f"engine.workers.{role}.parallelism cannot combine preset with independent knobs {independent}"
        )
    if preset == "default":
        return "default", None
    if isinstance(preset, list):
        return "flat", [_parallel_mapping(entry, f"engine.workers.{role}.parallelism.preset") for entry in preset]
    if preset not in (False, {}):
        raise ValueError(f"invalid parallelism preset for role {role}")
    choices: dict[str, list[int] | None] = {}
    log_ranges: dict[str, list[int]] = {}
    for key in _PARALLEL_KEYS:
        if key not in parallel:
            choices[key] = None
            continue
        values, log_range = _integer_domain(parallel[key], default=[])
        choices[key] = values
        if log_range is not None:
            log_ranges[key] = log_range
    return "independent", {"choices": choices, "log_ranges": log_ranges}


def _parallel_mapping(value: Any, path: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} entries must be mappings")
    missing = set(_PARALLEL_KEYS) - set(value)
    unknown = set(value) - set(_PARALLEL_KEYS)
    if missing or unknown:
        raise ValueError(
            f"{path} entry must cover exactly {_PARALLEL_KEYS}; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    for key in _PARALLEL_KEYS:
        leaf = value[key]
        if type(leaf) is not int or leaf <= 0:
            raise ValueError(f"{path}.{key} must be a positive integer")
    return {key: value[key] for key in _PARALLEL_KEYS}


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
    dict[str, dict[str, list[int]]],
    dict[str, dict[str, list[dict[str, int]]]],
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
    independent_log_ranges: dict[str, dict[str, list[int]]] = {}
    custom: dict[str, dict[str, list[dict[str, int]]]] = {}
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
                for name, choices in value["choices"].items()
            }
            independent_log_ranges["agg"] = {
                {
                    "replicas": "replicas",
                    "tensor": "tp",
                    "pipeline": "pp",
                    "attention_data": "attention_dp",
                    "moe_tensor": "moe_tp",
                    "moe_expert": "moe_ep",
                }[name]: bounds
                for name, bounds in value["log_ranges"].items()
            }
    if "disagg" in modes:
        prefill_kind, prefill = role_specs["prefill"]
        decode_kind, decode = role_specs["decode"]
        if prefill_kind == decode_kind == "flat":
            pinned["disagg"] = [
                {"prefill": _legacy_parallel(p), "decode": _legacy_parallel(d)}
                for p, d in itertools.product(prefill, decode)
            ]
            flat_modes.append("disagg")
        elif prefill_kind != "default" or decode_kind != "default":
            combined: dict[str, list[int] | None] = {}
            combined_log_ranges: dict[str, list[int]] = {}
            mapping = {
                "replicas": "replicas",
                "tensor": "tp",
                "pipeline": "pp",
                "attention_data": "attention_dp",
                "moe_tensor": "moe_tp",
                "moe_expert": "moe_ep",
            }
            for role, kind, values in (
                ("prefill", prefill_kind, prefill),
                ("decode", decode_kind, decode),
            ):
                if kind == "independent":
                    for name, choices in values["choices"].items():
                        combined[f"{role}_{mapping[name]}"] = choices
                    for name, bounds in values["log_ranges"].items():
                        combined_log_ranges[f"{role}_{mapping[name]}"] = bounds
                elif kind == "flat":
                    custom.setdefault("disagg", {})[role] = [_legacy_parallel(entry) for entry in values]
            if combined:
                independent["disagg"] = combined
            if combined_log_ranges:
                independent_log_ranges["disagg"] = combined_log_ranges
    independent_log_ranges = {mode: fields for mode, fields in independent_log_ranges.items() if fields}
    return pinned, flat_modes, independent, independent_log_ranges, custom


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
        if source.get("nested_timestamp_basis") is not None:
            result["weka_nested_timestamp_basis"] = source["nested_timestamp_basis"]
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
            if load.get("agentic_lanes") is not None:
                result["agentic_lanes"] = load["agentic_lanes"]
        if isinstance(stop, dict) and stop.get("max_virtual_time_seconds") is not None:
            result["max_sim_time_ms"] = 1_000.0 * float(stop["max_virtual_time_seconds"])
        return result
    if source_type == "synthetic":
        result.update(isl=source.get("input_tokens", 1024), osl=source.get("output_tokens", 128))
        if source.get("images") is not None:
            result["images"] = deepcopy(source["images"])
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
        _configure_load_domain(
            result,
            load.get("fraction"),
            field="kv_load_ratio",
            integer=False,
        )
    else:
        raise ValueError(f"unsupported synthetic load type {load_type!r}")
    if count is not None:
        result["request_count"] = count
    else:
        result["num_request_ratio"] = ratio
    return result


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
        if bounds["min"] == bounds["max"]:
            result[field] = bounds["min"]
            return
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


def _goal(config: CoreRecommendationConfig) -> dict[str, Any]:
    target = config.optimization.target
    payload: dict[str, Any] = {
        "target": target,
        "strict_sla": config.optimization.strict_sla,
    }
    sla = config.evaluation.sla
    if sla is not None:
        payload["sla"] = sla.model_dump(mode="json", exclude_none=True)
    return payload


def _candidate_prediction(
    source: CoreRecommendationConfig,
    sample: dict[str, Any],
    replay_spec: ReplaySpec,
    *,
    adapter_sections: Mapping[str, str],
) -> dict[str, Any]:
    deployment = replay_spec.backend_deployment
    engine: dict[str, Any] = {
        "mode": (
            "aggregated"
            if deployment.deployment_mode == "agg"
            else "afd"
            if deployment.deployment_mode in {"afd", "afd+pd"}
            else "disaggregated"
        ),
        "model": sample["model_name"],
        "hardware": sample["hardware_sku"],
        "backend": sample["backend"],
        "backend_version": sample.get("backend_version") or None,
        "context_length": sample.get("context_length") or "max",
        "workers": {},
    }
    raw_engine = source.engine.model_dump(mode="python", exclude_none=True)
    if deployment.forward_pass_estimators:
        for name in (
            "database_mode",
            "transfer_policy",
            "systems_paths",
            "estimation_mode",
            "fallback_policy",
            "estimator_config",
        ):
            if name in raw_engine:
                engine[name] = deepcopy(raw_engine[name])
    if sample.get("speculation") is not None:
        engine["speculation"] = deepcopy(sample["speculation"])
    if deployment.encoder is not None:
        from .config.epd import encoder_prediction_fields

        engine["workers"]["encoder"] = encoder_prediction_fields(deployment.encoder)
    if deployment.deployment_mode in {"afd", "afd+pd"}:
        raw_afd = sample.get("afd")
        if not isinstance(raw_afd, dict):
            raise InfeasibleCandidate("AFD candidate omitted its concrete topology")
        engine["afd"] = {
            name: raw_afd[name]
            for name in (
                "phase",
                "combined_with_pd",
                "n_a_nodes",
                "n_f_nodes",
                "tp_a",
                "a_batch_size",
                "f_moe_ep_size",
                "num_microbatches",
                "pipeline_model",
                "comm_overhead_factor",
                "boundary_on_attn",
            )
        }
        roles = (str(sample["afd_companion_role"]),) if deployment.deployment_mode == "afd+pd" else ()
    else:
        roles = ("agg",) if deployment.deployment_mode == "agg" else ("prefill", "decode")
    for role in roles:
        prefix = "" if role == "agg" else f"{role}_"
        public_role = "aggregated" if role == "agg" else role
        raw_worker = (
            raw_engine.get("workers", {}).get(public_role, {}) if isinstance(raw_engine.get("workers"), dict) else {}
        )
        block_size = sample[f"{role}_block_size"]
        if block_size is None:
            block_size = {"vllm": 64, "sglang": 1, "trtllm": 32}[sample["backend"]]
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
            timing = {"type": "default", "forward_model": sample.get(f"{role}_forward_model") or "op_level"}
        estimator = deployment.forward_pass_estimators.get(role)
        if estimator is not None:
            resolved = estimator.config
            timing = {
                "type": "default",
                "estimation_mode": resolved["estimation_mode"],
                "fallback_policy": "deny",
                "estimator_config": deepcopy(resolved["estimator_config"]),
            }
            # Per-role roots can differ; preserve them next to the role's timing.
            timing["systems_paths"] = list(resolved["systems_paths"])
            timing["database_mode"] = resolved["database_mode"]
            policy = resolved["transfer_policy"]
            timing["transfer_policy"] = list(policy) if policy is not None else None
        kv_cache = {
            "block_size": block_size,
            "prefix_caching": sample[f"{role}_enable_prefix_caching"],
            "bytes_per_token": sample[f"{role}_kv_bytes_per_token"],
            "capacity": capacity,
        }
        role_args = (
            deployment.agg_engine_args
            if role == "agg"
            else deployment.prefill_engine_args
            if role == "prefill"
            else deployment.decode_engine_args
        )
        if isinstance(role_args, dict) and role_args.get("kv_cache_bytes_per_token") is not None:
            kv_cache["bytes_per_token"] = role_args["kv_cache_bytes_per_token"]
        if sample.get(f"{role}_native_host_offload") is not None:
            kv_cache["host_offload"] = deepcopy(sample[f"{role}_native_host_offload"])
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
            "kv_cache": kv_cache,
            "timing": timing,
            "startup_seconds": sample.get(f"{role}_startup_time")
            if sample.get(f"{role}_startup_time") is not None
            else raw_worker.get("startup_seconds", 0),
        }
        if deployment.deployment_mode == "disagg" and raw_worker.get("hardware") is not None:
            engine["workers"][public_role]["hardware"] = sample[f"{role}_hardware_sku"]
    if deployment.deployment_mode == "disagg" and raw_engine.get("kv_transfer") is not None:
        engine["kv_transfer"] = deepcopy(raw_engine["kv_transfer"])

    traffic = _candidate_traffic(
        source.traffic.model_dump(mode="python", exclude_none=True) if source.traffic is not None else None,
        sample,
    )
    concrete_adapters: dict[str, dict[str, Any]] = {}
    for name, adapter in replay_spec.adapters.items():
        section = adapter_sections.get(name)
        if section is None:
            raise ValueError(f"recommendation candidate used unknown config adapter {name!r}")
        concrete_adapters[section] = deepcopy(adapter.config)
    try:
        prediction = CorePredictionConfig.model_validate(
            {
                "traffic": traffic,
                "engine": engine,
                "evaluation": source.evaluation.model_dump(mode="python", exclude_none=True),
                "execution": source.execution.model_dump(mode="python", exclude_none=True),
            }
        )
    except ValueError as exc:
        raise InfeasibleCandidate(f"generated prediction config is infeasible: {exc}") from exc
    public = prediction.model_dump(mode="python", exclude_none=True)
    public.update(concrete_adapters)
    return public


def _candidate_traffic(raw: dict[str, Any] | None, sample: dict[str, Any]) -> dict[str, Any]:
    if raw is None:
        return TrafficPredictionConfig.default().model_dump(mode="python", exclude_none=True)
    traffic = deepcopy(raw)
    load = traffic["load"]
    if load.get("type") == "kv_capacity_fraction":
        load.clear()
        load.update(type="concurrency", concurrency=sample["concurrency"])
        return traffic
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
            else:
                raise ValueError(f"candidate did not materialize traffic.load.{name}")
    return traffic
