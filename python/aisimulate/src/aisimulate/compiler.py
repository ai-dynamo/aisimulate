# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower the public configuration model to runner and Sweeper contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .aic import (
    estimate_kv_bytes_per_token,
    materialize_aic_num_gpu_blocks,
    resolve_model_context_length,
)
from .config_adapter import PredictionAdapterContext, SimulationConfigAdapter
from .public_config import (
    EngineConfig,
    PredictionConfig,
    SyntheticSessionSource,
    SyntheticSource,
    TraceSource,
    WorkerConfig,
)
from .sweeper.provider import AdapterReplaySpec, JSONValue
from .sweeper.replay import BackendDeploymentSpec, ReplaySpec


def prediction_to_replay_spec(
    config: PredictionConfig,
    *,
    stack: str,
    adapters: Mapping[str, SimulationConfigAdapter] | None = None,
) -> ReplaySpec:
    """Compile one concrete public prediction config."""

    deployment = _deployment(config.engine)
    workload, concurrency = _traffic(config)
    evaluation = config.evaluation.model_dump(mode="json", exclude_none=True)
    goal: dict[str, JSONValue] = {
        "sla": evaluation.get("sla") if evaluation else None,
    }
    adapter_specs: dict[str, AdapterReplaySpec] = {}
    requested: list[tuple[str, dict[str, JSONValue]]] = []
    router = config.router.model_dump(mode="json", exclude_none=True)
    if config.router.policy != "round_robin":
        requested.append(("router", router))
    planner = config.planner.model_dump(mode="json", exclude_none=True)
    if config.planner.policy != "disabled":
        requested.append(("planner", planner))

    available = dict(adapters or {})
    context = PredictionAdapterContext(
        engine=config.engine.model_dump(mode="json", exclude_none=True),
        traffic=config.traffic.model_dump(mode="json", exclude_none=True),
        evaluation=evaluation,
    )
    for section, section_config in requested:
        name = f"{stack}.{section}"
        adapter = available.get(name)
        if adapter is None:
            raise ValueError(
                f"stack {stack!r} does not provide the active {section} adapter {name!r}"
            )
        adapter_specs[name] = adapter.materialize_prediction(
            section_config, context
        )

    return ReplaySpec(
        backend_deployment=deployment,
        workload=workload,
        goal=goal,
        concurrency=concurrency,
        adapters=adapter_specs,
    )


def _deployment(engine: EngineConfig) -> BackendDeploymentSpec:
    mode = "agg" if engine.mode == "aggregated" else "disagg"
    common: dict[str, Any] = {
        "deployment_mode": mode,
        "backend": engine.backend,
        "backend_version": engine.backend_version or "",
    }
    if mode == "agg":
        assert engine.workers.aggregated is not None
        worker = engine.workers.aggregated
        parallel = _parallel_mapping(worker, prefix="")
        return BackendDeploymentSpec(
            parallel_config=parallel,
            agg_engine_args=_worker_engine_args(engine, worker, "aggregated"),
            num_workers=worker.parallelism.replicas,
            **common,
        )
    assert engine.workers.prefill is not None and engine.workers.decode is not None
    prefill = engine.workers.prefill
    decode = engine.workers.decode
    parallel = {
        **_parallel_mapping(prefill, prefix="prefill_"),
        **_parallel_mapping(decode, prefix="decode_"),
    }
    return BackendDeploymentSpec(
        parallel_config=parallel,
        prefill_engine_args=_worker_engine_args(engine, prefill, "prefill"),
        decode_engine_args=_worker_engine_args(engine, decode, "decode"),
        num_prefill_workers=prefill.parallelism.replicas,
        num_decode_workers=decode.parallelism.replicas,
        **common,
    )


def _parallel_mapping(worker: WorkerConfig, *, prefix: str) -> dict[str, JSONValue]:
    parallel = worker.parallelism
    return {
        f"{prefix}replicas": parallel.replicas,
        f"{prefix}tp": parallel.tensor,
        f"{prefix}pp": parallel.pipeline,
        f"{prefix}attention_dp": parallel.attention_data,
        f"{prefix}moe_tp": parallel.moe_tensor,
        f"{prefix}moe_ep": parallel.moe_expert,
    }


def _worker_engine_args(
    engine: EngineConfig, worker: WorkerConfig, role: str
) -> dict[str, JSONValue]:
    backend = engine.backend
    parallel = worker.parallelism
    cache = worker.kv_cache
    capacity = cache.capacity
    memory_fraction = capacity.memory_fraction
    if capacity.type == "default" and memory_fraction is None:
        memory_fraction = 0.88 if backend == "sglang" else 0.9
    block_size = cache.block_size
    if block_size is None:
        block_size = {"vllm": 64, "sglang": 1, "trtllm": 32}[backend]
    payload: dict[str, JSONValue] = {
        "worker_type": role,
        "engine_type": backend,
        "aic_backend": backend,
        "aic_system": engine.hardware,
        "aic_model_path": engine.model,
        "aic_tp_size": parallel.tensor,
        "aic_attention_dp_size": parallel.attention_data,
        "max_num_batched_tokens": worker.scheduler.max_batched_tokens,
        "max_num_seqs": worker.scheduler.max_sequences,
        "block_size": block_size,
        "enable_prefix_caching": cache.prefix_caching,
        "startup_time": worker.startup_seconds,
    }
    if engine.backend_version is not None:
        payload["aic_backend_version"] = engine.backend_version
    if parallel.pipeline != 1:
        payload["aic_pp_size"] = parallel.pipeline
    if parallel.moe_tensor * parallel.moe_expert > 1:
        payload["aic_moe_tp_size"] = parallel.moe_tensor
        payload["aic_moe_ep_size"] = parallel.moe_expert
    if backend == "vllm":
        payload["max_model_len"] = (
            engine.context_length
            if isinstance(engine.context_length, int)
            else resolve_model_context_length(engine.model)
        )
    if capacity.type == "fixed":
        assert capacity.blocks is not None
        payload["num_gpu_blocks"] = capacity.blocks
    else:
        assert memory_fraction is not None
        payload[
            {
                "vllm": "gpu_memory_utilization",
                "sglang": "mem_fraction_static",
                "trtllm": "free_gpu_memory_fraction",
            }[backend]
        ] = memory_fraction
    if worker.timing.type == "fixed":
        payload["timing_model"] = {
            "type": "fixed",
            "prefill_ms": worker.timing.prefill_ms,
            "decode_ms": worker.timing.decode_ms,
        }
    elif worker.timing.type == "polynomial":
        payload["timing_model"] = {"type": "polynomial"}
    if worker.timing.type != "default":
        if capacity.type == "default":
            payload = materialize_aic_num_gpu_blocks(payload)
        for name in (
            "aic_backend_version",
            "aic_system",
            "aic_model_path",
            "aic_moe_tp_size",
            "aic_moe_ep_size",
        ):
            payload.pop(name, None)
    if engine.kv_transfer is not None:
        transfer = engine.kv_transfer
        payload["kv_bytes_per_token"] = (
            estimate_kv_bytes_per_token(
                engine.model,
                tp_size=parallel.tensor,
                pp_size=parallel.pipeline,
                moe_tp_size=parallel.moe_tensor,
                moe_ep_size=parallel.moe_expert,
            )
            if transfer.bytes_per_token == "auto"
            else transfer.bytes_per_token
        )
        if transfer.bandwidth_gb_per_second is not None:
            payload["kv_transfer_bandwidth"] = transfer.bandwidth_gb_per_second
        payload["kv_transfer_timing_mode"] = transfer.timing_mode
    return payload


def _traffic(config: PredictionConfig) -> tuple[dict[str, JSONValue], int | None]:
    traffic = config.traffic
    source = traffic.source
    load = traffic.load
    stop = traffic.stop
    workload: dict[str, JSONValue] = {
        "source_type": source.type,
        "load_type": load.type,
    }
    concurrency: int | None = None
    if isinstance(source, TraceSource):
        workload.update(
            trace_paths=list(source.paths),
            trace_path=source.paths[0],
            trace_format=source.format,
            trace_block_size=source.block_size,
        )
        if load.type == "concurrency":
            workload["replay_concurrency"] = load.concurrency
        else:
            workload["arrival_speedup_ratio"] = load.speedup or 1.0
        if stop is not None and stop.max_virtual_time_seconds is not None:
            workload["max_sim_time_ms"] = 1_000.0 * stop.max_virtual_time_seconds
        return workload, None

    if isinstance(source, SyntheticSource):
        workload.update(isl=source.input_tokens, osl=source.output_tokens)
        stop_count = stop.requests if stop is not None else None
        relative = stop.requests_per_load_unit if stop is not None else None
    else:
        assert isinstance(source, SyntheticSessionSource)
        workload.update(
            isl=source.new_input_tokens_per_turn,
            osl=source.output_tokens_per_turn,
            turns_per_session=source.session.turns,
            shared_prefix_ratio=source.session.shared_prefix_ratio,
            num_prefix_groups=source.session.prefix_groups,
            inter_turn_delay_ms=source.session.inter_turn_delay_ms,
        )
        stop_count = stop.sessions if stop is not None else None
        relative = stop.sessions_per_load_unit if stop is not None else None

    if load.type == "concurrency":
        concurrency = load.concurrency
        workload["concurrency"] = concurrency
        load_unit = float(concurrency or 0)
    elif load.type == "poisson":
        rate = load.requests_per_second or load.sessions_per_second
        workload["request_rate"] = rate
        workload["arrival_seed"] = load.seed if load.seed is not None else 42
        load_unit = float(rate or 0.0)
    elif load.type == "constant_rate":
        rate = load.requests_per_second or load.sessions_per_second
        workload["arrival_interval_ms"] = 1_000.0 / float(rate or 0.0)
        load_unit = float(rate or 0.0)
    else:
        # Candidate-relative materialization is shared with the Sweeper. A
        # concrete predict compiler retains the requested fraction so the
        # runner can derive the in-flight cap from concrete KV capacity.
        workload["kv_load_ratio"] = load.fraction
        load_unit = float(load.fraction or 0.0)
    if stop_count is not None:
        workload["request_count"] = stop_count
    else:
        assert relative is not None
        workload["num_request_ratio"] = relative
        if load.type != "kv_capacity_fraction":
            workload["request_count"] = max(1, round(relative * load_unit))
    return workload, concurrency
