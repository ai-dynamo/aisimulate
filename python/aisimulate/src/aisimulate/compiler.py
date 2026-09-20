# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lower the public configuration model to runner and Sweeper contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .capacity import (
    estimate_kv_bytes_per_token,
    materialize_aic_num_gpu_blocks,
    resolve_model_context_length,
)
from .config.cli import CorePredictionConfig
from .config.common import ENGINE_MODEL_CONTROL_FIELDS, omit_inactive_moe_controls
from .config.engine import EnginePredictionConfig, WorkerPredictionConfig
from .config.traffic import SyntheticSessionSource, SyntheticSource, TraceSource
from .sweeper.afd_parallel import AFDParallelConfig, AFDTopology
from .sweeper.afd_perfmodel import (
    AFDPerformanceModel,
    AICAFDPerformanceModel,
    attach_afd_measurements,
)
from .sweeper.kv_estimate import resolve_backend_version
from .sweeper.model_hw import resolve_model_hardware
from .sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from .sweeper.provider import AdapterReplaySpec, JSONValue, validate_router_prefill_hardware
from .sweeper.replay import BackendDeploymentSpec, ReplaySpec


def prediction_to_replay_spec(
    config: CorePredictionConfig,
    *,
    adapter_specs: dict[str, AdapterReplaySpec] | None = None,
    afd_performance_model: AFDPerformanceModel | None = None,
    execution_mode: str = "offline",
) -> ReplaySpec:
    """Compile one concrete public prediction config."""

    if config.engine.speculation is not None and (adapter_specs or execution_mode != "offline"):
        raise ValueError("ngram speculation requires the offline engine stack without adapters")
    workload, concurrency = _traffic(config)
    deployment = _deployment(
        config.engine,
        workload=workload,
        afd_performance_model=afd_performance_model,
    )
    deployment = _pin_estimator_version_aliases(deployment)
    if config.engine.mode == "disaggregated":
        assert config.engine.workers.prefill is not None
        for adapter in (adapter_specs or {}).values():
            validate_router_prefill_hardware(adapter, config.engine.workers.prefill.hardware or config.engine.hardware)
    if config.engine.workers.encoder is not None:
        if adapter_specs or execution_mode != "offline":
            raise ValueError("analytical EPD requires the offline engine stack without adapters")
        deployment = replace(deployment, encoder=_prediction_encoder(config, workload))
    evaluation = config.evaluation.model_dump(mode="json", exclude_none=True)
    goal: dict[str, JSONValue] = {
        "sla": evaluation.get("sla") if evaluation else None,
    }
    if deployment.encoder is not None and goal["sla"] is not None:
        goal["strict_sla"] = True
    return ReplaySpec(
        backend_deployment=deployment,
        workload=workload,
        goal=goal,
        execution_mode=execution_mode,
        concurrency=concurrency,
        adapters=dict(adapter_specs or {}),
    )


def _pin_estimator_version_aliases(deployment: BackendDeploymentSpec) -> BackendDeploymentSpec:
    if deployment.backend_version not in {"current", "previous", "next"}:
        return deployment
    from aisimulate_core.sdk import RustForwardPassPerfModel

    updates = {}
    metadata = dict(deployment.performance_model_metadata)
    versions = set()
    for role, field in (
        ("aggregated", "agg_engine_args"),
        ("prefill", "prefill_engine_args"),
        ("decode", "decode_engine_args"),
    ):
        args = getattr(deployment, field)
        timing = (args or {}).get("timing_model", {})
        if timing.get("provider") != "aic" or "estimation_mode" not in timing.get("config", {}):
            continue
        config = dict(timing["config"])
        memory = {
            name: config.pop(name)
            for name in (
                "gpu_memory_utilization",
                "mem_fraction_static",
                "free_gpu_memory_fraction",
                "cuda_graph_reserved_bytes",
            )
            if name in config
        }
        model = RustForwardPassPerfModel.best_available(config)
        try:
            diagnostics = model.diagnostics()
        finally:
            model.close()
        if diagnostics["readiness"] != "ready":
            raise ValueError("regression estimator is not ready; replay requires training observations")
        resolved = diagnostics["provenance"]["config"]
        versions.add(resolved["backend_version"])
        updates[field] = {**args, "timing_model": {**timing, "config": {**resolved, **memory}}}
        metadata[role] = {"provider": "aic", "config": resolved, "selection": diagnostics}
    if len(versions) > 1:
        raise ValueError(
            f"estimator version alias resolves to different backend versions across roles: {sorted(versions)}"
        )
    if versions:
        updates.update(backend_version=versions.pop(), performance_model_metadata=metadata)
    return replace(deployment, **updates)


def _prediction_encoder(config: CorePredictionConfig, workload):
    from .sweeper.config import EncoderSearch, Workload
    from .sweeper.epd import resolve_encoder_pools

    engine = config.engine
    encoder = engine.workers.encoder
    assert encoder is not None
    shape = EncoderSearch(
        hardware_sku=encoder.hardware,
        backend_version=encoder.backend_version,
        tp=[encoder.tensor],
        batch_size=[encoder.batch_size],
        workers=[encoder.replicas],
        latency_correction=encoder.latency_correction,
        rate_degradation=encoder.rate_degradation,
    )
    pools = resolve_encoder_pools(
        model_name=engine.model,
        hardware_sku=engine.hardware,
        backends=[engine.backend],
        backend_version=engine.backend_version,
        context_length=(
            resolve_model_context_length(engine.model) if engine.context_length == "max" else engine.context_length
        ),
        encoder=shape,
        workload=Workload.model_validate(workload),
    )
    if len(pools) != 1:
        raise ValueError("prediction requires exactly one feasible encoder pool")
    return next(iter(pools.values()))


def _deployment(
    engine: EnginePredictionConfig,
    *,
    workload: dict[str, JSONValue],
    afd_performance_model: AFDPerformanceModel | None,
) -> BackendDeploymentSpec:
    if engine.mode == "afd":
        return _afd_deployment(
            engine,
            workload=workload,
            performance_model=afd_performance_model or AICAFDPerformanceModel(),
        )
    mode = "agg" if engine.mode == "aggregated" else "disagg"
    if mode == "disagg":
        from aisimulate_core.sdk.perf_database import load_system_spec

        from .sweeper.forward_pass_estimator import resolve_systems_paths

        workers = (engine.workers.prefill, engine.workers.decode)
        root_kwargs = {}
        for role, worker in zip(("prefill", "decode"), workers, strict=True):
            paths = engine.systems_paths
            if worker is not None and worker.timing.systems_paths is not None:
                paths = worker.timing.systems_paths
            root_kwargs[role] = {"systems_paths": list(resolve_systems_paths(paths))}
            if (
                worker is not None
                and worker.hardware is not None
                and not load_system_spec(worker.hardware, **root_kwargs[role])
            ):
                raise ValueError(f"unknown workers.{role}.hardware {worker.hardware!r}: no system configuration found")
        if engine.backend_version is None and any(
            worker is not None and (worker.hardware is not None or worker.timing.systems_paths is not None)
            for worker in workers
        ):
            versions = {
                role: resolve_backend_version(worker.hardware or engine.hardware, engine.backend, **root_kwargs[role])
                for role, worker in zip(("prefill", "decode"), workers, strict=True)
                if worker is not None
            }
            if len(set(versions.values())) != 1:
                raise ValueError(
                    "heterogeneous P/D hardware requires one common backend_version; "
                    f"latest versions for backend={engine.backend!r} are {versions}. "
                    "Set engine.backend_version to a version supported by both SKUs."
                )
            engine = engine.model_copy(update={"backend_version": next(iter(versions.values()))})
    common: dict[str, Any] = {
        "deployment_mode": mode,
        "backend": engine.backend,
        "backend_version": engine.backend_version or "",
    }
    if mode == "agg":
        assert engine.workers.aggregated is not None
        worker, vl_metadata = _resolve_host_profile(engine, engine.workers.aggregated, workload)
        parallel = _parallel_mapping(worker, prefix="")
        return BackendDeploymentSpec(
            parallel_config=parallel,
            performance_model_metadata={
                "aggregated": {**_worker_performance_model_metadata(engine, worker), **vl_metadata},
            },
            agg_engine_args=_worker_engine_args(
                engine,
                worker,
                "aggregated",
                transfer_bytes_per_token=None,
                # Images without a dedicated encoder pool are encoded on the language worker.
                vision=workload.get("images") is not None and engine.workers.encoder is None,
            ),
            num_workers=worker.parallelism.replicas,
            **common,
        )
    assert engine.workers.prefill is not None and engine.workers.decode is not None
    prefill = engine.workers.prefill
    decode = engine.workers.decode
    transfer_bytes_per_token = None
    if engine.kv_transfer is not None:
        transfer_bytes_per_token = _resolve_kv_bytes_per_token(
            engine,
            prefill,
            engine.kv_transfer.bytes_per_token,
        )
    parallel = {
        **_parallel_mapping(prefill, prefix="prefill_"),
        **_parallel_mapping(decode, prefix="decode_"),
    }
    return BackendDeploymentSpec(
        parallel_config=parallel,
        performance_model_metadata={
            "prefill": _worker_performance_model_metadata(engine, prefill),
            "decode": _worker_performance_model_metadata(engine, decode),
        },
        prefill_engine_args=_worker_engine_args(
            engine, prefill, "prefill", transfer_bytes_per_token=transfer_bytes_per_token
        ),
        decode_engine_args=_worker_engine_args(
            engine, decode, "decode", transfer_bytes_per_token=transfer_bytes_per_token
        ),
        num_prefill_workers=prefill.parallelism.replicas,
        num_decode_workers=decode.parallelism.replicas,
        **common,
    )


def _afd_deployment(
    engine: EnginePredictionConfig,
    *,
    workload: dict[str, JSONValue],
    performance_model: AFDPerformanceModel,
) -> BackendDeploymentSpec:
    afd = engine.afd
    assert afd is not None
    facts = resolve_model_hardware(engine.model, engine.hardware, backend=engine.backend)
    topology = AFDTopology(
        **afd.model_dump(mode="python"),
        gpus_per_node=facts.gpus_per_node,
        is_moe=facts.is_moe,
        num_experts=facts.num_experts,
    )
    backend_version = engine.backend_version or resolve_backend_version(engine.hardware, engine.backend)
    resolved_engine = engine.model_copy(update={"backend_version": backend_version})
    companion_role = "decode" if topology.phase.value == "prefill" else "prefill"
    companion_worker = getattr(engine.workers, companion_role) if topology.combined_with_pd else None
    companion: ReplicaParallelConfig | None = None
    if companion_worker is not None:
        parallel = companion_worker.parallelism
        companion = ReplicaParallelConfig(
            shape=ParallelShape(
                tp=parallel.tensor,
                pp=parallel.pipeline,
                dp=parallel.attention_data,
                moe_tp=parallel.moe_tensor,
                moe_ep=parallel.moe_expert,
            ),
            replicas=parallel.replicas,
        )
    afd_parallel = AFDParallelConfig(topology=topology, companion=companion)
    parallel_config: dict[str, JSONValue] = {
        "afd": topology.provenance()["topology"],
        "afd_provenance": afd_parallel.provenance(),
    }
    performance_metadata: dict[str, dict[str, JSONValue]] = {
        "afd": {
            "provider": "unresolved",
            "measurement_required": True,
            "config": topology.provenance()["topology"],
            "provenance": afd_parallel.provenance(),
        }
    }
    deployment_kwargs: dict[str, Any] = {}
    if companion_worker is not None:
        assert companion is not None
        prefix = f"{companion_role}_"
        shape = companion.shape
        parallel_config.update(
            {
                f"{prefix}tp": shape.tp,
                f"{prefix}pp": shape.pp,
                f"{prefix}attention_dp": shape.dp,
                f"{prefix}moe_tp": shape.moe_tp,
                f"{prefix}moe_ep": shape.moe_ep,
                f"{prefix}strategy": shape.strategy,
                f"{prefix}replicas": companion.replicas,
            }
        )
        companion_metadata = _worker_performance_model_metadata(resolved_engine, companion_worker)
        performance_metadata[companion_role] = companion_metadata
        deployment_kwargs[f"{companion_role}_engine_args"] = _worker_engine_args(
            resolved_engine,
            companion_worker,
            companion_role,
            transfer_bytes_per_token=None,
        )
        deployment_kwargs[f"num_{companion_role}_workers"] = companion.replicas
    deployment = BackendDeploymentSpec(
        deployment_mode=topology.adapter_topology,
        backend=engine.backend,
        backend_version=backend_version,
        parallel_config=parallel_config,
        performance_model_metadata=performance_metadata,
        **deployment_kwargs,
    )
    sample = {
        "model_name": engine.model,
        "hardware_sku": engine.hardware,
        "backend": engine.backend,
        "backend_version": backend_version,
        "context_length": (
            engine.context_length
            if isinstance(engine.context_length, int)
            else resolve_model_context_length(engine.model)
        ),
        "afd": topology.provenance()["topology"],
    }
    return attach_afd_measurements(
        deployment,
        sample=sample,
        workload=workload,
        performance_model=performance_model,
    )


def _parallel_mapping(worker: WorkerPredictionConfig, *, prefix: str) -> dict[str, JSONValue]:
    parallel = worker.parallelism
    return {
        f"{prefix}replicas": parallel.replicas,
        f"{prefix}tp": parallel.tensor,
        f"{prefix}pp": parallel.pipeline,
        f"{prefix}attention_dp": parallel.attention_data,
        f"{prefix}moe_tp": parallel.moe_tensor,
        f"{prefix}moe_ep": parallel.moe_expert,
    }


def _resolve_host_profile(
    engine: EnginePredictionConfig, worker: WorkerPredictionConfig, workload: dict[str, JSONValue]
) -> tuple[WorkerPredictionConfig, dict[str, JSONValue]]:
    """Fill the worker's host and frontend tables from its measured profile, if it names one."""
    if worker.host_profile is None:
        return worker, {}
    from .vl.profile import profile_digest, resolve_host_profile

    images = workload.get("images")
    if not isinstance(images, dict):
        raise ValueError("host_profile requires an image workload")
    profile, host, frontend = resolve_host_profile(
        worker.host_profile,
        model=engine.model,
        images=images,
        tensor_parallel=worker.parallelism.tensor,
        text_tokens=workload.get("isl"),
    )
    resolved = worker.model_copy(update={"host": host, "frontend": frontend, "host_profile": None})
    return resolved, {"vl": {"host_profile_digest": profile_digest(profile), "frontend": worker.host_profile.frontend}}


def _worker_performance_model_metadata(
    engine: EnginePredictionConfig, worker: WorkerPredictionConfig
) -> dict[str, JSONValue]:
    parallel = worker.parallelism
    sharded_moe = parallel.moe_tensor * parallel.moe_expert > 1
    return {
        "provider": "aic",
        "config": {
            "backend": engine.backend,
            "backend_version": engine.backend_version,
            "system": worker.hardware or engine.hardware,
            "model_path": engine.model,
            "tp_size": parallel.tensor,
            "attention_dp_size": parallel.attention_data,
            "moe_tp_size": parallel.moe_tensor if sharded_moe else None,
            "moe_ep_size": parallel.moe_expert if sharded_moe else None,
            "nextn": engine.nextn or None,
            **({"speculation": engine.speculation.cost_config()} if engine.speculation is not None else {}),
            "forward_model": worker.timing.forward_model,
            **{
                name: getattr(engine, name)
                for name in ENGINE_MODEL_CONTROL_FIELDS
                if getattr(engine, name) not in (None, False)
            },
            **({"decoder_replay": True} if engine.decoder_replay else {}),
            **{
                field: getattr(engine, field)
                for field in ("database_mode", "enable_shared_layer", "strict_provenance")
                if getattr(engine, field) is not None
            },
        },
    }


def _worker_engine_args(
    engine: EnginePredictionConfig,
    worker: WorkerPredictionConfig,
    role: str,
    *,
    transfer_bytes_per_token: int | None,
    vision: bool = False,
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
        "aic_system": worker.hardware or engine.hardware,
        "aic_model_path": engine.model,
        "aic_tp_size": parallel.tensor,
        "aic_attention_dp_size": parallel.attention_data,
        "max_num_batched_tokens": worker.scheduler.max_batched_tokens,
        "max_num_seqs": worker.scheduler.max_sequences,
        "prefill_schedule_interval": worker.scheduler.prefill_schedule_interval,
        "prefill_decode_interval": worker.scheduler.prefill_decode_interval,
        "block_size": block_size,
        "enable_prefix_caching": cache.prefix_caching,
        "startup_time": worker.startup_seconds,
    }
    if engine.speculation is not None:
        payload["speculation"] = engine.speculation.model_dump(mode="json")
    if engine.backend_version is not None:
        payload["aic_backend_version"] = engine.backend_version
    if engine.decoder_replay:
        payload["aic_decoder_replay"] = True
    for field in ("database_mode", "enable_shared_layer", "strict_provenance"):
        value = getattr(engine, field)
        if value is not None:
            payload[f"aic_{field}"] = value
    if parallel.pipeline != 1:
        payload["aic_pp_size"] = parallel.pipeline
    if parallel.moe_tensor * parallel.moe_expert > 1:
        payload["aic_moe_tp_size"] = parallel.moe_tensor
        payload["aic_moe_ep_size"] = parallel.moe_expert
    if worker.timing.type == "default" and worker.timing.forward_model != "op_level":
        # Only the non-default forward model is spelled out, so op_level specs stay byte-identical.
        payload["aic_forward_model"] = worker.timing.forward_model
    if backend == "vllm" or isinstance(engine.context_length, int):
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
        payload["cuda_graph_reserved_bytes"] = capacity.cuda_graph_reserved_bytes
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
    if worker.timing.type == "default" and engine.mode != "afd" and engine.workers.encoder is None:
        from aisimulate_core.sdk import ForwardPassPerfModelConfig

        from .sweeper.forward_pass_estimator import resolve_systems_paths

        timing = worker.timing
        sharded_moe = parallel.moe_tensor * parallel.moe_expert > 1
        canonical = ForwardPassPerfModelConfig(
            model=engine.model,
            system=worker.hardware or engine.hardware,
            backend=backend,
            backend_version=engine.backend_version,
            worker_type=role,
            decoder_replay=engine.decoder_replay,
            enable_shared_layer=engine.enable_shared_layer,
            strict_provenance=bool(engine.strict_provenance),
            tp=parallel.tensor,
            pp=parallel.pipeline,
            attention_dp=parallel.attention_data,
            moe_tp_size=parallel.moe_tensor if sharded_moe else None,
            moe_ep_size=parallel.moe_expert if sharded_moe else None,
            kv_block_size=block_size,
            nextn=engine.nextn,
            **{name: getattr(engine, name) for name in ENGINE_MODEL_CONTROL_FIELDS},
            speculation=engine.speculation.cost_config() if engine.speculation is not None else None,
            estimation_mode=timing.estimation_mode or engine.estimation_mode,
            fallback_policy=timing.fallback_policy or engine.fallback_policy,
            estimator_config=timing.estimator_config
            if timing.estimator_config is not None
            else engine.estimator_config,
            database_mode=timing.database_mode or engine.database_mode,
            transfer_policy=timing.transfer_policy if timing.transfer_policy is not None else engine.transfer_policy,
            systems_paths=resolve_systems_paths(timing.systems_paths or engine.systems_paths),
            # A rank hosting the vision tower prices encoder calls through the same estimator.
            encoder_parallel=(worker.vision.encoder_parallel if worker.vision is not None else "tp")
            if vision
            else None,
        )
        timing_config = omit_inactive_moe_controls(canonical.to_dict())
        for key in (
            "gpu_memory_utilization",
            "mem_fraction_static",
            "free_gpu_memory_fraction",
            "cuda_graph_reserved_bytes",
        ):
            if key in payload:
                timing_config[key] = payload[key]
        payload["timing_model"] = {"type": "external", "provider": "aic", "config": timing_config}
        payload["tensor_parallel_size"] = parallel.tensor
        payload["dp_size"] = parallel.attention_data
        for key in tuple(payload):
            if key.startswith("aic_") and key != "aic_nextn":
                payload.pop(key)
    if engine.enable_chunked_prefill is not None and role != "decode":
        payload["enable_chunked_prefill"] = engine.enable_chunked_prefill
    if engine.nextn:
        payload["aic_nextn"] = engine.nextn
        payload["aic_nextn_accepted"] = engine.nextn_accepted
    host_offload = cache.host_offload
    if host_offload is not None:
        payload["kv_cache_bytes_per_token"] = _resolve_kv_bytes_per_token(
            engine,
            worker,
            cache.bytes_per_token,
        )
    if transfer_bytes_per_token is not None:
        payload["kv_transfer_bytes_per_token"] = transfer_bytes_per_token
    if host_offload is not None:
        payload["native_host_offload"] = host_offload.model_dump(mode="json")
    if cache.g3_offload is not None:
        payload["g3_offload"] = cache.g3_offload.model_dump(mode="json")
    if engine.kv_transfer is not None:
        transfer = engine.kv_transfer
        if transfer.bandwidth_gb_per_second is not None:
            payload["kv_transfer_bandwidth"] = transfer.bandwidth_gb_per_second
        payload["kv_transfer_timing_mode"] = transfer.timing_mode
    if backend == "sglang":
        # SGLang reads its batch budget from `sglang.*`; the public token knob only
        # becomes effective on the host-aware VL path so existing predictions keep
        # their bytes.
        sglang: dict[str, JSONValue] = {}
        if worker.host is not None or vision or worker.scheduler.max_prefill_tokens is not None:
            sglang["chunked_prefill_size"] = worker.scheduler.max_batched_tokens
        if worker.scheduler.max_prefill_tokens is not None:
            sglang["max_prefill_tokens"] = worker.scheduler.max_prefill_tokens
        if vision:
            sglang["vlm_cache_bytes"] = (worker.vision.cache_mib if worker.vision is not None else 100) << 20
            payload["vision"] = True
        if worker.host is not None:
            sglang["host"] = worker.host.model_dump(mode="json")
        if sglang:
            payload["sglang"] = sglang
        if worker.frontend is not None:
            payload["frontend"] = worker.frontend.model_dump(mode="json")
    return payload


def _resolve_kv_bytes_per_token(
    engine: EnginePredictionConfig,
    worker: WorkerPredictionConfig,
    configured: int | str,
) -> int:
    if configured != "auto":
        return configured
    parallel = worker.parallelism
    return estimate_kv_bytes_per_token(
        engine.model,
        tp_size=parallel.tensor,
        pp_size=parallel.pipeline,
        moe_tp_size=parallel.moe_tensor,
        moe_ep_size=parallel.moe_expert,
        **({"kvcache_quant_mode": engine.kvcache_quant_mode} if engine.kvcache_quant_mode else {}),
    )


def _traffic(
    config: CorePredictionConfig,
) -> tuple[dict[str, JSONValue], int | None]:
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
        if source.nested_timestamp_basis is not None:
            workload["weka_nested_timestamp_basis"] = source.nested_timestamp_basis
        if load.type == "concurrency":
            workload["replay_concurrency"] = load.concurrency
        else:
            workload["arrival_speedup_ratio"] = load.speedup or 1.0
            if load.agentic_lanes is not None:
                workload["agentic_lanes"] = load.agentic_lanes
        if stop is not None and stop.max_virtual_time_seconds is not None:
            workload["max_sim_time_ms"] = 1_000.0 * stop.max_virtual_time_seconds
        return workload, None

    if isinstance(source, SyntheticSource):
        workload.update(
            isl=source.input_tokens, osl=source.output_tokens, cached_prefix_tokens=source.cached_prefix_tokens
        )
        if source.images is not None:
            workload["images"] = source.images.model_dump(mode="json")
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
