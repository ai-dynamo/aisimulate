# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified native port; original paths and license: root THIRD_PARTY_NOTICES.md.
# Source: https://github.com/LPhgh/aisimulate/tree/af56328c4ba7ffa1b8c42c8375244568c2e7bb8c

"""Lower the public configuration model to runner and Sweeper contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from aisimulate_core._native import validate_afd_attention_tp, validate_afd_routing_location

from .capacity import (
    estimate_kv_bytes_per_token,
    materialize_aic_num_gpu_blocks,
    reject_pipeline_parallelism,
    resolve_model_context_length,
)
from .config.cli import CorePredictionConfig
from .config.common import ENGINE_MODEL_CONTROL_FIELDS, omit_inactive_moe_controls
from .config.engine import (
    EnginePredictionConfig,
    FrontendMeasurementConfig,
    FrontendPredictionConfig,
    HostWorkerPredictionConfig,
    VisionPredictionConfig,
    WorkerPredictionConfig,
    has_afd,
    is_afd_worker,
    resolve_block_size,
)
from .config.traffic import SyntheticSessionSource, SyntheticSource, TraceSource
from .estimator_readiness import unready_estimator_message
from .state_size import resolve_state_size
from .sweeper.afd_parallel import AFDTopology, validate_afd_model
from .sweeper.kv_estimate import resolve_backend_version
from .sweeper.model_hw import resolve_model_hardware
from .sweeper.provider import AdapterReplaySpec, JSONValue, validate_router_prefill_hardware
from .sweeper.replay import BackendDeploymentSpec, ReplaySpec, resolve_afd_replay_identity


def prediction_to_replay_spec(
    config: CorePredictionConfig,
    *,
    adapter_specs: dict[str, AdapterReplaySpec] | None = None,
    execution_mode: str = "offline",
) -> ReplaySpec:
    """Compile one concrete public prediction config."""

    if (
        config.engine.speculation is not None
        and config.engine.speculation.kind == "ngram"
        and (adapter_specs or execution_mode != "offline")
    ):
        raise ValueError("ngram speculation requires the offline engine stack without adapters")
    if config.engine.speculation is not None and execution_mode != "offline":
        raise ValueError("speculation requires offline execution")
    if has_afd(config.engine.workers) and (adapter_specs or execution_mode != "offline"):
        raise ValueError("AFD roles require the offline engine stack without adapters")
    for role in ("aggregated", "prefill", "decode"):
        worker = getattr(config.engine.workers, role)
        if is_afd_worker(worker):
            validate_afd_routing_location(worker.afd.routing_location)
            validate_afd_attention_tp(worker.afd.tp_a)
    workload, concurrency = _traffic(config)
    deployment = _deployment(config.engine, workload=workload)
    deployment = _pin_estimator_version_aliases(deployment)
    metadata = dict(deployment.performance_model_metadata)
    for role, field in (
        ("aggregated", "agg_engine_args"),
        ("prefill", "prefill_engine_args"),
        ("decode", "decode_engine_args"),
    ):
        timing = (getattr(deployment, field) or {}).get("timing_model", {})
        if timing.get("provider") == "aic" and "estimation_mode" in timing.get("config", {}):
            identity = {
                key: value
                for key, value in timing["config"].items()
                if key
                not in {
                    "gpu_memory_utilization",
                    "mem_fraction_static",
                    "free_gpu_memory_fraction",
                    "cuda_graph_reserved_bytes",
                }
            }
            metadata[role] = {**metadata.get(role, {}), "provider": "aic", "config": identity}
    deployment = replace(deployment, performance_model_metadata=metadata)
    if config.engine.mode == "disaggregated":
        assert config.engine.workers.prefill is not None
        for adapter in (adapter_specs or {}).values():
            validate_router_prefill_hardware(
                adapter,
                (
                    config.engine.workers.prefill.afd.a_system
                    if is_afd_worker(config.engine.workers.prefill)
                    else config.engine.workers.prefill.hardware
                )
                or config.engine.hardware,
            )
    if config.engine.workers.encoder is not None:
        if adapter_specs or execution_mode != "offline":
            raise ValueError("encoder pools require the offline engine stack without adapters")
        pool = _prediction_encoder(config, workload)
        metadata = dict(deployment.performance_model_metadata)
        if pool.native is not None:
            from .config.epd import encoder_metadata

            metadata["encoder"] = encoder_metadata(pool)
        deployment = replace(deployment, encoder=pool, performance_model_metadata=metadata)
    evaluation = config.evaluation.model_dump(mode="json", exclude_none=True)
    goal: dict[str, JSONValue] = {
        "sla": evaluation.get("sla") if evaluation else None,
    }
    if deployment.encoder is not None and deployment.encoder.mode == "analytical" and goal["sla"] is not None:
        goal["strict_sla"] = True
    return resolve_afd_replay_identity(
        ReplaySpec(
            backend_deployment=deployment,
            workload=workload,
            goal=goal,
            execution_mode=execution_mode,
            concurrency=concurrency,
            adapters=dict(adapter_specs or {}),
        )
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
            raise ValueError(unready_estimator_message(diagnostics))
        resolved = diagnostics["provenance"]["config"]
        versions.add(resolved["backend_version"])
        updates[field] = {**args, "timing_model": {**timing, "config": {**resolved, **memory}}}
        metadata[role] = {**metadata.get(role, {}), "provider": "aic", "config": resolved, "selection": diagnostics}
    if len(versions) > 1:
        raise ValueError(
            f"estimator version alias resolves to different backend versions across roles: {sorted(versions)}"
        )
    if versions:
        updates.update(backend_version=versions.pop(), performance_model_metadata=metadata)
    return replace(deployment, **updates)


def _prediction_encoder(config: CorePredictionConfig, workload):
    from .sweeper.config import EncoderSearch
    from .sweeper.epd import replay_workload, resolve_encoder_pools

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
        mode=encoder.mode,
        host_profile=encoder.host_profile.model_dump(mode="json") if encoder.host_profile is not None else None,
        frontend=encoder.frontend.model_dump(mode="json") if encoder.frontend is not None else None,
        transfer_bandwidth_gb_per_second=(
            encoder.transfer.bandwidth_gb_per_second if encoder.transfer is not None else None
        ),
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
        workload=replay_workload(workload),
    )
    if len(pools) != 1:
        raise ValueError("prediction requires exactly one feasible encoder pool")
    return next(iter(pools.values()))


def _tokenization(engine: EnginePredictionConfig) -> dict[str, JSONValue] | None:
    tokenization = getattr(engine, "tokenization", None)
    if tokenization is None or (tokenization.base_ms == 0 and tokenization.per_input_token_us == 0):
        return None
    return {
        "workers": tokenization.workers,
        "base_ms": tokenization.base_ms,
        "per_input_token_ms": tokenization.per_input_token_us / 1000.0,
    }


def _detokenization(engine: EnginePredictionConfig) -> dict[str, JSONValue] | None:
    detokenization = getattr(engine, "detokenization", None)
    if detokenization is None or not detokenization.active:
        return None
    return {
        "workers": detokenization.workers,
        "base_ms": detokenization.base_ms,
        "per_output_ms": detokenization.per_output_us / 1000.0,
    }


def _deployment(
    engine: EnginePredictionConfig,
    *,
    workload: dict[str, JSONValue],
) -> BackendDeploymentSpec:
    if has_afd(engine.workers):
        return _roles_with_afd_deployment(engine, workload=workload)
    mode = "agg" if engine.mode == "aggregated" else "disagg"
    # Check every replayed role before any role sizes its state or KV cache.
    for role in ("aggregated",) if mode == "agg" else ("prefill", "decode"):
        worker = getattr(engine.workers, role)
        if worker is not None:
            reject_pipeline_parallelism(role, worker.parallelism.pipeline)
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
        "tokenization": _tokenization(engine),
        "detokenization": _detokenization(engine),
    }
    if mode == "agg":
        assert engine.workers.aggregated is not None
        worker, vl_metadata, vision = _host_worker(engine, engine.workers.aggregated, workload)
        _require_exclusive_context_parallelism(worker)
        parallel = _parallel_mapping(worker, prefix="")
        worker, state_size = _resolve_state_cache(engine, worker)
        return BackendDeploymentSpec(
            parallel_config=parallel,
            performance_model_metadata={"aggregated": {**_state_metadata(engine, worker, state_size), **vl_metadata}},
            agg_engine_args=_worker_engine_args(
                engine, worker, "aggregated", transfer_bytes_per_token=None, vision=vision
            ),
            num_workers=worker.parallelism.replicas,
            **common,
        )
    assert engine.workers.prefill is not None and engine.workers.decode is not None
    prefill, vl_metadata, vision = _host_worker(engine, engine.workers.prefill, workload)
    _require_disaggregated_context_parallelism(engine.backend, prefill, engine.workers.decode)
    prefill, prefill_state = _resolve_state_cache(engine, prefill)
    decode, decode_state = _resolve_state_cache(engine, engine.workers.decode)
    transfer_bytes_per_token = None
    if engine.kv_transfer is not None:
        configured = engine.kv_transfer.bytes_per_token
        if configured == "auto" and prefill_state is not None:
            # The state estimate already resolved this role's token KV bytes.
            configured = prefill.kv_cache.bytes_per_token
        transfer_bytes_per_token = _resolve_kv_bytes_per_token(
            engine,
            prefill,
            configured,
            role="prefill",
        )
    parallel = {
        **_parallel_mapping(prefill, prefix="prefill_"),
        **_parallel_mapping(decode, prefix="decode_"),
    }
    return BackendDeploymentSpec(
        parallel_config=parallel,
        performance_model_metadata={
            "prefill": {**_state_metadata(engine, prefill, prefill_state), **vl_metadata},
            "decode": _state_metadata(engine, decode, decode_state),
        },
        prefill_engine_args=_worker_engine_args(
            engine, prefill, "prefill", transfer_bytes_per_token=transfer_bytes_per_token, vision=vision
        ),
        decode_engine_args=_worker_engine_args(
            engine, decode, "decode", transfer_bytes_per_token=transfer_bytes_per_token
        ),
        num_prefill_workers=prefill.parallelism.replicas,
        num_decode_workers=decode.parallelism.replicas,
        **common,
    )


def _resolve_state_cache(
    engine: EnginePredictionConfig, worker: WorkerPredictionConfig
) -> tuple[WorkerPredictionConfig, dict[str, Any] | None]:
    """Return the worker with resolved state geometry and its sizing report."""
    cache = worker.kv_cache
    if cache.state_cache is None:
        return worker, None
    size = resolve_state_size(engine, worker)
    resolved = cache.model_copy(
        update={
            "state_cache": cache.state_cache.model_copy(update={"bytes_per_request": size["bytes_per_request"]}),
            "block_size": size["block_size"],
            "bytes_per_token": size["kv_bytes_per_token"],
        }
    )
    return worker.model_copy(update={"kv_cache": resolved}), size


def _state_metadata(
    engine: EnginePredictionConfig, worker: WorkerPredictionConfig, state_size: dict[str, Any] | None
) -> dict[str, Any]:
    metadata = _worker_performance_model_metadata(engine, worker)
    if state_size is not None:
        metadata["state_cache"] = state_size
    return metadata


def _afd_role(engine, worker, role):
    afd = worker.afd
    facts = resolve_model_hardware(
        engine.model, engine.hardware, backend=engine.backend, systems_paths=engine.systems_paths
    )
    a_system, f_system = afd.a_system or engine.hardware, afd.f_system or engine.hardware
    pool_facts = [
        facts
        if system == engine.hardware
        else resolve_model_hardware(engine.model, system, backend=engine.backend, systems_paths=engine.systems_paths)
        for system in (a_system, f_system)
    ]
    topology = AFDTopology.from_mapping(
        {
            **afd.model_dump(
                mode="python", exclude={"a_system", "f_system", "chunked_prefill_size", "tp_a", "a_batch_size"}
            ),
            "a_tp_size": afd.tp_a,
            "batch_size_per_a_replica": afd.a_batch_size,
            "gpus_per_node": pool_facts[0].gpus_per_node,
            "f_gpus_per_node": pool_facts[1].gpus_per_node,
            "is_moe": facts.is_moe,
            "num_experts": facts.num_experts,
            "a_system": a_system,
            "f_system": f_system,
        }
    )
    performance_metadata: dict[str, dict[str, JSONValue]] = {
        role: {
            "provider": "afd",
            "measurement_required": False,
            "config": topology.provenance()["topology"],
            "provenance": topology.provenance(),
            # Identity for the native materializer: the runner does not read
            # the frozen measurement provenance.
            "model": engine.model,
            "hardware": engine.hardware,
            # KV window for the native memory gate: the qualification reserves
            # KV for the full context window per request — "max" resolves to
            # the model's max context length.
            "context_length": (
                worker.context_length or engine.context_length
                if isinstance(worker.context_length or engine.context_length, int)
                else resolve_model_context_length(engine.model)
            ),
            # Estimator policy passthrough for the afd provider: the
            # native path honors these verbatim.
            "estimator": {
                key: value
                for key, value in {
                    "database_mode": engine.database_mode,
                    "transfer_policy": engine.transfer_policy,
                    "systems_paths": engine.systems_paths,
                    "estimation_mode": engine.estimation_mode,
                    "fallback_policy": engine.fallback_policy,
                    "estimator_config": engine.estimator_config,
                    "enable_shared_layer": engine.enable_shared_layer,
                    "strict_provenance": engine.strict_provenance,
                }.items()
                if value not in (None, "SILICON", "auto", "deny", {})
            },
            "quant": {
                key: value
                for key, value in {
                    "gemm_quant_mode": engine.gemm_quant_mode,
                    "moe_quant_mode": engine.moe_quant_mode,
                    "fmha_quant_mode": engine.fmha_quant_mode,
                    "kvcache_quant_mode": engine.kvcache_quant_mode,
                    "comm_quant_mode": engine.comm_quant_mode,
                }.items()
                if value is not None
            },
            # MTP draft depth for the native verify-width cost + sampler
            # acceptance; 0 when MTP is off. nextn_accepted feeds
            # the rank's burst sampler (the runner lowers it to conditional
            # accept rates); EnginePredictionConfig validates it is non-None
            # and within [0, nextn] whenever nextn > 0.
            "nextn": engine.nextn or 0,
            **{
                name: getattr(engine, name)
                for name in ("attention_backend", "moe_kernel_source")
                if getattr(engine, name) is not None
            },
            **({"nextn_accepted": engine.nextn_accepted} if engine.nextn else {}),
        }
    }
    raw = {
        "afd": {
            **topology.provenance()["topology"],
            **({"chunked_prefill_size": afd.chunked_prefill_size} if afd.chunked_prefill_size is not None else {}),
        },
        "startup_time": worker.startup_seconds,
        "aic_model_path": engine.model,
    }
    return raw, performance_metadata[role], topology


def _roles_with_afd_deployment(engine, *, workload):
    roles = ("aggregated",) if engine.mode == "aggregated" else ("prefill", "decode")
    for role in roles:
        worker = getattr(engine.workers, role)
        if not is_afd_worker(worker) and (_prefill_cp(worker.parallelism) != 1 or _decode_cp(worker.parallelism) != 1):
            raise ValueError(f"AFD companion ({role}) workers do not support context parallelism; set both to 1")
    validate_afd_model(engine.model)
    # All role hardware uses one literal backend version. Check each unique data root.
    from .sweeper.forward_pass_estimator import resolve_afd_backend_version, resolve_systems_paths

    paths = list(resolve_systems_paths(engine.systems_paths))
    systems = set()
    for role in roles:
        worker = getattr(engine.workers, role)
        if is_afd_worker(worker):
            systems.update(
                (system, tuple(paths))
                for system in (worker.afd.a_system or engine.hardware, worker.afd.f_system or engine.hardware)
            )
        else:
            role_paths = list(resolve_systems_paths(worker.timing.systems_paths or engine.systems_paths))
            systems.add((worker.hardware or engine.hardware, tuple(role_paths)))
    version = resolve_afd_backend_version(engine.backend, engine.backend_version, tuple(sorted(systems)))
    # Literal-version availability is validated by each pool's canonical Rust
    # model constructor. Rechecking it through the global Python catalog scans
    # unrelated platforms and duplicates the same validation at replay startup.
    engine = engine.model_copy(update={"backend_version": version})
    args, metadata, parallel, counts = {}, {}, {}, {}
    transfer_bytes = None
    if engine.kv_transfer is not None:
        worker = engine.workers.prefill
        if is_afd_worker(worker):
            configured = engine.kv_transfer.bytes_per_token
            transfer_bytes = (
                configured
                if configured != "auto"
                else estimate_kv_bytes_per_token(
                    engine.model,
                    tp_size=worker.afd.tp_a,
                    pp_size=1,
                    moe_tp_size=1,
                    moe_ep_size=1,  # KV belongs to one A replica, independently of F-pool EP.
                    kvcache_quant_mode=engine.kvcache_quant_mode,
                )
            )
        else:
            transfer_bytes = _resolve_kv_bytes_per_token(
                engine, worker, engine.kv_transfer.bytes_per_token, role="prefill"
            )
    for role in roles:
        worker = getattr(engine.workers, role)
        if is_afd_worker(worker):
            raw, meta, topology = _afd_role(engine, worker, role)
            metadata[role] = meta
            parallel[role] = {"type": "afd", **topology.provenance()}
            counts[role] = 1
            parallel["replicas" if role == "aggregated" else f"{role}_replicas"] = 1
            if engine.kv_transfer is not None:
                raw.update(
                    kv_transfer_bytes_per_token=transfer_bytes, kv_transfer_timing_mode=engine.kv_transfer.timing_mode
                )
                if engine.kv_transfer.bandwidth_gb_per_second is not None:
                    raw["kv_transfer_bandwidth"] = engine.kv_transfer.bandwidth_gb_per_second
        else:
            if isinstance(worker, HostWorkerPredictionConfig):
                unsupported = [name for name in ("vision", "host_profile") if getattr(worker, name) is not None]
                if unsupported:
                    raise ValueError(f"AFD companion ({role}) does not support {unsupported}")
                try:
                    _check_frontend_measurement(engine, worker, workload)
                except ValueError as exc:
                    raise ValueError(f"AFD companion ({role}) frontend: {exc}") from exc
            _require_exclusive_context_parallelism(worker)
            prefix = "" if role == "aggregated" else f"{role}_"
            parallel.update(_parallel_mapping(worker, prefix=prefix))
            metadata[role] = _worker_performance_model_metadata(engine, worker)
            raw = _worker_engine_args(engine, worker, role, transfer_bytes_per_token=transfer_bytes)
            counts[role] = worker.parallelism.replicas
        args["agg_engine_args" if role == "aggregated" else f"{role}_engine_args"] = raw
    return BackendDeploymentSpec(
        deployment_mode="agg" if engine.mode == "aggregated" else "disagg",
        backend=engine.backend,
        backend_version=version,
        parallel_config=parallel,
        performance_model_metadata=metadata,
        num_workers=counts.get("aggregated", 0),
        num_prefill_workers=counts.get("prefill", 0),
        num_decode_workers=counts.get("decode", 0),
        **args,
    )


def _require_exclusive_context_parallelism(worker: WorkerPredictionConfig) -> None:
    """Aggregated workers model prefill CP and decode CP as mutually exclusive.

    This is a deployment-topology rule, deliberately NOT a ModelConfig rule:
    disaggregated deployments carry the two knobs on different workers, so the
    same ModelConfig fields are valid there without any cross-check. In one
    aggregated engine the frameworks either do not compose the two at all
    (SGLang's prefill CP and DCP paths never reference each other) or only for
    a narrow model class (vLLM: sparse MLA with ``ag_rs``), so the simulator
    prices them one at a time.
    """
    parallel = worker.parallelism
    if _prefill_cp(parallel) > 1 and _decode_cp(parallel) > 1:
        raise ValueError(
            "aggregated workers support at most one of parallelism.prefill_context and "
            f"parallelism.decode_context above 1 (got prefill_context={parallel.prefill_context}, "
            f"decode_context={parallel.decode_context}); use a disaggregated deployment to apply "
            "prefill CP on the prefill worker and decode CP on the decode worker"
        )


def _require_disaggregated_context_parallelism(
    backend: str, prefill: WorkerPredictionConfig, decode: WorkerPredictionConfig
) -> None:
    """Disaggregated roles carry their own knobs, with two layout constraints.

    A prefill engine gains nothing from striping its KV (the KV only passes
    through), so decode CP on the prefill worker is accepted only as the
    PCP+DCP layout, i.e. equal to its prefill CP. vLLM's NIXL connector
    additionally refuses to pair a replicated-PCP prefill (pcp > 1, dcp = 1)
    with a DCP-sharded decode, and requires the two DCP sizes to divide one
    another (``nixl/base_worker.py``). SGLang re-lays the KV out per decode
    DCP rank on the prefill side and has no such pairing rule.
    """
    p, d = prefill.parallelism, decode.parallelism
    p_dcp, d_dcp = _decode_cp(p), _decode_cp(d)
    if p_dcp not in (1, _prefill_cp(p)):
        raise ValueError(
            f"prefill workers accept parallelism.decode_context only as 1 or equal to prefill_context "
            f"(got decode_context={p.decode_context}, prefill_context={p.prefill_context}); a prefill "
            "engine only stripes its KV to match a PCP+DCP layout"
        )
    if backend != "vllm" or d_dcp == 1:
        return
    if _prefill_cp(p) > 1 and p_dcp == 1:
        raise ValueError(
            f"vLLM cannot pair a replicated-PCP prefill worker (prefill_context={p.prefill_context}, "
            f"decode_context=1) with a DCP-sharded decode worker (decode_context={d.decode_context}); "
            "set the prefill worker's decode_context equal to its prefill_context or drop one knob"
        )
    if d_dcp % p_dcp and p_dcp % d_dcp:
        raise ValueError(
            f"vLLM requires the prefill and decode DCP sizes to divide one another (got prefill "
            f"decode_context={p.decode_context}, decode decode_context={d.decode_context})"
        )


def _parallel_mapping(worker: WorkerPredictionConfig, *, prefix: str) -> dict[str, JSONValue]:
    parallel = worker.parallelism
    mapping: dict[str, JSONValue] = {
        f"{prefix}replicas": parallel.replicas,
        f"{prefix}tp": parallel.tensor,
        f"{prefix}pp": parallel.pipeline,
        f"{prefix}attention_dp": parallel.attention_data,
        f"{prefix}moe_tp": parallel.moe_tensor,
        f"{prefix}moe_ep": parallel.moe_expert,
        **({f"{prefix}dcp": parallel.decode_context} if parallel.decode_context is not None else {}),
        **_prefill_cp_knob(parallel, f"{prefix}cp"),
    }
    return mapping


def _prefill_cp_knob(parallel: Any, cp_key: str) -> dict[str, int]:
    """Prefill CP, spelled out only when above one so cp=1 deployments, engine
    args and estimator identities stay byte-identical to pre-CP outputs. Decode
    CP is carried by ``decode_context`` itself: ``None`` (not requested) is
    distinct from an explicit 1 for the FPM cell identity."""
    return {cp_key: _prefill_cp(parallel)} if _prefill_cp(parallel) != 1 else {}


def _prefill_cp(parallel: Any) -> int:
    """The priced prefill-CP size: unset (``None``) means one."""
    return parallel.prefill_context or 1


def _decode_cp(parallel: Any) -> int:
    """The priced decode-CP size: unrecorded (``None``) means one."""
    return parallel.decode_context or 1


def _host_worker(
    engine: EnginePredictionConfig, worker: HostWorkerPredictionConfig, workload: dict[str, JSONValue]
) -> tuple[HostWorkerPredictionConfig, dict[str, JSONValue], VisionPredictionConfig | None]:
    """Resolve the language worker that hosts the scheduler loop: the aggregated or the prefill worker.

    Images without a dedicated encoder pool are encoded on it.
    """
    worker, vl_metadata = _resolve_host_profile(engine, worker, workload)
    _check_frontend_measurement(engine, worker, workload)
    vision = None
    if workload.get("images") is not None and engine.workers.encoder is None:
        vision = worker.vision or VisionPredictionConfig()
    return worker, vl_metadata, vision


def _resolve_host_profile(
    engine: EnginePredictionConfig, worker: HostWorkerPredictionConfig, workload: dict[str, JSONValue]
) -> tuple[HostWorkerPredictionConfig, dict[str, JSONValue]]:
    """Fill the worker's frontend stages from its measured host cost table, if it names one."""
    if worker.host_profile is None:
        return worker, {}
    from .vl.table import resolve_frontend

    images = workload.get("images")
    if not isinstance(images, dict):
        raise ValueError("host_profile requires an image workload")
    frontend, digest = resolve_frontend(
        worker.host_profile, model=engine.model, images=images, tensor=worker.parallelism.tensor
    )
    resolved = worker.model_copy(update={"frontend": frontend, "host_profile": None})
    return resolved, {"vl": {"host_profile_digest": digest, "frontend": worker.host_profile.frontend}}


def _check_frontend_measurement(
    engine: EnginePredictionConfig, worker: HostWorkerPredictionConfig, workload: dict[str, JSONValue]
) -> None:
    """Stages measured on one model, workload and feature transport price only that combination."""
    measured = worker.frontend.measured_for if worker.frontend is not None else None
    if measured is None:
        return
    images = workload.get("images")
    if not isinstance(images, dict):
        raise ValueError(f"frontend stages were measured for {measured.describe()}; this workload has no images")
    needed = FrontendMeasurementConfig.for_workload(engine.model, measured.frontend, images, worker.parallelism.tensor)
    if needed != measured:
        raise ValueError(
            f"frontend stages were measured for {measured.describe()}; this prediction needs {needed.describe()}"
        )


def sglang_host_engine_args(
    *,
    host_loop: bool,
    vision: VisionPredictionConfig | None,
    max_batched_tokens: int,
    frontend: FrontendPredictionConfig | None,
) -> dict[str, JSONValue]:
    """Rank arguments of the SGLang host-aware VL path, shared by predict and the sweeper.

    SGLang reads its batch budget from `sglang.chunked_prefill_size`; the public
    token knob only becomes effective on this path so other predictions keep
    their bytes. The frontend's measurement record stays on the Python side.
    """
    result: dict[str, JSONValue] = {}
    sglang: dict[str, JSONValue] = {}
    if host_loop or vision is not None:
        sglang["chunked_prefill_size"] = max_batched_tokens
    if vision is not None:
        sglang["vlm_cache_bytes"] = vision.cache_mib << 20
        result["vision"] = True
    if host_loop:
        sglang["host_loop"] = True
    if sglang:
        result["sglang"] = sglang
    if frontend is not None:
        result["frontend"] = frontend.model_dump(mode="json", exclude={"measured_for"})
    return result


def _worker_performance_model_metadata(
    engine: EnginePredictionConfig, worker: WorkerPredictionConfig
) -> dict[str, JSONValue]:
    parallel = worker.parallelism
    sharded_moe = parallel.moe_tensor * parallel.moe_expert > 1
    config: dict[str, JSONValue] = {
        "backend": engine.backend,
        "backend_version": engine.backend_version,
        "system": worker.hardware or engine.hardware,
        "model_path": engine.model,
        "tp_size": parallel.tensor,
        "attention_dp_size": parallel.attention_data,
        "moe_tp_size": parallel.moe_tensor if sharded_moe else None,
        "moe_ep_size": parallel.moe_expert if sharded_moe else None,
        "nextn": engine.nextn or None,
        "forward_model": worker.timing.forward_model,
        **{
            name: getattr(engine, name)
            for name in ENGINE_MODEL_CONTROL_FIELDS
            if getattr(engine, name) not in (None, False)
        },
        **({"decoder_replay": True} if engine.decoder_replay else {}),
        **{
            field: getattr(engine, field)
            for field in ("enable_shared_layer", "strict_provenance")
            if getattr(engine, field) is not None
        },
    }
    if parallel.decode_context is not None:
        config["dcp"] = parallel.decode_context
    for field in (
        "gemm_quant_mode",
        "moe_quant_mode",
        "fmha_quant_mode",
        "kvcache_quant_mode",
        "comm_quant_mode",
        "attention_backend",
    ):
        value = getattr(worker.timing, field)
        if value is not None:
            config[field] = value
    config["database_mode"] = worker.timing.database_mode or engine.database_mode
    systems_paths = worker.timing.systems_paths or engine.systems_paths
    if systems_paths is not None:
        config["systems_paths"] = systems_paths
    if engine.speculation is not None:
        config["speculation"] = engine.speculation.cost_config()
    if worker.timing.fpm_parquet_path is not None:
        config["fpm_parquet_path"] = worker.timing.fpm_parquet_path
    config.update(_prefill_cp_knob(parallel, "cp_size"))
    return {
        "provider": "aic",
        "config": config,
    }


def _worker_engine_args(
    engine: EnginePredictionConfig,
    worker: WorkerPredictionConfig,
    role: str,
    *,
    transfer_bytes_per_token: int | None,
    vision: VisionPredictionConfig | None = None,
) -> dict[str, JSONValue]:
    backend = engine.backend
    parallel = worker.parallelism
    cache = worker.kv_cache
    capacity = cache.capacity
    memory_fraction = capacity.memory_fraction
    identity_fields = (
        "gemm_quant_mode",
        "moe_quant_mode",
        "fmha_quant_mode",
        "kvcache_quant_mode",
        "comm_quant_mode",
        "attention_backend",
    )
    if engine.workers.encoder is not None and any(
        getattr(worker.timing, field) is not None for field in identity_fields
    ):
        raise ValueError("explicit timing identity requires the canonical forward-pass provider")
    if parallel.decode_context is not None:
        if parallel.tensor % parallel.decode_context:
            raise ValueError("decode_context must divide tensor parallelism")
        if worker.timing.type != "default" or engine.workers.encoder is not None:
            raise ValueError("DCP requires the canonical forward-pass timing provider")
    if capacity.type == "default" and memory_fraction is None:
        memory_fraction = 0.88 if backend == "sglang" else 0.9
    block_size = resolve_block_size(backend, cache.block_size)
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
    if worker.scheduler.async_scheduling is not None:
        payload["async_scheduling"] = worker.scheduler.async_scheduling
    if engine.speculation is not None:
        payload["speculation"] = engine.speculation.model_dump(mode="json")
    if engine.backend_version is not None:
        payload["aic_backend_version"] = engine.backend_version
    if engine.systems_paths is not None and worker.timing.type != "default":
        from .sweeper.forward_pass_estimator import resolve_systems_paths

        payload["systems_path"] = list(resolve_systems_paths(engine.systems_paths))
    if engine.decoder_replay:
        payload["aic_decoder_replay"] = True
    for field in ("database_mode", "enable_shared_layer", "strict_provenance"):
        value = getattr(engine, field)
        if value is not None:
            payload[f"aic_{field}"] = value
    if parallel.pipeline != 1:
        payload["aic_pp_size"] = parallel.pipeline
    payload.update(_prefill_cp_knob(parallel, "aic_cp_size"))
    if _decode_cp(parallel) != 1:
        payload["aic_dcp_size"] = parallel.decode_context
    if parallel.moe_tensor * parallel.moe_expert > 1:
        payload["aic_moe_tp_size"] = parallel.moe_tensor
        payload["aic_moe_ep_size"] = parallel.moe_expert
    if worker.timing.type == "default" and worker.timing.forward_model != "op_level":
        # Only the non-default forward model is spelled out, so op_level specs stay byte-identical.
        payload["aic_forward_model"] = worker.timing.forward_model
        if worker.timing.fpm_parquet_path is not None:
            payload["aic_fpm_parquet_path"] = worker.timing.fpm_parquet_path
    if backend == "vllm" or isinstance(engine.context_length, int) or worker.context_length is not None:
        effective_context_length = (
            worker.context_length
            if worker.context_length is not None
            else engine.context_length
            if isinstance(engine.context_length, int)
            else engine.fpm_profile.context_length
            if engine.fpm_profile is not None
            else resolve_model_context_length(engine.model)
        )
        payload["max_model_len"] = effective_context_length
    if cache.prefix_match_unit is not None:
        payload["prefix_match_unit"] = cache.prefix_match_unit
    if cache.state_cache is not None:
        payload["state_cache"] = {"bytes_per_request": cache.state_cache.bytes_per_request}
        payload["kv_cache_bytes_per_token"] = cache.bytes_per_token
    if capacity.type == "fixed":
        if capacity.blocks is not None:
            payload["num_gpu_blocks"] = capacity.blocks
        else:
            assert capacity.bytes is not None and isinstance(cache.bytes_per_token, int)
            payload["num_gpu_blocks"] = capacity.bytes // (block_size * cache.bytes_per_token)
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
        if capacity.type == "default" and cache.state_cache is None:
            payload = materialize_aic_num_gpu_blocks(payload)
        for name in (
            "systems_path",
            "aic_backend_version",
            "aic_system",
            "aic_model_path",
            "aic_moe_tp_size",
            "aic_moe_ep_size",
            "aic_cp_size",
            "aic_dcp_size",
        ):
            payload.pop(name, None)
    if worker.timing.type == "default" and engine.workers.encoder is None:
        from aisimulate_core.sdk import ForwardPassPerfModelConfig

        from .sweeper.forward_pass_estimator import resolve_systems_paths

        timing = worker.timing
        sharded_moe = parallel.moe_tensor * parallel.moe_expert > 1
        canonical = ForwardPassPerfModelConfig(
            model=engine.model,
            fpm_profile=engine.fpm_profile.model_dump(mode="json") if engine.fpm_profile is not None else None,
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
            dcp=parallel.decode_context,
            moe_tp_size=parallel.moe_tensor if sharded_moe else None,
            moe_ep_size=parallel.moe_expert if sharded_moe else None,
            **_prefill_cp_knob(parallel, "cp_size"),
            kv_block_size=block_size,
            nextn=engine.nextn,
            **{
                name: getattr(timing, name, None) if getattr(timing, name, None) is not None else getattr(engine, name)
                for name in ENGINE_MODEL_CONTROL_FIELDS
            },
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
            encoder_parallel=vision.encoder_parallel if vision is not None else None,
        )
        timing_config = omit_inactive_moe_controls(canonical.to_dict())
        if timing.fpm_parquet_path is not None:
            interpolation = timing_config["estimator_config"].setdefault("fpm_interpolation", {})
            if interpolation.get("fpm_parquet_path", timing.fpm_parquet_path) != timing.fpm_parquet_path:
                raise ValueError("conflicting fpm_parquet_path and estimator_config.fpm_interpolation.fpm_parquet_path")
            interpolation["fpm_parquet_path"] = timing.fpm_parquet_path
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
            role=role,
        )
    if transfer_bytes_per_token is not None:
        payload["kv_transfer_bytes_per_token"] = transfer_bytes_per_token
    if host_offload is not None:
        payload["native_host_offload"] = host_offload.model_dump(mode="json")
    if cache.g3_offload is not None:
        payload["g3_offload"] = cache.g3_offload.model_dump(mode="json")
    if cache.kvcr is not None:
        payload["kvcr"] = cache.kvcr.model_dump(mode="json")
    if engine.kv_transfer is not None:
        transfer = engine.kv_transfer
        if transfer.bandwidth_gb_per_second is not None:
            payload["kv_transfer_bandwidth"] = transfer.bandwidth_gb_per_second
        payload["kv_transfer_timing_mode"] = transfer.timing_mode
    if backend == "sglang" and isinstance(worker, HostWorkerPredictionConfig):
        payload.update(
            sglang_host_engine_args(
                host_loop=worker.host_loop,
                vision=vision,
                max_batched_tokens=worker.scheduler.max_batched_tokens,
                frontend=worker.frontend,
            )
        )
    return payload


def _resolve_kv_bytes_per_token(
    engine: EnginePredictionConfig,
    worker: WorkerPredictionConfig,
    configured: int | str,
    *,
    role: str,
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
        **(
            {
                "fpm_profile": engine.fpm_profile.model_dump(mode="json"),
                "worker_type": role,
                "system": worker.hardware or engine.hardware,
                "backend": engine.backend,
                "backend_version": engine.backend_version,
                "attention_dp_size": parallel.attention_data,
            }
            if engine.fpm_profile is not None
            else {}
        ),
        kvcache_quant_mode=worker.timing.kvcache_quant_mode or engine.kvcache_quant_mode,
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
            if load.agentic_snapshot is not None:
                workload["agentic_snapshot"] = load.agentic_snapshot.model_dump(mode="json")
            if load.agentic_warmup:
                workload["agentic_warmup"] = True
            if load.agentic_warmup_policy != "repeated_prefixes":
                workload["agentic_warmup_policy"] = load.agentic_warmup_policy
            if load.agentic_profile is not None:
                workload["agentic_profile"] = load.agentic_profile.model_dump(mode="json")
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
