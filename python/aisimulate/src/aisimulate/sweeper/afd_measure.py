# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer A/F measurement for AFD candidates.

This is the measurement core behind
:class:`aisimulate.sweeper.afd_perfmodel.AICAFDPerformanceModel`.
"""

from __future__ import annotations

from typing import Any

from aisimulate.sdk.backends.factory import get_backend
from aisimulate.sdk.config import AFDConfig, RuntimeConfig
from aisimulate.sdk.config_builders import apply_nextn, build_model_config
from aisimulate.sdk.inference_session import AFDInferenceSession
from aisimulate.sdk.models import (
    resolve_context_fmha_by_data,
    resolve_dsv4_moe_arch,
    resolve_nvfp4_for_system,
)
from aisimulate_core.sdk.perf_database import (
    PerfDatabase,
    get_database_view,
    get_latest_database_version,
)

from .afd_parallel import AFDInfeasible, AFDPhase, AFDReasonCategory, AFDTopology
from .afd_perfmodel import AFD_MEASUREMENT_API_VERSION, AFDLayerTimes

__all__ = ["measure_afd_layer_times"]


def _load_database(
    system_name: str,
    backend_name: str,
    backend_version: str | None,
) -> tuple[PerfDatabase, str]:
    version = backend_version
    if version is None:
        version = get_latest_database_version(system=system_name, backend=backend_name)
    if version is None:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            f"no perf database found for system={system_name!r}, backend={backend_name!r}",
            provenance={"system": system_name, "backend": backend_name},
        )
    database = get_database_view(
        system_name,
        backend_name,
        version,
        database_mode="SILICON",
        allow_missing_data=False,
    )
    if database is None:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            f"failed to load the perf database for system={system_name!r}, "
            f"backend={backend_name!r}, version={version!r}",
            provenance={"system": system_name, "backend": backend_name, "version": version},
        )
    return database, version


def _layer_times(
    payload: dict[str, Any],
    *,
    phases: tuple[AFDPhase, ...],
    provenance: dict[str, Any],
) -> tuple[AFDLayerTimes, ...]:
    measurements: list[AFDLayerTimes] = []
    for phase in phases:
        item = payload.get(phase.value)
        if not isinstance(item, dict):
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                f"the AFD run omitted the {phase.value!r} layer measurement",
                provenance={"available_phases": sorted(payload)},
            )
        try:
            measurements.append(
                AFDLayerTimes(
                    phase=phase,
                    attention_ms=float(item["attention_ms"]),
                    ffn_ms=float(item["ffn_ms"]),
                    a_to_f_ms=float(item["a_to_f_ms"]),
                    f_to_a_ms=float(item["f_to_a_ms"]),
                    num_layers=int(item["num_layers"]),
                    provenance=provenance,
                )
            )
        except AFDInfeasible:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise AFDInfeasible(
                AFDReasonCategory.INVALID_MEASUREMENT,
                f"the AFD run returned an invalid {phase.value!r} layer measurement: {exc}",
                provenance={"measurement": dict(item)},
            ) from exc
    return tuple(measurements)


def measure_afd_layer_times(
    *,
    model_path: str,
    system_name: str,
    backend_name: str,
    backend_version: str | None = None,
    isl: int,
    osl: int,
    topology: AFDTopology,
    prefix: int = 0,
    nextn: int = 0,
    max_seq_len: int | None = None,
) -> tuple[AFDLayerTimes, ...]:
    """Measure per-layer A/F times for one AFD topology at one workload point.

    Returns one :class:`AFDLayerTimes` per simulated phase: prefill and decode
    when ``topology.phase`` is ``both``, otherwise just that phase.

    MTP is measured cost-side only: ``apply_nextn`` writes the draft depth
    into both pool configs so the walked op lists carry the widened verify
    width, and the returned layer times stay raw. Acceptance (the benefit
    side) is deliberately not projected here — agg/disagg apply accepted-token
    progress above core (the ``run_agg`` scheduler's
    ``decode_tokens_per_iteration``; the replay engine's speculative sampler),
    and AFD will do the same in its evaluation/replay layer once AFD-MTP
    lands. Until then the config validators (``config/engine.py`` and
    ``sweeper/config.py`` ``_validate_engine_controls``) reject nextn>0 for
    AFD and ``sweeper/replay.py`` rejects explicit acceptance for AFD
    deployments, so ``nextn`` arrives as 0 through every public entry;
    nonzero values remain a direct-API surface for the ongoing MTP work.

    The perf database is always loaded with the SILICON defaults
    (``allow_missing_data=False``, no transfer policy, default systems paths).
    ``aisimulate.config.engine._supported_estimator_policies`` lists AFD as an
    unsupported estimator provider and rejects every custom estimator policy, so
    no other database configuration can reach this function.

    Note: this SILICON-only hardcoding is coupled to AFD's Python-side
    orchestration today — A/F partition, ping-pong pipeline, and comm ops all
    live in Python and the Rust engine only serves per-op latency lookups (see
    ``crates/core/perfmodel/docs/parity-audit-2026-07-14.md``. Once AFD
    estimation is ported into the Rust engine and gains first-class
    ``DatabaseMode`` / power / energy support like the agg and disagg surfaces,
    ``_supported_estimator_policies`` can drop AFD from ``unsupported_provider``
    and this function can start honouring ``database_mode`` / ``transfer_policy``
    / ``systems_paths`` / ``estimator_config`` from the caller instead of
    hardcoding SILICON defaults.
    """
    database, resolved_version = _load_database(system_name, backend_name, backend_version)
    backend = get_backend(backend_name)

    # gpus_per_node drives BW selection in perf_database / AFDTransfer, so the
    # hardware fact in the database must agree with the one this topology was
    # enumerated against. resolve_model_hardware reads the same system_spec key;
    # a mismatch means the candidate was built from a different SKU definition.
    gpus_per_node = int(database.system_spec["node"]["num_gpus_per_node"])
    if gpus_per_node != topology.gpus_per_node:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_TOPOLOGY,
            f"topology gpus_per_node={topology.gpus_per_node} disagrees with the "
            f"{system_name!r} system spec, which reports {gpus_per_node}",
            provenance={
                "system": system_name,
                "topology_gpus_per_node": topology.gpus_per_node,
                "system_spec_gpus_per_node": gpus_per_node,
            },
        )

    # A-Worker: attention-only pool; the MoE dims are irrelevant but must satisfy
    #   tp_size * attention_dp_size == moe_tp_size * moe_ep_size.
    # F-Worker: FFN/MoE pool; moe_tp_size = f_tp_size / f_moe_ep_size so the
    #   product constraint holds with attention_dp_size = 1.
    f_tp_size = topology.n_f_nodes * gpus_per_node
    # AFDTopology.__post_init__ already rejects f_moe_ep_size < 1 via _positive_int,
    # so this guard is unreachable today. Keep it as a fail-loud tripwire in case
    # future callers bypass AFDTopology validation.
    if topology.f_moe_ep_size < 1:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_TOPOLOGY,
            f"f_moe_ep_size must be a positive integer, got {topology.f_moe_ep_size!r}",
            provenance={"field": "f_moe_ep_size", "value": topology.f_moe_ep_size},
        )
    if f_tp_size % topology.f_moe_ep_size:
        raise AFDInfeasible(
            AFDReasonCategory.EXPERT_DIVISIBILITY,
            f"f_moe_ep_size={topology.f_moe_ep_size} must divide f_tp_size={f_tp_size} "
            f"(= n_f_nodes * gpus_per_node = {topology.n_f_nodes} * {gpus_per_node}) "
            "so that f_moe_tp = f_tp / f_moe_ep is an integer",
            provenance={"f_moe_ep_size": topology.f_moe_ep_size, "f_tp_size": f_tp_size},
        )
    f_moe_tp_size = f_tp_size // topology.f_moe_ep_size

    a_model_config = build_model_config(topology.tp_a, 1, 1, topology.tp_a, 1)
    f_model_config = build_model_config(f_tp_size, 1, 1, f_moe_tp_size, topology.f_moe_ep_size)

    # TODO: AFDTransfer still models committed decode-token volume only; recalibrate
    # MTP transfer amplification once the serving semantics are finalized.
    apply_nextn(a_model_config, nextn)
    apply_nextn(f_model_config, nextn)
    # The A-worker runs context attention whenever the phase covers prefill
    # ("prefill" or "both"), so resolve fmha against the perf data then. The
    # F-worker is FFN/MoE only and never touches FMHA.
    resolve_context_fmha_by_data(
        a_model_config,
        model_path,
        database,
        backend_name,
        is_context_role=topology.phase in (AFDPhase.PREFILL, AFDPhase.BOTH),
    )
    resolve_dsv4_moe_arch(a_model_config, model_path, system_name=system_name, backend_name=backend_name)
    resolve_nvfp4_for_system(a_model_config, system_name, model_path, backend_name=backend_name)
    resolve_dsv4_moe_arch(f_model_config, model_path, system_name=system_name, backend_name=backend_name)
    resolve_nvfp4_for_system(f_model_config, system_name, model_path, backend_name=backend_name)

    afd_config = AFDConfig(
        n_a_nodes=topology.n_a_nodes,
        n_f_nodes=topology.n_f_nodes,
        gpus_per_node=gpus_per_node,
        tp_a=topology.tp_a,
        # tp_f is derived inside AFDConfig (Phase 1: F-DP=1).
        f_moe_ep_size=topology.f_moe_ep_size,
        a_batch_size=topology.a_batch_size,
        num_microbatches=topology.num_microbatches,
        pipeline_model=topology.pipeline_model.value,
        # Layer measurements are uncalibrated inputs: the backend-neutral evaluator
        # applies this candidate's factor exactly once, so measuring with it would
        # fold it in twice.
        comm_overhead_factor=1.0,
        phase=topology.phase.value,
        # A measurement covers the AFD pool on its own. Pairing it with a static
        # pool is the caller's business, and phase="both" forbids it outright.
        combined_with_pd=False,
        boundary_on_attn=topology.boundary_on_attn,
    )
    runtime_config = RuntimeConfig(
        isl=isl,
        osl=osl,
        batch_size=afd_config.n_a_workers * topology.a_batch_size,
        prefix=prefix,
    )

    session = AFDInferenceSession(
        model_path=model_path,
        a_model_config=a_model_config,
        f_model_config=f_model_config,
        database=database,
        backend=backend,
        afd_config=afd_config,
    )
    summary = session.run_afd(
        runtime_config,
        phase=topology.phase.value,
        max_seq_len=max_seq_len,
    )

    if summary.check_oom():
        raise AFDInfeasible(
            AFDReasonCategory.OUT_OF_MEMORY,
            f"the model {model_path!r} does not fit in GPU memory on {system_name!r} "
            f"with this AFD topology (phase={topology.phase.value}); widen tp_a or the "
            "F pool (which widens the F-replica under Phase 1 F-DP=1), lower "
            "a_batch_size, or use a system with more VRAM per GPU",
            provenance={
                "model": model_path,
                "system": system_name,
                "phase": topology.phase.value,
                "tp_a": topology.tp_a,
                "n_f_nodes": topology.n_f_nodes,
                "a_batch_size": topology.a_batch_size,
            },
        )

    result_dict = summary.get_result_dict()
    payload = result_dict.get("afd_layer_measurements") if isinstance(result_dict, dict) else None
    if not isinstance(payload, dict) or not payload:
        raise AFDInfeasible(
            AFDReasonCategory.INVALID_MEASUREMENT,
            "the AFD run produced no layer measurements; the configuration may be invalid",
            provenance={"model": model_path, "system": system_name, "phase": topology.phase.value},
        )

    phases = (AFDPhase.PREFILL, AFDPhase.DECODE) if topology.phase is AFDPhase.BOTH else (topology.phase,)
    return _layer_times(
        payload,
        phases=phases,
        provenance={
            "provider": "aic",
            "source": "aisimulate.sweeper.afd_measure.measure_afd_layer_times",
            "api_version": AFD_MEASUREMENT_API_VERSION,
            "units": "milliseconds_per_layer",
            "communication_calibration": "unscaled",
            "model": model_path,
            "hardware": system_name,
            "backend": backend_name,
            "backend_version": resolved_version,
            "input_length": isl,
            "output_length": osl,
        },
    )
