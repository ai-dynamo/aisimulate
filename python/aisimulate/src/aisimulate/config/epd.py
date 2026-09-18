# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless public encoding of a resolved analytical encoder pool."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sweeper.replay import EncoderPoolSpec, ReplaySpec


def encoder_prediction_fields(encoder: EncoderPoolSpec) -> dict:
    """Pin inputs, including resolved data version; never accept user-supplied timing estimates."""
    return {
        "hardware": encoder.system,
        "backend_version": encoder.backend_version,
        "tensor": encoder.tp,
        "replicas": encoder.workers,
        "batch_size": encoder.batch_size,
        "latency_correction": encoder.latency_correction,
        "rate_degradation": encoder.rate_degradation,
    }


def validate_epd_prediction_mapping(value: dict, spec: ReplaySpec) -> None:
    """A callback must preserve the scored encoder, workload and language replay."""
    from ..compiler import _parallel_mapping, prediction_to_replay_spec
    from ..runner import _materialize_sla
    from .cli import CorePredictionConfig
    from .traffic import SyntheticSource

    encoder = spec.backend_deployment.encoder
    assert encoder is not None
    try:
        prediction = CorePredictionConfig.model_validate(value)
        engine = prediction.engine
        if engine.workers.encoder is None or engine.workers.encoder.model_dump() != encoder_prediction_fields(encoder):
            raise ValueError("encoder parameters or resolved database version changed")
        if engine.model != encoder.model or engine.backend != encoder.backend:
            raise ValueError("encoder model/backend identity changed")
        if spec.backend_deployment.backend_version and engine.backend_version is None:
            raise ValueError("selected language backend version was dropped")
        source = prediction.traffic.source
        if not isinstance(source, SyntheticSource) or source.images is None:
            raise ValueError("fixed image workload was dropped")
        if source.images.model_dump() != spec.workload["images"]:
            raise ValueError("image profile changed")
        if (source.input_tokens, source.output_tokens) != (spec.workload["isl"], spec.workload["osl"]):
            raise ValueError("text lengths changed")
        if prediction.traffic.load.concurrency != (spec.concurrency or spec.workload.get("concurrency")):
            raise ValueError("fixed concurrency changed")
        stop = prediction.traffic.stop
        assert stop is not None
        expected_count = spec.workload.get("request_count")
        if expected_count is None:
            expected_count = max(1, round(spec.workload["num_request_ratio"] * prediction.traffic.load.concurrency))
        count = stop.requests
        if count is None:
            count = max(1, round(stop.requests_per_load_unit * prediction.traffic.load.concurrency))
        if count != expected_count:
            raise ValueError("request count changed")
        deployment = spec.backend_deployment
        mode = "agg" if engine.mode == "aggregated" else "disagg"
        if mode != deployment.deployment_mode:
            raise ValueError("language layout changed")
        roles = (("aggregated", ""),) if mode == "agg" else (("prefill", "prefill_"), ("decode", "decode_"))
        for role, prefix in roles:
            parallel = _parallel_mapping(getattr(engine.workers, role), prefix=prefix)
            if any(deployment.parallel_config.get(key) != val for key, val in parallel.items()):
                raise ValueError("language GPU topology changed")
        # Compile through the public reload path, including visual-context admission.
        # Compare the same engine descriptors the runner executes, not a second list
        # of scheduler/cache/timing fields that can drift from the actual consumer.
        compiled = prediction_to_replay_spec(prediction)
        if _language_execution(compiled) != _language_execution(spec):
            raise ValueError("language replay settings changed")
        if _materialize_sla(compiled) != _materialize_sla(spec):
            raise ValueError("evaluation SLA changed")
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        raise ValueError(f"EPD prediction-ready output must preserve the encoder and workload: {exc}") from exc


def _language_execution(spec: ReplaySpec) -> dict:
    """Normalize compiler/Sweeper spellings at the existing execution boundary."""
    from aisimulate_core.sdk.perf_database import get_latest_database_version, resolve_query_version

    from ..runner import _materialize_engine_role
    from ..sweeper.forward_pass_estimator import resolve_systems_paths
    from .engine import SchedulerPredictionConfig

    deployment = spec.backend_deployment
    roles = (
        (("aggregated", deployment.agg_engine_args),)
        if deployment.deployment_mode == "agg"
        else (("prefill", deployment.prefill_engine_args), ("decode", deployment.decode_engine_args))
    )
    result = {}
    for role, args in roles:
        assert args is not None
        engine = _materialize_engine_role(
            deployment.backend, deployment.backend_version, deployment.parallel_config, args, role
        )
        rank = engine["rank"]
        rank.setdefault("prefill_schedule_interval", SchedulerPredictionConfig().prefill_schedule_interval)
        rank.setdefault("prefill_decode_interval", SchedulerPredictionConfig().prefill_decode_interval)
        timing = rank["timing_model"]["config"]
        roots = timing.get("systems_paths")
        if not roots and timing.get("systems_path") is not None:
            roots = [timing["systems_path"]]
        resolved_roots = list(resolve_systems_paths(roots)) if roots is not None else None
        version = timing.get("backend_version")
        if not version:
            version = get_latest_database_version(
                timing["system"],
                timing["backend"],
                **({"systems_paths": resolved_roots} if resolved_roots is not None else {}),
            )
            if version is None:
                raise ValueError(f"no perf database for system={timing['system']!r}, backend={timing['backend']!r}")
        timing["backend_version"] = resolve_query_version(
            timing["system"],
            timing["backend"],
            version,
            systems_paths=resolved_roots,
        )
        timing.setdefault("cuda_graph_reserved_bytes", 0)
        # HandoffTransferTiming::delay_ms uses the same fallback for either mode
        # when a complete byte-count/bandwidth transfer model is unavailable.
        if rank.get("kv_transfer_bytes_per_token") is None or rank.get("kv_transfer_bandwidth") is None:
            rank.pop("kv_transfer_timing_mode", None)
        result[role] = engine
    return result
