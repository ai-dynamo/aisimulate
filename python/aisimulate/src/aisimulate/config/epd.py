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
    """A callback must retain the resolved EPD identity, traffic and GPU topology."""
    from ..compiler import _parallel_mapping
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
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        raise ValueError(f"EPD prediction-ready output must preserve the encoder and workload: {exc}") from exc
