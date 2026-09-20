# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless public encoding of a native VL replay candidate."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sweeper.replay import ReplaySpec


def validate_vl_prediction_mapping(value: dict, spec: ReplaySpec) -> None:
    """A callback must preserve the scored workload, host tables and language replay."""
    from ..compiler import _parallel_mapping, prediction_to_replay_spec
    from ..runner import _materialize_sla
    from .cli import CorePredictionConfig
    from .engine import native_vl_worker
    from .epd import _language_execution
    from .traffic import SyntheticSource

    try:
        prediction = CorePredictionConfig.model_validate(value)
        engine = prediction.engine
        worker = native_vl_worker(engine)
        if engine.workers.encoder is not None or worker is None:
            raise ValueError("host-aware aggregated worker was dropped")
        source = prediction.traffic.source
        if not isinstance(source, SyntheticSource):
            raise ValueError("synthetic workload was dropped")
        images = source.images.model_dump(mode="json") if source.images is not None else None
        if images != spec.workload.get("images"):
            raise ValueError("image profile changed")
        if (source.input_tokens, source.output_tokens) != (spec.workload["isl"], spec.workload["osl"]):
            raise ValueError("text lengths changed")
        load = prediction.traffic.load
        if load.type == "concurrency":
            if load.concurrency != (spec.concurrency or spec.workload.get("concurrency")):
                raise ValueError("fixed concurrency changed")
        elif (load.requests_per_second or load.sessions_per_second) != spec.workload.get("request_rate"):
            raise ValueError("request rate changed")
        deployment = spec.backend_deployment
        if deployment.deployment_mode != "agg":
            raise ValueError("language layout changed")
        parallel = _parallel_mapping(engine.workers.aggregated, prefix="")
        if any(deployment.parallel_config.get(key) != val for key, val in parallel.items()):
            raise ValueError("language GPU topology changed")
        compiled = prediction_to_replay_spec(prediction)
        # The engine descriptors carry the host, frontend and vision tables; compare
        # what the runner executes rather than a second list of fields.
        if _language_execution(compiled) != _language_execution(spec):
            raise ValueError("language replay settings changed")
        if _materialize_sla(compiled) != _materialize_sla(spec):
            raise ValueError("evaluation SLA changed")
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        raise ValueError(
            f"native VL prediction-ready output must preserve the workload and host tables: {exc}"
        ) from exc
