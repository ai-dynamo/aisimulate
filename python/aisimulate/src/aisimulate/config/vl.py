# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless public encoding of a native VL replay candidate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..sweeper.replay import ReplaySpec

_SEARCH_ONLY_TRAFFIC = ("load_search_field", "load_choices", "load_range", "load_integer", "load_log_scale")


def _execution_traffic(spec: ReplaySpec) -> dict[str, Any]:
    """The traffic the runner executes for `spec`, in one spelling.

    The sweeper spells out defaults and keeps relative forms (`num_request_ratio`,
    `kv_load_ratio`) that the compiler resolves into concrete values, so both are
    normalized through the sweeper's workload model and the runner's rules:
    the resolved concurrency travels on the spec, and the request count of a
    relative stop is the runner's `round(ratio * load)`.
    """
    from ..sweeper.config import Workload

    workload = dict(spec.workload)
    # The compiler spells a constant-rate load as the interval it derives from the rate,
    # which the workload model does not know; validate it as the rate.
    interval = workload.pop("arrival_interval_ms", None)
    if interval is not None and workload.get("request_rate") is None:
        workload["request_rate"] = 1_000.0 / float(interval)
    traffic = Workload.model_validate(workload).model_dump(mode="json")
    for key in _SEARCH_ONLY_TRAFFIC:
        traffic.pop(key, None)
    rate = traffic.pop("request_rate", None)
    if rate is not None:
        # Compared as the interval, computed as the compiler computes it.
        traffic["arrival_interval_ms"] = interval if interval is not None else 1_000.0 / float(rate)
    if spec.concurrency is not None:
        # A KV-capacity or searched load resolved to this concurrency before scoring.
        traffic["concurrency"] = spec.concurrency
        traffic["load_type"] = "concurrency"
    traffic.pop("kv_load_ratio", None)
    ratio = traffic.pop("num_request_ratio", None)
    if traffic.get("request_count") is None and ratio is not None:
        load = traffic.get("concurrency") or traffic.get("request_rate")
        if load is not None:
            traffic["request_count"] = max(1, round(ratio * load))
    return traffic


def _differences(mine: dict[str, Any], scored: dict[str, Any]) -> str:
    keys = sorted(key for key in set(mine) | set(scored) if mine.get(key) != scored.get(key))
    return ", ".join(f"{key}={mine.get(key)!r} vs {scored.get(key)!r}" for key in keys)


def validate_vl_prediction_mapping(value: dict, spec: ReplaySpec) -> None:
    """A callback must reproduce the scored traffic, SLA and the language replay that ran it."""
    from ..compiler import _parallel_mapping, prediction_to_replay_spec
    from ..runner import _materialize_sla
    from .cli import CorePredictionConfig
    from .epd import _language_execution

    try:
        prediction = CorePredictionConfig.model_validate(value)
        engine = prediction.engine
        mode = "agg" if engine.mode == "aggregated" else "disagg"
        host_role = "aggregated" if mode == "agg" else "prefill"
        if engine.workers.encoder is not None or getattr(engine.workers, host_role) is None:
            raise ValueError(f"{host_role} worker was dropped")
        compiled = prediction_to_replay_spec(prediction)
        # Compare what the runner executes, in both directions: a stop condition or
        # seed that the saved prediction drops is as much a change as one it adds.
        # The search target itself is not part of a prediction; its SLA is checked below.
        mine, scored = _execution_traffic(compiled), _execution_traffic(spec)
        if mine != scored:
            raise ValueError(f"traffic changed: {_differences(mine, scored)}")
        deployment = spec.backend_deployment
        if deployment.deployment_mode != mode:
            raise ValueError("language layout changed")
        roles = (("aggregated", ""),) if mode == "agg" else (("prefill", "prefill_"), ("decode", "decode_"))
        for role, prefix in roles:
            parallel = _parallel_mapping(getattr(engine.workers, role), prefix=prefix)
            if any(deployment.parallel_config.get(key) != val for key, val in parallel.items()):
                raise ValueError("language GPU topology changed")
        # The engine descriptors carry the host, frontend and vision tables.
        if _language_execution(compiled) != _language_execution(spec):
            raise ValueError("language replay settings changed")
        if _materialize_sla(compiled) != _materialize_sla(spec):
            raise ValueError("evaluation SLA changed")
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        raise ValueError(
            f"native VL prediction-ready output must preserve the workload and host tables: {exc}"
        ) from exc
