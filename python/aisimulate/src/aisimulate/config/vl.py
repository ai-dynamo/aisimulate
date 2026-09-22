# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless public encoding of a native VL replay candidate."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..sweeper.replay import ReplaySpec


def validate_vl_prediction_mapping(value: dict, spec: ReplaySpec) -> None:
    """A callback must reproduce the scored workload and the language replay that ran it."""
    from ..compiler import _parallel_mapping, prediction_to_replay_spec
    from ..runner import _materialize_sla
    from .cli import CorePredictionConfig
    from .epd import _language_execution

    try:
        prediction = CorePredictionConfig.model_validate(value)
        if prediction.engine.workers.encoder is not None or prediction.engine.workers.aggregated is None:
            raise ValueError("aggregated worker was dropped")
        compiled = prediction_to_replay_spec(prediction)
        # The compiled workload, goal and engine descriptors carry every image,
        # load, stop, host, frontend and vision setting the runner executes;
        # compare those rather than a second list of fields. The sweeper spells
        # out defaults the compiler leaves to the runner, so every setting the
        # prediction produces must match the scored value, not the reverse.
        for name, mine, scored in (
            ("workload", compiled.workload, spec.workload),
            ("goal", compiled.goal, spec.goal),
        ):
            changed = {key for key, value in mine.items() if value is not None and scored.get(key) != value}
            if changed:
                raise ValueError(
                    f"{name} changed: "
                    + ", ".join(f"{key}={mine[key]!r} vs {scored.get(key)!r}" for key in sorted(changed))
                )
        if compiled.concurrency != spec.concurrency:
            raise ValueError(f"concurrency changed: {compiled.concurrency!r} vs {spec.concurrency!r}")
        deployment = spec.backend_deployment
        if deployment.deployment_mode != "agg":
            raise ValueError("language layout changed")
        parallel = _parallel_mapping(prediction.engine.workers.aggregated, prefix="")
        if any(deployment.parallel_config.get(key) != val for key, val in parallel.items()):
            raise ValueError("language GPU topology changed")
        if _language_execution(compiled) != _language_execution(spec):
            raise ValueError("language replay settings changed")
        if _materialize_sla(compiled) != _materialize_sla(spec):
            raise ValueError("evaluation SLA changed")
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        raise ValueError(
            f"native VL prediction-ready output must preserve the workload and host tables: {exc}"
        ) from exc
