# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD layer measurements whose MoE stage comes from a FastAFD split-stage profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aisimulate_core.sdk.fastafd_profile import FastAFDMoEStageMeasurement, FastAFDMoEStageProfile

from .afd_parallel import AFDInfeasible, AFDPhase, AFDReasonCategory, AFDTopology
from .afd_perfmodel import AFD_MEASUREMENT_API_VERSION, AFDLayerTimes, AFDMeasurementRequest, AFDPerformanceModel

# Per-layer attention components an attention provider must expose for the
# recomposition to be exact: FastAFD runs the router on the A side outside its
# measured stage and folds the top-k combine into that stage.
_ATTENTION_COMPONENTS = ("attention_compute_ms", "a_combine_ms", "router_ms")


def _infeasible(detail: str, **provenance: Any) -> AFDInfeasible:
    return AFDInfeasible(AFDReasonCategory.INVALID_MEASUREMENT, detail, provenance=provenance)


@dataclass(frozen=True)
class FastAFDMoEStagePerformanceModel:
    """Combine a measured FastAFD split MoE stage with an attention-side provider.

    The profile entry is an exact measurement of the complete decode MoE stage for
    one ``<A>A<F>F`` topology, per-A-rank batch, MTP degree, microbatch count, and
    layer count. Its latency is spread evenly over ``moe_layers * microbatches``
    pipeline units and becomes ``ffn_ms``; both transfer stages are zero because
    dispatch and return are inside the measured stage. ``attention_ms`` is the
    attention provider's A-side compute without its own combine term, plus the
    router the provider attributes to the F side. Lookup is exact-only.
    """

    profile_path: str | Path
    moe_backend: str
    attention_model: AFDPerformanceModel
    moe_precision: str | None = None

    def measure(self, request: AFDMeasurementRequest) -> tuple[AFDLayerTimes, ...]:
        topology = request.topology
        self._validate_topology(topology)
        attention = self._attention(request)
        components = self._components(attention)
        profile = FastAFDMoEStageProfile.load(self.profile_path)
        entry = self._select(profile, request, attention.num_layers)
        units = entry.key.moe_layers * entry.key.microbatches
        return (
            AFDLayerTimes(
                phase=AFDPhase.DECODE,
                attention_ms=components["attention_compute_ms"] + components["router_ms"],
                ffn_ms=entry.latency_ms / units,
                a_to_f_ms=0.0,
                f_to_a_ms=0.0,
                num_layers=entry.key.moe_layers,
                provenance={
                    "provider": "fastafd",
                    "api_version": AFD_MEASUREMENT_API_VERSION,
                    "units": "milliseconds_per_layer_microbatch",
                    "composition": {
                        "attention_ms": "attention_compute_ms + router_ms",
                        "ffn_ms": "stage_latency_ms / (moe_layers * microbatches)",
                        "a_to_f_ms": "inside the measured stage",
                        "f_to_a_ms": "inside the measured stage",
                    },
                    "stage": {
                        "profile_path": str(profile.path),
                        "profile_sha256": profile.profile_sha256,
                        "key": entry.key.__dict__,
                        "latency_ms": entry.latency_ms,
                        **entry.provenance(),
                    },
                    "attention": {
                        "components": dict(components),
                        **dict(attention.provenance),
                    },
                },
            ),
        )

    def _validate_topology(self, topology: AFDTopology) -> None:
        if topology.phase is not AFDPhase.DECODE:
            raise _infeasible(
                "FastAFD split MoE stage profiles measure decode only; "
                f"topology phase={topology.phase.value!r} needs another provider",
                phase=topology.phase.value,
            )
        if topology.tp_a != 1:
            raise _infeasible(
                f"FastAFD split MoE stage profiles measure one attention rank per A GPU; tp_a={topology.tp_a}",
                tp_a=topology.tp_a,
            )
        if topology.f_moe_ep_size != topology.ffn_gpus:
            raise _infeasible(
                "FastAFD split MoE stage profiles measure experts parallel across every F GPU; "
                f"f_moe_ep_size={topology.f_moe_ep_size} but the F pool has {topology.ffn_gpus} GPUs",
                f_moe_ep_size=topology.f_moe_ep_size,
                ffn_gpus=topology.ffn_gpus,
            )
        if not topology.boundary_on_attn:
            raise _infeasible(
                "FastAFD split MoE stage profiles keep the layer boundary on the A side; set boundary_on_attn=true"
            )

    def _attention(self, request: AFDMeasurementRequest) -> AFDLayerTimes:
        measurements = tuple(self.attention_model.measure(request))
        decode = [item for item in measurements if isinstance(item, AFDLayerTimes) and item.phase is AFDPhase.DECODE]
        if len(decode) != 1:
            raise _infeasible("the attention provider must return exactly one decode AFDLayerTimes")
        return decode[0]

    @staticmethod
    def _components(attention: AFDLayerTimes) -> dict[str, float]:
        raw = attention.provenance.get("components")
        if not isinstance(raw, Mapping) or any(name not in raw for name in _ATTENTION_COMPONENTS):
            raise _infeasible(
                "the attention provider does not expose the per-layer attention_compute_ms, "
                "a_combine_ms, and router_ms components needed to recompose the FastAFD boundary",
                provider=attention.provenance.get("provider"),
            )
        components = {name: float(raw[name]) for name in _ATTENTION_COMPONENTS}
        if components["router_ms"] <= 0.0:
            raise _infeasible(
                "the attention provider attributed no router time to the F side; "
                "FastAFD excludes the router from its measured stage, so it cannot be recomposed"
            )
        return components

    def _select(
        self,
        profile: FastAFDMoEStageProfile,
        request: AFDMeasurementRequest,
        model_layers: int,
    ) -> FastAFDMoEStageMeasurement:
        topology = request.topology
        wanted = {
            "model_path": request.model_name,
            "system": request.hardware_sku,
            "stage": "afd",
            "topology": f"{topology.attention_gpus}A{topology.ffn_gpus}F",
            "logical_batch_per_source_rank": topology.a_batch_size,
            "mtp_nextn": request.nextn,
            "microbatches": topology.num_microbatches,
            "moe_backend": self.moe_backend,
        }
        if self.moe_precision is not None:
            wanted["moe_precision"] = self.moe_precision
        matches = [
            entry
            for entry in profile.entries
            if all(getattr(entry.key, field) == value for field, value in wanted.items())
        ]
        if not matches:
            raise _infeasible(f"no FastAFD split MoE stage measurement matches {wanted}", **wanted)
        if len(matches) != 1:
            variants = sorted(entry.key.moe_precision for entry in matches)
            raise _infeasible(
                "several FastAFD split MoE stage measurements match; set engine.moe_quant_mode to pick the "
                f"profile precision: {variants}",
                **wanted,
            )
        entry = matches[0]
        if entry.key.moe_layers != model_layers:
            raise _infeasible(
                f"the profile measured {entry.key.moe_layers} MoE layers but the attention provider models "
                f"{model_layers} layers; FastAFD profiles cover models whose layers are all MoE layers",
                moe_layers=entry.key.moe_layers,
                model_layers=model_layers,
            )
        return entry


__all__ = ["FastAFDMoEStagePerformanceModel"]
