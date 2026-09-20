# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model- and hardware-aware starting points for one onboarding worker.

These byte-budget prechecks do not predict performance or establish runtime
compatibility. The generated plan and serving runtime retain admission authority.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import yaml

from .config_profile import (
    ModelConfig,
    ProfileDraft,
    ProfileRequestError,
    derive_profile,
    packaged_hardware_path,
    validate_overrides,
)
from .schema import SupportRequest

_NON_KV_BYTES = ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes")
_RANK_BYTES = frozenset((*_NON_KV_BYTES, "kv_bytes_per_token"))
_TOPOLOGY_FIELDS = ("tensor_parallel", "attention_data_parallel", "moe_tensor_parallel", "moe_expert_parallel")
_FAMILIES = ("pure_tp", "dep", "tep")


@dataclass(frozen=True)
class HardwareEnvelope:
    system: str
    declared_interconnect: str
    source: str
    sha256: str
    per_gpu_bytes: int
    memory_budget_bytes: int
    gpus_per_node: int
    gpus_per_rack: int | None
    fast_domain: Literal["single_gpu", "node", "rack"]
    fast_domain_gpus: int
    assumption: str


@dataclass(frozen=True)
class TopologyCandidate:
    family: Literal["tp", "pure_tp", "dep", "tep"]
    sizes: tuple[int, int, int, int]
    status: Literal["estimated_fit", "needs_inputs", "rejected"]
    reasons: tuple[str, ...]
    draft: ProfileDraft | None
    known_required_bytes: int | None
    estimated_required_bytes: int | None
    memory_budget_bytes: int
    resident_sequences_per_rank: int | None

    @property
    def topology(self) -> dict[str, int]:
        return dict(zip(_TOPOLOGY_FIELDS, self.sizes, strict=True))

    @property
    def required_gpus(self) -> int:
        return self.sizes[0] * self.sizes[1]

    @property
    def cli_flags(self) -> tuple[str, ...]:
        return tuple(
            item for key, value in self.topology.items() for item in ("--" + key.replace("_", "-"), str(value))
        )

    @property
    def missing(self) -> dict[str, str]:
        return dict(self.draft.missing) if self.draft is not None else {}

    def apply(self, request: SupportRequest) -> SupportRequest:
        """Return the selected topology with the caller's other request inputs intact."""
        payload = request.model_dump()
        payload["search"].update(self.topology)
        return SupportRequest.model_validate(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "topology": self.topology,
            "required_gpus": self.required_gpus,
            "cli_flags": list(self.cli_flags),
            "status": self.status,
            "reasons": list(self.reasons),
            "missing": self.missing,
            "known_required_bytes": self.known_required_bytes,
            "estimated_required_bytes": self.estimated_required_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "resident_sequences_per_rank": self.resident_sequences_per_rank,
            "resolved_profile_fields": dict(self.draft.resolved) if self.draft is not None else {},
            "field_sources": dict(self.draft.sources) if self.draft is not None else {},
        }


@dataclass(frozen=True)
class TopologySuggestions:
    config_sha256: str
    identity: dict[str, Any]
    collection: dict[str, Any]
    context_length: int
    overrides: dict[str, Any]
    hardware: HardwareEnvelope
    candidates: tuple[TopologyCandidate, ...]
    rejected_candidates: tuple[TopologyCandidate, ...]
    default: TopologyCandidate | None
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "identity": dict(self.identity),
            "collection": dict(self.collection),
            "context_length": self.context_length,
            "overrides": dict(self.overrides),
            "hardware": asdict(self.hardware),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "rejected_candidates": [candidate.to_dict() for candidate in self.rejected_candidates],
            "default": self.default.to_dict() if self.default is not None else None,
            "assumptions": list(self.assumptions),
        }


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if type(value) is not int or not minimum <= value <= 2**53:
        raise ValueError(f"packaged hardware {field} must be an integer in {minimum}..2**53")
    return value


def _bandwidth(value: Any, field: str) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"packaged hardware {field} must be a positive finite bandwidth in bytes/second")
    return value


def _hardware_envelope(system: str, interconnect: str, memory_fraction: float) -> HardwareEnvelope:
    path = packaged_hardware_path(system)
    if not path.is_file():
        raise ValueError(f"no packaged hardware metadata for {system}; select an explicit topology")
    payload = path.read_bytes()
    spec = yaml.safe_load(payload)
    if not isinstance(spec, dict):
        raise ValueError(f"packaged hardware metadata for {system} must be an object")
    for section in ("gpu", "node"):
        if not isinstance(spec.get(section), dict):
            raise ValueError(f"packaged hardware {section} must be an object")
    capacity = _integer(spec["gpu"].get("mem_capacity"), "gpu.mem_capacity")
    node = spec["node"]
    node_width = _integer(node.get("num_gpus_per_node"), "node.num_gpus_per_node")
    intra_bw = _bandwidth(node.get("intra_node_bw"), "node.intra_node_bw")
    inter_bw = _bandwidth(node.get("inter_node_bw"), "node.inter_node_bw")
    rack_width = None
    if "num_gpus_per_rack" in node:
        rack_width = _integer(node["num_gpus_per_rack"], "node.num_gpus_per_rack")
        if rack_width < node_width or rack_width % node_width:
            raise ValueError("packaged hardware node.num_gpus_per_rack must be a multiple of num_gpus_per_node")
    # Missing reservation estimates remain explicit profile inputs. Malformed
    # entries, including ones outside the candidate widths, are not absence.
    misc = spec.get("misc", {})
    if not isinstance(misc, dict):
        raise ValueError("packaged hardware misc must be an object")
    if "other_mem" in misc:
        _integer(misc["other_mem"], "misc.other_mem", minimum=0)
    nccl = misc.get("nccl_mem", {})
    if not isinstance(nccl, dict):
        raise ValueError("packaged hardware misc.nccl_mem must be a TP-indexed object")
    for tp, value in nccl.items():
        _integer(tp, "misc.nccl_mem TP key")
        _integer(value, f"misc.nccl_mem[{tp}]", minimum=0)
    fabric = interconnect.lower().replace(" ", "").replace("-", "")
    nvlink = fabric in {"nvlink", "nvswitch", "nvlink/nvswitch", "nvlink+nvswitch", "nvlinknvswitch"}
    use_rack = rack_width is not None and inter_bw >= intra_bw and nvlink
    domain, domain_width = "node", node_width
    assumption = (
        "Use the packaged node width as the fast domain; this is hardware architecture, not available GPU capacity."
    )
    if fabric == "none":
        domain, domain_width = "single_gpu", 1
        assumption = "The declared interconnect is 'none'; automatic choices are limited to one GPU."
    elif use_rack:
        domain, domain_width = "rack", rack_width
        assumption = (
            "Declared rack width is a fast domain because inter_node_bw >= intra_node_bw "
            "and the request declares NVLink/NVSwitch; assumes the packaged within-rack fabric. "
            "This is hardware architecture, not available GPU capacity."
        )
    elif not nvlink:
        assumption += (
            f" Declared fabric {interconnect!r} does not establish the packaged NVLink/NVSwitch rack; "
            "fabric compatibility is unverified. Explicit wider topology choices remain available."
        )
    return HardwareEnvelope(
        system=system,
        declared_interconnect=interconnect,
        source=f"packaged systems/{system}.yaml",
        sha256=hashlib.sha256(payload).hexdigest(),
        per_gpu_bytes=capacity,
        memory_budget_bytes=int(capacity * memory_fraction),
        gpus_per_node=node_width,
        gpus_per_rack=rack_width,
        fast_domain=domain,
        fast_domain_gpus=domain_width,
        assumption=assumption,
    )


def _assess(
    config: ModelConfig,
    request: SupportRequest,
    overrides: dict[str, Any],
    hardware: HardwareEnvelope,
    family: str,
    width: int,
) -> TopologyCandidate:
    sizes = {
        "tp": (width, 1, 1, 1),
        "pure_tp": (width, 1, width, 1),
        "dep": (1, width, 1, width),
        "tep": (width, 1, 1, width),
    }[family]
    payload = request.model_dump()
    payload["search"].update(zip(_TOPOLOGY_FIELDS, sizes, strict=True))
    selected = SupportRequest.model_validate(payload)
    try:
        draft = derive_profile(config, selected, overrides)
    except ProfileRequestError as error:
        if error.field not in _TOPOLOGY_FIELDS:
            raise
        return TopologyCandidate(
            family, sizes, "rejected", (str(error),), None, None, None, hardware.memory_budget_bytes, None
        )
    resolved = draft.resolved
    # Admission establishes room for one declared full-context request. The
    # scheduler's sequence limit is not an allocation of full contexts; runtime
    # total KV capacity is the remaining memory divided by bytes per token.
    resident = 1
    known = sum(resolved[field] for field in _NON_KV_BYTES if field in resolved)
    grouped = resolved.get("cache_layout") == "grouped"
    no_grouped_cache_budget = grouped and known >= hardware.memory_budget_bytes
    if grouped and draft.profile is not None and known < hardware.memory_budget_bytes:
        try:
            from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

            estimate = RustForwardPassPerfModel.estimate_cache_budget(
                {
                    "model": selected.identity.model,
                    "system": selected.identity.gpu,
                    "backend": selected.identity.framework,
                    "backend_version": selected.identity.framework_version,
                    "worker_type": "aggregated",
                    "tp": sizes[0],
                    "pp": 1,
                    "attention_dp": sizes[1],
                    "moe_tp_size": sizes[2],
                    "moe_ep_size": sizes[3],
                    "fpm_profile": draft.profile.model_dump(mode="json"),
                },
                {
                    "total_gpu_capacity_bytes": hardware.per_gpu_bytes,
                    "memory_fraction_kind": "of_total",
                    "memory_fraction_value": request.collection.memory_fraction,
                    "max_num_tokens": resolved["max_num_tokens"],
                    "max_batch_size": resolved["max_batch_size"],
                    "context_length": request.search.context_length,
                },
            )
        except (ImportError, AttributeError) as exc:
            raise ValueError(
                "grouped cache planning requires this checkout's compiled AISimulate extension; install or rebuild it"
            ) from exc
        known += estimate["request_peak_cache_bytes"]
    elif "kv_bytes_per_token" in resolved:
        known += resolved["kv_bytes_per_token"] * (request.search.context_length + 1)
    complete_resources = draft.profile is not None if grouped else resolved.keys() >= _RANK_BYTES
    estimated = known if complete_resources else None
    if known > hardware.memory_budget_bytes or no_grouped_cache_budget:
        status = "rejected"
        bound = "Conservative grouped peak" if grouped and complete_resources else "Known per-rank resource lower bound"
        reasons = (
            f"Non-KV resources {known} bytes leave no grouped cache budget ({hardware.memory_budget_bytes}-byte limit)."
            if no_grouped_cache_budget
            else f"{bound} {known} bytes exceeds the {request.collection.memory_fraction * 100:g}% per-GPU budget "
            f"of {hardware.memory_budget_bytes} bytes.",
        )
    elif draft.missing or not complete_resources:
        status = "needs_inputs"
        reasons = (
            "Known geometry checks passed; resource/precision inputs are incomplete. "
            "A known byte lower bound below the budget does not establish fit.",
        )
    else:
        status = "estimated_fit"
        reasons = (
            f"Complete declared/estimated per-rank resources including one full-context cache need {known} bytes, "
            f"within the {request.collection.memory_fraction * 100:g}% per-GPU budget "
            f"of {hardware.memory_budget_bytes} bytes; "
            "this is a conservative admission precheck, not runtime qualification.",
        )
    return TopologyCandidate(
        family, sizes, status, reasons, draft, known, estimated, hardware.memory_budget_bytes, resident
    )


def suggest_topologies(
    config: ModelConfig, request: SupportRequest, overrides: Mapping[str, Any] | None = None
) -> TopologySuggestions:
    """Shortlist one worker's starting points, without user allocation or timing data.

    The request supplies actual deployment identity, collection bounds and context. Its
    topology is replaced during enumeration. Exact per-rank bounds and attached
    deployment profiles must instead use the existing explicit-topology route.
    """
    supplied = validate_overrides(overrides)
    rank_overrides = sorted((_RANK_BYTES | {"cache_groups"}) & supplied.keys())
    if rank_overrides:
        raise ValueError(
            "automatic topology suggestions cannot transfer rank-local byte overrides across topologies: "
            + ", ".join(rank_overrides)
            + "; select an exact topology and use explicit onboarding with these bounds"
        )
    if request.fpm_profile is not None:
        raise ValueError("an attached FPM profile belongs to an exact topology; use explicit onboarding")
    hardware = _hardware_envelope(
        request.identity.gpu, request.identity.interconnect, request.collection.memory_fraction
    )
    candidates = []
    rejected = []
    families = _FAMILIES if request.identity.model_kind == "moe" else ("tp",)
    for family in families:
        passing, pending = [], []
        width = 1
        while width <= hardware.fast_domain_gpus:
            if width > 1 or family == families[0]:
                candidate = _assess(config, request, supplied, hardware, family, width)
                if candidate.status == "estimated_fit":
                    passing.append(candidate)
                elif candidate.status == "needs_inputs":
                    pending.append(candidate)
                else:
                    rejected.append(candidate)
            width *= 2
        candidates.extend((passing or pending)[:2])
    passing = [candidate for candidate in candidates if candidate.status == "estimated_fit"]
    default = min(passing, key=lambda candidate: candidate.required_gpus, default=None)
    payload = request.model_dump()
    payload["collection"].update(
        {field: supplied[field] for field in ("max_num_tokens", "max_batch_size") if field in supplied}
    )
    collection = SupportRequest.model_validate(payload).collection_settings()
    for field, scheduler_field in (("max_num_tokens", "max_batched_tokens"), ("max_batch_size", "max_sequences")):
        if field in supplied:
            collection["sources"][scheduler_field] = "user resource override; shared rank-local scheduler bound"
    grouped = any(
        candidate.draft and candidate.draft.resolved.get("cache_layout") == "grouped" for candidate in candidates
    )
    return TopologySuggestions(
        config_sha256=config.sha256,
        identity=request.identity.model_dump(mode="json"),
        collection=collection,
        context_length=request.search.context_length,
        overrides=supplied,
        hardware=hardware,
        candidates=tuple(candidates),
        rejected_candidates=tuple(rejected),
        default=default,
        assumptions=(
            hardware.assumption,
            "Enumerate powers-of-two widths within the fast domain; explicit topology choices may exceed it.",
            (
                "Check one request's conservative peak grouped allocation using Rust's window, block-rounding "
                "and transient prefill bound at the selected context and scheduler limits. No aggregate token "
                "capacity is assigned to mixed cache groups. Concurrent workload fit remains unchecked."
                if grouped
                else "Check room for one full search.context_length request plus one cached token of strict capacity "
                "headroom per rank. max_batch_size is a scheduler bound, not that many full-context cache allocations. "
                "Total KV capacity is estimated from the remaining per-rank memory; this check does not guarantee "
                "a particular concurrent workload fits or assume balanced attention-DP routing."
            ),
            f"The {request.collection.memory_fraction * 100:g}% byte budget uses the reviewed GPU memory fraction. "
            "CUDA graph reservations are excluded; "
            "generated plan and serving-runtime admission remain authoritative.",
            "Only known config geometry constraints are checked. Unknown architecture constraints, "
            "runtime compatibility, timing coverage and measured performance remain unchecked.",
            "Shortlist the first two complete fits per family, or pending-input choices when none fit; "
            "default to the smallest complete fit, breaking width ties by TP, DEP, then TEP. "
            "This ordering is not a performance ranking.",
            *([config.notes["modeling_scope"]] if "modeling_scope" in config.notes else []),
        ),
    )
