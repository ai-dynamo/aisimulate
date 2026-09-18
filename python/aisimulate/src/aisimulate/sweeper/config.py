# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input schema for a Sweeper smart-search run.

These Pydantic models are the single source of truth for the search inputs. See the repository's
``docs/sweeper/architecture.md`` for the experimental design:

- :class:`SearchSpace`        — the knobs to sweep + pinned context, per component
- :class:`Workload`           — the traffic every candidate is evaluated against
- :class:`OptimizationGoal`   — what "better" means + the SLA constraint
- :class:`SweepConfig`        — sweep run-control
- :class:`SmartSearchConfig`  — top-level bundle; one YAML maps to this
- :class:`Candidate`          — one evaluated configuration + its replay metrics

Field names are snake_case to match AIConfigurator's ``Task`` convention so the
eventual merge into an AIC sweep task is mechanical.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator


class OptimizationTarget(str, Enum):
    """What the search optimizes for.

    All members except ``pareto`` are scalar (single-objective) targets. ``pareto`` is a
    multi-objective mode: the search optimizes the Pareto tradeoff between the scalar
    targets listed in :attr:`OptimizationGoal.pareto_objectives` (default: throughput per
    GPU vs per-user throughput — the InferenceX tok/s/gpu vs tok/s/user frontier).
    """

    THROUGHPUT = "throughput"  # maximize replay throughput
    THROUGHPUT_PER_GPU = "throughput_per_gpu"  # maximize throughput / avg GPU (tok/s/gpu)
    THROUGHPUT_PER_USER = "throughput_per_user"  # maximize mean per-user output throughput (tok/s/user)
    TTFT = "ttft"  # minimize mean time to first token
    E2E_LATENCY = "e2e_latency"  # minimize mean end-to-end latency
    GOODPUT = "goodput"  # maximize SLA-satisfying throughput
    GOODPUT_PER_GPU = "goodput_per_gpu"  # maximize goodput / avg GPU (tok/s/gpu)
    PARETO = "pareto"  # multi-objective: Pareto front over pareto_objectives

    @property
    def maximize(self) -> bool:
        """True when larger is better (everything except e2e_latency).

        Raises for ``pareto`` — it has no single direction; use the per-objective
        directions in :attr:`OptimizationGoal.pareto_objectives` instead.
        """
        if self is OptimizationTarget.PARETO:
            raise ValueError("'pareto' is multi-objective and has no scalar direction")
        return self not in {OptimizationTarget.TTFT, OptimizationTarget.E2E_LATENCY}


class SLATarget(BaseModel):
    """Latency bounds in milliseconds.

    Replay treats every configured field as an independent per-request goodput
    bound; an unset field is unbounded. :attr:`OptimizationGoal.strict_sla`
    controls only the additional aggregate-mean filter.
    """

    model_config = ConfigDict(extra="forbid")

    ttft_ms: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)
    itl_ms: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)
    e2e_ms: float | None = Field(default=None, strict=True, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _validate_form(self) -> SLATarget:
        token_form = self.ttft_ms is not None or self.itl_ms is not None
        if token_form and self.e2e_ms is not None:
            raise ValueError("e2e_ms is mutually exclusive with ttft_ms/itl_ms")
        return self

    @property
    def has_bound(self) -> bool:
        """Whether at least one SLA bound is configured."""
        return any(value is not None for value in (self.ttft_ms, self.itl_ms, self.e2e_ms))


# Goodput-based scalar targets — the only ones that need an SLA (their metric counts
# only SLA-satisfying requests). Used to gate the SLA requirement on both the scalar
# target and the per-objective list under a pareto goal.
_SLA_TARGETS = frozenset({OptimizationTarget.GOODPUT, OptimizationTarget.GOODPUT_PER_GPU})

# Default Pareto objectives: throughput per GPU (y) vs mean per-user throughput (x) —
# the InferenceX tok/s/gpu vs tok/s/user frontier.
_DEFAULT_PARETO_OBJECTIVES = (
    OptimizationTarget.THROUGHPUT_PER_GPU,
    OptimizationTarget.THROUGHPUT_PER_USER,
)


class OptimizationGoal(BaseModel):
    """User-owned objective and SLA. Pinned; never searched."""

    model_config = ConfigDict(extra="forbid")

    target: OptimizationTarget = OptimizationTarget.THROUGHPUT
    sla: SLATarget | None = None  # required for goodput / goodput_per_gpu (scalar or pareto objective)
    # Only meaningful when target == pareto: the >=2 scalar objectives whose Pareto
    # front is sought. None -> the default pair (throughput_per_gpu, throughput_per_user).
    pareto_objectives: list[OptimizationTarget] | None = None
    # Preserve replay goodput's per-request SLA semantics by default. When
    # enabled, configured SLA bounds also gate aggregate mean metrics before
    # scalar ranking or Pareto dominance (legacy ``--strict-sla`` parity).
    strict_sla: bool = Field(default=False, strict=True)

    @property
    def resolved_pareto_objectives(self) -> list[OptimizationTarget]:
        """The effective Pareto objective list: the configured one, or the default pair only
        when unset (``None``). An explicitly-supplied empty/short list is kept as-is so the
        validator's ``len < 2`` guard rejects it (rather than silently using the default).
        """
        return list(_DEFAULT_PARETO_OBJECTIVES) if self.pareto_objectives is None else list(self.pareto_objectives)

    @property
    def is_pareto(self) -> bool:
        return self.target is OptimizationTarget.PARETO

    @model_validator(mode="after")
    def _validate_goal(self) -> OptimizationGoal:
        # pareto_objectives only applies to a pareto target.
        if not self.is_pareto and self.pareto_objectives is not None:
            raise ValueError("pareto_objectives is only valid when target is 'pareto'")
        if self.is_pareto:
            objs = self.resolved_pareto_objectives
            if len(objs) < 2:
                raise ValueError("a pareto goal needs at least 2 objectives")
            if OptimizationTarget.PARETO in objs:
                raise ValueError("pareto_objectives cannot contain 'pareto' itself (objectives must be scalar)")
            if len(set(objs)) != len(objs):
                raise ValueError(f"pareto_objectives must be distinct, got {[o.value for o in objs]}")
            effective = set(objs)
        else:
            effective = {self.target}
        # Any goodput-based objective (scalar target or pareto objective) needs an SLA.
        needs_sla = bool(effective & _SLA_TARGETS)
        has_sla = self.sla is not None and self.sla.has_bound
        if needs_sla and not has_sla:
            culprits = sorted(t.value for t in (effective & _SLA_TARGETS))
            raise ValueError(f"{culprits} require at least one SLA bound")
        if self.strict_sla and (self.sla is None or not self.sla.has_bound):
            raise ValueError("strict_sla requires at least one SLA bound")
        return self


class ImageWorkload(BaseModel):
    """One fixed image profile shared by every request (AIC encoder semantics)."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    height: int = Field(strict=True, gt=0)
    width: int = Field(strict=True, gt=0)
    count: int = Field(default=1, strict=True, gt=0)


class EncoderSearch(BaseModel):
    """Dedicated encoder pool; its backend follows the language backend."""

    model_config = ConfigDict(extra="forbid")
    hardware_sku: str | None = Field(default=None, min_length=1)
    backend_version: str | None = Field(default=None, min_length=1)
    tp: list[int] = [1, 2, 4, 8]
    batch_size: list[int] = [1, 2, 4, 8]
    workers: list[int] = [1, 2, 4, 8]
    latency_correction: float = Field(default=1.0, strict=True, gt=0, allow_inf_nan=False)
    rate_degradation: float = Field(default=0.9, strict=True, gt=0, le=1, allow_inf_nan=False)

    @field_validator("tp", "batch_size", "workers", mode="before")
    @classmethod
    def _positive_choices(cls, value):
        if not isinstance(value, list) or not value or any(type(v) is not int or v <= 0 for v in value):
            raise ValueError("encoder choices must be nonempty lists of positive integers")
        return list(dict.fromkeys(value))

    @field_validator("batch_size")
    @classmethod
    def _batch_limit(cls, value):
        if max(value) > 8:
            raise ValueError("encoder batch_size must be <= 8, matching AIC EPD")
        return value


class Workload(BaseModel):
    """Traffic every candidate is evaluated against (KV load may be searched for Pareto).

    Exactly one of **four load shapes** (all replayable with or without the planner):

    1. **mooncake trace** — set ``trace_path``. Open-loop at the trace's arrival
       timestamps (scale with ``arrival_speedup_ratio``); set ``replay_concurrency``
       to drive it **closed-loop** (cap N in flight, ignore timestamps).
    2. **synthetic request-rate** — set ``request_rate`` (+ ``isl``/``osl``/``num_request_ratio``):
       open-loop at a fixed QPS.
    3. **synthetic concurrency** — set ``concurrency`` (+ ``isl``/``osl``/``num_request_ratio``):
       closed-loop, cap N in flight.
    4. **synthetic KV load** — set ``kv_load_ratio`` (+ ``isl``/``osl``/``num_request_ratio``):
       closed-loop, with concurrency derived from each candidate's aggregate decode/agg KV
       capacity. A two-value range is searchable only under a ``pareto`` goal.

    The mode is inferred from which field is set; see the validator.

    ``concurrency`` is always one fixed positive integer. Under a ``pareto`` goal,
    ``kv_load_ratio`` may instead be a ``[min, max]`` continuous search range; when no
    synthetic load is specified, :class:`SmartSearchConfig` defaults that range to ``[0, 1]``.

    ``num_request_ratio`` (synthetic only) sets the request count **relative to the load**:
    ``num_requests = round(num_request_ratio * load)`` where ``load`` is ``concurrency``
    (closed-loop) or ``request_rate`` (open-loop). So the synthetic trace length scales with
    the concrete concurrency automatically — e.g. ratio 10 at concurrency 256 -> 2560 requests.
    """

    model_config = ConfigDict(extra="forbid")

    # synthetic workload (used when trace_path is unset): exactly one of
    # request_rate (open-loop QPS), concurrency (fixed closed-loop in-flight cap), or
    # kv_load_ratio (candidate-relative closed-loop load).
    images: ImageWorkload | None = None
    isl: int | None = None
    osl: int | None = None
    concurrency: int | None = None
    kv_load_ratio: float | list[float] | None = None
    request_rate: float | None = None
    request_count: int | None = None
    num_request_ratio: float | None = None  # request count multiplier for concrete concurrency or request_rate
    random_range_ratio: float = 1.0
    random_seed: int = 0
    shared_prefix_ratio: float = 0.0  # cache-locality / prefix sharing
    num_prefix_groups: int = 0
    turns_per_session: int = 1  # multi-turn sessions
    inter_turn_delay_ms: float = 0.0  # think-time between turns (multi-turn synthetic)
    arrival_seed: int = 42
    source_type: str | None = None
    load_type: str | None = None
    # Unified-CLI internal search dimension for a public traffic.load field.
    load_search_field: str | None = None
    load_choices: list[int | float] | None = None
    load_range: list[float] | None = None
    load_integer: bool = False
    load_log_scale: bool = False

    # dynamic trace source (mutually exclusive with the synthetic fields)
    trace_path: str | None = None
    trace_paths: list[str] | None = None
    trace_block_size: int | None = None
    trace_format: str = "mooncake"  # replay-ready trace schema
    weka_nested_timestamp_basis: Literal["auto", "absolute", "relative"] | None = None
    arrival_speedup_ratio: float = 1.0  # scale trace inter-arrival times
    agentic_lanes: int | None = Field(default=None, strict=True, gt=0)
    # Closed-loop replay over a *trace*: cap in-flight requests at this many (the
    # trace's timestamps are ignored; a new request starts as one finishes). For a
    # *synthetic* closed-loop workload use ``concurrency`` or ``kv_load_ratio`` instead.
    replay_concurrency: int | None = None
    max_sim_time_ms: float | None = None

    def require_fixed_epd(self) -> None:
        """Validate the analytical approximation at both search and replay boundaries."""
        if self.images is None or self.isl is None or self.osl is None or self.isl <= 0 or self.osl <= 0:
            raise ValueError("EPD requires positive text lengths and an image profile")
        if type(self.concurrency) is not int or self.concurrency <= 0:
            raise ValueError("analytical EPD requires fixed positive concurrency")
        if (
            self.trace_path is not None
            or self.trace_paths is not None
            or self.source_type not in (None, "synthetic")
            or (self.source_type is None) != (self.load_type is None)
            or self.kv_load_ratio is not None
            or self.load_search_field is not None
            or self.load_choices is not None
            or self.load_range is not None
            or self.random_range_ratio != 1.0
            or self.turns_per_session != 1
            or self.shared_prefix_ratio != 0.0
            or self.num_prefix_groups != 0
            or self.inter_turn_delay_ms != 0.0
            or self.max_sim_time_ms is not None
            or self.load_type not in (None, "concurrency")
            or self.replay_concurrency is not None
        ):
            raise ValueError("analytical EPD requires fixed synthetic traffic without traces, sessions or load search")

    @field_validator("random_range_ratio", mode="before")
    @classmethod
    def _validate_random_range_ratio_type(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"random_range_ratio must be a number, got {value!r}")
        return value

    @field_validator("random_seed", mode="before")
    @classmethod
    def _validate_random_seed_type(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"random_seed must be an unsigned 64-bit integer, got {value!r}")
        return value

    @property
    def is_trace_based(self) -> bool:
        return self.trace_path is not None

    @property
    def is_synthetic(self) -> bool:
        return self.trace_path is None

    @property
    def kv_load_ratio_range(self) -> tuple[float, float] | None:
        """The Pareto-only continuous KV-load range, or ``None`` for a scalar/other load."""
        if isinstance(self.kv_load_ratio, list):
            return float(self.kv_load_ratio[0]), float(self.kv_load_ratio[1])
        return None

    def effective_in_flight_cap(self, concurrency_override: int | None = None) -> int | None:
        """Closed-loop in-flight cap (``None`` = open-loop). ``concurrency_override`` (the
        per-trial value derived from KV load) wins; then ``replay_concurrency`` for a
        trace, then the fixed ``concurrency`` for a synthetic workload. KV-load mode always
        supplies its candidate-derived concurrency as the override."""
        if self.trace_path is not None:
            return self.replay_concurrency
        if concurrency_override is not None:
            return concurrency_override
        if self.concurrency is not None:
            return self.concurrency
        return None

    def resolved_request_count(self, concurrency_override: int | None = None) -> int:
        """Synthetic request count = ``round(num_request_ratio * load)`` (>= 1), where
        ``load`` is the in-flight concurrency (closed-loop) or the request rate (open-loop).
        ``concurrency_override`` is the candidate-specific concurrency in KV-load mode.
        """
        if concurrency_override is not None:
            load: float = concurrency_override
        elif self.concurrency is not None:
            load = self.concurrency
        elif self.request_rate is not None:
            load = self.request_rate
        else:
            raise ValueError("resolved_request_count needs a concurrency_override for a kv_load_ratio workload")
        return max(1, round((self.num_request_ratio or 0.0) * load))

    @property
    def synthetic_arrival_interval_ms(self) -> float | None:
        """Mean inter-arrival for a synthetic request-rate workload.

        Closed-loop workloads return ``None`` so Replay receives only their
        ``replay_concurrency`` load controller.
        """
        if self.request_rate is None:
            return None
        return 1000.0 / self.request_rate

    @model_validator(mode="after")
    def _validate_workload(self) -> Workload:
        if self.weka_nested_timestamp_basis is not None and (
            self.trace_path is None or self.source_type != "trace" or self.trace_format != "weka"
        ):
            raise ValueError("weka_nested_timestamp_basis requires Weka trace input")
        if self.agentic_lanes is not None:
            if (
                self.trace_path is None
                or self.source_type != "trace"
                or self.trace_format not in {"weka", "agentic_mooncake", "dynamo"}
            ):
                raise ValueError("agentic_lanes requires weka, agentic_mooncake, or agentic Dynamo trace input")
            if (
                self.load_type != "trace_timestamps"
                or self.replay_concurrency is not None
                or self.load_search_field == "replay_concurrency"
            ):
                raise ValueError("agentic_lanes requires trace_timestamps load without replay_concurrency")
        synthetic_only = (
            "isl",
            "osl",
            "request_rate",
            "concurrency",
            "kv_load_ratio",
            "num_request_ratio",
        )
        if self.trace_path is not None:
            set_syn = [n for n in synthetic_only if getattr(self, n) is not None]
            if self.random_range_ratio != 1.0:
                set_syn.append("random_range_ratio")
            if self.random_seed != 0:
                set_syn.append("random_seed")
            if set_syn:
                raise ValueError(f"trace workload (trace_path set) must not set synthetic fields {set_syn}")
            if self.replay_concurrency is not None and self.replay_concurrency <= 0:
                raise ValueError(f"replay_concurrency must be a positive integer, got {self.replay_concurrency}")
            return self
        # synthetic: exactly one load mode, plus isl/osl/num_request_ratio
        loads = [n for n in ("request_rate", "concurrency", "kv_load_ratio") if getattr(self, n) is not None]
        if len(loads) != 1:
            raise ValueError(
                "a synthetic workload needs exactly one of request_rate, concurrency, or kv_load_ratio "
                "(or set trace_path for a trace workload)"
            )
        missing = [n for n in ("isl", "osl") if getattr(self, n) is None]
        if self.request_count is None and self.num_request_ratio is None:
            missing.append("request_count or num_request_ratio")
        if missing:
            raise ValueError(f"a synthetic workload requires {missing}")
        if self.replay_concurrency is not None:
            raise ValueError("replay_concurrency is for trace workloads; use 'concurrency' for synthetic closed-loop")
        if self.concurrency is not None and (isinstance(self.concurrency, bool) or self.concurrency <= 0):
            raise ValueError(f"concurrency must be a positive integer, got {self.concurrency!r}")
        if self.request_count is not None and (isinstance(self.request_count, bool) or self.request_count <= 0):
            raise ValueError("request_count must be a positive integer")
        if self.load_choices is not None and not self.load_choices:
            raise ValueError("load_choices must be nonempty")
        if self.load_range is not None and (len(self.load_range) != 2 or self.load_range[0] >= self.load_range[1]):
            raise ValueError("load_range must contain [min, max] with min < max")
        if self.kv_load_ratio is not None:
            ratios = self.kv_load_ratio if isinstance(self.kv_load_ratio, list) else [self.kv_load_ratio]
            if isinstance(self.kv_load_ratio, list) and len(ratios) != 2:
                raise ValueError("kv_load_ratio range must contain exactly [min, max]")
            if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in ratios):
                raise ValueError(f"kv_load_ratio values must be finite and non-negative, got {self.kv_load_ratio!r}")
            if isinstance(self.kv_load_ratio, list) and float(ratios[0]) >= float(ratios[1]):
                raise ValueError(f"kv_load_ratio range needs min < max, got {self.kv_load_ratio!r}")
        for name in ("request_rate", "isl", "osl", "num_request_ratio"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        if (
            not math.isfinite(self.random_range_ratio)
            or self.random_range_ratio <= 0.0
            or self.random_range_ratio > 1.0
        ):
            raise ValueError(f"random_range_ratio must be finite and in (0.0, 1.0], got {self.random_range_ratio!r}")
        for name in ("isl", "osl"):
            length = getattr(self, name)
            if length is not None and int(length * self.random_range_ratio) == 0:
                raise ValueError(
                    f"random_range_ratio={self.random_range_ratio} gives a zero-token lower bound for {name}={length}"
                )
        if isinstance(self.random_seed, bool) or self.random_seed < 0 or self.random_seed > 0xFFFF_FFFF_FFFF_FFFF:
            raise ValueError(f"random_seed must be an unsigned 64-bit integer, got {self.random_seed!r}")
        if self.random_range_ratio != 1.0 and self.turns_per_session != 1:
            raise ValueError("random_range_ratio currently only supports single-turn synthetic workloads")
        return self


# Allowed choices for each swept search-space dimension. A configured value must
# be a non-empty subset of these (one or more); the field defaults below use the
# full set (or a sensible subset, e.g. ``backend``). Centralized here so the
# candidate generator can reuse it. Pinned scalars and the generated
# ``parallel_configs`` are intentionally not choice-constrained.
SEARCH_CHOICES: dict[str, tuple] = {
    "deployment_mode": ("disagg", "agg", "afd", "afd+pd"),
    "backend": ("vllm", "sglang", "trtllm"),
}

FORWARD_MODEL_CHOICES: tuple[str, ...] = ("op_level", "fpm")


class SearchSpace(BaseModel):
    """Dynamo-independent backend inputs to one search run.

    Each group lists its swept knobs (list-typed candidate sets; a
    single-element list pins that knob) followed by the pinned knobs that group
    needs (scalars). When ``deployment_mode`` lists both branches the optimizer
    runs one flat study per branch and ranks across both. Dynamo Planner and
    Router search spaces live under :class:`SmartSearchConfig.adapters`.
    """

    model_config = ConfigDict(extra="forbid")

    # deployment: branch + backend + legal parallel shapes
    deployment_mode: list[str] = ["disagg", "agg"]  # branches to explore; pin with one
    backend: list[str] = ["vllm"]  # vllm | sglang | trtllm
    backend_version: str | dict[str, str] | None = None
    parallel_configs: list[dict[str, Any]] = Field(default_factory=list)  # generated when empty
    parallel_configs_by_mode: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    flat_parallel_modes: list[str] = Field(default_factory=list)
    parallel_independent_by_mode: dict[str, dict[str, list[int] | None]] = Field(default_factory=dict)
    parallel_independent_log_ranges_by_mode: dict[str, dict[str, list[int]]] = Field(default_factory=dict)
    parallel_custom_configs_by_mode: dict[str, dict[str, list[dict[str, Any]]]] = Field(default_factory=dict)
    # pinned
    model_name: str  # HF id or private model name
    hardware_sku: str  # e.g. "h200_sxm"
    database_mode: Literal["SILICON", "HYBRID", "EMPIRICAL", "SOL"] = "SILICON"
    transfer_policy: str | list[str] | None = None
    systems_paths: list[str] | None = Field(default=None, min_length=1)
    estimation_mode: Literal["auto", "op_level", "fpm_interpolation", "fpm_regression"] = "auto"
    fallback_policy: Literal["deny", "allow"] = "deny"
    estimator_config: dict[str, Any] = Field(default_factory=dict)
    role_estimator_controls: dict[str, dict[str, Any]] = Field(default_factory=dict)
    prefill_hardware_sku: str | None = Field(default=None, min_length=1)
    decode_hardware_sku: str | None = Field(default=None, min_length=1)
    gpu_budget: int = 32  # max GPUs per candidate
    min_gpu_budget: int | None = None
    context_length: int | None = None
    startup_time: float | None = None
    aic_nextn: int | None = None  # speculative-decode (MTP) depth, 1..5
    encoder: EncoderSearch | None = None

    # Attention--FFN disaggregation. The A/F topology is a finite, complete
    # domain; ``afd+pd`` pairs each topology with one opposite-phase companion.
    afd_pinned_topologies: list[dict[str, Any]] = Field(default_factory=list)
    afd_companion_parallel_configs: list[dict[str, Any]] = Field(default_factory=list)
    afd_tp_a_candidates: list[int] | None = None
    afd_batch_size_candidates: list[int] | None = None
    afd_f_moe_ep_size_candidates: list[int | str] | None = None
    afd_microbatch_candidates: list[int] = Field(default_factory=lambda: [2, 3, 4])
    afd_pipeline_model_candidates: list[str] = Field(default_factory=lambda: ["optimistic", "conservative"])
    afd_phase: str = "decode"
    afd_comm_overhead_factor: float = Field(default=1.0, gt=0)
    afd_boundary_on_attn: bool = True
    afd_max_af_ratio: float = Field(default=4.0, gt=0)
    afd_max_candidates: int = Field(default=10_000, ge=1)

    # prefill engine (disagg branch): scheduler batching capacity
    prefill_max_num_batched_tokens: list[int] = [8192, 16384, 32768]
    prefill_max_num_seqs: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # pinned
    prefill_block_size: int | list[int] | None = 64
    prefill_gpu_memory_utilization: float | list[float] | None = 0.9
    prefill_enable_prefix_caching: bool = True
    prefill_kv_bytes_per_token: int | str = "auto"
    prefill_native_host_offload: dict[str, Any] | None = None
    prefill_num_gpu_blocks: int | None = None
    prefill_timing_model: dict[str, Any] | None = None
    prefill_forward_model: str = "op_level"
    prefill_startup_time: float | None = None

    # decode engine (disagg branch): scheduler batching capacity
    decode_max_num_batched_tokens: list[int] = [8192]
    decode_max_num_seqs: list[int] = [256, 512, 1024]
    # pinned
    decode_block_size: int | list[int] | None = 64
    decode_gpu_memory_utilization: float | list[float] | None = 0.9
    decode_enable_prefix_caching: bool = False  # forced off for decode workers
    decode_kv_bytes_per_token: int | str = "auto"
    decode_native_host_offload: dict[str, Any] | None = None
    decode_num_gpu_blocks: int | None = None
    decode_timing_model: dict[str, Any] | None = None
    decode_forward_model: str = "op_level"
    decode_startup_time: float | None = None

    # agg engine (agg branch): scheduler batching capacity
    agg_max_num_batched_tokens: list[int] = [8192, 16384, 32768]
    agg_max_num_seqs: list[int] = [256, 512, 1024]
    # pinned
    agg_block_size: int | list[int] | None = 64
    agg_gpu_memory_utilization: float | list[float] | None = 0.9
    agg_enable_prefix_caching: bool = True
    agg_kv_bytes_per_token: int | str = "auto"
    agg_native_host_offload: dict[str, Any] | None = None
    agg_num_gpu_blocks: int | None = None
    agg_timing_model: dict[str, Any] | None = None
    agg_forward_model: str = "op_level"
    agg_startup_time: float | None = None
    kv_transfer_bytes_per_token: int | str | None = None
    kv_transfer_bandwidth: float | None = None
    kv_transfer_timing_mode: str = "destination_missing"
    engine_float_ranges: dict[str, list[float]] = Field(default_factory=dict)
    engine_log_ranges: list[str] = Field(default_factory=list)
    engine_log_discrete: list[str] = Field(default_factory=list)
    engine_integer_log_ranges: dict[str, list[int]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_estimator_selection(cls, value):
        if not isinstance(value, dict):
            return value
        modes = value.get("deployment_mode")
        if (isinstance(modes, list) and any(mode in ("afd", "afd+pd") for mode in modes)) or value.get(
            "encoder"
        ) is not None:
            return value
        value = dict(value)
        raw_controls = value.get("role_estimator_controls", {})
        if not isinstance(raw_controls, dict) or any(
            not isinstance(settings, dict) for settings in raw_controls.values()
        ):
            return value
        controls = {role: dict(settings) for role, settings in raw_controls.items()}
        for role in ("agg", "prefill", "decode"):
            legacy = value.get(f"{role}_forward_model")
            if legacy == "op_level" and value.get(f"{role}_timing_model") is not None:
                continue
            if isinstance(legacy, str) and legacy in {"op_level", "fpm"}:
                controls.setdefault(role, {}).setdefault(
                    "estimation_mode", "fpm_interpolation" if legacy == "fpm" else "op_level"
                )
        value["role_estimator_controls"] = controls
        return value

    @model_validator(mode="after")
    def _validate_search_choices(self) -> SearchSpace:
        """Every backend dimension is a non-empty subset of its allowed choices."""
        for field_name, allowed in SEARCH_CHOICES.items():
            values = getattr(self, field_name)
            if not values:
                raise ValueError(f"{field_name} must list at least one choice; allowed: {list(allowed)}")
            for v in values:
                if v not in allowed:
                    raise ValueError(f"{field_name} has invalid choice {v!r}; allowed: {list(allowed)}")
        for field_name in (
            "prefill_max_num_batched_tokens",
            "prefill_max_num_seqs",
            "decode_max_num_batched_tokens",
            "decode_max_num_seqs",
            "agg_max_num_batched_tokens",
            "agg_max_num_seqs",
        ):
            values = getattr(self, field_name)
            if not values or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
            ):
                raise ValueError(f"{field_name} must contain positive integers")
        for field_name in ("prefill_forward_model", "decode_forward_model", "agg_forward_model"):
            value = getattr(self, field_name)
            if value not in FORWARD_MODEL_CHOICES:
                raise ValueError(f"{field_name} has invalid choice {value!r}; allowed: {list(FORWARD_MODEL_CHOICES)}")
        if not self._uses_legacy_estimator_provider():
            for role in ("agg", "prefill", "decode"):
                mode = self.role_estimator_controls.get(role, {}).get("estimation_mode", self.estimation_mode)
                setattr(self, f"{role}_forward_model", "fpm" if mode == "fpm_interpolation" else "op_level")
        return self

    def _uses_legacy_estimator_provider(self) -> bool:
        return bool(set(self.deployment_mode) & {"afd", "afd+pd"}) or self.encoder is not None

    @model_serializer(mode="wrap")
    def _serialize_estimator_selection(self, handler):
        result = handler(self)
        # Ordinary workers serialize only the canonical controls. Existing AFD
        # and encoder providers still consume the legacy selector, including
        # when a saved search is reloaded.
        if not self._uses_legacy_estimator_provider():
            for role in ("agg", "prefill", "decode"):
                result.pop(f"{role}_forward_model", None)
        return result

    @model_validator(mode="after")
    def _validate_role_hardware(self) -> SearchSpace:
        """Role-specific hardware is an override for ordinary P/D only."""
        if (self.prefill_hardware_sku is not None or self.decode_hardware_sku is not None) and (
            "disagg" not in self.deployment_mode
        ):
            raise ValueError("prefill_hardware_sku and decode_hardware_sku require deployment_mode to include 'disagg'")
        return self

    def hardware_sku_for(self, role: str) -> str:
        """Return the effective hardware SKU for an ordinary engine role."""
        if role == "agg":
            return self.hardware_sku
        if role == "prefill":
            return self.prefill_hardware_sku or self.hardware_sku
        if role == "decode":
            return self.decode_hardware_sku or self.hardware_sku
        raise ValueError(f"unknown engine role {role!r}")

    def systems_paths_for(self, role: str) -> list[str] | None:
        return self.role_estimator_controls.get(role, {}).get("systems_paths", self.systems_paths)

    @field_validator(
        "afd_tp_a_candidates",
        "afd_batch_size_candidates",
        "afd_microbatch_candidates",
        mode="before",
    )
    @classmethod
    def _validate_afd_positive_candidates(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list) or not value:
            raise ValueError("AFD candidate lists must be nonempty lists")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
            raise ValueError(f"AFD candidate lists need positive integers, got {value!r}")
        if len(set(value)) != len(value):
            raise ValueError("AFD candidate lists must not contain duplicates")
        return value

    @field_validator("afd_f_moe_ep_size_candidates", mode="before")
    @classmethod
    def _validate_afd_ep_candidates(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list) or not value:
            raise ValueError("afd_f_moe_ep_size_candidates must be a nonempty list")
        allowed_symbols = {"n_f_nodes", "ffn_tp", "tp_f"}
        invalid = [
            item
            for item in value
            if not (
                (isinstance(item, int) and not isinstance(item, bool) and item > 0)
                or (isinstance(item, str) and item in allowed_symbols)
            )
        ]
        if invalid:
            raise ValueError(
                "afd_f_moe_ep_size_candidates accepts positive integers, "
                f"'n_f_nodes', 'ffn_tp', or 'tp_f'; got {invalid!r}"
            )
        if len({(type(item).__name__, item) for item in value}) != len(value):
            raise ValueError("afd_f_moe_ep_size_candidates must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _validate_afd_contract(self) -> SearchSpace:
        afd_modes = [mode for mode in dict.fromkeys(self.deployment_mode) if mode in {"afd", "afd+pd"}]
        if self.afd_phase not in {"prefill", "decode", "both"}:
            raise ValueError("afd_phase must be 'prefill', 'decode', or 'both'")
        if any(value not in {"optimistic", "conservative", "serial"} for value in self.afd_pipeline_model_candidates):
            raise ValueError("afd_pipeline_model_candidates accepts optimistic, conservative, or serial")
        if len(set(self.afd_pipeline_model_candidates)) != len(self.afd_pipeline_model_candidates):
            raise ValueError("afd_pipeline_model_candidates must not contain duplicates")
        if "afd+pd" in afd_modes and self.afd_phase == "both":
            raise ValueError("deployment_mode='afd+pd' requires afd_phase prefill or decode")
        if afd_modes and not self.afd_pinned_topologies and self.afd_batch_size_candidates is None:
            raise ValueError(
                "searched AFD requires explicit, memory-qualified afd_batch_size_candidates; "
                "legacy topology-specific batch derivation must not become an implicit fixed batch"
            )
        if self.afd_pinned_topologies:
            if len(afd_modes) != 1:
                raise ValueError("afd_pinned_topologies requires exactly one AFD deployment_mode")
            required = {"n_a_nodes", "n_f_nodes", "tp_a", "a_batch_size"}
            for index, topology in enumerate(self.afd_pinned_topologies):
                if not isinstance(topology, dict):
                    raise ValueError(f"afd_pinned_topologies[{index}] must be a mapping")
                missing = sorted(required - topology.keys())
                if missing:
                    raise ValueError(f"afd_pinned_topologies[{index}] is missing {missing}")
        if self.afd_companion_parallel_configs and "afd+pd" not in afd_modes:
            raise ValueError("afd_companion_parallel_configs requires deployment_mode='afd+pd'")
        for index, companion in enumerate(self.afd_companion_parallel_configs):
            if not isinstance(companion, dict) or "tp" not in companion:
                raise ValueError(f"afd_companion_parallel_configs[{index}] must be a flat shape with tp")
            replicas = companion.get("replicas", 1)
            if isinstance(replicas, bool) or not isinstance(replicas, int) or replicas < 1:
                raise ValueError(f"afd_companion_parallel_configs[{index}].replicas must be positive")
        return self

    @field_validator("database_mode", mode="before")
    @classmethod
    def _normalize_estimator_database_mode(cls, value):
        return value.upper() if isinstance(value, str) else value

    @field_validator("systems_paths")
    @classmethod
    def _validate_estimator_roots(cls, value):
        if value is None:
            return value
        if any(not path.strip() for path in value):
            raise ValueError("systems_paths entries must be nonempty")
        return value

    @model_validator(mode="after")
    def _validate_estimator_controls(self):
        allowed = {
            "estimation_mode",
            "fallback_policy",
            "estimator_config",
            "systems_paths",
            "database_mode",
            "transfer_policy",
        }
        for role, controls in self.role_estimator_controls.items():
            if role not in {"agg", "prefill", "decode"}:
                raise ValueError(f"unknown estimator role {role!r}; expected agg, prefill, or decode")
            if set(controls) - allowed:
                raise ValueError(f"unknown estimator override for role {role!r}: {sorted(set(controls) - allowed)}")
            if controls and self._uses_legacy_estimator_provider():
                raise ValueError(
                    "role_estimator_controls require regular language workers; "
                    "AFD and encoder providers are unsupported"
                )
            if controls and getattr(self, f"{role}_timing_model") is not None:
                raise ValueError(f"{role} estimator settings require default timing")
            mode = controls.get("database_mode")
            if isinstance(mode, str) and mode.upper() == "SOL_FULL":
                raise ValueError(
                    f"role_estimator_controls.{role}.database_mode cannot be SOL_FULL; it is a per-call diagnostic"
                )
            if "systems_paths" in controls:
                paths = controls["systems_paths"]
                if (
                    not isinstance(paths, list)
                    or not paths
                    or any(not isinstance(p, str) or not p.strip() for p in paths)
                ):
                    raise ValueError(f"{role} systems_paths must contain nonempty strings")
        nondefault = (
            self.database_mode != "SILICON"
            or self.transfer_policy is not None
            or self.systems_paths not in (None, ["default"])
            or self.estimation_mode != "auto"
            or self.fallback_policy != "deny"
            or bool(self.estimator_config)
        )
        roles = ({"agg"} if "agg" in self.deployment_mode else set()) | (
            {"prefill", "decode"} if "disagg" in self.deployment_mode else set()
        )
        if nondefault and (
            set(self.deployment_mode) & {"afd", "afd+pd"}
            or self.encoder is not None
            or any(getattr(self, f"{role}_timing_model") is not None for role in roles)
        ):
            raise ValueError("estimator policies require regular language workers with default timing in every role")
        return self

    @model_validator(mode="after")
    def _validate_backend_versions(self) -> SearchSpace:
        configured = list(dict.fromkeys(self.backend))
        if isinstance(self.backend_version, str):
            if not self.backend_version.strip():
                raise ValueError("backend_version must be a non-empty version")
            if len(configured) != 1:
                raise ValueError(
                    "a string backend_version requires exactly one configured backend; "
                    "use a {backend: version} mapping for a multi-backend search"
                )
            self.backend_version = self.backend_version.strip()
        elif isinstance(self.backend_version, dict):
            unknown = sorted(set(self.backend_version) - set(configured))
            if unknown:
                raise ValueError(f"backend_version contains unconfigured backend(s): {unknown}")
            invalid = [
                backend
                for backend, version in self.backend_version.items()
                if not isinstance(version, str) or not version.strip()
            ]
            if invalid:
                raise ValueError(f"backend_version needs a non-empty version for {sorted(invalid)}")
            self.backend_version = {backend: version.strip() for backend, version in self.backend_version.items()}
        return self

    def requested_backend_version(self, backend: str) -> str | None:
        """Return the version pin for ``backend``; ``None`` means resolve latest."""

        if isinstance(self.backend_version, str):
            return self.backend_version
        if isinstance(self.backend_version, dict):
            return self.backend_version.get(backend)
        return None

    @model_validator(mode="after")
    def _validate_gpu_budget(self) -> SearchSpace:
        """A minimum GPU budget must be positive and within the maximum budget."""
        if self.min_gpu_budget is not None and not (0 < self.min_gpu_budget <= self.gpu_budget):
            raise ValueError(
                f"min_gpu_budget must satisfy 0 < min_gpu_budget <= gpu_budget "
                f"(got min_gpu_budget={self.min_gpu_budget}, gpu_budget={self.gpu_budget})"
            )
        return self

    @model_validator(mode="after")
    def _validate_parallel_configs(self) -> SearchSpace:
        """A pinned ``parallel_configs`` (non-empty) must match a single deployment
        mode and have the right shape: an agg entry is a flat shape dict (needs
        ``tp``); a disagg entry nests ``prefill`` + ``decode`` shape dicts. Full
        legality (MoE width, KV feasibility, GPU budget) is checked in
        ``enumerate_branches`` against the model+hardware."""

        def validate_shape_dict(value: Any, label: str) -> None:
            if not isinstance(value, dict):
                raise ValueError(f"{label} parallel_configs shape must be a dict")
            if "tp" not in value:
                raise ValueError(f"{label} parallel_configs shape needs a 'tp' field")

        if self.parallel_configs and len(self.deployment_mode) != 1:
            raise ValueError(
                "pinning parallel_configs requires deployment_mode to list exactly one mode "
                f"(got {self.deployment_mode}); pin the mode too"
            )
        configured: dict[str, list[dict[str, Any]]] = dict(self.parallel_configs_by_mode)
        if self.parallel_configs:
            configured[self.deployment_mode[0]] = self.parallel_configs
        unknown_modes = set(configured) - {"agg", "disagg"}
        if unknown_modes:
            raise ValueError(f"parallel_configs_by_mode has unknown modes {sorted(unknown_modes)}")
        inactive_modes = set(configured) - set(self.deployment_mode)
        if inactive_modes:
            raise ValueError(f"parallel_configs_by_mode configures inactive modes {sorted(inactive_modes)}")
        flat_modes = set(self.flat_parallel_modes)
        independent_modes = set(self.parallel_independent_by_mode)
        independent_log_modes = set(self.parallel_independent_log_ranges_by_mode)
        custom_modes = set(self.parallel_custom_configs_by_mode)
        invalid_special = (flat_modes | independent_modes | independent_log_modes | custom_modes) - set(
            self.deployment_mode
        )
        if invalid_special:
            raise ValueError(f"parallel search mode configures inactive modes {sorted(invalid_special)}")
        if flat_modes - set(configured):
            raise ValueError("flat_parallel_modes require pinned configs for each mode")
        if flat_modes & independent_modes:
            raise ValueError("parallel mode cannot be both flat and independent")
        if flat_modes & (independent_log_modes | custom_modes):
            raise ValueError("flat parallel mode cannot also configure mixed-role search")
        for mode, fields in self.parallel_independent_by_mode.items():
            if not fields:
                raise ValueError(f"parallel_independent_by_mode.{mode} must be nonempty")
            base_names = {
                "replicas",
                "tp",
                "pp",
                "attention_dp",
                "moe_tp",
                "moe_ep",
            }
            allowed_names = (
                base_names
                if mode == "agg"
                else {f"{role}_{name}" for role in ("prefill", "decode") for name in base_names}
            )
            unknown_names = set(fields) - allowed_names
            if unknown_names:
                raise ValueError(f"parallel_independent_by_mode.{mode} has unknown knobs {sorted(unknown_names)}")
            for name, values in fields.items():
                if values is not None and (
                    not values
                    or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values)
                ):
                    raise ValueError(f"parallel_independent_by_mode.{mode}.{name} must contain positive integers")
        for mode, fields in self.parallel_independent_log_ranges_by_mode.items():
            independent = self.parallel_independent_by_mode.get(mode, {})
            unknown_names = set(fields) - set(independent)
            if unknown_names:
                raise ValueError(
                    f"parallel_independent_log_ranges_by_mode.{mode} has knobs that are not "
                    f"independent: {sorted(unknown_names)}"
                )
            for name, bounds in fields.items():
                if (
                    len(bounds) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in bounds)
                    or bounds[0] <= 0
                    or bounds[0] > bounds[1]
                ):
                    raise ValueError(
                        f"parallel_independent_log_ranges_by_mode.{mode}.{name} "
                        "must be positive integer [min, max] bounds"
                    )
        for mode, roles in self.parallel_custom_configs_by_mode.items():
            allowed_roles = {"agg"} if mode == "agg" else {"prefill", "decode"}
            unknown_roles = set(roles) - allowed_roles
            if unknown_roles:
                raise ValueError(f"parallel_custom_configs_by_mode.{mode} has unknown roles {sorted(unknown_roles)}")
            for role, entries in roles.items():
                if not entries:
                    raise ValueError(f"parallel_custom_configs_by_mode.{mode}.{role} must be nonempty")
                for entry in entries:
                    validate_shape_dict(entry, f"a {mode} {role}")
        for mode, entries in configured.items():
            if not entries:
                raise ValueError(f"parallel_configs_by_mode.{mode} must be nonempty")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("each parallel_configs entry must be a dict")
                if mode == "agg":
                    validate_shape_dict(entry, "an agg")
                else:
                    if "prefill" not in entry or "decode" not in entry:
                        raise ValueError("a disagg parallel_configs entry needs 'prefill' and 'decode' sub-dicts")
                    validate_shape_dict(entry["prefill"], "a disagg prefill")
                    validate_shape_dict(entry["decode"], "a disagg decode")
        for name, bounds in self.engine_integer_log_ranges.items():
            if (
                len(bounds) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in bounds)
                or bounds[0] <= 0
                or bounds[0] > bounds[1]
            ):
                raise ValueError(f"engine_integer_log_ranges.{name} must be positive integer [min, max] bounds")
        return self


class SweepConfig(BaseModel):
    """Sweep run-control."""

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=20, ge=1)  # total Vizier/replay barrier rounds
    parallel_evals: int = Field(default=16, ge=1)  # replay worker fan-out and default candidates per round
    # Successful unique replay configs per round; duplicate projections are told from
    # cache and replaced. Defaults to parallel_evals.
    candidates_per_round: int | None = Field(default=None, ge=1)
    # Per-candidate wall-clock cap for the replay. A candidate whose replay exceeds this is
    # killed and reported as infeasible ("exceed runtime") so the optimizer avoids that region
    # instead of hanging the sweep (e.g. an over-subscribed config that churns). Only enforced
    # on the worker-pool path (parallel_evals > 1); None disables the cap.
    max_eval_seconds: float | None = Field(default=600.0, gt=0)
    # Unified-CLI controls. ``None`` preserves the historical round-based SDK
    # contract; a concrete value is a hard suggestion budget across branches.
    max_trials: int | None = Field(default=None, ge=1)
    algorithm: str = "bayesian"
    seed: int = Field(default=42, ge=0)

    @field_validator("algorithm")
    @classmethod
    def _validate_algorithm(cls, value: str) -> str:
        if value not in {"bayesian", "random"}:
            raise ValueError("algorithm must be 'bayesian' or 'random'")
        return value


class AdapterSearchConfig(BaseModel):
    """One optional simulation adapter and the search space it owns."""

    model_config = ConfigDict(extra="forbid")

    search_space: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    """One evaluated configuration and its replay performance."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]  # backend assignment plus namespaced adapter selections
    used_gpus: int
    score: float  # objective score, normalized so higher is better (pareto: the first objective's value)
    metrics: dict[str, float | None]  # replay performance: throughput, ttft, itl, e2e, goodput
    # Per-objective raw values (natural units/direction) under a pareto goal, keyed by
    # OptimizationTarget value (e.g. {"throughput_per_gpu": .., "throughput_per_user": ..});
    # None for a single-objective sweep. Drives Pareto dominance in score.pareto_front.
    objectives: dict[str, float] | None = None
    # Optional public concrete prediction configuration attached by the unified
    # CLI. Legacy Sweeper callers keep the historical internal ``config`` only.
    prediction_config: dict[str, Any] | None = None


class SmartSearchConfig(BaseModel):
    """Top-level config integrating every search input; one YAML maps to this."""

    model_config = ConfigDict(extra="forbid")

    search_space: SearchSpace
    adapters: dict[str, AdapterSearchConfig] = Field(default_factory=dict)
    workload: Workload
    goal: OptimizationGoal = Field(default_factory=OptimizationGoal)
    sweep: SweepConfig = Field(default_factory=SweepConfig)

    @model_validator(mode="after")
    def _validate_epd(self) -> SmartSearchConfig:
        encoder, workload = self.search_space.encoder, self.workload
        if (encoder is None) != (workload.images is None):
            raise ValueError("EPD requires both search_space.encoder and workload.images")
        if encoder is None:
            return self
        if any(mode not in {"agg", "disagg"} for mode in self.search_space.deployment_mode):
            raise ValueError("analytical EPD supports only agg/disagg language deployments; AFD is unsupported")
        if self.adapters:
            raise ValueError("analytical EPD does not support adapters")
        if self.search_space.min_gpu_budget is not None:
            raise ValueError("EPD currently supports gpu_budget only, not min_gpu_budget")
        if encoder.backend_version is not None and len(set(self.search_space.backend)) != 1:
            raise ValueError("encoder.backend_version requires a single backend")
        targets = self.goal.resolved_pareto_objectives if self.goal.is_pareto else [self.goal.target]
        if set(targets) & _SLA_TARGETS or (self.goal.sla is not None and not self.goal.strict_sla):
            raise ValueError("analytical EPD supports aggregate strict_sla, not per-request goodput")
        workload.require_fixed_epd()
        for role in ("agg", "prefill", "decode"):
            if getattr(self.search_space, f"{role}_forward_model") != "op_level":
                raise ValueError("EPD requires op_level forward models")
            if getattr(self.search_space, f"{role}_timing_model") is not None:
                raise ValueError("EPD does not support custom language timing")
            if getattr(self.search_space, f"{role}_startup_time") not in (None, 0.0):
                raise ValueError("analytical EPD requires static worker pools")
        if self.search_space.startup_time not in (None, 0.0):
            raise ValueError("analytical EPD requires static worker pools")
        return self

    @model_validator(mode="before")
    @classmethod
    def _default_pareto_kv_load_ratio(cls, data: Any) -> Any:
        """A synthetic Pareto workload with no explicit load searches KV load in [0, 1]."""
        if not isinstance(data, dict):
            return data
        goal = data.get("goal")
        if isinstance(goal, OptimizationGoal):
            is_pareto = goal.is_pareto
        elif isinstance(goal, dict):
            is_pareto = goal.get("target", OptimizationTarget.THROUGHPUT) in {
                OptimizationTarget.PARETO,
                OptimizationTarget.PARETO.value,
            }
        else:
            is_pareto = False
        workload = data.get("workload")
        if not is_pareto or not isinstance(workload, dict) or workload.get("trace_path") is not None:
            return data
        if any(workload.get(name) is not None for name in ("request_rate", "concurrency", "kv_load_ratio")):
            return data
        updated = dict(data)
        updated_workload = dict(workload)
        updated_workload["kv_load_ratio"] = [0.0, 1.0]
        updated["workload"] = updated_workload
        return updated

    @model_validator(mode="after")
    def _validate_kv_load_ratio_range(self) -> SmartSearchConfig:
        """Only a Pareto study may search a KV-load range; scalar ratios work for any goal."""
        if self.workload.kv_load_ratio_range is not None and not self.goal.is_pareto and self.sweep.max_trials is None:
            raise ValueError(
                "a ranged workload.kv_load_ratio is only allowed when goal.target is 'pareto' "
                f"(got target={self.goal.target.value}); use one scalar kv_load_ratio"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> SmartSearchConfig:
        """Load + validate one YAML file into the nested config."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
