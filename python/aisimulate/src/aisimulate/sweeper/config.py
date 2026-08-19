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
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OptimizationTarget(str, Enum):
    """What the search optimizes for.

    All members except ``pareto`` are scalar (single-objective) targets. ``pareto`` is a
    multi-objective mode: the search optimizes the Pareto tradeoff between the scalar
    targets listed in :attr:`OptimizationGoal.pareto_objectives` (default: throughput per
    GPU vs per-user throughput — the InferenceX tok/s/gpu vs tok/s/user frontier).
    """

    THROUGHPUT = "throughput"  # maximize replay throughput
    THROUGHPUT_PER_GPU = (
        "throughput_per_gpu"  # maximize throughput / avg GPU (tok/s/gpu)
    )
    THROUGHPUT_PER_USER = (
        "throughput_per_user"  # maximize mean per-user output throughput (tok/s/user)
    )
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
        return self is not OptimizationTarget.E2E_LATENCY


class SearchPolicy(str, Enum):
    """How Sweeper chooses candidates from the configured search space.

    ``rapid`` keeps the optimizer-guided, bounded search used by the refactored
    Sweeper. ``thorough`` visits every runnable point in a finite discrete search
    space in a canonical order.
    """

    RAPID = "rapid"
    THOROUGH = "thorough"


class SLATarget(BaseModel):
    """Per-request latency bounds in ms. Set ttft_ms+itl_ms, or e2e_ms."""

    model_config = ConfigDict(extra="forbid")

    ttft_ms: float | None = Field(default=None, gt=0)
    itl_ms: float | None = Field(default=None, gt=0)
    e2e_ms: float | None = Field(default=None, gt=0)


# Goodput-based scalar targets — the only ones that need an SLA (their metric counts
# only SLA-satisfying requests). Used to gate the SLA requirement on both the scalar
# target and the per-objective list under a pareto goal.
_SLA_TARGETS = frozenset(
    {OptimizationTarget.GOODPUT, OptimizationTarget.GOODPUT_PER_GPU}
)

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
    sla: SLATarget | None = (
        None  # required for goodput / goodput_per_gpu (scalar or pareto objective)
    )
    # Only meaningful when target == pareto: the >=2 scalar objectives whose Pareto
    # front is sought. None -> the default pair (throughput_per_gpu, throughput_per_user).
    pareto_objectives: list[OptimizationTarget] | None = None

    @property
    def resolved_pareto_objectives(self) -> list[OptimizationTarget]:
        """The effective Pareto objective list: the configured one, or the default pair only
        when unset (``None``). An explicitly-supplied empty/short list is kept as-is so the
        validator's ``len < 2`` guard rejects it (rather than silently using the default).
        """
        return (
            list(_DEFAULT_PARETO_OBJECTIVES)
            if self.pareto_objectives is None
            else list(self.pareto_objectives)
        )

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
                raise ValueError(
                    "pareto_objectives cannot contain 'pareto' itself (objectives must be scalar)"
                )
            if len(set(objs)) != len(objs):
                raise ValueError(
                    f"pareto_objectives must be distinct, got {[o.value for o in objs]}"
                )
            effective = set(objs)
        else:
            effective = {self.target}
        # Any goodput-based objective (scalar target or pareto objective) needs an SLA.
        needs_sla = bool(effective & _SLA_TARGETS)
        has_sla = self.sla is not None and (
            self.sla.e2e_ms is not None
            or (self.sla.ttft_ms is not None and self.sla.itl_ms is not None)
        )
        if needs_sla and not has_sla:
            culprits = sorted(t.value for t in (effective & _SLA_TARGETS))
            raise ValueError(
                f"{culprits} require an SLA target (ttft_ms+itl_ms or e2e_ms)"
            )
        return self


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
    isl: int | None = None
    osl: int | None = None
    concurrency: int | None = None
    kv_load_ratio: float | list[float] | None = None
    request_rate: float | None = None
    num_request_ratio: float | None = (
        None  # request count multiplier for concrete concurrency or request_rate
    )
    random_range_ratio: float = 1.0
    random_seed: int = 0
    shared_prefix_ratio: float = 0.0  # cache-locality / prefix sharing
    num_prefix_groups: int = 0
    turns_per_session: int = 1  # multi-turn sessions
    inter_turn_delay_ms: float = 0.0  # think-time between turns (multi-turn synthetic)

    # dynamic trace source (mutually exclusive with the synthetic fields)
    trace_path: str | None = None
    trace_format: str = "mooncake"  # replay-ready trace schema
    arrival_speedup_ratio: float = 1.0  # scale trace inter-arrival times
    # Closed-loop replay over a *trace*: cap in-flight requests at this many (the
    # trace's timestamps are ignored; a new request starts as one finishes). For a
    # *synthetic* closed-loop workload use ``concurrency`` or ``kv_load_ratio`` instead.
    replay_concurrency: int | None = None

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
            raise ValueError(
                f"random_seed must be an unsigned 64-bit integer, got {value!r}"
            )
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

    def effective_in_flight_cap(
        self, concurrency_override: int | None = None
    ) -> int | None:
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
            raise ValueError(
                "resolved_request_count needs a concurrency_override for a kv_load_ratio workload"
            )
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
                raise ValueError(
                    f"trace workload (trace_path set) must not set synthetic fields {set_syn}"
                )
            if self.replay_concurrency is not None and self.replay_concurrency <= 0:
                raise ValueError(
                    f"replay_concurrency must be a positive integer, got {self.replay_concurrency}"
                )
            return self
        # synthetic: exactly one load mode, plus isl/osl/num_request_ratio
        loads = [
            n
            for n in ("request_rate", "concurrency", "kv_load_ratio")
            if getattr(self, n) is not None
        ]
        if len(loads) != 1:
            raise ValueError(
                "a synthetic workload needs exactly one of request_rate, concurrency, or kv_load_ratio "
                "(or set trace_path for a trace workload)"
            )
        missing = [
            n for n in ("isl", "osl", "num_request_ratio") if getattr(self, n) is None
        ]
        if missing:
            raise ValueError(f"a synthetic workload requires {missing}")
        if self.replay_concurrency is not None:
            raise ValueError(
                "replay_concurrency is for trace workloads; use 'concurrency' for synthetic closed-loop"
            )
        if self.concurrency is not None and (
            isinstance(self.concurrency, bool) or self.concurrency <= 0
        ):
            raise ValueError(
                f"concurrency must be a positive integer, got {self.concurrency!r}"
            )
        if self.kv_load_ratio is not None:
            ratios = (
                self.kv_load_ratio
                if isinstance(self.kv_load_ratio, list)
                else [self.kv_load_ratio]
            )
            if isinstance(self.kv_load_ratio, list) and len(ratios) != 2:
                raise ValueError("kv_load_ratio range must contain exactly [min, max]")
            if any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in ratios
            ):
                raise ValueError(
                    f"kv_load_ratio values must be finite and non-negative, got {self.kv_load_ratio!r}"
                )
            if isinstance(self.kv_load_ratio, list) and float(ratios[0]) >= float(
                ratios[1]
            ):
                raise ValueError(
                    f"kv_load_ratio range needs min < max, got {self.kv_load_ratio!r}"
                )
        for name in ("request_rate", "isl", "osl", "num_request_ratio"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        if (
            not math.isfinite(self.random_range_ratio)
            or self.random_range_ratio <= 0.0
            or self.random_range_ratio > 1.0
        ):
            raise ValueError(
                "random_range_ratio must be finite and in (0.0, 1.0], got "
                f"{self.random_range_ratio!r}"
            )
        for name in ("isl", "osl"):
            length = getattr(self, name)
            if length is not None and int(length * self.random_range_ratio) == 0:
                raise ValueError(
                    f"random_range_ratio={self.random_range_ratio} gives a zero-token "
                    f"lower bound for {name}={length}"
                )
        if (
            isinstance(self.random_seed, bool)
            or self.random_seed < 0
            or self.random_seed > 0xFFFF_FFFF_FFFF_FFFF
        ):
            raise ValueError(
                "random_seed must be an unsigned 64-bit integer, got "
                f"{self.random_seed!r}"
            )
        if self.random_range_ratio != 1.0 and self.turns_per_session != 1:
            raise ValueError(
                "random_range_ratio currently only supports single-turn "
                "synthetic workloads"
            )
        return self


# Allowed choices for each swept search-space dimension. A configured value must
# be a non-empty subset of these (one or more); the field defaults below use the
# full set (or a sensible subset, e.g. ``backend``). Centralized here so the
# candidate generator can reuse it. Pinned scalars and the generated
# ``parallel_configs`` are intentionally not choice-constrained.
SEARCH_CHOICES: dict[str, tuple] = {
    "deployment_mode": ("disagg", "agg", "afd", "afd+pd"),
    "backend": ("vllm", "sglang", "trtllm"),
    "afd_pipeline_model_candidates": ("optimistic", "conservative", "serial"),
    "prefill_max_num_batched_tokens": (8192, 16384, 32768),
    "prefill_max_num_seqs": (1, 2, 4, 8, 16, 32, 64, 128, 256),
    "decode_max_num_batched_tokens": (8192,),
    "decode_max_num_seqs": (256, 512, 1024),
    "agg_max_num_batched_tokens": (8192, 16384, 32768),
    "agg_max_num_seqs": (256, 512, 1024),
}

_ROLE_CANDIDATE_SUFFIXES = (
    "num_gpu_candidates",
    "tp_candidates",
    "pp_candidates",
    "dp_candidates",
    "moe_tp_candidates",
    "moe_ep_candidates",
    "cp_candidates",
    "batch_size_candidates",
    "context_tokens_candidates",
    "num_workers_candidates",
)
_ROLE_CANDIDATE_FIELDS = tuple(
    f"{role}_{suffix}"
    for role in ("agg", "prefill", "decode")
    for suffix in _ROLE_CANDIDATE_SUFFIXES
)


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
    parallel_configs: list[dict[str, Any]] = Field(
        default_factory=list
    )  # generated when empty
    # pinned
    model_name: str  # HF id or private model name
    hardware_sku: str  # e.g. "h200_sxm"
    gpu_budget: int = 32  # max GPUs per candidate
    min_gpu_budget: int | None = None
    context_length: int | None = None
    startup_time: float | None = None
    aic_nextn: int | None = None  # speculative-decode (MTP) depth, 1..5

    # Attention--FFN disaggregation. ``deployment_mode`` selects pure ``afd`` or
    # combined ``afd+pd``; the latter adds one opposite-phase companion from the
    # same per-role legal domain used by aggregate/P/D search.
    afd_pinned_topologies: list[dict[str, Any]] = Field(default_factory=list)
    afd_tp_a_candidates: list[int] | None = None
    # Legacy AFD derives this per topology from A/F partition memory. Generic
    # Sweeper cannot reproduce that adapter-owned calculation safely, so a
    # searched AFD domain must declare memory-qualified batches explicitly.
    afd_batch_size_candidates: list[int] | None = None
    afd_f_moe_ep_size_candidates: list[int | str] | None = None
    afd_microbatch_candidates: list[int] = [2, 3, 4]
    afd_pipeline_model_candidates: list[str] = ["optimistic", "conservative"]
    afd_phase: str = "decode"
    afd_comm_overhead_factor: float = Field(default=1.0, gt=0)
    afd_boundary_on_attn: bool = True
    afd_max_af_ratio: float = Field(default=4.0, gt=0)
    afd_max_candidates: int = Field(default=10_000, ge=1)
    afd_candidate_overflow: str = "error"

    # Explicit per-role topology domains. ``None`` selects model/hardware/backend
    # defaults; a non-empty list is authoritative. Batch/context candidates are
    # replay scheduler limits and therefore remain distinct from topology legality.
    agg_num_gpu_candidates: list[int] | None = None
    agg_tp_candidates: list[int] | None = None
    agg_pp_candidates: list[int] | None = None
    agg_dp_candidates: list[int] | None = None
    agg_moe_tp_candidates: list[int] | None = None
    agg_moe_ep_candidates: list[int] | None = None
    agg_cp_candidates: list[int] | None = None
    agg_batch_size_candidates: list[int] | None = None
    agg_context_tokens_candidates: list[int] | None = None
    agg_num_workers_candidates: list[int] | None = None

    prefill_num_gpu_candidates: list[int] | None = None
    prefill_tp_candidates: list[int] | None = None
    prefill_pp_candidates: list[int] | None = None
    prefill_dp_candidates: list[int] | None = None
    prefill_moe_tp_candidates: list[int] | None = None
    prefill_moe_ep_candidates: list[int] | None = None
    prefill_cp_candidates: list[int] | None = None
    prefill_batch_size_candidates: list[int] | None = None
    prefill_context_tokens_candidates: list[int] | None = None
    prefill_num_workers_candidates: list[int] | None = None

    decode_num_gpu_candidates: list[int] | None = None
    decode_tp_candidates: list[int] | None = None
    decode_pp_candidates: list[int] | None = None
    decode_dp_candidates: list[int] | None = None
    decode_moe_tp_candidates: list[int] | None = None
    decode_moe_ep_candidates: list[int] | None = None
    decode_cp_candidates: list[int] | None = None
    decode_batch_size_candidates: list[int] | None = None
    decode_context_tokens_candidates: list[int] | None = None
    decode_num_workers_candidates: list[int] | None = None

    # Disaggregated P/D cell controls. The allowed-size ladder and ceiling are
    # applied before either search policy sees the domain.
    num_gpu_per_replica: list[int] = Field(
        default_factory=lambda: [1, 2, 4, 8, *range(16, 129, 8)]
    )
    max_gpu_per_replica: int = Field(default=128, ge=1)
    max_prefill_workers: int = Field(default=32, ge=1)
    max_decode_workers: int = Field(default=32, ge=1)

    # prefill engine (disagg branch): scheduler batching capacity
    prefill_max_num_batched_tokens: list[int] = [8192, 16384, 32768]
    prefill_max_num_seqs: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # pinned
    prefill_block_size: int = 64
    prefill_gpu_memory_utilization: float = 0.9
    prefill_enable_prefix_caching: bool = True

    # decode engine (disagg branch): scheduler batching capacity
    decode_max_num_batched_tokens: list[int] = [8192]
    decode_max_num_seqs: list[int] = [256, 512, 1024]
    # pinned
    decode_block_size: int = 64
    decode_gpu_memory_utilization: float = 0.9
    decode_enable_prefix_caching: bool = False  # forced off for decode workers

    # agg engine (agg branch): scheduler batching capacity
    agg_max_num_batched_tokens: list[int] = [8192, 16384, 32768]
    agg_max_num_seqs: list[int] = [256, 512, 1024]
    # pinned
    agg_block_size: int = 64
    agg_gpu_memory_utilization: float = 0.9
    agg_enable_prefix_caching: bool = True

    @field_validator(*_ROLE_CANDIDATE_FIELDS, "num_gpu_per_replica", mode="before")
    @classmethod
    def _validate_positive_candidate_lists(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list) or not value:
            raise ValueError("candidate lists must be non-empty lists")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
            raise ValueError(f"candidate lists need positive integers, got {value!r}")
        if len(set(value)) != len(value):
            raise ValueError(f"candidate lists must not contain duplicates, got {value!r}")
        return value

    @field_validator(
        "afd_tp_a_candidates",
        "afd_batch_size_candidates",
        "afd_microbatch_candidates",
        mode="before",
    )
    @classmethod
    def _validate_afd_positive_candidate_lists(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list) or not value:
            raise ValueError("AFD candidate lists must be non-empty lists")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
            raise ValueError(f"AFD candidate lists need positive integers, got {value!r}")
        if len(set(value)) != len(value):
            raise ValueError(f"AFD candidate lists must not contain duplicates, got {value!r}")
        return value

    @field_validator("afd_f_moe_ep_size_candidates", mode="before")
    @classmethod
    def _validate_afd_ep_candidates(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list) or not value:
            raise ValueError("afd_f_moe_ep_size_candidates must be a non-empty list")
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
        if self.afd_phase not in {"prefill", "decode", "both"}:
            raise ValueError("afd_phase must be 'prefill', 'decode', or 'both'")
        if self.afd_candidate_overflow not in {"error", "truncate"}:
            raise ValueError("afd_candidate_overflow must be 'error' or 'truncate'")
        if "afd+pd" in self.deployment_mode and self.afd_phase == "both":
            raise ValueError("deployment_mode='afd+pd' requires afd_phase prefill or decode")
        has_afd_mode = any(mode in {"afd", "afd+pd"} for mode in self.deployment_mode)
        if has_afd_mode and not self.afd_pinned_topologies and self.afd_batch_size_candidates is None:
            raise ValueError(
                "searched AFD requires explicit, memory-qualified "
                "afd_batch_size_candidates; legacy's topology-specific "
                "batch derivation must not be replaced by an implicit fixed batch"
            )
        if self.afd_pinned_topologies:
            modes = list(dict.fromkeys(self.deployment_mode))
            if len(modes) != 1 or modes[0] not in {"afd", "afd+pd"}:
                raise ValueError("afd_pinned_topologies requires exactly one AFD deployment_mode")
            required = {"n_a_nodes", "n_f_nodes", "tp_a", "a_batch_size"}
            for index, topology in enumerate(self.afd_pinned_topologies):
                if not isinstance(topology, dict):
                    raise ValueError(f"afd_pinned_topologies[{index}] must be a mapping")
                missing = sorted(required - topology.keys())
                if missing:
                    raise ValueError(f"afd_pinned_topologies[{index}] is missing {missing}")
        return self

    @model_validator(mode="after")
    def _validate_decode_context_parallelism(self) -> SearchSpace:
        """Decode is token-serial and cannot use prefill context parallelism."""

        if self.decode_cp_candidates is not None and any(
            value != 1 for value in self.decode_cp_candidates
        ):
            raise ValueError(
                "decode_cp_candidates must contain only 1; context parallelism applies "
                "to aggregate/prefill execution"
            )
        return self

    @model_validator(mode="after")
    def _validate_search_choices(self) -> SearchSpace:
        """Every backend dimension is a non-empty subset of its allowed choices."""
        for field_name, allowed in SEARCH_CHOICES.items():
            values = getattr(self, field_name)
            if not values:
                raise ValueError(
                    f"{field_name} must list at least one choice; allowed: {list(allowed)}"
                )
            for v in values:
                if v not in allowed:
                    raise ValueError(
                        f"{field_name} has invalid choice {v!r}; allowed: {list(allowed)}"
                    )
        return self

    @model_validator(mode="after")
    def _validate_gpu_budget(self) -> SearchSpace:
        """A minimum GPU budget must be positive and within the maximum budget."""
        if self.min_gpu_budget is not None and not (
            0 < self.min_gpu_budget <= self.gpu_budget
        ):
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
        if not self.parallel_configs:
            return self

        if any(mode in {"afd", "afd+pd"} for mode in self.deployment_mode):
            raise ValueError("parallel_configs only pins agg/disagg shapes; use afd_pinned_topologies for AFD")

        def validate_shape_dict(value: Any, label: str) -> None:
            if not isinstance(value, dict):
                raise ValueError(f"{label} parallel_configs shape must be a dict")
            if "tp" not in value:
                raise ValueError(f"{label} parallel_configs shape needs a 'tp' field")

        if len(self.deployment_mode) != 1:
            raise ValueError(
                "pinning parallel_configs requires deployment_mode to list exactly one mode "
                f"(got {self.deployment_mode}); pin the mode too"
            )
        mode = self.deployment_mode[0]
        for entry in self.parallel_configs:
            if not isinstance(entry, dict):
                raise ValueError("each parallel_configs entry must be a dict")
            if mode == "agg":
                validate_shape_dict(entry, "an agg")
            else:
                if "prefill" not in entry or "decode" not in entry:
                    raise ValueError(
                        "a disagg parallel_configs entry needs 'prefill' and 'decode' sub-dicts"
                    )
                validate_shape_dict(entry["prefill"], "a disagg prefill")
                validate_shape_dict(entry["decode"], "a disagg decode")
        return self


class SweepConfig(BaseModel):
    """Sweep run-control."""

    model_config = ConfigDict(extra="forbid")

    policy: SearchPolicy = SearchPolicy.RAPID
    # Seeds the default rapid Vizier designer. Thorough enumeration has a canonical
    # order, so its seed is recorded for provenance but deliberately does not reorder
    # candidates.
    seed: int = Field(default=0, ge=0, le=(2**32 - 1))
    max_rounds: int = Field(default=20, ge=1)  # total Vizier/replay barrier rounds
    parallel_evals: int = Field(
        default=16, ge=1
    )  # replay worker fan-out and default candidates per round
    # Successful unique replay configs per round; duplicate projections are told from
    # cache and replaced. Defaults to parallel_evals.
    candidates_per_round: int | None = Field(default=None, ge=1)
    # Per-candidate wall-clock cap for the replay. A candidate whose replay exceeds this is
    # killed and reported as infeasible ("exceed runtime") so the optimizer avoids that region
    # instead of hanging the sweep (e.g. an over-subscribed config that churns). Only enforced
    # on the worker-pool path (parallel_evals > 1); None disables the cap.
    max_eval_seconds: float | None = Field(default=600.0, gt=0)

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed_type(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"seed must be an unsigned 32-bit integer, got {value!r}")
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
    metrics: dict[str, float]  # replay performance: throughput, ttft, itl, e2e, goodput
    # Per-objective raw values (natural units/direction) under a pareto goal, keyed by
    # OptimizationTarget value (e.g. {"throughput_per_gpu": .., "throughput_per_user": ..});
    # None for a single-objective sweep. Drives Pareto dominance in score.pareto_front.
    objectives: dict[str, float] | None = None


class SmartSearchConfig(BaseModel):
    """Top-level config integrating every search input; one YAML maps to this."""

    model_config = ConfigDict(extra="forbid")

    search_space: SearchSpace
    adapters: dict[str, AdapterSearchConfig] = Field(default_factory=dict)
    workload: Workload
    goal: OptimizationGoal = Field(default_factory=OptimizationGoal)
    sweep: SweepConfig = Field(default_factory=SweepConfig)

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_flat_fields(cls, data: Any) -> Any:
        """Fail clearly instead of silently accepting the pre-adapter Sweeper schema."""
        if not isinstance(data, dict):
            return data
        search_space = data.get("search_space")
        if not isinstance(search_space, dict):
            return data

        kvbm_fields = {
            "num_g2_blocks",
            "kv_bytes_per_token",
            "bandwidth_g1_to_g2_gbps",
            "bandwidth_g2_to_g1_gbps",
            "offload_batch_size",
            "host_cache_hit_weight",
            "disk_cache_hit_weight",
        }
        planner_fields = {
            "min_endpoint",
            "prefill_min_endpoint",
            "decode_min_endpoint",
            "planner_scaling_policy",
            "planner_fpm_sampling",
            "planner_load_sensitivity",
            "load_predictor_candidates",
        }
        router_fields = {
            "router_mode",
            "overlap_score_credit",
            "prefill_load_scale",
            "router_temperature",
            "active_decode_blocks_threshold",
            "active_decode_tokens_threshold",
            "active_prefill_tokens_threshold",
            "active_prefill_tokens_threshold_frac",
            "no_admission_control",
        }

        present_kvbm = sorted(kvbm_fields.intersection(search_space))
        if present_kvbm:
            raise ValueError(
                "KVBM sweep fields are not supported by the AISimulate engine "
                f"and replay path; remove {present_kvbm}"
            )
        present_planner = sorted(planner_fields.intersection(search_space))
        if present_planner:
            raise ValueError(
                "Planner search fields now belong to an adapter; move "
                f"{present_planner} to adapters['dynamo.planner'].search_space"
            )
        present_router = sorted(router_fields.intersection(search_space))
        if present_router:
            raise ValueError(
                "Router search fields now belong to an adapter; move "
                f"{present_router} to adapters['dynamo.router'].search_space"
            )
        return data

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
        if (
            not is_pareto
            or not isinstance(workload, dict)
            or workload.get("trace_path") is not None
        ):
            return data
        if any(
            workload.get(name) is not None
            for name in ("request_rate", "concurrency", "kv_load_ratio")
        ):
            return data
        updated = dict(data)
        updated_workload = dict(workload)
        updated_workload["kv_load_ratio"] = [0.0, 1.0]
        updated["workload"] = updated_workload
        return updated

    @model_validator(mode="after")
    def _validate_kv_load_ratio_range(self) -> SmartSearchConfig:
        """Only a Pareto study may search a KV-load range; scalar ratios work for any goal."""
        if self.workload.kv_load_ratio_range is not None and not self.goal.is_pareto:
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
