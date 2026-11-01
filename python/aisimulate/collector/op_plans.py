# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Op plans: one case list per op, shared by every CUDA backend.

An op plan expands the op's cases once from ``cases/`` YAML, the same for
every backend. :func:`plan` then leaves out what ``collector/skips.py`` skips
for the backend and SM. Every other case runs; if it fails, the failure is
recorded.

Where the plan picks the kernel or runner a backend measures a case on (the
one serving selects, Rule 2), the op declares it in ``dispatch``; a case that
serving runs on several kernels (one per model that plans it) is measured on
each. A ``collector/skips.py`` entry may name a ``dispatch`` to leave a case
out on that kernel only. A backend whose run function takes more arguments
than the case declares ``rows``: the run arguments for one case and its
dispatch. Rows decide how a case runs, never whether it runs, so every case
gives at least one row.

Planning never imports a framework, so ``--plan-only`` runs on any CPU.

Each backend's getter is one line, ``return plan_cases("<op>", "<backend>",
get_sm_version())``. To move an op onto a plan, add its case type, its expand
function and an :class:`OpPlan` entry in ``OP_PLANS``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import NamedTuple

from collector import skips
from collector.case_generator import (
    BASE_OP_CASES_DIR,
    AttentionOwner,
    MoeCommonTestCase,
    get_attention_encoder_head_configs,
    get_attention_shapes,
    get_common_gdn_test_cases,
    get_common_kda_test_cases,
    get_common_mamba2_test_cases,
    get_common_mhc_test_cases,
    get_common_moe_test_cases,
    get_compute_scale_case_specs,
    get_context_mla_case_specs,
    get_gemm_case_specs,
    get_gemm_type_specs,
    get_generation_mla_case_specs,
    get_mla_bmm_case_specs,
    get_moe_quantization_module_config,
    get_sglang_moe_backend,
    get_shared_base_op_case_specs,
    moe_model_layer,
    moe_planned_modes,
    without_models,
)

# Plan logs go to the collector log, never stdout: --plan-only prints JSON there.
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class OpPlan:
    """How one op's shared cases are expanded."""

    case_type: type
    # Every case, in run order, before whole regions are removed.
    expand: Callable[[], list]
    # The case field that holds the dtype or quant mode.
    dtype_field: str
    # The case fields whose values share one module build in a worker.
    group_fields: tuple[str, ...]
    # backend -> the kernel or runner the backend measures a case on, where the
    # plan picks it; a tuple when serving runs the case on several.
    dispatch: Mapping[str, Callable[[object, int | None], str | tuple[str, ...]]] = MappingProxyType({})
    # backend -> the run arguments for one case and its dispatch (None without
    # one), for a run function that takes more than the case.
    rows: Mapping[str, Callable[[object, int | None, str | None], list]] = MappingProxyType({})


@dataclasses.dataclass(frozen=True)
class RemovedRegion:
    """What a plan leaves out (``dtype=fp8_block``, ``model=<architecture>``), how many cases, and why.

    ``reason`` is the skips.py entry that left the cases out.
    """

    region: str
    cases: int
    reason: str


@dataclasses.dataclass(frozen=True)
class Plan:
    """The cases one backend runs for an op on one SM."""

    cases: list
    removed: list[RemovedRegion]
    # The dispatches of each case, for a backend whose plan picks them; else empty.
    dispatches: list[tuple[str, ...]] = dataclasses.field(default_factory=list)
    # Dispatches a skips.py entry left out of a case that still runs on another.
    removed_runs: list[RemovedRegion] = dataclasses.field(default_factory=list)
    # The backend's rows function and where it runs, applied on demand: rows may
    # build the backend's framework objects (a torch dtype), cases never do.
    to_rows: Callable[[object, int | None, str | None], list] | None = None
    sm_version: int | None = None
    where: str = ""
    # The model rows the plan left out; rows read the model profiles without them.
    skipped_models: frozenset[str] = frozenset()

    @property
    def by_dispatch(self) -> dict[str, int]:
        """How many cases run on each dispatch."""
        return dict(Counter(runner for runners in self.dispatches for runner in runners))

    @property
    def rows(self) -> list:
        """What the getter queues: the cases, or the backend's rows for them."""
        if self.to_rows is None:
            return self.cases
        runs_on = self.dispatches or [(None,)] * len(self.cases)
        rows = []
        with without_models(self.skipped_models):
            for case, runners in zip(self.cases, runs_on, strict=True):
                for runner in runners:
                    case_rows = self.to_rows(case, self.sm_version, runner)
                    if not case_rows:
                        raise ValueError(f"{self.where} rows dropped case {case}; rows may only say how a case runs")
                    rows.extend(case_rows)
        return rows


class _Run(NamedTuple):
    """One case on one of its dispatches; a skips.py entry sees the case's fields and ``dispatch``."""

    case: object
    dispatch: str

    def __getattr__(self, name: str):
        return getattr(self.case, name)


def plan(op: str, backend: str, sm_version: int | None) -> Plan:
    """Expand an op's shared cases and leave out what ``skips.SKIPS`` skips for this backend on this SM.

    A ``model`` entry drops the model's rows before cases are merged, so a case
    another model also needs stays; the plan reports how many cases each such
    entry removes on its own.
    """
    spec = OP_PLANS[op]
    model_entries = skips.model_entries(backend, op, sm_version)
    skipped_models = skips.models(backend, op, sm_version)
    with without_models(skipped_models):
        expanded = spec.expand()
    removed_models = []
    for entry, architectures in model_entries:
        with without_models(skipped_models - architectures):
            count = len(spec.expand()) - len(expanded)
        if count:
            removed_models.append(RemovedRegion(f"model={','.join(sorted(architectures))}", count, repr(entry)))

    def dtypes_of(case) -> tuple[str]:
        return (getattr(case, spec.dtype_field),)

    kept, skipped = skips.split(expanded, backend, op, sm_version, dtypes_of, dispatch_entries=False)
    removed = Counter((region, repr(entry)) for _case, entry, region in skipped)
    removed_runs = Counter()
    dispatches = []
    dispatch = spec.dispatch.get(backend)
    if dispatch is not None:
        # Entries that name a dispatch leave a case out on that kernel only; a
        # case left on no kernel counts as left out.
        with without_models(skipped_models):
            runs = [_Run(case, runner) for case in kept for runner in _runners(dispatch(case, sm_version))]
        runs, skipped_runs = skips.split(
            runs, backend, op, sm_version, lambda run: dtypes_of(run.case), dispatch_entries=True
        )
        runners_of: dict = {}
        for run in runs:
            runners_of.setdefault(run.case, []).append(run.dispatch)
        left_out = set()
        for run, entry, region in skipped_runs:
            if run.case in runners_of:
                removed_runs[(region, repr(entry))] += 1
            elif run.case not in left_out:
                left_out.add(run.case)
                removed[(region, repr(entry))] += 1
        kept = [case for case in kept if case in runners_of]
        dispatches = [tuple(runners_of[case]) for case in kept]
    return Plan(
        cases=kept,
        removed=[
            *(RemovedRegion(region, count, reason) for (region, reason), count in removed.items()),
            *removed_models,
        ],
        dispatches=dispatches,
        removed_runs=[RemovedRegion(region, count, reason) for (region, reason), count in removed_runs.items()],
        to_rows=spec.rows.get(backend),
        sm_version=sm_version,
        where=f"{backend} {op}",
        skipped_models=skipped_models,
    )


def _runners(dispatched: str | tuple[str, ...]) -> tuple[str, ...]:
    return dispatched if isinstance(dispatched, tuple) else (dispatched,)


def plan_cases(op: str, backend: str, sm_version: int | None) -> list:
    """Return what one backend's getter queues for an op on one SM (see :func:`plan`)."""
    return plan(op, backend, sm_version).rows


def summarize(op: str, op_plan: Plan, case_id: Callable[[object], str]) -> dict:
    """Count a plan per dtype and list the regions it leaves out.

    ``case_id`` maps a queued row to the ID a collection run records, so
    ``plan_hash`` equals the case_plan_hash of a run that attempts every case.
    """
    from collector.provenance import case_plan_hash

    spec = OP_PLANS[op]
    summary = {
        "cases": len(op_plan.cases),
        "groups": len({tuple(getattr(case, field) for field in spec.group_fields) for case in op_plan.cases}),
        "by_dtype": dict(Counter(getattr(case, spec.dtype_field) for case in op_plan.cases)),
        "removed": [dataclasses.asdict(region) for region in op_plan.removed],
    }
    if op_plan.removed_runs:
        summary["removed_runs"] = [dataclasses.asdict(region) for region in op_plan.removed_runs]
    try:
        rows = op_plan.rows
    except ImportError as exc:
        # This backend's rows hold framework objects; without the framework the
        # cases are still counted, but the run IDs (and plan_hash) are not known.
        summary["plan_hash"] = None
        summary["rows_need"] = str(exc)
    else:
        summary["plan_hash"] = case_plan_hash([case_id(row) for row in rows])
        if len(rows) != len(op_plan.cases):
            summary["rows"] = len(rows)
    if op_plan.by_dispatch:
        summary["by_dispatch"] = op_plan.by_dispatch
    return summary


def _list_form(case: tuple) -> str:
    """A case's string: the list form case IDs, checkpoints and ``--case-filter`` match on (tuples print as lists)."""
    return str([list(value) if isinstance(value, tuple) else value for value in case])


def _tuple_or_none(values) -> tuple | None:
    return None if values is None else tuple(values)


# --- GEMM ---------------------------------------------------------------------


class GemmCase(NamedTuple):
    """One GEMM case.

    ``str()`` keeps the list form (``['fp8', 64, 4096, 7168]``) that case IDs,
    checkpoints, ``--cases-from`` replay and ``--case-filter`` match on.
    """

    gemm_type: str
    m: int
    n: int
    k: int

    __str__ = __repr__ = _list_form


def _expand_gemm() -> list[GemmCase]:
    """The ``gemm_types`` axis of gemm.yaml times the shared shapes (base grid plus model widths).

    Cases come one weight shape (n, k) at a time, largest first, with m
    descending, so a worker that built the module for (n, k) reuses it.
    """
    gemm_types = get_gemm_type_specs()
    if not gemm_types:
        raise RuntimeError(f"{BASE_OP_CASES_DIR / 'gemm.yaml'} must declare gemm_types")
    m_by_weight: dict[tuple[int, int], list[int]] = defaultdict(list)
    for spec in get_gemm_case_specs():
        m_by_weight[(spec.n, spec.k)].append(spec.x)
    return [
        GemmCase(gemm_type, m, n, k)
        for n, k in sorted(m_by_weight, reverse=True)
        for gemm_type in gemm_types
        for m in sorted(m_by_weight[(n, k)], reverse=True)
    ]


# --- MoE ----------------------------------------------------------------------


class MoeCase(NamedTuple):
    """One MoE case: a quant mode and one routed-expert layer, timed over every token count.

    ``str()`` keeps the list form that case IDs use (``num_tokens`` prints as a
    list); ``num_tokens`` is a tuple so that a case is hashable.
    """

    moe_type: str
    num_tokens: tuple[int, ...]
    hidden_size: int
    inter_size: int
    topk: int
    num_experts: int
    tp: int
    ep: int
    model_name: str
    distribution: str
    power_law_alpha: float | None

    __str__ = __repr__ = _list_form

    @property
    def parallel(self) -> str:
        """How the routed experts are split: ``tp``, ``ep``, or ``tp_ep`` when both exceed 1."""
        if self.tp > 1 and self.ep > 1:
            return "tp_ep"
        return "ep" if self.ep > 1 else "tp"


# SMs on which a row's SGLang runner pins are compared when two rows share a consumer key.
_MOE_SMS = (80, 86, 89, 90, 100, 103, 120)
_FP4_EXPERT_MODES = frozenset({"w4a16_mxfp4", "w4a8_mxfp4_mxfp8"})


def _is_fp4_experts(layer: MoeCommonTestCase, moe_type: str) -> bool:
    return layer.architecture == "DeepseekV4ForCausalLM" and moe_type in _FP4_EXPERT_MODES


def _moe_layer_identity(layer: MoeCommonTestCase, moe_type: str) -> tuple:
    """What a MoE case times besides its consumer key: routing and activation, runner pins, quant config."""
    routing = tuple(
        (field.name, getattr(layer, field.name))
        for field in dataclasses.fields(layer)
        if field.name.startswith("sglang_moe_") and field.name != "sglang_moe_backends"
    )
    runners = []
    for sm in _MOE_SMS:
        try:
            runners.append(get_sglang_moe_backend(layer, moe_type, sm))
        except ValueError:
            runners.append(None)
    module_configs = tuple(
        json.dumps(get_moe_quantization_module_config(backend, moe_type, model_name=layer.model_name), sort_keys=True)
        for backend in ("sglang", "trtllm", "vllm")
    )
    return routing, tuple(runners), module_configs, _is_fp4_experts(layer, moe_type)


def _expand_moe() -> list[MoeCase]:
    """Each model row's checkpoint quant modes times the shared TP, EP and distribution grid.

    A row plans its ``quantization_modes``, or every mode of the shared axis
    when it declares none (:func:`case_generator.moe_planned_modes`). Rows
    that time the same layer share one case, owned by the first row. Two rows
    that would time different layers under one consumer key raise: the
    moe_perf consumer cannot tell them apart.
    """
    modes: dict[tuple, list[str]] = {}
    identities: dict[tuple, tuple] = {}
    owners: dict[MoeCase, tuple[tuple, str]] = {}
    cases = []
    for layer in get_common_moe_test_cases():
        row = (layer.model_name, layer.hidden_size, layer.inter_size, layer.topk, layer.num_experts)
        if row not in modes:
            modes[row] = moe_planned_modes(layer.model_name, layer.quantization_modes)
        for moe_type in modes[row]:
            case = MoeCase(
                moe_type,
                tuple(layer.num_tokens_list),
                layer.hidden_size,
                layer.inter_size,
                layer.topk,
                layer.num_experts,
                layer.tp,
                layer.ep,
                layer.model_name,
                layer.token_expert_distribution,
                layer.power_law_alpha,
            )
            if (row, moe_type) not in identities:
                identities[(row, moe_type)] = _moe_layer_identity(layer, moe_type)
            identity = identities[(row, moe_type)]
            consumer_key = case._replace(model_name="")
            owner = owners.get(consumer_key)
            if owner is None:
                owners[consumer_key] = (identity, layer.model_name)
                cases.append(case)
            elif owner[0] != identity:
                raise ValueError(
                    f"MoE rows {owner[1]!r} and {layer.model_name!r} time different layers under one "
                    f"consumer key {consumer_key}; moe_perf cannot represent both"
                )
    empty = sorted({row[0] for row, row_modes in modes.items() if not row_modes})
    if empty:
        logger.info("moe: %d model row(s) plan no case (quantization_modes: []): %s", len(empty), ", ".join(empty))
    return cases


def _moe_layer_of(case: MoeCase) -> MoeCommonTestCase:
    return moe_model_layer(case.model_name, case.hidden_size, case.inter_size, case.topk, case.num_experts)


def _sglang_moe_runner(case: MoeCase, sm_version: int | None) -> str:
    """The MoE runner SGLang serving picks for this case's checkpoint and quant mode on this SM.

    Pinned in cases/base_ops/moe.yaml moe_sglang.backends and a model row's
    sglang_moe_backends.
    """
    return get_sglang_moe_backend(_moe_layer_of(case), case.moe_type, sm_version)


def _sglang_moe_rows(case: MoeCase, sm_version: int | None, runner: str | None) -> list[list]:
    """SGLang's run_moe_torch also takes the row's routing and activation and the runner."""
    layer = _moe_layer_of(case)
    return [
        [
            case.moe_type,
            list(case.num_tokens),
            *case[2:11],
            layer.sglang_moe_swiglu_limit,
            runner,
            layer.sglang_moe_activation,
            layer.sglang_moe_is_gated,
            layer.sglang_moe_has_bias,
            layer.sglang_moe_gemm1_alpha,
            layer.sglang_moe_gemm1_clamp_limit,
            layer.sglang_moe_scoring_func,
            layer.sglang_moe_routing_method_type,
            layer.sglang_moe_routed_scaling_factor,
            layer.sglang_moe_renormalize,
            layer.sglang_moe_has_correction_bias,
            layer.sglang_moe_num_expert_group,
            layer.sglang_moe_topk_group,
            layer.sglang_moe_apply_router_weight_on_input,
            _is_fp4_experts(layer, case.moe_type),
        ]
    ]


def _trtllm_moe_rows(case: MoeCase, sm_version: int | None, runner: str | None) -> list[list]:
    """TRT-LLM's run_moe_torch takes min_latency_mode after ep.

    An nvfp4 case on SM100/103 with at most 256 experts also times the
    min-latency (trtllm-gen) kernel, a second row of moe_perf
    (kernel_source moe_torch_flow_min_latency); its DeepSeek router supports
    at most 256 experts. That row is a second measurement, not just how the
    case runs; it becomes a declared variant when the Record counts rows per
    consumer key.
    """
    min_latency_modes = [False]
    if case.moe_type == "nvfp4" and sm_version in (100, 103) and case.num_experts <= 256:
        min_latency_modes.append(True)
    head = [case.moe_type, list(case.num_tokens), *case[2:8]]
    return [[*head, min_latency, *case[8:11]] for min_latency in min_latency_modes]


# --- compute_scale ------------------------------------------------------------


class ComputeScaleCase(NamedTuple):
    """One dynamic FP8 scale computation over an [m, k] activation."""

    m: int
    k: int

    __str__ = __repr__ = _list_form

    @property
    def dtype(self) -> str:
        """The op quantizes activations to fp8."""
        return "fp8"


def _expand_compute_scale() -> list[ComputeScaleCase]:
    """The compute_scale sweep of compute_scale.yaml: token counts times input widths, largest K last."""
    return [ComputeScaleCase(spec.m, spec.k) for spec in get_compute_scale_case_specs()]


# --- encoder attention --------------------------------------------------------


class EncoderAttentionCase(NamedTuple):
    """One non-causal (encoder) bf16 MHA case."""

    batch_size: int
    seq_len: int
    num_heads: int
    head_dim: int

    __str__ = __repr__ = _list_form

    @property
    def dtype(self) -> str:
        return "bfloat16"


# Per-case budgets every backend shares: the workload's token budget, and the
# q/k/v/o activations (4 tensors of bf16) stay inside 32-bit indexing.
_ENCODER_ATTENTION_MAX_TOKENS = 131072
_ENCODER_ATTENTION_MAX_ACTIVATION_BYTES = 2**31


def _expand_encoder_attention() -> list[EncoderAttentionCase]:
    """Each encoder head profile times the batch and sequence sweep, within the per-case budgets."""
    cases = []
    over_budget = 0
    for shape_sweep in get_shared_base_op_case_specs("encoder_attention"):
        batch_sizes = sorted(int(value) for value in shape_sweep["batch_sizes"])
        sequence_lengths = sorted(int(value) for value in shape_sweep["sequence_lengths"])
        for head_config in get_attention_encoder_head_configs(shape_sweep):
            n, head_dim = head_config.num_heads, head_config.head_dim
            for s in sequence_lengths:
                for b in batch_sizes:
                    tokens = b * s
                    if (
                        tokens > _ENCODER_ATTENTION_MAX_TOKENS
                        or 4 * tokens * n * head_dim * 2 >= _ENCODER_ATTENTION_MAX_ACTIVATION_BYTES
                    ):
                        over_budget += 1
                        continue
                    cases.append(EncoderAttentionCase(b, s, n, head_dim))
    logger.info(
        "encoder_attention: left out %d of %d cases over the per-case budgets", over_budget, len(cases) + over_budget
    )
    return cases


# --- linear attention: GDN, KDA, Mamba2 ---------------------------------------


class GdnCase(NamedTuple):
    """One Gated DeltaNet case: a TP-local head geometry and its batch (and, for context, sequence) sweep."""

    phase: str
    d_model: int
    d_conv: int
    num_k_heads: int
    head_k_dim: int
    num_v_heads: int
    head_v_dim: int
    batch_sizes: tuple[int, ...] | None
    seq_lens: tuple[int, ...] | None
    model_name: str
    # Always float32: the persisted GDN key has no state-dtype dimension.
    mamba_ssm_dtype: str

    __str__ = __repr__ = _list_form

    @property
    def dtype(self) -> str:
        return "bfloat16"


def _expand_gdn() -> list[GdnCase]:
    return [
        GdnCase(
            case.phase,
            case.d_model,
            case.d_conv,
            case.num_k_heads,
            case.head_k_dim,
            case.num_v_heads,
            case.head_v_dim,
            _tuple_or_none(case.batch_size_list),
            _tuple_or_none(case.seq_len_list) if case.phase == "context" else None,
            case.model_name,
            case.mamba_ssm_dtype,
        )
        for case in get_common_gdn_test_cases()
    ]


def _gdn_rows_without_state_dtype(case: GdnCase, sm_version: int | None, runner: str | None) -> list[list]:
    """TRT-LLM's and vLLM's GDN run functions take no SSM state dtype; they run the float32 state."""
    return [[list(value) if isinstance(value, tuple) else value for value in case[:10]]]


class KdaCase(NamedTuple):
    """One Kimi Delta Attention case: a TP-local head geometry and its batch and sequence (or draft) sweep."""

    phase: str
    d_model: int
    d_conv: int
    num_k_heads: int
    head_k_dim: int
    num_v_heads: int
    head_v_dim: int
    batch_sizes: tuple[int, ...] | None
    seq_lens: tuple[int, ...] | None
    model_name: str

    __str__ = __repr__ = _list_form

    @property
    def dtype(self) -> str:
        return "bfloat16"


def _expand_kda() -> list[KdaCase]:
    return [
        KdaCase(
            case.phase,
            case.d_model,
            case.d_conv,
            case.num_k_heads,
            case.head_k_dim,
            case.num_v_heads,
            case.head_v_dim,
            _tuple_or_none(case.batch_size_list),
            _tuple_or_none(case.seq_len_list),
            case.model_name,
        )
        for case in get_common_kda_test_cases()
    ]


class Mamba2Case(NamedTuple):
    """One Mamba2 SSM case: a head geometry and its batch (and, for context, sequence) sweep."""

    phase: str
    d_model: int
    d_state: int
    d_conv: int
    nheads: int
    head_dim: int
    n_groups: int
    chunk_size: int
    batch_sizes: tuple[int, ...] | None
    seq_lens: tuple[int, ...] | None
    model_name: str

    __str__ = __repr__ = _list_form

    @property
    def dtype(self) -> str:
        return "bfloat16"


def _expand_mamba2() -> list[Mamba2Case]:
    return [
        Mamba2Case(
            case.phase,
            case.d_model,
            case.d_state,
            case.d_conv,
            case.nheads,
            case.head_dim,
            case.n_groups,
            case.chunk_size,
            _tuple_or_none(case.batch_size_list),
            _tuple_or_none(case.seq_len_list) if case.phase == "context" else None,
            case.model_name,
        )
        for case in get_common_mamba2_test_cases()
    ]


# --- MLA generation BMMs --------------------------------------------------------


class MlaBmmCase(NamedTuple):
    """One MLA generation BMM case (the W_UK_T absorb or the W_UV projection), TP-local heads."""

    num_tokens: int
    num_heads: int
    dtype: str
    num_warmups: int
    num_runs: int

    __str__ = __repr__ = _list_form


def _mla_bmm_expand(op: str) -> Callable[[], list[MlaBmmCase]]:
    def expand() -> list[MlaBmmCase]:
        """The token, head and dtype sweep of cases/base_ops/<op>.yaml."""
        return [
            MlaBmmCase(spec.num_tokens, spec.num_heads, spec.dtype, spec.num_warmups, spec.num_runs)
            for spec in get_mla_bmm_case_specs(None, op)
        ]

    return expand


# --- MLA attention: mla_context, mla_generation ---------------------------------

# The MLA geometry both collectors' harnesses build (the DeepSeek-V3 family). A
# catalog entry with another geometry raises rather than being timed under the
# wrong key.
_MLA_GEOMETRY = {
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
}
_MLA_TP_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


class MlaCase(NamedTuple):
    """One MLA attention case: sequence length, batch, KV-cache dtype, and total heads split over TP."""

    input_len: int
    batch_size: int
    kv_cache_dtype: str
    num_heads: int
    tp_size: int
    is_context_phase: bool

    __str__ = __repr__ = _list_form


def _mla_expand(is_context: bool) -> Callable[[], list[MlaCase]]:
    def expand() -> list[MlaCase]:
        """The MLA catalog per KV dtype and TP. The perf key holds local heads, not total heads or TP,
        so of the TP splits that give the same local heads only the smallest is kept."""
        cases: dict[tuple, MlaCase] = {}
        for spec in get_context_mla_case_specs() if is_context else get_generation_mla_case_specs():
            geometry = {name: getattr(spec, name) for name in _MLA_GEOMETRY}
            if geometry != _MLA_GEOMETRY:
                raise ValueError(f"Unsupported MLA geometry for {spec.model_name}: {geometry}")
            for kv_cache_dtype in ("bfloat16", "fp8"):
                for tp_size in _MLA_TP_SIZES:
                    if spec.num_heads % tp_size:
                        continue
                    key = (kv_cache_dtype, spec.num_heads // tp_size, spec.batch_size, spec.input_len)
                    if key not in cases or tp_size < cases[key].tp_size:
                        cases[key] = MlaCase(
                            spec.input_len, spec.batch_size, kv_cache_dtype, spec.num_heads, tp_size, is_context
                        )
        return list(cases.values())

    return expand


def _mla_run_args(case: MlaCase, kv_cache_dtype, tokens_per_block: int) -> list:
    """The legacy run_mla tuple both backends take; output_len 1, 10 warm-up and 6 timed iterations."""
    return [
        case.input_len,
        case.batch_size,
        1,
        kv_cache_dtype,
        case.num_heads,
        case.tp_size,
        case.tp_size,
        tokens_per_block,
        10,
        6,
        case.is_context_phase,
    ]


def _sglang_mla_rows(case: MlaCase, sm_version: int | None, runner: str | None) -> list[list]:
    """SGLang's run_mla takes a torch dtype and the page size of its MLA backend (64 for trtllm_mla, else 1).

    The backend depends on the runtime CUDA version, so it is resolved here, in
    the SGLang image, not when the plan is expanded.
    """
    import torch

    from collector.sglang.collect_mla import _select_default_mla_backend

    page_size = 64 if _select_default_mla_backend() == "trtllm_mla" else 1
    dtype = torch.bfloat16 if case.kv_cache_dtype == "bfloat16" else torch.float8_e4m3fn
    return [_mla_run_args(case, dtype, page_size)]


def _trtllm_mla_rows(case: MlaCase, sm_version: int | None, runner: str | None) -> list[list]:
    """TRT-LLM's run_mla takes a TensorRT-LLM dtype and the KV block size serving uses on this SM."""
    import tensorrt_llm

    from collector.trtllm.collect_mla import _mla_tokens_per_block

    data_type = tensorrt_llm.bindings.DataType
    dtype = data_type.BF16 if case.kv_cache_dtype == "bfloat16" else data_type.FP8
    return [_mla_run_args(case, dtype, _mla_tokens_per_block())]


# --- mHC: mhc_module ---------------------------------------------------------------

_DSV4_ARCHITECTURE = "DeepseekV4ForCausalLM"
# The mHC call sites each architecture's serving forward runs. GLM-5.3-Flash also
# fuses the previous sublayer's post into the next pre and expands/contracts the
# residual stream once per forward (collector/vllm/collect_mhc_module.py).
_MHC_SITES = {
    _DSV4_ARCHITECTURE: ("pre", "post"),
    "Glm5NextForConditionalGeneration": ("pre", "post", "fused_post_pre", "expand", "contract"),
}
# The case-ID prefix of each architecture's rows.
_MHC_ID_PREFIX = {_DSV4_ARCHITECTURE: "mhc", "Glm5NextForConditionalGeneration": "mhc_glm5next"}


class MhcCase(NamedTuple):
    """One mHC call site of an architecture at one geometry, with the token sweep it runs."""

    architecture: str
    site: str
    hidden_size: int
    hc_mult: int
    num_tokens: tuple[int, ...]
    # The first model declared with this architecture and geometry; SGLang builds its layer from it.
    model_name: str

    __str__ = __repr__ = _list_form

    @property
    def dtype(self) -> str:
        return "bfloat16"


def _expand_mhc() -> list[MhcCase]:
    """One case per call site of each (architecture, hidden size, hc_mult); models that share one reuse it."""
    cases: dict[tuple, MhcCase] = {}
    for spec in get_common_mhc_test_cases():
        if spec.architecture not in _MHC_SITES:
            raise ValueError(f"No mHC call sites declared for {spec.architecture} ({spec.model_name})")
        for site in _MHC_SITES[spec.architecture]:
            key = (spec.architecture, site, spec.hidden_size, spec.hc_mult)
            cases.setdefault(key, MhcCase(*key, tuple(spec.num_tokens_list), spec.model_name))
    return list(cases.values())


def _mhc_row_id(case: MhcCase, suffix: object) -> str:
    return f"{_MHC_ID_PREFIX[case.architecture]}_{case.site}_hs{case.hidden_size}_hcm{case.hc_mult}_{suffix}"


def _trtllm_mhc_rows(case: MhcCase, sm_version: int | None, runner: str | None) -> list[dict]:
    """One row per token count; TRT-LLM drives the DeepSeek-V4 module API, which takes no model."""
    return [
        {"id": _mhc_row_id(case, tokens), "params": [case.site, tokens, case.hidden_size, case.hc_mult]}
        for tokens in case.num_tokens
    ]


def _vllm_mhc_rows(case: MhcCase, sm_version: int | None, runner: str | None) -> list[dict]:
    """One row per token count; rows other than DeepSeek-V4's also name their architecture and model."""
    model = [] if case.architecture == _DSV4_ARCHITECTURE else [case.architecture, case.model_name]
    return [
        {"id": _mhc_row_id(case, tokens), "params": [case.site, tokens, case.hidden_size, case.hc_mult, *model]}
        for tokens in case.num_tokens
    ]


def _sglang_mhc_rows(case: MhcCase, sm_version: int | None, runner: str | None) -> list[dict]:
    """One row per site: SGLang builds the model's layer once and sweeps every token count in it."""
    architecture = [] if case.architecture == _DSV4_ARCHITECTURE else [case.architecture]
    return [
        {
            "id": _mhc_row_id(case, case.model_name.replace("/", "_")),
            "params": [case.site, case.model_name, *architecture],
        }
    ]


# --- Attention ----------------------------------------------------------------


class AttentionCase(NamedTuple):
    """One dense-attention point: shape, batch, length and precision.

    ``input_len`` is the prefill length in context and the KV length before
    the decoded token in generation. ``attn_dtype`` is the compute dtype.
    """

    batch_size: int
    input_len: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    window_size: int
    kv_cache_dtype: str
    attn_dtype: str

    __str__ = __repr__ = _list_form

    @property
    def q_per_kv(self) -> int:
        """Query heads per KV head."""
        return self.num_heads // self.num_kv_heads


def _attention_points(sweep: dict, phase: str, num_heads: int, num_kv_heads: int, head_dim: int) -> list:
    """The (batch, length) points of one shape under the sweep's budgets, in run order."""
    batch_sizes = [int(value) for value in sweep["batch_sizes"]]
    sequence_lengths = [int(value) for value in sweep["sequence_lengths"]]
    if phase == "context":
        is_mha = num_kv_heads == num_heads
        points = []
        for seq in sorted(sequence_lengths, reverse=True):
            for batch in sorted(batch_sizes, reverse=True):
                if is_mha:
                    if batch * seq > int(sweep["max_tokens_self_attention"]):
                        continue
                    if batch > int(sweep["max_batch_size_self_attention"]):
                        continue
                elif batch * seq > int(sweep["max_tokens_grouped_query_attention"]):
                    continue
                if batch * seq * num_kv_heads * head_dim * 2 >= int(sweep["max_kv_elements"]):
                    continue
                points.append((batch, seq))
        return points

    # Generation: per KV length, every batch within the per-step token budget,
    # scaled up for head_dim below 128; keep a length only with enough batches.
    max_tokens = int(sweep["max_mha_tokens_per_step" if num_kv_heads == num_heads else "max_xqa_tokens_per_step"])
    batches_of: dict[int, set[int]] = {}
    for seq in sequence_lengths:
        max_batch = max_tokens * 128 // min(head_dim, 128) // seq // num_heads
        for batch in sorted(batch_sizes):
            if batch > max_batch:
                break
            batches_of.setdefault(seq, set()).add(batch)
    lengths_of: dict[int, set[int]] = {}
    for seq, batches in batches_of.items():
        if len(batches) < int(sweep["min_batch_options_per_sequence"]):
            continue
        for batch in batches:
            lengths_of.setdefault(batch, set()).add(seq - 1)
    points = []
    for batch, lengths in lengths_of.items():
        kept = sorted(lengths)
        if batch >= int(sweep["drop_largest_sequence_for_batch_at_least"]):
            kept = kept[:-1]
        points.extend((batch, length) for length in kept)
    return points


def _attention_expand(phase: str) -> Callable[[], list[AttentionCase]]:
    def expand() -> list[AttentionCase]:
        (sweep,) = get_shared_base_op_case_specs(f"attention_{phase}")
        cases = []
        for num_heads, num_kv_heads, head_dim, window_size in get_attention_shapes(phase):
            for batch, length in _attention_points(sweep, phase, num_heads, num_kv_heads, head_dim):
                for precision in sweep["precision_cases"]:
                    cases.append(
                        AttentionCase(
                            batch,
                            length,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            window_size,
                            "fp8" if precision["fp8_kv_cache"] else "bfloat16",
                            "fp8" if precision.get("fp8_context_fmha") else "bfloat16",
                        )
                    )
        return cases

    return expand


def _attention_owners(phase: str, case: AttentionCase) -> tuple[AttentionOwner, ...]:
    """The profiles that plan a case's shape, models first (a model's semantics win over the base grid's)."""
    owners = get_attention_shapes(phase)[(case.num_heads, case.num_kv_heads, case.head_dim, case.window_size)]
    return tuple(sorted(owners, key=lambda owner: not owner.from_model))


def _sglang_attention_dispatch(phase: str) -> Callable[[AttentionCase, int | None], tuple[str, ...]]:
    """The SGLang backends serving runs a case on: one per model that plans the shape.

    A shape two models serve on different backends (Nemotron-H on flashinfer,
    the generic grid on trtllm_mha) is measured on both.
    """

    def dispatch(case: AttentionCase, sm_version: int | None) -> tuple[str, ...]:
        backends = []
        for owner in _attention_owners(phase, case):
            backend = owner.sglang_backend(sm_version)
            if backend is None:
                raise ValueError(f"No SGLang attention backend for SM{sm_version}")
            if backend not in backends:
                backends.append(backend)
        return tuple(backends)

    return dispatch


def _sglang_owner(phase: str, case: AttentionCase, sm_version: int | None, backend: str) -> AttentionOwner:
    """The profile whose runtime semantics SGLang's row on this backend takes; profiles must agree."""
    owners = [owner for owner in _attention_owners(phase, case) if owner.sglang_backend(sm_version) == backend]
    signatures = {_sglang_signature(case, owner) for owner in owners if owner.from_model}
    if len(signatures) > 1:
        raise ValueError(
            f"SGLang attention profiles plan {case} on {backend} with different runtime semantics: "
            f"{sorted(map(str, signatures))}"
        )
    return owners[0]


def _sglang_signature(case: AttentionCase, owner: AttentionOwner) -> tuple:
    return (
        owner.v_head_dim or case.head_dim,
        _sglang_runtime_window(case, owner),
        owner.sglang_attention_chunk_size if case.window_size > 0 else None,
        owner.has_attention_sink,
        owner.sglang_scaling,
    )


def _sglang_runtime_window(case: AttentionCase, owner: AttentionOwner) -> int:
    if owner.sglang_runtime_window_size is not None:
        return owner.sglang_runtime_window_size
    return case.window_size or -1


def _sglang_attention_rows(phase: str):
    is_context = phase == "context"

    def rows(case: AttentionCase, sm_version: int | None, backend: str | None) -> list[list]:
        owner = _sglang_owner(phase, case, sm_version, backend)
        return [
            [
                case.batch_size,
                case.input_len,
                case.num_heads,
                case.num_kv_heads,
                case.head_dim,
                case.kv_cache_dtype == "fp8",
                case.attn_dtype == "fp8",
                is_context,
                case.window_size,
                owner.v_head_dim or case.head_dim,
                _sglang_runtime_window(case, owner),
                owner.sglang_attention_chunk_size,
                owner.has_attention_sink,
                owner.sglang_scaling,
                backend,
                owner.architecture,
            ]
        ]

    return rows


def _trtllm_attention_rows(phase: str):
    is_context = phase == "context"

    # TRT-LLM serves every collected model on its default TRTLLM attention
    # backend. Its run function adds attention sinks to every head_dim 64 row:
    # GPT-OSS has them, Kimi-K3's draft attention does not.
    def rows(case: AttentionCase, sm_version: int | None, runner: str | None) -> list[list]:
        return [
            [
                case.batch_size,
                case.input_len,
                case.num_heads,
                case.num_kv_heads,
                case.head_dim,
                case.window_size,
                case.kv_cache_dtype == "fp8",
                case.attn_dtype == "fp8",
                is_context,
                "TRTLLM",
            ]
        ]

    return rows


def _vllm_attention_rows(phase: str):
    is_context = phase == "context"

    # vLLM's attention collector takes one head_dim and no sinks: it times
    # MiMo-V2-Flash's head_dim 192 rows with V=192 (serving's V head is 128)
    # and GPT-OSS's rows without attention sinks.
    def rows(case: AttentionCase, sm_version: int | None, runner: str | None) -> list[list]:
        return [
            [
                case.batch_size,
                case.input_len,
                case.num_heads,
                case.num_kv_heads,
                case.head_dim,
                case.kv_cache_dtype == "fp8",
                is_context,
                case.window_size,
            ]
        ]

    return rows


def _attention_plan(phase: str) -> OpPlan:
    return OpPlan(
        AttentionCase,
        _attention_expand(phase),
        dtype_field="kv_cache_dtype",
        group_fields=("num_heads", "num_kv_heads", "head_dim", "window_size"),
        dispatch={"sglang": _sglang_attention_dispatch(phase)},
        rows={
            "sglang": _sglang_attention_rows(phase),
            "trtllm": _trtllm_attention_rows(phase),
            "vllm": _vllm_attention_rows(phase),
        },
    )


OP_PLANS: dict[str, OpPlan] = {
    "gemm": OpPlan(GemmCase, _expand_gemm, dtype_field="gemm_type", group_fields=("gemm_type", "n", "k")),
    "attention_context": _attention_plan("context"),
    "attention_generation": _attention_plan("generation"),
    "moe": OpPlan(
        MoeCase,
        _expand_moe,
        dtype_field="moe_type",
        # Every case builds its own layer.
        group_fields=MoeCase._fields,
        dispatch={"sglang": _sglang_moe_runner},
        rows={"sglang": _sglang_moe_rows, "trtllm": _trtllm_moe_rows},
    ),
    "compute_scale": OpPlan(ComputeScaleCase, _expand_compute_scale, dtype_field="dtype", group_fields=("m", "k")),
    "encoder_attention": OpPlan(
        EncoderAttentionCase, _expand_encoder_attention, dtype_field="dtype", group_fields=EncoderAttentionCase._fields
    ),
    "gdn": OpPlan(
        GdnCase,
        _expand_gdn,
        dtype_field="dtype",
        group_fields=GdnCase._fields,
        rows={"trtllm": _gdn_rows_without_state_dtype, "vllm": _gdn_rows_without_state_dtype},
    ),
    "kda": OpPlan(KdaCase, _expand_kda, dtype_field="dtype", group_fields=KdaCase._fields),
    "mamba2": OpPlan(Mamba2Case, _expand_mamba2, dtype_field="dtype", group_fields=Mamba2Case._fields),
    "mla_context": OpPlan(
        MlaCase,
        _mla_expand(is_context=True),
        dtype_field="kv_cache_dtype",
        group_fields=MlaCase._fields,
        rows={"sglang": _sglang_mla_rows, "trtllm": _trtllm_mla_rows},
    ),
    "mla_generation": OpPlan(
        MlaCase,
        _mla_expand(is_context=False),
        dtype_field="kv_cache_dtype",
        group_fields=MlaCase._fields,
        rows={"sglang": _sglang_mla_rows, "trtllm": _trtllm_mla_rows},
    ),
    "mla_bmm_gen_pre": OpPlan(
        MlaBmmCase, _mla_bmm_expand("mla_bmm_gen_pre"), dtype_field="dtype", group_fields=MlaBmmCase._fields
    ),
    "mla_bmm_gen_post": OpPlan(
        MlaBmmCase, _mla_bmm_expand("mla_bmm_gen_post"), dtype_field="dtype", group_fields=MlaBmmCase._fields
    ),
    "mhc_module": OpPlan(
        MhcCase,
        _expand_mhc,
        dtype_field="dtype",
        group_fields=MhcCase._fields,
        rows={"sglang": _sglang_mhc_rows, "trtllm": _trtllm_mhc_rows, "vllm": _vllm_mhc_rows},
    ),
}
