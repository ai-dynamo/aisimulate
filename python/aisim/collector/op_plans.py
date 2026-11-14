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
gives at least one row. Where one task times several cases on one module
build, the backend declares ``batch_rows`` instead: each row carries the cases
it times as explicit points, and every kept case is timed by exactly one row
(on each of its dispatches, where the plan picks them).

Planning never imports a framework, so ``--plan-only`` runs on any CPU.

Each backend's getter is one line, ``return plan_cases("<op>", "<backend>",
get_sm_version())``. To move an op onto a plan, add its case type, its expand
function and an :class:`OpPlan` entry in ``OP_PLANS``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

from collector import skips
from collector.case_generator import (
    BASE_OP_CASES_DIR,
    AttentionOwner,
    Dsv4Sweep,
    MoeCommonTestCase,
    _attention_profile_shapes,
    _framework_specific_model_case_values,
    _get_model_path_filter,
    _model_case_values,
    get_attention_encoder_head_configs,
    get_attention_shapes,
    get_base_common_case_values,
    get_common_gdn_test_cases,
    get_common_kda_test_cases,
    get_common_mamba2_test_cases,
    get_common_mhc_test_cases,
    get_common_moe_test_cases,
    get_compute_scale_case_specs,
    get_context_mla_case_specs,
    get_dsv4_models,
    get_dsv4_sweep,
    get_gemm_case_specs,
    get_gemm_type_specs,
    get_generation_mla_case_specs,
    get_mla_bmm_case_specs,
    get_mla_module_model_specs,
    get_mla_module_precision_specs,
    get_mla_module_sweep_spec,
    get_moe_quantization_module_config,
    get_sglang_moe_backend,
    get_shared_base_op_case_specs,
    is_wideep_moe_model,
    moe_model_layer,
    moe_planned_modes,
    without_models,
)
from collector.coverage import includes_long_context, long_context_coverage, long_context_points
from collector.helper import _AIC_MODEL_CONFIG_DIR

# Plan logs go to the collector log, never stdout: --plan-only prints JSON there.
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class OpPlan:
    """How one op's shared cases are expanded."""

    case_type: type
    # Every case, in run order, before whole regions are removed.
    expand: Callable[[], list]
    # The case field, or fields, that hold its dtype or quant mode.
    dtype_field: str | tuple[str, ...]
    # The case fields whose values share one module build in a worker.
    group_fields: tuple[str, ...]
    # The mapping defaults come from factories: a mappingproxy default is
    # rejected by dataclasses on Python 3.11, where it is unhashable.
    # backend -> the kernel or runner the backend measures a case on, where the
    # plan picks it; a tuple when serving runs the case on several.
    dispatch: Mapping[str, Callable[[object, int | None], str | tuple[str, ...]]] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    # backend -> the run arguments for one case and its dispatch (None without
    # one), for a run function that takes more than the case.
    rows: Mapping[str, Callable[[object, int | None, str | None], list]] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )
    # backend -> the tasks for its kept cases where one task times several of
    # them on one module build: (row, the cases it times) pairs. A row carries
    # its points explicitly; every kept case is timed by exactly one row. With a
    # dispatch, the function is given the (case, dispatch) runs instead, and
    # every run is timed by exactly one row.
    batch_rows: Mapping[str, Callable[[list, int | None], list[tuple[object, tuple]]]] = dataclasses.field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        both = sorted(set(self.batch_rows) & set(self.rows))
        if both:
            raise ValueError(f"a backend declares batch_rows or rows, not both: {both}")

    def dtypes(self, case) -> tuple[str, ...]:
        """The case's dtypes, which skips.py ``dtype`` entries and the per-dtype counts read."""
        fields = (self.dtype_field,) if isinstance(self.dtype_field, str) else self.dtype_field
        return tuple(getattr(case, field) for field in fields)


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
    # Or the backend's batch rows: (row, the cases it times) pairs.
    to_batch_rows: Callable[[list, int | None], list[tuple[object, tuple]]] | None = None
    sm_version: int | None = None
    where: str = ""
    # The model rows the plan left out; rows read the model profiles without them.
    skipped_models: frozenset[str] = frozenset()
    include_long_context: bool = False

    @property
    def by_dispatch(self) -> dict[str, int]:
        """How many cases run on each dispatch."""
        return dict(Counter(runner for runners in self.dispatches for runner in runners))

    @property
    def rows(self) -> list:
        """What the getter queues, retaining the coverage selected when this plan was built."""
        with long_context_coverage(self.include_long_context):
            return self._rows()

    def _rows(self) -> list:
        if self.to_batch_rows is not None:
            units = self.cases
            if self.dispatches:
                units = [
                    _Run(case, runner)
                    for case, runners in zip(self.cases, self.dispatches, strict=True)
                    for runner in runners
                ]
            with without_models(self.skipped_models):
                pairs = self.to_batch_rows(units, self.sm_version)
            timed = Counter(unit for _row, timed_units in pairs for unit in timed_units)
            wrong = [unit for unit in units if timed[unit] != 1] + list(set(timed) - set(units))
            if wrong:
                raise ValueError(f"{self.where} batch rows must time every kept case exactly once: {wrong[:3]}")
            return [row for row, _units in pairs]
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


def plan(op: str, backend: str, sm_version: int | None, *, include_long_context: bool | None = None) -> Plan:
    """Plan ordinary coverage, optionally unioned with declared model-owned long-context points."""
    enabled = includes_long_context() if include_long_context is None else include_long_context
    with long_context_coverage(enabled):
        return _plan(op, backend, sm_version)


def _plan(op: str, backend: str, sm_version: int | None) -> Plan:
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

    dtypes_of = spec.dtypes
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
        to_batch_rows=spec.batch_rows.get(backend),
        sm_version=sm_version,
        where=f"{backend} {op}",
        skipped_models=skipped_models,
        include_long_context=includes_long_context(),
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
        "by_dtype": dict(Counter("/".join(spec.dtypes(case)) for case in op_plan.cases)),
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

# SGLang serves these architectures' MoE on triton_kernel at EP1 where the per-SM
# pin is Triton (_sglang_moe_runner). Source: sglang v0.5.23. GPT-OSS's model
# override turns an auto runner into triton_kernel when ep_size == 1, the
# triton_kernels package is installed and no --quantization is passed
# (srt/arg_groups/model_overrides/gpt_oss.py:102-120): sglang-kernel ships the
# package (kernels/aot/CMakeLists.txt:548-553; is_triton_kernels_available,
# srt/utils/common.py:4814-4815) and GPT-OSS checkpoints are served without the
# flag. SM100/103 and SM120 take flashinfer_mxfp4 first (gpt_oss.py:64-75). At
# EP>1 the runner stays auto, which Mxfp4MoEMethod resolves to Triton
# (srt/layers/quantization/mxfp4.py:1443-1450): the per-SM pin
# (cases/base_ops/moe.yaml moe_sglang.backends.w4a16_mxfp4, Triton on SM89/90).
SGLANG_EP1_TRITON_KERNELS_ARCHITECTURES = frozenset({"GptOssForCausalLM"})
# SGLang routes these architectures' experts with the model's own routing
# function, not the row's routing fields; collector/sglang/collect_moe.py builds
# that TopK (_gemma4_routing_function).
SGLANG_GEMMA4_ROUTING_ARCHITECTURES = frozenset({"Gemma4ForConditionalGeneration"})
_SGLANG_ARCHITECTURE_SPECIFIC_MOE = SGLANG_EP1_TRITON_KERNELS_ARCHITECTURES | SGLANG_GEMMA4_ROUTING_ARCHITECTURES


def _is_fp4_experts(layer: MoeCommonTestCase, moe_type: str) -> bool:
    return layer.architecture == "DeepseekV4ForCausalLM" and moe_type in _FP4_EXPERT_MODES


def _moe_layer_identity(layer: MoeCommonTestCase, moe_type: str) -> tuple:
    """What a MoE case times besides its consumer key: routing and activation, runner pins, quant config.

    An architecture SGLang builds its own way (EP1 runner, routing function)
    is part of what the case times.
    """
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
    sglang_architecture = layer.architecture if layer.architecture in _SGLANG_ARCHITECTURE_SPECIFIC_MOE else None
    return routing, tuple(runners), module_configs, _is_fp4_experts(layer, moe_type), sglang_architecture


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
    sglang_moe_backends; an EP1 case of SGLANG_EP1_TRITON_KERNELS_ARCHITECTURES
    runs triton_kernel where that pin is Triton, as serving's model override
    decides by EP size.
    """
    layer = _moe_layer_of(case)
    runner = get_sglang_moe_backend(layer, case.moe_type, sm_version)
    if runner == "triton" and case.ep == 1 and layer.architecture in SGLANG_EP1_TRITON_KERNELS_ARCHITECTURES:
        return "triton_kernel"
    return runner


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


# --- Wide EP: moe_ep, deepep_ll, deepep_normal -----------------------------------


class MoeEpCase(NamedTuple):
    """One large-EP MoE expert-compute build: a quant mode, one routed-expert layer at one EP size, its EPLB layout.

    A task times the build over a token sweep. TRT-LLM runs one task per
    power-law draw of the shared MoE grid (``power_law_alphas``) over its
    ``num_tokens``; SGLang loads the model's checkpoint and sweeps its own
    prefill and decode grid in one task.
    """

    moe_type: str
    hidden_size: int
    inter_size: int
    topk: int
    num_experts: int
    ep: int
    model_name: str
    use_eplb: bool
    num_slots: int
    num_tokens: tuple[int, ...]
    power_law_alphas: tuple[float, ...]
    # moe_type is a quant mode the model's checkpoint plans.
    own_quant: bool

    __str__ = __repr__ = _list_form


_MOE_EP_MODES = ("fp8_block", "nvfp4")
# The redundant-expert layout EPLB serves: 288 slots.
_MOE_EP_REDUNDANT_SLOTS = 288


def _expand_moe_ep() -> list[MoeEpCase]:
    """The ``wideep: true`` MoE rows at every EP size of the shared grid (TP 1), per quant mode and EPLB layout.

    EPLB runs off (``num_slots == num_experts``), on with the same slots, and on
    with the 288-slot redundant layout where it holds every expert. Slots shard
    evenly over the EP ranks. ``moe_expert_compute_perf`` has no model column, so
    models whose builds write the same key share one case, owned by the first
    model whose checkpoint plans the quant mode, else by the first model.
    Ordered by model, quant mode, EP size and layout.
    """
    sweeps: dict[str, dict[int, dict]] = {}
    for layer in get_common_moe_test_cases():
        if not is_wideep_moe_model(layer.model_name) or layer.tp != 1 or layer.ep <= 1:
            continue
        sweep = sweeps.setdefault(layer.model_name, {}).setdefault(
            layer.ep, {"layer": layer, "num_tokens": set(), "alphas": set()}
        )
        if layer.token_expert_distribution == "power_law":
            sweep["num_tokens"].add(tuple(layer.num_tokens_list))
            sweep["alphas"].add(layer.power_law_alpha)
    cases = []
    for model_name in sorted(sweeps):
        first = next(iter(sweeps[model_name].values()))["layer"]
        planned = moe_planned_modes(model_name, first.quantization_modes)
        for moe_type in _MOE_EP_MODES:
            for ep, sweep in sorted(sweeps[model_name].items()):
                layer = sweep["layer"]
                if len(sweep["num_tokens"]) > 1:
                    raise ValueError(f"moe_ep: {model_name} at EP {ep} declares more than one token sweep")
                num_tokens = next(iter(sweep["num_tokens"]), ())
                layouts = [(False, layer.num_experts), (True, layer.num_experts)]
                if layer.num_experts <= _MOE_EP_REDUNDANT_SLOTS:
                    layouts.append((True, _MOE_EP_REDUNDANT_SLOTS))
                for use_eplb, num_slots in layouts:
                    if num_slots % ep:
                        continue
                    build = (moe_type, layer.hidden_size, layer.inter_size, layer.topk, layer.num_experts, ep)
                    build += (use_eplb, num_slots, num_tokens, tuple(sorted(sweep["alphas"])))
                    cases.append(MoeEpCase(*build[:6], model_name, *build[6:], own_quant=moe_type in planned))
    owners: dict[MoeEpCase, MoeEpCase] = {}
    for case in cases:
        key = case._replace(model_name="", own_quant=False)
        owner = owners.setdefault(key, case)
        if case.own_quant and not owner.own_quant:
            owners[key] = case
    kept = set(owners.values())
    return [case for case in cases if case in kept]


def _sglang_moe_ep_rows(case: MoeEpCase, sm_version: int | None, runner: str | None) -> list[list]:
    """SGLang's run_moe_ep loads the rank-local expert shard of the model's checkpoint."""
    return [
        [
            case.num_experts // case.ep,
            case.ep,
            case.hidden_size,
            case.inter_size,
            case.topk,
            case.num_experts,
            case.num_slots,
            case.moe_type,
            case.model_name,
        ]
    ]


def _trtllm_moe_ep_rows(case: MoeEpCase, sm_version: int | None, runner: str | None) -> list[list]:
    """TRT-LLM's run_moe_ep times one power-law draw per task on the CUTLASS runner, without min-latency mode.

    The last argument keeps its accurate wide-EP simulation: the router sees one
    DP rank's tokens and the experts only the tokens routed to rank 0.
    """
    head = [case.moe_type, "cutlass", list(case.num_tokens), *case[1:5], 1, case.ep, False, case.model_name]
    return [[*head, "power_law", alpha, case.use_eplb, case.num_slots, True] for alpha in case.power_law_alphas]


class DeepepCase(NamedTuple):
    """One DeepEP dispatch/combine shape; a task sweeps its token counts (and, in normal mode, SM budgets) inside."""

    hidden_size: int
    num_experts: int
    topk: int

    __str__ = __repr__ = _list_form


def _expand_deepep() -> list[DeepepCase]:
    """Every distinct ``(hidden_size, num_experts, topk)`` among the packaged MoE model configs, in order.

    Field names differ across DeepSeek, Qwen, Llama, MiniMax and gpt-oss
    configs; VL models nest the language config under ``text_config``.
    """
    shapes = set()
    for path in Path(_AIC_MODEL_CONFIG_DIR).glob("*_config.json"):
        # A packaged config that does not parse is a packaging error, not a shape to skip.
        config = json.loads(path.read_text())
        text = config.get("text_config") or {}
        num_experts = (
            config.get("n_routed_experts")
            or config.get("num_experts")
            or config.get("num_local_experts")
            or text.get("n_routed_experts")
            or text.get("num_experts")
            or text.get("num_local_experts")
        )
        topk = (
            config.get("num_experts_per_tok")
            or config.get("moe_topk")
            or config.get("num_experts_per_token")
            or config.get("topk")
            or text.get("num_experts_per_tok")
        )
        hidden_size = config.get("hidden_size") or text.get("hidden_size")
        if num_experts and topk and hidden_size:
            shapes.add(DeepepCase(int(hidden_size), int(num_experts), int(topk)))
    return sorted(shapes)


def _deepep_rows(case: DeepepCase, sm_version: int | None, runner: str | None) -> list[dict]:
    """The full-node DeepEP runners take each shape as a dict."""
    return [case._asdict()]


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

    @property
    def local_heads(self) -> int:
        """The heads one TP rank runs."""
        return self.num_heads // self.tp_size


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


# --- Attention modules: the MLA and DSA context and generation modules -----------------

# The section of cases/base_ops/mla_module.yaml (mla_module_plan) the module op plans
# share; the per-backend sections serve the module ops still on their own getters.
_MODULE_SWEEP = "plan"


class ModuleCase(NamedTuple):
    """One timed point of an attention module.

    ``seq_len`` is the query length (context) or the KV length (generation);
    ``prefix_len`` is the cached context behind a context point.
    """

    seq_len: int
    batch_size: int
    num_heads: int
    kv_cache_dtype: str
    compute_dtype: str
    gemm_type: str
    model_path: str
    prefix_len: int = 0

    __str__ = __repr__ = _list_form

    @property
    def checkpoint_quant(self) -> str:
        """The checkpoint's weight quantization: fp8_block, nvfp4 or bfloat16 (none)."""
        return _checkpoint_quant(self.model_path)

    @property
    def own_gemm(self) -> bool:
        """Whether gemm_type is the GEMM the checkpoint's own attention projections run."""
        return self.gemm_type == _projection_gemm(self.model_path)


def _mla_module_models() -> list:
    """The MLA module's model rows: the canonical declared model, or a targeted run's own.

    The MLA module table has no model key, so a full plan builds every module from
    the one canonical model (the loader collapses the declared aliases), and a
    targeted run (COLLECTOR_MODEL_PATH) measures its own model. Rows that either
    collecting backend declares count, so both plan the same models.
    """
    models: dict[str, object] = {}
    for backend in ("trtllm", "vllm"):
        for spec in get_mla_module_model_specs(attention_type="mla", backend=backend):
            models.setdefault(spec.model_path, spec)
    return list(models.values())


def _mla_module_expand(phase: str) -> Callable[[], list[ModuleCase]]:
    def expand() -> list[ModuleCase]:
        """Every (precision, heads, batch, length) point of each model, within the phase's token budget."""
        sweep = get_mla_module_sweep_spec(_MODULE_SWEEP)
        if phase == "context":
            batches, lengths = sweep.context_batch_sizes, sweep.context_sequence_lengths
            budget = sweep.context_max_tokens
        else:
            batches, lengths = sweep.generation_batch_sizes, sweep.generation_sequence_lengths
            budget = sweep.generation_max_tokens
        precisions = get_mla_module_precision_specs(_MODULE_SWEEP, phase=phase, attention_type="mla")
        cases = []
        for model in _mla_module_models():
            # A row that declares its tensor_parallel_sizes measures only the heads those shards run.
            shards = model.tensor_parallel_sizes
            shard_heads = {model.native_num_heads // tp for tp in shards} if shards else None
            for precision in precisions:
                for num_heads in sweep.inner_sweep_head_counts:
                    if shard_heads is not None and num_heads not in shard_heads:
                        continue
                    for batch_size in batches:
                        for length in lengths:
                            if batch_size * length > budget:
                                continue
                            cases.append(
                                ModuleCase(
                                    length,
                                    batch_size,
                                    num_heads,
                                    precision.kv_cache_dtype,
                                    precision.compute_dtype,
                                    precision.gemm_type,
                                    model.model_path,
                                )
                            )
        return _with_history_module_cases(cases, "mla", phase, sweep)

    return expand


def _module_series(cases: list[ModuleCase]) -> list[tuple[tuple, list[ModuleCase]]]:
    """The cases grouped by the module build they share (batch, heads, precision, model), in plan order."""
    series: dict[tuple, list[ModuleCase]] = {}
    for case in cases:
        precision = (case.kv_cache_dtype, case.compute_dtype, case.gemm_type)
        series.setdefault((case.batch_size, case.num_heads, *precision, case.model_path), []).append(case)
    return list(series.items())


def _trtllm_module_rows(*trailing: str) -> Callable[[list, int | None], list[tuple[list, tuple]]]:
    def rows(cases: list[ModuleCase], sm_version: int | None) -> list[tuple[list, tuple]]:
        """One series worker task per module: its (seq_len, prefix_len) points, the key, then
        ``trailing`` (the attention type, for the MLA module's worker)."""
        tasks = []
        for key, members in _module_series(cases):
            points = sorted(([case.seq_len, case.prefix_len] for case in members), reverse=True)
            tasks.append(([points, *key, *trailing], tuple(members)))
        return tasks

    return rows


def _vllm_module_rows(attn_type: str) -> Callable[[list, int | None], list[tuple[list, tuple]]]:
    def rows(cases: list[ModuleCase], sm_version: int | None) -> list[tuple[list, tuple]]:
        """One run_mla_module_worker task per module: its lengths, largest first, then the key.

        A module with cached prefixes lists its (seq_len, prefix_len) points instead.
        """
        tasks = []
        for key, members in _module_series(cases):
            if any(case.prefix_len for case in members):
                points = sorted(([case.seq_len, case.prefix_len] for case in members), reverse=True)
            else:
                points = sorted((case.seq_len for case in members), reverse=True)
            tasks.append(([points, *key, attn_type], tuple(members)))
        return tasks

    return rows


_MODULE_GROUP = ("model_path", "num_heads", "kv_cache_dtype", "compute_dtype", "gemm_type", "batch_size")


def _mla_module_plan(phase: str) -> OpPlan:
    return OpPlan(
        ModuleCase,
        _mla_module_expand(phase),
        dtype_field=("kv_cache_dtype", "compute_dtype", "gemm_type"),
        group_fields=_MODULE_GROUP,
        batch_rows={"trtllm": _trtllm_module_rows("mla"), "vllm": _vllm_module_rows("mla")},
    )


@functools.cache
def _packaged_config(model_path: str, kind: str = "config") -> dict:
    """A checkpoint's packaged config.json (kind "config") or hf_quant_config.json.

    Every planned model ships its config.json; a missing one is a packaging
    error and raises, since the plan would otherwise read every field as
    absent and queue a different case set. hf_quant_config.json ships for
    ModelOpt checkpoints only, so a missing one is {}.
    """
    path = Path(_AIC_MODEL_CONFIG_DIR) / f"{model_path.replace('/', '--')}_{kind}.json"
    if path.exists():
        return json.loads(path.read_text())
    if kind == "config":
        raise FileNotFoundError(f"{model_path} has no packaged config.json: expected {path}")
    return {}


def _max_position(model_path: str) -> int | None:
    value = _packaged_config(model_path).get("max_position_embeddings")
    return int(value) if value else None


def _modelopt_quant(model_path: str) -> dict:
    """The checkpoint's ModelOpt quantization section (hf_quant_config.json first), or its quantization_config."""
    quant = _packaged_config(model_path).get("quantization_config") or {}
    return _packaged_config(model_path, "hf_quant_config").get("quantization") or quant


def _fp4_lane(lane: object) -> bool:
    return isinstance(lane, dict) and "FP4" in str(lane.get("quant_algo", "")).upper()


def _checkpoint_quant(model_path: str) -> str:
    """The checkpoint's weight quantization, read from its packaged config: fp8_block, nvfp4 or bfloat16."""
    quant = _packaged_config(model_path).get("quantization_config") or {}
    if quant.get("quant_method") == "fp8" and quant.get("weight_block_size"):
        return "fp8_block"
    modelopt = _modelopt_quant(model_path)
    algo = str(modelopt.get("quant_algo", "")).upper()
    if algo == "NVFP4" or (
        algo == "MIXED_PRECISION" and any(_fp4_lane(lane) for lane in (modelopt.get("quantized_layers") or {}).values())
    ):
        return "nvfp4"
    return "bfloat16"


def _projection_gemm(model_path: str) -> str:
    """The GEMM a checkpoint's attention projections run, read from its packaged config.

    An NVFP4 checkpoint runs nvfp4 there when it quantizes o_proj, since the SDK keys the
    DSA module by o_proj (_dsa_attention_quant_modes in sdk/models/deepseek_v32.py):
    DeepSeek-V3.2-NVFP4 quantizes it, while the GLM-5.x NVFP4 checkpoints keep self_attn
    in BF16. A ModelOpt MIXED_PRECISION export quantizes only the modules its
    quantized_layers name.
    """
    quant = _checkpoint_quant(model_path)
    if quant != "nvfp4":
        return quant
    modelopt = _modelopt_quant(model_path)
    if str(modelopt.get("quant_algo", "")).upper() == "MIXED_PRECISION":
        lanes = modelopt.get("quantized_layers") or {}
        projection_lanes = [lane for key, lane in lanes.items() if key.endswith("o_proj")]
        if any(_fp4_lane(lane) for lane in projection_lanes):
            return "nvfp4"
        # Match the SDK's MXFP8 -> fp8_block projection approximation
        # (sdk/models/helpers.py:_infer_mixed_precision_quant_modes).
        if any(str(lane.get("quant_algo", "")).upper() == "MXFP8" for lane in projection_lanes):
            return "fp8_block"
        return "bfloat16"
    patterns = [str(pattern) for pattern in modelopt.get("exclude_modules") or []]
    keeps_o_proj = any(
        "o_proj" in pattern or ("self_attn" in pattern and pattern.rsplit("self_attn", 1)[1] in ("", "*", ".*"))
        for pattern in patterns
    )
    return "bfloat16" if keeps_o_proj else "nvfp4"


def _shares_dsa_index(model_path: str) -> bool:
    """Whether the checkpoint's layers share one DSA top-k index (index_topk_freq > 1), so most
    of them skip the indexer."""
    return int(_packaged_config(model_path).get("index_topk_freq") or 1) > 1


def _sparse_module_models(attention_type: str) -> list[tuple[object, tuple[str, ...]]]:
    """The DSA or MSA checkpoints the plan times, each with the projection GEMMs it stands for.

    One checkpoint stands for each (architecture, projection GEMM): the longest-context
    one whose projections run that GEMM, the first declared on a tie. A GEMM no
    checkpoint of the architecture runs is timed on its longest-context checkpoint. A
    targeted run (COLLECTOR_MODEL_PATH) declares only its own checkpoint, which then
    stands for every GEMM.
    """
    precisions = get_mla_module_precision_specs(_MODULE_SWEEP, attention_type=attention_type)
    gemms = list(dict.fromkeys(precision.gemm_type for precision in precisions))
    by_architecture: dict[str, list] = {}
    for spec in get_mla_module_model_specs(attention_type=attention_type):
        by_architecture.setdefault(spec.architecture, []).append(spec)
    stands_for: dict[object, list[str]] = {}
    for specs in by_architecture.values():
        for gemm in gemms:
            own = [spec for spec in specs if _projection_gemm(spec.model_path) == gemm]
            longest = max(own or specs, key=lambda spec: _max_position(spec.model_path) or 0)
            stands_for.setdefault(longest, []).append(gemm)
    return [(spec, tuple(gemms)) for spec, gemms in stands_for.items()]


def _long_context_points(phase: str, sweep, max_position: int | None) -> list[tuple[int, int, int]]:
    """The (batch, seq_len, prefix_len) points of one DSA or MSA module, batch by batch.

    Context follows serving's chunked prefill (the mla_module_plan comments): at most
    chunk_prefill_size new tokens and max_full_sequence_length KV tokens per forward,
    and prefix + seq_len within the model's context. Generation is the decode grid plus the long KV lengths
    at small batch; a KV length leaves room for the decoded token.
    """
    if phase == "context":
        return [
            (batch, length, prefix)
            for batch in sweep.context_batch_sizes
            for length in sweep.context_sequence_lengths
            for prefix in sweep.context_prefix_lengths
            if batch * length <= sweep.context_chunk_prefill_size
            and batch * (length + prefix) <= sweep.context_max_full_sequence_length
            and (max_position is None or prefix + length <= max_position)
        ]
    grid = {
        (batch, length)
        for batch in sweep.generation_batch_sizes
        for length in sweep.generation_sequence_lengths
        if batch * length <= sweep.generation_max_tokens
    }
    grid |= {
        (batch, length) for batch in sweep.generation_long_kv_batch_sizes for length in sweep.generation_long_kv_lengths
    }
    return [(batch, length, 0) for batch, length in sorted(grid) if max_position is None or length < max_position]


def _sparse_module_expand(
    attention_type: str, phase: str, *, shared_index_only: bool = False
) -> Callable[[], list[ModuleCase]]:
    def expand() -> list[ModuleCase]:
        """Every (precision, heads, point) of each checkpoint, at the projection GEMMs it stands for."""
        sweep = get_mla_module_sweep_spec(_MODULE_SWEEP)
        precisions = get_mla_module_precision_specs(_MODULE_SWEEP, phase=phase, attention_type=attention_type)
        cases = []
        for model, gemms in _sparse_module_models(attention_type):
            if shared_index_only and not _shares_dsa_index(model.model_path):
                continue
            # A row that declares its tensor_parallel_sizes measures only the heads those shards run.
            shards = model.tensor_parallel_sizes
            shard_heads = {model.native_num_heads // tp for tp in shards} if shards else None
            points = _long_context_points(phase, sweep, _max_position(model.model_path))
            for precision in precisions:
                if precision.gemm_type not in gemms:
                    continue
                for num_heads in sweep.inner_sweep_head_counts:
                    if shard_heads is not None and num_heads not in shard_heads:
                        continue
                    cases += [
                        ModuleCase(
                            length,
                            batch,
                            num_heads,
                            precision.kv_cache_dtype,
                            precision.compute_dtype,
                            precision.gemm_type,
                            model.model_path,
                            prefix,
                        )
                        for batch, length, prefix in points
                    ]
        return _with_history_module_cases(cases, attention_type, phase, sweep, shared_index_only=shared_index_only)

    return expand


def _history_config(model_path: str) -> dict:
    config = _packaged_config(model_path)
    return config.get("text_config") or config


def _history_limit(model_path: str) -> int:
    limit = _history_config(model_path).get("max_position_embeddings")
    if type(limit) is not int or limit <= 0:
        raise ValueError(f"long-context model {model_path} has no valid max_position_embeddings")
    return limit


def _history_attention_unquantized(model_path: str, config: dict) -> bool:
    """Prove a routed-expert quantization recipe leaves every self-attention module alone."""
    quant = config.get("quantization_config") or {}
    if not quant:
        return True
    modelopt = _modelopt_quant(model_path)
    algo = str(modelopt.get("quant_algo", "")).upper()
    if algo == "MIXED_PRECISION":
        lanes = modelopt.get("quantized_layers")
        return isinstance(lanes, dict) and bool(lanes) and all(".mlp." in path for path in lanes)
    if algo != "NVFP4":
        return False
    layers = config.get("num_hidden_layers")
    # Only whole-attention wildcards establish exclusion of all current and
    # future projections/indexers; a match for one named projection is insufficient.
    patterns = [
        pattern
        for pattern in modelopt.get("exclude_modules") or []
        if isinstance(pattern, str) and pattern.endswith("*")
    ]
    return (
        type(layers) is int
        and layers > 0
        and all(
            any(
                fnmatchcase(f"model.layers.{layer}.self_attn", pattern)
                or fnmatchcase(f"model.layers.{layer}.self_attn.", pattern)
                for pattern in patterns
            )
            for layer in range(layers)
        )
    )


def _history_runtime_signature(model_path: str) -> str:
    config = dict(_history_config(model_path))
    for label in ("_name_or_path", "_commit_hash", "transformers_version"):
        config.pop(label, None)
    if config.get("model_type") == "glm_moe_dsa":
        # These are outside the extracted self_attn module. Its native builders
        # take explicit qk_nope/qk_rope/v dimensions, not HF's redundant head_dim:
        # vllm.collect_mla_module._create_attention_module; TRT create_attention_layer;
        # SGLang DeepseekV2AttentionMLA.__init__ (deepseek_v2.py:1993-1998
        # @e00930c5489053f26d86b179cee0d087f846acbb) takes those explicit args;
        # its local config loader strips HF layer_types before constructing it.
        for field in ("bos_token_id", "eos_token_id", "pad_token_id", "num_experts", "mlp_bias", "moe_router_dtype"):
            config.pop(field, None)
        layer_types = config.get("layer_types")
        if isinstance(layer_types, list) and set(layer_types) == {"deepseek_sparse_attention"}:
            config.pop("layer_types")
        if all(config.get(field) is not None for field in ("qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")):
            config.pop("head_dim", None)
        # Never strip arbitrary quantization: every self_attn must be excluded,
        # or the mixed recipe must explicitly name only MLP modules. KV precision
        # is separately explicit in ModuleCase and the workers' cache config.
        if _history_attention_unquantized(model_path, config):
            config.pop("quantization_config", None)
        else:
            config["resolved_modelopt_quantization"] = _modelopt_quant(model_path)
    return json.dumps(config, sort_keys=True)


def _with_history_module_cases(
    cases: list[ModuleCase], attention_type: str, phase: str, sweep, *, shared_index_only: bool = False
) -> list[ModuleCase]:
    """Union exact model points before deduplicating consumer keys; keep ordinary owners first."""
    if not includes_long_context():
        return cases
    # Reuse the pinned planner's representative for an equivalent runtime,
    # even when a newly requested point was absent from its old grid.
    signatures = {path: _history_runtime_signature(path) for path in {case.model_path for case in cases}}
    representatives = {}
    for case in cases:
        architecture = (
            None if attention_type == "mla" else tuple(_history_config(case.model_path).get("architectures", []))
        )
        representatives.setdefault((architecture, case.gemm_type, signatures[case.model_path]), case.model_path)
    profiles = _model_case_values("mla_module")
    for backend in ("vllm", "sglang", "trtllm"):
        profiles += _framework_specific_model_case_values("mla_module", backend)
    precisions = get_mla_module_precision_specs(_MODULE_SWEEP, phase=phase, attention_type=attention_type)
    for profile in profiles:
        if profile.get("attention_type") != attention_type or not profile.get("long_context_cases"):
            continue
        model_path = profile["model_path"]
        if shared_index_only and not _shares_dsa_index(model_path):
            continue
        points = long_context_points(profile, phase, _history_limit(model_path))
        signatures.setdefault(model_path, _history_runtime_signature(model_path))
        architecture = None if attention_type == "mla" else tuple(_history_config(model_path).get("architectures", []))
        native_heads = int(profile["native_num_heads"])
        shards = profile.get("long_context_tensor_parallel_sizes", profile.get("tensor_parallel_sizes"))
        if not shards or any(type(tp) is not int or tp <= 0 or native_heads % tp for tp in shards):
            raise ValueError(f"long-context module {model_path} requires valid tensor_parallel_sizes")
        heads = sorted({native_heads // tp for tp in shards}, reverse=True)
        # Resolve declared artifact precision before selecting the canonical
        # measurement owner: aliases may have different projection GEMMs.
        artifacts = [model_path, *profile.get("model_aliases", [])]
        selected = _get_model_path_filter()
        if selected in artifacts:
            artifacts = [selected]
        projection_gemms = {_projection_gemm(artifact) for artifact in artifacts}
        for precision in precisions:
            if precision.gemm_type not in projection_gemms:
                continue
            owner = representatives.get((architecture, precision.gemm_type, signatures[model_path]), model_path)
            for num_heads in heads:
                for batch, length, prefix in points:
                    cases.append(
                        ModuleCase(
                            length,
                            batch,
                            num_heads,
                            precision.kv_cache_dtype,
                            precision.compute_dtype,
                            precision.gemm_type,
                            owner,
                            prefix,
                        )
                    )
    # Keep #676's ordinary owners, then union equal-runtime points onto them.
    # Config signatures are memoized only within this plan, never across inputs.
    physical = {}
    for case in cases:
        architecture = (
            None if attention_type == "mla" else tuple(_history_config(case.model_path).get("architectures", []))
        )
        key = (*case[:6], architecture, case.prefix_len)
        existing = physical.get(key)
        if existing is not None and signatures[existing.model_path] != signatures[case.model_path]:
            raise ValueError(
                f"long-context coverage collides at one {attention_type} persisted key with incompatible runtime "
                f"semantics: {existing.model_path} versus {case.model_path}; declare a compatible representative"
            )
        physical.setdefault(key, case)
    return list(physical.values())


def _sglang_dsa_backend(phase: str) -> Callable[[ModuleCase, int | None], str]:
    def dispatch(case: ModuleCase, sm_version: int | None) -> str:
        """The DSA attention backend SGLang serves the case on.

        FP8 KV runs trtllm on Blackwell and flashmla_kv on Hopper, which the collector
        pins (collect_mla_module.load_model_runner). BF16 KV stays on SGLang's own
        default: flashmla_sparse for prefill, trtllm or fa3 for decode.
        """
        blackwell = sm_version is not None and sm_version >= 100
        if case.kv_cache_dtype == "fp8":
            return "trtllm" if blackwell else "flashmla_kv"
        if phase == "context":
            return "flashmla_sparse"
        return "trtllm" if blackwell else "fa3"

    return dispatch


def _sglang_dsa_rows(phase: str) -> Callable[[list, int | None], list[tuple[list, tuple]]]:
    def rows(runs: list[_Run], sm_version: int | None) -> list[tuple[list, tuple]]:
        """One run_mla_module_worker task per ModelRunner: one context batch, or every decode batch.

        A row keeps the worker's positional fields (batch 0 for decode), then the target
        TP of the heads, the FP8-KV backend the collector pins (None keeps SGLang's
        default) and the [batch, seq_len, prefix_len] points, shortest prefix first, so a
        fault at a long prefix leaves the shorter points measured.
        """
        native_heads = {
            spec.model_path: spec.native_num_heads
            for spec in get_mla_module_model_specs(attention_type="dsa", apply_model_filter=False)
        }
        tasks: dict[tuple, list[_Run]] = {}
        for run in runs:
            batch = run.batch_size if phase == "context" else 0
            precision = (run.kv_cache_dtype, run.compute_dtype, run.gemm_type)
            tasks.setdefault((batch, run.num_heads, *precision, run.model_path, run.dispatch), []).append(run)
        rows = []
        for (batch, heads, kv, compute, gemm, model, backend), members in tasks.items():
            points = sorted(
                ([run.batch_size, run.seq_len, run.prefix_len] for run in members), key=lambda p: (p[2], p[0], p[1])
            )
            pinned = backend if kv == "fp8" else None
            row = [0, batch, heads, kv, compute, gemm, model, "dsa", None, native_heads[model] // heads, pinned, points]
            rows.append((row, tuple(members)))
        return rows

    return rows


def _dsa_module_plan(phase: str, *, skip_indexer: bool = False) -> OpPlan:
    """A DSA module op. The skip_indexer ops (SGLang only) time the checkpoints whose layers
    share one top-k index, with the indexer left out."""
    batch_rows = {"sglang": _sglang_dsa_rows(phase)}
    if not skip_indexer:
        batch_rows.update(trtllm=_trtllm_module_rows("dsa"), vllm=_vllm_module_rows("dsa"))
    return OpPlan(
        ModuleCase,
        _sparse_module_expand("dsa", phase, shared_index_only=skip_indexer),
        dtype_field=("kv_cache_dtype", "compute_dtype", "gemm_type"),
        group_fields=_MODULE_GROUP,
        dispatch={"sglang": _sglang_dsa_backend(phase)},
        batch_rows=batch_rows,
    )


def _sglang_msa_rows(phase: str) -> Callable[[list, int | None], list[tuple[list, tuple]]]:
    def rows(cases: list[ModuleCase], sm_version: int | None) -> list[tuple[list, tuple]]:
        """One run_msa_module_worker task per ModelRunner: one context batch, or every decode batch.

        A row keeps the worker's positional fields (batch 0 for decode), then the target
        TP of the heads and the [batch, seq_len, prefix_len] points, shortest prefix
        first, so a fault at a long prefix leaves the shorter points measured.
        """
        native_heads = {
            spec.model_path: spec.native_num_heads
            for spec in get_mla_module_model_specs(attention_type="msa", apply_model_filter=False)
        }
        tasks: dict[tuple, list[ModuleCase]] = {}
        for case in cases:
            batch = case.batch_size if phase == "context" else 0
            precision = (case.kv_cache_dtype, case.compute_dtype, case.gemm_type)
            tasks.setdefault((batch, case.num_heads, *precision, case.model_path), []).append(case)
        rows = []
        for (batch, heads, kv, compute, gemm, model), members in tasks.items():
            points = sorted(
                ([case.batch_size, case.seq_len, case.prefix_len] for case in members), key=lambda p: (p[2], p[0], p[1])
            )
            rows.append(
                ([0, batch, heads, kv, compute, gemm, model, native_heads[model] // heads, points], tuple(members))
            )
        return rows

    return rows


def _msa_module_plan(phase: str) -> OpPlan:
    """An MSA module op. vLLM times one point per task (run_msa_module_worker takes the case)."""
    return OpPlan(
        ModuleCase,
        _sparse_module_expand("msa", phase),
        dtype_field=("kv_cache_dtype", "compute_dtype", "gemm_type"),
        group_fields=_MODULE_GROUP,
        batch_rows={"trtllm": _trtllm_module_rows(), "sglang": _sglang_msa_rows(phase)},
    )


# --- DeepSeek-V4 attention modules: csa/hca context and generation -------------------


class Dsv4Case(NamedTuple):
    """One timed point of a DeepSeek-V4 attention module; each attention kind is its own op.

    ``seq_len`` is the query length (context) or the KV length (generation);
    ``prefix_len`` is the cached context behind a context point. Every backend serves
    DeepSeek-V4 on an FP8 KV cache and computes attention in BF16, so neither is a case
    field: the rows name them.
    """

    seq_len: int
    batch_size: int
    tp_size: int
    gemm_type: str
    model_path: str
    attn_kind: str
    prefix_len: int = 0

    __str__ = __repr__ = _list_form


def _dsv4_decode_points(sweep: Dsv4Sweep, max_position: int | None) -> list[tuple[int, int, int]]:
    """The (batch, KV length, 0) decode points within the KV-token budget and the batch ladder.

    A KV length leaves room for the decoded token.
    """
    return [
        (batch, length, 0)
        for batch in sweep.generation_batch_sizes
        for length in sweep.generation_sequence_lengths
        if batch * length <= sweep.generation_max_kv_tokens
        and not any(length >= floor and batch > largest for floor, largest in sweep.decode_batch_ladder)
        and (max_position is None or length < max_position)
    ]


def _dsv4_points(phase: str, model_path: str) -> list[tuple[int, int, int]]:
    """The (batch, seq_len, prefix_len) points of one DeepSeek-V4 module.

    Context takes the DSA and MSA modules' long-context grid; its prefix anchors keep the
    top-k knees (512 tokens for DeepSeek-V4-Flash, 1024 for DeepSeek-V4-Pro).
    """
    max_position = _max_position(model_path)
    if phase == "context":
        return _long_context_points("context", get_mla_module_sweep_spec(_MODULE_SWEEP), max_position)
    return _dsv4_decode_points(get_dsv4_sweep(), max_position)


def _dsv4_expand(attn_kind: str, phase: str) -> Callable[[], list[Dsv4Case]]:
    def expand() -> list[Dsv4Case]:
        """Every (projection GEMM, TP shard, point) of each planned checkpoint."""
        sweep = get_dsv4_sweep()
        cases = []
        for model_path in get_dsv4_models():
            points = _dsv4_points(phase, model_path)
            for gemm_type in sweep.gemm_types:
                for tp_size in sweep.tp_sizes:
                    cases += [
                        Dsv4Case(length, batch, tp_size, gemm_type, model_path, attn_kind, prefix)
                        for batch, length, prefix in points
                    ]
        return cases

    return expand


def _dsv4_series(cases: list[Dsv4Case]) -> list[tuple[tuple, list[Dsv4Case]]]:
    """The cases grouped by the module build they share (model, kind, TP, GEMM, batch), in plan order."""
    series: dict[tuple, list[Dsv4Case]] = {}
    for case in cases:
        key = (case.model_path, case.attn_kind, case.tp_size, case.gemm_type, case.batch_size)
        series.setdefault(key, []).append(case)
    return list(series.items())


def _trtllm_dsv4_rows(cases: list[Dsv4Case], sm_version: int | None) -> list[tuple[list, tuple]]:
    """One run_dsv4_attn_series_worker task per module: its (seq_len, prefix_len) points, then
    run_dsv4_attn_worker's positional fields.

    TRT-LLM serves the FP8 KV cache as the packed fp8_ds_mla pool on SM90 and as the
    plain FP8 pool elsewhere (collect_dsv4_attn picks it per SM).
    """
    tasks = []
    for (model, kind, tp_size, gemm, batch), members in _dsv4_series(cases):
        points = sorted(([case.seq_len, case.prefix_len] for case in members), reverse=True)
        tasks.append(([points, batch, tp_size, "fp8", "bfloat16", gemm, model, kind], tuple(members)))
    return tasks


def _vllm_dsv4_rows(phase: str) -> Callable[[Dsv4Case, int | None, str | None], list[list]]:
    def rows(case: Dsv4Case, sm_version: int | None, runner: str | None) -> list[list]:
        """run_dsv4_attn_worker's arguments, one point per task; a context point adds its prefix."""
        row = [case.seq_len, case.batch_size, case.tp_size, "fp8", "bfloat16", case.gemm_type, case.model_path]
        row += [case.attn_kind, None]
        return [row + [case.prefix_len] if phase == "context" else row]

    return rows


def _sglang_dsv4_rows(cases: list[Dsv4Case], sm_version: int | None) -> list[tuple[list, tuple]]:
    """One run_dsv4_attn_worker task per ModelRunner (model, kind, TP, GEMM, batch).

    A row keeps the worker's positional fields, then its points as (prefix_len,
    seq_lens) groups, shortest prefix first and longest seq_len first within one, so a
    fault at a long prefix leaves the shorter points measured.
    """
    tasks = []
    for (model, kind, tp_size, gemm, batch), members in _dsv4_series(cases):
        seq_lens: dict[int, list[int]] = defaultdict(list)
        for case in members:
            seq_lens[case.prefix_len].append(case.seq_len)
        points = [[prefix, sorted(lengths, reverse=True)] for prefix, lengths in sorted(seq_lens.items())]
        tasks.append(([0, batch, tp_size, "fp8", "bfloat16", gemm, model, kind, None, points], tuple(members)))
    return tasks


def _dsv4_module_plan(attn_kind: str, phase: str) -> OpPlan:
    """A DeepSeek-V4 module op. vLLM times one point per task (run_dsv4_attn_worker takes it)."""
    return OpPlan(
        Dsv4Case,
        _dsv4_expand(attn_kind, phase),
        dtype_field="gemm_type",
        group_fields=("model_path", "attn_kind", "tp_size", "gemm_type", "batch_size"),
        rows={"vllm": _vllm_dsv4_rows(phase)},
        batch_rows={"trtllm": _trtllm_dsv4_rows, "sglang": _sglang_dsv4_rows},
    )


# --- DeepSeek-V4 sparse kernels: paged MQA logits and the CSA top-k calibration -------


class Dsv4SparseCase(NamedTuple):
    """One point of the CSA indexer's paged MQA logits kernel, in run_dsv4_sparse_kernel_worker's
    (vLLM) order: batch, new tokens, cached tokens, TP, the kernel, then the checkpoint."""

    batch_size: int
    seq_len: int
    prefix_len: int
    tp_size: int
    kernel: str
    model_path: str

    __str__ = __repr__ = _list_form

    @property
    def dtype(self) -> str:
        """The kernel reads FP8 queries and an FP8 indexer cache."""
        return "fp8"

    @property
    def variant(self) -> None:
        """The kernel has one variant."""
        return None


def _expand_dsv4_paged_mqa_logits() -> list[Dsv4SparseCase]:
    """The sparse lists of each planned checkpoint: batch x new tokens within the prefill chunk,
    batch x (new + cached tokens) within the KV-token budget."""
    sweep = get_dsv4_sweep()
    return [
        Dsv4SparseCase(batch, length, prefix, tp_size, "paged_mqa_logits", model_path)
        for model_path in get_dsv4_models()
        for tp_size in sweep.sparse_tp_sizes
        for batch in sweep.sparse_batch_sizes
        for length in sweep.sparse_input_lengths
        if batch * length <= sweep.sparse_chunk_prefill_size
        for prefix in sweep.sparse_past_kv_lengths
        if batch * (length + prefix) <= sweep.sparse_max_full_sequence_length
    ]


class Dsv4TopkCase(NamedTuple):
    """One CSA module point of the top-k calibration, with the top-k kernel serving runs there:
    v1 for a prefill (it also emits the raw indices), v2 for a decode step."""

    batch_size: int
    seq_len: int
    prefix_len: int
    variant: str
    model_path: str

    __str__ = __repr__ = _list_form

    @property
    def tp_size(self) -> int:
        """The top-k kernel scans each token's compressed KV whole: TP does not shard it."""
        return 1


def _expand_dsv4_topk_calib() -> list[Dsv4TopkCase]:
    """The CSA module plan's points of each calibrated checkpoint, batch by batch: v1 on the
    context points, v2 on the decode points, where one new token follows the KV length.

    Consumers key the DELTA per native head geometry, so each checkpoint gets its own.
    """
    cases = []
    for model_path in get_dsv4_models():
        context = [Dsv4TopkCase(b, s, p, "v1", model_path) for b, s, p in _dsv4_points("context", model_path)]
        decode = [Dsv4TopkCase(b, 1, kv, "v2", model_path) for b, kv, _p in _dsv4_points("generation", model_path)]
        cases += sorted(context + decode, key=lambda case: case.batch_size)
    return cases


def _sglang_dsv4_kernel_rows(kernel: str) -> Callable[[list, int | None], list[tuple[list, tuple]]]:
    def rows(cases: list, sm_version: int | None) -> list[tuple[list, tuple]]:
        """One run_dsv4_sparse_kernel_worker task per (model, TP, batch): the checkpoint, the
        kernel, the batch and TP, then its [prefix_len, seq_len, variant] points."""
        tasks: dict[tuple, list] = {}
        for case in cases:
            tasks.setdefault((case.model_path, case.batch_size, case.tp_size), []).append(case)
        return [
            ([model, kernel, batch, tp_size, [[c.prefix_len, c.seq_len, c.variant] for c in members]], tuple(members))
            for (model, batch, tp_size), members in tasks.items()
        ]

    return rows


# --- GLM-5 sparse kernels: MQA logits, top-k and sparse attention (SGLang) -----------

# The checkpoints whose geometry the kernels read: the BF16 GLM-5, and the first declared
# NVFP4 one (GLM-5.2-NVFP4 reaches 1M tokens). collector/skips.py picks one per SM.
_GLM5_SPARSE_MODELS = ("zai-org/GLM-5", ("nvidia/GLM-5.2-NVFP4", "nvidia/GLM-5-NVFP4"))


class Glm5SparseCase(NamedTuple):
    """One shape of a GLM-5 DSA sparse kernel: batch, new tokens and cached tokens.

    The shapes are the DSA module plan's points of the checkpoint; a decode step is
    one new token over its KV length.
    """

    batch_size: int
    seq_len: int
    prefix_len: int
    kernel: str
    model_path: str

    __str__ = __repr__ = _list_form

    @property
    def dtype(self) -> str:
        """The sparse attention reads BF16 queries and KV; the indexer kernels FP8 keys."""
        return "bfloat16" if self.kernel == "dsa_attn" else "fp8"


def _glm5_sparse_models() -> list[str]:
    declared = {spec.model_path for spec in get_mla_module_model_specs(attention_type="dsa")}
    models = []
    for choice in _GLM5_SPARSE_MODELS:
        options = choice if isinstance(choice, tuple) else (choice,)
        models += [model for model in options if model in declared][:1]
    return models


def _glm5_sparse_expand(kernel: str) -> Callable[[], list[Glm5SparseCase]]:
    def expand() -> list[Glm5SparseCase]:
        """The DSA module plan's context and decode points of each checkpoint, batch by batch."""
        sweep = get_mla_module_sweep_spec(_MODULE_SWEEP)
        cases = []
        for model_path in _glm5_sparse_models():
            max_position = _max_position(model_path)
            shapes = {(b, s, p) for b, s, p in _long_context_points("context", sweep, max_position)}
            shapes |= {(b, 1, kv) for b, kv, _p in _long_context_points("generation", sweep, max_position)}
            cases += [
                Glm5SparseCase(batch, length, prefix, kernel, model_path)
                for batch, length, prefix in sorted(shapes, key=lambda shape: (shape[0], shape[2], shape[1]))
            ]
        return cases

    return expand


def _sglang_glm5_sparse_rows(cases: list, sm_version: int | None) -> list[tuple[list, tuple]]:
    """One run_glm5_dsa_sparse_kernel_worker task per (model, batch): the checkpoint, the kernel,
    the batch, then its [prefix_len, seq_len] points, shortest prefix first."""
    tasks: dict[tuple, list] = {}
    for case in cases:
        tasks.setdefault((case.model_path, case.kernel, case.batch_size), []).append(case)
    return [
        ([model, kernel, batch, [[c.prefix_len, c.seq_len] for c in members]], tuple(members))
        for (model, kernel, batch), members in tasks.items()
    ]


def _glm5_sparse_plan(kernel: str) -> OpPlan:
    return OpPlan(
        Glm5SparseCase,
        _glm5_sparse_expand(kernel),
        dtype_field="dtype",
        group_fields=("model_path", "batch_size"),
        batch_rows={"sglang": _sglang_glm5_sparse_rows},
    )


# --- DeepSeek-V4.1 native components: dsv41_module (SGLang) --------------------------


class Dsv41Case(NamedTuple):
    """One native DeepSeek-V4.1 collection: a checkpoint at one TP with one execution profile.

    The profiles write separate tables, so a run collects one of them
    (``--case-filter dsv41_<profile>_``).
    """

    model_path: str
    tp_size: int
    execution_profile: str

    __str__ = __repr__ = _list_form


def _expand_dsv41() -> list[Dsv41Case]:
    """Every TP and execution profile each declared checkpoint lists."""
    return [
        Dsv41Case(value["model_path"], tp_size, profile)
        for value in _framework_specific_model_case_values("dsv41_module", "sglang")
        for tp_size in value["tensor_parallel_sizes"]
        for profile in value["execution_profiles"]
    ]


def _sglang_dsv41_rows(case: Dsv41Case, sm_version: int | None, runner: str | None) -> list[dict]:
    """The native runner's {id, params} row: checkpoint, TP, profile and the workload sweep."""
    sweep = get_base_common_case_values("dsv41_module")
    if not sweep:
        raise ValueError("missing V4.1 workload sweep")
    model = case.model_path.replace("/", "_")
    return [
        {
            "id": f"dsv41_{case.execution_profile}_tp{case.tp_size}_{model}",
            "params": [case.model_path, case.tp_size, case.execution_profile, sweep],
        }
    ]


# --- mHC: mhc_module ---------------------------------------------------------------

_DSV4_ARCHITECTURE = "DeepseekV4ForCausalLM"
# The mHC call sites each architecture's serving forward runs. GLM-5.3-Flash also
# fuses the previous sublayer's post into the next pre and expands/contracts the
# residual stream once per forward (collector/vllm/collect_mhc_module.py).
#
# DeepSeek-V4 is collected as separate pre (no fused norm) and post on purpose,
# although serving fuses post + next pre (+ norm) at every layer boundary but the
# first and last, on all three frameworks. Measured on H20 (hc_mult 4, hidden 4096,
# 1k/8k tokens, 2026-07): fused post+pre+norm is within 1-2% of pre + post, and the
# fused norm adds 2-3% to pre; the SDK's DeepSeekV4 model prices pre + attn_norm +
# post per layer. Collecting fused_post_pre for DeepSeek-V4 is a producer + consumer
# contract change (SDK pricing, aiconfigurator#1580), not a pin-bump item: a pin
# bump keeps these two sites and only notes framework-specific gaps at the
# collectors. Reopen it with an end-to-end measurement that shows the gap.
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
        if includes_long_context():
            for profile in _model_case_values("attention"):
                if not profile.get("long_context_cases"):
                    continue
                points = long_context_points(profile, phase, _history_limit(profile["model_path"]))
                if any(prefix for _batch, _length, prefix in points):
                    raise ValueError("dense long_context_cases use full lookup sequence with prefix=0")
                for num_heads, num_kv_heads, head_dim, window_size in _attention_profile_shapes(profile, sweep, phase):
                    for batch, length, _prefix in points:
                        # Preserve the existing universal native indexing bound.
                        sequence = length if phase == "context" else length + 1
                        if batch * sequence * num_kv_heads * head_dim * 2 >= 2**31 - 1:
                            continue
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
        return list(dict.fromkeys(cases))

    return expand


def _attention_owners(phase: str, case: AttentionCase) -> tuple[AttentionOwner, ...]:
    """The profiles that plan a case's shape, models first (a model's semantics win over the base grid's)."""
    owners = get_attention_shapes(phase)[(case.num_heads, case.num_kv_heads, case.head_dim, case.window_size)]
    return tuple(sorted(owners, key=lambda owner: not owner.from_model))


def _sglang_attention_dispatch(phase: str) -> Callable[[AttentionCase, int | None], tuple[str, ...]]:
    """The SGLang backends serving runs a case on: one per model that plans the shape.

    A shape two models serve on different backends (on SM100, Qwen3.5 on
    triton and Qwen3.8 on trtllm_mha) is measured on both.
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
    # MiMo-V2-Flash's head_dim 192 rows with V=192 on the native FLASH_ATTN
    # path, where serving runs V=128 on FLASH_ATTN_DIFFKV or TRITON_ATTN_DIFFKV
    # (vllm model_executor/models/mimo_v2.py:300-321), and GPT-OSS's rows
    # without attention sinks.
    # A shape that model profiles plan names them as a last argument, one
    # [architecture, model] pair per profile. vLLM applies a model's own
    # attention decisions through a per-architecture config hook
    # (model_executor/models/config.py MODELS_CONFIG_MAP, applied by
    # VllmConfig.try_verify_and_update_config, config/vllm.py:2629-2658 at
    # v0.32.0); for the owners that have one, the collector builds its
    # VllmConfig from the model's config so the hook decides as in serving
    # (vllm/collect_attn.py models_with_config_hook). A shape only the base
    # grid plans names none and keeps the generic config, as do owners
    # without a hook.
    def rows(case: AttentionCase, sm_version: int | None, runner: str | None) -> list[list]:
        row = [
            case.batch_size,
            case.input_len,
            case.num_heads,
            case.num_kv_heads,
            case.head_dim,
            case.kv_cache_dtype == "fp8",
            is_context,
            case.window_size,
        ]
        owners = [owner for owner in _attention_owners(phase, case) if owner.from_model]
        if any(owner.model_path is None or owner.architecture is None for owner in owners):
            raise ValueError(f"an attention profile that plans {case} names no architecture or no model_path")
        named = [list(pair) for pair in dict.fromkeys((owner.architecture, owner.model_path) for owner in owners)]
        return [[*row, named]] if named else [row]

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
    "moe_ep": OpPlan(
        MoeEpCase,
        _expand_moe_ep,
        dtype_field="moe_type",
        # Every case builds its own layer.
        group_fields=MoeEpCase._fields,
        rows={"sglang": _sglang_moe_ep_rows, "trtllm": _trtllm_moe_ep_rows},
    ),
    "deepep_ll": OpPlan(
        DeepepCase, _expand_deepep, dtype_field=(), group_fields=DeepepCase._fields, rows={"sglang": _deepep_rows}
    ),
    "deepep_normal": OpPlan(
        DeepepCase, _expand_deepep, dtype_field=(), group_fields=DeepepCase._fields, rows={"sglang": _deepep_rows}
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
    "mla_context_module": _mla_module_plan("context"),
    "mla_generation_module": _mla_module_plan("generation"),
    "dsa_context_module": _dsa_module_plan("context"),
    "dsa_generation_module": _dsa_module_plan("generation"),
    "dsa_context_module_skip_indexer": _dsa_module_plan("context", skip_indexer=True),
    "dsa_generation_module_skip_indexer": _dsa_module_plan("generation", skip_indexer=True),
    "msa_context_module": _msa_module_plan("context"),
    "msa_generation_module": _msa_module_plan("generation"),
    "dsv4_csa_context_module": _dsv4_module_plan("csa", "context"),
    "dsv4_csa_generation_module": _dsv4_module_plan("csa", "generation"),
    "dsv4_hca_context_module": _dsv4_module_plan("hca", "context"),
    "dsv4_hca_generation_module": _dsv4_module_plan("hca", "generation"),
    "dsv4_paged_mqa_logits_module": OpPlan(
        Dsv4SparseCase,
        _expand_dsv4_paged_mqa_logits,
        dtype_field="dtype",
        group_fields=("model_path", "tp_size", "batch_size"),
        batch_rows={"sglang": _sglang_dsv4_kernel_rows("paged_mqa_logits")},
    ),
    "dsv4_csa_topk_calib": OpPlan(
        Dsv4TopkCase,
        _expand_dsv4_topk_calib,
        dtype_field="variant",
        group_fields=("model_path", "batch_size"),
        batch_rows={"sglang": _sglang_dsv4_kernel_rows("topk")},
    ),
    "dsv41_module": OpPlan(
        Dsv41Case,
        _expand_dsv41,
        dtype_field="execution_profile",
        group_fields=Dsv41Case._fields,
        rows={"sglang": _sglang_dsv41_rows},
    ),
    "glm5_mqa_logits_module": _glm5_sparse_plan("mqa"),
    "glm5_topk_module": _glm5_sparse_plan("topk"),
    "glm5_dsa_attn_module": _glm5_sparse_plan("dsa_attn"),
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
