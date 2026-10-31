# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Op plans: one case list per op, shared by every CUDA backend.

An op plan expands the op's cases once from ``cases/`` YAML, the same for
every backend. :func:`plan` then leaves out only whole regions: the
``capabilities.yaml`` floors and backend gaps (:func:`capabilities.region_reason`).
A case inside a kept region always runs; if it fails, the failure is recorded.

A backend whose run function takes more arguments than the shared case
declares ``rows``: the run arguments for one case. Rows decide how a case
runs, never whether it runs, so every case gives at least one row.

Each backend's getter is one line, ``return plan_cases("<op>", "<backend>",
get_sm_version())``. To move an op onto a plan, add its case type, its expand
function and an :class:`OpPlan` entry in ``OP_PLANS``.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import NamedTuple

from collector.capabilities import region_reason
from collector.case_generator import (
    BASE_OP_CASES_DIR,
    MoeCommonTestCase,
    get_common_moe_test_cases,
    get_gemm_case_specs,
    get_gemm_type_specs,
    get_moe_quantization_module_config,
    get_sglang_moe_backend,
    moe_model_layer,
    moe_planned_modes,
)


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
    # Other case fields a capabilities.yaml backend gap may name, besides dtype.
    region_fields: tuple[str, ...] = ()
    # backend -> the run arguments for one case, for a run function that takes
    # more than the case.
    rows: Mapping[str, Callable[[object, int | None], list]] = MappingProxyType({})


@dataclasses.dataclass(frozen=True)
class RemovedRegion:
    """A whole region a plan leaves out, how many cases it holds, and why."""

    region: str
    cases: int
    reason: str


@dataclasses.dataclass(frozen=True)
class Plan:
    """The cases one backend runs for an op on one SM."""

    cases: list
    # What the getter queues: the cases, or the backend's rows for them.
    rows: list
    removed: list[RemovedRegion]


def plan(op: str, backend: str, sm_version: int | None) -> Plan:
    """Expand an op's shared cases and leave out the whole regions this backend cannot run on this SM."""
    spec = OP_PLANS[op]
    reasons: dict[tuple, tuple[str, str] | None] = {}
    kept = []
    removed: Counter[tuple[str, str]] = Counter()
    for case in spec.expand():
        region = {"dtype": getattr(case, spec.dtype_field)}
        region.update((field, getattr(case, field)) for field in spec.region_fields)
        key = tuple(region.values())
        if key not in reasons:
            reasons[key] = region_reason(op, region, backend, sm_version)
        if reasons[key] is None:
            kept.append(case)
        else:
            removed[reasons[key]] += 1

    rows = kept
    to_rows = spec.rows.get(backend)
    if to_rows is not None:
        rows = []
        for case in kept:
            case_rows = to_rows(case, sm_version)
            if not case_rows:
                raise ValueError(f"{backend} {op} rows dropped case {case}; rows may only say how a case runs")
            rows.extend(case_rows)
    return Plan(
        cases=kept,
        rows=rows,
        removed=[RemovedRegion(region, count, reason) for (region, reason), count in removed.items()],
    )


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
        "plan_hash": case_plan_hash([case_id(row) for row in op_plan.rows]),
    }
    if len(op_plan.rows) != len(op_plan.cases):
        summary["rows"] = len(op_plan.rows)
    return summary


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

    def __str__(self) -> str:
        return str(list(self))

    __repr__ = __str__


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

    def __str__(self) -> str:
        return str([list(value) if isinstance(value, tuple) else value for value in self])

    __repr__ = __str__

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
        print(f"moe: {len(empty)} model row(s) plan no case (quantization_modes: []): {', '.join(empty)}")
    return cases


def _moe_layer_of(case: MoeCase) -> MoeCommonTestCase:
    return moe_model_layer(case.model_name, case.hidden_size, case.inter_size, case.topk, case.num_experts)


def _sglang_moe_rows(case: MoeCase, sm_version: int | None) -> list[list]:
    """SGLang's run_moe_torch also takes the row's routing and activation and the runner serving picks on this SM."""
    layer = _moe_layer_of(case)
    return [
        [
            case.moe_type,
            list(case.num_tokens),
            *case[2:11],
            layer.sglang_moe_swiglu_limit,
            get_sglang_moe_backend(layer, case.moe_type, sm_version),
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


def _trtllm_moe_rows(case: MoeCase, sm_version: int | None) -> list[list]:
    """TRT-LLM's run_moe_torch takes min_latency_mode after ep.

    An nvfp4 case on SM100/103 with at most 256 experts also times the
    min-latency (trtllm-gen) kernel, a second row of moe_perf
    (kernel_source moe_torch_flow_min_latency); its DeepSeek router supports
    at most 256 experts.
    """
    min_latency_modes = [False]
    if case.moe_type == "nvfp4" and sm_version in (100, 103) and case.num_experts <= 256:
        min_latency_modes.append(True)
    head = [case.moe_type, list(case.num_tokens), *case[2:8]]
    return [[*head, min_latency, *case[8:11]] for min_latency in min_latency_modes]


OP_PLANS: dict[str, OpPlan] = {
    "gemm": OpPlan(GemmCase, _expand_gemm, dtype_field="gemm_type", group_fields=("gemm_type", "n", "k")),
    "moe": OpPlan(
        MoeCase,
        _expand_moe,
        dtype_field="moe_type",
        # Every case builds its own layer.
        group_fields=MoeCase._fields,
        region_fields=("parallel",),
        rows={"sglang": _sglang_moe_rows, "trtllm": _trtllm_moe_rows},
    ),
}
