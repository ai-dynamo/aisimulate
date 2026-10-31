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
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import NamedTuple

from collector.capabilities import region_reason
from collector.case_generator import BASE_OP_CASES_DIR, get_gemm_case_specs, get_gemm_type_specs


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


OP_PLANS: dict[str, OpPlan] = {
    "gemm": OpPlan(GemmCase, _expand_gemm, dtype_field="gemm_type", group_fields=("gemm_type", "n", "k")),
}
