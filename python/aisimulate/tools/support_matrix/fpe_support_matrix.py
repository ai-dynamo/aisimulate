# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict-native Forward Pass Engine support-matrix probes.

This module deliberately stops at the FPE boundary.  It compiles one native
engine per resolved identity and runs representative prefill, decode, and
mixed-step calls.  It does not run the CLI, Sweeper, scheduler simulation,
Replay, or disaggregated rate matching, so its output must not be presented as
full AISimulate or deployment support.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import resource
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from itertools import groupby
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

STATUS_PASS = "PASS"
STATUS_SDK_UNREPRESENTABLE = "SDK_UNREPRESENTABLE"
STATUS_PERF_DATA_MISSING = "PERF_DATA_MISSING"
STATUS_MODEL_UNSUPPORTED = "MODEL_UNSUPPORTED"
STATUS_HW_INCOMPATIBLE = "HW_INCOMPATIBLE"
STATUS_FRAMEWORK_INCOMPATIBLE = "FRAMEWORK_INCOMPATIBLE"
STATUS_BUILD_FAILED = "BUILD_FAILED"
STATUS_QUERY_FAILED = "QUERY_FAILED"

VALID_STATUSES = frozenset(
    {
        STATUS_PASS,
        STATUS_SDK_UNREPRESENTABLE,
        STATUS_PERF_DATA_MISSING,
        STATUS_MODEL_UNSUPPORTED,
        STATUS_HW_INCOMPATIBLE,
        STATUS_FRAMEWORK_INCOMPATIBLE,
        STATUS_BUILD_FAILED,
        STATUS_QUERY_FAILED,
    }
)

ROLE_ORDER = {"agg": 0, "prefill": 1, "decode": 2}
PHASE_ORDER = {"prefill": 0, "decode_start": 1, "decode_end": 2, "mixed": 3}


@dataclass(frozen=True, order=True)
class ParallelTopology:
    """One public FPE parallel identity."""

    tp_size: int
    pp_size: int
    attention_dp_size: int
    moe_tp_size: int
    moe_ep_size: int
    cp_size: int

    @classmethod
    def from_choice(cls, choice: Sequence[int]) -> ParallelTopology:
        if len(choice) != 6:
            raise ValueError(f"parallel choice must have 6 entries, got {choice!r}")
        return cls(*(int(value) for value in choice))


@dataclass(frozen=True)
class ProbeWorkload:
    """Representative forward-pass shapes used for every engine identity."""

    isl: int = 256
    osl: int = 256
    prefix: int = 128
    prefill_batch: int = 1
    decode_batch: int = 8
    mixed_context_tokens: int = 256
    mixed_decode_tokens: int = 8

    def validate(self) -> None:
        positive = {
            "isl": self.isl,
            "osl": self.osl,
            "prefill_batch": self.prefill_batch,
            "decode_batch": self.decode_batch,
            "mixed_context_tokens": self.mixed_context_tokens,
            "mixed_decode_tokens": self.mixed_decode_tokens,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.prefix < 0 or self.prefix > self.isl:
            raise ValueError(f"prefix must be in [0, isl], got prefix={self.prefix}, isl={self.isl}")


@dataclass(frozen=True)
class ProbeCall:
    phase: str
    method: str
    kwargs: dict[str, int]


@dataclass(frozen=True)
class EngineProbePlan:
    """A unique strict-native engine build plus the roles that consume it."""

    model: str
    architecture: str
    system: str
    backend: str
    backend_version: str
    forward_model: str
    topology: ParallelTopology
    roles: tuple[str, ...]
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None
    nextn: int = 0
    attention_backend: str | None = None
    unrepresentable_reasons: tuple[str, ...] = ()
    planning_status: str = ""
    planning_error_type: str = ""
    planning_error_message: str = ""

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.system,
            self.backend,
            self.backend_version,
            self.model,
            self.forward_model,
            self.topology,
            self.roles,
        )

    def compile_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "backend_version": self.backend_version,
            "tp_size": self.topology.tp_size,
            "pp_size": self.topology.pp_size,
            "attention_dp_size": self.topology.attention_dp_size,
            "moe_tp_size": self.topology.moe_tp_size,
            "moe_ep_size": self.topology.moe_ep_size,
            "nextn": self.nextn,
            "forward_model": self.forward_model,
        }
        optional = {
            "gemm_quant_mode": self.gemm_quant_mode,
            "moe_quant_mode": self.moe_quant_mode,
            "kvcache_quant_mode": self.kvcache_quant_mode,
            "fmha_quant_mode": self.fmha_quant_mode,
            "comm_quant_mode": self.comm_quant_mode,
            "attention_backend": self.attention_backend,
        }
        kwargs.update({name: value for name, value in optional.items() if value is not None})
        return kwargs

    def identity_key(self) -> tuple[Any, ...]:
        return (
            self.model,
            self.architecture,
            self.system,
            self.backend,
            self.backend_version,
            self.forward_model,
            self.topology,
            self.gemm_quant_mode,
            self.moe_quant_mode,
            self.kvcache_quant_mode,
            self.fmha_quant_mode,
            self.comm_quant_mode,
            self.nextn,
            self.attention_backend,
            self.unrepresentable_reasons,
            self.planning_status,
            self.planning_error_type,
            self.planning_error_message,
        )


@dataclass(frozen=True)
class FPEProbeResult:
    model: str
    architecture: str
    system: str
    backend: str
    backend_version: str
    forward_model: str
    roles: str
    tp_size: int
    pp_size: int
    attention_dp_size: int
    moe_tp_size: int
    moe_ep_size: int
    cp_size: int
    gemm_quant_mode: str
    moe_quant_mode: str
    kvcache_quant_mode: str
    fmha_quant_mode: str
    comm_quant_mode: str
    nextn: int
    phase: str
    status: str
    latency_ms: float | None
    source: str
    failure_stage: str
    error_type: str
    error_message: str
    reproducer: str
    source_version: str
    source_sha: str
    attention_backend: str | None = None

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.model,
            self.architecture,
            self.system,
            self.backend,
            self.backend_version,
            self.forward_model,
            self.tp_size,
            self.pp_size,
            self.attention_dp_size,
            self.moe_tp_size,
            self.moe_ep_size,
            self.cp_size,
            PHASE_ORDER[self.phase],
            self.roles,
        )


@dataclass(frozen=True)
class MatrixRunMetrics:
    wall_time_seconds: float
    cpu_time_seconds: float
    peak_rss_kib: int
    max_workers: int
    plan_count: int
    result_count: int


def _enum_token(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _role_attr(task: Any, role: str, name: str) -> Any:
    return getattr(task, name if role == "agg" else f"{role}_{name}")


def _unrepresentable_reasons(model_config: Any, topology: ParallelTopology) -> tuple[str, ...]:
    """Fail closed when the stable public builder cannot encode a Task choice."""
    reasons: list[str] = []
    if topology.cp_size != 1:
        reasons.append("public EngineHandle.compile does not expose cp_size")
    if getattr(model_config, "moe_comm_backend", None):
        reasons.append("public EngineHandle.compile does not expose moe_comm_backend")
    if getattr(model_config, "enable_eplb", False):
        reasons.append("public EngineHandle.compile does not expose enable_eplb")
    if getattr(model_config, "moe_backend", None):
        reasons.append("public EngineHandle.compile does not expose moe_backend")
    if getattr(model_config, "language_only", False):
        reasons.append("public EngineHandle.compile does not expose language_only")
    if not getattr(model_config, "enable_encoder_dp", True):
        reasons.append("public EngineHandle.compile does not expose enable_encoder_dp=False")
    return tuple(sorted(set(reasons)))


def _make_plan(
    *,
    task: Any,
    role: str,
    topology: ParallelTopology,
    architecture: str,
    forward_model: str,
) -> EngineProbePlan:
    choice = (
        topology.tp_size,
        topology.pp_size,
        topology.attention_dp_size,
        topology.moe_tp_size,
        topology.moe_ep_size,
        topology.cp_size,
    )
    model_config = task.build_model_config(role=role, parallel=choice)
    return EngineProbePlan(
        model=str(_role_attr(task, role, "model_path")),
        architecture=architecture,
        system=str(_role_attr(task, role, "system_name")),
        backend=str(_role_attr(task, role, "backend_name")),
        backend_version=str(_role_attr(task, role, "backend_version")),
        forward_model=forward_model,
        topology=topology,
        roles=(role,),
        gemm_quant_mode=_enum_token(getattr(model_config, "gemm_quant_mode", None)),
        moe_quant_mode=_enum_token(getattr(model_config, "moe_quant_mode", None)),
        kvcache_quant_mode=_enum_token(getattr(model_config, "kvcache_quant_mode", None)),
        fmha_quant_mode=_enum_token(getattr(model_config, "fmha_quant_mode", None)),
        comm_quant_mode=_enum_token(getattr(model_config, "comm_quant_mode", None)),
        nextn=int(getattr(model_config, "nextn", 0) or 0),
        attention_backend=(
            str(model_config.attention_backend)
            if getattr(model_config, "attention_backend", None) is not None
            else None
        ),
        unrepresentable_reasons=_unrepresentable_reasons(model_config, topology),
    )


def merge_equivalent_plans(plans: Iterable[EngineProbePlan]) -> list[EngineProbePlan]:
    """Compile an identical engine once even if agg/prefill/decode share it."""
    merged: dict[tuple[Any, ...], tuple[EngineProbePlan, set[str]]] = {}
    for plan in plans:
        key = plan.identity_key()
        if key not in merged:
            merged[key] = (plan, set(plan.roles))
        else:
            merged[key][1].update(plan.roles)
    result = []
    for plan, roles in merged.values():
        ordered_roles = tuple(sorted(roles, key=ROLE_ORDER.__getitem__))
        result.append(replace(plan, roles=ordered_roles))
    return sorted(result, key=EngineProbePlan.sort_key)


def build_probe_plans(
    *,
    models: set[str] | None = None,
    systems: set[str] | None = None,
    backends: set[str] | None = None,
    backend_versions: set[str] | None = None,
    forward_models: Sequence[str] = ("op_level",),
    max_topologies_per_role: int | None = None,
    matrix: Any | None = None,
    create_task: Callable[..., Any] | None = None,
    constraints_for_model: Callable[[str], Any] | None = None,
) -> list[EngineProbePlan]:
    """Resolve the live curated inventory into strict public-FPE plans."""
    unsupported_forward_models = set(forward_models) - {"op_level"}
    if unsupported_forward_models:
        raise ValueError(f"unsupported forward models: {sorted(unsupported_forward_models)}")
    if max_topologies_per_role is not None and max_topologies_per_role <= 0:
        raise ValueError("max_topologies_per_role must be positive when provided")

    if matrix is None or create_task is None or constraints_for_model is None:
        from tools.support_matrix.support_matrix import SupportMatrix, _get_test_constraints

        matrix = matrix or SupportMatrix()
        create_task = create_task or SupportMatrix._create_task
        constraints_for_model = constraints_for_model or _get_test_constraints

    combinations = sorted(matrix.generate_combinations())
    seeds: list[EngineProbePlan] = []
    for model, system, backend, version in combinations:
        if models is not None and model not in models:
            continue
        if systems is not None and system not in systems:
            continue
        if backends is not None and backend not in backends:
            continue
        if backend_versions is not None and version not in backend_versions:
            continue

        architecture = matrix.get_architecture(model)
        constraints = constraints_for_model(model)
        for forward_model in forward_models:
            for mode, roles in (("agg", ("agg",)), ("disagg", ("prefill", "decode"))):
                mode_seeds: list[EngineProbePlan] = []
                try:
                    task = create_task(
                        mode=mode,
                        model=model,
                        system=system,
                        backend=backend,
                        version=version,
                        constraints=constraints,
                        database_mode="SILICON",
                    )
                    task.forward_model = forward_model
                    for role in roles:
                        choices = sorted({tuple(choice) for choice in task.iter_parallel(role)})
                        if max_topologies_per_role is not None:
                            choices = choices[:max_topologies_per_role]
                        for choice in choices:
                            topology = ParallelTopology.from_choice(choice)
                            try:
                                plan = _make_plan(
                                    task=task,
                                    role=role,
                                    topology=topology,
                                    architecture=architecture,
                                    forward_model=forward_model,
                                )
                            except Exception as error:
                                # One rejected choice must not erase valid earlier
                                # choices or stop discovery of later topologies.
                                plan = EngineProbePlan(
                                    model=model,
                                    architecture=architecture,
                                    system=system,
                                    backend=backend,
                                    backend_version=version,
                                    forward_model=forward_model,
                                    topology=topology,
                                    roles=(role,),
                                    planning_status=classify_failure(error, stage="build"),
                                    planning_error_type=type(error).__name__,
                                    planning_error_message=_error_message(error),
                                )
                            mode_seeds.append(plan)
                except Exception as error:
                    mode_seeds = [
                        EngineProbePlan(
                            model=model,
                            architecture=architecture,
                            system=system,
                            backend=backend,
                            backend_version=version,
                            forward_model=forward_model,
                            topology=ParallelTopology(1, 1, 1, 1, 1, 1),
                            roles=roles,
                            planning_status=classify_failure(error, stage="build"),
                            planning_error_type=type(error).__name__,
                            planning_error_message=_error_message(error),
                        )
                    ]
                seeds.extend(mode_seeds)
    return merge_equivalent_plans(seeds)


def probe_calls_for_roles(roles: Sequence[str], workload: ProbeWorkload) -> list[ProbeCall]:
    workload.validate()
    phases: set[str] = set()
    for role in roles:
        if role == "agg":
            phases.update(PHASE_ORDER)
        elif role == "prefill":
            phases.add("prefill")
        elif role == "decode":
            phases.update(("decode_start", "decode_end"))
        else:
            raise ValueError(f"unsupported FPE role: {role!r}")

    calls = {
        "prefill": ProbeCall(
            phase="prefill",
            method="predict_prefill_latency",
            kwargs={"bs": workload.prefill_batch, "isl": workload.isl, "prefix": workload.prefix},
        ),
        "decode_start": ProbeCall(
            phase="decode_start",
            method="predict_decode_latency",
            kwargs={"bs": workload.decode_batch, "isl": workload.isl, "osl": 2},
        ),
        "decode_end": ProbeCall(
            phase="decode_end",
            method="predict_decode_latency",
            kwargs={"bs": workload.decode_batch, "isl": workload.isl, "osl": workload.osl},
        ),
        "mixed": ProbeCall(
            phase="mixed",
            method="mixed_step_latency",
            kwargs={
                "ctx_tokens": workload.mixed_context_tokens,
                "gen_tokens": workload.mixed_decode_tokens,
                "isl": workload.isl,
                "osl": workload.osl,
                "prefix": workload.prefix,
            },
        ),
    }
    return [calls[phase] for phase in sorted(phases, key=PHASE_ORDER.__getitem__)]


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def classify_failure(error: BaseException, *, stage: str) -> str:
    """Map native/Python failures to stable, conservative matrix categories."""
    chain = _exception_chain(error)
    # Explicit SDK preflights from moe_comm_resolver and models/base, moe,
    # hybrid_moe, gemma4, deepseek_v41. Match type and complete diagnostic only while building;
    # unrelated or query-time failures must retain their blocking category.
    topology_rejections = (
        (
            ValueError,
            r"Cross-node EP requires pure expert parallelism \(moe_tp_size=1\); "
            r"got moe_tp_size=\d+, moe_ep_size=\d+\.",
        ),
        (AssertionError, r"num_heads \d+ should be divisible by tp_size \d+"),
        (AssertionError, r"dense Gemma 4 variants require moe_ep_size=1, got \d+"),
        (
            ValueError,
            r"Invalid quantized MoE configuration: \(moe_intermediate_size=\d+ / moe_tp_size=\d+\) "
            r"% weight_block_size=\d+ != 0\.",
        ),
        (
            NotImplementedError,
            r"DeepSeek-V4\.1 text baseline requires attention_dp_size=1, pp_size=1 and cp_size=1; "
            r"DP Engram collectives and PP cache ownership need separate contracts",
        ),
        (
            ValueError,
            r"DeepSeek-V4\.1 EP must divide experts; TP must divide index heads and output groups",
        ),
    )
    if stage == "build" and any(
        isinstance(item, error_type) and re.fullmatch(pattern, str(item).strip())
        for item in chain
        for error_type, pattern in topology_rejections
    ):
        return STATUS_SDK_UNREPRESENTABLE
    names = " ".join(type(item).__name__ for item in chain).lower()
    messages = " ".join(str(item) for item in chain).lower()
    evidence = f"{names} {messages}"
    if "perfdata" in names or "performance data not available" in evidence or "missing perf" in evidence:
        return STATUS_PERF_DATA_MISSING
    if (
        "missingsystemflops" in names
        or "hardware incompatible" in evidence
        or "unsupported datatype" in evidence
        or ("not supported on" in evidence and any(token in evidence for token in ("hopper", "ampere", "blackwell")))
    ):
        return STATUS_HW_INCOMPATIBLE
    if "framework" in names or "framework unsupported" in evidence or "unsupported on backend" in evidence:
        return STATUS_FRAMEWORK_INCOMPATIBLE
    if "unsupportedmodel" in names or "unknown model" in evidence or "unsupported model" in evidence:
        return STATUS_MODEL_UNSUPPORTED
    return STATUS_BUILD_FAILED if stage == "build" else STATUS_QUERY_FAILED


def _error_message(error: BaseException) -> str:
    message = " | ".join(str(item).strip() for item in _exception_chain(error) if str(item).strip())
    return " ".join(message.split())[:2000]


def _reproducer(plan: EngineProbePlan, call: ProbeCall) -> str:
    if plan.planning_status:
        return json.dumps(
            {
                "api": "tools.support_matrix.fpe_support_matrix.build_probe_plans",
                "error": plan.planning_error_message,
                "error_type": plan.planning_error_type,
                "phase": call.phase,
                "status": plan.planning_status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    payload = {
        "api": "aisimulate_core.sdk.EngineHandle.compile",
        "compile": {
            "model_path": plan.model,
            "system": plan.system,
            "backend": plan.backend,
            **plan.compile_kwargs(),
        },
        "probe": {"method": call.method, "kwargs": call.kwargs},
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _result(
    plan: EngineProbePlan,
    call: ProbeCall,
    *,
    status: str,
    source_version: str,
    source_sha: str,
    latency_ms: float | None = None,
    source: str = "",
    failure_stage: str = "",
    error: BaseException | None = None,
    error_message: str = "",
) -> FPEProbeResult:
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid FPE support-matrix status: {status}")
    topology = plan.topology
    return FPEProbeResult(
        model=plan.model,
        architecture=plan.architecture,
        system=plan.system,
        backend=plan.backend,
        backend_version=plan.backend_version,
        forward_model=plan.forward_model,
        roles="|".join(plan.roles),
        tp_size=topology.tp_size,
        pp_size=topology.pp_size,
        attention_dp_size=topology.attention_dp_size,
        moe_tp_size=topology.moe_tp_size,
        moe_ep_size=topology.moe_ep_size,
        cp_size=topology.cp_size,
        gemm_quant_mode=plan.gemm_quant_mode or "",
        moe_quant_mode=plan.moe_quant_mode or "",
        kvcache_quant_mode=plan.kvcache_quant_mode or "",
        fmha_quant_mode=plan.fmha_quant_mode or "",
        comm_quant_mode=plan.comm_quant_mode or "",
        nextn=plan.nextn,
        phase=call.phase,
        status=status,
        latency_ms=latency_ms,
        source=source,
        failure_stage=failure_stage,
        error_type=type(error).__name__ if error is not None else "",
        error_message=_error_message(error) if error is not None else error_message,
        reproducer=_reproducer(plan, call),
        source_version=source_version,
        source_sha=source_sha,
        attention_backend=plan.attention_backend,
    )


def _default_engine_factory(plan: EngineProbePlan) -> Any:
    from aisimulate_core.sdk import EngineHandle

    return EngineHandle.compile(plan.model, plan.system, plan.backend, **plan.compile_kwargs())


def probe_plan(
    plan: EngineProbePlan,
    *,
    workload: ProbeWorkload,
    source_version: str,
    source_sha: str,
    engine_factory: Callable[[EngineProbePlan], Any] = _default_engine_factory,
) -> list[FPEProbeResult]:
    calls = probe_calls_for_roles(plan.roles, workload)
    if plan.planning_status:
        return [
            _result(
                plan,
                call,
                status=plan.planning_status,
                source_version=source_version,
                source_sha=source_sha,
                failure_stage="plan",
                error_message=f"{plan.planning_error_type}: {plan.planning_error_message}",
            )
            for call in calls
        ]
    if plan.unrepresentable_reasons:
        message = "; ".join(plan.unrepresentable_reasons)
        return [
            _result(
                plan,
                call,
                status=STATUS_SDK_UNREPRESENTABLE,
                source_version=source_version,
                source_sha=source_sha,
                failure_stage="plan",
                error_message=message,
            )
            for call in calls
        ]

    try:
        engine = engine_factory(plan)
    except Exception as error:
        status = classify_failure(error, stage="build")
        return [
            _result(
                plan,
                call,
                status=status,
                source_version=source_version,
                source_sha=source_sha,
                failure_stage="build",
                error=error,
            )
            for call in calls
        ]

    results: list[FPEProbeResult] = []
    for call in calls:
        try:
            latency_ms = float(getattr(engine, call.method)(**call.kwargs))
            if not math.isfinite(latency_ms) or latency_ms <= 0:
                raise ValueError(f"{call.method} returned non-positive/non-finite latency {latency_ms!r}")
            provenance = engine.last_provenance() if hasattr(engine, "last_provenance") else None
            results.append(
                _result(
                    plan,
                    call,
                    status=STATUS_PASS,
                    source_version=source_version,
                    source_sha=source_sha,
                    latency_ms=latency_ms,
                    source=str(provenance or "silicon"),
                )
            )
        except Exception as error:
            results.append(
                _result(
                    plan,
                    call,
                    status=classify_failure(error, stage="query"),
                    source_version=source_version,
                    source_sha=source_sha,
                    failure_stage="query",
                    error=error,
                )
            )
    return results


def run_probe_plans(
    plans: Sequence[EngineProbePlan],
    *,
    workload: ProbeWorkload,
    source_version: str,
    source_sha: str,
    max_workers: int | None = None,
    engine_factory: Callable[[EngineProbePlan], Any] = _default_engine_factory,
) -> tuple[list[FPEProbeResult], MatrixRunMetrics]:
    """Run one bounded thread pool per database identity to limit live data sets."""
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    resolved_workers = max_workers or (os.cpu_count() or 1)
    if resolved_workers <= 0:
        raise ValueError("max_workers must be positive")

    sorted_plans = sorted(plans, key=EngineProbePlan.sort_key)
    results: list[FPEProbeResult] = []
    group_key = lambda plan: (plan.system, plan.backend, plan.backend_version)
    for _database_identity, group_iter in groupby(sorted_plans, key=group_key):
        group = list(group_iter)
        with ThreadPoolExecutor(max_workers=min(resolved_workers, len(group))) as executor:
            futures = {
                executor.submit(
                    probe_plan,
                    plan,
                    workload=workload,
                    source_version=source_version,
                    source_sha=source_sha,
                    engine_factory=engine_factory,
                ): plan
                for plan in group
            }
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception:
                    plan = futures[future]
                    error = RuntimeError(f"unhandled FPE probe failure:\n{traceback.format_exc()}")
                    for call in probe_calls_for_roles(plan.roles, workload):
                        results.append(
                            _result(
                                plan,
                                call,
                                status=STATUS_QUERY_FAILED,
                                source_version=source_version,
                                source_sha=source_sha,
                                failure_stage="worker",
                                error=error,
                            )
                        )

    results.sort(key=FPEProbeResult.sort_key)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss_kib = int(usage.ru_maxrss)
    if sys.platform == "darwin":
        # macOS reports bytes; Linux reports KiB.
        peak_rss_kib //= 1024
    metrics = MatrixRunMetrics(
        wall_time_seconds=time.perf_counter() - started_wall,
        cpu_time_seconds=time.process_time() - started_cpu,
        peak_rss_kib=peak_rss_kib,
        max_workers=resolved_workers,
        plan_count=len(plans),
        result_count=len(results),
    )
    return results, metrics


def _csv_rows(results: Sequence[FPEProbeResult]) -> list[dict[str, Any]]:
    rows = []
    for result in sorted(results, key=FPEProbeResult.sort_key):
        row = asdict(result)
        row["latency_ms"] = "" if result.latency_ms is None else f"{result.latency_ms:.9f}"
        rows.append(row)
    return rows


def _markdown_summary(results: Sequence[FPEProbeResult], metadata: dict[str, Any]) -> str:
    counts = Counter(result.status for result in results)
    lines = [
        "# AISimulate FPE coverage matrix",
        "",
        "> This artifact validates strict native Forward Pass Engine construction and representative",
        "> forward-pass queries only. It does not certify the AISimulate CLI, Sweeper, scheduler,",
        "> Replay, disaggregated rate matching, deployment validity, or prediction accuracy.",
        "",
        f"- Schema version: `{SCHEMA_VERSION}`",
        f"- AISimulate version: `{metadata['source_version']}`",
        f"- Source SHA: `{metadata['source_sha']}`",
        f"- Engine plans: `{metadata['plan_count']}`",
        f"- Probe results: `{len(results)}`",
        "",
        "## Status summary",
        "",
        "| Status | Count |",
        "| --- | ---: |",
    ]
    for status in sorted(VALID_STATUSES):
        if counts[status]:
            lines.append(f"| `{status}` | {counts[status]} |")
    lines.extend(
        [
            "",
            "PASS means the public strict-native engine built and the named representative call returned",
            "a positive finite latency. `SDK_UNREPRESENTABLE` is fail-closed: the current public builder",
            "cannot encode that resolved topology or feature, so no substitute configuration was tested.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    results: Sequence[FPEProbeResult],
    metrics: MatrixRunMetrics,
    output_dir: str | Path,
    source_version: str,
    source_sha: str,
    workload: ProbeWorkload,
) -> dict[str, Path]:
    """Write deterministic result artifacts and separate nondeterministic run metrics."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=FPEProbeResult.sort_key)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "source_version": source_version,
        "source_sha": source_sha,
        "wheel_sha256": os.environ.get("FPE_WHEEL_SHA256"),
        "plan_count": metrics.plan_count,
        "workload": asdict(workload),
    }

    json_path = destination / "fpe_support_matrix.json"
    csv_path = destination / "fpe_support_matrix.csv"
    markdown_path = destination / "README.md"
    metrics_path = destination / "run_metrics.json"

    payload = {"metadata": metadata, "results": [asdict(result) for result in sorted_results]}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_rows = _csv_rows(sorted_results)
    fieldnames = list(asdict(sorted_results[0]).keys()) if sorted_results else list(FPEProbeResult.__annotations__)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    markdown_path.write_text(_markdown_summary(sorted_results, metadata), encoding="utf-8")
    metrics_path.write_text(json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
        "metrics": metrics_path,
    }
