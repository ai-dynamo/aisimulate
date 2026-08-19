# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enumerate legal per-worker parallel shapes and worker counts within a GPU budget.

The per-worker shape enumeration mirrors
``aiconfigurator.sdk.utils.enumerate_parallel_config``. The MoE width constraint
``dp*tp*cp == moe_tp*moe_ep`` holds; for MoE only the pure TEP / DEP / MoE-TP patterns are
kept (MoE-TP — moe_ep==1 under tensor- or DP-attention — gated by ``allow_moe_pure_tp``,
now enabled for every MoE model incl. MLA); dense models use plain TP.
The backend-specific MoE filters are mirrored too.

``enumerate_parallel_config`` stops at *one worker's* shape
(GPUs/worker = tp*pp*attention_dp*cp).
On top of that, this module iterates the replica counts ``r`` such that
``gpus_per_worker * r`` fits the GPU budget — the replica/worker count AIC derives
separately in its sweep layer.

Kept standalone (no aiconfigurator import) so it is light and unit-testable;
parity with AIC's rules is covered by tests. ``is_moe`` is an input here
(resolved from the model via AIC's ``check_is_moe`` by the caller); the KV-cache
feasibility of each shape is applied separately by :mod:`aisimulate.sweeper.model_hw`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace

# GPUs-per-worker ladder (matches AIC's default num_gpu_per_worker).
_DEFAULT_GPUS_PER_WORKER: tuple[int, ...] = (1, 2, 4, 8, 16)
# Ladder used to enumerate tp / dp / moe_tp / moe_ep candidates within a worker.
_DIM_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)


@dataclass(frozen=True)
class RoleParallelCandidates:
    """Explicit finite topology and worker-count domain for one engine role."""

    gpus_per_worker: tuple[int, ...] = _DEFAULT_GPUS_PER_WORKER
    tp: tuple[int, ...] = _DIM_LADDER
    pp: tuple[int, ...] = (1,)
    attention_dp: tuple[int, ...] = _DIM_LADDER
    moe_tp: tuple[int, ...] = _DIM_LADDER
    moe_ep: tuple[int, ...] = _DIM_LADDER
    cp: tuple[int, ...] = (1,)
    workers: tuple[int, ...] | None = None


@dataclass(frozen=True)
class EnumerationDiagnostics:
    """Deterministic counts for a finite enumeration and its pruning rules."""

    considered: int
    accepted: int
    pruned: tuple[tuple[str, int], ...]

    @classmethod
    def from_counter(
        cls, *, considered: int, accepted: int, pruned: Counter[str]
    ) -> EnumerationDiagnostics:
        return cls(
            considered=considered,
            accepted=accepted,
            pruned=tuple(sorted(pruned.items())),
        )

    def as_dict(self) -> dict[str, int]:
        return dict(self.pruned)


@dataclass(frozen=True)
class ParallelShape:
    """One worker's parallel shape.

    ``dp`` is the attention data-parallel size (attention_dp_size).
    """

    tp: int
    dp: int
    moe_tp: int
    moe_ep: int
    pp: int = 1
    cp: int = 1

    @property
    def gpus_per_worker(self) -> int:
        return self.tp * self.pp * self.dp * self.cp

    @property
    def strategy(self) -> str:
        """Label per AIC's real-silicon patterns: ``tp`` (dense, or MoE tensor-parallel
        under tensor-attention), ``tep`` (attention-TP + expert-EP), ``dep``
        (attention-DP + expert-EP), ``dtp`` (attention-DP + MoE tensor-parallel —
        e.g. InferenceX GLM-5's EP=1 + DP-attention)."""
        if self.moe_tp == 1 and self.moe_ep == 1:
            return "tp"  # dense
        if (
            self.tp * self.cp > 1
            and self.dp == 1
            and self.moe_tp == 1
            and self.moe_ep > 1
        ):
            return "tep"
        if self.tp == 1 and self.dp > 1 and self.moe_tp == 1 and self.moe_ep > 1:
            return "dep"
        if (
            self.tp * self.cp > 1
            and self.dp == 1
            and self.moe_tp > 1
            and self.moe_ep == 1
        ):
            return "tp"  # MoE tensor-parallel, tensor-attention
        if self.tp == 1 and self.dp > 1 and self.moe_tp > 1 and self.moe_ep == 1:
            return "dtp"  # MoE tensor-parallel, DP-attention
        return "mixed"


@dataclass(frozen=True)
class ReplicaParallelConfig:
    """A worker shape plus how many replicas of it run, under a GPU budget."""

    shape: ParallelShape
    replicas: int

    @property
    def total_gpus(self) -> int:
        return self.shape.gpus_per_worker * self.replicas


@dataclass(frozen=True)
class DisaggParallelConfig:
    """A disaggregated candidate: a prefill worker config + a decode worker
    config sharing the GPU budget. prefill and decode are independent (shape and
    replica count may differ)."""

    prefill: ReplicaParallelConfig
    decode: ReplicaParallelConfig

    @property
    def total_gpus(self) -> int:
        return self.prefill.total_gpus + self.decode.total_gpus


def _backend_allows_moe_tp(
    backend: str, *, enable_wideep: bool, moe_backend: str | None
) -> bool:
    """sglang's EP-only MoE *kernels* (deepep_moe / megamoe) require moe_tp=1. wideEP
    (multinode wide expert-parallelism) does NOT force it on its own: real GLM-5 sglang
    deployments run MoE tensor-parallel multinode (InferenceX reports EP=1), so MoE-TP
    stays available. ``enable_wideep`` is accepted for call-site compatibility."""
    del enable_wideep  # no longer gates MoE-TP; kept in the signature for callers
    return not (backend == "sglang" and moe_backend in {"deepep_moe", "megamoe"})


def enumerate_worker_shapes_with_diagnostics(
    *,
    is_moe: bool,
    backend: str,
    gpus_per_worker: int,
    tp_candidates: tuple[int, ...] = _DIM_LADDER,
    pp_candidates: tuple[int, ...] = (1,),
    attention_dp_candidates: tuple[int, ...] = _DIM_LADDER,
    moe_tp_candidates: tuple[int, ...] = _DIM_LADDER,
    moe_ep_candidates: tuple[int, ...] = _DIM_LADDER,
    cp_candidates: tuple[int, ...] = (1,),
    enable_wideep: bool = False,
    moe_backend: str | None = None,
    allow_moe_pure_tp: bool = True,
) -> tuple[list[ParallelShape], EnumerationDiagnostics]:
    """Legal per-worker shapes and pruning counts at exactly ``gpus_per_worker`` GPUs.

    Mirrors ``enumerate_parallel_config`` (width + backend filters) followed by
    ``filter_real_silicon_configs``. For MoE, TEP / DEP are always scanned; MoE
    tensor-parallel (moe_ep == 1, under tensor- or DP-attention) is kept when
    ``allow_moe_pure_tp`` — now enabled for every MoE model, MLA included, since
    real deployments run it (InferenceX GLM-5 reports EP=1). Dense models scan
    plain TP and are unaffected. Backend EP-only filters (sglang wideep) still apply.
    """
    considered = 0
    pruned: Counter[str] = Counter()
    shapes: list[ParallelShape] = []
    for tp in tp_candidates:
        for pp in pp_candidates:
            for dp in attention_dp_candidates:
                for moe_tp in moe_tp_candidates:
                    for moe_ep in moe_ep_candidates:
                        for cp in cp_candidates:
                            considered += 1
                            if tp * pp * dp * cp != gpus_per_worker:
                                pruned["gpu_count_mismatch"] += 1
                                continue
                            if not is_moe and (dp, moe_tp, moe_ep) != (1, 1, 1):
                                pruned["dense_parallelism"] += 1
                                continue
                            width = tp * dp * cp
                            if is_moe and moe_tp * moe_ep != width:
                                pruned["moe_width_mismatch"] += 1
                                continue
                            # Backend filters from enumerate_parallel_config.
                            if backend != "sglang" and cp > 1:
                                pruned["backend_context_parallelism"] += 1
                                continue
                            if backend == "sglang" and cp > 1 and (tp > 1 or dp > 1):
                                pruned["sglang_cp_attention_parallelism"] += 1
                                continue
                            if backend == "trtllm" and dp > 1 and tp > 1:
                                pruned["trtllm_attention_parallelism"] += 1
                                continue
                            if (
                                backend == "sglang"
                                and moe_tp > 1
                                and not _backend_allows_moe_tp(
                                    backend,
                                    enable_wideep=enable_wideep,
                                    moe_backend=moe_backend,
                                )
                            ):
                                pruned["sglang_ep_only_moe_backend"] += 1
                                continue
                            if backend == "vllm" and moe_tp > 1 and moe_ep > 1:
                                pruned["vllm_mixed_moe_parallelism"] += 1
                                continue
                            if not is_moe:
                                shapes.append(
                                    ParallelShape(
                                        tp=tp,
                                        pp=pp,
                                        dp=1,
                                        moe_tp=1,
                                        moe_ep=1,
                                        cp=cp,
                                    )
                                )
                                continue
                            # Real-silicon pure-pattern filter.
                            attention_tp_width = tp * cp
                            is_tep = (
                                attention_tp_width > 1
                                and dp == 1
                                and moe_tp == 1
                                and moe_ep > 1
                            )
                            is_dep = (
                                attention_tp_width == 1
                                and dp > 1
                                and moe_tp == 1
                                and moe_ep > 1
                            )
                            # MoE tensor-parallel (moe_ep==1) under tensor-attention
                            # (strategy "tp") or DP-attention (strategy "dtp").
                            is_moe_tp = (
                                allow_moe_pure_tp
                                and moe_tp > 1
                                and moe_ep == 1
                                and (
                                    (attention_tp_width > 1 and dp == 1)
                                    or (tp == 1 and cp == 1 and dp > 1)
                                )
                            )
                            if not (is_tep or is_dep or is_moe_tp):
                                pruned["non_silicon_moe_pattern"] += 1
                                continue
                            shapes.append(
                                ParallelShape(
                                    tp=tp,
                                    pp=pp,
                                    dp=dp,
                                    moe_tp=moe_tp,
                                    moe_ep=moe_ep,
                                    cp=cp,
                                )
                            )
    return shapes, EnumerationDiagnostics.from_counter(
        considered=considered, accepted=len(shapes), pruned=pruned
    )


def enumerate_worker_shapes(**kwargs) -> list[ParallelShape]:
    """Compatibility wrapper returning only legal per-worker shapes."""

    shapes, _ = enumerate_worker_shapes_with_diagnostics(**kwargs)
    return shapes


def enumerate_parallel_configs(
    *,
    is_moe: bool,
    backend: str,
    gpu_budget: int,
    min_gpu_budget: int | None = None,
    min_gpus_per_worker: int = 1,
    gpus_per_worker_candidates: tuple[int, ...] = _DEFAULT_GPUS_PER_WORKER,
    tp_candidates: tuple[int, ...] = _DIM_LADDER,
    pp_candidates: tuple[int, ...] = (1,),
    attention_dp_candidates: tuple[int, ...] = _DIM_LADDER,
    moe_tp_candidates: tuple[int, ...] = _DIM_LADDER,
    moe_ep_candidates: tuple[int, ...] = _DIM_LADDER,
    cp_candidates: tuple[int, ...] = (1,),
    worker_candidates: tuple[int, ...] | None = None,
    max_workers: int | None = None,
    enable_wideep: bool = False,
    moe_backend: str | None = None,
    allow_moe_pure_tp: bool = True,
) -> list[ReplicaParallelConfig]:
    """Enumerate ``(worker shape, replica count)`` configs that fit ``gpu_budget``.

    For each candidate GPUs-per-worker ``g`` (in ``[min_gpus_per_worker, budget]``),
    enumerate the legal worker shapes, then iterate replica counts ``r`` in
    ``1..(gpu_budget // g)`` so the total ``g * r`` stays within
    ``[min_gpu_budget, gpu_budget]``.

    ``min_gpus_per_worker`` is an optional lower bound on a worker's GPU count
    (default 1). :func:`aisimulate.sweeper.model_hw.parallel_configs_for` leaves it at 1 and
    applies the KV-cache feasibility filter instead of a static weight floor.

    This is branch-agnostic: call once for an ``agg`` worker, or once per role
    (prefill / decode) for ``disagg`` — the prefill/decode pairing under the
    shared budget is the downstream rate-matching step.
    """
    configs: list[ReplicaParallelConfig] = []
    for g in gpus_per_worker_candidates:
        if g > gpu_budget or g < min_gpus_per_worker:
            continue
        shapes = enumerate_worker_shapes(
            is_moe=is_moe,
            backend=backend,
            gpus_per_worker=g,
            tp_candidates=tp_candidates,
            pp_candidates=pp_candidates,
            attention_dp_candidates=attention_dp_candidates,
            moe_tp_candidates=moe_tp_candidates,
            moe_ep_candidates=moe_ep_candidates,
            cp_candidates=cp_candidates,
            enable_wideep=enable_wideep,
            moe_backend=moe_backend,
            allow_moe_pure_tp=allow_moe_pure_tp,
        )
        if not shapes:
            continue
        max_replicas = gpu_budget // g
        if max_workers is not None:
            max_replicas = min(max_replicas, max_workers)
        for shape in shapes:
            replicas = worker_candidates or tuple(range(1, max_replicas + 1))
            for r in replicas:
                if r > max_replicas:
                    continue
                total = g * r
                if min_gpu_budget is not None and total < min_gpu_budget:
                    continue
                configs.append(ReplicaParallelConfig(shape=shape, replicas=r))
    return configs


def enumerate_disagg_configs(
    *,
    is_moe: bool,
    backend: str,
    gpu_budget: int,
    min_gpu_budget: int | None = None,
    min_gpus_per_worker: int = 1,
    gpus_per_worker_candidates: tuple[int, ...] | None = None,
    prefill_candidates: RoleParallelCandidates | None = None,
    decode_candidates: RoleParallelCandidates | None = None,
    num_gpu_per_replica: tuple[int, ...] | None = None,
    max_gpu_per_replica: int | None = None,
    max_prefill_workers: int | None = None,
    max_decode_workers: int | None = None,
    enable_wideep: bool = False,
    moe_backend: str | None = None,
    allow_moe_pure_tp: bool = True,
) -> list[DisaggParallelConfig]:
    """Enumerate disagg ``(prefill, decode)`` configs that share the GPU budget.

    Both roles are enumerated from the same per-role candidate set (shared
    model / hardware / backend, first pass) and paired so that
    ``prefill.total_gpus + decode.total_gpus`` lies in
    ``[min_gpu_budget, gpu_budget]``. prefill and decode may differ in shape and
    replica count.

    Required for building the disagg sweep search space. The set grows quickly
    with the budget, so the smart sweep samples from it rather than
    grid-enumerating; prefill/decode throughput rate-matching is applied
    downstream when each candidate is evaluated.
    """
    prefill_candidates = prefill_candidates or RoleParallelCandidates()
    decode_candidates = decode_candidates or RoleParallelCandidates()
    if gpus_per_worker_candidates is not None:
        prefill_candidates = replace(
            prefill_candidates, gpus_per_worker=gpus_per_worker_candidates
        )
        decode_candidates = replace(
            decode_candidates, gpus_per_worker=gpus_per_worker_candidates
        )
    prefill_role = enumerate_parallel_configs(
        is_moe=is_moe,
        backend=backend,
        gpu_budget=gpu_budget,
        min_gpus_per_worker=min_gpus_per_worker,
        gpus_per_worker_candidates=prefill_candidates.gpus_per_worker,
        tp_candidates=prefill_candidates.tp,
        pp_candidates=prefill_candidates.pp,
        attention_dp_candidates=prefill_candidates.attention_dp,
        moe_tp_candidates=prefill_candidates.moe_tp,
        moe_ep_candidates=prefill_candidates.moe_ep,
        cp_candidates=prefill_candidates.cp,
        worker_candidates=prefill_candidates.workers,
        max_workers=max_prefill_workers,
        enable_wideep=enable_wideep,
        moe_backend=moe_backend,
        allow_moe_pure_tp=allow_moe_pure_tp,
    )
    decode_role = enumerate_parallel_configs(
        is_moe=is_moe,
        backend=backend,
        gpu_budget=gpu_budget,
        min_gpus_per_worker=min_gpus_per_worker,
        gpus_per_worker_candidates=decode_candidates.gpus_per_worker,
        tp_candidates=decode_candidates.tp,
        pp_candidates=decode_candidates.pp,
        attention_dp_candidates=decode_candidates.attention_dp,
        moe_tp_candidates=decode_candidates.moe_tp,
        moe_ep_candidates=decode_candidates.moe_ep,
        cp_candidates=decode_candidates.cp,
        worker_candidates=decode_candidates.workers,
        max_workers=max_decode_workers,
        enable_wideep=enable_wideep,
        moe_backend=moe_backend,
        allow_moe_pure_tp=allow_moe_pure_tp,
    )
    if not prefill_role or not decode_role:
        return []

    # Each role needs at least its smallest worker, so cap one role's footprint
    # at budget minus the other role's minimum (prunes pairs that can never fit).
    min_prefill = min(c.total_gpus for c in prefill_role)
    min_decode = min(c.total_gpus for c in decode_role)
    prefill_role = [
        c for c in prefill_role if c.total_gpus <= gpu_budget - min_decode
    ]
    decode_role = [c for c in decode_role if c.total_gpus <= gpu_budget - min_prefill]

    configs: list[DisaggParallelConfig] = []
    for prefill in prefill_role:
        for decode in decode_role:
            total = prefill.total_gpus + decode.total_gpus
            if total > gpu_budget:
                continue
            if max_gpu_per_replica is not None and total > max_gpu_per_replica:
                continue
            if num_gpu_per_replica is not None and total not in num_gpu_per_replica:
                continue
            if min_gpu_budget is not None and total < min_gpu_budget:
                continue
            configs.append(DisaggParallelConfig(prefill=prefill, decode=decode))
    return configs
