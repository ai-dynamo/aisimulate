# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aisimulate.sweeper.parallel_enum import (
    RoleParallelCandidates,
    enumerate_disagg_configs,
    enumerate_parallel_configs,
    enumerate_worker_shapes,
    enumerate_worker_shapes_with_diagnostics,
)


def _tuples(shapes):
    return {(s.tp, s.dp, s.moe_tp, s.moe_ep) for s in shapes}


def test_dense_shape_is_plain_tp():
    shapes = enumerate_worker_shapes(is_moe=False, backend="vllm", gpus_per_worker=4)
    assert _tuples(shapes) == {(4, 1, 1, 1)}
    s = shapes[0]
    assert s.pp == 1
    assert s.gpus_per_worker == 4
    assert s.strategy == "tp"


def test_moe_scans_tp_tep_dep():
    # Every MoE model scans TEP + DEP + MoE-TP under both attention modes (no MLA carve-out).
    shapes = enumerate_worker_shapes(is_moe=True, backend="vllm", gpus_per_worker=4)
    # TEP (4,1,1,4), DEP (1,4,1,4), MoE-TP+TP-attn (4,1,4,1), MoE-TP+DP-attn (1,4,4,1)
    assert _tuples(shapes) == {(4, 1, 1, 4), (1, 4, 1, 4), (4, 1, 4, 1), (1, 4, 4, 1)}
    assert {s.strategy for s in shapes} == {"tep", "dep", "tp", "dtp"}
    for s in shapes:  # width constraint dp*tp == moe_tp*moe_ep == gpus_per_worker
        assert s.dp * s.tp == s.moe_tp * s.moe_ep == 4
        assert s.gpus_per_worker == 4


def test_mla_excludes_pure_tp_dense_keeps_tp():
    # MLA+MoE (allow_moe_pure_tp=False): pure expert-TP (4,1,4,1) is dropped,
    # leaving only TEP / DEP.
    mla = enumerate_worker_shapes(
        is_moe=True, backend="vllm", gpus_per_worker=4, allow_moe_pure_tp=False
    )
    assert _tuples(mla) == {(4, 1, 1, 4), (1, 4, 1, 4)}
    assert {s.strategy for s in mla} == {"tep", "dep"}
    # GQA+MoE (default allow_moe_pure_tp=True): pure expert-TP is kept.
    gqa = enumerate_worker_shapes(
        is_moe=True, backend="vllm", gpus_per_worker=4, allow_moe_pure_tp=True
    )
    assert (4, 1, 4, 1) in _tuples(gqa)
    # Dense models do not go through the MoE pure-pattern gate: plain TP regardless.
    dense = enumerate_worker_shapes(
        is_moe=False, backend="vllm", gpus_per_worker=4, allow_moe_pure_tp=False
    )
    assert _tuples(dense) == {(4, 1, 1, 1)}


def test_allow_moe_pure_tp_threads_through_configs():
    # The carve-out propagates through enumerate_parallel_configs / enumerate_disagg_configs.
    agg = enumerate_parallel_configs(
        is_moe=True, backend="vllm", gpu_budget=8, allow_moe_pure_tp=False
    )
    assert agg
    assert all(c.shape.moe_tp == 1 for c in agg)  # no pure expert-TP
    disagg = enumerate_disagg_configs(
        is_moe=True, backend="vllm", gpu_budget=8, allow_moe_pure_tp=False
    )
    assert disagg
    assert all(
        c.prefill.shape.moe_tp == 1 and c.decode.shape.moe_tp == 1 for c in disagg
    )
    # default keeps pure expert-TP available
    agg_default = enumerate_parallel_configs(
        is_moe=True, backend="trtllm", gpu_budget=8
    )
    assert any(c.shape.moe_tp > 1 for c in agg_default)


def test_sglang_wideep_keeps_moe_tp():
    # wideEP no longer forces EP-only: sglang keeps MoE pure expert-TP (4,1,4,1)
    # alongside TEP/DEP, matching real GLM-5 multinode deployments (InferenceX EP=1).
    shapes = enumerate_worker_shapes(
        is_moe=True, backend="sglang", gpus_per_worker=4, enable_wideep=True
    )
    assert _tuples(shapes) == {(4, 1, 1, 4), (1, 4, 1, 4), (4, 1, 4, 1), (1, 4, 4, 1)}
    # The explicit EP-only MoE kernels still drop MoE-TP (both attention modes), even under wideep.
    for moe_backend in ("deepep_moe", "megamoe"):
        ep_only = enumerate_worker_shapes(
            is_moe=True,
            backend="sglang",
            gpus_per_worker=4,
            enable_wideep=True,
            moe_backend=moe_backend,
        )
        assert _tuples(ep_only) == {(4, 1, 1, 4), (1, 4, 1, 4)}, moe_backend


def test_sglang_moe_backend_filters():
    # moe_backend in {deepep_moe, megamoe} is EP-only -> pure expert-TP dropped (same as wideep).
    for moe_backend in ("deepep_moe", "megamoe"):
        shapes = enumerate_worker_shapes(
            is_moe=True, backend="sglang", gpus_per_worker=4, moe_backend=moe_backend
        )
        assert _tuples(shapes) == {(4, 1, 1, 4), (1, 4, 1, 4)}, moe_backend
    # A non-EP-only moe_backend leaves pure expert-TP available.
    shapes = enumerate_worker_shapes(
        is_moe=True, backend="sglang", gpus_per_worker=4, moe_backend="cutlass_moe"
    )
    assert (4, 1, 4, 1) in _tuples(shapes)


def test_moe_has_no_shape_below_two_gpus():
    assert enumerate_worker_shapes(is_moe=True, backend="vllm", gpus_per_worker=1) == []


def test_replica_iteration_within_budget():
    cfgs = enumerate_parallel_configs(is_moe=False, backend="vllm", gpu_budget=8)
    by_g: dict[int, set[int]] = {}
    for c in cfgs:
        by_g.setdefault(c.shape.gpus_per_worker, set()).add(c.replicas)
    assert by_g[8] == {1}
    assert by_g[4] == {1, 2}
    assert by_g[2] == {1, 2, 3, 4}
    assert by_g[1] == set(range(1, 9))
    assert all(c.total_gpus <= 8 for c in cfgs)


def test_min_gpus_per_worker_floor():
    # memory-fit floor: workers smaller than 4 GPUs are excluded
    cfgs = enumerate_parallel_configs(
        is_moe=False, backend="vllm", gpu_budget=8, min_gpus_per_worker=4
    )
    assert {c.shape.gpus_per_worker for c in cfgs} == {4, 8}


def test_min_gpu_budget_floor():
    cfgs = enumerate_parallel_configs(
        is_moe=False, backend="vllm", gpu_budget=8, min_gpu_budget=4
    )
    assert all(4 <= c.total_gpus <= 8 for c in cfgs)
    g2_replicas = sorted(c.replicas for c in cfgs if c.shape.gpus_per_worker == 2)
    assert g2_replicas == [2, 3, 4]  # total 4,6,8


def test_moe_budget_includes_pure_tp():
    cfgs = enumerate_parallel_configs(is_moe=True, backend="trtllm", gpu_budget=16)
    assert cfgs
    assert all(c.total_gpus <= 16 for c in cfgs)
    assert any(c.shape.moe_tp > 1 for c in cfgs)  # pure expert-TP scanned for all MoE


def test_disagg_pairs_share_budget():
    cfgs = enumerate_disagg_configs(is_moe=True, backend="trtllm", gpu_budget=8)
    assert len(cfgs) == 176
    for c in cfgs:
        assert c.total_gpus == c.prefill.total_gpus + c.decode.total_gpus <= 8
        assert (
            c.prefill.total_gpus >= 2 and c.decode.total_gpus >= 2
        )  # each role >= 1 worker
    # prefill and decode may differ in shape
    assert any(c.prefill.shape != c.decode.shape for c in cfgs)


def test_disagg_min_gpu_budget_full_utilization():
    cfgs = enumerate_disagg_configs(
        is_moe=True, backend="trtllm", gpu_budget=8, min_gpu_budget=8
    )
    assert len(cfgs) == 96
    assert all(c.total_gpus == 8 for c in cfgs)


def test_disagg_dense_small_budget():
    cfgs = enumerate_disagg_configs(is_moe=False, backend="vllm", gpu_budget=4)
    assert cfgs
    assert all(c.total_gpus <= 4 for c in cfgs)
    assert all(c.prefill.shape.strategy == "tp" for c in cfgs)


def test_explicit_pipeline_and_context_parallel_candidates_are_enumerated():
    pipeline = enumerate_worker_shapes(
        is_moe=False,
        backend="vllm",
        gpus_per_worker=4,
        tp_candidates=(2,),
        pp_candidates=(2,),
        attention_dp_candidates=(1,),
        moe_tp_candidates=(1,),
        moe_ep_candidates=(1,),
    )
    assert [(shape.tp, shape.pp, shape.cp) for shape in pipeline] == [(2, 2, 1)]

    context, context_diagnostics = enumerate_worker_shapes_with_diagnostics(
        is_moe=True,
        backend="sglang",
        gpus_per_worker=4,
        tp_candidates=(1, 2),
        pp_candidates=(1,),
        attention_dp_candidates=(1,),
        moe_tp_candidates=(1,),
        moe_ep_candidates=(4,),
        cp_candidates=(2, 4),
    )
    assert [
        (shape.tp, shape.pp, shape.dp, shape.moe_ep, shape.cp) for shape in context
    ] == [(1, 1, 1, 4, 4)]
    assert context_diagnostics.as_dict() == {
        "gpu_count_mismatch": 2,
        "sglang_cp_attention_parallelism": 1,
    }


def test_invalid_combination_pruning_counts_are_deterministic():
    shapes, diagnostics = enumerate_worker_shapes_with_diagnostics(
        is_moe=False,
        backend="vllm",
        gpus_per_worker=2,
        tp_candidates=(1, 2),
        pp_candidates=(1,),
        attention_dp_candidates=(1,),
        moe_tp_candidates=(1,),
        moe_ep_candidates=(1,),
        cp_candidates=(1,),
    )

    assert [(shape.tp, shape.pp) for shape in shapes] == [(2, 1)]
    assert diagnostics.considered == 2
    assert diagnostics.accepted == 1
    assert diagnostics.as_dict() == {"gpu_count_mismatch": 1}


def test_disagg_role_worker_lists_and_replica_ceiling_prune_before_search():
    prefill = RoleParallelCandidates(
        gpus_per_worker=(2,),
        tp=(2,),
        pp=(1,),
        attention_dp=(1,),
        moe_tp=(1,),
        moe_ep=(1,),
        cp=(1,),
        workers=(1, 2),
    )
    decode = RoleParallelCandidates(
        gpus_per_worker=(1,),
        tp=(1,),
        pp=(1,),
        attention_dp=(1,),
        moe_tp=(1,),
        moe_ep=(1,),
        cp=(1,),
        workers=(1, 3),
    )

    configs = enumerate_disagg_configs(
        is_moe=False,
        backend="vllm",
        gpu_budget=8,
        prefill_candidates=prefill,
        decode_candidates=decode,
        num_gpu_per_replica=(3, 5),
        max_gpu_per_replica=5,
        max_prefill_workers=1,
        max_decode_workers=3,
    )

    assert [
        (config.prefill.replicas, config.decode.replicas, config.total_gpus)
        for config in configs
    ] == [(1, 1, 3), (1, 3, 5)]
