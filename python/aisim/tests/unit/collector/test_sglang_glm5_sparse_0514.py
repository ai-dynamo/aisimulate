# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "glm5_dsa_sparse_modules.py"


def _load_pure_helpers(*names):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def test_glm5_topk_metadata_uses_causal_lengths_and_full_kv_offsets():
    helper = _load_pure_helpers("_glm5_topk_metadata")["_glm5_topk_metadata"]

    lengths, offsets, max_seqlen_k = helper(bs=2, isl=3, past_kv=4)

    assert lengths == [5, 6, 7, 5, 6, 7]
    assert offsets == [0, 0, 0, 7, 7, 7]
    assert max_seqlen_k == 7


def test_glm5_decode_metadata_keeps_full_kv_page_table_width():
    helper = _load_pure_helpers("_glm5_topk_metadata")["_glm5_topk_metadata"]

    lengths, offsets, max_seqlen_k = helper(bs=2, isl=1, past_kv=4095)

    assert lengths == [4096, 4096]
    assert offsets == [0, 4096]
    assert max_seqlen_k == 4096

    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "torch.arange(max_seqlen_k" in source
    assert "pt.view(1, max_seqlen_k)" in source


def test_glm5_flash_mla_heads_are_architecture_specific():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "q_heads = 128 if sm_major >= 10 else native_heads" in source


def _real_torch():
    torch = pytest.importorskip("torch")
    if not isinstance(getattr(torch, "__version__", None), str) or not hasattr(torch, "where"):
        pytest.skip("requires real torch, not another collector test's import stub")
    return torch


def _load_with(namespace, *names):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def _run_once(kernel_fn, **_kwargs):
    kernel_fn()
    return {"latency_ms": 1.0}


def test_glm5_flash_mla_keeps_the_production_topk_width_and_passes_serving_topk_length(monkeypatch):
    """The serving call (dsa_backend._forward_flashmla_sparse @v0.5.23): flash_mla_interface's
    kernel, topk columns per row, -1 past the row's causal count, and that count as topk_length."""
    import sys
    from types import ModuleType

    torch = _real_torch()
    calls = []
    interface = ModuleType("sglang.kernels.ops.attention.flash_mla_interface")
    interface.flash_mla_sparse_fwd = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, interface.__name__, interface)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))
    bench = _load_with({"torch": torch, "_bench_cuda_graph": _run_once}, "_bench_glm5_dsa_attn")["_bench_glm5_dsa_attn"]

    bs, isl, past_kv, topk = 2, 3, 4, 6
    bench(bs * isl, past_kv, isl, native_heads=64, d_qk=576, d_v=512, topk=topk, device="cpu")

    (call,) = calls
    assert call["q"].shape == (6, 128, 576)  # SM100 pads the heads to 128
    assert call["kv"].shape == (bs * (past_kv + isl), 1, 576)
    assert call["topk_length"].dtype == torch.int32
    assert call["topk_length"].tolist() == [5, 6, 6, 5, 6, 6]  # min(past_kv + p + 1, topk)
    indices = call["indices"]
    assert indices.shape == (6, 1, topk) and indices.dtype == torch.int32
    assert indices[0, 0].tolist() == [0, 1, 2, 3, 4, -1]
    assert indices[3, 0].tolist() == [7, 8, 9, 10, 11, -1]  # the second request's own segment
    assert indices[5, 0].tolist() == [7, 8, 9, 10, 11, 12]


def _topk_namespace(monkeypatch, torch, *, v2):
    import sys
    from types import ModuleType

    calls = []
    sgl_kernel = ModuleType("sgl_kernel")
    sgl_kernel.fast_topk_transform_fused = lambda **kwargs: calls.append(("fused", kwargs))
    sgl_kernel.fast_topk_v2 = lambda *args, **kwargs: calls.append(("v2_unfused", args, kwargs))
    topk_ops = ModuleType("sglang.kernels.ops.attention.dsv4.topk")
    topk_ops.plan_topk_v2 = lambda lengths: calls.append(("plan", lengths.tolist())) or "plan"
    topk_ops.topk_transform_paged_v2 = lambda *args: calls.append(("paged_v2", args))
    monkeypatch.setitem(sys.modules, "sgl_kernel", sgl_kernel)
    monkeypatch.setitem(sys.modules, topk_ops.__name__, topk_ops)
    namespace = {
        "torch": torch,
        "_bench_cuda_graph": _run_once,
        "_glm5_fuse_topk_enabled": lambda: True,
        "_glm5_topk_v2_enabled": lambda: v2,
    }
    _load_with(namespace, "_bench_glm5_topk", "_glm5_topk_metadata", "_make_glm5_topk_scores")
    return namespace, calls


def test_glm5_decode_topk_runs_the_serving_v2_paged_kernel(monkeypatch):
    """Decode PAGED top-k without row_starts goes to _topk_transform_v2_paged when
    SGLANG_OPT_USE_TOPK_V2 is on (dsa_topk_backend.py:123-133 @v0.5.23)."""
    torch = _real_torch()
    namespace, calls = _topk_namespace(monkeypatch, torch, v2=True)

    results, kernel_source = namespace["_bench_glm5_topk"](2, 4095, 1, 2, topk=2048, device="cpu")

    assert kernel_source == "topk_transform_v2"
    assert [mode for mode, _ in results] == ["top_last"]  # no flat row: the modules time written history
    plans = [call for call in calls if call[0] == "plan"]
    assert plans == [("plan", [4096, 4096])]  # once, outside the timed kernel
    paged = [call[1] for call in calls if call[0] == "paged_v2"]
    ((score, lengths, page_table, out, page_size, plan),) = paged
    assert score.shape == (2, 4096) and score.stride(0) % 4 == 0
    assert lengths.tolist() == [4096, 4096]
    assert page_table.shape == (2, 64) and page_size == 64 and plan == "plan"
    assert out.shape == (2, 2048) and out.dtype == torch.int32
    assert not [call for call in calls if call[0] == "fused"]


def test_glm5_decode_topk_keeps_the_legacy_fused_kernel_when_v2_is_off(monkeypatch):
    torch = _real_torch()
    namespace, calls = _topk_namespace(monkeypatch, torch, v2=False)

    _results, kernel_source = namespace["_bench_glm5_topk"](2, 4095, 1, 2, topk=2048, device="cpu")

    assert kernel_source == "fast_topk_transform_fused"
    fused = [call[1] for call in calls if call[0] == "fused"]
    assert len(fused) == 1 and all("row_starts" not in kwargs for kwargs in fused)
    assert not [call for call in calls if call[0] in {"plan", "paged_v2"}]


def test_glm5_context_topk_stays_on_the_fused_paged_transform_with_row_starts(monkeypatch):
    torch = _real_torch()
    namespace, calls = _topk_namespace(monkeypatch, torch, v2=True)
    namespace["_glm5_score_rows_per_chunk"] = lambda num_q, _num_k, _device: num_q

    _results, kernel_source = namespace["_bench_glm5_topk"](2 * 4096, 0, 4096, 2, topk=2048, device="cpu")

    assert kernel_source == "fast_topk_transform_fused"
    fused = [call[1] for call in calls if call[0] == "fused"]
    assert len(fused) == 1 and all(kwargs["row_starts"] is not None for kwargs in fused)
    assert not [call for call in calls if call[0] in {"plan", "paged_v2"}]


def test_glm5_sparse_collector_uses_the_dsa_fuse_topk_switch_not_the_nsa_spelling():
    # the file pin itself is enforced by test_stock_collectors_are_exactly_pinned_to_their_resolved_runtime
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "SGLANG_DSA_FUSE_TOPK" in source
    assert "SGLANG_NSA_FUSE_TOPK" not in source


def test_glm5_mqa_matches_sglang_query_row_chunking_contract():
    source = SOURCE_PATH.read_text(encoding="utf-8")

    assert "SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION" in source
    assert "budget_bytes = min(int(free_mem * free_mem_fraction), int(total_mem * 0.30))" in source
    assert "budget_bytes // (num_k * 4)" in source
    assert "_glm5_score_rows_per_chunk(M, full_s, device)" in source
    assert "for start in range(0, M, query_rows_per_chunk)" in source
    assert "q[start:end]" in source
    assert "ks[start:end]" in source
    assert "ke[start:end]" in source


def test_glm5_sparse_rows_carry_the_dsa_module_points_of_their_batch(monkeypatch):
    from collector.op_plans import _long_context_points, _max_position, get_mla_module_sweep_spec, plan

    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    sweep = get_mla_module_sweep_spec("plan")
    max_position = _max_position("zai-org/GLM-5")
    expected = {(b, s, p) for b, s, p in _long_context_points("context", sweep, max_position)}
    expected |= {(b, 1, kv) for b, kv, _p in _long_context_points("generation", sweep, max_position)}
    for kernel, op in (
        ("mqa", "glm5_mqa_logits_module"),
        ("topk", "glm5_topk_module"),
        ("dsa_attn", "glm5_dsa_attn_module"),
    ):
        rows = plan(op, "sglang", 90).rows
        assert len({row[2] for row in rows}) == len(rows)  # one task per batch
        got = set()
        for model, row_kernel, batch, points in rows:
            assert (model, row_kernel) == ("zai-org/GLM-5", kernel)
            assert points == sorted(points)  # shortest prefix first
            got |= {(batch, seq_len, prefix) for prefix, seq_len in points}
        assert got == expected


def test_glm5_sparse_platform_gating_lives_outside_the_getter():
    """The getter itself carries no SM predicate: pre-Hopper is dropped by
    collector/skips.py and SM120 is parked by registry maturity markers, so
    both stay auditable declaration-layer decisions."""
    from collector.sglang.registry import REGISTRY
    from collector.skips import filter_cases

    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "not in {90, 100, 103}" not in source

    glm5_ops = ("glm5_mqa_logits_module", "glm5_topk_module", "glm5_dsa_attn_module")
    for op in glm5_ops:
        assert filter_cases([[1]], "sglang", op, 89)[0] == []
        assert filter_cases([[1]], "sglang", op, 90)[0] == [[1]]
    markers = {entry.op: entry.unverified_sms for entry in REGISTRY if entry.op in glm5_ops}
    assert markers == dict.fromkeys(glm5_ops, (120,))


def test_dsv4_sparse_platform_gating_lives_outside_the_getter():
    """Same contract for the DSV4 sparse sub-kernel family. The flash_mla-less
    HCA/CSA FMLA sidecar ops were parked unverified until 2026-10-15 and are now
    removed outright (retired SDK placeholders that failed every registry-generated
    shard plan), so only the indexer scoring kernel and the topk calibration remain."""
    from collector.sglang.registry import REGISTRY
    from collector.skips import filter_cases

    dsv4_source = (REPO_ROOT / "collector" / "sglang" / "deepseekv4_sparse_modules.py").read_text(encoding="utf-8")
    assert "_dsv4_sparse_kernel_supported" not in dsv4_source
    assert "_dsv4_topk_kernel_supported" not in dsv4_source

    floors = ("dsv4_paged_mqa_logits_module", "dsv4_csa_topk_calib")
    for op in floors:
        assert filter_cases([[1]], "sglang", op, 89)[0] == []
        assert filter_cases([[1]], "sglang", op, 90)[0] == [[1]]
    by_op = {entry.op: entry for entry in REGISTRY}
    assert by_op["dsv4_paged_mqa_logits_module"].unverified_sms == (120,)
    assert by_op["dsv4_csa_topk_calib"].unverified_sms == (120,)
    assert not {"dsv4_hca_attn_module", "dsv4_csa_attn_module"} & set(by_op)
