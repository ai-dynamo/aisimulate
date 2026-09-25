# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""probe_driver record rules: kernel-name normalization, the taxonomy
contract, and the orphan keep rule that framework-mode probes depend on."""
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


@pytest.fixture(scope="module")
def pd(tmp_path_factory):
    # the driver resolves its workspace from the env at import; point it at a
    # scratch dir so no archive/ is touched
    import os
    os.environ["AIS_PROBE_WORKSPACE"] = str(tmp_path_factory.mktemp("ws"))
    spec = importlib.util.spec_from_file_location("probe_driver", COMPONENTS / "probe_driver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["probe_driver"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("raw, expected", [
    # Triton autotune tile suffix is not identity
    ("_matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1", "_matmul_ogs_NNT_bf16xbf16xmxfp4"),
    ("_matmul_ogs_NNT_bf16xbf16xmxfp4_128x256x128x1", "_matmul_ogs_NNT_bf16xbf16xmxfp4"),
    # cute-DSL instantiations bake shape params into the symbol
    ("kernel_cutlass_gdn_decode_bf16state_mtp_ilp4_kernel_tensorptrbf16gmemalign32o1291612812828",
     "kernel_cutlass_gdn_decode_bf16state_mtp_ilp4_kernel"),
    # templated C++ kernels: the qualified name, not the template args
    ("void flash::FlashAttnFwdSm90<int, 128, true>(flash::Params)", "flash::FlashAttnFwdSm90"),
    ("void deep_gemm::sm90_fp8_mqa_logits<64u, 128u, false>(int)", "deep_gemm::sm90_fp8_mqa_logits"),
])
def test_normalize_kernel(pd, raw, expected):
    assert pd.normalize_kernel(raw) == expected


def test_normalize_drops_denied_names(pd):
    assert pd.normalize_kernel("Memcpy DtoH (Device -> Pinned)") is None
    # raw CUDA runtime copies surface as lowercase kernels under graphs (sglang)
    assert pd.normalize_kernel("memcpy128") is None
    assert pd.normalize_kernel("memset32") is None


@pytest.mark.parametrize("kernel, family", [
    ("flash::FlashAttnFwdSm90", "fa3"),
    ("kernel_unified_attention", "triton"),
    ("_fwd_grouped_kernel_stage1", "triton"),
    ("_topk_index_merge_kernel", "triton"),
    ("flash_fwd_splitkv_mla_fp8_sparse_kernel", "flashmla"),
    ("deep_gemm::sm90_fp8_paged_mqa_logits", "dsa_nsa"),
    ("deep_gemm::sm90_fp8_gemm_1d2d_impl", "deepgemm"),
    ("nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN", "cublas"),
    ("fmha_v2_flash_attention_bf16_64_128_S_qkv_128_causal_tma_ws_sm90_kernel", "trtllm_mha"),
    ("fused_moe_kernel", "triton_fused_moe"),
    ("chunk_kda_fwd_kernel_inter_solve_fused", "fla_triton"),
])
def test_taxonomy_labels_identity_kernels(pd, kernel, family):
    labels, unmatched = pd.label_kernels([kernel])
    assert family in labels and not unmatched


@pytest.mark.parametrize("kernel", [
    "triton_poi_fused_mul_silu_slice_1",   # torch.compile Inductor fusion
    "triton_red_fused_fused_add_rms_norm_0",
    "_fused_dsa_decode_metadata_kernel",   # sglang graph-mode metadata
    "compute_position_kernel",
    "_min_p_kernel",                       # sampler
    "Compiled",                            # profiler region label
])
def test_taxonomy_glue_carries_no_identity(pd, kernel):
    labels, unmatched = pd.label_kernels([kernel])
    assert labels == set() and not unmatched, (labels, unmatched)


def test_only_attention_rules_emit_triton(pd):
    # the one-vocabulary contract: `triton` names the Triton ATTENTION backend
    for rx, backend, role in pd._TAXONOMY:
        if backend == "triton":
            assert role == "attention", rx.pattern


def test_orphans_keep_every_labeled_kernel_including_infra_families(pd):
    # framework-mode probes: no spans, GEMM/quant kernels arrive as orphans and
    # path_diff grades gemm-class ops on their names — they must survive
    noise = [{"kernel": f"void at::native::unrolled_elementwise_kernel_{i}<int>(int)", "us": 100 - i}
             for i in range(60)]
    facts = {"decode_kernels": noise + [
        {"kernel": "nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN", "us": 1.0},
        {"kernel": "void vllm::scaled_fp8_quant_kernel_strided<c10::BFloat16>(int)", "us": 0.5},
        {"kernel": "void flash::FlashAttnFwdSm90<int>(flash::Params)", "us": 0.1},
    ]}
    ops, orphans = pd.build_ops(facts)
    assert ops == []
    assert "nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN" in orphans
    assert "vllm::scaled_fp8_quant_kernel_strided" in orphans
    assert "flash::FlashAttnFwdSm90" in orphans
    # unlabeled remainder is capped by device time, never the labeled ones
    assert len(orphans) <= 3 + pd._ORPHAN_REST_CAP


def test_spans_and_launchers_are_not_orphans(pd):
    facts = {"prefill_kernels": [
        {"kernel": "AIC::attn::FlashAttentionImpl", "us": 5},
        {"kernel": "_vllm_fa3_C::fwd", "us": 5},
        {"kernel": "step 3", "us": 5},
        {"kernel": "void flash::FlashAttnFwdSm90<int>(flash::Params)", "us": 5},
    ]}
    _, orphans = pd.build_ops(facts)
    assert orphans == ["flash::FlashAttnFwdSm90"]


def _targets_for(repo: str, variants: list[str]) -> dict:
    return {
        "topologies": [{"tp": 1, "evidence": "real"}],
        "backends": {"vllm": {"versions": ["0.29.0"], "images": {"0.29.0": "img"}}},
        "families": {"fam": {"checkpoints": [{"repo": repo, "profile": "bfloat16", "variants": variants}]}},
    }


def test_capacity_fallback_advances_past_a_load_oom(pd, tmp_path, monkeypatch):
    """The representative dummy cut is the FIRST variant unless its raw probe
    OOMed at engine load on this backend; then the next smaller faithful cut is
    queued and the switch is recorded on the run (never silent)."""
    import json
    monkeypatch.setattr(pd, "ROOT", tmp_path)
    for v in ("depth8", "depth4"):
        (tmp_path / "dummy_models" / "generic" / f"Big__{v}").mkdir(parents=True)
    targets = _targets_for("org/Big", ["depth8", "depth4"])
    # no evidence yet: index 0 is the representative
    runs = [r for r in pd.enumerate_runs(targets, full=False, backends=["vllm"]) if "skip" not in r]
    assert {r["variant"] for r in runs} == {"depth8"}
    assert all("capacity_fallback_from" not in r for r in runs)
    # the depth8 probe OOMed at load -> depth4 becomes the representative
    oom_rid = next(r["id"] for r in runs if r["kv_dtype"] is None)
    (tmp_path / "archive" / "raw").mkdir(parents=True)
    (tmp_path / "archive" / "raw" / f"{oom_rid}.json").write_text(json.dumps(
        {"errors": {"load": "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 GiB"}}))
    runs = [r for r in pd.enumerate_runs(targets, full=False, backends=["vllm"]) if "skip" not in r]
    assert {r["variant"] for r in runs} == {"depth4"}
    assert {r["capacity_fallback_from"] for r in runs} == {"depth8"}
    # a non-OOM failure is NOT a capacity signal: the representative stays
    (tmp_path / "archive" / "raw" / f"{oom_rid}.json").write_text(json.dumps(
        {"errors": {"load": "RuntimeError: CUDA error: an illegal memory access was encountered"}}))
    runs = [r for r in pd.enumerate_runs(targets, full=False, backends=["vllm"]) if "skip" not in r]
    assert {r["variant"] for r in runs} == {"depth8"}


def test_capacity_fallback_stops_at_the_smallest_cut(pd, tmp_path, monkeypatch):
    """When every cut OOMed the smallest one stays queued (its failure is the
    honest matrix cell) — the fallback never invents a cut that does not exist."""
    import json
    monkeypatch.setattr(pd, "ROOT", tmp_path)
    for v in ("depth8", "depth4"):
        (tmp_path / "dummy_models" / "generic" / f"Big__{v}").mkdir(parents=True)
    (tmp_path / "archive" / "raw").mkdir(parents=True)
    ck = {"repo": "org/Big", "profile": "bfloat16"}
    # both OOM phrasings count: torch's and the trtllm executor's
    for v, msg in (("depth8", "CUDA out of memory"),
                   ("depth4", "RuntimeError: Executor creation failed due to insufficient GPU memory.")):
        rid = pd._run_id(ck, v, "vllm", "0.29.0", 1, None)
        (tmp_path / "archive" / "raw" / f"{rid}.json").write_text(json.dumps({"errors": {"load": msg}}))
    runs = [r for r in pd.enumerate_runs(_targets_for("org/Big", ["depth8", "depth4"]), full=False, backends=["vllm"])
            if "skip" not in r]
    assert {r["variant"] for r in runs} == {"depth4"}
    assert {r["capacity_fallback_from"] for r in runs} == {"depth8"}
