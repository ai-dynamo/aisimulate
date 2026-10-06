# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""probe_driver record rules: kernel-name normalization, the taxonomy
contract, and the orphan keep rule that framework-mode probes depend on."""
import importlib.util
import json
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
    # anonymous-namespace kernels: the kernel, not the namespace words (Kimi-K3 KDA decode)
    ("void (anonymous namespace)::kda_decode_fusion_many_heads_kernel<true, true, 96, 96>(int)",
     "kda_decode_fusion_many_heads_kernel"),
    # plain symbol with a parameter-type tail (sglang DSA metadata scheduler)
    ("(anonymous namespace)::smxx_paged_mqa_logits_metadata(MetadataParams)", "smxx_paged_mqa_logits_metadata"),
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
        rid = pd._run_id(ck, v, "vllm", "0.29.0", 1, None, "h20_sm90")  # ids carry the platform (default h20_sm90 without a targets platform)
        (tmp_path / "archive" / "raw" / f"{rid}.json").write_text(json.dumps({"errors": {"load": msg}}))
    runs = [r for r in pd.enumerate_runs(_targets_for("org/Big", ["depth8", "depth4"]), full=False, backends=["vllm"])
            if "skip" not in r]
    assert {r["variant"] for r in runs} == {"depth4"}
    assert {r["capacity_fallback_from"] for r in runs} == {"depth8"}


def test_customization_sms_scope(pd, monkeypatch):
    """`sms:` on a cli_extra_args entry limits it to those SMs (sm120 needs bf16 KV
    for NVFP4 MLA checkpoints; sm90/sm100 must not inherit it)."""
    entry = {"args": ["--generator-set", "x=1"], "fact": "f", "sms": ["sm120"]}
    assert pd._cea(entry, "sm120") == ["--generator-set", "x=1"]
    assert pd._cea(entry, "sm90") == []
    assert pd._cea({"args": ["-a"], "fact": "f"}, "sm90") == ["-a"]
    assert pd._cea(["-b"], "sm120") == ["-b"]


def test_run_id_carries_the_platform(pd):
    ck = {"repo": "org/m", "profile": "fp8"}
    legacy = pd._run_id(ck, "v", "vllm", "0.30.0", 1, None)
    h20 = pd._run_id(ck, "v", "vllm", "0.30.0", 1, None, "h20_sm90")
    b200 = pd._run_id(ck, "v", "vllm", "0.30.0", 1, None, "b200_sm100")
    assert len({legacy, h20, b200}) == 3 and len(h20) == 12


def test_migrate_run_ids_renames_artifacts_and_rewrites_ids(pd, tmp_path):
    ws = tmp_path / "ws"; (ws / "archive" / "raw").mkdir(parents=True); (ws / "archive" / "run_sh").mkdir()
    ck = {"repo": "org/m", "profile": "fp8"}
    old = pd._run_id(ck, "v", "vllm", "0.30.0", 1, None)
    new = pd._run_id(ck, "v", "vllm", "0.30.0", 1, None, "h20_sm90")
    run = {"id": old, "repo": "org/m", "profile": "fp8", "variant": "v", "backend": "vllm", "version": "0.30.0",
           "tp": 1, "kv_dtype": None, "platform": "h20_sm90"}
    (ws / "archive" / "plan_vllm.json").write_text(json.dumps([run]))
    (ws / "archive" / "raw" / f"{old}.json").write_text(json.dumps({"provenance": {"id": old}, "ok": True}))
    (ws / "archive" / "raw" / f"{old}.fp").write_text("deadbeef")
    (ws / "archive" / "run_sh" / f"{old}.sh").write_text("#!/bin/bash\n")
    (ws / "archive" / "records.jsonl").write_text(json.dumps({"id": old, "target": {"repo": "org/m"}}) + "\n")
    assert pd.migrate_run_ids(ws, "h20_sm90", apply=False)["runs"] == 1
    st = pd.migrate_run_ids(ws, "h20_sm90", apply=True)
    assert st["renamed"] == 3
    assert (ws / "archive" / "raw" / f"{new}.json").exists() and not (ws / "archive" / "raw" / f"{old}.json").exists()
    assert json.loads((ws / "archive" / "raw" / f"{new}.json").read_text())["provenance"]["id"] == new
    assert json.loads((ws / "archive" / "plan_vllm.json").read_text())[0]["id"] == new
    assert json.loads((ws / "archive" / "records.jsonl").read_text().strip())["id"] == new
    assert pd.migrate_run_ids(ws, "h20_sm90", apply=False)["runs"] == 0  # idempotent


def test_attention_identity_uses_taxonomy_backend_labels(pd):
    """The identity column shows the backend label, not an 80-char cubin name."""
    assert pd.attn_identity_label("flash::FlashAttnFwdSm90<...>") == "fa3"
    unknown = "some_totally_unknown_kernel_name"
    assert pd.attn_identity_label(unknown) == pd.normalize_kernel(unknown)


# --- golden render facts status + dummy adapter guard (sm120 V4.1 re-probe, 2026-10-01) ---

def test_golden_facts_status_flags_the_swallowed_resolution_failure(pd):
    log = ('WARNING pipeline.py:75 Fact resolution failed; continuing without facts.\n'
           'Traceback (most recent call last):\n  File "resolve.py", line 104\n'
           'KeyError: "Unknown hardware profile \'rtx_pro_6000_server\'. Available profiles: [\'b200\', \'h200\']"\n'
           'Generated 4 artifacts\n')
    st = pd.golden_facts_status(log)
    assert st["applied"] is False and "rtx_pro_6000_server" in st["reason"]
    assert pd.golden_facts_status("Generated 4 artifacts\n") == {"applied": True}
    assert pd.golden_facts_status("") == {"applied": True}


def _dummy_tree(tmp_path, *leaves):
    root = tmp_path / "dummy_models"
    for fam, leaf in leaves:
        (root / fam / leaf).mkdir(parents=True)
    return root


def test_select_dummy_dir_refuses_another_adapter_when_override_declared(pd, tmp_path):
    root = _dummy_tree(tmp_path, ("generic", "DeepSeek-V4.1-Flash__rep"))
    ck = {"dummy_overrides": {"family": "dsv41"}}
    vdir, why = pd.select_dummy_dir(root, "DeepSeek-V4.1-Flash", "rep", "roster", {}, ck)
    assert vdir is None and "generic" in why and "dsv41" in why and "dummies.py" in why
    (root / "dsv41" / "DeepSeek-V4.1-Flash__rep").mkdir(parents=True)
    vdir, why = pd.select_dummy_dir(root, "DeepSeek-V4.1-Flash", "rep", "roster", {}, ck)
    assert why is None and vdir.parent.name == "dsv41"


def test_select_dummy_dir_default_order_is_family_then_generic_then_any(pd, tmp_path):
    root = _dummy_tree(tmp_path, ("generic", "M__rep"), ("glm", "M__rep"), ("zzz", "N__rep"))
    assert pd.select_dummy_dir(root, "M", "rep", "glm", {}, {})[0].parent.name == "glm"
    assert pd.select_dummy_dir(root, "M", "rep", "roster", {}, {})[0].parent.name == "generic"
    assert pd.select_dummy_dir(root, "N", "rep", "roster", {}, {})[0].parent.name == "zzz"
    vdir, why = pd.select_dummy_dir(root, "Q", "rep", "roster", {}, {})
    assert vdir is None and why.startswith("no dummy dir")
