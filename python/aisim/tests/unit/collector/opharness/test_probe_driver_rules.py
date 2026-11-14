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


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Triton autotune tile suffix is not identity
        ("_matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1", "_matmul_ogs_NNT_bf16xbf16xmxfp4"),
        ("_matmul_ogs_NNT_bf16xbf16xmxfp4_128x256x128x1", "_matmul_ogs_NNT_bf16xbf16xmxfp4"),
        # cute-DSL instantiations bake shape params into the symbol
        (
            "kernel_cutlass_gdn_decode_bf16state_mtp_ilp4_kernel_tensorptrbf16gmemalign32o1291612812828",
            "kernel_cutlass_gdn_decode_bf16state_mtp_ilp4_kernel",
        ),
        # templated C++ kernels: the qualified name, not the template args
        ("void flash::FlashAttnFwdSm90<int, 128, true>(flash::Params)", "flash::FlashAttnFwdSm90"),
        ("void deep_gemm::sm90_fp8_mqa_logits<64u, 128u, false>(int)", "deep_gemm::sm90_fp8_mqa_logits"),
        # anonymous-namespace kernels: the kernel, not the namespace words (Kimi-K3 KDA decode)
        (
            "void (anonymous namespace)::kda_decode_fusion_many_heads_kernel<true, true, 96, 96>(int)",
            "kda_decode_fusion_many_heads_kernel",
        ),
        # plain symbol with a parameter-type tail (sglang DSA metadata scheduler)
        ("(anonymous namespace)::smxx_paged_mqa_logits_metadata(MetadataParams)", "smxx_paged_mqa_logits_metadata"),
    ],
)
def test_normalize_kernel(pd, raw, expected):
    assert pd.normalize_kernel(raw) == expected


def test_normalize_drops_denied_names(pd):
    assert pd.normalize_kernel("Memcpy DtoH (Device -> Pinned)") is None
    # raw CUDA runtime copies surface as lowercase kernels under graphs (sglang)
    assert pd.normalize_kernel("memcpy128") is None
    assert pd.normalize_kernel("memset32") is None


@pytest.mark.parametrize(
    "kernel, family",
    [
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
        # triton_kernels' MoE matmul under its current name (SGLang triton_kernel runner,
        # TRT-LLM TritonFusedMoE), and the earlier _matmul_ogs_ one
        ("_matmul_NNT_bf16xbf16xmxfp4_swiglu", "triton_fused_moe"),
        ("_matmul_NNT_bf16xbf16xmxfp4", "triton_fused_moe"),
        ("_matmul_ogs_NNT_bf16xbf16xmxfp4", "triton_fused_moe"),
        ("chunk_kda_fwd_kernel_inter_solve_fused", "fla_triton"),
    ],
)
def test_taxonomy_labels_identity_kernels(pd, kernel, family):
    labels, unmatched = pd.label_kernels([kernel], "sm90")
    assert family in labels and not unmatched


@pytest.mark.parametrize("sm", ["sm89", "sm90", "sm100", "sm103", "sm120"])
@pytest.mark.parametrize(
    "kernel",
    [
        "moe_align_block_size_kernel",
        "moe_sum_reduce_kernel",
        # SGLang's Triton align (top-k numel <= 64) and sum-reduce
        "_moe_align_small_numel_kernel",
        "_moe_sum_reduce_kernel",
    ],
)
def test_taxonomy_labels_moe_align_and_sum_reduce_as_moe_infra(pd, kernel, sm):
    assert pd.kernel_role(kernel, sm) == ("sgl_kernel", "moe_infra")


@pytest.mark.parametrize(
    "kernel",
    [
        "triton_poi_fused_mul_silu_slice_1",  # torch.compile Inductor fusion
        "triton_red_fused_fused_add_rms_norm_0",
        "_fused_dsa_decode_metadata_kernel",  # sglang graph-mode metadata
        "compute_position_kernel",
        "_min_p_kernel",  # sampler
        "Compiled",  # profiler region label
    ],
)
def test_taxonomy_glue_carries_no_identity(pd, kernel):
    labels, unmatched = pd.label_kernels([kernel], "sm90")
    assert labels == set() and not unmatched, (labels, unmatched)


@pytest.mark.parametrize("sm", ["sm89", "sm90", "sm100", "sm103", "sm120"])
def test_only_attention_rules_emit_triton(pd, sm):
    # the one-vocabulary contract: `triton` names the Triton ATTENTION backend
    for rx, backend, role in pd.taxonomy(sm):
        if backend == "triton":
            assert role == "attention", rx.pattern


def test_sm_scoped_rules_label_only_on_their_sm(pd):
    # a record is labeled with the rules of the SM its raw ran on, not a configured box
    sm89_only = next(
        r
        for r in __import__("yaml").safe_load((COMPONENTS / "kernel_taxonomy.yaml").read_text())["rules"]
        if r.get("sms") == ["sm89"]
    )
    assert any(rx.pattern == sm89_only["match"] for rx, _, _ in pd.taxonomy("sm89"))
    assert not any(rx.pattern == sm89_only["match"] for rx, _, _ in pd.taxonomy("sm100"))


def test_orphans_keep_every_labeled_kernel_including_infra_families(pd):
    # framework-mode probes: no spans, GEMM/quant kernels arrive as orphans and
    # path_diff grades gemm-class ops on their names — they must survive
    noise = [
        {"kernel": f"void at::native::unrolled_elementwise_kernel_{i}<int>(int)", "us": 100 - i} for i in range(60)
    ]
    facts = {
        "decode_kernels": noise
        + [
            {"kernel": "nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN", "us": 1.0},
            {"kernel": "void vllm::scaled_fp8_quant_kernel_strided<c10::BFloat16>(int)", "us": 0.5},
            {"kernel": "void flash::FlashAttnFwdSm90<int>(flash::Params)", "us": 0.1},
        ]
    }
    ops, orphans = pd.build_ops(facts, "sm90")
    assert ops == []
    assert "nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_TNN" in orphans
    assert "vllm::scaled_fp8_quant_kernel_strided" in orphans
    assert "flash::FlashAttnFwdSm90" in orphans
    # unlabeled remainder is capped by device time, never the labeled ones
    assert len(orphans) <= 3 + pd._ORPHAN_REST_CAP


def test_spans_and_launchers_are_not_orphans(pd):
    facts = {
        "prefill_kernels": [
            {"kernel": "AIC::attn::FlashAttentionImpl", "us": 5},
            {"kernel": "_vllm_fa3_C::fwd", "us": 5},
            {"kernel": "step 3", "us": 5},
            {"kernel": "void flash::FlashAttnFwdSm90<int>(flash::Params)", "us": 5},
        ]
    }
    _, orphans = pd.build_ops(facts, "sm90")
    assert orphans == ["flash::FlashAttnFwdSm90"]


def _targets_for(repo: str, variants: list[str], **extra) -> dict:
    return {
        "topologies": [{"tp": 1, "evidence": "real"}, {"tp": 2, "evidence": "real"}],
        "backends": {"vllm": {"default_version": "0.29.0", "versions": ["0.29.0"], "images": {"0.29.0": "img"}}},
        "families": {"fam": {"checkpoints": [{"repo": repo, "profile": "bfloat16", "variants": variants}]}},
        **extra,
    }


def test_a_campaign_renders_and_identifies_by_its_system(pd, tmp_path, monkeypatch):
    """The system names the render, the SM and the run id: the same case on h200 and on
    b200 is two runs, so two campaigns' raws never overwrite each other."""
    monkeypatch.setattr(pd, "ROOT", tmp_path)
    (tmp_path / "dummy_models" / "generic" / "M__rep").mkdir(parents=True)
    targets = _targets_for("org/M", ["rep"])
    h200 = [r for r in pd.enumerate_runs(targets, "h200_sxm", False, ["vllm"]) if "skip" not in r]
    b200 = [r for r in pd.enumerate_runs(targets, "b200_sxm", False, ["vllm"]) if "skip" not in r]
    assert {(r["system"], r["sm"]) for r in h200} == {("h200_sxm", "sm90")}
    assert {(r["system"], r["sm"]) for r in b200} == {("b200_sxm", "sm100")}
    assert not {r["id"] for r in h200} & {r["id"] for r in b200}
    assert pd.system_sm("b300_sxm") == "sm103" and pd.system_sm("rtx_pro_6000_server") == "sm120"


def test_the_representative_cut_is_the_first_variant_unless_overridden(pd, tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "ROOT", tmp_path)
    for v in ("depth8", "depth4"):
        (tmp_path / "dummy_models" / "generic" / f"Big__{v}").mkdir(parents=True)
    targets = _targets_for("org/Big", ["depth8", "depth4"])
    runs = [r for r in pd.enumerate_runs(targets, "h200_sxm", False, ["vllm"]) if "skip" not in r]
    assert {r["variant"] for r in runs} == {"depth8"} and {r["tp"] for r in runs} == {1}
    targets["families"]["fam"]["checkpoints"][0]["variant_overrides"] = {"vllm": "depth4"}
    runs = [r for r in pd.enumerate_runs(targets, "h200_sxm", False, ["vllm"]) if "skip" not in r]
    assert {r["variant"] for r in runs} == {"depth4"}


def test_tp_required_scoped_by_the_campaign_sm(pd, tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "ROOT", tmp_path)
    (tmp_path / "dummy_models" / "generic" / "K__rep").mkdir(parents=True)
    policy = {"tp_required": [{"repo": "org/K", "backend": "vllm", "tp": 2, "sms": [100]}]}
    targets = _targets_for("org/K", ["rep"], topology_policy=policy)
    tps = lambda system: {r["tp"] for r in pd.enumerate_runs(targets, system, False, ["vllm"]) if "skip" not in r}
    assert tps("b200_sxm") == {2} and tps("h200_sxm") == {1}


def test_customization_sms_scope(pd, monkeypatch):
    """`sms:` on a cli_extra_args entry limits it to those SMs (sm120 needs bf16 KV
    for NVFP4 MLA checkpoints; sm90/sm100 must not inherit it)."""
    entry = {"args": ["--generator-set", "x=1"], "fact": "f", "sms": ["sm120"]}
    assert pd._cea(entry, "sm120") == ["--generator-set", "x=1"]
    assert pd._cea(entry, "sm90") == []
    assert pd._cea({"args": ["-a"], "fact": "f"}, "sm90") == ["-a"]
    assert pd._cea(["-b"], "sm120") == ["-b"]


def test_the_marlin_rescue_of_nvfp4_deepseek_v4_holds_on_sm90_only(pd):
    """SGLang picks its NVFP4 runner for these checkpoints only while the runner is auto, on every
    SM; a forced marlin on SM100 would record weight-only kernels where serving runs NVFP4."""
    import yaml

    targets = yaml.safe_load((COMPONENTS.parent / "targets.yaml").read_text())
    entries = {}
    for fam in targets["families"].values():
        for ck in fam.get("checkpoints") or []:
            entries[ck["repo"]] = (ck.get("cli_extra_args") or {}).get("sglang")
        for repo, override in (fam.get("checkpoint_overrides") or {}).items():
            entries[repo] = ((override or {}).get("cli_extra_args") or {}).get("sglang")
    forced = {repo: v for repo, v in entries.items() if v and "marlin" in str(v)}
    assert set(forced) == {
        "nvidia/DeepSeek-V4-Pro-NVFP4",
        "nvidia/DeepSeek-V4-Flash-NVFP4",
        "nvidia/DeepSeek-V4.1-Flash-NVFP4",
    }
    for v in forced.values():
        assert pd._cea(v, "sm90") == ["--generator-set", "params.agg.moe_backend=marlin"]
        assert [pd._cea(v, sm) for sm in ("sm89", "sm100", "sm103", "sm120")] == [[], [], [], []]


def test_customization_list_scopes_each_entry(pd):
    """A list of entries applies each on its own SMs: DeepSeek-V4's page size holds on
    every SM, its fp8_ds_mla KV only on Hopper (rc2 takes it off Hopper only at 256)."""
    entries = [
        {"args": ["--generator-set", "params.agg.tokens_per_block=128"], "fact": "f"},
        {"args": ["--generator-set", "params.agg.kv_cache_dtype=fp8_ds_mla"], "fact": "f", "sms": ["sm90"]},
    ]
    assert pd._cea(entries, "sm90") == [
        "--generator-set",
        "params.agg.tokens_per_block=128",
        "--generator-set",
        "params.agg.kv_cache_dtype=fp8_ds_mla",
    ]
    assert pd._cea(entries, "sm100") == ["--generator-set", "params.agg.tokens_per_block=128"]


def test_run_id_carries_the_system_and_kv(pd):
    ck = {"repo": "org/m", "profile": "fp8"}
    ids = {
        pd._run_id(ck, "v", "vllm", "0.30.0", 1, kv, sys) for kv in (None, "fp8") for sys in ("h200_sxm", "b200_sxm")
    }
    assert len(ids) == 4 and all(len(i) == 12 for i in ids)


# --- golden render facts status + dummy adapter guard (sm120 V4.1 re-probe, 2026-10-01) ---


def test_golden_facts_status_flags_the_swallowed_resolution_failure(pd):
    log = (
        "WARNING pipeline.py:75 Fact resolution failed; continuing without facts.\n"
        'Traceback (most recent call last):\n  File "resolve.py", line 104\n'
        "KeyError: \"Unknown hardware profile 'rtx_pro_6000_server'. Available profiles: ['b200', 'h200']\"\n"
        "Generated 4 artifacts\n"
    )
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


def test_the_roster_floor_is_every_model_path_the_case_yamls_declare(pd):
    repos = pd.collector_mentioned_repos()
    # orgs a fixed org list did not know, and a repo named only in a comment
    assert {"stepfun-ai/Step-3.7-Flash", "RadixArk/Qwen3.8-2.4T-A95B-NVFP4", "meta-models/Muse-Glimmer-30B"} <= repos
    assert "zai-org/GLM-5.3-FP8" not in repos


def test_a_roster_repo_without_a_dummy_needs_no_config(pd, tmp_path, monkeypatch):
    """--only DeepSeek-V3 also matches nvidia/DeepSeek-V3.1-NVFP4: a repo nobody built a dummy for
    is reported as skipped, not a stop for the whole campaign; a dummy without its config is."""
    monkeypatch.setattr(pd, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "dummy_models").mkdir()
    manifest = {"variants": [{"repo": "org/built", "variant": "built__rep"}]}
    (tmp_path / "dummy_models" / "variants_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(pd, "derived_repos", lambda fam, targets: ["org/built", "org/unbuilt"])
    with pytest.raises(SystemExit, match="org/built has dummies but no fetched config"):
        pd.derive_roster_checkpoints({}, {})
    monkeypatch.setattr(pd, "derive_profile", lambda repo, configs: "bf16")
    cks = {ck["repo"]: ck for ck in pd.derive_roster_checkpoints({}, {})}
    assert cks["org/built"]["variants"] == ["rep"] and cks["org/unbuilt"]["variants"] == []
