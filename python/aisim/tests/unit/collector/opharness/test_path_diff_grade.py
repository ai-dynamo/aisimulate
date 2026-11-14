# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""path_diff.grade — the verdict rule as a pure function, graded on the gate's
TARGET ROLES (review 2026-09-25 P1): auxiliary kernel overlap never proves a
gate, a role the collector ran must exist on the serving side, kernel drift
inside a role stays red, serving evidence is selected by PHASE, and auxiliary
work in the timed runs that serving never does fails the gate (2026-11-08).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, COMPONENTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


path_diff = _load("path_diff")

# a fixed toy taxonomy (backend, role) so the RULE is tested, not the yaml
_LABELS = {
    "flash::FlashAttnFwdSm90": ("fa3", "attention"),
    "flash::FlashAttnFwdCombine": ("fa3", "attention"),
    "kernel_unified_attention": ("triton", "attention"),
    "deep_gemm::sm90_fp8_mqa_logits": ("dsa_nsa", "attention"),
    "deep_gemm::sm90_fp8_paged_mqa_logits": ("dsa_nsa", "attention"),
    "nvjet_sm90_tst_128x256": ("cublas", "gemm"),
    "nvjet_sm90_tst_64x64": ("cublas", "gemm"),
    "nvjet_sm90_wrong_gemm": ("cublas", "gemm"),
    "nvjet_sm90_actual_gemm": ("cublas", "gemm"),
    "vllm::scaled_fp8_quant_kernel": ("vllm_kernel", "quant"),
    "per_token_group_quant_8bit_kernel": ("sgl_kernel", "quant"),
    "reshape_and_cache_kernel_flash": ("framework_native", "kvcache"),
    "deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl": ("deepgemm", "gemm"),
    "deep_gemm::pack_fp32_into_ue8m0": ("deepgemm", "quant"),
    "triton_per_fused__to_copy_abs_ceil_clamp_div_exp2_log2_max_view_0": ("framework_native", "activation"),
    "triton_": ("infra", "infra"),
    "vllm::per_token_group_quant_8bit_packed_kernel": ("vllm_kernel", "quant"),
    "deep_gemm::transpose_and_pack_fp32_into_ue8m0": ("deepgemm", "gemm"),
}


def role_of(k):
    return _LABELS.get(k)


def grade(col, srv, roles, cap_error=None, measured=None, srv_every=None):
    every = None if srv_every is None else set(srv_every)
    return path_diff.grade(sorted(col), set(srv), roles, cap_error, role_of, measured, every)


def test_subset_semantics_collector_one_op_serving_whole_model():
    col = ["flash::FlashAttnFwdSm90"]
    srv = ["flash::FlashAttnFwdSm90", "nvjet_sm90_tst_128x256", "per_token_group_quant_8bit_kernel"]
    g = grade(col, srv, ("attention",))
    assert g["verdict"] == "aligned" and g["only_col"] == [] and g["kernel_drift"] is None
    assert g["role_evidence"]["attention"]["matched"] == ["flash::FlashAttnFwdSm90"]


def test_collector_only_backend_family_inside_the_target_role_is_diverged():
    g = grade(["kernel_unified_attention"], ["flash::FlashAttnFwdSm90"], ("attention",))
    assert g["verdict"] == "diverged" and g["only_col"] == ["attention:triton"]


def test_same_family_different_kernel_is_drift():
    # 0.29 DSA indexer: same dsa_nsa family, prefill vs paged logits kernel
    g = grade(["deep_gemm::sm90_fp8_mqa_logits"], ["deep_gemm::sm90_fp8_paged_mqa_logits"], ("attention",))
    assert g["verdict"] == "diverged"
    assert g["kernel_drift"]["attention"]["collector_only_kernels"] == ["deep_gemm::sm90_fp8_mqa_logits"]


def test_crashed_capture_is_invalid_not_aligned():
    g = grade([], ["flash::FlashAttnFwdSm90"], ("attention",), cap_error="RuntimeError: boom")
    assert g["verdict"] == "invalid-capture"


def test_empty_capture_is_no_collector_signal():
    g = grade([], ["flash::FlashAttnFwdSm90"], ("attention",))
    assert g["verdict"] == "no-collector-signal"


def test_gemm_gate_graded_on_gemm_kernel_names():
    # cuBLAS IS the backend: aligned iff a GEMM instantiation serving launched appears
    g = grade(["nvjet_sm90_tst_128x256"], ["nvjet_sm90_tst_128x256", "flash::FlashAttnFwdSm90"], ("gemm",))
    assert g["verdict"] == "aligned"
    assert g["role_evidence"]["gemm"]["matched"] == ["nvjet_sm90_tst_128x256"]


def test_review_p1_different_gemm_shared_quant_kernel_is_not_aligned():
    # the reviewed counterexample: the quant kernel overlaps, the GEMM does not
    col = ["nvjet_sm90_wrong_gemm", "vllm::scaled_fp8_quant_kernel"]
    srv = ["nvjet_sm90_actual_gemm", "vllm::scaled_fp8_quant_kernel"]
    g = grade(col, srv, ("gemm",))
    assert g["verdict"] == "diverged"
    assert g["kernel_drift"]["gemm"]["collector_only_kernels"] == ["nvjet_sm90_wrong_gemm"]


def test_only_auxiliary_kernels_cannot_prove_a_gemm_gate():
    g = grade(["vllm::scaled_fp8_quant_kernel"], ["nvjet_sm90_actual_gemm", "vllm::scaled_fp8_quant_kernel"], ("gemm",))
    assert g["verdict"] == "no-collector-signal"


def test_quant_gate_with_fused_serving_quant_is_diverged_not_aligned():
    # compute_scale under torch.compile: serving fuses the quant, the role is absent
    g = grade(["vllm::scaled_fp8_quant_kernel"], ["nvjet_sm90_actual_gemm"], ("quant",))
    assert g["verdict"] == "diverged" and g["missing_roles"] == ["quant"]


def test_gemm_tile_differences_outside_the_target_role_never_count():
    # serving ran a GEMM tile the collector did not: not a divergence of an attention gate
    g = grade(
        ["flash::FlashAttnFwdSm90", "nvjet_sm90_tst_64x64"],
        ["flash::FlashAttnFwdSm90", "nvjet_sm90_tst_128x256"],
        ("attention",),
    )
    assert g["verdict"] == "aligned"
    assert g["aux_drift"] is None  # no timed-run kernels recorded: auxiliary work unchecked


def test_unscoped_gate_grades_every_non_glue_role_the_collector_ran():
    g = grade(
        ["flash::FlashAttnFwdSm90", "vllm::scaled_fp8_quant_kernel", "reshape_and_cache_kernel_flash"],
        ["flash::FlashAttnFwdSm90"],
        None,
    )
    assert g["target_roles"] == ["attention", "quant"] and g["verdict"] == "diverged"
    assert g["missing_roles"] == ["quant"]


# --------------------------------------------------------------------- phase


def test_review_p1_prefill_gate_cannot_borrow_decode_evidence():
    # record: FA3 ran in prefill only, the Triton unified kernel in decode only
    record = {
        "ops": [],
        "orphan_kernels": ["flash::FlashAttnFwdSm90", "kernel_unified_attention"],
        "kernel_phases": {"flash::FlashAttnFwdSm90": ["prefill"], "kernel_unified_attention": ["decode"]},
    }
    srv, scoped = path_diff.select_serving(record, "prefill")
    assert srv == {"flash::FlashAttnFwdSm90"} and scoped is True
    # a decode-only capture graded as a PREFILL gate: diverged (collector-only family)
    g = grade(["kernel_unified_attention"], srv, ("attention",))
    assert g["verdict"] == "diverged"
    # the same capture as a decode gate: aligned
    srv_dec, _ = path_diff.select_serving(record, "decode")
    assert grade(["kernel_unified_attention"], srv_dec, ("attention",))["verdict"] == "aligned"


def test_profile_run_evidence_is_a_separate_phase():
    record = {
        "ops": [],
        "orphan_kernels": ["flash::FlashAttnFwdSm90"],
        "kernel_phases": {"flash::FlashAttnFwdSm90": ["profile_run"]},
    }
    assert path_diff.select_serving(record, "prefill")[0] == set()
    assert path_diff.select_serving(record, "profile_run")[0] == {"flash::FlashAttnFwdSm90"}


def test_a_span_only_kernel_is_kept_but_leaves_the_selection_unscoped():
    record = {
        "ops": [{"phase": "decode", "kernels": ["kernel_unified_attention", "flash::FlashAttnFwdSm90"]}],
        "kernel_phases": {"kernel_unified_attention": ["decode"]},
    }
    srv, scoped = path_diff.select_serving(record, "prefill")
    assert srv == {"flash::FlashAttnFwdSm90"} and scoped is False


def test_serving_records_match_the_sm_kv_class_and_prompt_length(tmp_path, monkeypatch):
    """A gate graded on SM100 never reads an SM90 record; kv auto and bf16 are one class; a gate
    that names a prompt length reads only records probed at it; the newest import wins."""
    monkeypatch.setattr(path_diff.probe_driver, "ROOT", tmp_path)

    def rec(rid, sm, kv="auto", isl=4096, at=0.0):
        rt = {"backend": "vllm", "version": "1", "sm": sm, "kv_cache_dtype": kv, "isl": isl}
        return {"id": rid, "target": {"repo": "org/m"}, "runtime": rt, "outcome": {"status": "ok"}, "imported_at": at}

    for name, recs in {
        "h200": [rec("a", "sm90"), rec("b", "sm90", isl=512)],
        "b200": [rec("c", "sm100", at=1.0), rec("d", "sm100", at=2.0), rec("e", "sm100", kv="fp8")],
    }.items():
        (tmp_path / "campaigns" / name).mkdir(parents=True)
        (tmp_path / "campaigns" / name / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    ids = lambda *a, **k: [r["id"] for r in path_diff.serving_records("org/m", "vllm", "1", *a, **k)]
    assert ids("sm100", "bfloat16") == ["d", "c"]
    assert ids("sm100", "fp8") == ["e"]
    assert ids("sm90", "auto", 512) == ["b"]
    assert ids("sm103", "auto") == []


def test_a_capture_with_kernels_the_vocabulary_cannot_name_never_passes():
    # sm100 gdn_ctx, 2026-11-06: the Blackwell GDN cubins were unlabeled and the gate passed on conv1d
    g = grade(["flash::FlashAttnFwdSm90", "blackwell_gdn_cubin"], ["flash::FlashAttnFwdSm90"], ("attention",))
    assert g["verdict"] == "unlabeled-kernels" and g["unlabeled_kernels"] == ["blackwell_gdn_cubin"]
    assert grade(["blackwell_gdn_cubin"], ["x"], ("attention",))["verdict"] == "unlabeled-kernels"
    # a drift stays a drift: the unlabeled kernels are listed beside it
    g = grade(["kernel_unified_attention", "blackwell_gdn_cubin"], ["flash::FlashAttnFwdSm90"], ("attention",))
    assert g["verdict"] == "diverged" and g["unlabeled_kernels"] == ["blackwell_gdn_cubin"]


# ----------------------------------------------------------- auxiliary work


_B200_GEMM = "deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl"
_INDUCTOR_QUANT = "triton_per_fused__to_copy_abs_ceil_clamp_div_exp2_log2_max_view_0"
_SCALE_PACK = "deep_gemm::pack_fp32_into_ue8m0"
_SERVING_QUANT = "vllm::per_token_group_quant_8bit_packed_kernel"


def test_auxiliary_work_serving_never_does_fails_the_gate():
    # vLLM B200 fp8_block, 2026-11-07: the timed runs carried an inductor activation quant and a
    # DeepGEMM scale pack; serving runs one CUDA quant. The gemm matched, so the old rule passed it.
    col = [_INDUCTOR_QUANT, _SCALE_PACK, _B200_GEMM]
    srv = [_B200_GEMM, _SERVING_QUANT, "triton_"]  # "triton_" is a substring, but of another role
    assert grade(col, srv, ("gemm",))["verdict"] == "aligned"  # nothing measured: unchecked
    g = grade(col, srv, ("gemm",), measured=col)
    assert g["verdict"] == "diverged"
    assert g["aux_drift"] == {"activation": [_INDUCTOR_QUANT], "quant": [_SCALE_PACK]}
    assert g["kernel_drift"] is None and g["missing_roles"] == []  # the gemm itself is aligned


def test_auxiliary_kernels_serving_runs_keep_the_gate_aligned():
    col = [_SERVING_QUANT, _B200_GEMM]
    g = grade(col, [_B200_GEMM, _SERVING_QUANT], ("gemm",), measured=col)
    assert g["verdict"] == "aligned" and g["aux_drift"] == {}


def test_an_auxiliary_kernel_may_come_from_any_serving_phase():
    # the gate's phase scopes its own op's kernels; a quant serving runs only in decode is still
    # work serving does
    col = ["flash::FlashAttnFwdSm90", "vllm::scaled_fp8_quant_kernel"]
    srv_prefill = ["flash::FlashAttnFwdSm90"]
    g = grade(col, srv_prefill, ("attention",), measured=col, srv_every=[*srv_prefill, "vllm::scaled_fp8_quant_kernel"])
    assert g["verdict"] == "aligned" and g["aux_drift"] == {}


def test_only_the_timed_runs_are_checked_for_auxiliary_work():
    # an autotuner's warmup work sits in the timed range, not in the runs the row averages
    col = [_INDUCTOR_QUANT, _B200_GEMM]
    g = grade(col, [_B200_GEMM], ("gemm",), measured=[_B200_GEMM])
    assert g["verdict"] == "aligned" and g["aux_drift"] == {}


def test_a_warmup_only_kernel_of_the_target_role_is_not_graded():
    # SGLang B200 fp8_block, 2026-11-08: DeepGEMM packed a scale once on the first call; the
    # timed runs launch only the GEMM and serving's quant, as serving does
    pack = "deep_gemm::transpose_and_pack_fp32_into_ue8m0"
    col = [pack, _B200_GEMM, "per_token_group_quant_8bit_kernel"]
    srv = [_B200_GEMM, "per_token_group_quant_8bit_kernel"]
    assert grade(col, srv, ("gemm",))["verdict"] == "diverged"  # timed range only: the pack counts
    g = grade(col, srv, ("gemm",), measured=[_B200_GEMM, "per_token_group_quant_8bit_kernel"])
    assert g["verdict"] == "aligned" and g["kernel_drift"] is None and g["aux_drift"] == {}


# ---------------------------------------------------------------- tactics (2026-11-08)
# a kernel that differs from serving's only in its tactic is the same path: a tuner's near-tie
# flips between runs, so it is listed and never diverges the gate
_MOE = {
    "bmm_E4m3_E4m3E4m3_Fp32_t128x64x128u2_s6_et64x64": ("trtllm_gen", "moe_gemm"),
    "bmm_E4m3_E4m3E4m3_Fp32_t128x128x128_s4_et128x64": ("trtllm_gen", "moe_gemm"),
    "routingDeepSeek": ("trtllm_gen", "routing"),
}
FC_A, FC_B = "bmm_E4m3_E4m3E4m3_Fp32_t128x64x128u2_s6_et64x64", "bmm_E4m3_E4m3E4m3_Fp32_t128x128x128_s4_et128x64"


def test_a_tactic_difference_is_listed_and_does_not_diverge():
    col, srv = [FC_B], [FC_A, "routingDeepSeek"]
    g = path_diff.grade(col, set(srv), ("moe_gemm",), None, _MOE.get, col, set(srv))
    assert g["verdict"] == "aligned" and g["kernel_drift"] is None
    assert g["tactic_drift"]["moe_gemm"]["collector_only_kernels"] == col


def test_a_kernel_of_another_family_is_drift_not_a_tactic():
    labels = {**_MOE, "_matmul_ogs_NNT_bf16xbf16xmxfp4": ("triton", "moe_gemm")}
    col, srv = ["bmm_E4m3_E4m3E4m3_Fp32_x"], {"_matmul_ogs_NNT_bf16xbf16xmxfp4"}
    labels["bmm_E4m3_E4m3E4m3_Fp32_x"] = ("triton", "moe_gemm")
    g = path_diff.grade(col, srv, ("moe_gemm",), None, labels.get, col, srv)
    assert g["verdict"] == "diverged" and g["kernel_drift"] and g["tactic_drift"] is None
