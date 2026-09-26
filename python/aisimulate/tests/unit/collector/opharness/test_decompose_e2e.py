# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""decompose (families + residue from records through the taxonomy) and the
pure grading half of e2e_align (measurement contract, tolerance verdict)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

COMPONENTS = Path(__file__).resolve().parents[4] / "collector" / "opharness" / "components"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, COMPONENTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dc():
    return _load("decompose")


@pytest.fixture(scope="module")
def e2e():
    return _load("e2e_align")


def _record(rid, repo, backend, kv, kernels, orphans, eager=False, variant="rep"):
    return {"id": rid, "target": {"repo": repo, "variant": variant, "profile": "bfloat16"},
            "runtime": {"backend": backend, "version": "0.29.0", "kv_cache_dtype": kv,
                        "probe_eager": eager, "sm_measured": "sm90"},
            "identity": {}, "ops": [{"phase": "prefill", "op": "attn:X", "kernels": kernels}],
            "orphan_kernels": orphans, "outcome": {"status": "ok"}}


def test_taxonomy_labels_every_kernel_or_leaves_residue(dc):
    rules = dc.load_rules("sm90")
    rec = _record("a", "org/m", "vllm", "auto",
                  ["flash::FlashAttnFwdSm90"], ["nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT", "totally_unknown_kernel_xyz"])
    d = dc.decompose_record(rec, rules)
    assert d["families"]["attention"]["fa3"] == ["flash::FlashAttnFwdSm90"]
    assert d["families"]["gemm"]["cublas"] == ["nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT"]
    assert d["residue"] == ["totally_unknown_kernel_xyz"]
    assert d["coverage"] == {"labeled": 2, "residue": 1}
    assert d["ops_observed"] == ["attn:X"]


def test_first_match_wins_and_glue_is_not_residue(dc):
    rules = dc.load_rules("sm90")
    # the six 2026-09-25 glue rules: labeled (role infra/gemm/...), never residue
    for k in ("sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize64x128x32", "fmha_cutlassF_bf16_aligned_64x128_rf_sm80",
              "_extract_transpose_prefill_kernel", "store_kvcache", "flash_c4_decode"):
        assert dc.label(k, rules) is not None, k
    # the trtllm MoE tactic profiler is denied upstream (probe_driver.KERNEL_DENY), never a family
    assert dc.label("tensorrt_llm::kernels::delayStreamKernel", rules) is None
    # the sglang Triton block-fp8 GEMM keeps its own label: collector lane is DeepGEMM
    assert dc.label("_w8a8_block_fp8_matmul", rules) == ("gemm", "triton_block_fp8")


def test_representative_is_rendered_kv_framework_mode_and_fp8_variant_folds_in(dc, tmp_path):
    rules = dc.load_rules("sm90")
    recs = [
        _record("eager", "org/m", "vllm", "auto", ["flash::FlashAttnFwdSm90"], [], eager=True),
        _record("fw", "org/m", "vllm", "auto", ["flash::FlashAttnFwdSm90"], ["nvjet_sm90_tst_128x8_64x12_2x1_v_bz_TNT"]),
        _record("fp8", "org/m", "vllm", "fp8", ["flash::FlashAttnFwdSm90"], ["reshape_and_cache_kernel_flash", "mystery_fp8_kernel"]),
        {"id": "bad", "target": {"repo": "org/m", "variant": "rep"}, "runtime": {"backend": "vllm", "version": "0.29.0"},
         "outcome": {"status": "fail"}},
    ]
    p = tmp_path / "records.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    out = dc.decompose(p, "sm90", None, rules)
    entry = out[("vllm", "0.29.0")]["org/m"]
    assert entry["record"] == "fw"  # framework-mode rendered-KV wins over the eager probe
    assert entry["residue"] == []
    assert entry["kv_variants"]["fp8"]["added_kernels"] == ["mystery_fp8_kernel", "reshape_and_cache_kernel_flash"]
    assert entry["kv_variants"]["fp8"]["added_residue"] == ["mystery_fp8_kernel"]
    written = dc.write_outputs(out, "sm90", tmp_path / "out", "kernel_taxonomy_sm90.yaml", tmp_path / "evidence")
    yaml = __import__("yaml")
    doc = yaml.safe_load(written[0].read_text())
    assert doc["_meta"]["summary"] == {"repos": 1, "repos_with_residue": 0, "residue_kernels": []}
    # the committed file carries COUNTS, the evidence file carries the kernel names
    assert doc["results"]["org/m"]["families"] == {"attention": {"fa3": 1}, "gemm": {"cublas": 1}}
    assert "kv_variants" in doc["results"]["org/m"] and doc["results"]["org/m"]["kv_variants"]["fp8"]["added_kernels"] == 2
    full = yaml.safe_load((tmp_path / "evidence" / "vllm-0.29.0.yaml").read_text())
    assert full["results"]["org/m"]["families"]["attention"]["fa3"] == ["flash::FlashAttnFwdSm90"]
    assert full["_meta"]["kind"] == "evidence:kernels" and doc["_meta"]["kind"] == "summary"


def test_e2e_grade_tolerance(e2e):
    g = e2e.grade({"ttft_ms": 100.0, "tpot_ms": 10.0}, {"ttft_ms": 120.0, "tpot_ms": 9.0}, 0.25)
    assert g["verdict"] == "aligned"
    assert g["metrics"]["ttft_ms"]["rel_error"] == 0.2
    g = e2e.grade({"ttft_ms": 100.0, "tpot_ms": 10.0}, {"ttft_ms": 200.0, "tpot_ms": 9.0}, 0.25)
    assert g["verdict"] == "diverged"
    # nothing comparable -> not a verdict about the model
    assert e2e.grade({"ttft_ms": None}, {"ttft_ms": 1.0}, 0.25)["verdict"] == "not-comparable"


def test_e2e_measurement_contract_and_aiperf_adapter(e2e, tmp_path):
    ours = tmp_path / "m.json"
    ours.write_text(json.dumps({"isl": 4096, "osl": 512, "batch_size": 32, "ttft_ms": 800.0, "tpot_ms": 25.0, "source": "x"}))
    m = e2e.load_measurement(ours)
    assert (m["isl"], m["osl"], m["batch_size"], m["ttft_ms"], m["tpot_ms"]) == (4096, 512, 32, 800.0, 25.0)
    aiperf = tmp_path / "profile_export_aiperf.json"
    aiperf.write_text(json.dumps({"time_to_first_token": {"avg": 812.5, "unit": "ms"},
                                  "inter_token_latency": {"avg": 27500.0, "unit": "us"},
                                  "input_sequence_length": {"avg": 4096.0}, "output_sequence_length": {"avg": 512.0},
                                  "request_concurrency": 32}))
    m = e2e.load_measurement(aiperf)
    assert m["ttft_ms"] == 812.5 and m["tpot_ms"] == 27.5
    assert (m["isl"], m["osl"], m["batch_size"]) == (4096, 512, 32)


def test_e2e_golden_reader_takes_identity_from_the_render(e2e, tmp_path):
    g = tmp_path / "golden" / "org_m_naive_tp2_pp1_1"
    g.mkdir(parents=True)
    (g / "generator_config.yaml").write_text("ServiceConfig:\n  model_path: org/m\nK8sConfig:\n  system_name: h200_sxm\n")
    (g / "run.sh").write_text('engine_command=(python3 -m dynamo.vllm --model org/m --tensor-parallel-size 2 --max-num-seqs 512)\n')
    info = e2e.load_golden(tmp_path / "golden")
    assert (info["model_path"], info["system"], info["tp"], info["pp"]) == ("org/m", "h200_sxm", 2, 1)
    assert "--tensor-parallel-size 2" in info["engine_line"]
