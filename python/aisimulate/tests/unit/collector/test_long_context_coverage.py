# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional model-owned long-context coverage preserves the pinned planner baseline."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from collector import case_generator, op_plans
from collector.coverage import includes_long_context, long_context_coverage, long_context_points
from collector.model_cases import build_collection_case_plan
from collector.op_plans import plan

pytestmark = pytest.mark.unit
PKG_ROOT = Path(__file__).resolve().parents[3]
# Generated from actual plans at #676 69232f77ce94de85e8ca7c40da8c2e484a6bdff4; the DSA and MSA
# context entries regenerated after the context grid gained prefix 1, 1536, 3072, 6144, 10240 and
# 12288 and every sequence length at prefixes from 16k on; SGLang attention_context regenerated
# after the FP8-compute skip widened to triton and fa4 prefill (no FP8-compute path there) and
# fa4's FP8-KV prefill was left out (serving cannot run it).
BASELINE = {
    "sglang:attention_context": [28645, "6c0dfd64be096af68aee88ce03a86bb4e104e9e8ff536c56b04e8c9853681267"],
    "sglang:attention_generation": [36096, "403666ff16ab29e96d2ef9a79c7c1595a64d8208f2c4ce5c1d3abc7274994653"],
    "sglang:dsa_context_module": [32940, "0774184fcb3fc62175675a77e5d5d63c9dee99efb1840ecbb0d7be28c018c82f"],
    "sglang:dsa_generation_module": [5700, "995f99c01c986b3fa038f21eb7c3f306965cad81b18516c7bf8f774d1d8b1d13"],
    "sglang:mla_context_module": [13176, "d63d0534649bab9324ee70c23fcfcce0373bc7903537342c42850dcfa88e46fa"],
    "sglang:mla_generation_module": [13248, "da66d1707e49f320831b1acbd6e090a7a5e615b9fe9523c38817ac3816f75391"],
    "sglang:msa_context_module": [12350, "ec381d6168ee455b9e4a2666aee44d7fde609bf7ebe19412ac25e7704b59aaa1"],
    "sglang:msa_generation_module": [1930, "18a1aaa0f67336597f617632a1b8d7bb5893f0874e79cfdec84bf88dff21c7d6"],
    "trtllm:attention_context": [36261, "1dcb3ce260dd61381831f647b287b7c41640273508900ea18e361096109f3757"],
    "trtllm:attention_generation": [30054, "10f11fc455c7123272fa4fee898f3b962ab11d7a24f43bb1b3e34afd7ea77e2a"],
    "trtllm:dsa_context_module": [65220, "76c41b415880af8991b96ce5f79b1bc1326429703f358461a9c96a6b660a559c"],
    "trtllm:dsa_generation_module": [10152, "64009d7847fe4065c7a7f2047560c34535a96d614c69d50227bffdb36a314731"],
    "trtllm:mla_context_module": [8784, "be7ad7e1b3f779960be249f05a95cfb90e0fc341881528a77f1d8196fd56a29f"],
    "trtllm:mla_generation_module": [13248, "da66d1707e49f320831b1acbd6e090a7a5e615b9fe9523c38817ac3816f75391"],
    "trtllm:msa_context_module": [18525, "d078285ec076d16221ff7ed2d65321719a3062788be39412f2648474894e62f6"],
    "trtllm:msa_generation_module": [2895, "527698244a19b01034190b05e4d3f2fc134e60fa47f9d5222c4a5d2cfcd188dd"],
    "vllm:attention_context": [28390, "a1ed1fd7cdd56512af271c151a4b141d771a3e5585a88fc9c2e1f18aad471db3"],
    "vllm:attention_generation": [36096, "403666ff16ab29e96d2ef9a79c7c1595a64d8208f2c4ce5c1d3abc7274994653"],
    "vllm:dsa_context_module": [65220, "76c41b415880af8991b96ce5f79b1bc1326429703f358461a9c96a6b660a559c"],
    "vllm:dsa_generation_module": [10152, "64009d7847fe4065c7a7f2047560c34535a96d614c69d50227bffdb36a314731"],
    "vllm:mla_context_module": [13176, "d63d0534649bab9324ee70c23fcfcce0373bc7903537342c42850dcfa88e46fa"],
    "vllm:mla_generation_module": [13248, "da66d1707e49f320831b1acbd6e090a7a5e615b9fe9523c38817ac3816f75391"],
    "vllm:msa_context_module": [37050, "f6e84a28e1f2b84992b264097c70884126e42c00dbc939b7b1da8c86496bd370"],
    "vllm:msa_generation_module": [5790, "e4fad3bb4e4c216880d90f66c7ff38906d069ae57e581b3ad8165958d53b2235"],
}


@pytest.fixture(autouse=True)
def clear_target(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    case_generator._attention_shapes.cache_clear()
    yield
    case_generator._attention_shapes.cache_clear()


@pytest.mark.parametrize("key", sorted(BASELINE))
def test_off_preserves_pinned_676_default_history_population(key):
    backend, op = key.split(":")
    cases = plan(op, backend, 103).cases
    count, fingerprint = BASELINE[key]
    assert len(cases) == count
    assert hashlib.sha256("\n".join(map(str, cases)).encode()).hexdigest() == fingerprint


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
@pytest.mark.parametrize("sm", [90, 103, 120])
@pytest.mark.parametrize("op", sorted({key.split(":")[1] for key in BASELINE}))
def test_on_is_additive_across_frameworks_and_sms(backend, sm, op):
    normal = plan(op, backend, sm)
    extended = plan(op, backend, sm, include_long_context=True)
    assert set(normal.cases) <= set(extended.cases)
    assert len(extended.cases) == len(set(extended.cases))
    assert not includes_long_context()


def test_qwen_points_use_own_geometry_and_aliases_without_cross_product(monkeypatch):
    sets = []
    for model in ("Qwen/Qwen3.5-397B-A17B", "nvidia/Qwen3.5-397B-A17B-NVFP4"):
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", model)
        normal = plan("attention_context", "vllm", 103)
        extra = set(plan("attention_context", "vllm", 103, include_long_context=True).cases) - set(normal.cases)
        assert extra
        assert {(case.num_heads, case.num_kv_heads, case.head_dim) for case in extra} <= {
            (32, 2, 256),
            (16, 1, 256),
            (8, 1, 256),
            (4, 1, 256),
            (2, 1, 256),
            (1, 1, 256),
        }
        assert max(case.input_len for case in extra) == 262144
        assert all(case.batch_size <= 16 for case in extra)
        sets.append(extra)
    assert sets[0] == sets[1]


def test_sparse_points_keep_query_prefix_and_decode_past_distinct(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/GLM-5.2-NVFP4")
    context = plan("dsa_context_module", "sglang", 103, include_long_context=True)
    decode = plan("dsa_generation_module", "sglang", 103, include_long_context=True)
    assert any(case.seq_len == 32768 and case.prefix_len == 917504 for case in context.cases)
    assert any(case.batch_size == 32 and case.seq_len == 1048575 for case in decode.cases)
    assert all(case.prefix_len + case.seq_len <= 1048576 for case in context.cases)
    assert all(case.seq_len + 1 <= 1048576 for case in decode.cases)


def test_nested_checkpoint_config_limits_are_used(monkeypatch):
    monkeypatch.setattr(
        op_plans, "_packaged_config", lambda _path: {"text_config": {"max_position_embeddings": 262144}}
    )
    assert op_plans._history_limit("example/nested") == 262144
    with long_context_coverage(True), pytest.raises(ValueError, match="exceeding model limit"):
        long_context_points(
            {"model_path": "example/nested", "long_context_cases": {"generation": [[1, 262144, 0]]}},
            "generation",
            op_plans._history_limit("example/nested"),
        )
    monkeypatch.setattr(op_plans, "_packaged_config", lambda _path: {})
    with pytest.raises(ValueError, match="no valid max_position_embeddings"):
        op_plans._history_limit("example/missing")


@pytest.mark.parametrize("point", [[0, 512, 0], [1, -1, 0], [1, 512, -1], [True, 512, 0], [1, 512], [1, "512", 0]])
def test_malformed_points_raise_and_coverage_scope_resets(point):
    with pytest.raises(ValueError), long_context_coverage(True):
        long_context_points({"long_context_cases": {"context": [point]}}, "context", 1048576)
    assert not includes_long_context()


def test_runtime_geometry_prevents_wrong_cross_model_dedup(monkeypatch):
    configs = {
        "a": {"max_position_embeddings": 1048576, "kv_lora_rank": 512},
        "b": {"max_position_embeddings": 1048576, "kv_lora_rank": 256},
    }
    monkeypatch.setattr(op_plans, "_packaged_config", configs.__getitem__)
    monkeypatch.setattr(op_plans, "_model_case_values", lambda _op: [])
    monkeypatch.setattr(op_plans, "_framework_specific_model_case_values", lambda _op, _backend: [])
    monkeypatch.setattr(op_plans, "get_mla_module_precision_specs", lambda *_args, **_kwargs: [])
    cases = [op_plans.ModuleCase(524287, 1, 8, "fp8", "bfloat16", "bfloat16", path) for path in ("a", "b")]
    with long_context_coverage(True), pytest.raises(ValueError, match="incompatible runtime semantics"):
        op_plans._with_history_module_cases(cases, "mla", "generation", None)
    configs["b"] = {**configs["a"], "_name_or_path": "b"}
    with long_context_coverage(True):
        assert len(op_plans._with_history_module_cases(cases, "mla", "generation", None)) == 1


def test_plan_metadata_records_default_and_opt_in():
    for enabled in (False, True):
        selected = build_collection_case_plan(
            backend="sglang", model_path="zai-org/GLM-5.2", include_long_context=enabled
        )
        assert selected.to_log_dict()["include_long_context"] is enabled


def test_group_rows_keep_coverage_when_materialized_later(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "zai-org/GLM-5.2")
    selected = plan("dsa_generation_module", "sglang", 103, include_long_context=True)
    assert not includes_long_context()
    rows = selected.rows
    points = [point for row in rows for point in row[-1]]
    assert [32, 1048575, 0] in points
    assert len(points) == len(selected.cases)
    assert not includes_long_context()


def test_cli_plan_and_resume_do_not_lose_new_group_points(tmp_path):
    # Exercise the real CLI and the real row-ID checkpoint filter, without loading a GPU framework.
    env = dict(os.environ)
    env.pop("COLLECTOR_MODEL_PATH", None)
    env["PYTHONPATH"] = os.pathsep.join([str(PKG_ROOT), str(PKG_ROOT / "collector"), str(PKG_ROOT / "src")])
    command = [
        sys.executable,
        str(PKG_ROOT / "collector/collect.py"),
        "--backend",
        "sglang",
        "--model-path",
        "zai-org/GLM-5.2",
        "--ops",
        "dsa_generation_module",
        "--sm",
        "103",
        "--plan-only",
    ]
    normal = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
    extended = subprocess.run(
        [*command, "--include-long-context"], env=env, text=True, capture_output=True, timeout=120
    )
    assert normal.returncode == extended.returncode == 0, normal.stderr + extended.stderr
    before, after = json.loads(normal.stdout), json.loads(extended.stdout)
    assert not before["include_long_context"] and after["include_long_context"]
    assert before["plans"]["dsa_generation_module"]["cases"] < after["plans"]["dsa_generation_module"]["cases"]
    script = """
from collector.collect import ResumeCheckpoint, _run_case_id
from collector.op_plans import plan
import logging, sys
import collector.collect as collect
collect.logger = logging.getLogger("coverage-resume")
checkpoint = ResumeCheckpoint("sglang", "sglang.dsa_generation_module", "run", sys.argv[1])
normal = plan("dsa_generation_module", "sglang", 103)
expanded = plan("dsa_generation_module", "sglang", 103, include_long_context=True)
old_ids = {_run_case_id(row, "run", "sglang.dsa_generation_module") for row in normal.rows}
for case_id in old_ids: checkpoint.mark_passed(case_id)
tasks = [{"id": _run_case_id(row,"run","sglang.dsa_generation_module"), "params":row} for row in expanded.rows]
pending = checkpoint.filter_done(tasks)
assert pending
assert any([32,1048575,0] in task["params"][-1] for task in pending)
for task in pending: checkpoint.mark_passed(task["id"])
assert checkpoint.filter_done(tasks) == []
"""
    env["COLLECTOR_MODEL_PATH"] = "zai-org/GLM-5.2"
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], env=env, text=True, capture_output=True, timeout=120
    )
    assert result.returncode == 0, result.stderr


def test_qwen_measured_666_anchor_batches_are_retained(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "Qwen/Qwen3.5-397B-A17B")
    points = plan("attention_context", "vllm", 103, include_long_context=True).cases
    actual = {
        (case.batch_size, case.input_len)
        for case in points
        if (case.num_heads, case.num_kv_heads, case.head_dim, case.kv_cache_dtype) == (4, 1, 256, "bfloat16")
    }
    assert {(batch, length) for batch in (1, 2, 4, 8, 16) for length in (24576, 32768, 40960)} <= actual


def test_glm_measured_665_batch52_frontier_is_declared_for_native_head8(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "nvidia/GLM-5.2-NVFP4")
    expected = {
        1,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
        262144,
        393216,
        524288,
        655360,
        720896,
    }
    for op in ("dsa_generation_module", "dsa_generation_module_skip_indexer"):
        cases = plan(op, "sglang", 103, include_long_context=True).cases
        selected = [case for case in cases if case.batch_size == 52]
        assert {case.seq_len for case in selected} == expected
        assert {case.num_heads for case in selected} == {8}


def test_glm_expert_only_quantization_and_redundant_head_dim_are_safe_to_union():
    reference = op_plans._history_runtime_signature("zai-org/GLM-5.2")
    assert op_plans._history_runtime_signature("nvidia/GLM-5.2-NVFP4") == reference
    assert op_plans._history_runtime_signature("nvidia/GLM-5.3-NVFP4") == reference


def test_unknown_quantization_and_actual_head_geometry_are_never_erased(monkeypatch):
    reference = op_plans._history_config("zai-org/GLM-5.2")
    changed = {**reference, "qk_nope_head_dim": 128}
    configs = {"a": reference, "b": changed}
    monkeypatch.setattr(
        op_plans, "_packaged_config", lambda path, kind="config": configs[path] if kind == "config" else {}
    )
    assert op_plans._history_runtime_signature("a") != op_plans._history_runtime_signature("b")
    configs["b"] = {**reference, "quantization_config": {"quant_method": "unknown", "target": "self_attn"}}
    assert op_plans._history_runtime_signature("a") != op_plans._history_runtime_signature("b")


def test_added_module_points_have_one_persisted_key_per_runtime_owner():
    for op in ("dsa_context_module", "dsa_generation_module"):
        cases = plan(op, "vllm", 103, include_long_context=True).cases
        keys = {
            (*case[:6], tuple(op_plans._history_config(case.model_path).get("architectures", [])), case.prefix_len)
            for case in cases
        }
        assert len(keys) == len(cases)


def test_equivalent_model_points_reuse_existing_canonical_owner():
    for op in ("dsa_context_module", "dsa_generation_module"):
        normal = plan(op, "vllm", 103)
        extended = plan(op, "vllm", 103, include_long_context=True)
        owners = {case.model_path for case in normal.cases}
        assert {case.model_path for case in extended.cases} <= owners


def test_partial_quantization_exclusions_do_not_prove_attention_unquantized(monkeypatch):
    config = {"quantization_config": {"quant_algo": "NVFP4"}, "num_hidden_layers": 2}
    monkeypatch.setattr(
        op_plans,
        "_modelopt_quant",
        lambda _path: {"quant_algo": "NVFP4", "exclude_modules": ["model.layers.*.self_attn.probe*"]},
    )
    assert not op_plans._history_attention_unquantized("example/model", config)


@pytest.mark.parametrize("op", ["mla_context_module", "mla_generation_module"])
def test_unqualified_kimi_nope_extension_is_not_enabled(monkeypatch, op):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "moonshotai/Kimi-K3")
    baseline = plan(op, "vllm", 103)
    extended = plan(op, "vllm", 103, include_long_context=True)
    assert baseline.cases
    assert extended.cases == baseline.cases
    # The pin's ordinary geometry remains; no new long-context recipe is
    # inferred from Kimi's 1M context limit or native-head declaration.
    profiles = case_generator._framework_specific_model_case_values("mla_module", "vllm")
    assert profiles and all("long_context_cases" not in profile for profile in profiles)
    assert all("long_context_tensor_parallel_sizes" not in profile for profile in profiles)


@pytest.mark.parametrize("backend", ["vllm", "trtllm"])
@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize(
    ("selected", "expected_gemms"),
    [
        ("MiniMaxAI/MiniMax-M3", {"bfloat16"}),
        ("nvidia/MiniMax-M3-NVFP4", {"fp8_block"}),
        (None, {"bfloat16", "fp8_block"}),
    ],
)
def test_minimax_long_context_preserves_artifact_projection_lanes(
    monkeypatch, backend, phase, selected, expected_gemms
):
    if selected:
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", selected)
    normal = plan(f"msa_{phase}_module", backend, 103)
    expanded = plan(f"msa_{phase}_module", backend, 103, include_long_context=True)
    added = set(expanded.cases) - set(normal.cases)
    length, prefix = (32768, 917504) if phase == "context" else (917504, 0)
    points = [
        case
        for case in added
        if (case.batch_size, case.seq_len, case.prefix_len, case.num_heads, case.kv_cache_dtype)
        == (1, length, prefix, 8, "bfloat16")
    ]
    assert {case.gemm_type for case in points} == expected_gemms
    assert {case.model_path for case in points} == {"MiniMaxAI/MiniMax-M3"}
    assert set(normal.cases) <= set(expanded.cases)
