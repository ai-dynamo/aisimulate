# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared attention plan: one case list for every backend, each model's attention
properties on the rows of the backends that can take them."""

import ast
from pathlib import Path

import pytest
from collector import case_generator
from collector.case_generator import (
    AttentionHeadConfig,
    get_attention_context_shape_sweeps,
    get_attention_head_configs,
    get_attention_shapes,
    sglang_default_attention_backend,
)
from collector.op_plans import OP_PLANS, AttentionCase, _attention_points, plan

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

SGLANG_FIELDS = (
    "batch_size",
    "input_len",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "use_fp8_kv_cache",
    "use_fp8_context_fmha",
    "is_context_phase",
    "window_size",
    "v_head_dim",
    "runtime_window_size",
    "attention_chunk_size",
    "has_attention_sink",
    "scaling",
    "attn_backend_name",
    "architecture",
)


@pytest.fixture(autouse=True)
def _fresh_shapes(monkeypatch):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    case_generator._attention_shapes.cache_clear()
    yield
    case_generator._attention_shapes.cache_clear()


@pytest.fixture
def model_files(monkeypatch):
    """Replace the packaged model case files with the given ones."""

    def install(*files):
        monkeypatch.setattr(case_generator, "_load_model_cases_data", lambda: files)
        case_generator._attention_shapes.cache_clear()

    return install


def _profile(architecture, heads=8, kv_heads=2, head_dim=128, window=0, **properties):
    row = {
        "model_path": f"org/{architecture}",
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "window_size": window,
        "tensor_parallel_sizes": [1],
        **properties,
    }
    return {"architecture": architecture, "model_case_values": {"attention": [row]}}


def _sglang_rows(monkeypatch, model_path, sm_version, phase="context"):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    return [dict(zip(SGLANG_FIELDS, row, strict=True)) for row in plan(f"attention_{phase}", "sglang", sm_version).rows]


def _by_shape(rows):
    shapes = {}
    for row in rows:
        shape = (row["num_heads"], row["num_kv_heads"], row["head_dim"], row["window_size"])
        shapes.setdefault(shape, []).append(row)
    return shapes


def test_the_shared_grid_is_generic_full_attention_at_head_dim_128():
    for phase, base_shapes in (("context", 44), ("generation", 22)):
        shapes = get_attention_shapes(phase)
        base = [shape for shape, owners in shapes.items() if any(not owner.from_model for owner in owners)]
        assert len(base) == base_shapes
        assert {(head_dim, window) for _n, _kv, head_dim, window in base} == {(128, 0)}
        assert max(num_heads for num_heads, *_ in base) == 64


def test_model_profiles_add_every_other_head_dim_and_window():
    for phase in ("context", "generation"):
        shapes = get_attention_shapes(phase)
        assert {
            (16, 8, 256, 1024),  # Gemma 4 sliding window
            (16, 2, 512, 0),  # Gemma 4 global
            (40, 8, 128, 8192),  # Llama 4 chunked local
            (64, 4, 192, 0),  # MiMo-V2-Flash global
            (64, 8, 192, 128),  # MiMo-V2-Flash sliding window
            (64, 8, 64, 128),  # gpt-oss sliding window
            (96, 8, 128, 512),  # Step-3.7-Flash sliding window
            (128, 8, 128, 0),  # Llama 3.1 405B at TP1
        } <= set(shapes)
    # 122/110: the 44/22 base shapes plus every model shape outside them.
    assert len(get_attention_shapes("context")) == 122
    assert len(get_attention_shapes("generation")) == 110


def test_decode_budget_scales_up_below_head_dim_128_only():
    (sweep,) = case_generator.get_shared_base_op_case_specs("attention_generation")

    def points(head_dim):
        return set(_attention_points(sweep, "generation", 8, 1, head_dim))

    assert points(64) > points(128)
    assert points(128) == points(256) == points(512)


def test_every_case_carries_its_precision_on_the_case():
    case = OP_PLANS["attention_context"].expand()[0]
    assert isinstance(case, AttentionCase)
    assert {case.kv_cache_dtype, case.attn_dtype} <= {"bfloat16", "fp8"}
    assert case.q_per_kv == case.num_heads // case.num_kv_heads


def test_sglang_rows_carry_each_models_attention_properties(monkeypatch):
    mimo = _by_shape(_sglang_rows(monkeypatch, "XiaomiMiMo/MiMo-V2-Flash", 90))
    assert {row["v_head_dim"] for rows in mimo.values() for row in rows} == {128}
    assert not any(row["has_attention_sink"] for row in mimo[(64, 4, 192, 0)])
    assert all(row["has_attention_sink"] and row["runtime_window_size"] == 128 for row in mimo[(64, 8, 192, 128)])
    assert {row["attn_backend_name"] for rows in mimo.values() for row in rows} == {"fa3"}

    gemma = _by_shape(_sglang_rows(monkeypatch, "google/gemma-4-26B-A4B", 90))
    assert {row["runtime_window_size"] for row in gemma[(16, 8, 256, 1024)]} == {1023}
    assert {row["scaling"] for rows in gemma.values() for row in rows} == {1.0}
    assert {row["attn_backend_name"] for rows in gemma.values() for row in rows} == {"triton"}

    llama = _by_shape(_sglang_rows(monkeypatch, "meta-llama/Llama-4-Scout-17B-16E", 90))
    rows = llama[(40, 8, 128, 0)] + llama[(40, 8, 128, 8192)]
    assert {(row["attention_chunk_size"], row["runtime_window_size"]) for row in rows} == {(8192, -1)}

    gpt_oss = _by_shape(_sglang_rows(monkeypatch, "openai/gpt-oss-120b", 90, "generation"))
    assert {row["runtime_window_size"] for row in gpt_oss[(64, 8, 64, 128)]} == {127}
    assert all(row["has_attention_sink"] for rows in gpt_oss.values() for row in rows)
    assert {row["attn_backend_name"] for rows in gpt_oss.values() for row in rows} == {"fa3"}

    muse = _by_shape(_sglang_rows(monkeypatch, "meta-models/Muse-Glimmer-30B", 90))
    assert {row["runtime_window_size"] for row in muse[(32, 2, 128, 2048)]} == {2047}
    assert {row["runtime_window_size"] for row in muse[(32, 2, 128, 0)]} == {-1}
    assert not any(row["has_attention_sink"] for rows in muse.values() for row in rows)

    nemotron = _sglang_rows(monkeypatch, "nvidia/Nemotron-H-56B-Base-8K", 100)
    assert {row["attn_backend_name"] for row in nemotron} == {"flashinfer"}
    qwen35 = _sglang_rows(monkeypatch, "Qwen/Qwen3.5-27B", 100)
    assert {row["attn_backend_name"] for row in qwen35} == {"triton"}


def test_a_shape_two_models_serve_on_different_backends_runs_on_both(model_files):
    model_files(_profile("ModelA", sglang_backends={100: "flashinfer"}), _profile("ModelB"))
    rows = plan("attention_generation", "sglang", 100).rows
    shapes = _by_shape(dict(zip(SGLANG_FIELDS, row, strict=True)) for row in rows)
    rows = shapes[(8, 2, 128, 0)]
    assert {row["attn_backend_name"] for row in rows} == {"flashinfer", "trtllm_mha"}
    by_backend = {row["attn_backend_name"]: row["architecture"] for row in rows}
    assert by_backend == {"flashinfer": "ModelA", "trtllm_mha": "ModelB"}


def test_a_model_profile_wins_over_the_base_grid_on_the_same_backend(model_files):
    model_files(_profile("ModelA", v_head_dim=64))
    rows = [dict(zip(SGLANG_FIELDS, row, strict=True)) for row in plan("attention_generation", "sglang", 90).rows]
    assert {row["v_head_dim"] for row in rows if (row["num_heads"], row["num_kv_heads"]) == (8, 2)} == {64}


def test_two_models_that_disagree_on_one_backend_raise(model_files):
    model_files(_profile("ModelA", v_head_dim=128), _profile("ModelB", v_head_dim=64))
    with pytest.raises(ValueError, match="different runtime semantics"):
        _ = plan("attention_context", "sglang", 90).rows

    model_files(_profile("ModelA", window=8192), _profile("ModelB", window=8192, sglang_attention_chunk_size=8192))
    with pytest.raises(ValueError, match="different runtime semantics"):
        _ = plan("attention_context", "sglang", 90).rows


def test_a_chunk_size_on_a_full_attention_shape_does_not_split_it(model_files):
    model_files(_profile("ModelA"), _profile("ModelB", sglang_attention_chunk_size=8192))
    assert plan("attention_context", "sglang", 90).rows


def test_sglang_default_backend_follows_serving_selection():
    assert sglang_default_attention_backend(90, False) == "fa3"
    assert sglang_default_attention_backend(100, False) == sglang_default_attention_backend(103, False) == "trtllm_mha"
    assert sglang_default_attention_backend(89, False) == sglang_default_attention_backend(120, False) == "flashinfer"
    assert sglang_default_attention_backend(89, True) == sglang_default_attention_backend(120, True) == "triton"
    assert sglang_default_attention_backend(80, False) is None

    dispatch = OP_PLANS["attention_context"].dispatch["sglang"]
    case = OP_PLANS["attention_context"].expand()[0]
    with pytest.raises(ValueError, match="No SGLang attention backend for SM80"):
        dispatch(case, 80)
    # The plan never asks: SGLang's attention is left out on SM80/86 as a whole.
    assert plan("attention_context", "sglang", 80).cases == []


def test_a_dispatch_entry_leaves_a_case_out_on_that_kernel_only():
    sm100 = plan("attention_context", "sglang", 100)
    runs = dict(zip(sm100.cases, sm100.dispatches, strict=True))

    def case(kv_cache_dtype, attn_dtype):
        return AttentionCase(1, 1024, 32, 2, 128, 0, kv_cache_dtype, attn_dtype)

    # Nemotron-H serves (32, 2) on flashinfer, the other models and the generic grid on trtllm_mha.
    assert set(runs[case("bfloat16", "bfloat16")]) == {"flashinfer", "trtllm_mha"}
    # trtllm_mha quantizes Q itself on an FP8 KV cache; flashinfer has no FP8 compute.
    assert runs[case("fp8", "bfloat16")] == ("flashinfer",)
    assert runs[case("fp8", "fp8")] == ("trtllm_mha",)
    assert sm100.removed_runs


def test_bf16_prefill_on_fp8_kv_never_reaches_a_backend_that_quantizes_q():
    for sm_version in (89, 90, 100, 103, 120):
        rows = [
            dict(zip(SGLANG_FIELDS, row, strict=True)) for row in plan("attention_context", "sglang", sm_version).rows
        ]
        for row in rows:
            backend = row["attn_backend_name"]
            if row["use_fp8_kv_cache"] and not row["use_fp8_context_fmha"]:
                assert backend != "trtllm_mha" and not (backend == "fa3" and row["head_dim"] <= 256), row
            if row["use_fp8_context_fmha"]:
                assert backend != "flashinfer", row


def test_sglang_attention_runtime_fields_are_not_persisted_dimensions():
    source_path = REPO_ROOT / "collector" / "sglang" / "collect_attn.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch")
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call)]

    pool = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "MHATokenToKVPool")
    radix = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "RadixAttention")
    log_perf = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "log_perf")
    layer_call = next(call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "layer")

    assert "v_head_dim" in {keyword.arg for keyword in pool.keywords}
    assert {"v_head_dim", "sliding_window_size", "use_irope"} <= {keyword.arg for keyword in radix.keywords}
    # Serving passes ``sinks`` only for sink-carrying models (FlashInferAttnBackend
    # takes no ``sinks`` kwarg), so the layer call expands a conditional kwargs dict.
    assert {keyword.arg for keyword in layer_call.keywords} == {None}
    kwargs_expansion = next(keyword.value for keyword in layer_call.keywords if keyword.arg is None)
    assert isinstance(kwargs_expansion, ast.Name) and kwargs_expansion.id == "layer_kwargs"
    layer_kwargs_assign = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "layer_kwargs" for target in node.targets)
    )
    conditional = layer_kwargs_assign.value
    assert isinstance(conditional, ast.IfExp)
    assert isinstance(conditional.test, ast.Name) and conditional.test.id == "has_attention_sink"

    item_list = next(keyword.value for keyword in log_perf.keywords if keyword.arg == "item_list")
    row_fields = {key.value for key in item_list.elts[0].keys if isinstance(key, ast.Constant)}
    assert not {"v_head_dim", "attention_chunk_size", "runtime_window_size"} & row_fields
    source_kw = next(keyword.value for keyword in log_perf.keywords if keyword.arg == "kernel_source")
    assert isinstance(source_kw, ast.Name) and source_kw.id == "attn_backend_name"


def test_only_the_xpu_collector_reads_the_framework_neutral_helper():
    for relative_path, calls in (
        ("collector/vllm/collect_attn_xpu.py", 2),
        ("collector/vllm/collect_attn.py", 0),
        ("collector/sglang/collect_attn.py", 0),
        ("collector/trtllm/collect_attn.py", 0),
    ):
        tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_attention_head_configs"
        ]
        assert len(found) == calls, relative_path


def test_framework_neutral_population_keeps_one_config_per_key(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "test/model")
    profile = {
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "window_size": 0,
        "tensor_parallel_sizes": [1],
    }
    configs = get_attention_head_configs({"head_profiles": [profile, dict(profile)]}, phase="context")

    assert configs == [AttentionHeadConfig(8, 2, 128, 0)]


@pytest.mark.parametrize(
    ("model_path", "expected"),
    [
        # Global rows, then sliding-window rows, each over its TP shards.
        ("openai/gpt-oss-120b", [(heads, window) for window in (0, 128) for heads in (64, 32, 16, 8, 4, 2, 1)]),
        ("meta-llama/Llama-4-Scout-17B-16E", [(heads, window) for heads in (40, 20, 10, 5) for window in (0, 8192)]),
    ],
)
def test_framework_neutral_model_profiles_keep_their_order(monkeypatch, model_path, expected):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", model_path)
    sweep = get_attention_context_shape_sweeps("vllm_xpu")[0]
    configs = get_attention_head_configs(sweep, phase="context")

    assert [(config.num_heads, config.window_size) for config in configs] == expected
