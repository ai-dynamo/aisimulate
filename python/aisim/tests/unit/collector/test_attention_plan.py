# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared attention plan: one case list for every backend, each model's attention
properties on the rows of the backends that can take them."""

import ast
from pathlib import Path

import pytest
import yaml
from collector import case_generator
from collector.case_generator import (
    SGLANG_ATTENTION_BACKENDS,
    AttentionHeadConfig,
    AttentionOwner,
    get_attention_architecture_models,
    get_attention_context_shape_sweeps,
    get_attention_head_configs,
    get_attention_shapes,
    sglang_default_attention_backend,
)
from collector.helper import _AIC_MODEL_CONFIG_DIR, _resolve_local_model_path
from collector.op_plans import OP_PLANS, AttentionCase, _attention_points, plan

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

MODEL_CASE_FILES = sorted((REPO_ROOT / "collector" / "cases" / "models").glob("*_cases.yaml"))

VLLM_FIELDS = (
    "batch_size",
    "input_len",
    "num_heads",
    "num_kv_heads",
    "head_dim",
    "use_fp8_kv_cache",
    "is_context_phase",
    "window_size",
)

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

    # nemotron_h.py _nemotron_h_overrides declares trtllm_mha on SM100/103, the default.
    nemotron = _sglang_rows(monkeypatch, "nvidia/Nemotron-H-56B-Base-8K", 100)
    assert {row["attn_backend_name"] for row in nemotron} == {"trtllm_mha"}
    qwen35 = _sglang_rows(monkeypatch, "Qwen/Qwen3.5-27B", 100)
    assert {row["attn_backend_name"] for row in qwen35} == {"triton"}


def test_a_shape_two_models_serve_on_different_backends_runs_on_both(model_files, monkeypatch):
    monkeypatch.setitem(SGLANG_ATTENTION_BACKENDS, "ModelA", {100: "flashinfer"})
    model_files(_profile("ModelA"), _profile("ModelB"))
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


def test_model_case_files_declare_no_sglang_backend(model_files):
    """A model case file holds structure; the backend per architecture lives in collector code."""
    assert MODEL_CASE_FILES
    for path in MODEL_CASE_FILES:
        profiles = (yaml.safe_load(path.read_text(encoding="utf-8")).get("model_case_values") or {}).get("attention")
        for profile in profiles or []:
            assert "sglang_backends" not in profile, path.name

    model_files(_profile("ModelA", sglang_backends={100: "flashinfer"}))
    with pytest.raises(ValueError, match="no sglang_backends field"):
        get_attention_shapes("context")


# The backend the plan dispatches each architecture on, per SM (89, 90, 100,
# 103, 120): its table entry where it has one, else the default.
_SGLANG_BACKENDS_BY_SM = {
    "Gemma4ForConditionalGeneration": ("triton", "triton", "trtllm_mha", "trtllm_mha", "triton"),
    "GptOssForCausalLM": ("triton", "fa3", "trtllm_mha", "trtllm_mha", "triton"),
    "KimiK3ForConditionalGeneration": ("flashinfer", "fa3", "trtllm_mha", "trtllm_mha", "flashinfer"),
    "Llama4ForConditionalGeneration": ("triton", "fa3", "trtllm_mha", "trtllm_mha", "triton"),
    "MiMoV2FlashForCausalLM": ("triton", "fa3", "fa4", "fa4", "triton"),
    "NemotronHForCausalLM": ("flashinfer", "fa3", "trtllm_mha", "trtllm_mha", "flashinfer"),
    "Qwen3_5ForConditionalGeneration": ("flashinfer", "fa3", "triton", "triton", "flashinfer"),
    "Qwen3_5MoeForCausalLM": ("flashinfer", "fa3", "trtllm_mha", "trtllm_mha", "flashinfer"),
    "Qwen3_5MoeForConditionalGeneration": ("flashinfer", "fa3", "triton", "triton", "flashinfer"),
}


def test_the_sglang_backend_table_has_every_architecture_that_declared_one():
    """An architecture on the default has an explicit empty entry, so the table lists every reviewed one."""
    assert set(SGLANG_ATTENTION_BACKENDS) == set(_SGLANG_BACKENDS_BY_SM)
    assert {name for name, backends in SGLANG_ATTENTION_BACKENDS.items() if not backends} == {
        "KimiK3ForConditionalGeneration",
        "NemotronHForCausalLM",
        "Qwen3_5MoeForCausalLM",
    }
    for architecture, backends in SGLANG_ATTENTION_BACKENDS.items():
        assert (REPO_ROOT / "collector" / "cases" / "models" / f"{architecture}_cases.yaml").exists(), architecture
        assert set(backends) <= {89, 90, 100, 103, 120}, architecture
        assert set(backends.values()) <= {"fa3", "fa4", "flashinfer", "triton", "trtllm_mha"}, architecture


@pytest.mark.parametrize("architecture", sorted(_SGLANG_BACKENDS_BY_SM))
def test_the_sglang_backend_table_keeps_each_architectures_backends(architecture):
    sink = architecture == "GptOssForCausalLM"  # every GPT-OSS layer has attention sinks
    owner = AttentionOwner(from_model=True, architecture=architecture, has_attention_sink=sink)
    assert tuple(owner.sglang_backend(sm) for sm in (89, 90, 100, 103, 120)) == _SGLANG_BACKENDS_BY_SM[architecture]


def test_an_architecture_outside_the_table_and_the_base_grid_take_the_default():
    for owner in (AttentionOwner(), AttentionOwner(from_model=True, architecture="Qwen3ForCausalLM")):
        for sm in (89, 90, 100, 103, 120):
            assert owner.sglang_backend(sm) == sglang_default_attention_backend(sm, False)


def test_the_sglang_backend_table_cites_the_override_of_every_entry():
    """Each entry sits under a comment naming the SGLang file and symbol that decide it."""
    source = (REPO_ROOT / "collector" / "case_generator.py").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(source) if line.startswith("SGLANG_ATTENTION_BACKENDS"))
    comment: list[str] = []
    cited = {}
    for line in source[start + 1 :]:
        stripped = line.strip()
        if stripped == "}":
            break
        if stripped.startswith("#"):
            comment.append(stripped.lstrip("# "))
            continue
        cited[stripped.split('"')[1]] = " ".join(comment)
        comment = []
    assert set(cited) == set(SGLANG_ATTENTION_BACKENDS)
    providers = {
        "Gemma4ForConditionalGeneration": "gemma4.py _gemma4_overrides",
        "GptOssForCausalLM": "gpt_oss.py _gpt_oss_overrides",
        "KimiK3ForConditionalGeneration": "kimi_k3.py _kimi_k3_overrides",
        "Llama4ForConditionalGeneration": "llama4.py _llama4_overrides",
        "MiMoV2FlashForCausalLM": "mimo_v2.py _mimo_v2_overrides",
        "NemotronHForCausalLM": "nemotron_h.py _nemotron_h_overrides",
        "Qwen3_5ForConditionalGeneration": "qwen3_5.py _qwen3_5_hybrid_overrides",
        # Shares the comment above the entry before it.
        "Qwen3_5MoeForConditionalGeneration": "",
        "Qwen3_5MoeForCausalLM": "qwen3_5.py registers _qwen3_5_hybrid_overrides",
    }
    for architecture, provider in providers.items():
        assert provider in cited[architecture], architecture


def test_vllm_rows_name_the_models_that_plan_a_shape():
    """A model-planned shape names each profile's [architecture, model]; a base-grid shape names none."""
    for phase in ("context", "generation"):
        shapes = get_attention_shapes(phase)
        rows = plan(f"attention_{phase}", "vllm", 90).rows
        seen = set()
        for row in rows:
            shape = (row[2], row[3], row[4], row[7])
            owners = [owner for owner in shapes[shape] if owner.from_model]
            if owners:
                assert len(row) == len(VLLM_FIELDS) + 1, row
                assert row[-1] == [
                    list(pair) for pair in dict.fromkeys((owner.architecture, owner.model_path) for owner in owners)
                ]
            else:
                assert len(row) == len(VLLM_FIELDS), row
            seen.add(shape)
        assert seen == set(shapes)
        base_only = {shape for shape, owners in shapes.items() if not any(owner.from_model for owner in owners)}
        assert base_only and {head_dim for _n, _kv, head_dim, _w in base_only} == {128}


def test_vllm_gemma_4_rows_name_the_gemma_4_models():
    architecture = "Gemma4ForConditionalGeneration"
    gemma = {"google/gemma-4-26B-A4B", "nvidia/Gemma-4-31B-IT-NVFP4"}
    for op in ("attention_context", "attention_generation"):
        rows = plan(op, "vllm", 89).rows
        gemma_rows = [row for row in rows if (row[4], row[7]) in {(256, 1024), (512, 0)}]
        assert gemma_rows
        assert all({name for name, _model in row[-1]} == {architecture} for row in gemma_rows)
        assert {tuple(model for _name, model in row[-1]) for row in gemma_rows if row[2] == 16} == {
            tuple(sorted(gemma))
        }
        assert {tuple(model for _name, model in row[-1]) for row in gemma_rows if row[2] == 32} == {
            ("nvidia/Gemma-4-31B-IT-NVFP4",)
        }
        # No other shape names a Gemma 4 model.
        others = [row for row in rows if (row[4], row[7]) not in {(256, 1024), (512, 0)}]
        assert not any(
            name == architecture for row in others if len(row) > len(VLLM_FIELDS) for name, _model in row[-1]
        )
    for model in gemma:
        assert Path(_resolve_local_model_path(model), "config.json").exists()


def test_vllm_rows_name_an_architecture_that_is_its_models_own():
    """The worker gates on the architecture: it must be the one in the model's config."""
    configs = Path(_AIC_MODEL_CONFIG_DIR)
    named = {
        tuple(pair)
        for row in plan("attention_context", "vllm", 90).rows
        if len(row) > len(VLLM_FIELDS)
        for pair in row[-1]
    }
    assert named
    checked = 0
    for architecture, model in named:
        config = configs / f"{model.replace('/', '--')}_config.json"
        if config.exists():
            assert yaml.safe_load(config.read_text(encoding="utf-8"))["architectures"] == [architecture], model
            checked += 1
    assert checked > 25


def test_vllm_rows_keep_the_case_and_only_grow_by_the_models():
    """The row decides how a case runs: every case gives one row with the case's own values."""
    for sm_version in (89, 90, 100, 103, 120):
        vllm = plan("attention_generation", "vllm", sm_version)
        assert len(vllm.rows) == len(vllm.cases)
        for case, row in zip(vllm.cases, vllm.rows, strict=True):
            assert row[: len(VLLM_FIELDS)] == [
                case.batch_size,
                case.input_len,
                case.num_heads,
                case.num_kv_heads,
                case.head_dim,
                case.kv_cache_dtype == "fp8",
                False,
                case.window_size,
            ]


def test_a_model_profile_without_a_model_path_cannot_name_its_vllm_config(model_files):
    profile = _profile("ModelA")
    del profile["model_case_values"]["attention"][0]["model_path"]
    model_files(profile)
    with pytest.raises(ValueError, match="names no architecture or no model_path"):
        _ = plan("attention_context", "vllm", 90).rows


def test_an_architectures_models_are_its_profiles_models_then_aliases():
    assert get_attention_architecture_models("Gemma4ForConditionalGeneration") == (
        "google/gemma-4-26B-A4B",
        "nvidia/Gemma-4-26B-A4B-NVFP4",
        "nvidia/Gemma-4-31B-IT-NVFP4",
    )
    assert get_attention_architecture_models("NoSuchArchitecture") == ()


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
        return AttentionCase(1, 1024, 4, 1, 256, 0, kv_cache_dtype, attn_dtype)

    # Qwen3.5 serves (4, 1) at head_dim 256 on triton, Qwen3.8 on trtllm_mha.
    assert set(runs[case("bfloat16", "bfloat16")]) == {"triton", "trtllm_mha"}
    # trtllm_mha quantizes Q itself on an FP8 KV cache; triton has no FP8 compute.
    assert runs[case("fp8", "bfloat16")] == ("triton",)
    assert runs[case("fp8", "fp8")] == ("trtllm_mha",)
    assert sm100.removed_runs

    # A shape every owner serves on trtllm_mha has no BF16-compute prefill on an FP8 KV cache.
    nemotron = AttentionCase(1, 1024, 32, 2, 128, 0, "fp8", "bfloat16")
    assert nemotron not in runs
    assert runs[nemotron._replace(attn_dtype="fp8")] == ("trtllm_mha",)
    assert "flashinfer" not in sm100.by_dispatch


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
