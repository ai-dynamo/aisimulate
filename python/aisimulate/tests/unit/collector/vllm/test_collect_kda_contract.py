# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Phase coverage of the shared KDA case generator is pinned once, in
# tests/unit/collector/sglang/test_collect_kda_contract.py (both backend
# getters adapt the same generator).

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit
SOURCE_PATH = Path(__file__).resolve().parents[4] / "collector" / "vllm" / "collect_kda.py"


def test_kda_context_conv_int32_overflow_guard_is_resolved_not_present():
    # Resolved at the 0.27.0 era bump: the old unverified FIXME
    # (kernel-limit) guard `nt * proj >= 2 ** 31` was deleted after 0.27.0's
    # causal_conv1d was verified int64 throughout and GB300 silicon passed
    # the formerly-vetoed cells. Pin that the guard does not silently come
    # back without re-verification, and that the resolution comment keeps
    # its evidence citations.
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    guard_tests = {ast.unparse(node.test) for node in ast.walk(tree) if isinstance(node, ast.If)}
    assert "nt * proj >= 2 ** 31" not in guard_tests
    source = SOURCE_PATH.read_text(encoding="utf-8")
    assert "stride_x_token: tl.int64" in source


def test_kda_context_conv_passes_serve_parity_metadata():
    # Serving prefill hands the step-cached GDNAttentionMetadata into every
    # layer's causal_conv1d_fn call; omitting it selects the non-serving
    # metadata=None branch that rebuilds token offsets with numpy + a D2H
    # sync inside every timed call (the 0.1.dev19262 pollute-flat bug). Pin
    # that run_kda_context_benchmark passes metadata= structurally.
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_kda_context_benchmark":
            conv_calls = [
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "causal_conv1d_fn"
            ]
            # Materialized before asserting: a generator is always truthy and
            # all() over an empty one is vacuously true, so an assertion built
            # on a generator passes even when no causal_conv1d_fn call exists.
            assert len(conv_calls) == 1, (
                f"run_kda_context_benchmark must call causal_conv1d_fn at exactly one site, found {len(conv_calls)}"
            )
            metadata_keywords = [kw for kw in conv_calls[0].keywords if kw.arg == "metadata"]
            assert len(metadata_keywords) == 1
            metadata_value = metadata_keywords[0].value
            assert isinstance(metadata_value, ast.Name) and metadata_value.id == "conv_metadata", (
                "run_kda_context_benchmark's causal_conv1d_fn call must receive the "
                "serve-parity conv_metadata object; metadata=None adds a numpy + D2H "
                "floor inside every timed call"
            )
            return
    raise AssertionError("run_kda_context_benchmark not found / no causal_conv1d_fn call")


def test_kda_context_seq_len_one_routes_through_decode_kernels(monkeypatch):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    run_entrypoint = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "run_kda_torch"
    )

    calls = []

    def record(kind):
        return lambda **kwargs: calls.append((kind, kwargs))

    namespace = {
        "WORKER_RESTART": 23,
        "_is_glm5_next_kda": lambda model_name: False,
        "_require_audited_runtime": lambda model_name, version: None,
        "run_kda_context_benchmark": record("context"),
        "run_kda_generation_benchmark": record("decode"),
        "run_kda_verify_benchmark": record("verify"),
    }
    module = ast.Module(body=[run_entrypoint], type_ignores=[])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)

    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    version_module = ModuleType("vllm.version")
    version_module.__version__ = "0.27.0"
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.version", version_module)

    kwargs = {
        "phase": "context",
        "d_model": 7168,
        "d_conv": 4,
        "num_k_heads": 12,
        "head_k_dim": 128,
        "num_v_heads": 12,
        "head_v_dim": 128,
        "batch_size_list": [1, 2],
        "seq_len_list": [1, 2, 4],
        "model_name": "moonshotai/Kimi-K3",
        "perf_filename": "unused.txt",
    }
    assert namespace["run_kda_torch"](**kwargs) == 23
    assert [(kind, call["seq_len_list"] if kind == "context" else call["row_phase"]) for kind, call in calls] == [
        ("decode", "context"),
        ("context", [2, 4]),
    ]

    with pytest.raises(ValueError, match="at least one sequence length"):
        namespace["run_kda_torch"](**{**kwargs, "seq_len_list": []})
    with pytest.raises(ValueError, match="sequence lengths must be positive"):
        namespace["run_kda_torch"](**{**kwargs, "seq_len_list": [0]})


def test_kda_dispatch_mirrors_serving():
    # The collector must dispatch prefill like serving (FlashKDA when
    # supported, Triton fallback) and probe the fused decode kernel via the
    # same predicate serving uses — never pin a kernel unconditionally.
    # AST name references (not substring greps), so docstrings/comments
    # cannot satisfy the contract — mirrors the sglang twin test.
    import ast

    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "is_flashkda_supported" in referenced
    assert "is_fused_kda_decode_supported" in referenced


def _exec_functions(names, namespace):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    body = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in names for t in node.targets))
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def _fake_vllm_version(monkeypatch, version="0.31.0"):
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    version_module = ModuleType("vllm.version")
    version_module.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.version", version_module)


GLM_SHAPE = {
    "d_model": 4096,
    "d_conv": 4,
    "num_k_heads": 32,
    "head_k_dim": 128,
    "num_v_heads": 32,
    "head_v_dim": 128,
}


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("zai-org/GLM-5.3-Flash", "glm"),
        ("nvidia/GLM-5.3-Flash-NVFP4", "glm"),
        ("moonshotai/Kimi-K3", "kimi"),
    ],
)
def test_glm5_next_rows_route_to_the_glm5next_dispatch(monkeypatch, model_name, expected):
    # GLM-5.3-Flash is served by vllm/models/glm5next/common/kda.py, not the
    # Kimi-K3 layer; the entry point must hand its rows to the GLM router and
    # leave every other model on the unchanged Kimi path.
    calls = []
    namespace = {
        "WORKER_RESTART": 23,
        "_require_audited_runtime": lambda model_name, version: None,
        "run_glm5_next_kda_torch": lambda *args, **kwargs: calls.append(("glm", kwargs["vllm_version"])),
        "run_kda_generation_benchmark": lambda **kwargs: calls.append(("kimi", kwargs["vllm_version"])),
    }
    _exec_functions({"GLM5_NEXT_KDA_MODEL_PATHS", "_is_glm5_next_kda", "run_kda_torch"}, namespace)
    _fake_vllm_version(monkeypatch)
    result = namespace["run_kda_torch"](
        phase="generation",
        batch_size_list=[1],
        seq_len_list=None,
        model_name=model_name,
        perf_filename="unused.txt",
        **GLM_SHAPE,
    )
    assert result == 23
    assert calls == [(expected, "0.31.0")]


def test_glm5_next_phase_router_matches_serving_decode_threshold():
    # gdn_attn.py split_decodes_and_prefills(decode_threshold=1): one-token
    # cells of the context grid are decodes (row phase stays "context");
    # verify is not collected for the nextn=0 GLM baseline.
    calls = []
    namespace = {
        "run_glm5_next_kda_decode": lambda **kwargs: calls.append(
            ("decode", kwargs.get("row_phase", "generation"), None)
        ),
        "run_glm5_next_kda_context": lambda **kwargs: calls.append(("context", None, kwargs["seq_len_list"])),
    }
    _exec_functions({"run_glm5_next_kda_torch"}, namespace)
    router = namespace["run_glm5_next_kda_torch"]
    args = (4096, 4, 16, 128, 16, 128, [1, 2])
    router("context", *args, [1, 2, 131072], model_name="m")
    router("generation", *args, None, model_name="m")
    assert calls == [
        ("decode", "context", None),
        ("context", None, [2, 131072]),
        ("decode", "generation", None),
    ]
    with pytest.raises(NotImplementedError, match="verify"):
        router("verify", *args, [2, 4], model_name="m")
    with pytest.raises(ValueError, match="symmetric"):
        router("context", 4096, 4, 16, 128, 8, 128, [1], [2], model_name="m")


def _function(name):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _calls(node, name):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and (
            (isinstance(call.func, ast.Name) and call.func.id == name)
            or (isinstance(call.func, ast.Attribute) and call.func.attr == name)
        )
    ]


def test_glm5_next_context_mirrors_serving_prefill():
    # kda.py:569-582 runs ONE merged q|k|v causal_conv1d_fn with the step
    # metadata; the prefill core is picked by the framework's own resolver and
    # wrapped by the state gather/scatter (kda.py:644-692).
    context = _function("run_glm5_next_kda_context")
    conv_calls = _calls(context, "causal_conv1d_fn")
    assert len(conv_calls) == 1
    assert {kw.arg for kw in conv_calls[0].keywords} >= {"metadata", "has_initial_state", "cache_indices"}
    resolver_calls = _calls(context, "_resolve_kda_prefill_backend")
    assert len(resolver_calls) == 1 and isinstance(resolver_calls[0].args[0], ast.Constant)
    assert resolver_calls[0].args[0].value == "auto"
    assert len(_calls(context, "gather_initial_states")) == 2
    assert len(_calls(context, "scatter_states")) == 2
    assert len(_calls(context, "fwd")) == 1
    assert len(_calls(context, "chunk_kda_with_fused_gate")) == 1


def test_glm5_next_decode_mirrors_serving_decode():
    # kda.py:583-596,693-720: merged conv update, then the glm5next
    # fused_recurrent_kda with the in-kernel bounded gate.
    decode = _function("run_glm5_next_kda_decode")
    assert len(_calls(decode, "causal_conv1d_update")) == 1
    (recurrent,) = _calls(decode, "fused_recurrent_kda")
    keywords = {kw.arg: kw.value for kw in recurrent.keywords}
    assert isinstance(keywords["compute_gate"], ast.Constant) and keywords["compute_gate"].value is True
    assert isinstance(keywords["sigmoid_beta"], ast.Constant) and keywords["sigmoid_beta"].value is True
    assert ast.unparse(keywords["lower_bound"]) == "GLM5_NEXT_KDA_LOWER_BOUND"


@pytest.mark.parametrize(
    ("version", "accepted"),
    [
        ("0.1.dev19262+gb6bbf29dd", True),
        ("0.30.0+glm53tail.eb4704514fdf", True),
        ("0.31.0", True),
        ("0.31.1", False),
    ],
)
def test_kda_compat_admits_kimi_preview_and_glm_runtime(version, accepted):
    from collector.version_resolver import _check_compat

    declaration = next(
        node.value.value
        for node in ast.parse(SOURCE_PATH.read_text(encoding="utf-8")).body
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__compat__" for t in node.targets)
    )
    assert _check_compat(declaration, version) is accepted


_GATE_NAMES = {
    "GLM5_NEXT_KDA_MODEL_PATHS",
    "_is_glm5_next_kda",
    "KIMI_K3_KDA_ARCHITECTURE",
    "GLM5_NEXT_KDA_ARCHITECTURE",
    "_KDA_ARCHITECTURE_COMPAT",
    "KdaRuntimeNotAuditedError",
    "_kda_architecture",
    "_require_audited_runtime",
}


def _gate_namespace():
    from collector.version_resolver import _check_compat

    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    body = [
        node
        for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in _GATE_NAMES)
        or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in _GATE_NAMES for t in node.targets))
    ]
    namespace = {"_check_compat": _check_compat}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace


def test_kda_audited_release_sets_do_not_overlap():
    from collector.version_resolver import _check_compat

    compat = _gate_namespace()["_KDA_ARCHITECTURE_COMPAT"]
    audited = {arch: spec.split("==", 1)[1] for arch, spec in compat.items()}
    assert set(audited) == {"KimiK3ForConditionalGeneration", "Glm5NextForConditionalGeneration"}
    for arch, version in audited.items():
        for other, spec in compat.items():
            assert _check_compat(spec, version) is (other == arch), (arch, other)


@pytest.mark.parametrize(
    ("model_name", "version", "audited"),
    [
        ("moonshotai/Kimi-K3", "0.1.dev19262+gb6bbf29dd", True),
        ("moonshotai/Kimi-K3", "0.27.0", False),
        ("moonshotai/Kimi-K3", "0.30.0+glm53tail.eb4704514fdf", False),
        ("moonshotai/Kimi-K3", "0.31.0", False),
        ("zai-org/GLM-5.3-Flash", "0.31.0", True),
        ("nvidia/GLM-5.3-Flash-NVFP4", "0.31.0", True),
        ("zai-org/GLM-5.3-Flash", "0.30.0+glm53tail.eb4704514fdf", False),
        ("zai-org/GLM-5.3-Flash", "0.30.0", False),
        ("zai-org/GLM-5.3-Flash", "0.29.0", False),
        ("nvidia/GLM-5.3-Flash-NVFP4", "0.1.dev19262", False),
    ],
)
def test_kda_runtime_gate_rejects_unaudited_releases(model_name, version, audited):
    namespace = _gate_namespace()
    if audited:
        namespace["_require_audited_runtime"](model_name, version)
    else:
        with pytest.raises(namespace["KdaRuntimeNotAuditedError"], match="not an audited KDA runtime"):
            namespace["_require_audited_runtime"](model_name, version)


def test_glm5_next_imports_follow_the_v031_module_layout():
    # vLLM 0.31.0 moved the GLM layer glm5next/nvidia/kda.py ->
    # glm5next/common/kda.py (byte-identical); the CUDA ops stay under
    # glm5next/nvidia/ops/third_party/kda (common/kda.py:42-51). The GLM path
    # is 0.31.0-only, so no import from the 0.30.0 location may remain.
    glm_imports = {
        (node.module, alias.name)
        for name in ("run_glm5_next_kda_context", "run_glm5_next_kda_decode")
        for node in ast.walk(_function(name))
        if isinstance(node, ast.ImportFrom) and node.module and "glm5next" in node.module
        for alias in node.names
    }
    assert glm_imports == {
        ("vllm.models.glm5next.common.kda", "_cast_sigmoid"),
        ("vllm.models.glm5next.common.kda", "_resolve_kda_prefill_backend"),
        ("vllm.models.glm5next.nvidia.ops.third_party.kda", "chunk_kda_with_fused_gate"),
        ("vllm.models.glm5next.nvidia.ops.third_party.kda", "fused_recurrent_kda"),
    }
