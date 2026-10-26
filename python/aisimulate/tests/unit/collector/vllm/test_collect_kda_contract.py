# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Phase coverage of the shared KDA case generator is pinned once, in
# tests/unit/collector/sglang/test_collect_kda_contract.py (both backend
# getters adapt the same generator).

import ast
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
        "_require_audited_runtime": lambda model_name, version, **kwargs: None,
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
        "_require_audited_runtime": lambda model_name, version, **kwargs: None,
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
    router("context", *args, [1, 2, 32768], model_name="m")
    router("generation", *args, None, model_name="m")
    assert calls == [
        ("decode", "context", None),
        ("context", None, [2, 32768]),
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
        ("moonshotai/Kimi-K3", "0.31.0", False),
        ("zai-org/GLM-5.3-Flash", "0.31.0", True),
        ("nvidia/GLM-5.3-Flash-NVFP4", "0.31.0", True),
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


def _compile_function(name, namespace):
    tree = ast.parse(SOURCE_PATH.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("backend", ["flashkda", "triton", "flashinfer"])
def test_target_prefill_resolver_uses_fp32_state_and_propagates_selection(monkeypatch, backend):
    from collector.version_resolver import _check_compat

    calls = []
    native = ModuleType("vllm.models.kimi_k3.nvidia.kda")
    native.resolve_kda_prefill_backend = lambda *args: (calls.append(args), backend)[1]
    native._flashkda_prefill = object()
    monkeypatch.setitem(sys.modules, native.__name__, native)
    vllm = ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm._flashkda_C", ModuleType("vllm._flashkda_C"))
    resolve = _compile_function(
        "_resolve_prefill_kernel",
        {
            "torch": SimpleNamespace(dtype=object, float32="fp32"),
            "KDA_LOWER_BOUND": -5.0,
            "_check_compat": _check_compat,
        },
    )
    if backend == "flashinfer":
        with pytest.raises(RuntimeError, match="Unexpected KDA prefill backend"):
            resolve("bf16", "0.31.0")
    else:
        source, kernel = resolve("bf16", "0.31.0")
        assert (source, kernel) == (
            ("flashkda_fwd", native._flashkda_prefill) if backend == "flashkda" else ("chunk_kda_with_fused_gate", None)
        )
    assert calls == [("auto", 128, "bf16", "fp32", -5.0)]
    if backend == "flashkda":
        monkeypatch.setitem(sys.modules, "vllm._flashkda_C", None)
        with pytest.raises(ImportError):
            resolve("bf16", "0.31.0")


class _ContractTensor:
    def __init__(self, *shape, dtype=None, **kwargs):
        self.shape, self.dtype = shape, dtype

    def reshape(self, *shape):
        return _ContractTensor(*shape, dtype=self.dtype)

    def transpose(self, first, second):
        shape = list(self.shape)
        shape[first], shape[second] = shape[second], shape[first]
        return _ContractTensor(*shape, dtype=self.dtype)

    def contiguous(self):
        return self

    def __rmul__(self, scalar):
        return self


@pytest.mark.parametrize("native_failure", [False, True])
def test_target_decode_uses_native_output_and_never_times_unused_fallback(monkeypatch, native_failure):
    from collector.version_resolver import _check_compat

    calls, published, selected, cleanups = [], [], [], []

    def native_decode(**kwargs):
        calls.append(kwargs)
        if native_failure:
            raise RuntimeError("native kernel failed")

    def unused(*args, **kwargs):
        raise AssertionError("unused fallback was called")

    vllm = ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    for name, attributes in {
        "vllm._custom_ops": {"fused_kda_decode": native_decode},
        "vllm.model_executor.layers.mamba.ops.causal_conv1d": {"causal_conv1d_update": unused},
        "vllm.models.kimi_k3.nvidia.kda": {
            "is_fused_kda_decode_supported": unused,
            "resolve_kda_decode_backend": lambda *args: (selected.append(args), "native")[1],
        },
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": {"fused_recurrent_kda_packed_decode": unused},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    @contextmanager
    def benchmark(**kwargs):
        kwargs["kernel_func"]()
        yield {"latency_ms": 1.0, "power_stats": None}

    torch = SimpleNamespace(
        device=lambda device: device,
        cuda=SimpleNamespace(set_device=lambda device: None),
        bfloat16="bf16",
        float32="fp32",
        int32="int32",
        randn=_ContractTensor,
        zeros=_ContractTensor,
        empty=_ContractTensor,
        empty_like=lambda tensor: _ContractTensor(*tensor.shape, dtype=tensor.dtype),
        arange=_ContractTensor,
    )
    run = _compile_function(
        "run_kda_generation_benchmark",
        {
            "torch": torch,
            "_check_compat": _check_compat,
            "KDA_LOWER_BOUND": -5.0,
            "NUM_WARMUPS": 3,
            "NUM_RUNS": 10,
            "benchmark_with_power": benchmark,
            "_log": lambda *args: published.append(args),
            "_cleanup": cleanups.append,
            "_format_failures": lambda failures: ";".join(failures),
        },
    )
    args = (7168, 4, 12, 128, 12, 128, [1], "moonshotai/Kimi-K3", "isolated.txt", "0.31.0")
    if native_failure:
        with pytest.raises(RuntimeError, match="native kernel failed"):
            run(*args)
        assert published == []
    else:
        run(*args)
        assert [record[2] for record in published] == ["fused_kda_decode"]
    assert selected == [("auto", 12, 128, 4, 0, "bf16", "bf16", "fp32")]
    assert len(calls) == 1
    assert calls[0]["out"].shape == (1, 1, 12, 128)
    assert calls[0]["state"].shape == (1, 12, 128, 128)
    assert calls[0]["state"].dtype == "fp32"
    assert calls[0]["conv_state"].shape == (1, 4608, 3)
    assert cleanups == ["generation"]


def test_target_flashkda_buffers_are_allocated_before_timed_callback():
    from collector.version_resolver import _check_compat

    tree = ast.parse(SOURCE_PATH.read_text())
    context = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_kda_context_benchmark")
    branch = next(
        n
        for n in ast.walk(context)
        if isinstance(n, ast.If)
        and ast.dump(n.test) == ast.dump(ast.parse('prefill_source == "flashkda_fwd"', mode="eval").body)
    )
    calls, allocations = [], []

    def allocate(*shape, **kwargs):
        allocations.append(shape)
        return _ContractTensor(*shape, **kwargs)

    torch = SimpleNamespace(
        float32="fp32",
        uint8="uint8",
        empty=allocate,
        empty_like=lambda tensor: allocate(*tensor.shape, dtype=tensor.dtype),
        ops=SimpleNamespace(_flashkda_C=SimpleNamespace(get_workspace_size=lambda *args: 1234)),
    )
    namespace = {
        "torch": torch,
        "_check_compat": _check_compat,
        "vllm_version": "0.31.0",
        "nt": 128,
        "nh": 12,
        "batch_size": 2,
        "device": "cuda:0",
        "KDA_LOWER_BOUND": -5.0,
        "q": _ContractTensor(1, 128, 12, 128, dtype="bf16"),
        "init_state": _ContractTensor(2, 12, 128, 128, dtype="fp32"),
        "flashkda_prefill": lambda *args, **kwargs: calls.append(kwargs),
        "flashkda_buffers": {},
    }
    namespace.update(dict.fromkeys(["k", "v", "raw_g", "raw_beta", "a_log", "dt_bias", "cu"], object()))
    exec(compile(ast.Module(body=branch.body, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    assert len(allocations) == 3
    namespace["run_prefill"]()
    namespace["run_prefill"]()
    assert len(allocations) == 3
    assert calls[0]["out"].shape == (1, 128, 12, 128)
    assert calls[0]["final_state"].shape == (2, 12, 128, 128)
    assert calls[0]["final_state"].dtype == "fp32"
    assert calls[0]["workspace"].shape == (1234,)
    assert calls[0]["workspace"].dtype == "uint8"
    assert calls[1]["workspace"] is calls[0]["workspace"]


@pytest.mark.parametrize("backend", ["flashkda_fwd", "chunk_kda_with_fused_gate"])
@pytest.mark.parametrize("first_core_fails", [False, True])
def test_prefill_releases_completed_phase_storage_and_failed_point_buffers(monkeypatch, backend, first_core_fails):
    from collector.version_resolver import _check_compat

    mixed_refs, buffer_refs, core_calls, conv_calls, timings, published, cleanups = [], [], [], [], [], [], []

    class Storage:
        pass

    class Tensor(_ContractTensor):
        def __init__(self, *shape, dtype=None, storage=None, **kwargs):
            super().__init__(*shape, dtype=dtype)
            self.storage = storage if storage is not None else Storage()

        def transpose(self, first, second):
            shape = list(self.shape)
            shape[first], shape[second] = shape[second], shape[first]
            return Tensor(*shape, dtype=self.dtype, storage=self.storage)

        def split(self, sizes, dim=0):
            return [
                Tensor(*self.shape[:dim], size, *self.shape[dim + 1 :], dtype=self.dtype, storage=self.storage)
                for size in sizes
            ]

    def randn(*shape, **kwargs):
        if len(shape) == 2 and shape[1] == 4608:
            assert all(ref() is None for ref in buffer_refs), "Previous point retains native buffers"
        if len(shape) == 4:
            assert all(ref() is None for ref in mixed_refs), "Completed conv storage overlaps core"
        tensor = Tensor(*shape, **kwargs)
        if len(shape) == 2 and shape[1] == 4608:
            mixed_refs.append(weakref.ref(tensor.storage))
        return tensor

    def buffer(*shape, **kwargs):
        tensor = Tensor(*shape, **kwargs)
        buffer_refs.append(weakref.ref(tensor.storage))
        return tensor

    def native(*args, **kwargs):
        core_calls.append(args[0].shape)
        if backend == "flashkda_fwd":
            assert kwargs["out"].shape == args[0].shape
            assert kwargs["final_state"].shape == (1, 12, 128, 128)
            assert kwargs["final_state"].dtype == "fp32"
            assert kwargs["workspace"].shape == (1234,)
        if first_core_fails and len(core_calls) == 1:
            raise MemoryError("native core allocation failed")

    def conv(x, weight, bias, **kwargs):
        assert x.storage is mixed_refs[-1]()
        assert kwargs["metadata"].num_prefills == 1
        conv_calls.append(x.shape)

    def cleanup(tag):
        assert all(ref() is None for ref in buffer_refs), "Point cleanup retains native buffers"
        cleanups.append(tag)

    for name, attrs in {
        "vllm.model_executor.layers.mamba.ops.causal_conv1d": {"causal_conv1d_fn": conv},
        "vllm.models.kimi_k3.nvidia.ops.third_party.kda": {"chunk_kda_with_fused_gate": native},
        "vllm.v1.attention.backends.gdn_attn": {"GDNAttentionMetadata": SimpleNamespace},
        "vllm.v1.attention.backends.utils": {"compute_causal_conv1d_metadata": lambda *a, **kw: ({}, None, None)},
        "vllm.model_executor.layers.mamba.mamba_utils": {"is_conv_state_dim_first": lambda: False},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    @contextmanager
    def benchmark(**kwargs):
        timings.append((kwargs["num_warmups"], kwargs["num_runs"], kwargs["repeat_n"]))
        kwargs["kernel_func"]()
        yield {"latency_ms": 1.0, "power_stats": None}

    torch = SimpleNamespace(
        device=lambda device: device,
        cuda=SimpleNamespace(set_device=lambda device: None),
        bfloat16="bf16",
        float32="fp32",
        int32="int32",
        bool="bool",
        uint8="uint8",
        randn=randn,
        zeros=Tensor,
        arange=Tensor,
        empty=buffer,
        empty_like=lambda t: buffer(*t.shape, dtype=t.dtype),
        ops=SimpleNamespace(_flashkda_C=SimpleNamespace(get_workspace_size=lambda *args: 1234)),
    )
    run = _compile_function(
        "run_kda_context_benchmark",
        {
            "torch": torch,
            "_check_compat": _check_compat,
            "KDA_LOWER_BOUND": -5.0,
            "NUM_WARMUPS": 3,
            "NUM_RUNS": 10,
            "benchmark_with_power": benchmark,
            "_resolve_prefill_kernel": lambda *args: (backend, native),
            "_log": lambda common, latency, source, *args: published.append((common.copy(), source)),
            "_cleanup": cleanup,
            "_format_failures": lambda failures: ";".join(failures),
        },
    )
    args = (7168, 4, 12, 128, 12, 128, [1], [2, 4], "moonshotai/Kimi-K3", "unused.txt", "0.31.0")
    if first_core_fails:
        with pytest.raises(RuntimeError, match="batch_size=1 seq_len=2: MemoryError"):
            run(*args)
    else:
        run(*args)
    assert core_calls == [(1, 2, 12, 128), (1, 4, 12, 128)]
    assert conv_calls == [(1536, 2)] * 3 + [(1536, 4)] * 3
    assert timings == [(3, 10, 10), (3, 10, 1)] * 2
    assert cleanups == ["context", "context"]
    assert [(row["seq_len"], source) for row, source in published] == (
        [(2, "causal_conv1d_fn_qkv3")]
        + ([] if first_core_fails else [(2, backend)])
        + [(4, "causal_conv1d_fn_qkv3"), (4, backend)]
    )


@pytest.mark.parametrize("phase", ["context", "generation"])
def test_kimi031_fixed_profile_gate_admits_only_b200_sd(monkeypatch, phase):
    namespace = _gate_namespace()
    namespace["torch"] = SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda device: "NVIDIA B200"))
    native = ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
    native.is_conv_state_dim_first = lambda: False
    monkeypatch.setitem(sys.modules, native.__name__, native)
    gate = namespace["_require_audited_runtime"]
    gate("moonshotai/Kimi-K3", "0.31.0", phase=phase)
    native.is_conv_state_dim_first = lambda: True
    with pytest.raises(namespace["KdaRuntimeNotAuditedError"], match="SD conv state"):
        gate("moonshotai/Kimi-K3", "0.31.0", phase=phase)
    native.is_conv_state_dim_first = lambda: False
    namespace["torch"].cuda.get_device_name = lambda device: "NVIDIA H100"
    with pytest.raises(namespace["KdaRuntimeNotAuditedError"], match="NVIDIA B200"):
        gate("moonshotai/Kimi-K3", "0.31.0", phase=phase)


@pytest.mark.parametrize("phase", [None, "verify", "unknown"])
def test_kimi031_verify_rejected_before_native_import(phase):
    namespace = _gate_namespace()
    with pytest.raises(namespace["KdaRuntimeNotAuditedError"], match="context/nonspec"):
        namespace["_require_audited_runtime"]("moonshotai/Kimi-K3", "0.31.0", phase=phase)


def test_kimi031_entrypoint_preserves_phase_at_gate(monkeypatch):
    calls = []
    namespace = {
        "WORKER_RESTART": 23,
        "_require_audited_runtime": lambda model, version, **kwargs: calls.append((model, version, kwargs)),
        "_is_glm5_next_kda": lambda model: False,
        "run_kda_generation_benchmark": lambda **kwargs: None,
    }
    _exec_functions({"run_kda_torch"}, namespace)
    _fake_vllm_version(monkeypatch)
    namespace["run_kda_torch"](
        "generation", 7168, 4, 12, 128, 12, 128, [1], [1], "moonshotai/Kimi-K3", perf_filename="unused.txt"
    )
    assert calls == [("moonshotai/Kimi-K3", "0.31.0", {"phase": "generation", "device": "cuda:0"})]


def test_kimi031_registered_structural_keys_remain_distinct_from_glm():
    from collector.case_generator import get_common_kda_test_cases

    keys = {"kimi": set(), "glm": set()}
    for case in get_common_kda_test_cases():
        family = "kimi" if case.model_name == "moonshotai/Kimi-K3" else "glm"
        keys[family].add(
            (case.d_model, case.d_conv, case.num_k_heads, case.head_k_dim, case.num_v_heads, case.head_v_dim)
        )
    assert len(keys["kimi"]) == len(keys["glm"]) == 4
    assert keys["kimi"].isdisjoint(keys["glm"])


@pytest.mark.parametrize("target", [False, True])
def test_kimi_prefill_target_reserves_native_null_slot(target):
    # Native v0.31 causal_conv1d returns immediately for cache index0. Execute
    # the collector's allocation statements so a zero-based regression fails.
    from collector.version_resolver import _check_compat

    tree = ast.parse(SOURCE_PATH.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_kda_context_benchmark")
    names = {"target_runtime", "first_slot", "idx", "has_init"}
    assignments = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
    ]
    assignments.sort(key=lambda n: n.lineno)
    namespace = {
        "_check_compat": _check_compat,
        "vllm_version": "0.31.0" if target else "0.1.dev19262",
        "batch_size": 1,
        "device": "cuda:0",
        "torch": SimpleNamespace(
            int32="int32", bool="bool", arange=lambda a, b, **kw: list(range(a, b)), zeros=lambda n, **kw: [False] * n
        ),
    }
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    assert namespace["idx"] == ([1] if target else [0])
    if target:
        assert all(slot != 0 for slot in namespace["idx"]), "Native null-slot branch skips all B1 work"
    target_branch = next(
        n
        for n in ast.walk(function)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "target_runtime"
    )
    cache_allocations = [
        n
        for n in ast.walk(ast.Module(body=target_branch.body, type_ignores=[]))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "zeros"
    ]
    assert len(cache_allocations) == 2
    assert all(ast.unparse(n.args[0]) == "batch_size + 1" for n in cache_allocations)
