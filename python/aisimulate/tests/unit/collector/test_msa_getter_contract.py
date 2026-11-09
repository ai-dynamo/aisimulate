# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public MSA getter and worker contracts for the TRT-LLM and vLLM collectors."""

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = "MiniMaxAI/MiniMax-M3"
ARCHITECTURE = "MiniMaxM3ForCausalLM"


class _Dummy:
    def __init__(self, *_args, **kwargs):
        self.__dict__.update(kwargs)


def _noop(*_args, **_kwargs):
    return None


def _stub_module(monkeypatch, name: str, **attrs):
    module = types.ModuleType(name)
    module.__path__ = []
    module.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _stub_torch(monkeypatch):
    return _stub_module(
        monkeypatch,
        "torch",
        Tensor=object,
        bfloat16="bfloat16",
        float8_e4m3fn="float8_e4m3fn",
        float32="float32",
        int32="int32",
        long="long",
        uint8="uint8",
        cuda=SimpleNamespace(
            OutOfMemoryError=RuntimeError,
            Stream=_noop,
            empty_cache=_noop,
            is_available=lambda: False,
        ),
        device=lambda value: value,
        inference_mode=nullcontext,
    )


class _PeriodicGCStub:
    def maybe_collect(self):
        return False


def _install_helper_stub(monkeypatch):
    _stub_module(
        monkeypatch,
        "collector.helper",
        PeriodicGarbageCollector=_PeriodicGCStub,
        _resolve_local_model_path=lambda path: path,
        benchmark_with_power=_noop,
        get_sm_version=lambda: 100,
        log_perf=lambda **_kwargs: True,
    )


def _install_trtllm_stubs(monkeypatch):
    _stub_torch(monkeypatch)
    transformers = _stub_module(monkeypatch, "transformers", PretrainedConfig=_Dummy)
    trtllm = _stub_module(monkeypatch, "tensorrt_llm", __version__="1.4.0rc2")
    for package in (
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.attention",
        "tensorrt_llm._torch.attention.backends",
        "tensorrt_llm._torch.attention.backends.sparse",
        "tensorrt_llm._torch.attention.backends.sparse.minimax_m3",
        "tensorrt_llm._torch.models",
        "tensorrt_llm._torch.modules",
        "tensorrt_llm._torch.pyexecutor",
        "tensorrt_llm.bindings",
        "tensorrt_llm.bindings.internal",
        "tensorrt_llm.llmapi",
        "tensorrt_llm.models",
        "tensorrt_llm.quantization",
    ):
        _stub_module(monkeypatch, package)
    _stub_module(monkeypatch, "tensorrt_llm._torch.attention.backends.interface", AttentionRuntimeFeatures=_Dummy)
    _stub_module(monkeypatch, "tensorrt_llm._torch.attention.backends.utils", get_attention_backend=_noop)
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_backend",
        MiniMaxM3MsaSparseAttentionMetadata=_Dummy,
    )
    _stub_module(monkeypatch, "tensorrt_llm._torch.autotuner", autotune=nullcontext)
    _stub_module(monkeypatch, "tensorrt_llm._torch.modules.linear", Linear=_Dummy)
    _stub_module(monkeypatch, "tensorrt_llm._torch.metadata", KVCacheParams=_Dummy)
    _stub_module(monkeypatch, "tensorrt_llm._torch.model_config", ModelConfig=_Dummy)
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.models.modeling_minimaxm3",
        MiniMaxM3DecoderLayer=_Dummy,
    )
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.pyexecutor._util",
        get_kv_cache_manager_cls=_noop,
    )
    _stub_module(monkeypatch, "tensorrt_llm._torch.pyexecutor.config_utils", _CONFIG_REGISTRY={})
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.pyexecutor.model_loader",
        initialize_dummy_weights=_noop,
    )
    _stub_module(
        monkeypatch,
        "tensorrt_llm._torch.utils",
        AuxStreamType=SimpleNamespace(MoeChunkingOverlap="moe"),
        get_model_extra_attrs=lambda: {},
        model_extra_attrs=lambda *_args: SimpleNamespace(__enter__=_noop, __exit__=_noop),
    )
    _stub_module(monkeypatch, "tensorrt_llm._utils", torch_dtype_to_binding=_noop)
    _stub_module(
        monkeypatch,
        "tensorrt_llm.llmapi.llm_args",
        KvCacheConfig=_Dummy,
        MiniMaxM3SparseAttentionConfig=_Dummy,
    )
    _stub_module(
        monkeypatch,
        "tensorrt_llm.bindings.internal.batch_manager",
        CacheType=SimpleNamespace(SELF="self"),
    )
    _stub_module(monkeypatch, "tensorrt_llm.functional", AllReduceStrategy=SimpleNamespace(AUTO="auto"))
    _stub_module(
        monkeypatch,
        "tensorrt_llm.models.modeling_utils",
        QuantConfig=_Dummy,
    )
    _stub_module(monkeypatch, "tensorrt_llm.quantization.mode", QuantAlgo=SimpleNamespace())
    configs = _stub_module(monkeypatch, "tensorrt_llm._torch.configs")
    trtllm._torch = SimpleNamespace(configs=configs)
    configs.PretrainedConfig = transformers.PretrainedConfig
    _install_helper_stub(monkeypatch)


def _install_vllm_stubs(monkeypatch):
    _stub_torch(monkeypatch)
    _stub_module(monkeypatch, "vllm")
    for package in (
        "vllm.transformers_utils",
        "vllm.v1",
        "vllm.v1.worker",
        "collector.vllm",
    ):
        _stub_module(monkeypatch, package)
    _stub_module(monkeypatch, "vllm.config", set_current_vllm_config=_noop)
    _stub_module(monkeypatch, "vllm.forward_context", set_forward_context=_noop)
    _stub_module(monkeypatch, "vllm.transformers_utils.config", _CONFIG_REGISTRY={})
    _stub_module(monkeypatch, "vllm.v1.worker.workspace", init_workspace_manager=_noop)
    _stub_module(monkeypatch, "vllm.version", __version__="0.24.0")
    _stub_module(
        monkeypatch,
        "collector.vllm.utils",
        BatchSpec=_Dummy,
        create_common_attn_metadata=_noop,
        create_vllm_config=_noop,
        enable_engine_fused_ops=_noop,
        setup_distributed=_noop,
        with_exit_stack=lambda function: function,
    )
    _install_helper_stub(monkeypatch)


def _load_backend(monkeypatch, backend: str):
    import collector.op_plans  # noqa: F401  (the real plans, before collector.helper is stubbed)

    monkeypatch.setattr(sys, "path", list(sys.path))
    if backend == "trtllm":
        _install_trtllm_stubs(monkeypatch)
    else:
        _install_vllm_stubs(monkeypatch)
    module_name = f"msa_{backend}_getter_contract"
    path = REPO_ROOT / "collector" / backend / "collect_msa_module.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", ["trtllm", "vllm"])
def test_public_msa_getters_queue_the_op_plan(monkeypatch, backend):
    from collector.op_plans import plan

    module = _load_backend(monkeypatch, backend)
    for op, getter in (
        ("msa_context_module", module.get_msa_context_module_test_cases),
        ("msa_generation_module", module.get_msa_generation_module_test_cases),
    ):
        rows = getter()
        assert rows and rows == plan(op, backend, 100).rows
        assert {row[6] for row in rows} == {MODEL_PATH}


def test_trtllm_series_rows_fit_the_series_worker(monkeypatch):
    module = _load_backend(monkeypatch, "trtllm")
    captured = []
    monkeypatch.setattr(module, "run_msa_module_series", lambda *args, **kwargs: captured.append((args, kwargs)))

    row = module.get_msa_context_module_test_cases()[0]
    module.run_msa_module_series_worker(*row, perf_filename="context.txt")

    points, batch_size, num_heads, kv_dtype, compute, gemm, model_path = row
    assert captured == [
        (
            (points, batch_size, num_heads, kv_dtype, compute, gemm, "context.txt"),
            {"model_path": model_path, "device": "cuda:0"},
        )
    ]


def test_vllm_cases_fit_the_point_worker(monkeypatch):
    module = _load_backend(monkeypatch, "vllm")
    captured = []
    monkeypatch.setattr(module, "run_msa_module", lambda **kwargs: captured.append(kwargs) or 1.0)

    context = next(case for case in module.get_msa_context_module_test_cases() if case.prefix_len)
    generation = module.get_msa_generation_module_test_cases()[0]
    module.run_msa_module_worker(*context, perf_filename="context.txt")
    module.run_msa_module_worker(*generation, perf_filename="generation.txt")

    assert captured[0]["prefix_len"] == context.prefix_len
    assert captured[0]["perf_filename"] == "context.txt"
    assert captured[1]["prefix_len"] == 0
    assert captured[1]["seq_len"] == generation.seq_len


def test_vllm_refuses_a_decode_point_whose_kv_cache_cannot_fit_before_allocating(monkeypatch):
    module = _load_backend(monkeypatch, "vllm")
    monkeypatch.setattr(
        module.torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=2**30), raising=False
    )
    monkeypatch.setattr(module, "setup_distributed", lambda _device: pytest.fail("allocated for a refused point"))

    with pytest.raises(RuntimeError, match=r"^FIXME\(kernel-limit\): does not fit: the KV cache needs"):
        module.run_msa_module(
            None,
            seq_len=131072,
            batch_size=64,
            num_heads=64,
            kv_cache_dtype="bfloat16",
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            perf_filename="msa_generation_module_perf.txt",
            model_path=MODEL_PATH,
        )


def test_vllm_standalone_cli_runs_the_plan_cases_and_quick_defaults_to_prefix_zero(monkeypatch):
    from collector.op_plans import ModuleCase

    module = _load_backend(monkeypatch, "vllm")
    cases = [
        ModuleCase(16, 1, 64, "bfloat16", "bfloat16", "bfloat16", MODEL_PATH, 0),
        ModuleCase(16, 1, 64, "bfloat16", "bfloat16", "bfloat16", MODEL_PATH, 512),
    ]
    monkeypatch.setattr(module, "plan", lambda *_args: SimpleNamespace(rows=cases))
    calls = []
    monkeypatch.setattr(module, "run_msa_module", lambda **kwargs: calls.append(kwargs))

    monkeypatch.setattr(sys, "argv", ["collect_msa_module.py", "--mode", "context"])
    module.main()
    assert [call["prefix_len"] for call in calls] == [0, 512]

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["collect_msa_module.py", "--mode", "context", "--quick"])
    module.main()
    assert [call["prefix_len"] for call in calls] == [0]
