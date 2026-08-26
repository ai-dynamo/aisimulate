# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public MSA getter contracts for the TRT-LLM and vLLM collectors."""

from __future__ import annotations

import importlib.util
import sys
import types
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
    )


def _install_helper_stub(monkeypatch):
    _stub_module(
        monkeypatch,
        "collector.helper",
        _resolve_local_model_path=lambda path: path,
        benchmark_with_power=_noop,
        get_sm_version=lambda: 100,
        log_perf=lambda **_kwargs: True,
    )


def _install_trtllm_stubs(monkeypatch):
    _stub_torch(monkeypatch)
    transformers = _stub_module(monkeypatch, "transformers", PretrainedConfig=_Dummy)
    trtllm = _stub_module(monkeypatch, "tensorrt_llm", __version__="1.3.0rc23")
    for package in (
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.attention_backend",
        "tensorrt_llm._torch.models",
        "tensorrt_llm._torch.pyexecutor",
        "tensorrt_llm.bindings",
        "tensorrt_llm.bindings.internal",
        "tensorrt_llm.llmapi",
        "tensorrt_llm.models",
        "tensorrt_llm.quantization",
    ):
        _stub_module(monkeypatch, package)
    _stub_module(monkeypatch, "tensorrt_llm._torch.attention_backend.interface", AttentionRuntimeFeatures=_Dummy)
    _stub_module(monkeypatch, "tensorrt_llm._torch.attention_backend.utils", get_attention_backend=_noop)
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


def _invocation_identity(case: list[object]) -> tuple[object, ...]:
    return tuple(case)


def _consumer_key(case: list[object], phase: str) -> tuple[object, ...]:
    seq_len, batch_size, num_heads, kv_dtype, compute_dtype, gemm_type, _model_path, *rest = case
    if phase == "context":
        prefix_len = rest[0]
        return (ARCHITECTURE, compute_dtype, kv_dtype, gemm_type, num_heads, prefix_len, seq_len, batch_size)
    return (ARCHITECTURE, kv_dtype, gemm_type, num_heads, batch_size, seq_len + 1)


@pytest.mark.parametrize("backend", ["trtllm", "vllm"])
def test_public_msa_getters_populate_both_phases_without_duplicate_identities(monkeypatch, backend):
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    module = _load_backend(monkeypatch, backend)

    for phase, getter in (
        ("context", module.get_msa_context_module_test_cases),
        ("generation", module.get_msa_generation_module_test_cases),
    ):
        cases = getter()
        assert cases, f"{backend}/{phase} returned no cases"
        assert all(case[6] == MODEL_PATH for case in cases)
        assert len(cases) == len({_invocation_identity(case) for case in cases})
        consumer_keys = [_consumer_key(case, phase) for case in cases]
        assert len(consumer_keys) == len(set(consumer_keys))

        expected_length = 8 if phase == "context" else 7
        assert {len(case) for case in cases} == {expected_length}


@pytest.mark.parametrize("backend", ["trtllm", "vllm"])
def test_targeted_msa_getters_keep_prefixes_and_worker_tuple_compatibility(monkeypatch, backend):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL_PATH)
    module = _load_backend(monkeypatch, backend)
    context_cases = module.get_msa_context_module_test_cases()
    generation_cases = module.get_msa_generation_module_test_cases()

    assert {case[7] for case in context_cases} == {0, 128}
    assert generation_cases
    assert {case[6] for case in context_cases + generation_cases} == {MODEL_PATH}

    captured = []
    monkeypatch.setattr(module, "run_msa_module", lambda **kwargs: captured.append(kwargs) or 1.0)
    module.run_msa_module_worker(*context_cases[0], perf_filename="context.txt")
    module.run_msa_module_worker(*generation_cases[0], perf_filename="generation.txt")

    assert captured[0]["prefix_len"] == context_cases[0][7]
    assert captured[0]["perf_filename"] == "context.txt"
    assert captured[1]["prefix_len"] == 0
    assert captured[1]["perf_filename"] == "generation.txt"


def test_vllm_public_generation_getter_applies_live_memory_filter(monkeypatch, capsys):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL_PATH)
    module = _load_backend(monkeypatch, "vllm")
    monkeypatch.setattr(module, "_device_total_memory_bytes", lambda: 1)

    assert module.get_msa_generation_module_test_cases() == []
    assert "msa_generation_module: dropped" in capsys.readouterr().out


def test_vllm_standalone_full_sweep_keeps_declared_prefixes_and_quick_defaults_to_zero(monkeypatch):
    module = _load_backend(monkeypatch, "vllm")
    model_spec = SimpleNamespace(model_path=MODEL_PATH)
    monkeypatch.setattr(module, "get_mla_module_model_specs", lambda **_kwargs: [model_spec])
    monkeypatch.setattr(
        module,
        "get_context_test_cases",
        lambda: [
            [16, 1, 64, "bfloat16", "bfloat16", "bfloat16", 0],
            [16, 1, 64, "bfloat16", "bfloat16", "bfloat16", 128],
        ],
    )
    calls = []
    monkeypatch.setattr(module, "run_msa_module", lambda **kwargs: calls.append(kwargs))

    monkeypatch.setattr(sys, "argv", ["collect_msa_module.py", "--mode", "context"])
    module.main()
    assert [call["prefix_len"] for call in calls] == [0, 128]

    calls.clear()
    monkeypatch.setattr(sys, "argv", ["collect_msa_module.py", "--mode", "context", "--quick"])
    module.main()
    assert [call["prefix_len"] for call in calls] == [0]
