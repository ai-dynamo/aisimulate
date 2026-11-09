# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest
from collector.helper import create_test_case_id

pytestmark = pytest.mark.unit


@pytest.fixture
def dsv4_module(monkeypatch):
    def install_module(name: str, *, package: bool = False, **attrs):
        module = ModuleType(name)
        if package:
            module.__path__ = []
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class QuantizeMethodBase:
        pass

    class DeepseekV4Attention:
        pass

    def context(*_args, **_kwargs):
        return nullcontext()

    torch = install_module(
        "torch",
        OutOfMemoryError=RuntimeError,
        bfloat16=object(),
        device=lambda value: value,
        full=lambda *_args, **_kwargs: object(),
        inference_mode=context,
    )
    torch.cuda = SimpleNamespace(
        OutOfMemoryError=RuntimeError,
        empty_cache=lambda: None,
        get_device_name=lambda _device: "fake-gpu",
        synchronize=lambda: None,
    )
    install_module("vllm", package=True)
    install_module("vllm.config", set_current_vllm_config=context)
    install_module("vllm.forward_context", set_forward_context=context)
    install_module("vllm.model_executor", package=True)
    install_module("vllm.model_executor.layers", package=True)
    install_module("vllm.model_executor.layers.quantization", package=True)
    install_module(
        "vllm.model_executor.layers.quantization.base_config",
        QuantizeMethodBase=QuantizeMethodBase,
    )
    install_module("vllm.models", package=True)
    install_module("vllm.models.deepseek_v4", package=True)
    install_module("vllm.models.deepseek_v4.attention", DeepseekV4Attention=DeepseekV4Attention)
    install_module("vllm.models.deepseek_v4.nvidia", package=True)
    install_module("vllm.models.deepseek_v4.nvidia.model", _select_dsv4_attn_cls=lambda _config: None)
    install_module("vllm.platforms", current_platform=SimpleNamespace())
    install_module("vllm.utils", package=True)
    install_module(
        "vllm.utils.deep_gemm",
        fp8_fp4_paged_mqa_logits=lambda *_args, **_kwargs: None,
        get_paged_mqa_logits_metadata=lambda *_args, **_kwargs: None,
    )
    install_module("vllm.utils.platform_utils", num_compute_units=lambda *_args, **_kwargs: 1)
    install_module("vllm.utils.torch_utils", set_default_torch_dtype=context)
    install_module("vllm.version", __version__="0.24.0")
    install_module("vllm.v1", package=True)
    install_module("vllm.v1.worker", package=True)
    install_module("vllm.v1.worker.workspace", init_workspace_manager=lambda *_args, **_kwargs: None)
    install_module(
        "collector.vllm.utils",
        BatchSpec=SimpleNamespace,
        create_common_attn_metadata=lambda *_args, **_kwargs: None,
        create_vllm_config=lambda *_args, **_kwargs: None,
        enable_engine_fused_ops=lambda *_args, **_kwargs: None,
        setup_distributed=lambda *_args, **_kwargs: None,
    )

    module_name = "collector.vllm.collect_dsv4_attn"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


def test_rows_are_the_plan_points_and_carry_the_context_prefix(dsv4_module, monkeypatch):
    from collector.op_plans import plan

    mod = dsv4_module
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8")
    monkeypatch.setattr(mod, "get_sm_version", lambda: 100)

    cases = mod.get_dsv4_csa_context_test_cases()

    assert cases == plan("dsv4_csa_context_module", "vllm", 100).rows
    assert {len(case) for case in cases} == {10}
    # vLLM times the FP8 checkpoints' own projections on an FP8 KV cache.
    assert {tuple(case[3:6]) for case in cases} == {("fp8", "bfloat16", "fp8_block")}
    assert all(case[1] * case[0] <= 16384 for case in cases)
    assert all(case[1] * (case[9] + case[0]) <= 1048576 for case in cases)
    assert max(case[9] for case in cases) == 1048575

    task_ids = {create_test_case_id(case, "run_dsv4_attn_worker", mod.__name__) for case in cases}
    assert len(task_ids) == len(cases)

    generation_cases = mod.get_dsv4_csa_generation_test_cases()
    assert generation_cases
    assert {len(case) for case in generation_cases} == {9}
    assert max(case[0] for case in generation_cases) == 1048575

    forwarded = {}
    monkeypatch.setattr(mod, "_init_cuda", lambda _device: None)
    monkeypatch.setattr(mod, "_bench_attention_shape", lambda **kwargs: forwarded.update(kwargs))
    nonzero_prefix_case = next(case for case in cases if case[9] == 2048)
    mod.run_dsv4_attn_worker(
        *nonzero_prefix_case,
        perf_filename="dsv4_csa_context_module_perf.txt",
    )
    assert forwarded["prefix_len"] == 2048
    assert (forwarded["warming_up"], forwarded["test_ite"]) == (5, 10)


def test_sparse_rows_are_the_plan_cases_including_short_shapes(dsv4_module, monkeypatch):
    from collector.op_plans import plan

    mod = dsv4_module
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "sgl-project/DeepSeek-V4-Flash-FP8")
    monkeypatch.setattr(mod, "get_sm_version", lambda: 100)

    rows = mod.get_dsv4_paged_mqa_logits_test_cases()

    assert rows == plan("dsv4_paged_mqa_logits_module", "vllm", 100).cases
    # run_dsv4_sparse_kernel_worker's arguments; the case prints as the list it replaces.
    assert str(rows[0]) == "[1, 1, 0, 1, 'paged_mqa_logits', 'sgl-project/DeepSeek-V4-Flash-FP8']"
    assert all(b * s <= 8192 and b * (s + p) <= 1048576 for b, s, p, *_rest in rows)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"kv_cache_dtype": "bfloat16"}, "kv_cache_dtype"),
        ({"compute_dtype": "float16"}, "compute_dtype"),
        ({"attention_backend": "FLASH_ATTN"}, "attention_backend must be unset"),
    ],
)
def test_dsv4_worker_rejects_ignored_runtime_inputs(dsv4_module, override, message):
    args = {
        "seq_len": 64,
        "batch_size": 1,
        "tp_size": 1,
        "kv_cache_dtype": "fp8",
        "compute_dtype": "bfloat16",
        "gemm_type": "fp8_block",
        "model_path": "model",
        "attn_kind": "csa",
        "attention_backend": None,
        "perf_filename": "dsv4_csa_context_module_perf.txt",
    }
    args.update(override)

    with pytest.raises(ValueError, match=message):
        dsv4_module.run_dsv4_attn_worker(**args)


@pytest.mark.parametrize("kernel", ["paged_mqa_logits"])
def test_sparse_worker_observes_short_shapes(dsv4_module, monkeypatch, kernel):
    calls = {}
    monkeypatch.setattr(dsv4_module, "_init_cuda", lambda device: calls.setdefault("device", device))
    monkeypatch.setattr(dsv4_module, "_bench_sparse_kernel_shape", lambda **kwargs: calls.update(kwargs))

    dsv4_module.run_dsv4_sparse_kernel_worker(
        1,
        1,
        0,
        1,
        kernel,
        "model",
        perf_filename="perf.txt",
    )

    assert calls["kernel"] == kernel
    assert calls["isl"] == 1
    assert calls["past_kv"] == 0


def test_sparse_worker_raises_beyond_max_position_embeddings(dsv4_module, monkeypatch):
    monkeypatch.setattr(dsv4_module, "_init_cuda", lambda device: None)
    monkeypatch.setattr(
        dsv4_module, "_bench_sparse_kernel_shape", lambda **kwargs: pytest.fail("must raise before benchmarking")
    )

    over_limit = dsv4_module.get_dsv4_sweep().sparse_max_full_sequence_length + 1
    with pytest.raises(ValueError, match="max_position_embeddings"):
        dsv4_module.run_dsv4_sparse_kernel_worker(
            1,
            1,
            over_limit,
            1,
            "paged_mqa_logits",
            "model",
            perf_filename="perf.txt",
        )


@pytest.mark.parametrize(
    ("mode", "seq_len", "prefix_len", "metadata_seq_len", "query_len", "logged_isl", "logged_step"),
    [
        ("context", 64, 128, 192, 64, 64, 128),
        ("generation", 64, 0, 65, 1, 1, 64),
    ],
)
def test_bench_uses_total_cache_length_and_fresh_query_length(
    dsv4_module,
    monkeypatch,
    mode,
    seq_len,
    prefix_len,
    metadata_seq_len,
    query_len,
    logged_isl,
    logged_step,
):
    mod = dsv4_module
    calls = {}

    class FakeAttention:
        prefix = "model.layers.0.attn"
        n_local_heads = 8

        def __call__(self, *_args):
            return None

    class FakeLayer:
        def get_attn_backend(self):
            return SimpleNamespace(get_name=lambda: "FAKE_BACKEND")

        def get_kv_cache_spec(self, _config):
            return object()

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"],
                hidden_size=8,
                # Native head count; must equal FakeAttention.n_local_heads * tp_size
                # to satisfy the unified geometry cross-check at the write site.
                num_attention_heads=8,
            )
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={"model.layers.0.attn": FakeLayer()},
        ),
    )

    @contextmanager
    def create_module(**kwargs):
        calls["create"] = kwargs
        yield FakeAttention(), config

    @contextmanager
    def benchmark(**_kwargs):
        yield {"latency_ms": 1.25, "power_stats": None}

    monkeypatch.setattr(mod, "_tp_simulation", lambda _tp: nullcontext())
    monkeypatch.setattr(mod, "_create_dsv4_attention_module", create_module)

    def make_metadata(**kwargs):
        calls["metadata"] = kwargs
        return SimpleNamespace(positions=object())

    monkeypatch.setattr(mod, "_make_common_metadata", make_metadata)
    monkeypatch.setattr(mod, "_build_metadata_and_bind_caches", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(mod, "set_current_vllm_config", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(mod, "set_forward_context", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(mod, "benchmark_with_power", benchmark)
    monkeypatch.setattr(mod, "log_perf", lambda **kwargs: calls.setdefault("log", kwargs))
    monkeypatch.setattr(mod.torch, "full", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(mod.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(mod.torch.cuda, "get_device_name", lambda _device: "fake-gpu")
    monkeypatch.setattr(mod.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(mod.gc, "collect", lambda: None)

    latency = mod._bench_attention_shape(
        model_path="model",
        attn_kind="csa",
        mode=mode,
        batch_size=2,
        seq_len=seq_len,
        prefix_len=prefix_len,
        tp_size=1,
        gemm_type="fp8_block",
        device="cuda:0",
        perf_filename="perf.txt",
        warming_up=1,
        test_ite=1,
    )

    assert latency == 1.25
    assert calls["create"]["seq_len"] == metadata_seq_len
    assert calls["create"]["query_len"] == query_len
    assert calls["metadata"]["seq_len"] == metadata_seq_len
    assert calls["metadata"]["query_len"] == query_len
    row = calls["log"]["item_list"][0]
    assert row["isl"] == logged_isl
    assert row["step"] == logged_step
    # kernel_source must be the registered layer's actual backend, not an
    # assumed name.
    assert calls["log"]["kernel_source"] == "FAKE_BACKEND"


@pytest.mark.parametrize(("mode", "expected_prefix"), [("context", 128), ("generation", 0)])
def test_cli_past_kv_is_context_prefix_only(dsv4_module, monkeypatch, mode, expected_prefix):
    mod = dsv4_module
    calls = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_dsv4_attn.py",
            "--mode",
            mode,
            "--seq-len",
            "64",
            "--past-kv",
            "128",
        ],
    )
    monkeypatch.setattr(mod, "_init_cuda", lambda _device: None)
    monkeypatch.setattr(mod, "_bench_attention_shape", lambda **kwargs: calls.update(kwargs))

    mod.main()

    assert calls["prefix_len"] == expected_prefix
    assert calls["seq_len"] == 64


@pytest.mark.parametrize(
    ("max_position", "seq_len", "expected"),
    [
        # Serving's default: the checkpoint's max_position_embeddings (1M for DeepSeek-V4).
        (1048576, 16385, 1048576),
        (1048576, 1048576, 1048576),
        # No declared limit: the previous 4096 floor, never below the point itself.
        (None, 192, 4096),
        (None, 8193, 8193),
    ],
)
def test_module_is_built_at_serving_default_max_model_len(dsv4_module, monkeypatch, max_position, seq_len, expected):
    """DSV4 plans its prefill chunks, indexer buffer and decode logits by max_model_len
    (sparse_swa.py:250-329, deepseek_v4/attention.py:995-1002 @v0.32.0): build at serving's."""
    mod = dsv4_module
    seen = {}

    class Built(Exception):
        pass

    def create_vllm_config(**kwargs):
        seen.update(kwargs)
        raise Built

    config = {} if max_position is None else {"max_position_embeddings": max_position}
    monkeypatch.setattr(mod, "_patched_config_dir", lambda *_args, **_kwargs: nullcontext("config-dir"))
    monkeypatch.setattr(mod, "_read_model_config", lambda _model: config)
    monkeypatch.setattr(mod, "create_vllm_config", create_vllm_config)

    with (
        pytest.raises(Built),
        mod._create_dsv4_attention_module(
            model_path="sgl-project/DeepSeek-V4-Flash-FP8",
            attn_kind="csa",
            batch_size=256,
            seq_len=seq_len,
            tp_size=1,
            is_context=True,
            device="cuda:0",
            query_len=1,
        ),
    ):
        pass

    assert seen["max_model_len"] == expected
    assert seen["model_name"] == "config-dir"
