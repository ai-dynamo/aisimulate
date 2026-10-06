# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the module collector against fake framework state and timing."""

import ast
import math
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
COLLECTOR = Path(__file__).resolve().parents[3] / "collector"


def _load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _module_runner(*, graph_flag=True, failure=None):
    events, published, benchmark_calls = [], [], []
    error = RuntimeError(f"{failure} failed")
    active_contexts = []

    @contextmanager
    def context(name):
        active_contexts.append(name)
        try:
            yield
        finally:
            assert active_contexts.pop() == name
            events.append(f"exit {name}")

    class Tensor:
        def unchanged(self, *args, **kwargs):
            return self

        unsqueeze = expand = reshape = contiguous = unchanged

    def forward(*args):
        if failure == "forward":
            raise error
        events.append("forward")

    layer = SimpleNamespace(kv_cache_dtype="fp8_ds_mla")
    module = SimpleNamespace(forward=forward, mla_attn=SimpleNamespace(mla_attn=layer))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=7168),
            hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
        ),
        compilation_config=SimpleNamespace(static_forward_context={"model.layers.0.self_attn.attn": SimpleNamespace()}),
    )

    def create_module(**kwargs):
        if failure == "construction":
            raise error
        assert kwargs["max_query_len"] == 16
        assert kwargs["max_seq_len"] in {16, 144}
        return module, config

    @contextmanager
    def benchmark(**kwargs):
        benchmark_calls.append(kwargs)
        assert kwargs["use_cuda_graph"] is True
        assert kwargs["allow_graph_fail"] is False
        if failure == "capture":
            raise error
        yield {"used_cuda_graph": graph_flag, "latency_ms": 0.015, "power_stats": None}
        events.append("graph teardown")

    def cleanup():
        assert active_contexts == []
        events.append("cleanup")

    namespace = {
        "ExitStack": ExitStack,
        "wraps": wraps,
        "torch": SimpleNamespace(
            device=lambda value: value,
            long="long",
            bfloat16="bfloat16",
            arange=lambda *args, **kwargs: Tensor(),
            full=lambda *args, **kwargs: Tensor(),
            inference_mode=nullcontext,
            cuda=SimpleNamespace(
                set_device=lambda value: None,
                get_device_name=lambda value: "B200",
                OutOfMemoryError=MemoryError,
            ),
        ),
        "setup_distributed": lambda device: None,
        "enable_engine_fused_ops": lambda: None,
        "init_workspace_manager": lambda device: events.append("workspace init"),
        "_create_attention_module": create_module,
        "_process_module_weights": lambda *args: None,
        "_create_kv_cache_and_metadata": lambda **kwargs: (Tensor(), object(), None, None, None),
        "set_current_vllm_config": lambda config: context("config"),
        "set_forward_context": lambda *args: context("forward"),
        "benchmark_with_power": benchmark,
        "_mla_backend_name": lambda *args: "FLASHINFER_MLA_SPARSE",
        "log_perf": lambda **kwargs: published.append(kwargs),
        "_cleanup": cleanup,
        "vllm_version": "0.25.0",
        "traceback": SimpleNamespace(print_exc=lambda: None),
    }
    _load_function(COLLECTOR / "vllm/utils.py", "with_exit_stack", namespace)
    run = _load_function(COLLECTOR / "vllm/collect_mla_module.py", "run_mla_module", namespace)

    def execute(attn_type="dsa", phase="context"):
        return run(
            seq_len=16,
            batch_size=2,
            num_heads=8,
            kv_cache_dtype="fp8",
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            perf_filename=f"{attn_type}_{phase}_module_perf.txt",
            prefix_len=128,
            model_path="fixture/model",
            attn_type=attn_type,
        )

    return execute, events, published, benchmark_calls, error


@pytest.mark.parametrize("attn_type", ["mla", "dsa"])
@pytest.mark.parametrize("phase", ["context", "generation"])
def test_graph_publication_and_workspace_teardown_after_context_exit(attn_type, phase):
    execute, events, published, calls, _ = _module_runner()
    assert execute(attn_type, phase) == 0.015
    assert len(calls) == len(published) == 1
    assert events[-4:] == ["graph teardown", "exit forward", "exit config", "cleanup"]
    row = published[0]["item_list"][0]
    assert row["latency"] == "0.0150"
    assert row["step"] == (128 if phase == "context" else 16)
    assert published[0]["kernel_source"] == "FLASHINFER_MLA_SPARSE"


@pytest.mark.parametrize("failure", ["construction", "forward", "capture"])
def test_failure_propagates_and_workspace_is_released_without_publication(failure):
    execute, events, published, _, error = _module_runner(failure=failure)
    with pytest.raises(RuntimeError) as caught:
        execute()
    assert caught.value is error
    assert events[-1] == "cleanup"
    assert events.count("cleanup") == 1
    assert published == []


@pytest.mark.parametrize("graph_flag", [False, None, 0, 1, "true"])
def test_unproven_or_eager_latency_is_not_published(graph_flag):
    execute, events, published, _, _ = _module_runner(graph_flag=graph_flag)
    with pytest.raises(RuntimeError, match="refusing to publish eager timing"):
        execute()
    assert published == []
    assert events[-1] == "cleanup"


@pytest.mark.parametrize(
    "batch,query,prefix,is_context,expected_tokens",
    [
        (1, 1057, 1047488, True, 131072),
        (1, 1057, 0, True, 131072),
        (8, 32768, 1047488, True, 262144),
        (8, 32768, 0, True, 262144),
        (4, 1047488, 0, False, 4),
    ],
)
def test_indexer_output_budget_excludes_cached_prefix(monkeypatch, batch, query, prefix, is_context, expected_tokens):
    """Run actual construction and inspect both tensor shape and context capacity."""
    allocations, configs = [], []

    class Module:
        def __init__(self, **kwargs):
            assert kwargs["topk_indices_buffer"] == (expected_tokens, 2048)

        def parameters(self):
            return []

        named_parameters = named_buffers = parameters

        def to(self, device):
            return self

        def eval(self):
            pass

        def requires_grad_(self, enabled):
            assert enabled is False

    def create_config(**kwargs):
        configs.append(kwargs)
        hf = SimpleNamespace(
            index_topk=2048,
            hidden_size=7168,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            kv_lora_rank=512,
            max_position_embeddings=2097152,
        )
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_text_config=hf, dtype="bf16"),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=kwargs["max_num_batched_tokens"]),
            cache_config=object(),
        )

    def empty(*shape, **kwargs):
        allocations.append(shape)
        return shape

    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.deepseek_v2", SimpleNamespace(DeepseekV2MLAAttention=Module)
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.attention.mla_attention",
        SimpleNamespace(backend_supports_prefill_query_quantization=SimpleNamespace(cache_clear=lambda: None)),
    )
    monkeypatch.setitem(
        sys.modules, "vllm.utils.torch_utils", SimpleNamespace(set_default_torch_dtype=lambda dtype: nullcontext())
    )
    namespace = {
        "math": math,
        "torch": SimpleNamespace(empty=empty, int32="int32", no_grad=nullcontext),
        "_resolve_model_path": lambda model: model,
        "create_vllm_config": create_config,
        "_create_gemm_quant_config": lambda gemm: None,
        "_move_module_preserving_buffers": lambda module, device: module.to(device),
        "_initialize_synthetic_parameters": lambda module: None,
        "set_current_vllm_config": lambda config: nullcontext(),
    }
    create = _load_function(COLLECTOR / "vllm/collect_mla_module.py", "_create_attention_module", namespace)
    create(
        model_path="fixture/model",
        attn_type="dsa",
        num_heads=8,
        use_fp8_kv_cache=True,
        max_seq_len=prefix + query,
        max_batch_size=batch,
        max_query_len=query,
        is_context=is_context,
    )
    assert allocations == [(expected_tokens, 2048)]
    assert configs[0]["max_model_len"] == max(prefix + query, 4096)
    assert configs[0]["num_gpu_blocks"] == max(1 + math.ceil((prefix + query + 1) / 64) * batch, 8192)
