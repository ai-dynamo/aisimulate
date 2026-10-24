# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU execution tests for rejecting an observed, unqualified timing boundary.

Load complete production functions without importing Torch or SGLang globally.
The fake runtime exercises control flow and publication, not GPU qualification.
"""

import ast
import json
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/sglang/collect_mla_module.py"
GRAPH_SOURCE = SOURCE.with_name("dsa_prefill_graph.py")


def _load(names, namespace=None, source=SOURCE):
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    loaded = dict(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), loaded)
    return loaded


@pytest.fixture
def runtime(monkeypatch):
    events, rows = [], []

    def record(name):
        return lambda *_args, **_kwargs: events.append(name)

    def module(name, **attrs):
        fake = ModuleType(name)
        fake.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, fake)

    class Request(SimpleNamespace):
        def set_extend_input_len(self, length):
            self.extend_input_len = length

    batch = SimpleNamespace(prepare_for_extend=record("prepare"))
    forward_batch = SimpleNamespace()
    module(
        "sglang.srt.layers.communicator",
        AttentionInputs=lambda *_: object(),
        get_attn_tp_context=lambda: SimpleNamespace(set_attn_inputs=record("inputs")),
    )
    module(
        "sglang.srt.managers.schedule_batch",
        Req=Request,
        ScheduleBatch=SimpleNamespace(init_new=lambda **_: batch),
    )
    module("sglang.srt.mem_cache.cache_init_params", CacheInitParams=SimpleNamespace)
    module("sglang.srt.mem_cache.chunk_cache", ChunkCache=lambda _: object())
    module(
        "sglang.srt.model_executor.forward_batch_info",
        ForwardBatch=SimpleNamespace(init_new=lambda *_: forward_batch),
    )
    module("sglang.srt.sampling.sampling_params", SamplingParams=SimpleNamespace)
    module("sglang.srt.speculative.spec_info", SpeculativeAlgorithm=SimpleNamespace(NONE=None))
    module("sglang.srt.utils", BumpAllocator=lambda **_: object())
    module("sglang.srt.managers.overlap_utils", resolve_forward_inputs=record("resolve"))
    module("flashinfer.autotuner", autotune=lambda _: nullcontext())

    positions = Mock()
    positions.unsqueeze.return_value.expand.return_value.contiguous.return_value.flatten.return_value = object()

    def event(**_):
        return SimpleNamespace(record=record("event"), elapsed_time=lambda _: 6.0)

    torch = SimpleNamespace(
        randint=lambda _low, _high, shape: SimpleNamespace(tolist=lambda: [0] * shape[0]),
        randn=lambda *_args, **_kwargs: object(),
        arange=lambda *_args, **_kwargs: positions,
        bfloat16="bfloat16",
        float32="float32",
        no_grad=nullcontext,
        OutOfMemoryError=MemoryError,
        cuda=SimpleNamespace(
            Event=event,
            synchronize=record("sync"),
            empty_cache=record("empty_cache"),
            get_device_name=lambda _: "FakeGPU",
            OutOfMemoryError=MemoryError,
        ),
    )
    attention = Mock(side_effect=record("attention"))
    attention.q_lora_rank = 2048
    attention.kv_lora_rank = 512
    attention.qk_rope_head_dim = 64
    runner = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(hidden_size=6144),
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)]),
        ),
        model_config=object(),
        server_args=SimpleNamespace(
            attention_backend="dsa",
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend="tc_piecewise", bs=[4, 144, 2048])),
        ),
        req_to_token_pool=SimpleNamespace(clear=record("request_clear")),
        token_to_kv_pool_allocator=SimpleNamespace(clear=record("kv_clear"), page_size=64),
        attn_backend=SimpleNamespace(
            init_forward_metadata=record("metadata"), use_mha=False, dsa_prefill_impl="flashmla_sparse"
        ),
    )

    class GraphContext:
        close_error = None

        def __enter__(self):
            events.append("native_enter")
            return record("native_replay")

        def __exit__(self, *_):
            events.append("native_close")
            if self.close_error is not None:
                raise self.close_error

    graph = GraphContext()
    bucket = _load(["graph_token_bucket"], source=GRAPH_SOURCE)["graph_token_bucket"]
    module(
        "collector.sglang.dsa_prefill_graph",
        graph_token_bucket=bucket,
        dsa_prefill_graph=lambda *_args, **_kwargs: graph,
    )

    def log_perf(**row):
        rows.append(row)
        return True

    loaded = _load(
        ["MeasurementBoundaryError", "PerfLogWriteError", "_run_prefill", "run_attention_torch"],
        {
            "torch": torch,
            "json": json,
            "traceback": traceback,
            "_import_sglang_forward_context": lambda: (SimpleNamespace, lambda *_: nullcontext()),
            "_alloc_prefix_indices": lambda _runner, size, _prefix: [object()] * size,
            "_temporarily_chunked_alloc_extend": lambda *_: nullcontext(),
            "_dsa_forward_input_relay": lambda *_: object(),
            "_validate_dsa_forward_tokens": lambda *_: None,
            "_initialize_dsa_history": record("history"),
            "_runtime_chunk_size": lambda _: 8192,
            "_dsa_skip_indexer_enabled": lambda *_: False,
            "_resolve_perf_path": lambda _path, filename: filename,
            "_module_model_architecture": lambda _: "GlmMoeDsaForCausalLM",
            "get_version": lambda _: "0.5.14",
            "log_perf": log_perf,
        },
    )
    kwargs = dict(
        model_runner=runner,
        attention_module=attention,
        batch_size=1,
        seq_length=1,
        head_num=8,
        num_warmup=2,
        num_iterations=3,
        device="cuda",
        output_path=None,
        dummy_qkv_latent_func=attention.prepare_qkv_latent,
        attn_type="dsa",
        model_path="glm",
        architecture="GlmMoeDsaForCausalLM",
        backend_name="dsa",
        version="0.5.14",
        device_name="FakeGPU",
        log_mla_dtype="bfloat16",
        log_kv_dtype="fp8",
        log_gemm_type="bfloat16",
        target_tp_size=1,
    )
    return SimpleNamespace(events=events, rows=rows, graph=graph, loaded=loaded, runner=runner, kwargs=kwargs)


@pytest.mark.parametrize("seq_length", [1, 129])
def test_observed_native_tc_rejects_before_timing_or_publication_and_cleans_up(runtime, seq_length):
    runtime.kwargs["seq_length"] = seq_length
    error_type = runtime.loaded["MeasurementBoundaryError"]
    with pytest.raises(error_type, match="sglang_dsa_indexer_flashmla_sparse"):
        runtime.loaded["_run_prefill"](**runtime.kwargs)
    assert "metadata" in runtime.events
    assert "native_enter" in runtime.events
    assert not runtime.rows
    assert "native_replay" not in runtime.events
    assert "event" not in runtime.events
    assert "attention" not in runtime.events  # No eager fallback after native dispatch.
    assert runtime.events[-4:] == ["native_close", "request_clear", "kv_clear", "empty_cache"]


def test_graph_close_failure_still_cleans_remaining_pools_and_preserves_boundary_cause(runtime):
    runtime.graph.close_error = RuntimeError("close failed")
    with pytest.raises(RuntimeError, match="native_prefill_graph.close: RuntimeError: close failed") as exc:
        runtime.loaded["_run_prefill"](**runtime.kwargs)
    assert isinstance(exc.value.__context__, runtime.loaded["MeasurementBoundaryError"])
    assert not runtime.rows
    assert runtime.events[-4:] == ["native_close", "request_clear", "kv_clear", "empty_cache"]


@pytest.mark.parametrize("seq_length", [4096, 8192])
def test_prefill_outside_native_graph_coverage_still_measures_and_writes(runtime, seq_length):
    runtime.kwargs["seq_length"] = seq_length
    assert runtime.loaded["_run_prefill"](**runtime.kwargs) is True
    assert "native_enter" not in runtime.events
    assert runtime.events.count("attention") == 5  # Two warmups and three timed forwards.
    assert len(runtime.rows) == 1
    row = runtime.rows[0]
    assert row["op_name"] == "dsa_context_module"
    assert row["kernel_source"] == "sglang_dsa_indexer_flashmla_sparse"
    assert row["item_list"][0]["isl"] == seq_length
    assert row["item_list"][0]["latency"] == "2.0000"
    assert runtime.events[-3:] == ["request_clear", "kv_clear", "empty_cache"]


def _run_group(runtime, cases):
    return runtime.loaded["run_attention_torch"](
        runtime.runner,
        cases,
        8,
        0,
        2,
        3,
        "cuda",
        None,
        attn_type="dsa",
        model_path="glm",
        kv_cache_dtype="fp8",
        compute_dtype="bfloat16",
        gemm_type="bfloat16",
    )


def test_mixed_group_preserves_eligible_rows_and_reports_each_boundary_failure(runtime, capsys):
    decode_calls = []

    def decode(**kwargs):
        # Group contract: an eligible decode worker remains reachable and its
        # success contributes to logged_count; decode kernels have separate tests.
        decode_calls.append(kwargs)
        runtime.rows.append({"op_name": "dsa_generation_module"})
        return True

    runtime.loaded["_run_decode"] = decode
    cases = [(1, 1, True, 0), (1, 4096, True, 0), (1, 129, True, 128), (1, 8192, False)]
    with pytest.raises(runtime.loaded["MeasurementBoundaryError"], match="collection incomplete") as exc:
        _run_group(runtime, cases)
    receipt = json.loads(str(exc.value).split(": ", 1)[1])
    assert receipt["logged_count"] == 2
    assert [failure["seq_length"] for failure in receipt["boundary_failures"]] == [1, 129]
    assert [failure["prefix_len"] for failure in receipt["boundary_failures"]] == [0, 128]
    assert all(failure["error_type"] == "MeasurementBoundaryError" for failure in receipt["boundary_failures"])
    assert [row["op_name"] for row in runtime.rows] == ["dsa_context_module", "dsa_generation_module"]
    assert len(decode_calls) == 1
    assert decode_calls[0]["seq_length"] == 8192
    assert capsys.readouterr().out.count("DSA measurement boundary failure:") == 2


@pytest.mark.parametrize("is_prefill", [False, True])
def test_group_does_not_swallow_or_reclassify_unrelated_worker_exceptions(runtime, is_prefill):
    failure = ValueError("unrelated worker failure")
    failed = Mock(side_effect=failure)
    later = Mock(return_value=True)
    runtime.loaded["_run_prefill"] = failed if is_prefill else later
    runtime.loaded["_run_decode"] = later if is_prefill else failed
    with pytest.raises(ValueError, match="unrelated worker failure") as exc:
        _run_group(runtime, [(1, 4096, is_prefill, 0), (1, 8192, not is_prefill, 0)])
    assert exc.value is failure
    later.assert_not_called()
    assert not runtime.rows
