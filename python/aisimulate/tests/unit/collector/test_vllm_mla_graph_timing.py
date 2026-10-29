# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute the module collector against fake framework state and timing."""

import ast
import math
import os
import sys
import time
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


def _module_runner(*, graph_flag=True, failure=None, reuse_frequency=1, reuse_pattern=None, point_failures=None):
    events, published, benchmark_calls = [], [], []
    point_failures = dict(point_failures or {})
    error = RuntimeError(f"{failure} failed")
    active_contexts = []
    active_slot_mapping = {}
    query_slots = object()
    attn_name = "model.layers.0.self_attn.attn"
    indexer_name = "model.layers.0.self_attn.indexer.k_cache"

    @contextmanager
    def forward_context(metadata, _config, *, slot_mapping=None):
        nonlocal active_slot_mapping
        assert set(slot_mapping or {}) == set(metadata)
        active_slot_mapping = slot_mapping or {}
        try:
            with context("forward"):
                yield
        finally:
            active_slot_mapping = {}

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
        # The native updater is a no-op if the layer mapping is absent. The
        # attention metadata alone does not supply this separate dependency.
        if active_slot_mapping.get(attn_name) is not None:
            assert active_slot_mapping[attn_name] is query_slots
            events.append(("current KV written", module.mla_attn.skip_topk))

    layer = SimpleNamespace(kv_cache_dtype="fp8_ds_mla")
    module = SimpleNamespace(forward=forward, mla_attn=SimpleNamespace(mla_attn=layer, skip_topk=False))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hidden_size=7168, index_topk_freq=reuse_frequency, index_topk_pattern=reuse_pattern
            ),
            hf_config=SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"]),
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={attn_name: SimpleNamespace(), indexer_name: SimpleNamespace()}
        ),
    )

    def create_module(**kwargs):
        if failure == "construction":
            raise error
        assert kwargs["max_query_len"] == 16
        assert kwargs["max_seq_len"] in {16, 144}
        return module, config

    @contextmanager
    def benchmark(**kwargs):
        benchmark_calls.append(dict(kwargs, skip_indexer=module.mla_attn.skip_topk))
        assert "forward" in events  # Real full dry run seeds the shared top-k buffer.
        assert kwargs["use_cuda_graph"] is True
        assert kwargs["allow_graph_fail"] is False
        if failure == "capture" or (failure == "skip_capture" and module.mla_attn.skip_topk):
            raise error
        kwargs["kernel_func"]()
        yield {"used_cuda_graph": graph_flag, "latency_ms": 0.015, "power_stats": None}
        events.append("graph teardown")

    def create_kv(**kwargs):
        events.append(("kv", kwargs["seq_len"]))
        if kwargs["seq_len"] in point_failures:
            raise point_failures[kwargs["seq_len"]]
        return (
            Tensor(),
            object(),
            SimpleNamespace(slot_mapping=query_slots),
            Tensor() if kwargs["attn_type"] == "dsa" else None,
            object() if kwargs["attn_type"] == "dsa" else None,
        )

    def cleanup():
        assert active_contexts == []
        assert module.mla_attn.skip_topk is False
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
                empty_cache=lambda: events.append("empty_cache"),
            ),
        ),
        "gc": SimpleNamespace(collect=lambda: None),
        "setup_distributed": lambda device: None,
        "enable_engine_fused_ops": lambda: None,
        "init_workspace_manager": lambda device: events.append("workspace init"),
        "_create_attention_module": create_module,
        "_install_rope_missing_key_shims": lambda module: ([], []),
        "_process_module_weights": lambda *args: None,
        "_create_kv_cache_and_metadata": create_kv,
        "set_current_vllm_config": lambda config: context("config"),
        "set_forward_context": forward_context,
        "benchmark_with_power": benchmark,
        "_mla_backend_name": lambda *args: "FLASHINFER_MLA_SPARSE",
        "log_perf": lambda **kwargs: published.append(kwargs),
        "_cleanup": cleanup,
        "vllm_version": "0.25.0",
        "traceback": SimpleNamespace(print_exc=lambda: None),
        "time": time,
    }
    _load_function(COLLECTOR / "vllm/utils.py", "with_exit_stack", namespace)
    # A prefix list expands through the plan predicate: budgets from the sweep spec,
    # ceiling from the model config (both faked here).
    namespace["get_mla_module_sweep_spec"] = lambda _backend=None: SimpleNamespace(
        context_dsa_chunk_prefill_size=16384, context_max_tokens=131072, context_dsa_max_full_sequence_length=1048576
    )
    namespace["_model_max_position_embeddings"] = lambda _model: None
    for helper in (
        "_series_seq_lens",
        "_dsa_context_point_allowed",
        "_dsa_context_budgets",
        "_series_points",
        "_is_fatal_cuda_error",
    ):
        _load_function(COLLECTOR / "vllm/collect_mla_module.py", helper, namespace)
    run = _load_function(COLLECTOR / "vllm/collect_mla_module.py", "run_mla_module", namespace)

    def execute(attn_type="dsa", phase="context", seq_len=16, prefix_len=128):
        return run(
            seq_len=seq_len,
            batch_size=2,
            num_heads=8,
            kv_cache_dtype="fp8",
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            perf_filename=f"{attn_type}_{phase}_module_perf.txt",
            prefix_len=prefix_len,
            model_path="fixture/model",
            attn_type=attn_type,
        )

    return execute, events, published, benchmark_calls, error


@pytest.mark.parametrize("attn_type", ["mla", "dsa"])
@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("reuse_frequency", [1, None])
def test_graph_publication_and_workspace_teardown_after_context_exit(attn_type, phase, reuse_frequency):
    execute, events, published, calls, _ = _module_runner(reuse_frequency=reuse_frequency)
    assert execute(attn_type, phase) == [0.015]
    assert len(calls) == len(published) == 1
    assert events[-4:] == ["graph teardown", "exit forward", "exit config", "cleanup"]
    row = published[0]["item_list"][0]
    assert row["latency"] == "0.0150"
    assert row["step"] == (128 if phase == "context" else 15)
    assert row["isl"] + row["step"] == (144 if phase == "context" else 16)
    assert published[0]["kernel_source"] == "FLASHINFER_MLA_SPARSE"
    writes = [event for event in events if isinstance(event, tuple) and event[0] == "current KV written"]
    assert writes == [("current KV written", False), ("current KV written", False)]


@pytest.mark.parametrize("phase", ["context", "generation"])
@pytest.mark.parametrize("reuse", [{"reuse_frequency": 4}, {"reuse_frequency": None, "reuse_pattern": "FFSS"}])
def test_reuse_checkpoint_publishes_native_full_and_skip_measurements(phase, reuse):
    execute, events, published, calls, _ = _module_runner(**reuse)
    assert execute(phase=phase) == [0.015]
    assert [call["skip_indexer"] for call in calls] == [False, True]
    assert [row["op_name"] for row in published] == [
        f"dsa_{phase}_module",
        f"dsa_{phase}_module_skip_indexer",
    ]
    assert published[0]["perf_filename"] == published[1]["perf_filename"]
    assert published[0]["item_list"][0] == published[1]["item_list"][0]
    writes = [event for event in events if isinstance(event, tuple) and event[0] == "current KV written"]
    assert writes == [
        ("current KV written", False),
        ("current KV written", False),
        ("current KV written", True),
    ]


def test_failed_skip_capture_is_raised_and_does_not_publish_a_skip_row():
    execute, events, published, calls, error = _module_runner(reuse_frequency=4, failure="skip_capture")
    with pytest.raises(RuntimeError) as caught:
        execute()
    assert caught.value is error
    assert [call["skip_indexer"] for call in calls] == [False, True]
    assert [row["op_name"] for row in published] == ["dsa_context_module"]
    assert events[-1] == "cleanup"


def test_ordinary_mla_does_not_publish_dsa_reuse_rows():
    execute, _, published, calls, _ = _module_runner(reuse_frequency=4)
    execute(attn_type="mla")
    assert len(calls) == 1
    assert [row["op_name"] for row in published] == ["mla_context_module"]


@pytest.mark.parametrize("reuse_frequency", [1, 4])
def test_each_case_prints_one_phase_timing_line(capsys, reuse_frequency):
    execute, _, _, calls, _ = _module_runner(reuse_frequency=reuse_frequency)
    execute()
    lines = [line for line in capsys.readouterr().out.splitlines() if "module timing" in line]
    assert len(lines) == 1
    (line,) = lines
    assert line.startswith("  [DSA module timing] context b=2, s=16, prefix=128, heads=8, gemm=bfloat16, kv=fp8: ")
    phases = ["module", "weights", "kv_metadata", "dry_run", "measure_full"]
    if reuse_frequency == 4:
        phases.append("measure_skip")
    phases += ["indexer_fill", "total"]
    assert [field.split("=")[0] for field in line.split(": ", 1)[1].split()] == phases
    assert len(calls) == len([p for p in phases if p.startswith("measure")])


def test_series_builds_the_module_once_and_measures_every_point_largest_first(capsys):
    execute, events, published, calls, _ = _module_runner(reuse_frequency=4)
    assert execute(seq_len=[8, 16, 8]) == [0.015, 0.015]
    assert events.count("workspace init") == 1
    assert [e for e in events if isinstance(e, tuple) and e[0] == "kv"] == [("kv", 16), ("kv", 8)]
    assert [(row["op_name"], row["item_list"][0]["isl"]) for row in published] == [
        ("dsa_context_module", 16),
        ("dsa_context_module_skip_indexer", 16),
        ("dsa_context_module", 8),
        ("dsa_context_module_skip_indexer", 8),
    ]
    assert len(calls) == 4 and events[-1] == "cleanup" and events.count("cleanup") == 1
    out = capsys.readouterr().out
    timing = [line for line in out.splitlines() if "module timing" in line]
    assert len(timing) == 2
    assert "module=" in timing[0] and "weights=" in timing[0], "setup phases are reported once, on the first point"
    assert "module=" not in timing[1] and "kv_metadata=" in timing[1]
    series_lines = [line for line in out.splitlines() if "[DSA module series] context b=2, prefix=128" in line]
    assert len(series_lines) == 1 and "2/2 points" in series_lines[0]


def test_series_sweeps_the_isl_x_prefix_grid_of_one_module_largest_total_context_first(capsys):
    """A DSA context case carries its prefix list: one module, every (isl, prefix) point of the
    plan grid measured on it, largest total context first (module build was 0.9-1.2 s/pt under
    the campaign's 8 workers when it was amortised over one prefix only, 2026-10-29)."""
    execute, events, published, calls, _ = _module_runner(reuse_frequency=4)
    assert execute(seq_len=[8, 16], prefix_len=[0, 128]) == [0.015] * 4
    assert events.count("workspace init") == 1
    kv_events = [e for e in events if isinstance(e, tuple) and e[0] == "kv"]
    assert kv_events == [("kv", 16), ("kv", 8), ("kv", 16), ("kv", 8)]
    rows = [
        (row["item_list"][0]["isl"], row["item_list"][0]["step"])
        for row in published
        if row["op_name"] == "dsa_context_module"
    ]
    assert rows == [(16, 128), (8, 128), (16, 0), (8, 0)]
    out = capsys.readouterr().out
    assert "[DSA module] context b=2, s=[16, 8], prefix=[128, 0]," in out
    series_lines = [line for line in out.splitlines() if "[DSA module series] context b=2, prefix=[128, 0]" in line]
    assert len(series_lines) == 1 and "4/4 points" in series_lines[0]
    assert len(calls) == 8 and events.count("cleanup") == 1


def test_series_keeps_measuring_after_a_non_fatal_point_failure_and_raises_it_afterwards():
    oom = MemoryError("CUDA out of memory (fixture)")
    execute, events, published, calls, _ = _module_runner(point_failures={16: oom})
    with pytest.raises(MemoryError) as caught:
        execute(seq_len=[16, 8])
    assert caught.value is oom
    assert [e for e in events if isinstance(e, tuple) and e[0] == "kv"] == [("kv", 16), ("kv", 8)]
    assert "empty_cache" in events, "the failed point's memory is released before the next point"
    assert [row["item_list"][0]["isl"] for row in published] == [8]
    assert len(calls) == 1 and events[-1] == "cleanup"


def test_series_summarises_several_point_failures_in_one_classified_error():
    failures = {16: MemoryError("oom at 16"), 4: RuntimeError("Missing TRTLLM-GEN kernel (decode)")}
    execute, _, published, _, _ = _module_runner(point_failures=failures)
    with pytest.raises(RuntimeError, match=r"2/3 point\(s\) of the module series failed") as caught:
        execute(seq_len=[16, 8, 4])
    assert "s=16 prefix=128: MemoryError: oom at 16" in str(caught.value)
    assert "s=4 prefix=128: RuntimeError: Missing TRTLLM-GEN kernel" in str(caught.value)
    assert [row["item_list"][0]["isl"] for row in published] == [8]


def test_series_aborts_at_once_on_a_sticky_cuda_error():
    sticky = RuntimeError("CUDA error: an illegal memory access was encountered")
    execute, events, published, _, _ = _module_runner(point_failures={16: sticky})
    with pytest.raises(RuntimeError) as caught:
        execute(seq_len=[16, 8])
    assert caught.value is sticky
    kv_points = [e for e in events if isinstance(e, tuple) and e[0] == "kv"]
    assert kv_points == [("kv", 16)], "no further point on a dead context"
    assert published == [] and events[-1] == "cleanup"


def test_single_point_failures_keep_their_original_exception_and_timing():
    """A one-point case (CLI, single-point fills) behaves exactly like before the series layout."""
    oom = MemoryError("oom")
    execute, events, published, _, _ = _module_runner(point_failures={16: oom})
    with pytest.raises(MemoryError) as caught:
        execute(seq_len=16)
    assert caught.value is oom and published == [] and events[-1] == "cleanup"


def test_explicit_full_pattern_overrides_reuse_frequency():
    execute, _, published, calls, _ = _module_runner(reuse_frequency=4, reuse_pattern="FFFF")
    execute()
    assert len(calls) == 1
    assert [row["op_name"] for row in published] == ["dsa_context_module"]


@pytest.mark.parametrize("failure", ["construction", "forward", "capture"])
def test_failure_propagates_and_workspace_is_released_without_publication(failure):
    execute, events, published, _, error = _module_runner(failure=failure)
    with pytest.raises(RuntimeError) as caught:
        execute()
    assert caught.value is error
    assert events[-1] == "cleanup"
    assert events.count("cleanup") == 1
    assert published == []


@pytest.mark.parametrize("graph_flag", [False, None])
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
    # The generic serving-parity construction resolves the model class through
    # vLLM's ModelRegistry; the stubs below cover the explicit-class path, so
    # select it (AB-only switch) for this allocation check.
    monkeypatch.setenv("AIS_DSA_LEGACY_MODULE", "1")
    namespace = {
        "math": math,
        "os": os,
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
    # The module is built before the backend (and its page) is known: the KV
    # capacity is an upper bound at the smallest page (16) and is recomputed
    # per selected page in _create_kv_cache_and_metadata.
    assert configs[0]["num_gpu_blocks"] == max(1 + math.ceil((prefix + query + 1) / 16) * batch, 8192)
