# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import sys
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest
from collector.registry_types import PerfFile
from collector.sglang_rubin import collect_mla_module as module
from collector.sglang_rubin import glm5_dsa_sparse_modules as sparse

pytestmark = pytest.mark.unit


def test_pilot_cases_keep_bf16_attention_and_explicit_tp_identity(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", module.MODEL_PATH)
    context = module.get_dsa_context_module_test_cases()
    generation = module.get_dsa_generation_module_test_cases()

    assert context and generation
    for case in context + generation:
        assert len(case) == 11
        assert case[2] * case[9] == 64
        assert case[3:6] == ["fp8", "bfloat16", "bfloat16"]
        assert case[6:9] == [module.MODEL_PATH, "dsa", None]
        assert case[10] is None  # The runtime resolves the DSA sub-backends.
    assert any(case[2] == 16 and case[9] == 4 for case in context)
    assert module.get_dsa_context_module_skip_indexer_test_cases() == context
    assert module.get_dsa_generation_module_skip_indexer_test_cases() == generation


def test_an_unrelated_target_is_rejected_instead_of_substituted(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "zai-org/GLM-5.2-FP8")
    with pytest.raises(ValueError, match="require nvidia/GLM-5.2-NVFP4"):
        module.get_dsa_context_module_test_cases()
    with pytest.raises(ValueError, match="require nvidia/GLM-5.2-NVFP4"):
        sparse.get_glm5_mqa_test_cases()


def test_checkpoint_retains_native_shared_layer_offset_and_quant_exclusions():
    directory = module._local_model_config(module.MODEL_PATH, 16)
    try:
        config = json.loads(Path(directory, "config.json").read_text())
        quant = json.loads(Path(directory, "hf_quant_config.json").read_text())
        assert config["num_hidden_layers"] == 4
        assert config["num_attention_heads"] == 16
        assert config["index_skip_topk_offset"] == 3
        assert config["index_topk_freq"] == 4
        assert config["indexer_types"] == ["full", "full", "full", "shared"]
        assert quant["quantization"]["quant_algo"] == "NVFP4"
        assert "model.layers.3.self_attn*" in quant["quantization"]["exclude_modules"]
    finally:
        shutil.rmtree(directory)


@pytest.mark.parametrize("resolved_prefill_graph_disabled", [True, False])
def test_model_runner_initializes_bf16_backend_before_loading_projections(
    monkeypatch, tmp_path, resolved_prefill_graph_disabled
):
    events = []
    backend = SimpleNamespace(value="auto")
    args = SimpleNamespace(
        bf16_gemm_backend="auto",
        mem_fraction_static=0.8,
        is_startup_weight_load_overlap=False,
        disable_prefill_cuda_graph=resolved_prefill_graph_disabled,
    )

    def server_args(**kwargs):
        assert kwargs["disable_prefill_cuda_graph"] is True
        return args

    def initialize_bf16(received):
        assert received is args
        events.append("bf16")
        backend.value = "cutedsl"

    def model_runner(**kwargs):
        assert kwargs["server_args"] is args
        assert backend.value == "cutedsl"
        events.append("model_load")
        return SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[module.ARCHITECTURE])),
            model=SimpleNamespace(model=SimpleNamespace(layers=[])),
            alloc_memory_pool=lambda: events.append("memory"),
            init_attention_backends=lambda: events.append("attention"),
        )

    modules = {
        "torch": SimpleNamespace(device=lambda _: SimpleNamespace(index=0)),
        "sglang.srt.configs.model_config": SimpleNamespace(
            ModelConfig=SimpleNamespace(from_server_args=lambda received: received)
        ),
        "sglang.srt.distributed.parallel_state_wrapper": SimpleNamespace(
            ParallelState=SimpleNamespace(trivial=lambda **_: object())
        ),
        "sglang.srt.entrypoints.engine": SimpleNamespace(_set_envs_and_config=lambda _: events.append("envs")),
        "sglang.srt.layers.moe": SimpleNamespace(initialize_moe_config=lambda _: events.append("moe")),
        "sglang.srt.layers.quantization.fp8_utils": SimpleNamespace(
            initialize_fp8_gemm_config=lambda _: events.append("fp8")
        ),
        "sglang.srt.layers.quantization.fp4_utils": SimpleNamespace(
            initialize_fp4_gemm_config=lambda _: events.append("fp4")
        ),
        "sglang.srt.layers.quantization.unquant": SimpleNamespace(initialize_bf16_gemm_config=initialize_bf16),
        "sglang.srt.model_executor.model_runner": SimpleNamespace(ModelRunner=model_runner),
        "sglang.srt.server_args": SimpleNamespace(ServerArgs=server_args),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    directory = tmp_path / "model_config"
    directory.mkdir()
    monkeypatch.setattr(module, "_local_model_config", lambda *_: str(directory))

    if resolved_prefill_graph_disabled:
        module.load_model_runner(module.MODEL_PATH, 16, target_tp_size=4)
        assert events == ["envs", "moe", "fp8", "fp4", "bf16", "model_load", "memory", "attention"]
    else:
        with pytest.raises(RuntimeError, match="--disable-prefill-cuda-graph"):
            module.load_model_runner(module.MODEL_PATH, 16, target_tp_size=4)
        assert not events

    assert not directory.exists()


def test_dsa_runtime_requires_the_exact_installed_wheel(monkeypatch):
    from collector.sglang_rubin import collect_gemm, runtime

    package = {"version": collect_gemm.SGLANG_DISTRIBUTION_VERSION}
    monkeypatch.setattr(runtime, "collect_inventory", lambda: {"observed": {"package_versions": {"sglang": package}}})
    monkeypatch.setattr(runtime, "validate_runtime", lambda _: [])
    module._validate_runtime()
    package["version"] = "0.5.18"
    with pytest.raises(RuntimeError, match="Expected SGLang distribution"):
        module._validate_runtime()


def test_native_index_sharing_uses_layers_two_and_three_without_mutation():
    attentions = [
        SimpleNamespace(skip_topk=False, next_skip_topk=False, indexer=object()),
        SimpleNamespace(skip_topk=False, next_skip_topk=False, indexer=object()),
        SimpleNamespace(skip_topk=False, next_skip_topk=True, indexer=object()),
        SimpleNamespace(skip_topk=True, next_skip_topk=True, indexer=None),
    ]
    runner = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention) for attention in attentions])
        )
    )
    before = [vars(attention).copy() for attention in attentions]
    assert module._module_pair(runner, False) == (attentions[2], attentions[2])
    assert module._module_pair(runner, True) == (attentions[2], attentions[3])
    assert [vars(attention) for attention in attentions] == before
    attentions[3].indexer = object()
    with pytest.raises(RuntimeError, match="native index-sharing"):
        module._module_pair(runner, True)


def test_shape_filters_apply_equally_to_full_and_sparse_inputs(monkeypatch):
    monkeypatch.setenv("COLLECTOR_MODEL_PATH", module.MODEL_PATH)
    monkeypatch.setenv("AIC_DSA_CONTEXT_PREFIX_LENS", "4096")
    monkeypatch.setenv("AIC_DSA_CONTEXT_SEQ_LENS", "128")
    monkeypatch.setenv("AIC_DSA_CONTEXT_BATCH_SIZES", "1")
    monkeypatch.setenv("AIC_DSA_GENERATION_PREFIX_LENS", "4096")
    monkeypatch.setenv("AIC_DSA_GENERATION_BATCH_SIZES", "1")
    assert module._dsa_context_derived_shapes(module.MODEL_PATH) == [(4096, 128, 1)]
    assert module._dsa_generation_derived_shapes(module.MODEL_PATH) == [(4096, 1, 1)]
    for kernel, getter in (
        ("mqa", sparse.get_glm5_mqa_test_cases),
        ("topk", sparse.get_glm5_topk_test_cases),
        ("dsa_attn", sparse.get_glm5_dsa_attn_test_cases),
    ):
        assert getter() == [[module.MODEL_PATH, kernel, 1]]


class _Tensor:
    def __init__(self, size):
        self.size = size

    def to(self, device, *, non_blocking):
        assert device == "cuda" and non_blocking
        return self


class _Req:
    # Deliberately excludes retired fill_len / set_extend_input_len fields.
    __slots__ = (
        "extend_range",
        "full_untruncated_fill_ids",
        "logprob_start_len",
        "origin_input_ids",
        "origin_input_text",
        "output_ids",
        "prefix_indices",
        "rid",
        "sampling_params",
    )

    def __init__(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.output_ids = []

    def set_extend_range(self, start, end):
        self.extend_range = (start, end)


@pytest.mark.parametrize("is_prefill", [True, False])
def test_batch_preparation_uses_current_scheduler_and_forward_contract(monkeypatch, is_prefill):
    observed = {}

    class Batch:
        @classmethod
        def init_new(cls, **kwargs):
            batch = cls()
            batch.reqs = kwargs["reqs"]
            batch.device = "cuda"
            observed["batch"] = batch
            return batch

        def prepare_for_extend(self):
            self.input_ids = None
            self.prefill_input_ids_cpu = _Tensor(sum(end - start for start, end in [r.extend_range for r in self.reqs]))

        def prepare_for_decode(self):
            assert self.input_ids.size == len(self.reqs)
            assert all(req.output_ids == [0] for req in self.reqs)
            observed["decode"] = True

    class ForwardBatch:
        @staticmethod
        def init_new(batch, runner, *, return_hidden_states_before_norm):
            assert return_hidden_states_before_norm is False
            assert batch.prefill_input_ids_cpu is None
            return SimpleNamespace(input_ids=batch.input_ids)

    fake_torch = SimpleNamespace(int64="int64", zeros=lambda size, **kwargs: _Tensor(size))
    modules = {
        "torch": fake_torch,
        "sglang.srt.managers.schedule_batch": SimpleNamespace(Req=_Req, ScheduleBatch=Batch),
        "sglang.srt.mem_cache.cache_init_params": SimpleNamespace(CacheInitParams=lambda **kwargs: kwargs),
        "sglang.srt.mem_cache.chunk_cache": SimpleNamespace(ChunkCache=lambda params: params),
        "sglang.srt.model_executor.forward_batch_info": SimpleNamespace(ForwardBatch=ForwardBatch),
        "sglang.srt.sampling.sampling_params": SimpleNamespace(SamplingParams=lambda **kwargs: kwargs),
        "sglang.srt.speculative.spec_info": SimpleNamespace(SpeculativeAlgorithm=SimpleNamespace(NONE="none")),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(
        module, "alloc_prefix_indices", lambda runner, bs, prefix: [list(range(prefix)) for _ in range(bs)]
    )
    pool = SimpleNamespace(clear=lambda: None)
    runner = SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=pool,
        page_size=64,
        model_config=object(),
        device="cuda",
        attn_backend=SimpleNamespace(init_forward_metadata=lambda forward: observed.update(forward=forward)),
    )
    result = module._prepare_batch(
        runner, prefix=4096, isl=128 if is_prefill else 1, batch_size=2, is_prefill=is_prefill
    )
    batch = observed["batch"]
    assert observed["forward"] is result
    assert result.input_ids.size == (256 if is_prefill else 2)
    assert batch.reqs[0].extend_range == ((4096, 4224) if is_prefill else (4095, 4096))
    assert len(batch.reqs[0].prefix_indices) == (4096 if is_prefill else 4095)
    assert bool(observed.get("decode")) is not is_prefill


def test_worker_preserves_device_mapping_and_safe_json_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    unusual = tmp_path / 'space and "quoted"' / "dsa_context_module_skip_indexer_perf.txt"
    module.run_mla_module_worker(
        0,
        1,
        16,
        "fp8",
        "bfloat16",
        "bfloat16",
        module.MODEL_PATH,
        "dsa",
        target_tp_size=4,
        perf_filename=str(unusual),
        device="cuda:1",
    )
    command, kwargs = calls[0]
    assert command[1:3] == ["-m", "collector.sglang_rubin.collect_mla_module"]
    payload = json.loads(command[command.index("--payload") + 1])
    assert payload["output_path"] == str(unusual.parent)
    assert payload["skip_indexer"] is True
    assert payload["target_tp_size"] == 4
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "7"
    assert kwargs["check"] is True


def test_module_variants_share_canonical_files_with_distinct_operation_labels(monkeypatch, tmp_path):
    from collector.helper import finalize_perf_outputs

    variants = [
        (True, False, PerfFile.DSA_CONTEXT_MODULE, "dsa_context_module"),
        (False, False, PerfFile.DSA_GENERATION_MODULE, "dsa_generation_module"),
        (True, True, PerfFile.DSA_CONTEXT_MODULE, "dsa_context_module_skip_indexer"),
        (False, True, PerfFile.DSA_GENERATION_MODULE, "dsa_generation_module_skip_indexer"),
    ]
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda _: None, get_device_name=lambda _: "VR200"),
            inference_mode=nullcontext,
        ),
    )
    monkeypatch.setattr(module, "_validate_runtime", lambda: None)
    monkeypatch.setattr(module, "get_version", lambda _: module.__compat__.removeprefix("sglang=="))
    monkeypatch.setattr(module, "_dsa_context_derived_shapes", lambda _: [(4096, 128, 1)])
    monkeypatch.setattr(module, "_dsa_generation_derived_shapes", lambda _: [(4096, 1, 1)])
    monkeypatch.setattr(module, "load_model_runner", lambda *_, **__: object())
    monkeypatch.setattr(module, "_prepare_batch", lambda *_, **__: object())

    def measure_module(*_, skip_indexer, **__):
        return {"latency_ms": 0.125 if skip_indexer else 0.25, "power_stats": None}, "test_dsa_kernel"

    monkeypatch.setattr(module, "_measure_module", measure_module)

    for is_prefill, skip_indexer, _, _ in variants:
        module.run_mla_module(
            attn_type="dsa",
            head_num=16,
            model_path=module.MODEL_PATH,
            kv_cache_dtype="fp8",
            compute_dtype="bfloat16",
            gemm_type="bfloat16",
            is_prefill=is_prefill,
            gpu_id=0,
            output_path=str(tmp_path),
            target_tp_size=4,
            skip_indexer=skip_indexer,
        )

    assert {path.name for path in tmp_path.glob("*_perf.txt")} == {filename for _, _, filename, _ in variants}
    assert set(finalize_perf_outputs(tmp_path)) == {
        (tmp_path / filename).with_suffix(".parquet") for _, _, filename, _ in variants
    }
    for filename in (PerfFile.DSA_CONTEXT_MODULE, PerfFile.DSA_GENERATION_MODULE):
        rows = pq.read_table((tmp_path / filename).with_suffix(".parquet")).to_pylist()
        assert sorted((row["op_name"], row["isl"], row["latency"]) for row in rows) == sorted(
            (op_name, 128 if is_prefill else 1, 0.125 if skip_indexer else 0.25)
            for is_prefill, skip_indexer, expected_file, op_name in variants
            if expected_file == filename
        )


def test_sparse_rows_record_tp4_actual_fp8_kernel_and_fail_on_write(monkeypatch, tmp_path):
    from collector import helper

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(get_device_name=lambda _: "VR200")))
    rows = []
    monkeypatch.setattr(helper, "log_perf", lambda **kwargs: rows.append(kwargs) or True)
    arguments = dict(
        kernel="dsa_attn",
        bs=1,
        isl=128,
        prefix=4096,
        num_heads=16,
        latency=0.123,
        model_path=module.MODEL_PATH,
        source="flashinfer.trtllm_batch_decode_with_kv_cache_mla:trtllm-gen",
        device="cuda:0",
        mode=None,
    )
    sparse._write_row(str(tmp_path / "glm5_dsa_attn_module_perf.txt"), **arguments)
    row = rows[0]
    assert row["item_list"][0]["num_heads"] == 16
    assert row["item_list"][0]["tp_size"] == 4
    assert row["item_list"][0]["mla_dtype"] == "fp8_e4m3"
    assert "trtllm-gen" in row["kernel_source"]
    monkeypatch.setattr(helper, "log_perf", lambda **kwargs: False)
    with pytest.raises(RuntimeError, match="Failed to persist"):
        sparse._write_row(str(tmp_path / "glm5_dsa_attn_module_perf.txt"), **arguments)


def test_trtllm_prefill_and_decode_keep_different_serving_length_semantics():
    metadata = SimpleNamespace(dsa_cache_seqlens_int32=[2048, 2048], cache_seqlens_int32=[8192, 16384])
    assert sparse._trtllm_sequence_lengths(metadata, is_prefill=True) is metadata.dsa_cache_seqlens_int32
    assert sparse._trtllm_sequence_lengths(metadata, is_prefill=False) is metadata.cache_seqlens_int32


@pytest.mark.parametrize("measure", [sparse._bench_mqa, sparse._bench_topk])
@pytest.mark.parametrize("dense", [True, False])
def test_short_prefill_raw_kernels_raise_on_native_dense_or_k_only_dispatch(monkeypatch, measure, dense):
    prefix, isl = 0, 128
    forward = SimpleNamespace(seq_lens_cpu=[prefix + isl])
    observed = []

    def should_skip(received):
        assert received is forward
        observed.append(received.seq_lens_cpu)
        return True

    indexer = SimpleNamespace(_should_skip_logits_computation=should_skip, dsa_enable_prefill_cp=False)
    runner = SimpleNamespace(attn_backend=SimpleNamespace(use_mha=dense))
    monkeypatch.setattr(sparse, "_indexer", lambda _: indexer)
    monkeypatch.setattr(sparse, "_bench", lambda *_: pytest.fail("Unselected kernels must not be measured"))
    with pytest.raises(RuntimeError, match="framework-selected " + ("dense MHA" if dense else "K-only indexer")):
        measure(runner, forward, None, is_prefill=True, device="cuda:0")
    assert observed == [[128]]


def test_raw_kernel_gate_obeys_native_selector_instead_of_a_length_rule(monkeypatch):
    forward = SimpleNamespace(seq_lens_cpu=[128])
    indexer = SimpleNamespace(_should_skip_logits_computation=lambda _: False, dsa_enable_prefill_cp=False)
    runner = SimpleNamespace(attn_backend=SimpleNamespace(use_mha=False))
    monkeypatch.setattr(sparse, "_indexer", lambda _: indexer)
    assert sparse._require_full_indexer_path(runner, forward) is indexer


@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["eager", "graph-construction"])
@pytest.mark.parametrize("skip_indexer", [False, True], ids=["full-indexer", "shared-indexer"])
def test_module_call_projects_latent_on_every_invocation(monkeypatch, use_cuda_graph, skip_indexer):
    hidden = SimpleNamespace(shape=(8, 6144), dtype="bfloat16", device="cuda:0")
    forward = SimpleNamespace(positions=object())
    runner = SimpleNamespace(attn_backend=object())
    previous_topk = object() if skip_indexer else None
    projected, inputs, capture_entries = [], [], []
    current = SimpleNamespace(inputs=None, capturing=False)
    output = object()

    def prepare_qkv_latent(received_hidden, received_forward):
        assert received_hidden is hidden
        assert received_forward is forward
        assert current.capturing is use_cuda_graph
        projected.append(object())
        return projected[-1]

    def attention_inputs(received_hidden, received_forward, callback):
        # Model the framework's per-AttentionInputs memoization without
        # replacing the collector closure under test.
        value = SimpleNamespace(
            fetch_qkv_latent=lru_cache(maxsize=1)(lambda: callback(received_hidden, received_forward))
        )
        inputs.append(value)
        return value

    def attention(**kwargs):
        assert kwargs["positions"] is forward.positions
        assert kwargs["hidden_states"] is hidden
        assert kwargs["forward_batch"] is forward
        assert kwargs["prev_topk_indices"] is previous_topk
        assert kwargs["zero_allocator"]._pointer == 0
        kwargs["zero_allocator"]._pointer = 16
        # Multiple consumers within one attention call must share its latent.
        assert current.inputs.fetch_qkv_latent() is current.inputs.fetch_qkv_latent()
        return output

    attention.prepare_qkv_latent = prepare_qkv_latent
    attention.q_lora_rank, attention.kv_lora_rank, attention.qk_rope_head_dim = 2048, 512, 64

    @contextmanager
    def capture_mode():
        assert not current.capturing
        current.capturing = True
        capture_entries.append(True)
        try:
            yield
        finally:
            current.capturing = False

    modules = {
        "torch": SimpleNamespace(float32="float32", randn=lambda *_, **__: object()),
        "sglang.srt.layers.communicator": SimpleNamespace(
            AttentionInputs=attention_inputs,
            get_attn_tp_context=lambda: SimpleNamespace(
                set_attn_inputs=lambda value: setattr(current, "inputs", value)
            ),
        ),
        "sglang.srt.model_executor.forward_context": SimpleNamespace(
            ForwardContext=lambda **kwargs: kwargs,
            forward_context=nullcontext,
        ),
        "sglang.srt.model_executor.runner": SimpleNamespace(model_capture_mode=capture_mode),
        "sglang.srt.utils": SimpleNamespace(BumpAllocator=lambda **_: SimpleNamespace(_pointer=0)),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)

    call = module._make_module_call(
        runner, forward, attention, hidden, previous_topk=previous_topk, use_cuda_graph=use_cuda_graph
    )
    assert not projected
    # The helper invokes the closure for warmup and again while constructing
    # the graph (or measuring eagerly). Replays execute captured GPU work.
    for count in range(1, 5):
        assert call() is output
        assert len(projected) == count
        assert len(inputs) == count
    assert len(capture_entries) == (4 if use_cuda_graph else 0)


@pytest.mark.parametrize("skip_indexer", [True, False])
@pytest.mark.parametrize("is_prefill,use_mha", [(True, True), (True, False), (False, False)])
def test_module_measures_native_return_contract_outside_producer_timing(monkeypatch, skip_indexer, is_prefill, use_mha):
    from collector import helper

    calls = []
    producer, attention = object(), object()
    output = _Tensor(128)
    topk = None if use_mha else _Tensor(2048)
    forward = SimpleNamespace(input_ids=SimpleNamespace(numel=lambda: 128), batch_size=1)
    runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=6144)),
        attn_backend=SimpleNamespace(use_mha=use_mha, dsa_prefill_impl="trtllm", dsa_decode_impl="trtllm"),
        server_args=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(backend="cuda", bs=[1]))),
    )

    def make_call(_runner, _forward, selected, _hidden, **kwargs):
        assert _forward is forward
        assert kwargs["use_cuda_graph"] is (not is_prefill)
        assert kwargs.get("previous_topk") is (topk if selected is attention and skip_indexer else None)

        def call():
            calls.append(selected)
            return output if use_mha else (output, topk)

        return call

    @contextmanager
    def benchmark(**kwargs):
        assert kwargs["use_cuda_graph"] is (not is_prefill)
        assert calls == ([producer] if skip_indexer else []) + [attention] * 8
        kwargs["kernel_func"]()
        assert calls[-1] is attention
        yield {"latency_ms": 0.1}

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            Tensor=_Tensor,
            bfloat16="bfloat16",
            randn=lambda *_, **__: object(),
            cuda=SimpleNamespace(synchronize=lambda _: None),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.runner.flashinfer_autotune",
        SimpleNamespace(should_run_flashinfer_autotune=lambda _: False, flashinfer_autotune_context=nullcontext),
    )
    monkeypatch.setattr(module, "_module_pair", lambda *_: (producer, attention))
    monkeypatch.setattr(module, "_make_module_call", make_call)
    monkeypatch.setattr(helper, "benchmark_with_power", benchmark)

    results, source = module._measure_module(
        runner, forward, skip_indexer=skip_indexer, is_prefill=is_prefill, device="cuda:0"
    )

    assert results["latency_ms"] == 0.1
    density = "dense" if use_mha else "sparse"
    method = "eager" if is_prefill else "cuda_graph"
    assert source.endswith(f"_{density}_{method}")
    assert calls == ([producer] if skip_indexer else []) + [attention] * 9


@pytest.mark.parametrize(
    "producer_result,error",
    [
        pytest.param(_Tensor(128), "native .*contract", id="sparse-tensor"),
        pytest.param(None, "native .*contract", id="sparse-none"),
        pytest.param((), "native .*contract", id="sparse-empty-tuple"),
        pytest.param((_Tensor(128),), "native .*contract", id="sparse-short-tuple"),
        pytest.param((_Tensor(128), _Tensor(2048), None), "native .*contract", id="sparse-long-tuple"),
        pytest.param((_Tensor(128), None), "no topk indices for sparse attention", id="sparse-missing-topk"),
    ],
)
def test_sparse_index_sharing_rejects_missing_or_malformed_producer_result(monkeypatch, producer_result, error):
    from collector import helper

    producer, attention = object(), object()
    calls = []
    runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(hidden_size=6144)),
        attn_backend=SimpleNamespace(use_mha=False),
    )
    forward = SimpleNamespace(input_ids=SimpleNamespace(numel=lambda: 128), batch_size=1)

    def make_call(_runner, _forward, selected, _hidden, **kwargs):
        assert selected is producer
        assert kwargs.get("previous_topk") is None

        def call():
            calls.append(selected)
            return producer_result

        return call

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(Tensor=_Tensor, bfloat16="bfloat16", randn=lambda *_, **__: object())
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.model_executor.runner.flashinfer_autotune",
        SimpleNamespace(should_run_flashinfer_autotune=lambda _: False, flashinfer_autotune_context=nullcontext),
    )
    monkeypatch.setattr(module, "_module_pair", lambda *_: (producer, attention))
    monkeypatch.setattr(module, "_make_module_call", make_call)
    monkeypatch.setattr(helper, "benchmark_with_power", lambda **_: pytest.fail("Invalid sparse input reached timing"))

    with pytest.raises(RuntimeError, match=error):
        module._measure_module(runner, forward, skip_indexer=True, is_prefill=True, device="cuda:0")
    assert calls == [producer]
