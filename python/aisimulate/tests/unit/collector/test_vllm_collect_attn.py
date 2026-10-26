# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import os
import sys
import typing
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ATTN_SOURCE = ROOT / "collector/vllm/collect_attn_v1.py"
ENCODER_ATTN_SOURCE = ROOT / "collector/vllm/collect_attn_encoder_v1.py"
MLA_ATTN_SOURCE = ROOT / "collector/vllm/collect_mla_module.py"
UTILS_SOURCE = ROOT / "collector/vllm/utils.py"
BF16, FP8, UINT8 = object(), object(), object()


def _load_function(path: Path, name: str, namespace: dict):
    """Compile the real function body without importing unavailable vLLM."""
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _return_self(self, *_args, **_kwargs):
    return self


def _real_torch_or_skip():
    torch = pytest.importorskip("torch")
    if not isinstance(getattr(torch, "Tensor", None), type):
        pytest.skip("requires a real torch module")
    return torch


class FakeTensor:
    ndim = 5
    __getitem__ = contiguous = permute = to = transpose = _return_self


class Spec:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)


class FullAttentionSpec(Spec):
    pass


class SlidingWindowSpec(Spec):
    pass


@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]


@pytest.mark.parametrize("use_fp8_kv_cache", [False, True])
@pytest.mark.parametrize("native_runtime", [False, True])
@pytest.mark.parametrize("window_size", [0, 128])
@pytest.mark.parametrize("backend_name,fa_version", [("FLASH_ATTN", 3), ("FLASH_ATTN", 4), ("FLASHINFER", None)])
def test_generation_uses_total_runtime_length_and_production_call_order(
    use_fp8_kv_cache, native_runtime, window_size, backend_name, fa_version, monkeypatch
):
    calls, events = {}, []
    slot_mapping = object()
    history_slot_mapping = SimpleNamespace(numel=lambda: 128)

    torch = SimpleNamespace(
        bfloat16=BF16,
        cuda=SimpleNamespace(set_device=lambda _device: None, get_device_name=lambda _device: "fake-gpu"),
        tensor=lambda *_args, **_kwargs: FakeTensor(),
        randn=lambda *_args, **_kwargs: FakeTensor(),
        cat=lambda *_args, **_kwargs: FakeTensor(),
        empty_like=lambda *_args, **_kwargs: FakeTensor(),
    )

    class Platform:
        def fp8_dtype(self):
            return FP8

        def get_attn_backend_cls(self, _backend, selector, *, num_heads):
            calls["selector"] = selector
            calls["selector_num_heads"] = num_heads
            return "fake.backend"

    class Backend:
        forward_includes_kv_cache_update = False

        get_name = staticmethod(lambda: backend_name)
        get_required_kv_cache_layout = staticmethod(lambda: None)
        get_kv_cache_stride_order = staticmethod(lambda: (0, 1, 2, 3, 4))

    flashinfer = SimpleNamespace(get_per_layer_parameters=None)
    if backend_name == "FLASHINFER":
        # The production runner imports and patches this upstream population hook.
        for name in ("vllm", "vllm.v1", "vllm.v1.attention", "vllm.v1.attention.backends"):
            monkeypatch.setitem(sys.modules, name, SimpleNamespace(__path__=[]))
        monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.flashinfer", flashinfer)
        monkeypatch.setitem(
            sys.modules, "vllm.v1.attention.backends.utils", SimpleNamespace(PerLayerParameters=SimpleNamespace)
        )

    class FlashInferTrtllmAPIDecode:
        kernel = SimpleNamespace(value="trtllm-gen")

    class Builder:
        def __init__(self, _spec, layer_names, actual_config, _device):
            if backend_name == "FLASHINFER":
                calls["per_layer"] = flashinfer.get_per_layer_parameters(actual_config, layer_names, Impl)

        def build(self, **kwargs):
            calls["build"] = kwargs
            if backend_name == "FLASHINFER":
                return SimpleNamespace(num_prefills=0, prefill=None, decode=FlashInferTrtllmAPIDecode())
            return object()

    class Impl:
        supports_quant_query_input = backend_name == "FLASHINFER"
        vllm_flash_attn_version = fa_version

        def __init__(self, *_args, **kwargs):
            calls["impl"] = kwargs

        def do_kv_cache_update(self, layer, key, value, cache, slots):
            events.append(("update", layer, key, value, cache, slots))

        def forward(self, layer, query, key, value, cache, metadata, *, output):
            events.append(("forward", layer, query, key, value, cache, metadata, output))

    class MockAttentionLayer:
        def __init__(self, _device):
            self._q_scale = self._k_scale = self._v_scale = FakeTensor()
            self._q_scale_float = self._k_scale_float = self._v_scale_float = 1.0

    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, num_gpu_blocks=8192),
        model_config=SimpleNamespace(get_sliding_window=lambda: window_size or None),
    )

    def create_config(**kwargs):
        calls["config"] = kwargs
        return config

    def create_metadata(batch_spec, _block_size, _device):
        calls["batch"] = batch_spec
        return SimpleNamespace(slot_mapping=slot_mapping)

    @contextmanager
    def benchmark(**kwargs):
        calls["timing"] = kwargs
        kwargs["kernel_func"]()
        kwargs["kernel_func"]()
        yield {"latency_ms": 1.25, "power_stats": None}

    namespace = {
        "__file__": str(ATTN_SOURCE),
        "os": os,
        "torch": torch,
        "BatchSpec": BatchSpec,
        "MockAttentionLayer": MockAttentionLayer,
        "set_random_seed": lambda _seed: None,
        "create_vllm_config": create_config,
        "set_current_vllm_config": lambda _config: nullcontext(),
        "AttentionSelectorConfig": lambda **kwargs: SimpleNamespace(**kwargs),
        "current_platform": Platform(),
        "resolve_obj_by_qualname": lambda _path: Backend,
        "AttentionBackendEnum": {backend_name: object()},
        "create_standard_kv_cache_spec": lambda *_args: SimpleNamespace(dtype=UINT8 if use_fp8_kv_cache else BF16),
        "create_common_attn_metadata": create_metadata,
        "create_kv_cache_and_block_mappings": lambda **_kwargs: (FakeTensor(), history_slot_mapping),
        "backend_utils": SimpleNamespace(set_kv_cache_layout=lambda _layout: None),
        "get_attention_backend": lambda _backend: (Builder, Impl),
        "benchmark_with_power": benchmark,
        "log_perf": lambda **kwargs: calls.setdefault("log", kwargs),
        "vllm_version": "0.24.0",
    }
    _load_function(ATTN_SOURCE, "_dense_kernel_source", namespace)
    run = _load_function(ATTN_SOURCE, "run_attention_torch", namespace)
    runtime = None
    if native_runtime:

        def prepare_cache(batch, _config, _backend, _spec, _blocks, _device, _is_context):
            calls["batch"] = batch
            return SimpleNamespace(slot_mapping=slot_mapping), FakeTensor(), history_slot_mapping

        runtime = SimpleNamespace(
            validate_backend=lambda name: calls.setdefault("validated_backend", name), prepare_cache=prepare_cache
        )
        # The current route must never invoke the removed legacy metadata/layout APIs.
        namespace["create_common_attn_metadata"] = None
        namespace["create_kv_cache_and_block_mappings"] = None
        namespace["backend_utils"] = None
    with ExitStack() as stack:
        run(
            stack,
            batch_size=2,
            input_len=64,
            num_heads=8,
            num_kv_heads=2,
            head_dim=128,
            use_fp8_kv_cache=use_fp8_kv_cache,
            is_context_phase=False,
            window_size=window_size,
            perf_filename="generation_attention_perf.txt",
            _runtime=runtime,
        )

    assert calls["batch"] == BatchSpec(seq_lens=[65, 65], query_lens=[1, 1])
    assert calls["config"]["max_model_len"] == 65
    assert calls["config"]["sliding_window"] == (window_size or None)
    assert calls["impl"]["sliding_window"] == (window_size or None)
    assert calls["selector"].kv_cache_dtype == ("fp8" if use_fp8_kv_cache else "auto")
    assert calls["log"]["item_list"][0]["isl"] == 1
    assert calls["log"]["item_list"][0]["step"] == 64
    expected_source = (
        "vllm_flashinfer_flashinfertrtllmapidecode"
        if backend_name == "FLASHINFER"
        else f"vllm_flash_attn_fa{fa_version}"
    )
    assert calls["log"]["kernel_source"] == expected_source
    if backend_name == "FLASHINFER":
        params = calls["per_layer"]["placeholder"]
        assert params.window_left == (window_size - 1 if window_size else -1)
        assert params.logits_soft_cap == 0.0 and params.sm_scale == pytest.approx(128**-0.5)
        assert flashinfer.get_per_layer_parameters is None  # no leaked monkeypatch
    assert calls["build"]["common_prefix_len"] == 0
    assert (calls["timing"]["num_warmups"], calls["timing"]["num_runs"], calls["timing"]["repeat_n"]) == (3, 6, 1)
    assert [event[0] for event in events] == ["update", "update", "forward", "update", "forward"]
    assert events[0][5] is history_slot_mapping
    for update, forward in zip(events[1::2], events[2::2], strict=True):
        assert update[1:5] == (forward[1], forward[3], forward[4], forward[5])
        assert update[5] is slot_mapping


@pytest.mark.parametrize("randomize_blocks", [False, True])
def test_dense_kv_cache_history_slots_follow_block_table(randomize_blocks):
    torch = _real_torch_or_skip()

    namespace = {
        "torch": torch,
        "CommonAttentionMetadata": object,
        "cdiv": lambda value, divisor: (value + divisor - 1) // divisor,
    }
    create_cache = _load_function(UTILS_SOURCE, "create_kv_cache_and_block_mappings", namespace)
    metadata = SimpleNamespace(
        num_reqs=2,
        seq_lens_cpu=torch.tensor([70, 5], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([69, 4], dtype=torch.int32),
        block_table_tensor=torch.full((2, 2), -1, dtype=torch.int32),
        slot_mapping=torch.full((2,), -1, dtype=torch.long),
    )

    cache, history_slots = create_cache(
        block_size=64,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        num_blocks=8,
        common_attn_metadata=metadata,
        randomize_blocks=randomize_blocks,
    )

    expected_history_slots = []
    for request, context_len in enumerate([69, 4]):
        for offset in range(context_len):
            block_id = int(metadata.block_table_tensor[request, offset // 64])
            expected_history_slots.append(block_id * 64 + offset % 64)

    assert cache.shape == (2, 8, 64, 1, 8)
    assert sorted(metadata.block_table_tensor[0, :2].tolist() + metadata.block_table_tensor[1, :1].tolist()) == [
        1,
        2,
        3,
    ]
    assert metadata.slot_mapping.tolist() == [
        int(metadata.block_table_tensor[0, 1]) * 64 + 5,
        int(metadata.block_table_tensor[1, 0]) * 64 + 4,
    ]
    assert history_slots.tolist() == expected_history_slots


def test_mla_fp8_history_uses_framework_cache_writer():
    torch = _real_torch_or_skip()
    calls = []

    class Ops:
        @staticmethod
        def concat_and_cache_mla(kv_c, k_pe, kv_cache, slots, *, kv_cache_dtype, scale):
            calls.append((kv_c, k_pe, slots.clone(), kv_cache_dtype, scale.clone()))
            kv_cache.view(-1, kv_cache.shape[-1])[slots] = 17

    namespace = {
        "torch": torch,
        "ops": Ops,
        "CommonAttentionMetadata": object,
        "Optional": typing.Optional,
        "Union": typing.Union,
        "cdiv": lambda value, divisor: (value + divisor - 1) // divisor,
    }
    create_cache = _load_function(UTILS_SOURCE, "create_and_prepopulate_kv_cache_mla", namespace)
    metadata = SimpleNamespace(
        seq_lens_cpu=torch.tensor([3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
        num_computed_tokens_cpu=torch.tensor([2], dtype=torch.int32),
        block_table_tensor=torch.full((1, 1), -1, dtype=torch.int32),
        slot_mapping=torch.full((1,), -1, dtype=torch.long),
    )
    kv_c = torch.full((2, 4), 1.5, dtype=torch.bfloat16)
    k_pe = torch.full((2, 1, 2), -1.5, dtype=torch.bfloat16)

    cache = create_cache(
        kv_c_contexts=[kv_c],
        k_pe_contexts=[k_pe],
        block_size=64,
        head_size=6,
        dtype=torch.uint8,
        device=torch.device("cpu"),
        num_blocks=4,
        common_attn_metadata=metadata,
        randomize_blocks=False,
        kv_cache_dtype="fp8",
        scale=2.0,
    )

    assert len(calls) == 1
    assert calls[0][0] is kv_c
    assert torch.equal(calls[0][1], k_pe.squeeze(1))
    assert calls[0][2].tolist() == [64, 65]
    assert calls[0][3] == "fp8"
    assert calls[0][4].item() == 2.0
    assert torch.equal(cache[1, :2], torch.full((2, 6), 17, dtype=torch.uint8))


def test_encoder_attention_rejects_missing_flash_attention_version():
    flash_attn = object()
    torch = SimpleNamespace(
        bfloat16=object(),
        int32=object(),
        cuda=SimpleNamespace(set_device=lambda _device: None),
        randn=lambda *_args, **_kwargs: object(),
        arange=lambda *_args, **_kwargs: object(),
    )
    namespace = {
        "torch": torch,
        "get_vit_attn_backend": lambda **_kwargs: flash_attn,
        "AttentionBackendEnum": SimpleNamespace(FLASH_ATTN=flash_attn),
        "get_flash_attn_version": lambda **_kwargs: None,
    }
    run = _load_function(ENCODER_ATTN_SOURCE, "run_encoder_attention_torch", namespace)

    with pytest.raises(RuntimeError, match="without a concrete FA version"):
        run(1, 64, 4, 128, perf_filename="encoder_attention_perf.txt")


def test_encoder_031_route_keeps_native_version_wrapper_and_timing():
    calls = {}
    flash_attn = object()
    cu_seqlens = object()

    @contextmanager
    def benchmark(**kwargs):
        calls["timing"] = kwargs
        kwargs["kernel_func"]()
        yield {"latency_ms": 0.01, "power_stats": None}

    namespace = {
        "torch": SimpleNamespace(
            bfloat16=BF16,
            int32=object(),
            cuda=SimpleNamespace(set_device=lambda _: None, get_device_name=lambda _: "B200"),
            randn=lambda *_args, **_kwargs: FakeTensor(),
            arange=lambda *_args, **_kwargs: cu_seqlens,
        ),
        "get_vit_attn_backend": lambda **kwargs: calls.setdefault("selector", kwargs) and flash_attn,
        "AttentionBackendEnum": SimpleNamespace(FLASH_ATTN=flash_attn),
        "get_flash_attn_version": lambda **kwargs: calls.setdefault("fa_selector", kwargs) and 4,
        "vit_flash_attn_wrapper": lambda *args, **kwargs: calls.setdefault("wrapper", kwargs),
        "benchmark_with_power": benchmark,
        "log_perf": lambda **kwargs: calls.setdefault("row", kwargs),
        "vllm_version": "0.31.0",
    }
    implementation = _load_function(ENCODER_ATTN_SOURCE, "run_encoder_attention_torch", namespace)
    wrapper = _load_function(
        ROOT / "collector/vllm/collect_attn_encoder_v2.py",
        "run_encoder_attention_torch",
        {"collect_attn_encoder_v1": SimpleNamespace(run_encoder_attention_torch=implementation)},
    )
    wrapper(2, 64, 4, 128, perf_filename="encoder_attention_perf.txt")
    assert calls["selector"] == {"head_size": 128, "dtype": BF16}
    assert calls["fa_selector"] == {"head_size": 128}
    assert calls["wrapper"]["fa_version"] == 4
    assert calls["wrapper"]["cu_seqlens"] is cu_seqlens
    assert calls["wrapper"]["batch_size"] == 2
    assert (calls["timing"]["num_warmups"], calls["timing"]["num_runs"], calls["timing"]["repeat_n"]) == (3, 6, 1)
    assert calls["row"]["kernel_source"] == "vllm_vit_flash_attn_fa4"


class _FIPrefill:
    pass


class _FIDecode:
    pass


@pytest.mark.parametrize(
    ("num_prefills", "prefill", "decode", "expected"),
    [
        (2, _FIPrefill(), None, "vllm_flashinfer__fiprefill"),
        # s=1 "context" batches are classified entirely as decodes by
        # FlashInfer (reorder_batch_threshold=1): prefill is None and the
        # decode portion is the ground truth.
        (0, None, _FIDecode(), "vllm_flashinfer__fidecode"),
    ],
)
def test_dense_kernel_source_flashinfer_reads_populated_portion(num_prefills, prefill, decode, expected):
    source = _load_function(ATTN_SOURCE, "_dense_kernel_source", {})
    metadata = SimpleNamespace(num_prefills=num_prefills, prefill=prefill, decode=decode)

    assert source("FLASHINFER", impl=None, attn_metadata=metadata) == expected


def test_dense_kernel_source_flashinfer_raises_when_both_portions_missing():
    source = _load_function(ATTN_SOURCE, "_dense_kernel_source", {})
    metadata = SimpleNamespace(num_prefills=0, prefill=None, decode=None)

    with pytest.raises(RuntimeError, match="neither a prefill nor a decode portion"):
        source("FLASHINFER", impl=None, attn_metadata=metadata)


@pytest.mark.parametrize(
    ("attn_type", "is_context", "num_prefills", "expected"),
    [
        ("mla", True, 2, "PREFILL_BACKEND"),
        ("mla", True, 0, "DECODE_BACKEND"),  # s=1 context batch: all decodes
        ("mla", False, 0, "DECODE_BACKEND"),
        ("dsa", True, 2, "DECODE_BACKEND"),  # DSA always runs attn_backend
    ],
)
def test_mla_backend_name_records_actually_invoked_backend(attn_type, is_context, num_prefills, expected):
    backend_name = _load_function(MLA_ATTN_SOURCE, "_mla_backend_name", {})
    mla_layer = SimpleNamespace(
        attn_backend=SimpleNamespace(get_name=lambda: "DECODE_BACKEND"),
        prefill_backend=SimpleNamespace(get_name=lambda: "PREFILL_BACKEND"),
    )
    metadata = SimpleNamespace(num_prefills=num_prefills)

    assert backend_name(mla_layer, attn_type, is_context, metadata) == expected


def test_mla_backend_name_never_reads_num_prefills_off_the_mla_context_branch():
    """Sparse DSA metadata (FlashMLASparseMetadata @0.24.0) has no top-level
    ``num_prefills``; the helper must not read it for DSA or generation rows."""
    backend_name = _load_function(MLA_ATTN_SOURCE, "_mla_backend_name", {})
    mla_layer = SimpleNamespace(
        attn_backend=SimpleNamespace(get_name=lambda: "DECODE_BACKEND"),
        prefill_backend=SimpleNamespace(get_name=lambda: "PREFILL_BACKEND"),
    )
    sparse_metadata = SimpleNamespace()  # no num_prefills attribute

    assert backend_name(mla_layer, "dsa", True, sparse_metadata) == "DECODE_BACKEND"
    assert backend_name(mla_layer, "mla", False, sparse_metadata) == "DECODE_BACKEND"


@pytest.mark.parametrize(
    ("kv_cache_dtype", "compute_dtype", "attn_type", "message"),
    [
        ("bfloat16", "bfloat16", "unknown", "attention type"),
        ("float16", "bfloat16", "mla", "KV-cache dtype"),
        ("bfloat16", "float16", "mla", "query compute"),
        # FP8 prefill query compute is only honored with an FP8 KV cache.
        ("bfloat16", "fp8", "mla", "requires an FP8 KV cache"),
    ],
)
def test_mla_module_rejects_unknown_runtime_inputs(kv_cache_dtype, compute_dtype, attn_type, message):
    run = _load_function(MLA_ATTN_SOURCE, "run_mla_module", {})

    with pytest.raises(ValueError, match=message):
        run(
            None,
            64,
            1,
            8,
            kv_cache_dtype,
            compute_dtype,
            "fp8_block",
            "perf.txt",
            model_path="model",
            attn_type=attn_type,
        )


@pytest.mark.parametrize(
    ("use_fp8", "window", "expected_cls", "expected_dtype", "cache_dtype"),
    [
        (False, None, FullAttentionSpec, BF16, "auto"),
        (True, None, FullAttentionSpec, UINT8, "fp8"),
        (True, 128, SlidingWindowSpec, UINT8, "fp8"),
    ],
)
def test_standard_kv_cache_spec_matches_vllm_contract(use_fp8, window, expected_cls, expected_dtype, cache_dtype):
    namespace = {
        "VllmConfig": object,
        "FullAttentionSpec": FullAttentionSpec,
        "SlidingWindowSpec": SlidingWindowSpec,
        "get_kv_quant_mode": lambda dtype: ("quant", dtype),
        "kv_cache_dtype_str_to_dtype": lambda dtype, _model: BF16 if dtype == "auto" else UINT8,
    }
    create_spec = _load_function(UTILS_SOURCE, "create_standard_kv_cache_spec", namespace)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, cache_dtype=cache_dtype),
        parallel_config=object(),
        model_config=SimpleNamespace(
            dtype=BF16,
            get_num_kv_heads=lambda _parallel: 2,
            get_head_size=lambda: 128,
            get_sliding_window=lambda: window,
        ),
    )

    spec = create_spec(config, use_fp8)

    assert type(spec) is expected_cls
    assert (spec.block_size, spec.num_kv_heads, spec.head_size, spec.head_size_v) == (64, 2, 128, 128)
    assert spec.dtype is expected_dtype
    assert spec.kv_quant_mode == ("quant", cache_dtype)
    if window is not None:
        assert spec.sliding_window == window
