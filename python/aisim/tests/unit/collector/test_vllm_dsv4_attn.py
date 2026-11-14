# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from collector.helper import create_test_case_id

pytestmark = pytest.mark.unit

# numpy stand-ins for the torch dtypes the collector tells apart.
_FP8, _BF16 = np.float16, np.float64


class _Array(np.ndarray):
    """The slice of a torch tensor the input-preparation helpers touch, over numpy."""

    is_meta = False
    device = "cuda:0"

    def copy_(self, other):
        self[...] = other
        return self

    def zero_(self):
        return self.copy_(0)

    def fill_(self, value):
        return self.copy_(value)

    def view(self, *shape):
        return self.reshape(*shape)

    def numel(self):
        return self.size

    def to(self, *args, dtype=None, device=None):
        dtype = next((arg for arg in args if not isinstance(arg, str)), dtype)
        return self if dtype is None else self.astype(dtype)


def _array(shape, dtype, value=0.0):
    return np.full(shape, value, dtype=dtype).view(_Array)


class _Generator:
    def __init__(self, device):
        assert device == "cpu"

    def manual_seed(self, seed):
        self.rng = np.random.default_rng(seed)
        return self


def _numpy_torch(monkeypatch, mod):
    """Back the torch calls of the input-preparation helpers with numpy."""
    fakes = {
        "Tensor": _Array,
        "Generator": _Generator,
        "no_grad": nullcontext,
        "uint8": np.uint8,
        "long": np.int64,
        "float32": np.float32,
        "bfloat16": _BF16,
        "float8_e4m3fn": _FP8,
        "float8_e5m2": np.complex64,
        "empty": lambda shape, dtype, device: _array(shape, dtype),
        "arange": lambda start, end, dtype, device: np.arange(start, end, dtype=dtype).view(_Array),
        "randn": lambda shape, generator, dtype, device: (
            generator.rng.standard_normal(tuple(shape)).astype(dtype).view(_Array)
        ),
    }
    for name, fake in fakes.items():
        monkeypatch.setattr(mod.torch, name, fake, raising=False)


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
    monkeypatch.setattr(mod, "_write_indexer_history", lambda *_args, **kwargs: calls.setdefault("history", kwargs))
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
    # The cached history is everything before the fresh query tokens.
    assert calls["history"] == {"context_len": metadata_seq_len - query_len}
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


def test_dummy_fill_leaves_the_rotary_table_alone(dsv4_module, monkeypatch):
    mod = dsv4_module
    _numpy_torch(monkeypatch, mod)
    table = np.arange(6, dtype=np.float32).reshape(3, 2)
    parameters = {
        "wq_b.weight": _array((2, 2), _FP8, 7.0),
        "wq_b.weight_scale_inv": _array((1,), np.float32, 7.0),
        "q_norm.weight": _array((2,), _BF16, 7.0),
        "packed": _array((2,), np.uint8, 7),
    }
    buffers = {"rotary_emb.cos_sin_cache": table.copy().view(_Array)}
    module = SimpleNamespace(named_parameters=parameters.items, named_buffers=buffers.items)

    mod._init_dummy_module_tensors(module)

    assert np.array_equal(buffers["rotary_emb.cos_sin_cache"], table)
    assert not parameters["wq_b.weight"].any() and not parameters["packed"].any()
    assert np.all(parameters["wq_b.weight_scale_inv"] == 1.0)
    assert np.allclose(parameters["q_norm.weight"], 0.01)


@pytest.mark.parametrize("meta", [False, True])
def test_move_keeps_the_rotary_table_when_to_empty_reallocates_buffers(dsv4_module, monkeypatch, meta):
    mod = dsv4_module
    _numpy_torch(monkeypatch, mod)
    table = np.arange(6, dtype=np.float32).reshape(3, 2)

    class Module:
        def __init__(self):
            self.rotary = SimpleNamespace(cos_sin_cache=table.copy().view(_Array))
            self.moves = []

        def parameters(self):
            return [SimpleNamespace(is_meta=meta)]

        def modules(self):
            return [self, self.rotary]

        def to(self, device):
            self.moves.append(("to", device))
            return self

        def to_empty(self, *, device):
            self.moves.append(("to_empty", device))
            self.rotary.cos_sin_cache = _array((3, 2), np.float32, np.nan)
            return self

    module = Module()

    assert mod._move_to_device(module, "cuda:0") is module
    assert module.moves == [("to_empty" if meta else "to", "cuda:0")]
    assert np.array_equal(module.rotary.cos_sin_cache, table)


def _csa_module():
    def norm():
        return SimpleNamespace(weight=_array((5,), _BF16, 0.01))

    return SimpleNamespace(
        fused_wqa_wkv=SimpleNamespace(weight=_array((6, 5), _FP8)),
        q_norm=norm(),
        kv_norm=norm(),
        indexer=SimpleNamespace(
            wq_b=SimpleNamespace(weight=_array((8, 5), _FP8)),
            compressor=SimpleNamespace(norm=norm()),
        ),
    )


def test_index_query_projections_are_seeded_and_differ_between_heads(dsv4_module, monkeypatch):
    mod = dsv4_module
    _numpy_torch(monkeypatch, mod)
    # Smaller than either projection: the draw is taken a bounded block at a time.
    monkeypatch.setattr(mod, "_SAMPLE_CHUNK_ELEMENTS", 7)
    module, again = _csa_module(), _csa_module()

    mod._give_index_query_values(module)
    mod._give_index_query_values(again)

    for weight in (module.fused_wqa_wkv.weight, module.indexer.wq_b.weight):
        assert weight.dtype == _FP8
        assert np.all(weight != 0) and np.abs(weight).max() < 0.2
        assert len({row.tobytes() for row in weight}) == len(weight)
    assert np.array_equal(module.indexer.wq_b.weight, again.indexer.wq_b.weight)
    assert np.all(module.q_norm.weight == 1.0) and np.all(module.indexer.compressor.norm.weight == 1.0)
    # Not an index input: the main key norm keeps the dummy fill.
    assert np.allclose(module.kv_norm.weight, 0.01)


def test_a_module_without_an_indexer_keeps_its_dummy_weights(dsv4_module, monkeypatch):
    mod = dsv4_module
    _numpy_torch(monkeypatch, mod)
    module = _csa_module()
    module.indexer = None

    mod._give_index_query_values(module)

    assert not module.fused_wqa_wkv.weight.any()
    assert np.allclose(module.q_norm.weight, 0.01)


def _history_fixture(monkeypatch, mod, *, use_fp4_kv=False, entry_bytes=12):
    """A CSA indexer over a 4-state page and the framework writer's calls."""
    _numpy_torch(monkeypatch, mod)
    writes = []
    ops = ModuleType("vllm._custom_ops")
    ops.indexer_k_quant_and_cache = lambda keys, k_cache, slots, quant_block_size, scale_fmt: writes.append(
        (np.array(keys), k_cache, [int(slot) for slot in slots], quant_block_size, scale_fmt)
    )
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", ops)
    k_cache = _array((16, 4, entry_bytes), np.uint8)
    indexer = SimpleNamespace(
        compress_ratio=4,
        head_dim=8,
        quant_block_size=8,
        scale_fmt="ue8m0",
        use_fp4_kv=use_fp4_kv,
        k_cache=SimpleNamespace(kv_cache=k_cache),
    )
    # Two requests of three pages each, as create_common_attn_metadata numbers them.
    common = SimpleNamespace(block_table_tensor=np.arange(3, 9, dtype=np.int32).reshape(2, 3).view(_Array))
    return SimpleNamespace(indexer=indexer), common, k_cache, writes


def test_history_keys_go_through_the_framework_writer_at_the_history_slots(dsv4_module, monkeypatch):
    mod = dsv4_module
    module, common, k_cache, writes = _history_fixture(monkeypatch, mod)
    monkeypatch.setattr(mod, "_SAMPLE_CHUNK_ELEMENTS", 16)

    # 27 cached tokens hold 6 compressed keys per request; the 7th is the forward's to write.
    mod._write_indexer_history(module, common, context_len=27)

    assert len(writes) == 1
    keys, cache, slots, quant_block_size, scale_fmt = writes[0]
    assert cache is k_cache and (quant_block_size, scale_fmt) == (8, "ue8m0")
    # block * states_per_block + state: pages 3-4 of request 0, pages 6-7 of request 1.
    assert slots == [12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29]
    assert keys.shape == (12, 8)
    assert len({row.tobytes() for row in keys}) == 12
    assert 0.5 < keys.std() < 1.5


def test_a_long_history_repeats_one_block_of_keys(dsv4_module, monkeypatch):
    mod = dsv4_module
    module, common, _k_cache, writes = _history_fixture(monkeypatch, mod)
    monkeypatch.setattr(mod, "_HISTORY_ROWS", 5)

    mod._write_indexer_history(module, common, context_len=27)

    written = [slots for _keys, _cache, slots, *_rest in writes]
    assert [len(slots) for slots in written] == [5, 5, 2]
    assert [slot for slots in written for slot in slots] == [*range(12, 18), *range(24, 30)]
    block = writes[0][0]
    assert np.array_equal(writes[1][0], block) and np.array_equal(writes[2][0], block[:2])


@pytest.mark.parametrize("context_len", [0, 3])
def test_no_history_is_written_before_the_first_compressed_key(dsv4_module, monkeypatch, context_len):
    mod = dsv4_module
    module, common, _k_cache, writes = _history_fixture(monkeypatch, mod)

    mod._write_indexer_history(module, common, context_len=context_len)
    # A module without an indexer (HCA) has no index keys at all.
    mod._write_indexer_history(SimpleNamespace(indexer=None), common, context_len=4096)

    assert writes == []


@pytest.mark.parametrize("override", [{"use_fp4_kv": True}, {"entry_bytes": 5}])
def test_history_refuses_a_cache_that_is_not_the_fp8_layout(dsv4_module, monkeypatch, override):
    mod = dsv4_module
    module, common, _k_cache, writes = _history_fixture(monkeypatch, mod, **override)

    with pytest.raises(RuntimeError, match="FP8 indexer cache only"):
        mod._write_indexer_history(module, common, context_len=27)

    assert writes == []
