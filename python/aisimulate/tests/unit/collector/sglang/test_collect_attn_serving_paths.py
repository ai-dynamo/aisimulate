# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SGLang dense and encoder attention collectors build what sglang v0.5.23 serving builds.

Strict fakes stand in for every sglang module the collectors touch (a name missing from the
fake tree raises, as a moved or dropped sglang name would); torch is permissive. Only the
context phase is driven here: the decode branch compares tensor sizes, which a permissive
torch fake cannot answer.
"""

import ast
import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.unit
COLLECTOR_DIR = Path(__file__).resolve().parents[4] / "collector"


class _NullContextDecorator(contextlib.ContextDecorator):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _EnvInt:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


class _World:
    """Records what the collector builds, in order."""

    def __init__(self):
        self.events: list = []
        self.backends: list = []
        self.pools: list = []
        self.translators: list = []
        self.layer_calls: list = []
        self.knob = _EnvInt(384 * 1024 * 1024)
        self.buffers: dict = {}

    def surface(self) -> dict:
        world = self

        class _Backend:
            name = "?"

            def __init__(self, model_runner, *args, **kwargs):
                self.model_runner, self.args, self.kwargs = model_runner, args, kwargs
                world.backends.append(self)
                world.events.append(f"build:{self.name}")

            def init_forward_metadata(self, forward_batch):
                world.events.append(("init_forward_metadata", forward_batch.kv_loc_plan is not None))

        def backend(name):
            return type(name, (_Backend,), {"name": name})

        class _Plan:
            def bind(self, batch, reader):
                batch.kv_loc_plan = self
                world.events.append("bind")

        class _Translator:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                world.translators.append(self)

            def own_plan(self, forward_batch):
                return _Plan()

        class _Pool:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                world.pools.append(self)

            def set_kv_buffer(self, *args, **kwargs):
                pass

        class _ReqToTokenPool:
            def __init__(self, size, max_context_len, device, enable_memory_saver):
                self.req_to_token = MagicMock(name="req_to_token")

        class _ForwardBatch:
            def __init__(self, **kwargs):
                self.kv_loc_plan = None
                self.kwargs = kwargs
                for key, value in kwargs.items():
                    setattr(self, key, value)

        class _Layer:
            k_scale = None
            v_scale = None

            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def to(self, device):
                return self

            def __call__(self, q, k, v, forward_batch, **kwargs):
                world.layer_calls.append(SimpleNamespace(q=q, k=k, v=v, forward_batch=forward_batch, kwargs=kwargs))

        @contextlib.contextmanager
        def forward_context(ctx, **kwargs):
            yield

        def get_buffer(name, factory):
            if name not in world.buffers:
                world.events.append(f"allocate:{name}")
                world.buffers[name] = factory()
            return world.buffers[name]

        parallel = SimpleNamespace(override=lambda **kwargs: _NullContextDecorator())
        return {
            "sglang.srt.configs.model_config": {"AttentionArch": SimpleNamespace(MHA="MHA")},
            "sglang.srt.environ": {"envs": SimpleNamespace(SGLANG_FLASHINFER_WORKSPACE_SIZE=world.knob)},
            "sglang.srt.layers.attention.flashattention_backend": {"FlashAttentionBackend": backend("fa")},
            "sglang.srt.layers.attention.flashinfer_backend": {"FlashInferAttnBackend": backend("flashinfer")},
            "sglang.srt.layers.attention.trtllm_mha_backend": {"TRTLLMHAAttnBackend": backend("trtllm_mha")},
            "sglang.srt.layers.attention.triton_backend": {"TritonAttnBackend": backend("triton")},
            "sglang.srt.layers.radix_attention": {"RadixAttention": _Layer},
            "sglang.srt.mem_cache.kv_index_translator": {"KVIndexTranslator": _Translator},
            "sglang.srt.mem_cache.memory_pool": {"MHATokenToKVPool": _Pool, "ReqToTokenPool": _ReqToTokenPool},
            "sglang.srt.model_executor.forward_batch_info": {
                "ForwardBatch": _ForwardBatch,
                "ForwardMode": SimpleNamespace(EXTEND="EXTEND", DECODE="DECODE"),
            },
            "sglang.srt.model_executor.forward_context": {
                "ForwardContext": lambda attn_backend: SimpleNamespace(attn_backend=attn_backend),
                "forward_context": forward_context,
            },
            "sglang.srt.runtime_context": {"get_buffer": get_buffer, "get_parallel": lambda: parallel},
        }


def _install(monkeypatch, specs: dict) -> MagicMock:
    for name in tuple(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name)
    for path, attrs in specs.items():
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix not in sys.modules:
                monkeypatch.setitem(sys.modules, prefix, types.ModuleType(prefix))
        for key, value in attrs.items():
            setattr(sys.modules[path], key, value)
    torch = MagicMock(name="torch")
    # Distinct tensors per draw, tagged so a later .to() result is told apart.
    torch.randn.side_effect = lambda *args, **kwargs: MagicMock(kind="randn")
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def _import(monkeypatch, dotted: str, filename: str):
    monkeypatch.delitem(sys.modules, dotted, raising=False)
    spec = importlib.util.spec_from_file_location(dotted, COLLECTOR_DIR / "sglang" / filename)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, dotted, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def attn(monkeypatch):
    world = _World()
    torch = _install(monkeypatch, world.surface())
    module = _import(monkeypatch, "collector.sglang.collect_attn", "collect_attn.py")
    world.bench_kwargs = []
    world.perf_rows = []

    @contextlib.contextmanager
    def benchmark_with_power(**kwargs):
        world.bench_kwargs.append(kwargs)
        kwargs["kernel_func"]()
        yield {"latency_ms": 1.0, "power_stats": None}

    def log_perf(**kwargs):
        world.perf_rows.append(kwargs)
        return True

    # What SGLang serving resolves per model, (prefill, decode); None is an MLA model. A
    # model missing here resolves to the row's own backend, so the lane tests run unchecked.
    world.serving = {}
    world.resolved = []

    def resolve_serving_attention_backends(model):
        world.resolved.append(model)
        if model in world.serving:
            return world.serving[model]
        return (world.dispatch, world.dispatch)

    monkeypatch.setattr(module, "benchmark_with_power", benchmark_with_power)
    monkeypatch.setattr(module, "log_perf", log_perf)
    monkeypatch.setattr(module, "_dist_version", lambda name: "0.5.23")
    monkeypatch.setattr(module, "resolve_serving_attention_backends", resolve_serving_attention_backends)
    monkeypatch.setattr(module, "_resolve_local_model_path", lambda model: f"bundled:{model}")
    module._world = world
    return module, world, torch


def _run_context(
    module, sm, backend, *, head_dim=128, v_head_dim=None, architecture=None, fp8=False, is_context=True, input_len=1000
):
    module.get_sm_version = lambda: sm
    module._world.dispatch = backend
    module.run_attention_torch(
        2,
        input_len,
        8,
        2,
        head_dim,
        fp8,
        fp8,
        is_context,
        0,
        v_head_dim,
        -1,
        None,
        False,
        None,
        backend,
        architecture,
        perf_filename="context_attention_perf.txt",
    )


@pytest.mark.parametrize("sm", [100, 103])
def test_fa4_lane_is_servings_fa4_factory_at_page_size_128(attn, sm):
    """MiMo-V2 on SM100/103 serves on fa4 (arg_groups/model_overrides/mimo_v2.py:27-29), built
    as FlashAttentionBackend(runner, fa_impl_ver=4) (attention_registry.py:264-274) with the
    page size forced to 128 (arg_groups/overrides.py:1273-1297)."""
    module, world, _ = attn
    _run_context(module, sm, "fa4", head_dim=192, v_head_dim=128, architecture="MiMoV2FlashForCausalLM")
    (backend,) = world.backends
    assert backend.name == "fa" and backend.kwargs == {"fa_impl_ver": 4}
    assert world.pools[0].kwargs["page_size"] == 128
    assert world.pools[0].kwargs["head_dim"] == 192 and world.pools[0].kwargs["v_head_dim"] == 128
    assert world.translators[0].kwargs["page_size"] == 128
    assert world.perf_rows[0]["kernel_source"] == "fa4"
    assert "allocate:flashinfer_workspace" not in world.events


def test_trtllm_lane_preallocates_the_workspace_and_binds_the_kv_plan_before_metadata(attn):
    """TRTLLMHAAttnBackend reads its page table through the batch's KV-location plan
    (trtllm_mha_backend.py:1341-1343), which ForwardBatch.init_new binds
    (forward_batch_info.py:1139-1141)."""
    module, world, _ = attn
    _run_context(module, 100, "trtllm_mha")
    assert world.events == [
        "allocate:flashinfer_workspace",
        "build:trtllm_mha",
        "bind",
        ("init_forward_metadata", True),
    ]
    assert world.knob.get() == module._FLASHINFER_WORKSPACE_MIN_BYTES
    assert world.pools[0].kwargs["page_size"] == 64
    translator_kwargs = world.backends[0].model_runner.kv_index_translator.kwargs
    assert translator_kwargs["page_size"] == 64 and translator_kwargs["is_draft_worker"] is False


def test_extend_host_lengths_are_lists_as_serving_fills_them(attn):
    module, world, _ = attn
    _run_context(module, 90, "fa3")
    (layer_call,) = world.layer_calls
    assert layer_call.forward_batch.extend_seq_lens_cpu == [1000, 1000]
    assert layer_call.forward_batch.extend_prefix_lens_cpu == [0, 0]
    model_runner = world.backends[0].model_runner
    assert model_runner.prefill_attention_backend_str == "fa3"
    assert model_runner.decode_attention_backend_str == "fa3"
    assert world.pools[0].kwargs["page_size"] == 1
    assert "allocate:flashinfer_workspace" not in world.events


def test_fp8_compute_rows_hand_the_backend_bf16_inputs(attn):
    """Serving hands attention model-dtype Q/K/V; fa3 quantizes Q and the pool K/V inside the
    call (flashattention_backend.py:1426-1436, memory_pool.py:2926-2933), so the collector
    must not pre-cast them outside the timed call."""
    module, world, torch = attn
    _run_context(module, 90, "fa3", fp8=True)
    (layer_call,) = world.layer_calls
    assert layer_call.q.kind == "randn"
    assert layer_call.k.kind == "randn"
    assert layer_call.v.kind == "randn"
    assert world.pools[0].kwargs["dtype"] is torch.float8_e4m3fn
    assert world.perf_rows[0]["item_list"][0]["attn_dtype"] == "fp8"


def test_attention_rows_time_without_the_moe_first_point_warmup(attn):
    """The first-point warm-up stays MoE-only on all three frameworks until an attention smoke
    measures the transient on attention rows; the workspace is allocated before the backend."""
    module, world, _ = attn
    _run_context(module, 120, "flashinfer")

    assert "min_warmup_seconds" not in world.bench_kwargs[0]
    assert world.events[:2] == ["allocate:flashinfer_workspace", "build:flashinfer"]


# ------------------------------------------------- the plan's dispatch against serving
def test_a_row_runs_when_serving_resolves_the_plans_backend_for_its_architecture(attn):
    """Gemma 4 on SM90: the plan and gemma4.py _gemma4_overrides both say triton."""
    module, world, _ = attn
    world.serving["google/gemma-4-26B-A4B"] = ("triton", "triton")
    _run_context(module, 90, "triton", architecture="Gemma4ForConditionalGeneration")
    assert world.resolved == ["google/gemma-4-26B-A4B"]
    assert world.perf_rows[0]["kernel_source"] == "triton"


def test_a_row_fails_when_serving_resolves_another_backend_than_the_plan(attn):
    module, world, _ = attn
    world.serving["nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"] = ("trtllm_mha", "trtllm_mha")
    with pytest.raises(RuntimeError) as error:
        _run_context(module, 100, "flashinfer", architecture="NemotronHForCausalLM")
    message = str(error.value)
    assert "the plan dispatches NemotronHForCausalLM prefill attention on flashinfer" in message
    assert "SGLang serving resolves trtllm_mha for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" in message
    # Refused before anything is built or timed.
    assert world.backends == [] and world.pools == [] and world.perf_rows == []


def test_the_phase_the_op_times_is_the_one_checked(attn):
    """Serving may split the pair (Qwen3-VL on a large Hopper GPU decodes on flashinfer,
    arg_groups/model_overrides/qwen3_vl.py:129-130): a context row is checked against
    prefill, a generation row against decode."""
    module, world, _ = attn
    world.serving["Qwen/Qwen3-VL-2B-Instruct"] = ("fa3", "flashinfer")
    _run_context(module, 90, "fa3", architecture="Qwen3VLForConditionalGeneration")
    assert len(world.perf_rows) == 1

    with pytest.raises(RuntimeError, match="decode attention on fa3, but SGLang serving resolves flashinfer"):
        module.check_dispatch_is_servings("fa3", "Qwen3VLForConditionalGeneration", is_context_phase=False)
    module.check_dispatch_is_servings("flashinfer", "Qwen3VLForConditionalGeneration", is_context_phase=False)


def test_a_base_grid_row_is_checked_against_a_model_without_an_override(attn):
    module, world, _ = attn
    world.serving[module.DEFAULT_BACKEND_MODEL] = ("trtllm_mha", "trtllm_mha")
    _run_context(module, 100, "trtllm_mha")
    assert world.resolved == [module.DEFAULT_BACKEND_MODEL]

    with pytest.raises(RuntimeError, match="base-grid prefill attention on flashinfer.*resolves trtllm_mha"):
        _run_context(module, 100, "flashinfer")

    # The stand-in ships its config and its architecture has no table entry: it takes the default.
    config = COLLECTOR_DIR.parent / "src/aisimulate_core/model_configs/Qwen--Qwen3-32B_config.json"
    (architecture,) = json.loads(config.read_text(encoding="utf-8"))["architectures"]
    assert module.DEFAULT_BACKEND_MODEL == "Qwen/Qwen3-32B" and architecture == "Qwen3ForCausalLM"
    case_generator = sys.modules["collector.case_generator"]
    assert architecture not in case_generator.SGLANG_ATTENTION_BACKENDS


def test_serving_is_resolved_once_per_architecture_for_both_phases(attn):
    module, world, _ = attn
    world.serving["openai/gpt-oss-120b"] = ("fa3", "fa3")
    world.serving[module.DEFAULT_BACKEND_MODEL] = ("fa3", "fa3")
    for _ in range(3):
        module.check_dispatch_is_servings("fa3", "GptOssForCausalLM", is_context_phase=True)
        module.check_dispatch_is_servings("fa3", "GptOssForCausalLM", is_context_phase=False)
        module.check_dispatch_is_servings("fa3", None, is_context_phase=True)
    assert world.resolved == ["openai/gpt-oss-120b", module.DEFAULT_BACKEND_MODEL]


def test_an_mla_models_dense_rows_are_not_checked_against_its_own_attention_backend(attn):
    """Kimi-K3's own attention is MLA; its dense rows time the speculative draft."""
    module, world, _ = attn
    world.serving["moonshotai/Kimi-K3"] = None
    _run_context(module, 100, "trtllm_mha", architecture="KimiK3ForConditionalGeneration")
    assert world.resolved == ["moonshotai/Kimi-K3"] and len(world.perf_rows) == 1


def test_a_resolution_that_fails_fails_every_row_of_the_architecture_without_retrying(attn, monkeypatch):
    module, world, _ = attn
    attempts = []

    def broken(model):
        attempts.append(model)
        raise ValueError("config not recognized")

    monkeypatch.setattr(module, "resolve_serving_attention_backends", broken)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="cannot resolve the attention backend SGLang serves .*config not recog"):
            _run_context(module, 90, "fa3", architecture="Llama4ForConditionalGeneration")
    assert attempts == ["meta-llama/Llama-4-Scout-17B-16E-Instruct"]
    assert world.perf_rows == []


def test_an_architecture_is_resolved_through_its_first_model_with_a_config(attn, monkeypatch):
    module, world, _ = attn

    def bundled(model):
        if model in {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"}:
            raise FileNotFoundError(model)
        return f"bundled:{model}"

    monkeypatch.setattr(module, "_resolve_local_model_path", bundled)
    assert module._architecture_model("Qwen3_5ForConditionalGeneration") == "Qwen/Qwen3.5-27B"
    assert module._architecture_model(None) == module.DEFAULT_BACKEND_MODEL

    monkeypatch.setattr(
        module, "_resolve_local_model_path", lambda model: (_ for _ in ()).throw(FileNotFoundError(model))
    )
    with pytest.raises(RuntimeError, match="no model of Mistral3ForConditionalGeneration has a config"):
        _run_context(module, 90, "fa3", architecture="Mistral3ForConditionalGeneration")


def test_serving_is_resolved_by_servings_own_pipeline_on_the_models_config(attn, monkeypatch):
    """resolve_serving_attention_backends builds the model's ServerArgs, runs
    ServerArgs.resolve_once and reads the pair the pipeline resolved; it holds no table."""
    module, world, _ = attn
    monkeypatch.undo()  # the real function, over a fake sglang surface
    events = []

    class ServerArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            events.append(("server_args", kwargs))

        def resolve_once(self):
            events.append("resolve_once")

    override_base = {
        "attention_backends_of": lambda view: (events.append(("pair_of", view)), ("fa3", "flashinfer"))[1],
        "resolved_view": lambda server_args: ("view", server_args),
        "use_mla_backend": lambda server_args: server_args.kwargs["model_path"].endswith("Kimi-K3"),
    }
    for name, attrs in {
        "sglang.srt.arg_groups.model_override_base": override_base,
        "sglang.srt.server_args": {"ServerArgs": ServerArgs},
    }.items():
        fake = types.ModuleType(name)
        vars(fake).update(attrs)
        monkeypatch.setitem(sys.modules, name, fake)
    monkeypatch.setattr(module, "_resolve_local_model_path", lambda model: f"bundled:{model}")

    assert module.resolve_serving_attention_backends("Qwen/Qwen3-32B") == ("fa3", "flashinfer")
    server_args, resolve, pair = events
    assert server_args == (
        "server_args",
        {"model_path": "bundled:Qwen/Qwen3-32B", "load_format": "dummy", "trust_remote_code": True},
    )
    assert resolve == "resolve_once" and pair[0] == "pair_of" and pair[1][0] == "view"
    assert module.resolve_serving_attention_backends("moonshotai/Kimi-K3") is None

    source = (COLLECTOR_DIR / "sglang" / "collect_attn.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_serving_attention_backends"
    )
    literals = {n.value for n in ast.walk(resolver) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    literals.discard(ast.get_docstring(resolver, clean=False))
    assert not literals & {"fa3", "fa4", "flashinfer", "triton", "trtllm_mha"}


# ---------------------------------------------------------------- encoder
@pytest.fixture
def encoder(monkeypatch):
    calls: dict = {}

    def recorder(name):
        def kernel(*args, **kwargs):
            calls.setdefault(name, []).append((args, kwargs))

        return kernel

    _install(
        monkeypatch,
        {
            "sglang.kernels.ops.attention.flash_attention": {"flash_attn_varlen_func": recorder("fa3")},
            "sglang.kernels.ops.attention.flash_attention_v4": {"flash_attn_varlen_func": recorder("fa4")},
            "sglang.kernels.ops.attention.prefill_attention": {"context_attention_fwd": recorder("triton")},
        },
    )
    module = _import(monkeypatch, "collector.sglang.collect_attn_encoder", "collect_attn_encoder.py")
    return module, calls


def _encoder_lane(module, calls, sm):
    module.get_sm_version = lambda: sm
    run_iter, tag = module._build_kernel_runner(
        device=SimpleNamespace(type="cuda"),
        batch_size=2,
        seq_len=1024,
        num_heads=16,
        head_dim=64,
        dtype="bfloat16",
    )
    run_iter()
    ((name, recorded),) = calls.items()
    return tag, name, recorded[0][1]


@pytest.mark.parametrize("sm", [100, 103])
def test_encoder_fa4_lane_calls_fa4_directly_as_vision_flash4_attention_does(encoder, sm):
    """vision.py:62-68 hands ver=4 calls straight to flash_attention_v4, not the dispatcher."""
    tag, name, kwargs = _encoder_lane(*encoder, sm)
    assert (tag, name) == ("flash_attention_v4", "fa4")
    assert set(kwargs) == {"cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "softmax_scale"}
    assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 1024


def test_encoder_fa3_lane_matches_vision_flash3_attention(encoder):
    tag, name, kwargs = _encoder_lane(*encoder, 90)
    assert (tag, name) == ("flash_attention_v3", "fa3")
    assert kwargs["window_size"] == (-1, -1) and "ver" not in kwargs


@pytest.mark.parametrize("sm", [89, 120])
def test_encoder_triton_lane_matches_vision_triton_attention(encoder, sm):
    tag, name, kwargs = _encoder_lane(*encoder, sm)
    assert (tag, name) == ("triton", "triton")
    assert kwargs["is_causal"] is False
