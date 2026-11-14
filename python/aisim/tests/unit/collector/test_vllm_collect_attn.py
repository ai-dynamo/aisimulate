# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import os
import typing
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
ATTN_SOURCE = ROOT / "collector/vllm/collect_attn.py"
ENCODER_ATTN_SOURCE = ROOT / "collector/vllm/collect_attn_encoder.py"
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

    def compute_num_tokens(self):
        return sum(self.query_lens)


@pytest.mark.parametrize("use_fp8_kv_cache", [False, True])
def test_generation_uses_total_runtime_length_and_production_call_order(use_fp8_kv_cache):
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

    def get_attn_backend(*args, **kwargs):
        # serving's entry point (attention.py:350-360 @v0.32.0)
        calls["selector"] = (args, kwargs)
        return Backend

    def create_config(**kwargs):
        calls["config"] = kwargs
        return config

    class Backend:
        forward_includes_kv_cache_update = False

        get_name = staticmethod(lambda: "FLASH_ATTN")
        get_required_kv_cache_layout = staticmethod(lambda: None)
        get_kv_cache_stride_order = staticmethod(lambda: (0, 1, 2, 3, 4))

    class Builder:
        def __init__(self, *_args, **_kwargs):
            pass

        def build(self, **_kwargs):
            return object()

    class Impl:
        supports_quant_query_input = False
        vllm_flash_attn_version = 3

        def __init__(self, *_args, **_kwargs):
            pass

        def do_kv_cache_update(self, layer, key, value, cache, slots):
            events.append(("update", layer, key, value, cache, slots))

        def forward(self, layer, query, key, value, cache, metadata, *, output):
            events.append(("forward", layer, query, key, value, cache, metadata, output))

    class MockAttentionLayer:
        def __init__(self, _device):
            self._q_scale = self._k_scale = self._v_scale = FakeTensor()
            self._q_scale_float = self._k_scale_float = self._v_scale_float = 1.0

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=64,
            num_gpu_blocks=8192,
            # resolved layout (772728f3): identity order = HND
            get_resolved_kv_cache_layout=lambda: SimpleNamespace(name="LBHNC", layer_view_order=(0, 1, 2, 3)),
        ),
        model_config=SimpleNamespace(get_sliding_window=lambda: None),
    )

    def create_metadata(batch_spec, _block_size, _device):
        calls["batch"] = batch_spec
        return SimpleNamespace(slot_mapping=slot_mapping)

    @contextmanager
    def benchmark(**kwargs):
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
        "get_attn_backend": get_attn_backend,
        "_cached_get_attn_backend": SimpleNamespace(cache_clear=lambda: calls.setdefault("memo_cleared", True)),
        "SERVING_MAX_NUM_SEQS": 1024,
        "current_platform": Platform(),
        "AttentionBackendEnum": {"FLASH_ATTN": object()},
        "create_standard_kv_cache_spec": lambda *_args, **kwargs: (
            calls.setdefault("spec", kwargs),
            SimpleNamespace(dtype=UINT8 if use_fp8_kv_cache else BF16),
        )[1],
        "MODELS_CONFIG_MAP": {},
        "create_common_attn_metadata": create_metadata,
        "create_kv_cache_and_block_mappings": lambda **_kwargs: (FakeTensor(), history_slot_mapping),
        "set_kv_cache_layout": lambda _layout: None,
        "get_attention_backend": lambda _backend: (Builder, Impl),
        "benchmark_with_power": benchmark,
        "log_perf": lambda **kwargs: calls.setdefault("log", kwargs),
        "vllm_version": "0.24.0",
        # 1d0492eb: the page is asked from the framework per backend (collect_attn:201
        # -> utils.framework_kv_block_size); the test pins the same 16
        "framework_kv_block_size": lambda _backend_cls: 16,
        # framework-resolved KV layout (collect_attn:248-270, 772728f3): identity order = HND
        "get_supported_kv_cache_layouts": lambda _backends: [SimpleNamespace(name="LBHNC")],
        "resolve_kv_cache_layout": lambda _cfg, _layouts, _specs: None,
        "get_kv_cache_layout": lambda: SimpleNamespace(name="LBHNC", layer_view_order=(0, 1, 2, 3)),
        # spy: the paged KV must be laid out in the framework-RESOLVED layout order
        "kv_cache_in_resolved_layout": lambda kv, order: (calls.setdefault("kv_layout_order", order), kv)[1],
    }
    _load_function(ATTN_SOURCE, "_dense_kernel_source", namespace)
    _load_function(ATTN_SOURCE, "match_serving_fa_aot_schedule", namespace)
    _load_function(ATTN_SOURCE, "select_attention_backend", namespace)
    _load_function(ATTN_SOURCE, "models_with_config_hook", namespace)
    run = _load_function(ATTN_SOURCE, "run_attention_torch", namespace)
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
            perf_filename="generation_attention_perf.txt",
        )

    assert calls["batch"] == BatchSpec(seq_lens=[65, 65], query_lens=[1, 1])
    assert calls["config"]["max_model_len"] == 65
    # A base-grid row (no models): the generic config, shaped as the layer before it is built.
    assert calls["config"]["model_name"] == str(ATTN_SOURCE.parent / "fake_hf_model")
    assert (calls["config"]["head_dim"], calls["config"]["num_heads"], calls["config"]["num_kv_heads"]) == (128, 8, 2)
    assert calls["config"]["sliding_window"] is None
    assert calls["spec"] == {"sliding_window": None}
    # serving's scheduler bound, not the case's batch of 2 (CUDA-graph capture sizes)
    assert calls["config"]["max_num_seqs"] == 1024
    assert calls["selector"] == ((128, BF16, "fp8" if use_fp8_kv_cache else "auto"), {"has_sliding_window": False})
    assert calls["memo_cleared"]
    assert calls["log"]["item_list"][0]["isl"] == 1
    assert calls["log"]["item_list"][0]["step"] == 64
    assert calls["log"]["kernel_source"] == "vllm_flash_attn_fa3"
    assert calls["kv_layout_order"] == (0, 1, 2, 3)  # cache_config.get_resolved_kv_cache_layout()
    assert [event[0] for event in events] == ["update", "update", "forward", "update", "forward"]
    assert events[0][5] is history_slot_mapping
    for update, forward in zip(events[1::2], events[2::2], strict=True):
        assert update[1:5] == (forward[1], forward[3], forward[4], forward[5])
        assert update[5] is slot_mapping


class _StopAfterShape(Exception):
    """Ends run_attention_torch once the config, the selection and the layer shape are settled."""


class _AttentionConfig:
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text


HOOKED = "HookedForCausalLM"  # an architecture the stubbed MODELS_CONFIG_MAP has a hook for
PLAIN = "PlainForCausalLM"  # one it has none for


def _model_row_world(resolutions):
    """A namespace that runs the selection half of run_attention_torch; ``resolutions`` maps a
    model's bundled directory to the (backend name, attention config) vLLM resolves for it."""
    events = []
    current = []

    def create_vllm_config(**kwargs):
        events.append(("config", kwargs))
        name, attention = resolutions.get(kwargs["model_name"], ("FLASH_ATTN", "AttentionConfig()"))
        return SimpleNamespace(
            backend=name,
            attention_config=_AttentionConfig(attention),
            model_config=SimpleNamespace(kwargs=kwargs),
        )

    @contextmanager
    def set_current_vllm_config(config):
        current.append(config)
        try:
            yield
        finally:
            current.remove(config)

    def get_attn_backend(*args, **kwargs):
        (config,) = current[-1:]
        events.append(("select", config.model_config.kwargs["model_name"], args, kwargs))
        return SimpleNamespace(get_name=lambda: config.backend)

    def set_attention_layer_of_model(model_config, **shape):
        events.append(("layer", model_config.kwargs["model_name"], shape))
        raise _StopAfterShape

    namespace = {
        "__file__": str(ATTN_SOURCE),
        "os": os,
        "torch": SimpleNamespace(bfloat16=BF16, cuda=SimpleNamespace(set_device=lambda _device: None)),
        "BatchSpec": BatchSpec,
        "set_random_seed": lambda _seed: None,
        "create_vllm_config": create_vllm_config,
        "set_current_vllm_config": set_current_vllm_config,
        "get_attn_backend": get_attn_backend,
        "_cached_get_attn_backend": SimpleNamespace(cache_clear=lambda: events.append(("memo_cleared",))),
        "_resolve_local_model_path": lambda model: f"bundled:{model}",
        "set_attention_layer_of_model": set_attention_layer_of_model,
        "SERVING_MAX_NUM_SEQS": 1024,
        "AttentionBackendEnum": {"FLASH_ATTN": object(), "TRITON_ATTN": object()},
        # vLLM's own map, stubbed: only its keys matter to the gate.
        "MODELS_CONFIG_MAP": {HOOKED: object(), "AnotherHookedForCausalLM": object()},
        # Ends a generic row where a model row ends, once the config and the selection are settled.
        "framework_kv_block_size": lambda _backend_cls: (_ for _ in ()).throw(_StopAfterShape()),
    }
    for name in (
        "select_attention_backend",
        "models_with_config_hook",
        "model_attention_selection",
        "require_one_attention_selection",
    ):
        _load_function(ATTN_SOURCE, name, namespace)
    run = _load_function(ATTN_SOURCE, "run_attention_torch", namespace)
    return run, events, ExitStack


def _run_model_row(run, stack_cls, models, *, architecture=HOOKED, **overrides):
    """Run a row whose owners are ``models``, each of ``architecture`` (or an explicit ``owners``)."""
    case = dict(
        batch_size=2,
        input_len=4096,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        use_fp8_kv_cache=False,
        is_context_phase=True,
        window_size=1024,
        owners=[[architecture, model] for model in models],
        perf_filename="context_attention_perf.txt",
    )
    case.update(overrides)
    with stack_cls() as stack:
        run(stack, **case)


def test_a_model_row_builds_its_config_from_the_rows_model_and_shapes_the_layer_after_the_hooks():
    """The model's own config reaches VllmConfig untouched, so its hook sees the real model;
    the layer under test takes the row's shape only after the backend is selected."""
    run, events, stack_cls = _model_row_world({})
    with pytest.raises(_StopAfterShape):
        _run_model_row(run, stack_cls, ["org/model-a"])

    configs = [kwargs for kind, *rest in events if kind == "config" for kwargs in rest]
    assert {kwargs["model_name"] for kwargs in configs} == {"bundled:org/model-a"}
    for kwargs in configs:
        # Nothing of the layer is written before the hooks run, and the model keeps its own length.
        assert not {"head_dim", "num_heads", "num_kv_heads", "sliding_window"} & set(kwargs)
        assert kwargs["max_model_len"] is None
        assert kwargs["dtype"] == "bfloat16" and kwargs["block_size"] is None
    measured = configs[-1]
    assert measured["max_num_seqs"] == 1024 and measured["use_fp8_kv_cache"] is False

    kinds = [event[0] for event in events]
    assert kinds[-3:] == ["memo_cleared", "select", "layer"]
    assert events[-2] == ("select", "bundled:org/model-a", (256, BF16, "auto"), {"has_sliding_window": True})
    # The layer's window is not written into the model's config: it goes to the spec and the impl.
    assert events[-1] == (
        "layer",
        "bundled:org/model-a",
        {"max_model_len": 4096, "head_dim": 256, "num_heads": 8, "num_kv_heads": 4},
    )
    # Every selection starts from a cleared memo: the worker holds many models.
    assert kinds.count("memo_cleared") == kinds.count("select")


def test_a_full_attention_layer_of_a_windowed_model_leaves_the_models_window_alone():
    """Transformers types a model's sliding_window as an int and rejects None (Gemma 4 and
    GPT-OSS global layers), so the layer's window reaches the spec and the impl as arguments."""
    run, events, stack_cls = _model_row_world({})
    with pytest.raises(_StopAfterShape):
        _run_model_row(run, stack_cls, ["org/model-a"], head_dim=512, window_size=0)
    assert "sliding_window" not in events[-1][2]
    assert events[-2][2:] == ((512, BF16, "auto"), {"has_sliding_window": False})

    tree = ast.parse(ATTN_SOURCE.read_text(), filename=str(ATTN_SOURCE))
    run_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_attention_torch")
    calls = {
        call.func.id: {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        for call in ast.walk(run_node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }
    layer_window = "layer_shape['sliding_window']"
    assert calls["create_standard_kv_cache_spec"]["sliding_window"] == layer_window
    assert calls["impl_cls"]["sliding_window"] == layer_window
    assert "get_sliding_window" not in ast.unparse(run_node)


def test_a_row_whose_owners_have_no_config_hook_runs_on_the_generic_config():
    """vLLM decides nothing for such a model, so its row is a base-grid row: same config
    arguments, same selection, no model config built."""
    run, base_events, stack_cls = _model_row_world({})
    with pytest.raises(_StopAfterShape):
        _run_model_row(run, stack_cls, [])
    run, events, stack_cls = _model_row_world({})
    with pytest.raises(_StopAfterShape):
        _run_model_row(run, stack_cls, ["org/model-a", "org/model-b"], architecture=PLAIN)

    assert events == base_events
    (config,) = [kwargs for kind, *rest in events if kind == "config" for kwargs in rest]
    assert config["model_name"] == str(ATTN_SOURCE.parent / "fake_hf_model")
    assert (config["head_dim"], config["num_heads"], config["num_kv_heads"]) == (256, 8, 4)
    assert config["sliding_window"] == 1024 and config["max_model_len"] == 4096
    assert "dtype" not in config
    assert [event[0] for event in events] == ["config", "memo_cleared", "select"]


def test_only_owners_with_a_config_hook_are_built_and_must_agree():
    different = {
        "bundled:org/hooked": ("TRITON_ATTN", "AttentionConfig(backend=TRITON_ATTN)"),
        "bundled:org/plain": ("FLASH_ATTN", "AttentionConfig()"),
    }
    run, events, stack_cls = _model_row_world(different)
    with pytest.raises(_StopAfterShape):
        _run_model_row(run, stack_cls, [], owners=[[PLAIN, "org/plain"], [HOOKED, "org/hooked"]])

    # The un-hooked owner is never built, although vLLM would resolve it differently here.
    assert {event[1]["model_name"] for event in events if event[0] == "config"} == {"bundled:org/hooked"}
    assert events[-1][:2] == ("layer", "bundled:org/hooked")


def test_the_gate_reads_vllms_own_hook_map():
    namespace = {"MODELS_CONFIG_MAP": {HOOKED: object()}}
    hooked = _load_function(ATTN_SOURCE, "models_with_config_hook", namespace)
    assert hooked([[PLAIN, "org/a"], [HOOKED, "org/b"], [HOOKED, "org/c"]]) == ("org/b", "org/c")
    assert hooked([[PLAIN, "org/a"]]) == () and hooked(()) == ()

    tree = ast.parse(ATTN_SOURCE.read_text(), filename=str(ATTN_SOURCE))
    (imported,) = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and any(alias.name == "MODELS_CONFIG_MAP" for alias in node.names)
    ]
    assert imported.module == "vllm.model_executor.models.config"
    # The collector never builds or rebinds the map.
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "MODELS_CONFIG_MAP" and isinstance(node.ctx, ast.Store)
    ]
    assert not stores
    gate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "models_with_config_hook")
    literals = {n.value for n in ast.walk(gate) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert literals == {ast.get_docstring(gate, clean=False)}


def test_models_that_share_a_shape_and_agree_are_measured_once_under_the_first():
    same = ("TRITON_ATTN", "AttentionConfig(backend=TRITON_ATTN)")
    run, events, stack_cls = _model_row_world({"bundled:org/model-a": same, "bundled:org/model-b": same})
    with pytest.raises(_StopAfterShape):
        _run_model_row(run, stack_cls, ["org/model-a", "org/model-b"], use_fp8_kv_cache=True)

    selected = [event[1] for event in events if event[0] == "select"]
    assert selected == ["bundled:org/model-a", "bundled:org/model-b", "bundled:org/model-a"]
    assert events[-1][1] == "bundled:org/model-a"
    assert all(event[2][2] == "fp8" for event in events if event[0] == "select")


def test_models_that_share_a_shape_and_resolve_different_backends_raise():
    run, events, stack_cls = _model_row_world(
        {
            "bundled:org/model-a": ("FLASH_ATTN", "AttentionConfig(flash_attn_version=4)"),
            "bundled:org/model-b": ("FLASH_ATTN", "AttentionConfig(flash_attn_version=None)"),
        }
    )
    with pytest.raises(RuntimeError) as error:
        _run_model_row(run, stack_cls, ["org/model-a", "org/model-b"])
    message = str(error.value)
    assert "different attention backends" in message
    assert "org/model-a: FLASH_ATTN under AttentionConfig(flash_attn_version=4)" in message
    assert "org/model-b: FLASH_ATTN under AttentionConfig(flash_attn_version=None)" in message
    assert not any(event[0] == "layer" for event in events)


def test_a_row_whose_model_has_no_bundled_config_fails():
    run, _events, stack_cls = _model_row_world({})

    def missing(model):
        raise FileNotFoundError(f"{model} has no bundled config")

    run.__globals__["_resolve_local_model_path"] = missing
    with pytest.raises(FileNotFoundError, match="org/model-a has no bundled config"):
        _run_model_row(run, stack_cls, ["org/model-a"])


def _attention_architectures():
    return {path.name.removesuffix("_cases.yaml") for path in (ROOT / "collector/cases/models").glob("*_cases.yaml")}


def test_the_attention_collector_names_no_model_and_holds_no_backend_table():
    """The backend comes from vLLM under the row's model config: the collector has no
    architecture literal, no model-family branch and no table of backends."""
    tree = ast.parse(ATTN_SOURCE.read_text(), filename=str(ATTN_SOURCE))
    code = [node for node in tree.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))]
    literals = {
        node.value
        for statement in code
        for node in ast.walk(statement)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    # Docstrings are prose; drop them and keep the literals code can act on.
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module))
    }
    literals -= docstrings
    architectures = _attention_architectures()
    assert architectures and not literals & architectures
    assert not [literal for literal in literals if "gemma" in literal.lower()]

    # No mapping or sequence of backend names: the only backend literals are the
    # two the kernel_source label branches on.
    backend_names = {"FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION", "TRITON_FLASH_ATTN"}
    assert literals & backend_names == {"FLASH_ATTN", "FLASHINFER"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
            elements = node.keys + node.values if isinstance(node, ast.Dict) else node.elts
            assert not [e for e in elements if isinstance(e, ast.Constant) and e.value in backend_names]

    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch")
    assert "owners" in [arg.arg for arg in run.args.args]
    # The run never compares a model or architecture name, and never names a backend enum member.
    for node in ast.walk(run):
        if isinstance(node, ast.Compare):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            assert not names & {"models", "owners", "architecture"}
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id != "AttentionBackendEnum"


def test_the_model_branch_builds_its_config_from_the_bundled_config_of_the_rows_model():
    tree = ast.parse(ATTN_SOURCE.read_text(), filename=str(ATTN_SOURCE))
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_attention_torch")
    branch = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "models" and node.orelse
    )

    def config_call(statements):
        (call,) = [
            node
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_vllm_config"
        ]
        return {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}

    of_model, generic = config_call(branch.body), config_call(branch.orelse)
    assert of_model["model_name"] == "_resolve_local_model_path(models[0])"
    assert "fake_hf_model" in generic["model_name"]
    # Only the generic config is shaped before VllmConfig runs the model hooks.
    assert of_model[None] == "serving_config" and of_model["max_model_len"] == "None"
    selection = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "model_attention_selection"
    )
    assert config_call(selection.body)["model_name"] == "_resolve_local_model_path(model)"


def test_the_layer_shape_is_written_through_one_helper_before_or_after_the_hooks():
    """create_vllm_config shapes a generic model before VllmConfig is built; a real model is
    narrowed to the layer by set_attention_layer_of_model on the built config."""
    calls = []

    class TextConfig:
        """A Transformers config that types sliding_window as an int, as Gemma 4's and GPT-OSS's do."""

        def __setattr__(self, name, value):
            if name == "sliding_window" and not isinstance(value, int):
                raise TypeError(f"Field 'sliding_window' expected int, got {type(value).__name__}")
            super().__setattr__(name, value)

    def model_config():
        text_config = TextConfig()
        text_config.sliding_window = 1024
        return SimpleNamespace(
            hf_config=SimpleNamespace(),
            hf_text_config=text_config,
            model_arch_config=SimpleNamespace(
                head_size=512, total_num_attention_heads=16, total_num_kv_heads=8, is_deepseek_mla=True
            ),
            max_model_len=262144,
        )

    namespace = {"ModelConfig": object}
    shape = _load_function(UTILS_SOURCE, "set_attention_layer_shape", namespace)
    namespace["set_attention_layer_shape"] = lambda *args, **kwargs: (calls.append(kwargs), shape(*args, **kwargs))
    of_model = _load_function(UTILS_SOURCE, "set_attention_layer_of_model", namespace)

    config = model_config()
    of_model(config, head_dim=256, num_heads=8, num_kv_heads=4, max_model_len=4096)
    arch = config.model_arch_config
    assert (arch.head_size, arch.total_num_attention_heads, arch.total_num_kv_heads) == (256, 8, 4)
    assert config.hf_text_config.sliding_window == 1024  # the model's own, never rewritten
    assert arch.is_deepseek_mla is False and config.max_model_len == 4096
    assert calls == [{"head_dim": 256, "num_heads": 8, "num_kv_heads": 4}]

    untouched = model_config()
    shape(untouched)
    assert vars(untouched.model_arch_config) == vars(model_config().model_arch_config)
    of_model_source = ast.unparse(
        next(
            node
            for node in ast.parse(UTILS_SOURCE.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "set_attention_layer_of_model"
        ).body[1:]
    )
    assert "sliding_window" not in of_model_source

    tree = ast.parse(UTILS_SOURCE.read_text(), filename=str(UTILS_SOURCE))
    create = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_vllm_config")
    order = [
        node.func.id
        for node in ast.walk(create)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"ModelConfig", "set_attention_layer_shape", "VllmConfig"}
    ]
    assert set(order) == {"ModelConfig", "set_attention_layer_shape", "VllmConfig"}
    lines = {
        node.func.id: node.lineno
        for node in ast.walk(create)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in set(order)
    }
    assert lines["ModelConfig"] < lines["set_attention_layer_shape"] < lines["VllmConfig"]


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
    """Sparse DSA metadata (FlashMLASparseMetadata) has no top-level
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
        "_MODEL_WINDOW": object(),
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

    # A layer's own window replaces the model-wide one in either direction.
    assert type(create_spec(config, use_fp8, sliding_window=None)) is FullAttentionSpec
    layer = create_spec(config, use_fp8, sliding_window=512)
    assert type(layer) is SlidingWindowSpec and layer.sliding_window == 512
