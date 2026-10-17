# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU behavioral regression checks for per-forward DSA projection inputs."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/sglang/collect_mla_module.py"


def function(name, namespace, *, within=None):
    tree = ast.parse(SOURCE.read_text())
    if within:
        tree = next(
            node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == within
        )
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    loaded = dict(namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), loaded)
    return loaded[name]


@pytest.mark.parametrize(("attn_type", "ordinary"), [("dsa", False), ("mla", True), ("mla", False)])
def test_native_projection_callback_reaches_phase_workers(attn_type, ordinary):
    calls = []

    def projection(*_args):
        return object()

    attention = SimpleNamespace(q_lora_rank=2048, kv_lora_rank=512, qk_rope_head_dim=64, prepare_qkv_latent=projection)
    runner = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=attention)])),
        server_args=SimpleNamespace(attention_backend="dsa"),
        req_to_token_pool=SimpleNamespace(clear=lambda: None),
        token_to_kv_pool_allocator=SimpleNamespace(clear=lambda: None),
    )

    def worker(**kwargs):
        calls.append(kwargs)
        return True

    run = function(
        "run_attention_torch",
        {
            "_module_model_architecture": lambda _: "GlmMoeDsaForCausalLM",
            "get_version": lambda _: "0.5.14",
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(get_device_name=lambda _: "FakeGPU"), randn=lambda *_a, **_k: None
            ),
            "_validate_mla_projection_precision": lambda *_: None,
            "_run_prefill": worker,
            "_run_decode": worker,
        },
    )
    shapes = [(1, 128, True, 128)] if ordinary else [(1, 128, True, 128), (1, 8192, False, 0)]
    assert run(
        runner,
        shapes,
        8,
        0,
        2,
        3,
        "cuda",
        None,
        attn_type=attn_type,
        model_path="glm",
        kv_cache_dtype="fp8",
        compute_dtype="bfloat16",
        gemm_type="bfloat16",
        ordinary_mla=ordinary,
    ) == len(shapes)
    for call in calls:
        assert (call["dummy_qkv_latent_func"] is projection) == (attn_type == "dsa" or ordinary)


def invocation_namespace(skip):
    projected, outputs, seen_kwargs = [], [], []
    context = SimpleNamespace(inputs=None)
    context.set_attn_inputs = lambda inputs: setattr(context, "inputs", inputs)

    def project(hidden, batch):
        value = object()
        projected.append((hidden, batch, value))
        return value

    class Inputs:
        def __init__(self, hidden, batch, callback):
            self.values = hidden, batch
            self.callback = callback
            self.computed = False

        def latent(self):
            if not self.computed:
                self.value = self.callback(*self.values)
                self.computed = True
            return self.value

    def attention(**kwargs):
        seen_kwargs.append(kwargs)
        value = context.inputs.latent()
        assert context.inputs.latent() is value
        outputs.append(value)
        return value

    batch, hidden, indices = object(), object(), object()
    context.inputs = Inputs(hidden, batch, project)
    # A previous warmup already populated the cache. New forward must replace it.
    context.inputs.latent()
    projected.clear()
    namespace = {
        "ordinary_mla": False,
        "attn_type": "dsa",
        "AttentionInputs": Inputs,
        "get_attn_tp_context": lambda: context,
        "hidden_states": hidden,
        "forward_batch": batch,
        "decode_hidden": hidden,
        "forward_batch_decode": batch,
        "dummy_qkv_latent_func": project,
        "forward_context": lambda *_: nullcontext(),
        "forward_context_type": lambda **kw: kw,
        "model_runner": SimpleNamespace(attn_backend=SimpleNamespace()),
        "use_module_piecewise_context": False,
        "use_module_eager_dsa_context": False,
        "attention_module": attention,
        "positions": None,
        "decode_positions": None,
        "zero_allocator": None,
        "_skip_kwargs": lambda: {"prev_topk_indices": indices} if skip else {},
        "use_benchmark_cuda_graph": True,
        "model_capture_mode": nullcontext,
    }
    return namespace, projected, outputs, seen_kwargs, indices


@pytest.mark.parametrize("skip", [False, True])
@pytest.mark.parametrize(("owner", "name"), [("_run_prefill", "call_attention_module"), ("_run_decode", "kernel_func")])
def test_each_dsa_forward_recomputes_projection_and_preserves_reuse(owner, name, skip):
    namespace, projected, outputs, kwargs, indices = invocation_namespace(skip)
    invoke = function(name, namespace, within=owner)
    for _ in range(3):
        invoke()
    assert len(projected) == 3
    assert len({id(output) for output in outputs}) == 3
    assert all(call.get("prev_topk_indices") is (indices if skip else None) for call in kwargs)


def test_piecewise_attention_wrapper_refreshes_projection_inputs():
    namespace, projected, outputs, _, _ = invocation_namespace(False)
    namespace.update({"torch": SimpleNamespace(Tensor=object), "LogitsProcessorOutput": SimpleNamespace})
    invoke = function("forward", namespace, within="_AttentionOnlyLanguageModel")
    module = SimpleNamespace(
        hidden_states=[1, 2], logits=[0, 0], layers=[SimpleNamespace(self_attn=namespace["attention_module"])]
    )
    for _ in range(3):
        invoke(module, SimpleNamespace(shape=(2,)), None, namespace["forward_batch"])
    assert len(projected) == 3
    assert len({id(output) for output in outputs}) == 3
