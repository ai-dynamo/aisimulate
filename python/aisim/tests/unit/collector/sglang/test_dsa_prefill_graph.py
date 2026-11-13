# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang prefill graph adapter: off at 0.5.21 (eager DSA prefill), closed for any enabled backend."""

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
SOURCE = Path(__file__).resolve().parents[4] / "collector/sglang/dsa_prefill_graph.py"


def select_bucket(monkeypatch, backend, tokens):
    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "graph_token_bucket"
    )
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)
    prefill = SimpleNamespace(backend=backend, bs=[1, 128, 256, 2048])
    exec_bag = SimpleNamespace(graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(prefill=prefill)))
    for name in ("sglang", "sglang.srt"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    runtime_context = ModuleType("sglang.srt.runtime_context")
    runtime_context.get_exec = lambda: exec_bag  # 0.5.21: the resolved config, not server_args
    monkeypatch.setitem(sys.modules, runtime_context.__name__, runtime_context)
    return ns["graph_token_bucket"](SimpleNamespace(server_args=SimpleNamespace(cuda_graph_config=None)), tokens)


@pytest.mark.parametrize("tokens", [1, 129, 2048, 4096])
def test_disabled_prefill_graph_times_eagerly(monkeypatch, tokens):
    assert select_bucket(monkeypatch, "disabled", tokens) is None


@pytest.mark.parametrize("backend", ["tc_piecewise", "breakable", "full"])
def test_enabled_prefill_graph_fails_closed_until_the_adapter_is_ported(monkeypatch, backend):
    with pytest.raises(ValueError, match="not ported to 0.5.21"):
        select_bucket(monkeypatch, backend, 1)


def test_cleanup_removes_only_new_owned_native_compile_hooks():
    node = next(
        n
        for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_release_owned_compile_hooks"
    )
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), ns)

    def hook_for(module):
        def hook():
            return module

        return hook

    owned, foreign = object(), object()
    hooks = {1: hook_for(owned), 2: hook_for(owned), 3: hook_for(foreign), 4: object(), 5: lambda: None}
    # A constructor may fail after registering its hook but before returning a
    # runner. Cleanup requires only the owned module and original registry.
    ns["_release_owned_compile_hooks"](hooks, {1}, owned)
    assert set(hooks) == {1, 3, 4, 5}


def test_native_model_contexts_are_held_across_replays_and_exception():
    """Per-model scopes enter once, stay active during each layer replay and unwind."""
    from contextlib import contextmanager

    tree = ast.parse(SOURCE.read_text())
    graph = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "dsa_prefill_graph")
    scope = next(
        n
        for n in ast.walk(graph)
        if isinstance(n, ast.With)
        and any(isinstance(child, ast.Expr) and isinstance(child.value, ast.Yield) for child in n.body)
    )
    function = ast.FunctionDef(
        name="exercise",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[scope],
        decorator_list=[],
    )
    events = []

    @contextmanager
    def context(name):
        events.append(("enter", name))
        try:
            yield
        finally:
            events.append(("exit", name))

    batch = SimpleNamespace(input_ids=object(), positions=object())

    def installed_forward(input_ids, positions, forward_batch):
        assert input_ids is batch.input_ids and positions is batch.positions and forward_batch is batch
        assert not any(kind == "exit" for kind, _ in events)
        events.append(("forward", "native trampoline"))
        return "attention output"

    ns = dict(
        torch=SimpleNamespace(no_grad=lambda: context("no_grad")),
        runner=SimpleNamespace(
            backend=SimpleNamespace(replay_session=lambda: context("replay")),
            attention_layers=[],
            quant_config=None,
            moe_layers=[],
            moe_fusions=[],
            dsa_indexers=[],
        ),
        owned=SimpleNamespace(
            attn_backend=object(), model=SimpleNamespace(model=SimpleNamespace(forward=installed_forward))
        ),
        forward_context=lambda _: context("forward"),
        ForwardContext=lambda **_: object(),
        set_tc_piecewise_forward_context=lambda *_, **__: context("tc"),
        static_batch=batch,
        bucket=4,
        hidden_states=[0],
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(SOURCE), "exec"), ns)
    iterator = ns["exercise"]()
    replay = next(iterator)
    assert replay() == replay() == "attention output"
    assert [name for kind, name in events if kind == "enter"] == ["no_grad", "replay", "forward", "tc"]
    with pytest.raises(RuntimeError, match="measurement failed"):
        iterator.throw(RuntimeError("measurement failed"))
    assert [name for kind, name in events if kind == "exit"] == ["tc", "forward", "replay", "no_grad"]
