# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SGLang MSA collector runs the points its op plan row lists, and refuses the
ones the device cannot run with a classified reason before allocating them."""

import importlib.util
import sys
import types

import pytest

pytestmark = pytest.mark.unit

MODEL_PATH = "MiniMaxAI/MiniMax-M3"


def _import_module(monkeypatch, sm: int = 100):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        set_device=lambda *_: None,
        get_device_name=lambda *_: "GPU",
        OutOfMemoryError=MemoryError,
    )
    torch.OutOfMemoryError = MemoryError
    torch.Tensor = object
    monkeypatch.setitem(sys.modules, "torch", torch)
    import collector.op_plans  # noqa: F401  (imported once, before the module under test)

    mod_name = "collector.sglang.collect_msa_module"
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    spec = importlib.util.spec_from_file_location(mod_name, "collector/sglang/collect_msa_module.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "get_sm_version", lambda: sm)
    return mod


def test_getters_queue_the_op_plan_rows(monkeypatch):
    from collector.op_plans import plan

    mod = _import_module(monkeypatch)
    for op, getter in (
        ("msa_context_module", mod.get_msa_context_module_test_cases),
        ("msa_generation_module", mod.get_msa_generation_module_test_cases),
    ):
        rows = getter()
        assert rows and rows == plan(op, "sglang", 100).rows
        # run_msa_module_worker's fields: [0, batch, heads, kv, compute, gemm, model, target TP, points]
        assert {len(row) for row in rows} == {9}


def _runner(monkeypatch, *, device_chunk=8192, runtime_chunk=8192, kv_capacity=None):
    mod = _import_module(monkeypatch)
    timed = {"points": []}
    chunk_module = types.ModuleType("collector.sglang.deepseekv4_sparse_modules")
    chunk_module._sglang_chunked_prefill_size = lambda: device_chunk
    monkeypatch.setitem(sys.modules, "collector.sglang.deepseekv4_sparse_modules", chunk_module)
    runner = types.SimpleNamespace(
        model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=None)]))
    )
    monkeypatch.setattr(mod, "cleanup_distributed", lambda: None)
    monkeypatch.setattr(mod, "load_model_runner", lambda **kwargs: timed.update(load=kwargs) or runner)
    monkeypatch.setattr(mod, "runtime_chunk_size", lambda _runner: runtime_chunk)
    monkeypatch.setattr(mod, "kv_pool_capacity_tokens", lambda _runner: kv_capacity)
    monkeypatch.setattr(mod, "kv_pool_page_size", lambda _runner: 128)
    monkeypatch.setattr(
        mod,
        "_run_prefill_point",
        lambda _runner, _attn, b, s, p, **_kwargs: timed["points"].append((b, s, p)) or (1.0, None),
    )
    monkeypatch.setattr(mod, "_prefill_kernel_source", lambda _runner: "triton")
    monkeypatch.setattr(mod, "_log_msa_row", lambda **_kwargs: None)

    def run(points):
        return mod.run_msa_module(8, MODEL_PATH, "bfloat16", "bfloat16", "bfloat16", True, 0, inner_manifest=points)

    return run, timed


def test_points_past_the_device_chunk_are_refused_with_their_reason(monkeypatch):
    run, timed = _runner(monkeypatch)
    with pytest.raises(
        RuntimeError, match=r"^FIXME\(kernel-limit\): 1/2 point\(s\) refused: b=2 s=8192 prefix=0"
    ) as exc:
        run([[1, 4096, 512], [2, 8192, 0]])
    assert "16384 new tokens exceed the chunked_prefill_size=8192" in str(exc.value)
    # The refused point is never allocated: the ModelRunner is sized for the other.
    assert timed["points"] == [(1, 4096, 512)]
    assert timed["load"]["max_total_tokens"] < 2 * 8192
    assert timed["load"]["chunked_prefill_size"] is None


def test_points_past_the_kv_pool_are_refused_after_the_runner_is_built(monkeypatch):
    run, timed = _runner(monkeypatch, kv_capacity=10_000)
    with pytest.raises(
        RuntimeError, match=r"^FIXME\(kernel-limit\): 1/2 point\(s\) refused: b=1 s=16 prefix=16384"
    ) as exc:
        run([[1, 16, 512], [1, 16, 16384]])
    assert "the KV pool holds 10000" in str(exc.value)
    assert timed["points"] == [(1, 16, 512)]


def test_a_task_with_every_point_refused_builds_no_runner(monkeypatch):
    run, timed = _runner(monkeypatch)
    with pytest.raises(RuntimeError, match=r"^FIXME\(kernel-limit\): 1/1 point\(s\) refused"):
        run([[2, 8192, 0]])
    assert "load" not in timed


def test_the_run_takes_its_points_from_the_plan_row(monkeypatch):
    run, _timed = _runner(monkeypatch)
    with pytest.raises(RuntimeError, match=r"MSA module context has no points"):
        run([])


@pytest.mark.parametrize("rank", ["", "[rank0]: "])
def test_subprocess_keeps_the_kernel_limit_class_of_refused_points(monkeypatch, rank):
    mod = _import_module(monkeypatch)
    refusal = "FIXME(kernel-limit): 1/2 point(s) refused: b=2 s=8192 prefix=0: does not fit"

    def child(argv, *, env, cwd, timeout):
        output = (
            f"{rank}Traceback (most recent call last):\n"
            f'{rank}  File "<string>", line 3\n'
            f"{rank}RuntimeError: {refusal}\n"
        )
        return 1, (output + "[rank0]:[W1101] destroy_process_group() was not called before exit\n").encode()

    monkeypatch.setattr(mod, "run_measurement_subprocess", child)
    with pytest.raises(RuntimeError) as exc:
        mod.run_msa_module_worker(
            0,
            2,
            8,
            "bfloat16",
            "bfloat16",
            "bfloat16",
            MODEL_PATH,
            8,
            [[2, 8192, 0]],
            perf_filename="/t/msa_context_module_perf.txt",
        )
    assert str(exc.value) == refusal
