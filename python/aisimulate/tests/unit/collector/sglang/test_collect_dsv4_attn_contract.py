# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SGLang DeepSeek-V4 module collector runs the points its op plan row lists, and
refuses the ones the device cannot run with a classified reason before allocating them."""

import ast
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_PATH = REPO_ROOT / "collector" / "sglang" / "collect_dsv4_attn.py"
MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"


def _import_module(monkeypatch, sm: int = 100):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        empty_cache=lambda: None,
        get_device_name=lambda *_: "GPU",
        OutOfMemoryError=MemoryError,
    )
    torch.OutOfMemoryError = MemoryError
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sys, "path", list(sys.path))
    import collector.op_plans  # noqa: F401  (imported once, before the module under test)

    mod_name = "collector.sglang.collect_dsv4_attn"
    monkeypatch.delitem(sys.modules, mod_name, raising=False)
    spec = importlib.util.spec_from_file_location(mod_name, SOURCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, mod_name, mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "get_sm_version", lambda: sm)
    return mod


def test_getters_queue_the_op_plan_rows(monkeypatch):
    from collector.op_plans import plan

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", MODEL_PATH)
    mod = _import_module(monkeypatch)
    for op, getter in (
        ("dsv4_csa_context_module", mod.get_dsv4_csa_context_test_cases),
        ("dsv4_hca_generation_module", mod.get_dsv4_hca_generation_test_cases),
    ):
        rows = getter()
        assert rows and rows == plan(op, "sglang", 100).rows
        # run_dsv4_attn_worker's fields: [0, batch, tp, kv, compute, gemm, model, kind, backend, points]
        assert {len(row) for row in rows} == {10}
        assert {(row[3], row[4]) for row in rows} == {("fp8", "bfloat16")}
        assert {row[5] for row in rows} == {"bfloat16", "fp8_block"}
        for row in rows:
            prefixes = [prefix for prefix, _seq_lens in row[9]]
            # Shortest prefix first, longest seq_len first within one.
            assert prefixes == sorted(prefixes)
            assert all(list(seq_lens) == sorted(seq_lens, reverse=True) for _prefix, seq_lens in row[9])


def _runner(monkeypatch, *, device_chunk=8192, runtime_chunk=8192, kv_capacity=None, swa_capacity=None):
    mod = _import_module(monkeypatch)
    timed = {"points": []}
    chunk_module = types.ModuleType("collector.sglang.deepseekv4_sparse_modules")
    chunk_module._sglang_chunked_prefill_size = lambda: device_chunk
    monkeypatch.setitem(sys.modules, "collector.sglang.deepseekv4_sparse_modules", chunk_module)
    attention = types.SimpleNamespace(compress_ratio=4, n_heads=64)
    runner = types.SimpleNamespace(
        model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attention)])),
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(num_attention_heads=64), window_size=128),
        req_to_token_pool=types.SimpleNamespace(clear=lambda: None),
        token_to_kv_pool_allocator=types.SimpleNamespace(clear=lambda: None),
    )
    monkeypatch.setattr(mod, "_load_model_runner", lambda *args, **kwargs: timed.update(load=kwargs) or runner)
    monkeypatch.setattr(mod, "_effective_prefill_chunk_size", lambda _runner: runtime_chunk)
    monkeypatch.setattr(mod, "_kv_pool_capacity_tokens", lambda _runner: kv_capacity)
    monkeypatch.setattr(mod, "_kv_pool_page_size", lambda _runner: 256)
    monkeypatch.setattr(mod, "_swa_kv_pool_capacity_tokens", lambda _runner: swa_capacity)
    monkeypatch.setattr(mod, "get_version", lambda _package: "0.5.21")

    def build(_runner, batch_size, seq_len, *, is_prefill, prefix_len):
        timed["points"].append((batch_size, seq_len, prefix_len))

    monkeypatch.setattr(mod, "_build_forward_batch", build)
    monkeypatch.setattr(mod, "_make_inputs", lambda *_args, **_kwargs: (None, None))
    monkeypatch.setattr(mod, "_bench_cuda_events", lambda *_args, **_kwargs: dict.fromkeys(_STATS, 1.0))
    monkeypatch.setattr(mod, "_log_result", lambda **_kwargs: None)

    def run(points, mode="context"):
        return mod.run_dsv4_mla_module(
            model_path=MODEL_PATH,
            mode=mode,
            attn_kind="csa",
            batch_size=points[0][0] if points else 1,
            seq_lens_by_prefix={prefix: [s for b, s, p in points if p == prefix] for _b, _s, prefix in points},
            gemm_type="fp8_block",
        )

    return run, timed


_STATS = ("mean_ms", "median_ms", "min_ms", "max_ms", "std_ms", "n")


def test_points_past_the_device_chunk_are_refused_with_their_reason(monkeypatch):
    run, timed = _runner(monkeypatch)
    with pytest.raises(
        RuntimeError, match=r"^FIXME\(kernel-limit\): 1/2 point\(s\) refused: b=2 s=8192 prefix=0"
    ) as exc:
        run([(2, 2048, 512), (2, 8192, 0)])
    assert "16384 new tokens exceed the chunked_prefill_size=8192" in str(exc.value)
    # The refused point is never allocated: the CSA pool is sized for the other.
    assert timed["points"] == [(2, 2048, 512)]
    assert timed["load"]["csa_context_shapes"] == [(2, 2048, 512)]


def test_points_past_the_kv_pools_are_refused_after_the_runner_is_built(monkeypatch):
    run, timed = _runner(monkeypatch, kv_capacity=10_000)
    with pytest.raises(
        RuntimeError, match=r"^FIXME\(kernel-limit\): 1/2 point\(s\) refused: b=1 s=16 prefix=16384"
    ) as exc:
        run([(1, 16, 512), (1, 16, 16384)])
    assert "the KV pool holds 10000" in str(exc.value)
    assert timed["points"] == [(1, 16, 512)]

    run, timed = _runner(monkeypatch, swa_capacity=300)
    with pytest.raises(RuntimeError, match=r"^FIXME\(kernel-limit\): 1/2 point\(s\) refused: b=1 s=512") as exc:
        run([(1, 16, 0), (1, 512, 0)])
    assert "the SWA KV pool holds 300" in str(exc.value)
    assert timed["points"] == [(1, 16, 0)]


def test_a_row_with_every_point_refused_builds_no_runner(monkeypatch):
    run, timed = _runner(monkeypatch)
    with pytest.raises(RuntimeError, match=r"^FIXME\(kernel-limit\): 1/1 point\(s\) refused"):
        run([(2, 8192, 0)])
    assert "load" not in timed


def test_decode_points_are_not_bounded_by_the_prefill_chunk(monkeypatch):
    run, timed = _runner(monkeypatch, device_chunk=1)
    run([(4, 65536, 0)], mode="generation")
    assert timed["points"] == [(4, 65536, 0)]


def test_each_point_times_a_module_with_its_history_written(monkeypatch):
    """The index scores get their magnitude once per runner; every point then has the index
    keys of its history written: the prefix in context, the whole past KV in generation."""
    for mode, points, histories in (
        ("context", [(2, 2048, 0), (2, 2048, 512)], [0, 512]),
        ("generation", [(4, 65536, 0), (4, 1024, 0)], [65536, 1024]),
    ):
        run, _timed = _runner(monkeypatch)
        mod = sys.modules["collector.sglang.collect_dsv4_attn"]
        prepared, written = [], []
        monkeypatch.setattr(mod, "_give_index_scores_magnitude", lambda runner, attention: prepared.append(attention))
        monkeypatch.setattr(mod, "_write_index_history", lambda runner, attention, batch, n: written.append(n))
        run(points, mode=mode)
        assert len(prepared) == 1 and prepared[0].compress_ratio == 4
        assert written == histories


def test_the_worker_takes_its_points_from_the_plan_row(monkeypatch):
    mod = _import_module(monkeypatch)
    with pytest.raises(RuntimeError, match=r"requires the plan row's \(prefix, sequence_lengths\) points"):
        mod.run_dsv4_attn_worker(
            0,
            1,
            1,
            "fp8",
            "bfloat16",
            "fp8_block",
            MODEL_PATH,
            "csa",
            None,
            None,
            perf_filename="/t/dsv4_csa_generation_module_perf.txt",
        )
    captured = {}
    monkeypatch.setattr(mod, "_run_subprocess", lambda **kwargs: captured.update(kwargs))
    mod.run_dsv4_attn_worker(
        0,
        4,
        2,
        "fp8",
        "bfloat16",
        "fp8_block",
        MODEL_PATH,
        "hca",
        None,
        [[0, [65536, 2]]],
        perf_filename="/t/dsv4_hca_generation_module_perf.txt",
    )
    assert captured["mode"] == "generation"
    assert captured["inner_shapes"] == ((0, (65536, 2)),)
    assert captured["kv_cache_dtype_sglang"] == "fp8_e4m3"


@pytest.mark.parametrize("rank", ["", "[rank0]: "])
def test_subprocess_keeps_the_kernel_limit_class_of_refused_points(monkeypatch, tmp_path, rank):
    mod = _import_module(monkeypatch)
    refusal = "FIXME(kernel-limit): 1/2 point(s) refused: b=2 s=8192 prefix=0: does not fit"
    log = (
        f"{rank}Traceback (most recent call last):\n"
        f'{rank}  File "<string>", line 3\n'
        f"{rank}RuntimeError: {refusal}\n"
        "[rank0]:[W1101] destroy_process_group() was not called before exit\n"
    )

    class Child:
        returncode = 1

        def __init__(self, _argv, *, env, stdout, stderr, cwd):
            stdout.write(log.encode())

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(mod.subprocess, "Popen", Child)
    with pytest.raises(RuntimeError) as exc:
        mod._run_subprocess(
            mode="context",
            attn_kind="csa",
            model_path=MODEL_PATH,
            kv_cache_dtype_sglang="fp8_e4m3",
            batch_size=2,
            output_path=str(tmp_path),
            gpu_id=0,
            inner_shapes=((0, (8192,)),),
        )
    assert str(exc.value) == refusal


def test_dsv4_persisted_num_heads_is_rank_local_with_geometry_guard():
    """Persisted ``num_heads`` follows the unified convention: rank-local
    heads derived from the config-native count (never from module attributes,
    which flip between native and rank-local across sglang versions). A module
    head count matching neither reading fails the case instead of mislabeling
    the row."""
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"), filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_local_heads"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    resolve = namespace["_resolve_local_heads"]

    # Flash native=64: rank-local persists 64/tp whether the module reports
    # native (0.5.14 behavior) or rank-local heads.
    assert resolve(native_heads=64, module_heads=64, tp_size=8) == 8
    assert resolve(native_heads=64, module_heads=8, tp_size=8) == 8
    # Pro native=128 without a module attribute still derives from config.
    assert resolve(native_heads=128, module_heads=None, tp_size=4) == 32
    # Geometry surprises fail the case (observe, don't guess).
    with pytest.raises(RuntimeError, match=r"head geometry mismatch"):
        resolve(native_heads=64, module_heads=16, tp_size=8)
    with pytest.raises(RuntimeError, match=r"head geometry mismatch"):
        resolve(native_heads=64, module_heads=63, tp_size=1)


def test_dsv4_generation_enters_sglang_model_capture_mode():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_dsv4_mla_module"
    )
    function_source = ast.get_source_segment(source, function)

    assert "model_capture_mode() if not is_prefill" in function_source


def test_tp_rows_are_refused_before_a_model_or_child_is_built(monkeypatch, tmp_path):
    """SGLang shards DeepSeek-V4 attention by get_parallel(), which this process cannot
    emulate; a TP>1 row would time the TP=1 module under a rank-local label."""
    run, timed = _runner(monkeypatch)
    mod = sys.modules["collector.sglang.collect_dsv4_attn"]
    with pytest.raises(RuntimeError, match=r"tp_size=2: single-process TP emulation"):
        mod.run_dsv4_mla_module(
            model_path=MODEL_PATH,
            mode="generation",
            attn_kind="csa",
            batch_size=1,
            seq_lens_by_prefix={0: [1024]},
            tp_size=2,
        )
    assert "load" not in timed

    def no_child(*_args, **_kwargs):
        raise AssertionError("a TP>1 row must not start a child process")

    monkeypatch.setattr(mod.subprocess, "Popen", no_child)
    with pytest.raises(RuntimeError, match=r"tp_size=4: single-process TP emulation"):
        mod._run_subprocess(
            mode="generation",
            attn_kind="hca",
            model_path=MODEL_PATH,
            kv_cache_dtype_sglang="fp8_e4m3",
            batch_size=1,
            output_path=str(tmp_path),
            gpu_id=0,
            inner_shapes=((0, (1024,)),),
            tp_size=4,
        )


def test_a_row_fails_when_the_module_runs_other_heads_than_its_label(monkeypatch):
    mod = _import_module(monkeypatch)
    mod._check_module_local_heads(types.SimpleNamespace(n_local_heads=64), 64, 1)
    mod._check_module_local_heads(types.SimpleNamespace(), 8, 8)  # no attribute: nothing to compare
    with pytest.raises(RuntimeError, match=r"runs n_local_heads=64, but the row would persist num_heads=8"):
        mod._check_module_local_heads(types.SimpleNamespace(n_local_heads=64), 8, 8)


def test_the_child_leaves_the_fp8_wo_a_gemm_to_sglang(monkeypatch, tmp_path):
    mod = _import_module(monkeypatch)
    monkeypatch.setenv("SGLANG_OPT_FP8_WO_A_GEMM", "1")
    seen = {}

    class Child:
        returncode = 0

        def __init__(self, _argv, *, env, stdout, stderr, cwd):
            seen["env"] = env

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(mod.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(mod.subprocess, "Popen", Child)
    mod._run_subprocess(
        mode="generation",
        attn_kind="csa",
        model_path=MODEL_PATH,
        kv_cache_dtype_sglang="fp8_e4m3",
        batch_size=1,
        output_path=str(tmp_path),
        gpu_id=0,
        inner_shapes=((0, (1024,)),),
        gemm_type="fp8_block",
    )
    assert "SGLANG_OPT_FP8_WO_A_GEMM" not in seen["env"]


def test_a_prefix_is_held_in_the_request_row_with_only_its_swa_tail(monkeypatch):
    """0.5.23 requests carry no prefix indices: a held row's [0, prefix_len) is the
    prefix, and only the SWA window tail holds SWA slots (the PD-decode recipe)."""
    mod = _import_module(monkeypatch)
    swa = object()
    component_module = types.ModuleType("sglang.srt.mem_cache.unified_cache.component_type")
    component_module.ComponentType = types.SimpleNamespace(SWA=swa)
    monkeypatch.setitem(sys.modules, component_module.__name__, component_module)
    monkeypatch.setattr(mod, "_alloc_prefix_indices", lambda _runner, n, prefix: [f"slots{i}" for i in range(n)])
    writes = []

    def alloc(reqs):
        for row, req in enumerate(reqs, start=1):
            req.kv.req_pool_idx = row
        return [req.kv.req_pool_idx for req in reqs]

    runner = types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(
            alloc=alloc, write=lambda index, value: writes.append((index, value)), available_size=lambda: 8
        ),
        token_to_kv_pool_allocator=types.SimpleNamespace(page_size=256, alloc_extend_swa_tail=lambda *_: None),
        model_config=types.SimpleNamespace(window_size=128),
    )

    def req():
        kv = types.SimpleNamespace(evicted={})
        kv.set_evicted_seqlen = lambda component, length: kv.evicted.__setitem__(component, length)
        return types.SimpleNamespace(kv=kv)

    reqs = [req(), req()]
    mod._hold_prefix_rows(runner, reqs, 1000)
    assert writes == [((1, slice(0, 1000)), "slots0"), ((2, slice(0, 1000)), "slots1")]
    assert {(r.kv.kv_allocated_len, r.kv.kv_committed_len) for r in reqs} == {(1000, 1000)}
    # The page-aligned window start alloc_prefix_indices allocates SWA from.
    assert [r.kv.evicted for r in reqs] == [{swa: 768}, {swa: 768}]

    fresh = [req()]
    mod._hold_prefix_rows(runner, fresh, 0)
    assert not hasattr(fresh[0].kv, "req_pool_idx")


def test_sources_follow_the_v0_5_23_serving_calls():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    # ChunkCache is gone (sglang #43201); Req has no prefix_indices or set_extend_range.
    for removed in ("mem_cache.chunk_cache", "req.prefix_indices", "set_extend_range", "_tp_load_model_patch"):
        assert removed not in source
    # The pool configurator factory takes the KV-cache configurator, as serving passes it.
    assert "create_memory_pool_configurator(kv_cache_configurator)" in source
    sparse = (REPO_ROOT / "collector" / "sglang" / "deepseekv4_sparse_modules.py").read_text(encoding="utf-8")
    # Serving plans top-k v2 with the kernel's heuristic threshold.
    assert "plan_topk_v2(seq_lens)" in sparse
    assert "plan_topk_v2(seq_lens, 0)" not in sparse
