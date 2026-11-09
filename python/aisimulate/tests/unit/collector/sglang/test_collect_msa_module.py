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
    monkeypatch.setattr(mod, "serving_memory_defaults", lambda tp_size: (device_chunk, 256))
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


def _stub_modules(monkeypatch, modules: dict[str, dict]):
    """Install strict sglang fakes: each module carries only the names given."""
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)


class _Req:
    """A v0.5.23 Req surface: no set_extend_range, no prefix_indices (#42824, #42825)."""

    def __init__(self, rid, origin_input_text, origin_input_ids, sampling_params):
        self.rid = rid
        self.origin_input_ids = origin_input_ids
        self.prefix_len = 0
        self.extend_end = None


def test_requests_follow_the_v0523_admission_contract(monkeypatch):
    mod = _import_module(monkeypatch)
    _stub_modules(
        monkeypatch,
        {
            "sglang.srt.managers.schedule_batch": {"Req": _Req},
            "sglang.srt.sampling.sampling_params": {"SamplingParams": lambda **kwargs: kwargs},
        },
    )

    class _Ids:
        def __init__(self, n):
            self.n = n

        def numpy(self):
            return self

        def tobytes(self):
            return bytes(8 * self.n)

    monkeypatch.setattr(mod.torch, "randint", lambda low, high, size, dtype=None: _Ids(size[0]), raising=False)
    monkeypatch.setattr(mod.torch, "int64", "int64", raising=False)
    reqs = mod._make_reqs(3, 1024)
    assert [len(req.origin_input_ids) for req in reqs] == [1024] * 3
    assert [(req.prefix_len, req.extend_end) for req in reqs] == [(0, 1024)] * 3
    assert all(list(req.full_untruncated_fill_ids) == list(req.origin_input_ids) for req in reqs)


def test_a_cached_prefix_is_an_earlier_chunk_round_of_the_same_requests(monkeypatch):
    """With the radix cache off a prefix lives in the request's row from its previous
    chunk; the second round extends only the new tokens (mem_cache/allocation.py)."""
    import contextlib

    mod = _import_module(monkeypatch)
    rounds = []

    def checkpoint(req, tree_cache):  # mem_cache/common.py checkpoint_kv_cache
        req.prefix_len = req.extend_end

    _stub_modules(monkeypatch, {"sglang.srt.mem_cache.common": {"checkpoint_kv_cache": checkpoint}})
    trees = []
    monkeypatch.setattr(mod, "_radix_disabled_tree_cache", lambda _runner: trees.append(object()) or trees[-1])
    monkeypatch.setattr(mod, "temporarily_chunked_alloc_extend", lambda _runner, n: contextlib.nullcontext())

    class _Batch:
        device = "cuda"

        def __init__(self, reqs, tree_cache):
            self.reqs, self.tree_cache = reqs, tree_cache
            self.prefill_input_ids_cpu = types.SimpleNamespace(to=lambda device, non_blocking: ("ids", device))

        def prepare_for_extend(self):
            rounds.append((self.tree_cache, [(req.prefix_len, req.extend_end) for req in self.reqs]))

    monkeypatch.setattr(mod, "_make_schedule_batch", lambda _runner, reqs, tree_cache: _Batch(reqs, tree_cache))
    reqs = [types.SimpleNamespace(prefix_len=0, extend_end=1536) for _ in range(2)]
    batch = mod._extend_batch(object(), reqs, 512)

    assert [spans for _tree, spans in rounds] == [[(0, 512)] * 2, [(512, 1536)] * 2]
    assert rounds[0][0] is rounds[1][0] is trees[0]  # one tree cache across both rounds
    # The prefill ids reach the forward stream (benchmark/one_batch.py).
    assert batch.input_ids == ("ids", "cuda") and batch.prefill_input_ids_cpu is None

    rounds.clear()
    mod._extend_batch(object(), [types.SimpleNamespace(prefix_len=0, extend_end=64)], 0)
    assert [spans for _tree, spans in rounds] == [[(0, 64)]]


def _sparse(*, width=1048580, block=128, topk=16):
    return types.SimpleNamespace(
        req_to_token=types.SimpleNamespace(shape=(9, width)), block_size_k=block, topk_blocks=topk
    )


@pytest.mark.parametrize(
    "ceiling,bound,variant",
    [
        # Graph-captured decode is bounded by the 1M context: 8192 score blocks.
        (16384, 1048576, "topk_radix"),  # v0.5.23 CUDA ceiling
        (4096, 1048576, "topk_split"),  # v0.5.21 CUDA ceiling
        (4096, 4096 * 128, "topk_radix"),
    ],
)
def test_decode_topk_label_follows_the_kernel_gate(monkeypatch, ceiling, bound, variant):
    mod = _import_module(monkeypatch)
    _stub_modules(
        monkeypatch,
        {
            "sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx": {
                "_JIT_TOPK_MAX_NUM_BLOCKS": ceiling
            },
            "sglang.srt.environ": {
                "envs": types.SimpleNamespace(
                    SGLANG_OPT_USE_MINIMAX_DECODE_TOPK_RADIX=types.SimpleNamespace(get=lambda: True)
                )
            },
        },
    )
    assert mod._decode_topk_variant(_sparse(), bound) == variant
    # The score tensor never spans past req_to_token's width.
    assert mod._decode_topk_variant(_sparse(width=4096 * 128), 10**9) == "topk_radix"


@pytest.mark.parametrize(
    "target_tp,batch,expected",
    [
        (1, 256, 256),
        (1, 512, None),  # below 160 GiB a TP<4 server captures up to 256
        (8, 512, 512),  # a TP>=4 server captures up to 512
        (8, 1024, None),
        (8, 3, 4),  # serving pads to the next capture size
    ],
)
def test_decode_graph_coverage_is_the_target_tp_servers(monkeypatch, target_tp, batch, expected):
    mod = _import_module(monkeypatch)

    def generate(_server_args, max_bs):  # arg_groups/cuda_graph_hook.py
        sizes = [1, 2, 4, 8, 12] + list(range(16, 257, 8)) + list(range(272, 512, 16))
        sizes += list(range(512, max_bs + 1, 32))
        sizes = [bs for bs in sizes if bs <= max_bs]
        return sizes if max_bs in sizes else sizes + [max_bs]

    _stub_modules(
        monkeypatch, {"sglang.srt.arg_groups.cuda_graph_hook": {"generate_decode_cuda_graph_batch_sizes": generate}}
    )
    monkeypatch.setattr(mod, "serving_memory_defaults", lambda tp_size: (8192, 256 if tp_size < 4 else 512))
    decode = types.SimpleNamespace(backend="full")
    monkeypatch.setattr(mod, "resolved_arg", lambda _args, name: types.SimpleNamespace(decode=decode))
    runner = types.SimpleNamespace(server_args=object())
    assert mod._serving_decode_graph_batch(runner, batch, target_tp) == expected

    decode.backend = "disabled"
    assert mod._serving_decode_graph_batch(runner, batch, target_tp) is None


def test_an_msa_fallback_is_labelled_with_the_kernel_that_ran(monkeypatch):
    mod = _import_module(monkeypatch)
    ops = types.SimpleNamespace(_msa_fallback_warned=False)
    monkeypatch.setattr(mod, "_minimax_sparse_ops", lambda: ops)
    runner = types.SimpleNamespace(attn_backend=types.SimpleNamespace(sparse=types.SimpleNamespace(use_msa=True)))
    assert mod._prefill_kernel_source(runner) == "sglang_minimax_prefill_msa_fmha_sm100"
    ops._msa_fallback_warned = True  # minimax_sparse_prefill fell back to Triton
    assert mod._prefill_kernel_source(runner) == "sglang_minimax_prefill_triton_sparse"
    runner.attn_backend.sparse.use_msa = False
    ops._msa_fallback_warned = False
    assert mod._prefill_kernel_source(runner) == "sglang_minimax_prefill_triton_sparse"


def _attention(*, fused_width):
    linear = types.SimpleNamespace
    return types.SimpleNamespace(
        is_sparse_attention_layer=True,
        disable_index_value=True,
        num_heads=8,
        num_kv_heads=1,
        num_idx_heads=1,
        head_dim=128,
        idx_head_dim=128,
        qkv_proj=linear(output_size_per_partition=10 * 128),
        o_proj=linear(input_size_per_partition=8 * 128),
        index_qkv_proj=linear(output_size_per_partition=2 * 128),
        index_o_proj=None,
        _fused_qkv_index=None if fused_width is None else linear(output_size_per_partition=fused_width),
        attn=types.SimpleNamespace(layer_id=0),
    )


def test_the_module_must_run_servings_fused_qkv_index_projection(monkeypatch):
    mod = _import_module(monkeypatch)
    runner = types.SimpleNamespace(attn_backend=types.SimpleNamespace(sparse=object(), sparse_layer_ids=[0, 1]))
    mod._validate_msa_module(runner, _attention(fused_width=12 * 128), 8, 8, 128)
    with pytest.raises(RuntimeError, match="no fused qkv\\+index projection"):
        mod._validate_msa_module(runner, _attention(fused_width=None), 8, 8, 128)
    with pytest.raises(RuntimeError, match="fused_qkv_index_proj.output_size_per_partition"):
        mod._validate_msa_module(runner, _attention(fused_width=10 * 128), 8, 8, 128)
