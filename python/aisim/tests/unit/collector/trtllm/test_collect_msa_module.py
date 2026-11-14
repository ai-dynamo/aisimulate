# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-free tests for the TRT-LLM MiniMax-M3 MSA module collector's pure helpers.
The module imports tensorrt_llm at module scope, so the helpers are AST-extracted."""

from __future__ import annotations

import ast
import contextlib
import math
import re
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.op_plans import plan

pytestmark = pytest.mark.unit
SRC = Path(__file__).resolve().parents[4] / "collector" / "trtllm" / "collect_msa_module.py"

_HELPERS = (
    "_ceil_div",
    "_round_up",
    "_msa_implementation",
    "_shard_head_counts",
    "_kv_pool_tokens",
    "_autotune_projections",
    "_write_index_key_history",
)
_CONSTANTS = ("_M3_POOL_ROUNDING_BYTES", "_TUNED_NUM_TOKENS", "_HISTORY_CHUNK_ELEMENTS")


def _load(**namespace):
    tree = ast.parse(SRC.read_text())
    selected = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in _HELPERS)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in _CONSTANTS for t in n.targets))
        or (isinstance(n, ast.AnnAssign) and getattr(n.target, "id", None) in _CONSTANTS)
    ]
    assert len(selected) == len(_HELPERS) + len(_CONSTANTS)
    namespace.setdefault("weakref", weakref)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SRC), "exec"), namespace)
    return namespace


@pytest.mark.parametrize(
    "sm,implementation", [(100, "msa"), (103, "msa"), (90, "triton"), (89, "triton"), (120, "triton")]
)
def test_msa_runs_on_the_sm100_family_and_the_triton_reference_elsewhere(sm, implementation):
    assert _load()["_msa_implementation"](sm) == implementation


@pytest.mark.parametrize(
    "num_heads,expected",
    # MiniMax-M3: 64 query heads, 4 KV heads, 4 index heads. Index heads follow their KV
    # group (index_head_range): a rank holds 4 / min(tp, 4) of them.
    [(64, (1, 4, 4)), (32, (2, 2, 2)), (16, (4, 1, 1)), (8, (8, 1, 1)), (4, (16, 1, 1))],
)
def test_a_shard_holds_its_kv_groups_index_heads(num_heads, expected):
    assert _load()["_shard_head_counts"](64, 4, 4, num_heads) == expected


def test_a_head_count_that_is_no_tp_shard_is_refused():
    with pytest.raises(ValueError, match="is not a TP shard"):
        _load()["_shard_head_counts"](64, 4, 4, 48)


_MAX_UTIL_FOR_RESUME = 0.95  # KvCacheConfig.max_util_for_resume default


def _v2_slots(max_tokens: int, num_kv_heads: int, tokens_per_block: int) -> int:
    """Slots KVCacheManagerV2 gives a one-sparse-layer M3 pool group for max_tokens.

    The quota counts K/V bytes only (_get_quota_from_max_tokens_impl) over
    max_util_for_resume; the GPU tier rounds it to its allocation unit (2 MiB, doubled
    per GiB of quota up to 32 MiB, _gpu_alloc_granularity) and gives each pool of the
    group whole units (CacheLevelStorage::ratioToSlotCountList). Index-K pages match
    K/V pages at one KV head and share their pool; otherwise they have their own.
    """
    index_page = tokens_per_block * 128 * 2
    kv_page = tokens_per_block * num_kv_heads * 128 * 2
    pool_slot_bytes = [3 * index_page] if num_kv_heads == 1 else [2 * kv_page, index_page]
    quota = math.ceil(max_tokens * 2 * num_kv_heads * 128 * 2 / _MAX_UTIL_FOR_RESUME)
    ratio = quota // ((2 << 20) * 512)
    unit = (2 << 20) << (0 if ratio == 0 else min(4, int(math.log2(ratio))))
    units = max(len(pool_slot_bytes), -(-quota // unit))
    lo, hi = 0, units * unit // min(pool_slot_bytes)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(-(-mid * size // unit) for size in pool_slot_bytes) <= units:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _plan_requests():
    """(batch, KV heads, page size, max_seq, request tokens) of every planned point."""
    seen = set()
    for op in ("msa_context_module", "msa_generation_module"):
        for sm, tokens_per_block in ((100, 128), (90, 32)):
            for points, batch, num_heads, *_ in plan(op, "trtllm", sm).rows:
                for seq_len, prefix_len in points:
                    if op == "msa_context_module":
                        max_seq, tokens = prefix_len + seq_len + 1, prefix_len + seq_len
                    else:
                        max_seq, tokens = seq_len + 2, seq_len + 1
                    seen.add((batch, max(1, num_heads // 16), tokens_per_block, max_seq, tokens))
    return sorted(seen)


def test_the_kv_pool_holds_every_planned_point_after_v2_splits_the_quota():
    ns = _load()
    requests = _plan_requests()
    assert any(r[0] == 1 and r[3] <= 4 for r in requests) and any(r[3] > 10**6 for r in requests)
    short = []
    for batch, num_kv_heads, tokens_per_block, max_seq, tokens in requests:
        max_tokens = ns["_kv_pool_tokens"](
            batch * ns["_round_up"](max_seq, tokens_per_block), 2 * num_kv_heads * 128 * 2, 128 * 2, tokens_per_block
        )
        slots = _v2_slots(max_tokens, num_kv_heads, tokens_per_block)
        if slots * _MAX_UTIL_FOR_RESUME < batch * -(-tokens // tokens_per_block) or slots * tokens_per_block < max_seq:
            short.append((batch, num_kv_heads, tokens_per_block, max_seq))
    assert short == []


def test_the_request_size_alone_misses_the_index_k_pages():
    """max_tokens is charged at K/V bytes only, so the bare request size cannot hold it."""
    batch, max_seq, tokens_per_block = 1, 65538, 128
    blocks = batch * -(-max_seq // tokens_per_block)
    assert _v2_slots(blocks * tokens_per_block, 1, tokens_per_block) < blocks


def test_projections_are_tuned_at_each_count_under_autotune():
    """Serving's warmup pass forwards every projection under autotune() at its token counts."""
    events, tuning = [], [False]

    class Linear:
        def __init__(self, name, in_features):
            self.name, self.in_features, self.dtype = name, in_features, "bf16"

        def forward(self, x):
            events.append((self.name, x, tuning[-1]))

    class Module:
        def __init__(self, children):
            self.children = children

        def modules(self):
            return [self, *self.children]

    @contextlib.contextmanager
    def autotune():
        tuning.append(True)
        yield
        tuning.append(False)

    torch = SimpleNamespace(
        inference_mode=contextlib.nullcontext,
        randn=lambda shape, dtype, device: (shape, dtype, device),
        bfloat16="bf16",
        cuda=SimpleNamespace(synchronize=lambda: None),
    )
    ns = _load(time=__import__("time"), torch=torch, Linear=Linear, autotune=autotune)
    module = Module([Linear("qkv_proj", 6144), object(), Linear("o_proj", 8192)])
    ns["_autotune_projections"](module, (8192, 1), "cuda:0")
    assert events == [
        ("qkv_proj", ((8192, 6144), "bf16", "cuda:0"), True),
        ("o_proj", ((8192, 8192), "bf16", "cuda:0"), True),
        ("qkv_proj", ((1, 6144), "bf16", "cuda:0"), True),
        ("o_proj", ((1, 8192), "bf16", "cuda:0"), True),
    ]
    assert ns["_TUNED_NUM_TOKENS"][module] == 8192
    ns["_autotune_projections"](module, (32768,), "cuda:0")
    assert ns["_TUNED_NUM_TOKENS"][module] == 32768


def test_the_decode_contract_names_the_rc2_metadata_fields():
    """rc2 dropped msa_decode_proxy_plan: a pure decode step is a decode span from row 0."""
    text = SRC.read_text()
    assert "msa_decode_proxy_plan" not in text
    assert "span.row_first == 0" in text
    # build_attention_metadata passes max_num_heads_per_kv; num_heads_per_kv is gone.
    assert "max_num_heads_per_kv=" in text and not re.search(r"(?<!max_)num_heads_per_kv", text)


def _history_torch():
    return SimpleNamespace(
        bfloat16="bf16",
        Generator=lambda device: SimpleNamespace(manual_seed=lambda seed: (device, seed)),
        randn=lambda shape, generator, device: (shape, generator, device),
    )


def test_the_index_key_history_fills_the_whole_view_in_bounded_copies():
    """The indexer scores the cached history's index keys, which the manager zeroes and
    nothing else writes: seeded rows fill the view before the forward writes its own tokens."""
    copies = []

    class Part:
        def __init__(self, rows):
            self.shape = (rows, 1, 4)

        def copy_(self, source):
            copies.append((self.shape, source))

    class View:
        dtype = "bf16"
        shape = (10, 1, 4)

        def __getitem__(self, key):
            if isinstance(key, int):
                return SimpleNamespace(numel=lambda: 4)
            return Part(min(key.stop, 10) - key.start)

    namespace = _load(torch=_history_torch())
    namespace["_HISTORY_CHUNK_ELEMENTS"] = 12  # three rows of four elements per copy
    namespace["_write_index_key_history"](SimpleNamespace(get_index_k_buffer=lambda layer: View()), "cuda:0")
    assert [shape for shape, _ in copies] == [(3, 1, 4), (3, 1, 4), (3, 1, 4), (1, 1, 4)]
    assert all(source == (shape, ("cuda:0", 0), "cuda:0") for shape, source in copies)


def test_the_index_key_history_refuses_a_pool_it_does_not_know_how_to_write():
    namespace = _load(torch=_history_torch())
    manager = SimpleNamespace(get_index_k_buffer=lambda layer: SimpleNamespace(dtype="fp8"))
    with pytest.raises(RuntimeError, match="bf16 index keys only"):
        namespace["_write_index_key_history"](manager, "cuda:0")
    namespace["_write_index_key_history"](SimpleNamespace(get_index_k_buffer=lambda layer: None), "cuda:0")
