# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import types
from types import SimpleNamespace

import pytest
from collector.sglang import runtime_limits

pytestmark = pytest.mark.unit


class _ChunkedIndexer:
    def _should_chunk_mqa_logits(self):
        return True

    def _get_topk_ragged(self, start, q_offset):
        while start < q_offset:
            logits_chunk = start
            start += 1
        return logits_chunk


def test_dsa_chunking_source_contract_is_detected(monkeypatch):
    module = SimpleNamespace(Indexer=_ChunkedIndexer)
    monkeypatch.setattr(runtime_limits.importlib, "import_module", lambda _name: module)

    assert runtime_limits.sglang_dsa_mqa_logits_chunking_supported()


def test_dsa_chunking_detection_failure_does_not_enable_a_heuristic_filter(monkeypatch):
    def missing(_name):
        raise ModuleNotFoundError

    monkeypatch.setattr(runtime_limits.importlib, "import_module", missing)

    with pytest.raises(RuntimeError, match="chunking source contract was not detected"):
        runtime_limits.sglang_dsa_mqa_logits_chunking_supported()


def test_request_slots_are_sglang_floor_for_the_planned_batches():
    # Every planned DSA module batch (at most 1024) gets the floor, whatever the pool's token capacity.
    assert runtime_limits.request_slots([1, 256, 1024]) == runtime_limits.SGLANG_MIN_REQUEST_SLOTS == 2048
    assert runtime_limits.request_slots(batch for batch in (1,)) == 2048


def test_request_slots_cover_a_batch_past_the_floor():
    assert runtime_limits.request_slots([4096, 8]) == 4096


@pytest.mark.parametrize("gpu_mem_mib", [None, 16 * 1024, 32 * 1024, 48 * 1024, 81_559, 144_000, 183_359])
@pytest.mark.parametrize("tp_size", [1, 8])
def test_request_slots_keep_sglang_decode_graph_buckets(monkeypatch, gpu_mem_mib, tp_size):
    # SGLang lowers the decode CUDA-graph batch size to max_running_requests when that is smaller
    # (arg_groups/memory_hook.py:182-188 at v0.5.23); the floor is not below any device's default.
    common = types.ModuleType("sglang.srt.utils.common")
    common.get_device_memory_capacity = lambda: gpu_mem_mib
    for name in ("sglang", "sglang.srt", "sglang.srt.utils"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.common", common)

    assert runtime_limits.request_slots([1]) >= runtime_limits.serving_memory_defaults(tp_size)[1]
