# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from types import SimpleNamespace

import pytest
from collector.glm53flash_sglang_runtime import _TraceState, actual_coordinates, match_frozen_requests

pytestmark = pytest.mark.unit


class Tensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *shape):
        return self

    def tolist(self):
        return self.values


class Mode:
    def __init__(self, name):
        self.name = name

    def is_decode(self):
        return self.name == "DECODE"

    def is_extend(self):
        return self.name in ("EXTEND", "MIXED")

    def is_mixed(self):
        return self.name == "MIXED"


def batch(mode, *, query=None, prefix=None, lengths=None, tokens=()):
    return SimpleNamespace(
        forward_mode=Mode(mode),
        batch_size=1,
        rids=["actual-request"],
        extend_seq_lens_cpu=query,
        extend_prefix_lens_cpu=prefix,
        seq_lens_cpu=Tensor(lengths) if lengths is not None else None,
        input_ids=Tensor(list(tokens)),
    )


def test_inclusive_native_decode_and_exact_chunk_coordinates():
    decode = actual_coordinates(batch("DECODE", lengths=[129]))
    assert decode["total_past_kv_tokens"] == 128
    assert decode["total_new_tokens"] == 1
    assert decode["inclusive_sequence_lengths"] == [129]
    chunk = actual_coordinates(batch("EXTEND", query=[128], prefix=[4096]))
    assert chunk["inclusive_sequence_lengths"] == [4224]
    assert chunk["total_past_kv_tokens"] == 4096
    with pytest.raises(RuntimeError, match="mixed"):
        actual_coordinates(batch("MIXED", query=[128], prefix=[4096]))
    with pytest.raises(RuntimeError, match="mirror"):
        actual_coordinates(batch("DECODE"))


class Timer:
    def __init__(self, reporter):
        self.reporter = reporter

    def _report(self):
        pass


def test_native_futuremap_input_does_not_depend_on_lagging_req_outputs(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.device_timer", SimpleNamespace(DeviceTimer=Timer))
    runner = SimpleNamespace(ps=SimpleNamespace(tp_rank=0), device_timer=None)
    state = _TraceState(runner, tmp_path, {"backend": "sglang"}, None)
    # The host request deliberately lags: overlap scheduling has not appended
    # its sampled20 yet, while FutureMap has already resolved native input20.
    request = SimpleNamespace(rid="actual-request", origin_input_ids=[10, 11, 12, 13], output_ids=[])
    prefill = state.before(batch("EXTEND", query=[4], prefix=[0], tokens=[10, 11, 12, 13]), [request])
    state.on_timing(t=0.003, category="extend")
    state.after(prefill, SimpleNamespace(can_run_graph=False))
    state.finish_worker(prefill, SimpleNamespace(next_token_ids=Tensor([20]), delay_sample_func=None))
    decode = state.before(batch("DECODE", lengths=[5], tokens=[20]), [request])
    state.on_timing(t=0.004, category="decode")
    state.after(decode, SimpleNamespace(can_run_graph=False))
    state.finish_worker(decode, SimpleNamespace(next_token_ids=Tensor([21]), delay_sample_func=None))
    records = [json.loads(line) for line in (tmp_path / "forward-rank-0.jsonl").read_text().splitlines()]
    assert [record["native_forward_ms"] for record in records] == [3.0, 4.0]
    request = records[1]["requests"][0]
    assert request["native_query_token_ids"] == [20]
    assert request["output_token_ids_before"] == [20]
    assert request["computed_tokens_before"] == 4
    assert request["computed_tokens_after"] == 5
    assert request["previous_forward_id"] == records[0]["forward_id"]
    assert request["same_request_real_prefix"] is True
    assert records[1]["used_cuda_graph"] is False


def test_unobserved_cached_prefix_retains_failed_admission_evidence(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sglang.srt.utils.device_timer", SimpleNamespace(DeviceTimer=Timer))
    state = _TraceState(SimpleNamespace(ps=SimpleNamespace(tp_rank=0), device_timer=None), tmp_path, {}, None)
    request = SimpleNamespace(rid="actual-request", origin_input_ids=[10, 11, 12, 13], output_ids=[])
    index = state.before(batch("EXTEND", query=[1], prefix=[3], tokens=[13]), [request])
    assert state.records[index]["requests"][0]["same_request_real_prefix"] is False
    assert state.records[index]["requests"][0]["previous_forward_id"] is None
    with pytest.raises(RuntimeError, match="category"):
        state.on_timing(t=0.001, category="decode")


def test_frozen_target_needs_exact_batch_coordinates_and_real_chain():
    manifest = {
        "request_set": "heldout-text-v1",
        "dataset_role": "heldout",
        "corpus_sha256": "a" * 64,
        "requests": {
            rid: {
                "benchmark_id": 7,
                "repetition": 5,
                "sampling_role": "measurement",
                "target_phase": "context",
                "target_query": 129,
                "target_prefix": 4095,
                "target_batch_size": 2,
            }
            for rid in ("r1", "r2")
        },
    }
    record = {
        "request_ids": ["r1", "r2"],
        "batch_size": 2,
        "phase": "context",
        "query_lengths": [129, 129],
        "prefix_lengths": [4095, 4095],
        "requests": [{"same_request_real_prefix": True}, {"same_request_real_prefix": True}],
    }
    assert match_frozen_requests(record, manifest)["stage"] == "measure"
    assert match_frozen_requests({**record, "prefix_lengths": [4096, 4096]}, manifest)["stage"] == "seed"
    assert match_frozen_requests({**record, "batch_size": 1}, manifest)["stage"] == "seed"
    assert (
        match_frozen_requests({**record, "requests": [{"same_request_real_prefix": False}]}, manifest)["stage"]
        == "seed"
    )
