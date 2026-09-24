# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU admission tests; fixture latencies are never published as measurements."""

import json
from types import SimpleNamespace

import pytest

from collector.glm53flash_vllm_runtime import _TraceState, native_context_receipt, native_coordinates

pytestmark = pytest.mark.unit


def runner(prefix, *, prompt=128):
    return SimpleNamespace(
        max_model_len=131079,
        input_batch=SimpleNamespace(req_ids=["real-rid"], num_computed_tokens_cpu=[prefix]),
        requests={"real-rid": SimpleNamespace(num_prompt_tokens=prompt, prompt_token_ids=list(range(prompt)))},
    )


def test_native_phase_follows_prompt_boundary_not_query_size():
    schedule = SimpleNamespace(num_scheduled_tokens={"real-rid": 1})
    assert native_coordinates(runner(127), schedule)["phase"] == "context"
    result = native_coordinates(runner(128), schedule)
    assert result["phase"] == "generation"
    assert result["prefix_lengths"] == [128]
    schedule.num_scheduled_tokens["real-rid"] = 2
    with pytest.raises(RuntimeError, match="speculative"):
        native_coordinates(runner(128), schedule)


class Tensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def reshape(self, *args):
        return self

    def tolist(self):
        return self.values


def test_dynamic_frozen_mapping_requires_real_worker_prefix(monkeypatch, tmp_path):
    path = tmp_path / "request-map.json"
    path.write_text(
        json.dumps(
            {
                "request_set": "native-text",
                "dataset_role": "training",
                "corpus_sha256": "a" * 64,
                "requests": {
                    "real-rid": {
                        "benchmark_id": 1,
                        "repetition": 5,
                        "sampling_role": "measurement",
                        "target_phase": "generation",
                        "target_query": 1,
                        "target_prefix": 128,
                        "target_batch_size": 1,
                    }
                },
            }
        )
    )
    monkeypatch.setenv("AISIM_GLM53_REQUEST_MANIFEST", str(path))
    state = _TraceState.__new__(_TraceState)
    state.runner = runner(128)
    state.counter, state.rank, state.provenance = 0, 0, {}
    state.previous, state.matched = {}, set()
    state.layout, state.layout_sha256 = {"admitted": True}, "b" * 64
    started = []
    state.observer = SimpleNamespace(begin=started.append)
    schedule = SimpleNamespace(num_scheduled_tokens={"real-rid": 1})
    context = SimpleNamespace(cudagraph_runtime_mode=SimpleNamespace(name="NONE"))
    record, _ = state.before(schedule, Tensor([900]), context)
    assert record["stage"] == "seed"
    assert not started
    state.previous["real-rid"] = {"computed_tokens": 128, "tokens": list(range(128)), "forward_id": "real-prefill"}
    record, completed = state.before(schedule, Tensor([900]), context)
    assert record["stage"] == "measure"
    assert completed["real-rid"][-1] == 900
    assert started[0].history_ids == ("real-prefill",)
    with pytest.raises(RuntimeError, match="more than one"):
        state.before(schedule, Tensor([900]), context)
    context.cudagraph_runtime_mode.name = "FULL"
    with pytest.raises(RuntimeError, match="CUDA graph"):
        state.before(schedule, Tensor([900]), context)


def test_native_context_receipt_requires_actual_worker_headroom(monkeypatch):
    from collector.glm53flash_validation import _validate_vllm_context

    monkeypatch.setenv("DYN_FPM_GLM53FLASH_MEASURED_CONTEXT", "131072")
    actual = runner(0)
    receipt = native_context_receipt(actual)
    _validate_vllm_context(receipt)
    for bad in (131072, 131080):
        actual.max_model_len = bad
        with pytest.raises(RuntimeError, match="actual native"):
            native_context_receipt(actual)
        with pytest.raises(ValueError, match="actual vLLM"):
            _validate_vllm_context({**receipt, "native_max_model_len": bad})
    with pytest.raises(ValueError, match="actual vLLM"):
        _validate_vllm_context({})
