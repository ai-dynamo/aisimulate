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


def test_completed_native_targets_retire_host_history_but_keep_raw_receipts():
    state = _TraceState.__new__(_TraceState)
    state.previous = {}
    state.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    state.observer = SimpleNamespace(end=lambda: [])
    saved = []
    state.append = lambda kind, record: saved.append((kind, record))
    seed = {"stage": "seed", "forward_id": "seed-1", "requests": [{"request_id": "r", "computed_tokens_after": 2}]}
    state.after(seed, {"r": [11, 12]})
    assert state.previous["r"]["tokens"] == [11, 12]
    target = {
        "stage": "measure",
        "forward_id": "target-1",
        "requests": [{"request_id": "r", "computed_tokens_after": 3}],
    }
    state.after(target, {"r": [11, 12, 13]})
    assert not state.previous
    assert [row["forward_id"] for _, row in saved] == ["seed-1", "target-1"]
    assert all(row["gpu_completed"] is True for _, row in saved)


def test_v2_reads_real_input_batch_and_rejects_gpu_cpu_state_disagreement():
    from collector.glm53flash_vllm_runtime import native_v2_coordinates

    class Lengths(Tensor):
        def __getitem__(self, key):
            return Lengths(self.values[key])

    actual = SimpleNamespace(req_states=SimpleNamespace(req_id_to_index={"r": 3}))
    batch = SimpleNamespace(
        req_ids=["r"],
        num_scheduled_tokens=[3],
        num_computed_tokens_np=[4096],
        seq_lens=Lengths([4099]),
        is_prefilling_np=[True],
        num_tokens=3,
        num_tokens_after_padding=3,
        num_draft_tokens=0,
        idx_mapping_np=[3],
    )
    schedule = SimpleNamespace(num_scheduled_tokens={"r": 3})
    assert native_v2_coordinates(actual, schedule, batch)["prefix_lengths"] == [4096]
    batch.seq_lens = Lengths([4100])
    with pytest.raises(RuntimeError, match="actual inputs"):
        native_v2_coordinates(actual, schedule, batch)


def test_v2_finalizes_after_native_later_logits_and_sample_not_execute(monkeypatch, tmp_path):
    import importlib.metadata
    import sys

    from collector import glm53flash_vllm_runtime as runtime

    calls = []
    model = SimpleNamespace(
        forward=lambda **kwargs: calls.append("model"), compute_logits=lambda: calls.append("logits")
    )

    class NativeRunner:
        def prepare_inputs(self, schedule):
            calls.append("native_prepare")
            return "native_batch"

        def execute_model(self, schedule, intermediate_tensors=None, dummy_run=False):
            if dummy_run:
                return "native_dummy"
            self.prepare_inputs(schedule)
            self.model.forward(input_ids="native_ids")
            return None

        def sample(self):
            self.model.compute_logits()
            calls.append("native_sample")
            return SimpleNamespace(sampled_token_ids=Tensor([[7]])), None, None

        def sample_tokens(self):
            return self.sample()

    class Trace:
        def __init__(self, *args):
            calls.append("install_module_hooks")

        def before(self, schedule, input_ids, context, *, native_batch):
            assert input_ids == "native_ids" and native_batch == "native_batch"
            calls.append("before")
            return {"requests": [{}]}, {"r": [5]}

        def after(self, record, completed):
            assert record["requests"][0]["sampled_token_id"] == 7
            calls.append("after")

    for name in ("provenance", "manifest"):
        (tmp_path / f"{name}.json").write_text("{}")
    monkeypatch.setenv("AISIM_GLM53_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("AISIM_GLM53_PROVENANCE", str(tmp_path / "provenance.json"))
    monkeypatch.setenv("AISIM_GLM53_OPS_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("AISIM_GLM53_PURPOSE", "ops")
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.30.0")
    monkeypatch.setitem(
        sys.modules, "vllm.forward_context", SimpleNamespace(get_forward_context=lambda: "native_context")
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", SimpleNamespace(GPUModelRunner=NativeRunner))
    monkeypatch.setattr(runtime, "_TraceState", Trace)
    runtime.install_v2()
    runner = NativeRunner()
    runner.model = model
    schedule = SimpleNamespace(total_num_scheduled_tokens=1)
    assert runner.execute_model(schedule, dummy_run=True) == "native_dummy"
    assert not calls
    runner.execute_model(schedule)
    assert "after" not in calls
    runner.sample_tokens()
    assert calls == ["native_prepare", "install_module_hooks", "before", "model", "logits", "native_sample", "after"]
    assert list(tmp_path.glob("worker-activation-*.json"))
