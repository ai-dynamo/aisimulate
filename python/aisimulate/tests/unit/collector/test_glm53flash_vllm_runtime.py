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


class Event:
    def record(self, stream):
        self.stream = stream

    def elapsed_time(self, end):
        return 7.0


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
    state.counter, state.rank, state.provenance = 0, 0, {"backend_version": "0.30.0"}
    state.previous, state.matched = {}, set()
    state.layout, state.layout_sha256 = {"admitted": True}, "b" * 64
    started = []
    state.observer = SimpleNamespace(begin=started.append)
    state.torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: 1, Event=lambda **kwargs: Event()))
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
    context.cudagraph_runtime_mode.name = "NONE"
    path.write_text('{"requests": {}}')
    with pytest.raises(RuntimeError, match="absent from the frozen request manifest"):
        state.before(schedule, Tensor([900]), context)
    path.unlink()
    with pytest.raises(FileNotFoundError):
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
    state.whole_events, state.whole_end_recorded = (Event(), Event(), 1), True
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
    assert target["whole_forward_gpu_ms"] == 7.0
    assert target["whole_forward_boundary"] == "embedding_to_logits_gpu_v1"
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


@pytest.mark.parametrize("backend_version", ["0.30.0", "0.30.0+glm53kpool.bf5f6b0e689d"])
def test_v2_finalizes_after_native_later_logits_and_sample_not_execute(monkeypatch, tmp_path, backend_version):
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
        (tmp_path / f"{name}.json").write_text(json.dumps({"backend_version": backend_version}))
    monkeypatch.setenv("AISIM_GLM53_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("AISIM_GLM53_PROVENANCE", str(tmp_path / "provenance.json"))
    monkeypatch.setenv("AISIM_GLM53_OPS_MANIFEST", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("AISIM_GLM53_PURPOSE", "ops")
    monkeypatch.setattr(importlib.metadata, "version", lambda _: backend_version)
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
    with pytest.raises(RuntimeError, match="precedes successful"):
        runner.execute_model(schedule)
    assert not calls
    runner._aisim_glm53_ops_warming_up = True
    runner.execute_model(schedule)
    runner.sample_tokens()
    assert calls == ["native_prepare", "model", "logits", "native_sample"]
    calls.clear()
    runner._aisim_glm53_ops_warming_up = False
    runner._aisim_glm53_ops_serving_ready = True
    runner.execute_model(schedule)
    assert "after" not in calls
    runner.sample_tokens()
    assert calls == ["native_prepare", "install_module_hooks", "before", "model", "logits", "native_sample", "after"]
    assert list(tmp_path.glob("worker-activation-*.json"))


@pytest.mark.parametrize("fail", [False, True])
def test_native_compile_lifecycle_returns_readiness_only_after_success(monkeypatch, fail):
    import sys

    from collector import glm53flash_vllm_runtime as runtime

    class Worker:
        def compile_or_warm_up_model(self):
            assert self.model_runner._aisim_glm53_ops_warming_up is True
            assert self.model_runner._aisim_glm53_ops_serving_ready is False
            if fail:
                raise RuntimeError("native warmup failed")
            return "original-native-result"

    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", SimpleNamespace(Worker=Worker))
    runtime.install_worker_lifecycle()
    worker = Worker()
    worker.model_runner = SimpleNamespace()
    if fail:
        with pytest.raises(RuntimeError, match="native warmup failed"):
            worker.compile_or_warm_up_model()
        assert worker.model_runner._aisim_glm53_ops_serving_ready is False
    else:
        assert worker.compile_or_warm_up_model() == "original-native-result"
        assert worker.model_runner._aisim_glm53_ops_serving_ready is True
    assert worker.model_runner._aisim_glm53_ops_warming_up is False


def test_v2_graph_coordinates_bind_real_and_physical_geometry_separately():
    from dataclasses import make_dataclass
    from enum import Enum

    from collector.glm53flash_vllm_runtime import native_v2_coordinates
    from tests.unit.collector.test_glm53flash_vllm_graph_policy import snapshot

    class Lengths(Tensor):
        def __getitem__(self, key):
            return Lengths(self.values[key])

    modes = Enum("NativeModes", "NONE PIECEWISE FULL")
    descriptor_type = make_dataclass("NativeDescriptor", [(key, object) for key in snapshot()["full_graphs"][0]])
    values = dict(snapshot()["full_graphs"][-1], cg_mode=modes.FULL)
    desc = descriptor_type(**values)
    ids = ["a", "b", "c"]
    actual = SimpleNamespace(req_states=SimpleNamespace(req_id_to_index=dict(zip(ids, [4, 7, 8], strict=True))))
    batch = SimpleNamespace(
        req_ids=ids,
        num_scheduled_tokens=[1] * 3,
        num_computed_tokens_np=[128] * 3,
        seq_lens=Lengths([129, 129, 129, 0]),
        is_prefilling_np=[False] * 3,
        num_tokens=3,
        num_tokens_after_padding=4,
        num_reqs_after_padding=4,
        num_draft_tokens=0,
        idx_mapping_np=[4, 7, 8],
    )
    schedule = SimpleNamespace(num_scheduled_tokens=dict.fromkeys(ids, 1))
    kwargs = {"graph_policy": snapshot(), "native_descriptor": desc}
    result = native_v2_coordinates(actual, schedule, batch, **kwargs)
    assert result["batch_size"] == result["total_new_tokens"] == 3
    assert result["prefix_lengths"] == [128] * 3
    assert result["native_dispatch"]["physical_tokens"] == result["native_dispatch"]["physical_requests"] == 4
    with pytest.raises(RuntimeError, match="eager inputs"):
        native_v2_coordinates(actual, schedule, batch)
    batch.num_tokens_after_padding = 3
    with pytest.raises(RuntimeError, match="physical padding"):
        native_v2_coordinates(actual, schedule, batch, **kwargs)
    batch.num_tokens_after_padding = 4
    batch.is_prefilling_np = [True] * 3
    with pytest.raises(RuntimeError, match="selected descriptor"):
        native_v2_coordinates(actual, schedule, batch, **kwargs)
    batch.is_prefilling_np = [False] * 3
    batch.num_computed_tokens_np = [128, 127, 128]
    with pytest.raises(RuntimeError, match="homogeneous"):
        native_v2_coordinates(actual, schedule, batch, **kwargs)


def test_v2_full_holdout_starts_after_inputs_and_requires_actual_replay_before_later_logits(monkeypatch, tmp_path):
    import importlib.metadata
    import sys

    from collector import glm53flash_vllm_graph_ops as graph
    from collector import glm53flash_vllm_runtime as runtime

    calls = []
    descriptor = object()
    batch = SimpleNamespace(input_ids="native_padded_ids")
    model = SimpleNamespace(forward=lambda: calls.append("must_not_call_python_model"))

    class Manager:
        def run_fullgraph(self, desc):
            assert desc is descriptor
            calls.append("native_replay")
            return "original-result"

    class NativeRunner:
        def prepare_inputs(self, schedule, state, desc):
            calls.append("native_inputs")
            return batch

        def execute_model(self, schedule):
            self.prepare_inputs(schedule, "native_state", descriptor)
            calls.append("native_attention_metadata")
            self.cudagraph_manager.run_fullgraph(descriptor)

        def sample(self):
            calls.extend(["native_logits", "end_gpu_window", "native_sample"])
            return SimpleNamespace(sampled_token_ids=Tensor([[7]])), None, None

        def sample_tokens(self):
            return self.sample()

    class Trace:
        rank = 0

        def __init__(self, runner, output, provenance, manifest):
            assert manifest is None
            calls.append("no_eager_hooks")

        def before(self, schedule, ids, context, **kwargs):
            assert ids == "native_padded_ids" and context is None
            assert kwargs["native_batch"] is batch and kwargs["native_descriptor"] is descriptor
            calls.append("start_gpu_window")
            return {"runtime_mode": "FULL", "requests": [{}]}, {}

        def after(self, record, completed):
            assert record["native_graph_replay_completed"] is True
            assert record["requests"][0]["sampled_token_id"] == 7
            calls.append("completed_receipt")

    monkeypatch.setenv("AISIM_GLM53_TRACE_DIR", str(tmp_path))
    path = tmp_path / "provenance.json"
    path.write_text('{"backend_version":"0.30.0"}')
    monkeypatch.setenv("AISIM_GLM53_PROVENANCE", str(path))
    monkeypatch.setenv("AISIM_GLM53_PURPOSE", "ops_graph_holdout")
    monkeypatch.delenv("AISIM_GLM53_OPS_MANIFEST", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.30.0")
    monkeypatch.setitem(sys.modules, "vllm.forward_context", SimpleNamespace(get_forward_context=lambda: None))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", SimpleNamespace(GPUModelRunner=NativeRunner))
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker.gpu.cudagraph_utils", SimpleNamespace(ModelCudaGraphManager=Manager)
    )
    monkeypatch.setattr(graph, "install_holdout_capture", lambda output: calls.append("pre_request_capture_hook"))
    monkeypatch.setattr(graph, "holdout_policy", lambda manager, model: {"backend_version": "0.30.0", "tp_rank": 0})
    monkeypatch.setattr(runtime, "_TraceState", Trace)
    runtime.install_v2()
    runner = NativeRunner()
    runner.model, runner.cudagraph_manager = model, Manager()
    runner._aisim_glm53_ops_serving_ready = True
    schedule = SimpleNamespace(total_num_scheduled_tokens=1)
    runner.execute_model(schedule)
    assert "completed_receipt" not in calls and "native_logits" not in calls
    runner.sample_tokens()
    assert calls == [
        "pre_request_capture_hook",
        "native_inputs",
        "no_eager_hooks",
        "start_gpu_window",
        "native_attention_metadata",
        "native_replay",
        "native_logits",
        "end_gpu_window",
        "native_sample",
        "completed_receipt",
    ]
    with pytest.raises(RuntimeError, match="lacks its same-runner"):
        foreign = Manager()
        original_prepare = runner.prepare_inputs
        runner.prepare_inputs = lambda *args: (original_prepare(*args), foreign.run_fullgraph(descriptor))[0]
        runner.execute_model(schedule)


def test_graph_holdout_cannot_adopt_late_or_replaced_capture():
    from collector.glm53flash_vllm_graph_ops import holdout_policy

    model, original = object(), object()
    manager = SimpleNamespace(graphs={"native_descriptor": original})
    with pytest.raises(RuntimeError, match="pre-request"):
        holdout_policy(manager, model)
    manager._aisim_glm53_holdout_capture = (model, {"TEST_ONLY": True}, dict(manager.graphs))
    assert holdout_policy(manager, model) == {"TEST_ONLY": True}
    manager.graphs["native_descriptor"] = object()
    with pytest.raises(RuntimeError, match="changed its initialized"):
        holdout_policy(manager, model)


@pytest.mark.parametrize("mode,physical,stage", [("FULL", 4, "measure"), ("PIECEWISE", 4, "seed"), ("NONE", 3, "seed")])
def test_before_preserves_actual_graph_flags_padding_and_only_logical_query_tokens(
    monkeypatch, tmp_path, mode, physical, stage
):
    from collector import glm53flash_vllm_runtime as runtime

    ids = ["a", "b", "c"]
    queries = [7, 8, 9]
    prefix = 2 if mode == "FULL" else 0
    prompts = [[1, 2]] * 3 if mode == "FULL" else [[token, 2] for token in queries]

    class PromptBuffer:
        def __getitem__(self, key):
            row, columns = key
            return Tensor(prompts[row][columns])

    state = _TraceState.__new__(_TraceState)
    state.runner = SimpleNamespace(
        max_model_len=131079,
        req_states=SimpleNamespace(
            req_id_to_index=dict(zip(ids, range(3), strict=True)),
            prompt_len=SimpleNamespace(np=[2] * 3),
            all_token_ids=SimpleNamespace(gpu=PromptBuffer()),
        ),
    )
    state.counter, state.rank, state.provenance = 0, 0, {"backend_version": "0.30.0"}
    state.previous = {rid: {"computed_tokens": 2, "tokens": [1, 2], "forward_id": "seed-" + rid} for rid in ids}
    state.matched = set()
    state.layout, state.layout_sha256 = {"admitted": True}, "b" * 64
    state.observer = None
    state.torch = SimpleNamespace(cuda=SimpleNamespace(current_stream=lambda: 1, Event=lambda **kwargs: Event()))
    coords = {
        "phase": "generation" if mode == "FULL" else "context",
        "batch_size": 3,
        "request_ids": ids,
        "query_lengths": [1] * 3,
        "prefix_lengths": [prefix] * 3,
        "total_new_tokens": 3,
        "total_past_kv_tokens": prefix * 3,
        "native_dispatch": {"descriptor": {"cg_mode": mode}, "physical_tokens": physical},
    }
    monkeypatch.setattr(runtime, "native_v2_coordinates", lambda *args, **kwargs: coords)
    path = tmp_path / "request-map.json"
    path.write_text(
        json.dumps(
            {
                "request_set": "TEST_ONLY_native",
                "dataset_role": "holdout",
                "corpus_sha256": "a" * 64,
                "requests": {
                    rid: {
                        "benchmark_id": 1,
                        "repetition": 5,
                        "sampling_role": "measurement",
                        "target_phase": "generation",
                        "target_query": 1,
                        "target_prefix": 2,
                        "target_batch_size": 3,
                    }
                    for rid in ids
                },
            }
        )
    )
    monkeypatch.setenv("AISIM_GLM53_REQUEST_MANIFEST", str(path))
    record, completed = state.before(
        object(),
        Tensor(queries + ([999] if physical == 4 else [])),
        None,
        native_batch=object(),
        graph_policy={"TEST_ONLY": True},
        native_descriptor=object(),
    )
    assert record["stage"] == stage
    assert record["used_cuda_graph"] is (mode != "NONE")
    assert record["num_padded_tokens"] == physical
    assert record["total_new_tokens"] == 3
    assert [row["native_query_token_ids"] for row in record["requests"]] == [[7], [8], [9]]
    assert all(999 not in tokens for tokens in completed.values())
    if stage == "measure":
        assert state.whole_boundary == "native_metadata_to_logits_gpu_v1"
