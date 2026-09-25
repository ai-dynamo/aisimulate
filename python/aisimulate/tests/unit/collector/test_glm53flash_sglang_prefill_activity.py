# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY source-scope/correlation counterexamples, never measured rows."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector import glm53flash_sglang_prefill_activity as prefill
from collector.glm53flash_observer import NativeOperationObserver, NativeWorkload

pytestmark = pytest.mark.unit


def fixture():
    def scope(name, start, end):
        return {"name": name, "cat": "user_annotation", "ph": "X", "ts": start, "dur": end - start, "pid": 1, "tid": 2}

    descriptions = [
        ("embedding", 2, 9, None),
        ("mhc_pre_attn_3", 20, 29, None),
        ("attention_3", 30, 39, None),
        ("attention_3", 40, 69, None),
        ("attention_allreduce_3", 50, 59, 3),
        ("logits", 70, 89, None),
    ]
    calls, events = [], [scope(prefill.PREFILL_RANGE, 0, 100), scope(prefill.SETUP_RANGE, 10, 19)]
    for index, (name, start, end, parent) in enumerate(descriptions):
        calls.append(
            {
                "operation": name,
                "scope_name": f"{prefill.UNIT_PREFIX}{index}/{name}",
                "source": f"TEST_ONLY.native.part{index}",
                "completed": True,
                "included_sources": [],
                "excluded_collective_sources": ["TEST_ONLY.native.part4"] if index == 3 else [],
                "parent_operation": None if parent is None else descriptions[parent][0],
                "parent_scope_name": None
                if parent is None
                else f"{prefill.UNIT_PREFIX}{parent}/{descriptions[parent][0]}",
            }
        )
        events.append(scope(calls[-1]["scope_name"], start, end))
    # Kernel names intentionally lie about semantic ownership: an actual MoE
    # name in this test belongs to the source mHC call by exact correlation.
    for correlation, (start, category, name) in enumerate(
        [
            (4, "kernel", "embedding_kernel"),
            (12, "gpu_memset", "Memset"),
            (22, "kernel", "MoE_name_is_not_an_ownership_rule"),
            (32, "kernel", "latent_projection"),
            (42, "kernel", "attention"),
            (52, "kernel", "collective"),
            (72, "kernel", "logits"),
        ],
        1,
    ):
        events.append(
            {
                **scope("cudaMemsetAsync" if category == "gpu_memset" else "cudaLaunchKernel", start, start + 1),
                "cat": "cuda_runtime",
                "args": {"correlation": correlation},
            }
        )
        events.append(
            {
                "cat": category,
                "name": name,
                "ph": "X",
                "ts": start + 1,
                "dur": 1,
                "args": {
                    "correlation": correlation,
                    "stream": 7,
                    **(
                        {"bytes": 360}
                        if category == "gpu_memset"
                        else {"grid": [1, 1, 1], "block": [32, 1, 1], "shared memory": 0}
                    ),
                },
            }
        )
    return events, calls, list(dict.fromkeys(row[0] for row in descriptions))


def test_complete_calls_bind_disjoint_projection_and_exclusive_collective():
    events, calls, names = fixture()
    result = prefill.bind_prefill_activity(events, calls, names)
    assert result["contribution_counts"]["attention_3"] == 2
    assert result["contribution_counts"]["native_graph_setup"] == 1
    assert {row["launch_correlation"]: row["operation"] for row in result["activities"]} == {
        1: "embedding",
        2: "native_graph_setup",
        3: "mhc_pre_attn_3",
        4: "attention_3",
        5: "attention_3",
        6: "attention_allreduce_3",
        7: "logits",
    }
    signatures = prefill.dispatch_signatures(result)
    assert json.loads(signatures["mhc_pre_attn_3"][0])["name"] == "MoE_name_is_not_an_ownership_rule"
    assert json.loads(signatures["native_graph_setup"][0])["bytes"] == 360
    assert result["formal_admission"] is False
    assert result["dispatch_interpolation_admitted"] is False
    assert prefill.bind_prefill_activity(events + [dict(events[0], cat="gpu_user_annotation")], calls, names) == result


@pytest.mark.parametrize(
    "defect",
    [
        "missing_call",
        "duplicate_scope",
        "missing_cpu",
        "duplicate_cpu",
        "unknown_cpu",
        "wrong_thread",
        "wrong_category",
        "incomplete",
        "parent_part",
        "parent_source",
        "missing_collective",
        "compute_overlap",
        "setup_overlap",
        "graph_launch",
        "graph_activity",
        "missing_activity",
        "orphan_activity",
        "unknown_api",
        "duplicate_correlation",
        "boundary",
        "straddle",
        "unowned",
        "extra_scope_id",
        "repeated_nonattention",
        "repeated_kda",
    ],
)
def test_malformed_or_incomplete_actual_ownership_rejects(defect):
    events, calls, names = fixture()
    apis = [row for row in events if row["cat"] == "cuda_runtime"]
    gpu = [row for row in events if row["cat"] in ("kernel", "gpu_memset")]
    if defect == "missing_call":
        calls.pop()
    elif defect == "duplicate_scope":
        calls[-1]["scope_name"] = calls[0]["scope_name"]
    elif defect == "missing_cpu":
        events.pop(2)
    elif defect == "duplicate_cpu":
        events.append(copy.deepcopy(events[2]))
    elif defect == "unknown_cpu":
        events.append(dict(events[2], name="aisim.glm53/unknown"))
    elif defect == "wrong_thread":
        events[2]["tid"] = 99
    elif defect == "wrong_category":
        events[2]["cat"] = "gpu_user_annotation"
    elif defect == "incomplete":
        calls[0]["completed"] = False
    elif defect == "parent_part":
        calls[4]["parent_scope_name"] = calls[2]["scope_name"]
    elif defect == "parent_source":
        calls[4]["source"] = "TEST_ONLY.wrong_source"
    elif defect == "missing_collective":
        calls[3]["excluded_collective_sources"] = []
    elif defect == "compute_overlap":
        events[3].update(ts=8, dur=20)
    elif defect == "setup_overlap":
        events[1].update(ts=8, dur=20)
    elif defect == "graph_launch":
        apis[0]["name"] = "cudaGraphLaunch"
    elif defect == "graph_activity":
        gpu[0]["args"]["graph id"] = 5
    elif defect == "missing_activity":
        events.remove(gpu[0])
    elif defect == "orphan_activity":
        gpu[0]["args"]["correlation"] = 55
    elif defect == "unknown_api":
        apis[0]["name"] = "cudaUnknownLaunch"
    elif defect == "duplicate_correlation":
        apis[0]["args"]["correlation"] = 2
    elif defect == "boundary":
        apis[0].update(ts=2, dur=0)
    elif defect == "straddle":
        apis[0].update(ts=8, dur=3)
    elif defect == "unowned":
        apis[0].update(ts=96, dur=1)
    elif defect == "extra_scope_id":
        calls[0]["scope_name"] = f"{prefill.UNIT_PREFIX}98/embedding"
    elif defect in ("repeated_nonattention", "repeated_kda"):
        replacement = "embedding" if defect == "repeated_nonattention" else "attention_2"
        names[names.index("attention_3")] = replacement
        for index in (2, 3):
            calls[index]["operation"] = replacement
            calls[index]["scope_name"] = f"{prefill.UNIT_PREFIX}{index}/{replacement}"
    with pytest.raises(ValueError):
        prefill.bind_prefill_activity(events, calls, names)


def test_completed_empty_call_is_preserved_and_unowned_activity_is_durable_error_data():
    events, calls, names = fixture()
    events = [row for row in events if row.get("args", {}).get("correlation") != 3]
    result = prefill.bind_prefill_activity(events, calls, names)
    assert result["operation_activity_indices"]["mhc_pre_attn_3"] == []
    assert prefill.dispatch_signatures(result)["mhc_pre_attn_3"] == []
    [row for row in events if row["cat"] == "cuda_runtime"][0].update(ts=96, dur=1)
    with pytest.raises(ValueError, match="unowned") as caught:
        prefill.bind_prefill_activity(events, calls, names)
    assert caught.value.unowned_activities[0]["launch_correlation"] == 1


def test_authoritative_callback_does_not_consume_wrong_functionevent_parents():
    class Profile:
        def stop(self):
            pass

        def events(self):
            raise AssertionError("legacy FunctionEvent mapping must not run")

    observer = NativeOperationObserver(
        {"phases": {"context": [{"name": "x"}]}},
        {},
        0,
        torch_module=SimpleNamespace(),
        profile_callback=lambda *_: {"x": ["actual correlated kernel"]},
    )
    observer.workload = NativeWorkload("context", 1, 1, 0, "full_prefill", ("request",), (), 4, 1)
    observer.profiler = Profile()
    observer._finish_profile()
    assert observer.dispatches[observer._dispatch_key("x")] == ["actual correlated kernel"]


@pytest.mark.parametrize("dispatch", [{}, {"x": [None]}, {"x": "kernel"}, {"x": [], "extra": []}])
def test_authoritative_callback_requires_complete_named_inventory(dispatch):
    observer = NativeOperationObserver(
        {"phases": {"context": [{"name": "x"}]}},
        {},
        0,
        torch_module=SimpleNamespace(),
        profile_callback=lambda *_: dispatch,
    )
    observer.workload = NativeWorkload("context", 1, 1, 0, "full_prefill", ("request",), (), 4, 1)
    observer.profiler = SimpleNamespace(stop=lambda: None)
    with pytest.raises(RuntimeError, match="complete authoritative"):
        observer._finish_profile()


def lifecycle_fixture(monkeypatch, tmp_path):
    events, calls, names = fixture()
    log = []
    cuda = SimpleNamespace(time=0, stream=7)

    class Event:
        def record(self, stream):
            assert stream == cuda.stream
            self.time = cuda.time

        def elapsed_time(self, end):
            return end.time - self.time

    class Scope:
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            log.append(("enter", self.name))

        def __exit__(self, *_):
            log.append(("exit", self.name))

    class Profile:
        def start(self):
            log.append(("profile", "start"))

        def stop(self):
            log.append(("profile", "stop"))

        def events(self):
            raise AssertionError("legacy parent walk must remain unused")

        def export_chrome_trace(self, path):
            log.append(("profile", "export"))
            Path(path).write_text(json.dumps({"traceEvents": events}))

    cuda.Event = lambda **_: Event()
    cuda.current_stream = lambda: cuda.stream
    cuda.is_current_stream_capturing = lambda: False
    cuda.synchronize = lambda: log.append(("cuda", "synchronize"))
    torch = SimpleNamespace(
        cuda=cuda,
        float32="torch.float32",
        profiler=SimpleNamespace(
            profile=lambda **_: Profile(),
            record_function=Scope,
            ProfilerActivity=SimpleNamespace(CPU="cpu", CUDA="cuda"),
        ),
    )

    class Allocator:
        def __init__(self, buffer_size, dtype, device):
            log.append(("native", "allocator"))
            cuda.time += 1

    class Model:
        duplicate = False
        fail = False

        def forward(self, input_ids, positions, forward_batch):
            log.append(("native", "model"))
            Allocator(90, torch.float32, "cuda:0")
            if self.duplicate:
                Allocator(90, torch.float32, "cuda:0")
            if self.fail:
                raise RuntimeError("TEST_ONLY original model error")
            cuda.time += 3
            return input_ids

    native_module = SimpleNamespace(BumpAllocator=Allocator)
    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sglang.srt.models", SimpleNamespace(glm5_next=native_module))
    monkeypatch.setattr(
        prefill,
        "model_identity",
        lambda *_: {"source_pins": prefill.SOURCE_PINS, "native_model_contract": prefill.MODEL_CONTRACT},
    )
    model = Model()
    original_model, original_allocator = model.forward, Allocator.__init__
    observer = NativeOperationObserver(
        {"phases": {"context": [{"name": name} for name in names]}}, {"backend": "sglang"}, 0, torch_module=torch
    )
    observer.native_call_inventory = lambda: copy.deepcopy(calls)
    helper = prefill.NativeSglangPrefillExecution(SimpleNamespace(model=model), observer, tmp_path)
    batch = SimpleNamespace(
        can_run_tbo=False,
        contains_mm_inputs=lambda: False,
        input_embeds=None,
        forward_mode=SimpleNamespace(is_extend=lambda: True, is_mixed=lambda: False),
    )
    return helper, observer, model, batch, log, original_model, original_allocator, events


def record(sample):
    return {
        "run_id": "TEST_ONLY_RUN",
        "request_set": "TEST_ONLY_REQUESTS",
        "tp_rank": 0,
        "invocation": sample + 1,
        "forward_id": f"rank-0/forward-{sample + 1}",
        "phase": "context",
        "benchmark_id": 1,
        "repetition": sample,
        "sampling_role": "warmup" if sample < 5 else "measurement",
        "dataset_role": "calibration",
        "corpus_sha256": "a" * 64,
        "request_ids": ["TEST_ONLY_REQUEST"],
    }


def begin(helper, observer, sample):
    row = record(sample)
    observer.begin(NativeWorkload("context", 1, 32, 0, "full_prefill", ("TEST_ONLY_REQUEST",), (), sample, sample + 1))
    helper.begin(row)
    return row


def test_lifecycle_preserves_original_once_and_profiles_only_excluded_fifth_warmup(monkeypatch, tmp_path):
    helper, observer, model, batch, log, original, allocate, _ = lifecycle_fixture(monkeypatch, tmp_path)
    for sample in range(15):
        row = begin(helper, observer, sample)
        assert log.count(("profile", "start")) == (1 if sample > 4 else 0)
        token = object()
        assert model.forward(token, None, batch) is token
        assert observer.profiler is None
        log.append(("native", "sampling"))
        # This fixture substitutes the completed native-call inventory above;
        # the separate event-pool tests exercise real observer.end() and reads.
        observer.torch.cuda.synchronize()
        observer.event_pool.consumed_operations()
        helper.complete(row)
        assert row["native_prefill_setup"]["latency"] == 1
        assert row["whole_forward_gpu_ms"] == 4
        assert row["native_prefill_event_pool"]["contract"] == "sglang_prefill_preinitialized_events_v1"
        assert row["native_prefill_event_pool"]["bootstrap_sample"] == 0
        assert ("native_prefill_profile" in row) is (sample == 4)
        observer.workload = None
    assert log.count(("native", "model")) == log.count(("native", "allocator")) == 15
    assert log.count(("profile", "start")) == log.count(("profile", "stop")) == 1
    assert log.index(("profile", "stop")) < log.index(("profile", "export"))
    saved = list(tmp_path.glob("prefill-profile-*.json"))
    assert len(saved) == 1
    assert json.loads(saved[0].read_text())["aisim_native_forward"]["invocation"] == 5
    helper.close()
    assert model.forward == original and helper.allocator.__init__ is allocate


def test_setup_and_whole_pairs_remain_owned_until_operation_reads_complete(monkeypatch, tmp_path):
    helper, observer, model, batch, _, _, _, _ = lifecycle_fixture(monkeypatch, tmp_path)
    row = begin(helper, observer, 0)
    model.forward(object(), None, batch)
    with pytest.raises(RuntimeError, match="no complete"):
        helper.complete(row)
    assert helper.active is not None and observer.event_pool.active
    observer.torch.cuda.synchronize()
    observer.event_pool.consumed_operations()
    helper.complete(row)
    assert helper.active is None and not observer.event_pool.active


@pytest.mark.parametrize(
    "defect", ["duplicate", "native_failure", "copy_buffer", "tbo", "multimodal", "unowned", "trace_collision"]
)
def test_lifecycle_failed_original_trace_is_preserved_and_wrappers_restore(monkeypatch, tmp_path, defect):
    helper, observer, model, batch, log, original, allocate, events = lifecycle_fixture(monkeypatch, tmp_path)
    row = begin(helper, observer, 4)
    if defect == "duplicate":
        model.duplicate = True
    elif defect == "native_failure":
        model.fail = True
    elif defect == "copy_buffer":
        batch.input_embeds = object()
    elif defect == "tbo":
        batch.can_run_tbo = True
    elif defect == "multimodal":
        batch.contains_mm_inputs = lambda: True
    elif defect == "unowned":
        next(event for event in events if event["cat"] == "cuda_runtime").update(ts=96, dur=1)
    elif defect == "trace_collision":
        (tmp_path / "prefill-profile-rank-0-forward-5.json").write_text("ORIGINAL")
    with pytest.raises((RuntimeError, ValueError)) as error:
        model.forward(object(), None, batch)
    helper.abort(error.value)
    helper.close()
    assert helper.active is None and observer.profiler is None
    assert model.forward == original and helper.allocator.__init__ is allocate
    if defect == "unowned":
        failure = json.loads(next(tmp_path.glob("*.failure.json")).read_text())
        assert failure["unowned_device_activities"][0]["launch_correlation"] == 1
    elif defect == "trace_collision":
        assert (tmp_path / "prefill-profile-rank-0-forward-5.json").read_text() == "ORIGINAL"
    elif defect not in ("copy_buffer", "tbo", "multimodal"):
        assert (
            json.loads(next(tmp_path.glob("failed-prefill-profile-*.json")).read_text())["aisim_native_prefill"][
                "failed"
            ]
            is True
        )
    else:
        assert not list(tmp_path.iterdir())
        assert ("profile", "start") not in log
    assert "native_prefill_profile" not in row
