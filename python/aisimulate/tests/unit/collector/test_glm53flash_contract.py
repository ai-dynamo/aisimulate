# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission regressions for native GLM measurements; fixtures are not perf data."""

import copy
import json
from types import SimpleNamespace

import pytest

from collector.glm53flash_contract import (
    BACKENDS,
    CHECKPOINTS,
    aggregate_rank_records,
    canonical_json,
    validate_row,
    write_parquet,
)
from collector.glm53flash_observer import NativeOperationObserver, NativeWorkload

pytestmark = pytest.mark.unit


def test_native_launch_keeps_measured_128k_and_sglang_admission_headroom(tmp_path):
    from collector.collect_glm53flash import native_command

    command = native_command("sglang", "/models/fixture", "pinned", 4, "prefill", tmp_path, tmp_path / "text")
    assert command[command.index("--context-length") + 1] == "131079"
    assert command[command.index("--benchmark-max-context-length") + 1] == "131072"
    command = native_command("vllm", "/models/fixture", "pinned", 4, "prefill", tmp_path, tmp_path / "text")
    assert command[command.index("--max-model-len") + 1] == "131072"
    assert "--no-async-scheduling" in command
    assert "--no-enable-prefix-caching" in command
    assert "--enforce-eager" in command


def sample_row():
    return {
        "component": "attention",
        "geometry": canonical_json({"backend": "vllm", "checkpoint_format": "fp8", "is_context": True, "tp_size": 2}),
        "batch_size": 1,
        "prefix": 0,
        "x": 128,
        "latency": 2.0,
        "sample_count": 10,
        "dispatch_fingerprint": "",
        "dataset_role": "calibration",
        "request_set": "authored-unit-fixture-not-performance",
        "corpus_sha256": "d" * 64,
        "evidence_sha256": "e" * 64,
        "measurement_scope": "local_compute",
        "kv_seed_regime": "empty",
        "backend": "vllm",
        "backend_version": BACKENDS["vllm"][0],
        "backend_revision": BACKENDS["vllm"][1],
        "checkpoint_revision": CHECKPOINTS["fp8"][1],
        "source_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "runtime_digest": "sha256:" + "c" * 64,
        "used_cuda_graph": False,
        "kernel_source": "fixture.actual_module.forward",
        "state_mode": "full_prefill",
        "name": "attention_0",
        "phase": "context",
        "sample": 0,
        "invocation": 1,
        "tp_rank": 0,
        "stage": "measure",
        "benchmark_id": 0,
        "repetition": 0,
        "sampling_role": "measurement",
    }


def manifest(row, *, second_layer=False):
    entries = [{key: row[key] for key in ("name", "component", "geometry")}]
    if second_layer:
        entries.append({**entries[0], "name": "attention_1"})
    return {"schema_version": 1, "phases": {"context": entries, "generation": entries}}


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"prefix": 128, "state_mode": "cached_prefill", "kv_seed_regime": "fake_kv"}, "real-prefix"),
        ({"prefix": 128}, "state mode"),
        ({"x": 131073}, "128K"),
        ({"sample_count": True}, "uint32"),
        ({"latency": float("nan")}, "finite"),
        ({"latency": True}, "finite"),
        ({"runtime_digest": "latest"}, "digest"),
        ({"checkpoint_revision": "main"}, "checkpoint revision"),
        ({"backend_revision": "main"}, "backend revision"),
        ({"used_cuda_graph": "false"}, "boolean"),
        ({"kernel_source": " "}, "dispatch"),
    ],
)
def test_reject_unqualified_measurements(updates, match):
    row = {**sample_row(), **updates}
    with pytest.raises(ValueError, match=match):
        validate_row(row)


def rank_files(tmp_path, rows):
    paths = []
    for rank, records in enumerate(rows):
        path = tmp_path / f"rank-{rank}.jsonl"
        path.write_text("".join(json.dumps({**row, "tp_rank": rank}) + "\n" for row in records))
        paths.append(path)
    return paths


def test_rank_max_then_median_and_complete_layer_evidence(tmp_path):
    first = sample_row()

    def repetitions(rank):
        return [
            {
                **first,
                "sample": rep,
                "repetition": rep,
                "invocation": rep + 1,
                "sampling_role": "warmup" if rep < 5 else "measurement",
                "latency": 1000 if rep < 5 else (1 if rank == 0 else 3) if rep < 10 else (7 if rank == 0 else 5),
            }
            for rep in range(15)
        ]

    paths = rank_files(tmp_path, [repetitions(0), repetitions(1)])
    rows = aggregate_rank_records(paths, 2, manifest(first), evidence_sha256="e" * 64)
    assert rows[0]["latency"] == 5
    assert rows[0]["sample_count"] == 10
    with pytest.raises(ValueError, match="incomplete native context graph"):
        aggregate_rank_records(paths, 2, manifest(first, second_layer=True), evidence_sha256="e" * 64)
    paths = rank_files(tmp_path, [repetitions(0)[:14], repetitions(1)[:14]])
    with pytest.raises(ValueError, match="five warmups and ten"):
        aggregate_rank_records(paths, 2, manifest(first), evidence_sha256="e" * 64)


def test_rank_and_source_mismatch_cannot_be_repaired_by_merge(tmp_path):
    row = sample_row()
    paths = rank_files(tmp_path, [[row], [{**row, "config_sha256": "d" * 64}]])
    with pytest.raises(ValueError, match="incompatible native"):
        aggregate_rank_records(paths, 2, manifest(row), evidence_sha256="e" * 64)
    paths[1].write_text("")
    with pytest.raises(ValueError, match="incomplete TP rank"):
        aggregate_rank_records(paths, 2, manifest(row), evidence_sha256="e" * 64)
    paths[1].write_text(json.dumps({**row, "tp_rank": 0}) + "\n")
    with pytest.raises(ValueError, match="evidence file"):
        aggregate_rank_records(paths, 2, manifest(row), evidence_sha256="e" * 64)


def test_shared_physical_key_uses_frozen_point_owner_not_lower_latency(tmp_path):
    row = sample_row()
    records = [
        {
            **row,
            "benchmark_id": point,
            "sample": rep,
            "repetition": rep,
            "invocation": point * 100 + rep,
            "sampling_role": "warmup" if rep < 5 else "measurement",
            "latency": 9.0 if point == 1 else 1.0,
        }
        for point in (1, 2)
        for rep in range(15)
    ]
    paths = rank_files(tmp_path, [records, records])
    result = aggregate_rank_records(paths, 2, manifest(row), evidence_sha256="e" * 64)[0]
    assert (result["latency"], result["sample_count"], result["original_point_id"]) == (9.0, 10, 1)
    result = aggregate_rank_records(paths, 2, manifest(row), evidence_sha256="e" * 64, point_ids={1: 8, 2: 3})[0]
    assert (result["latency"], result["owner_benchmark_id"], result["original_point_id"]) == (1.0, 2, 3)
    records[-1]["dispatch_fingerprint"] = "a" * 64
    paths = rank_files(tmp_path, [records, records])
    with pytest.raises(ValueError, match="incompatible measured dispatch"):
        aggregate_rank_records(paths, 2, manifest(row), evidence_sha256="e" * 64)


def test_checkpoint_formats_remain_separate_physical_keys(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    fp8 = sample_row()
    nvfp4 = copy.deepcopy(fp8)
    shape = json.loads(nvfp4["geometry"])
    shape["checkpoint_format"] = "nvfp4"
    nvfp4.update(geometry=canonical_json(shape), checkpoint_revision=CHECKPOINTS["nvfp4"][1], config_sha256="d" * 64)
    path = tmp_path / "glm53flash_module_perf.parquet"
    write_parquet([fp8, nvfp4], path)
    assert pq.read_table(path).num_rows == 2
    with pytest.raises(ValueError, match="duplicate"):
        write_parquet([fp8, fp8], path)


def test_workload_requires_native_history_and_rejects_graph_relabeling():
    values = dict(
        phase="generation",
        batch_size=1,
        query=1,
        prefix=128,
        state_mode="decode",
        request_ids=("r1",),
        history_ids=(),
        sample=0,
        invocation=1,
    )
    with pytest.raises(ValueError, match="execution receipts"):
        NativeWorkload(**values)
    with pytest.raises(ValueError, match="graph replay"):
        NativeWorkload(**{**values, "history_ids": ("prefill-r1",), "used_cuda_graph": True})


class FakeCuda:
    def __init__(self):
        self.clock = 0.0

    def current_stream(self):
        return 0

    def is_current_stream_capturing(self):
        return False

    def synchronize(self):
        pass

    def Event(self, enable_timing):  # noqa: N802 - mirrors torch.cuda's public constructor
        assert enable_timing
        cuda = self

        class Event:
            def record(self, stream):
                self.time = cuda.clock

            def elapsed_time(self, other):
                return other.time - self.time

        return Event()


def test_native_observer_preserves_arguments_and_subtracts_only_witnessed_collective():
    cuda = FakeCuda()
    row = sample_row()
    recorder = NativeOperationObserver(manifest(row), row, 0, torch_module=SimpleNamespace(cuda=cuda))

    class Native:
        def collective(self, value):
            cuda.clock += 2
            return value

        def forward(self, value):
            cuda.clock += 1
            result = self.collective(value)
            cuda.clock += 1
            return result

    native = Native()
    original = native.forward
    recorder.wrap(native, "forward", "attention_0")
    recorder.wrap_collective(native, "collective")
    recorder.begin(NativeWorkload("context", 1, 128, 0, "full_prefill", ("r1",), (), 0, 1))
    sentinel = object()
    assert native.forward(sentinel) is sentinel
    measured = recorder.end()
    # The hand-authored timeline is 1 ms compute + 2 ms collective + 1 ms compute.
    assert measured[0]["latency"] == 2
    assert measured[0]["excluded_collectives"][0]["latency"] == 2
    assert measured[0]["used_cuda_graph"] is False
    recorder.close()
    assert native.forward == original


def test_native_graph_replay_without_python_events_is_not_coverage():
    row = sample_row()
    recorder = NativeOperationObserver(manifest(row), row, 0, torch_module=SimpleNamespace(cuda=FakeCuda()))
    recorder.begin(NativeWorkload("context", 1, 128, 0, "full_prefill", ("r1",), (), 0, 1))
    with pytest.raises(RuntimeError, match="incomplete native operation coverage"):
        recorder.end()


def test_collective_is_counted_once_in_complete_local_plus_communication_partition():
    cuda = FakeCuda()
    row = sample_row()
    graph = manifest(row)
    primitive = {
        "name": "attention_allreduce_0",
        "component": "primitive",
        "geometry": canonical_json(
            {"backend": "vllm", "checkpoint_format": "fp8", "role": "allreduce", "token_selection": "all_scheduled"}
        ),
    }
    for entries in graph["phases"].values():
        if primitive not in entries:
            entries.append(primitive)
    recorder = NativeOperationObserver(graph, row, 0, torch_module=SimpleNamespace(cuda=cuda))

    class Native:
        def collective(self, value):
            cuda.clock += 2
            return value

        def forward(self, value):
            cuda.clock += 3
            self.collective(value)
            cuda.clock += 5
            return value

    native = Native()
    recorder.wrap(native, "forward", "attention_0")
    recorder.wrap_collective(native, "collective", ("attention_allreduce_0",))
    recorder.begin(NativeWorkload("context", 1, 128, 0, "full_prefill", ("r1",), (), 0, 1))
    native.forward(object())
    rows = {result["component"]: result for result in recorder.end()}
    assert rows["attention"]["latency"] == 8
    assert rows["primitive"]["latency"] == 2
    assert rows["primitive"]["measurement_scope"] == "communication"
    assert sum(result["latency"] for result in rows.values()) == 10


def test_profile_kernel_attribution_partitions_nested_communication():
    row = sample_row()
    graph = manifest(row)
    graph["phases"]["context"].append(dict(graph["phases"]["context"][0], name="comm"))
    graph["phases"]["generation"].append(dict(graph["phases"]["generation"][0], name="comm"))
    recorder = NativeOperationObserver(graph, row, 0, torch_module=SimpleNamespace(cuda=FakeCuda()))
    recorder.workload = NativeWorkload("context", 1, 128, 0, "full_prefill", ("request",), (), 4, 4)
    local = SimpleNamespace(name="aisim.glm53/attention_0", cpu_parent=None)
    comm = SimpleNamespace(name="aisim.glm53/comm", cpu_parent=local)
    events = [
        SimpleNamespace(name="launch", cpu_parent=local, kernels=[SimpleNamespace(name="gemm_native")]),
        SimpleNamespace(name="launch", cpu_parent=comm, kernels=[SimpleNamespace(name="nccl_native")]),
    ]
    recorder.profiler = SimpleNamespace(stop=lambda: None, events=lambda: events)
    recorder._finish_profile()
    assert recorder.dispatches[recorder._dispatch_key("attention_0")] == ["gemm_native"]
    assert recorder.dispatches[recorder._dispatch_key("comm")] == ["nccl_native"]


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_public_population_preserves_native_checkpoint_and_targeted_plan(monkeypatch, backend):
    import importlib

    from collector.model_cases import build_collection_case_plan
    from collector.version_resolver import build_collections

    module = importlib.import_module(f"collector.{backend}.collect_glm53flash")
    registry = importlib.import_module(f"collector.{backend}.registry").REGISTRY
    monkeypatch.delenv("COLLECTOR_MODEL_PATH", raising=False)
    raw = module.get_glm53flash_test_cases()
    assert len(raw) == 8
    assert len({case["id"] for case in raw}) == len(raw)
    assert sum(len(case["params"][-1]) for case in raw) == 120
    for fmt, (path, _revision) in CHECKPOINTS.items():
        monkeypatch.setenv("COLLECTOR_MODEL_PATH", path)
        plan = build_collection_case_plan(backend=backend, model_path=path, sm_version=103)
        assert plan.selected_ops == {"glm53flash_module"}
        cases = module.get_glm53flash_test_cases()
        assert len(cases) == 4
        assert {tuple(case["params"][1:4]) for case in cases} == {(path, fmt, 2), (path, fmt, 4)}
    resolved = build_collections(registry, backend, BACKENDS[backend][0], ops=["glm53flash_module"])
    assert len(resolved) == 1 and resolved[0]["unverified"] is True
    # The public scheduler therefore queues zero cases until native qualification;
    # raw recipes are retained and never described as scheduled GPU coverage.
