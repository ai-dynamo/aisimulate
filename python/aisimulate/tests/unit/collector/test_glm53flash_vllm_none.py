# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY object/call evidence; no native GPU or accepted timing fixtures."""

import functools
import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from collector import glm53flash_vllm_none as none
from collector import glm53flash_vllm_runtime as runtime

pytestmark = pytest.mark.unit


def native_model_fixture(monkeypatch, tmp_path):
    """Distinct TEST_ONLY classes and actual code files stand in for imports."""

    class Module:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def modules(self):
            yield self
            for child in vars(self).values():
                if isinstance(child, Module):
                    yield from child.modules()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(nn=SimpleNamespace(Module=Module)))
    package = tmp_path / "vllm"
    package.mkdir()
    (package / "__init__.py").write_text("# TEST_ONLY\n")
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__file__=str(package / "__init__.py")))
    sources = [
        (
            "model_executor/models/glm4_1v.py",
            "class Glm4vForConditionalGeneration(Module):\n"
            "    def forward(self, value): return self.language_model.model(value)\n"
            "    def compute_logits(self, value): return self.language_model.compute_logits(value)\n",
        ),
        (
            "models/glm5next/nvidia/model.py",
            "class Glm5NextModel(Module):\n"
            "    def forward(self, value): return value\n"
            "class Glm5NextForCausalLM(Module):\n"
            "    def __init__(self): self.model = Glm5NextModel()\n"
            "    def compute_logits(self, value): return value\n"
            "class Glm5NextForConditionalGeneration(Glm4vForConditionalGeneration):\n"
            "    def __init__(self): self.language_model = Glm5NextForCausalLM()\n",
        ),
    ]
    namespace = {"Module": Module}
    for relative, source in sources:
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        module_name = "vllm." + relative[:-3].replace("/", ".")
        module = ModuleType(module_name)
        module.__dict__.update(namespace)
        exec(compile(source, str(path), "exec"), module.__dict__)
        monkeypatch.setitem(sys.modules, module_name, module)
        monkeypatch.setitem(none.SOURCE_PINS, relative, hashlib.sha256(path.read_bytes()).hexdigest())
        namespace.update({key: value for key, value in module.__dict__.items() if key.startswith("Glm")})
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.cudagraph_utils",
        SimpleNamespace(has_compiled_submodule=lambda model: getattr(model, "compiled", False)),
    )
    return namespace["Glm5NextForConditionalGeneration"](), package


@pytest.mark.parametrize("defect", [None, "class", "forward", "logits", "compiled", "module_compile", "source"])
def test_none_identity_binds_original_inherited_method_and_actual_classes(monkeypatch, tmp_path, defect):
    model, package = native_model_fixture(monkeypatch, tmp_path)
    if defect == "class":
        model.__class__ = type("Substitute", (type(model),), {})
    elif defect == "forward":
        model.forward = lambda value: value
    elif defect == "logits":
        model.language_model.compute_logits = lambda value: value
    elif defect == "compiled":
        model.compiled = True
    elif defect == "module_compile":
        model.language_model.model._compiled_call_impl = lambda value: value
    elif defect == "source":
        (package / "model_executor/models/glm4_1v.py").write_text("# changed TEST_ONLY source\n")
    if defect:
        with pytest.raises(RuntimeError, match="native serving NONE"):
            none.NativeNoneModelWitness(model)
    else:
        witness = none.NativeNoneModelWitness(model)
        witness.validate()
        sentinel = object()
        assert model(sentinel) is sentinel and model.compute_logits(sentinel) is sentinel
        assert witness.receipt["methods"][0]["method"].endswith("Glm4vForConditionalGeneration.forward")
        assert witness.receipt["admission"] == "DIAGNOSTIC_ONLY_NATIVE_CALLS_STILL_REQUIRED"


@pytest.mark.parametrize("replacement", ["root", "inner", "forward", "logits", "call", "compiled"])
def test_none_identity_rejects_post_init_replacement(monkeypatch, tmp_path, replacement):
    model, _ = native_model_fixture(monkeypatch, tmp_path)
    witness = none.NativeNoneModelWitness(model)
    original = model.forward

    @functools.wraps(original)
    def observed(value):
        return original(value)

    model.forward = observed
    witness.bind_observer_wrapper("forward", observed)
    with pytest.raises(RuntimeError, match="repeated"):
        witness.bind_observer_wrapper("forward", observed)
    if replacement == "root":
        model.language_model = type(model.language_model)()
    elif replacement == "inner":
        model.language_model.model = type(model.language_model.model)()
    elif replacement == "forward":
        model.forward = functools.wraps(observed)(lambda value: observed(value))
    elif replacement == "logits":
        model.language_model.compute_logits = lambda value: value
    elif replacement == "call":
        monkeypatch.setattr(type(model), "__call__", lambda self, value: value)
    else:
        model.language_model.model._compiled_call_impl = lambda value: value
    with pytest.raises(RuntimeError, match="identity changed"):
        witness.validate()


def none_record():
    return {
        "runtime_mode": "NONE",
        "phase": "context",
        "used_cuda_graph": False,
        "num_padded_tokens": 8,
        "total_new_tokens": 8,
        "batch_size": 2,
        "native_dispatch": {
            "descriptor": {"cg_mode": "NONE", "num_tokens": 8, "num_reqs": 2},
            "physical_tokens": 8,
            "physical_requests": 2,
            "policy_sha256": "a" * 64,
        },
        "invocation": 4,
        "forward_id": "rank-0/forward-4",
        "tp_rank": 0,
        "serving_none_model_sha256": "b" * 64,
        "native_operation_calls": {f"TEST_ONLY_{i}": 1 for i in range(277)},
    }


@pytest.mark.parametrize("defect", [None, "graph", "padding", "decode", "missing", "duplicate", "foreign"])
def test_none_raw_inventory_does_not_accept_graph_or_incomplete_native_calls(defect):
    record = none_record()
    rows = [
        {"name": f"TEST_ONLY_{i}", "used_cuda_graph": False, "invocation": 4, "tp_rank": 0, "phase": "context"}
        for i in range(277)
    ]
    if defect == "graph":
        record["runtime_mode"] = "FULL"
    elif defect == "padding":
        record["native_dispatch"]["physical_tokens"] = 16
    elif defect == "decode":
        record["phase"] = "generation"
    elif defect == "missing":
        rows.pop()
    elif defect == "duplicate":
        rows[-1] = dict(rows[0])
    elif defect == "foreign":
        rows[-1]["invocation"] = 3
    if defect:
        with pytest.raises(RuntimeError):
            none.diagnostic_operation_rows(record, rows)
    else:
        actual = none.diagnostic_operation_rows(record, rows)
        assert len(actual) == 277
        assert all(row["measurement_admission"] == "DIAGNOSTIC_ONLY_NO_TABLE_EXPORT" for row in actual)
        assert all(row["native_dispatch"] == record["native_dispatch"] for row in actual)


@pytest.mark.parametrize(
    "purpose,value,pw,capture",
    [
        ("ops", "1", "0", "0"),
        ("ops_holdout", "1", "0", "0"),
        ("ops_graph", "true", "0", "0"),
        ("ops_graph", "1", "1", "0"),
        ("ops_graph", "1", "0", "1"),
    ],
)
def test_none_observation_requires_separate_explicit_serving_diagnostic(monkeypatch, purpose, value, pw, capture):
    monkeypatch.setenv("AISIM_GLM53_SERVING_NONE_DIAGNOSTIC", value)
    monkeypatch.setenv("AISIM_GLM53_PIECEWISE_REPLAY", pw)
    monkeypatch.setenv("AISIM_GLM53_PIECEWISE_CAPTURE_ONLY", capture)
    with pytest.raises(RuntimeError, match="own explicit"):
        runtime._serving_none_enabled(purpose)


@pytest.mark.parametrize("measured", [False, True])
def test_none_raw_timings_cannot_write_eager_query_file(tmp_path, measured):
    class Event:
        def __init__(self, time):
            self.time = time

        def elapsed_time(self, other):
            return other.time - self.time

    state = runtime._TraceState.__new__(runtime._TraceState)
    state.output, state.rank, state.serving_none = tmp_path, 0, True
    state.serving_none_measured = measured
    state.previous, state.graph_execution = {}, None
    state.whole_events, state.whole_end_recorded = (Event(0), Event(20), 1), True
    state.none_boundaries = [Event(3), Event(17), Event(18)]
    state.whole_boundary = "native_metadata_to_logits_gpu_v1"
    state.observer = SimpleNamespace(
        events=[{"name": f"TEST_ONLY_{i}"} for i in range(277)],
        end=lambda: [
            {"name": f"TEST_ONLY_{i}", "used_cuda_graph": False, "invocation": 4, "tp_rank": 0, "phase": "context"}
            for i in range(277)
        ],
    )
    record = {
        **none_record(),
        "stage": "measure",
        "benchmark_id": 1,
        "repetition": 5,
        "sampling_role": "measurement",
        "dataset_role": "calibration",
        "request_set": "TEST_ONLY",
        "corpus_sha256": "c" * 64,
        "requests": [],
        "native_none_forward_completed": True,
    }
    if measured:
        from collector.glm53flash_vllm_none_activity import NONE_MEASUREMENT_CONTRACT

        record.update(measurement_contract=NONE_MEASUREMENT_CONTRACT, profiled=False)
    state.after(record, {})
    assert not (tmp_path / "rank-0.jsonl").exists()
    name = "serving-none-measured-ops-rank-0.jsonl" if measured else "serving-none-ops-rank-0.jsonl"
    rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
    assert len(rows) == 277
    if measured:
        assert all(
            row["measurement_contract"] == NONE_MEASUREMENT_CONTRACT and "measurement_admission" not in row
            for row in rows
        )
    else:
        assert all(row["measurement_admission"] == "DIAGNOSTIC_ONLY_NO_TABLE_EXPORT" for row in rows)
    assert record["whole_forward_gpu_ms"] == 20
    assert record["native_runtime_boundary_gpu_ms"] == {
        "prepared_inputs_to_raw_model_entry": 3,
        "raw_model_return_to_logits_entry": 1,
    }


@pytest.mark.parametrize(
    "purpose,enabled", [("ops_graph", True), ("ops_graph_holdout", True), ("ops", False), ("unknown", False)]
)
def test_none_measured_flag_keeps_legacy_diagnostic_and_eager_paths_separate(monkeypatch, purpose, enabled):
    monkeypatch.setenv("AISIM_GLM53_SERVING_NONE_MEASURED", "1")
    for key in (
        "AISIM_GLM53_SERVING_NONE_DIAGNOSTIC",
        "AISIM_GLM53_PIECEWISE_CAPTURE_ONLY",
        "AISIM_GLM53_PIECEWISE_REPLAY",
    ):
        monkeypatch.setenv(key, "0")
    if enabled:
        assert runtime._serving_none_measured_enabled(purpose)
        assert not runtime._serving_none_enabled(purpose)
    else:
        with pytest.raises(RuntimeError, match="own explicit"):
            runtime._serving_none_measured_enabled(purpose)


@pytest.mark.parametrize(
    "other",
    ["AISIM_GLM53_SERVING_NONE_DIAGNOSTIC", "AISIM_GLM53_PIECEWISE_CAPTURE_ONLY", "AISIM_GLM53_PIECEWISE_REPLAY"],
)
def test_none_measured_flag_rejects_combined_instrumentation(monkeypatch, other):
    monkeypatch.setenv("AISIM_GLM53_SERVING_NONE_MEASURED", "1")
    monkeypatch.setenv(other, "1")
    with pytest.raises(RuntimeError, match="own explicit"):
        runtime._serving_none_measured_enabled("ops_graph")


@pytest.mark.parametrize("calibration", [True, False])
@pytest.mark.parametrize("defect", [None, "omitted", "duplicate", "replaced_inner"])
@pytest.mark.parametrize("measured", [False, True])
def test_none_serving_adapter_preserves_native_forward_and_later_logits(
    monkeypatch, tmp_path, calibration, defect, measured
):
    import importlib.metadata

    from collector import glm53flash_vllm_graph_ops as graph

    model, _ = native_model_fixture(monkeypatch, tmp_path)
    calls, failed = [], []
    actual_inputs = object()

    class SampledTokens:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [[7]]

    class Manager:
        def run_fullgraph(self, *args):
            pytest.fail("native NONE must not replay a graph")

    class Runner:
        def prepare_inputs(self, schedule, state, descriptor):
            calls.append("prepare_inputs")
            return SimpleNamespace(input_ids=actual_inputs)

        def execute_model(self, schedule):
            self.prepare_inputs(schedule, None, object())
            calls.append("metadata")
            if defect == "omitted":
                return
            if defect == "replaced_inner":
                model.language_model.model.forward = lambda value: value
            self.hidden = model(actual_inputs)
            assert self.hidden is actual_inputs
            if defect == "duplicate":
                model(actual_inputs)

        def sample(self):
            assert model.compute_logits(self.hidden) is actual_inputs
            calls.append("native_logits")
            return SimpleNamespace(sampled_token_ids=SampledTokens()), None, None

        def sample_tokens(self):
            return self.sample()

    class Trace:
        rank = 0
        graph_execution = None

        def __init__(
            self,
            runner,
            output,
            provenance,
            manifest,
            *,
            graph_calibration=False,
            serving_none=False,
            serving_none_measured=False,
        ):
            assert serving_none and not graph_calibration
            assert serving_none_measured is measured
            assert (manifest is not None) is calibration
            self.none_witness = none.NativeNoneModelWitness(runner.model)
            self.none_execution = (
                SimpleNamespace(boundary=lambda value: calls.append("source_scope_" + value), abort=lambda error: None)
                if measured and calibration
                else None
            )

        def before(self, *args, **kwargs):
            calls.append("metadata_window_start")
            return {**none_record(), "stage": "measure", "requests": [{}]}, {}

        def mark_none_boundary(self):
            calls.append("raw_model_boundary")

        def append(self, name, row):
            assert name == "failed"
            failed.append(row)

        def after(self, record, completed):
            assert record["native_none_forward_completed"]
            assert not record["native_graph_replay_completed"]
            assert record["requests"][0]["sampled_token_id"] == 7
            calls.append("completed")

    monkeypatch.setenv("AISIM_GLM53_SERVING_NONE_DIAGNOSTIC", "0" if measured else "1")
    monkeypatch.setenv("AISIM_GLM53_SERVING_NONE_MEASURED", "1" if measured else "0")
    monkeypatch.setenv("AISIM_GLM53_PIECEWISE_REPLAY", "0")
    monkeypatch.setenv("AISIM_GLM53_PIECEWISE_CAPTURE_ONLY", "0")
    monkeypatch.setenv("AISIM_GLM53_PURPOSE", "ops_graph" if calibration else "ops_graph_holdout")
    monkeypatch.setenv("AISIM_GLM53_TRACE_DIR", str(tmp_path))
    provenance = tmp_path / "provenance.json"
    provenance.write_text('{"backend_version":"0.30.0"}')
    monkeypatch.setenv("AISIM_GLM53_PROVENANCE", str(provenance))
    if calibration:
        manifest = tmp_path / "manifest.json"
        manifest.write_text('{"TEST_ONLY":true}')
        monkeypatch.setenv("AISIM_GLM53_OPS_MANIFEST", str(manifest))
    else:
        monkeypatch.delenv("AISIM_GLM53_OPS_MANIFEST", raising=False)
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.30.0")
    monkeypatch.setitem(sys.modules, "vllm.forward_context", SimpleNamespace(get_forward_context=lambda: None))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", SimpleNamespace(GPUModelRunner=Runner))
    monkeypatch.setattr(
        sys.modules["vllm.v1.worker.gpu.cudagraph_utils"], "ModelCudaGraphManager", Manager, raising=False
    )
    monkeypatch.setattr(graph, "install", lambda *args, **kwargs: pytest.fail("NONE cannot install graph op hooks"))
    monkeypatch.setattr(graph, "install_holdout_capture", lambda *args: calls.append("independent_policy"))
    monkeypatch.setattr(graph, "holdout_policy", lambda *args: {"backend_version": "0.30.0", "tp_rank": 0})
    monkeypatch.setattr(runtime, "_TraceState", Trace)
    runtime.install_v2()
    runner = Runner()
    runner.model, runner.cudagraph_manager, runner._aisim_glm53_ops_serving_ready = model, Manager(), True
    schedule = SimpleNamespace(total_num_scheduled_tokens=8)
    if defect:
        with pytest.raises(RuntimeError, match="serving NONE"):
            runner.execute_model(schedule)
        assert len(failed) == 1 and "completed" not in calls
    else:
        runner.execute_model(schedule)
        assert "native_logits" not in calls
        runner.sample_tokens()
        assert calls == [
            "independent_policy",
            "prepare_inputs",
            "metadata_window_start",
            "metadata",
            "raw_model_boundary",
            *(["source_scope_model"] if measured and calibration else []),
            "raw_model_boundary",
            *(["source_scope_before_logits"] if measured and calibration else []),
            "native_logits",
            "completed",
        ]
