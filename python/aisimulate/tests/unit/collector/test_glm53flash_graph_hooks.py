# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture lifecycle regressions; fake CPU nodes are never calibration data."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from collector.glm53flash_graph_hooks import NativeGraphOperationObserver
from collector.glm53flash_sglang_graph_ops import finish_native_forward

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class ShapeKey:
    size: int


def fixture():
    captured = False
    nodes = {}
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_current_stream_capturing=lambda: captured,
            current_stream=lambda: SimpleNamespace(cuda_stream=1),
        )
    )
    api = SimpleNamespace(
        libraries={"TEST_ONLY": True},
        snapshot=lambda stream: {"capture_id": 1, "graph_id": 2, "nodes": dict(nodes), "edges": []},
    )
    manifest = {"phases": {phase: [{"name": "compute"}, {"name": "allreduce"}] for phase in ("context", "generation")}}
    observer = NativeGraphOperationObserver(manifest, {"TEST_ONLY": True}, 0, api, torch_module=torch)

    def capture(value):
        nonlocal captured
        captured = value

    class Native:
        def reduce(self, x):
            if captured:
                nodes[len(nodes) + 1] = {"node_type": 0}
            return x + 1

        def forward(self, x):
            if captured:
                nodes[len(nodes) + 1] = {"node_type": 0}
            return self.reduce(x) + 1

    native = Native()
    observer.wrap(native, "forward", "compute")
    observer.wrap_collective(native, "reduce", ("allreduce",))
    return observer, native, capture


def test_native_warmups_unchanged_then_one_complete_capture_without_replay_callbacks():
    observer, native, capture = fixture()
    assert native.forward(4) == 6
    assert observer.registry is None
    capture(True)
    observer.start(ShapeKey(4), torch_compile_enabled=False)
    assert native.forward(5) == 7
    registry = observer.finish()
    assert [row["name"] for row in registry["nodes"]] == ["compute", "allreduce"]
    assert registry["physical_padded_tokens"] == 4
    assert registry["formal_admission"] is False
    capture(False)
    observer.close()
    assert native.forward(6) == 8


def test_compiled_fusion_cannot_be_observed_with_python_capture_hooks():
    observer, _, capture = fixture()
    capture(True)
    with pytest.raises(RuntimeError, match="compiled fusion"):
        observer.start(ShapeKey(1), torch_compile_enabled=True)


def test_native_capture_without_complete_model_cannot_pass():
    observer, _, capture = fixture()
    capture(True)
    observer.start(ShapeKey(1), torch_compile_enabled=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        observer.finish()


@pytest.mark.parametrize("defect", [None, "completion", "mode", "sample", "pending"])
def test_native_graph_whole_window_requires_completed_actual_replay(defect):
    start = SimpleNamespace(elapsed_time=lambda end: 3.0)
    pending = {"start": start, "end": object(), "profiled": False, "ops_instrumented": False}
    runner = SimpleNamespace(_aisim_glm53_graph_pending={1: pending})
    record = {
        "stage": "measure",
        "invocation": 1,
        "gpu_completed": True,
        "runtime_mode": "FULL",
        "requests": [{"sampled_token_id": 5}],
    }
    if defect == "completion":
        record["gpu_completed"] = False
    elif defect == "mode":
        record["runtime_mode"] = "NONE"
    elif defect == "sample":
        record["requests"][0].pop("sampled_token_id")
    elif defect == "pending":
        runner._aisim_glm53_graph_pending.clear()
    if defect:
        with pytest.raises(RuntimeError):
            finish_native_forward(runner, record)
    else:
        result = finish_native_forward(runner, record)
        assert result["whole_forward_gpu_ms"] == 3.0
        assert result["whole_forward_boundary"] == "native_full_graph_metadata_to_logits_gpu_v1"
        assert result["ops_instrumented"] is False
        assert result["formal_admission"] is False
