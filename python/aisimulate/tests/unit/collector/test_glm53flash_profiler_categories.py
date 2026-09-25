# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY Kineto category regressions, not native GPU qualification."""

import copy

import pytest
from collector.glm53flash_graph_nodes import bind_execution_activity, bind_vllm_execution_activity
from collector.glm53flash_vllm_piecewise_activity import bind_piecewise_execution

from .test_glm53flash_graph_execution import fixture_events
from .test_glm53flash_vllm_graph_execution import trace
from .test_glm53flash_vllm_piecewise_activity import fixture

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("factory", "binder"),
    [
        (fixture_events, bind_execution_activity),
        (trace, bind_vllm_execution_activity),
        (fixture, bind_piecewise_execution),
    ],
)
@pytest.mark.parametrize("defect", [None, "gpu_only", "cpu_duplicate", "unknown_category"])
def test_same_named_gpu_annotations_are_not_cpu_launch_ownership(factory, binder, defect):
    capture, events = factory()
    expected = binder(copy.deepcopy(capture), copy.deepcopy(events))
    annotations = [row for row in events if row.get("cat") == "user_annotation"]
    # Actual Kineto614513 has one CPU annotation plus46 same-named device
    # stream ranges. Keep every GPU range without treating it as a CPU scope.
    events.extend(dict(row, cat="gpu_user_annotation", pid=0, tid=128 + i) for row in annotations for i in range(46))
    if defect == "gpu_only":
        events.remove(annotations[0])
    elif defect == "cpu_duplicate":
        events.append(dict(annotations[0]))
    elif defect == "unknown_category":
        annotations[0]["cat"] = "unknown_annotation"
    original = copy.deepcopy(events)
    if defect:
        with pytest.raises(ValueError):
            binder(capture, events)
    else:
        assert binder(capture, events) == expected
    assert events == original
