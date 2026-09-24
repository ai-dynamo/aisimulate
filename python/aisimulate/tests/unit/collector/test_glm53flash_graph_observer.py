# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native graph evidence must survive padding, replay and logits boundaries."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from collector.glm53flash_graph_observer import NativeFullGraphWindow


@dataclass(frozen=True)
class Descriptor:
    cg_mode: str = "FULL"
    num_tokens: int = 4
    num_reqs: int = 4


class Event:
    def __init__(self, cuda):
        self.cuda = cuda

    def record(self, stream):
        assert stream == self.cuda.stream
        self.time = self.cuda.time

    def elapsed_time(self, other):
        return other.time - self.time


class Cuda:
    def __init__(self):
        self.stream, self.time, self.capturing = 1, 0, False

    def current_stream(self):
        return self.stream

    def is_current_stream_capturing(self):
        return self.capturing

    def Event(self, *, enable_timing):  # noqa: N802 - native torch.cuda API
        assert enable_timing
        return Event(self)

    def synchronize(self):
        pass


def setup():
    cuda = Cuda()
    observer = NativeFullGraphWindow(SimpleNamespace(cuda=cuda))
    graph = SimpleNamespace(replay=lambda: None)
    row = {
        "native_mode": "FULL",
        "actual_padded_tokens": 4,
        "requests": [{"request_id": str(i), "query": 1, "is_prefilling": False} for i in range(3)],
    }
    return cuda, observer, graph, row


def advance(cuda, value):
    cuda.time += value
    return "original_result"


def test_actual_padded_graph_and_logits_exclude_later_sampling_time():
    cuda, observer, graph, row = setup()
    descriptor = Descriptor()
    observer.arm(row, descriptor, graph)
    assert observer.dispatch(descriptor, graph, advance, cuda, 5) == "original_result"
    assert observer.logits(advance, cuda, 2) == "original_result"
    advance(cuda, 123)
    row.update(model_execute_returned=True, gpu_completed=True)
    for item in row["requests"]:
        item["sampled_token_id"] = 99
    result = observer.finish()
    assert result["whole_forward_gpu_ms"] == 7
    assert result["native_graph_descriptor"]["num_tokens"] == 4
    assert len(result["requests"]) == 3
    assert result["constituent_operation_coverage"] == "NOT_EVALUATED"
    assert observer.pending is None


@pytest.mark.parametrize("mode", ["NONE", "PIECEWISE"])
def test_non_full_dispatch_is_not_relabeled(mode):
    _, observer, graph, row = setup()
    with pytest.raises(ValueError, match="actual FULL"):
        observer.arm(row, Descriptor(cg_mode=mode), graph)


@pytest.mark.parametrize("error", ["padding", "duplicate", "prefill", "tokens"])
def test_actual_native_coordinates_must_match_graph(error):
    _, observer, graph, row = setup()
    if error == "padding":
        row["actual_padded_tokens"] = 3
    elif error == "duplicate":
        row["requests"][1]["request_id"] = "0"
    elif error == "prefill":
        row["requests"][0]["is_prefilling"] = True
    else:
        row["requests"][0]["query"] = 2
    with pytest.raises(ValueError, match="geometry"):
        observer.arm(row, Descriptor(), graph)


@pytest.mark.parametrize("error", ["recapture", "key", "capture", "duplicate", "stream"])
def test_replay_identity_cannot_change(error):
    cuda, observer, graph, row = setup()
    desc = Descriptor()
    observer.arm(row, desc, graph)
    if error == "recapture":
        graph = SimpleNamespace(replay=lambda: None)
    elif error == "key":
        desc = Descriptor(num_tokens=8)
    elif error == "capture":
        cuda.capturing = True
    elif error == "duplicate":
        observer.dispatch(desc, graph, advance, cuda, 5)
    with pytest.raises(RuntimeError):
        observer.dispatch(
            desc, graph, (lambda: setattr(cuda, "stream", 2)) if error == "stream" else lambda: advance(cuda, 5)
        )


def test_missing_logits_or_original_gpu_completion_rejected():
    cuda, observer, graph, row = setup()
    observer.arm(row, Descriptor(), graph)
    observer.dispatch(Descriptor(), graph, advance, cuda, 5)
    with pytest.raises(RuntimeError, match="logits endpoint"):
        observer.finish()
    observer.logits(advance, cuda, 2)
    with pytest.raises(RuntimeError, match="completed original"):
        observer.finish()
