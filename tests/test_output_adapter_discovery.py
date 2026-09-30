# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from aisimulate.output_adapter import (
    OUTPUT_ADAPTER_API_VERSION,
    OutputAdapterExecutionError,
    OutputAdapterResolutionError,
    RecommendationOutputCallbacks,
    resolve_output_adapters,
    resolve_output_callbacks,
    write_output_adapters,
)
from aisimulate.sweeper.config import Candidate
from aisimulate.sweeper.result import CandidateProvenance, CandidateRecord, CandidateStatus


class _Adapter:
    api_version = OUTPUT_ADAPTER_API_VERSION

    def __init__(self, name: str) -> None:
        self.name = name

    def write(self, config, *, result, output_dir):
        del config, result
        path = Path("artifact.txt")
        (output_dir / path).write_text("artifact\n")
        return [path]


@dataclass
class _EntryPoint:
    name: str
    value: str
    provider: object
    loads: int = 0

    def load(self):
        self.loads += 1
        if isinstance(self.provider, Exception):
            raise self.provider
        return self.provider


def test_only_selected_output_adapter_is_loaded() -> None:
    selected = _EntryPoint("dgd", "example:create_dgd", lambda: _Adapter("dgd"))
    unused = _EntryPoint("chart", "example:create_chart", lambda: _Adapter("chart"))

    resolved = resolve_output_adapters(["dgd"], entry_points=[selected, unused])

    assert list(resolved) == ["dgd"]
    assert selected.loads == 1
    assert unused.loads == 0


def test_missing_output_adapter_lists_available_names() -> None:
    available = _EntryPoint("chart", "example:create_chart", lambda: _Adapter("chart"))

    with pytest.raises(OutputAdapterResolutionError, match="dgd.*installed adapters: chart"):
        resolve_output_adapters(["dgd"], entry_points=[available])

    assert available.loads == 0


@pytest.mark.parametrize(
    ("adapter", "message"),
    [
        (_Adapter("wrong"), "returned name"),
        (type("OldAdapter", (_Adapter,), {"api_version": 0})("dgd"), "API version"),
        (type("IncompleteAdapter", (), {"name": "dgd", "api_version": OUTPUT_ADAPTER_API_VERSION})(), "write"),
    ],
)
def test_invalid_output_adapter_abi_is_rejected(adapter, message) -> None:
    with pytest.raises(OutputAdapterResolutionError, match=message):
        resolve_output_adapters(["dgd"], injected={"dgd": adapter}, entry_points=[])


def test_reported_artifacts_must_be_relative_existing_paths(tmp_path) -> None:
    class InvalidPathAdapter(_Adapter):
        def write(self, config, *, result, output_dir):
            del config, result, output_dir
            return ["../outside"]

    with pytest.raises(OutputAdapterExecutionError, match="outside the output directory"):
        write_output_adapters(
            {"dgd": InvalidPathAdapter("dgd")},
            {"dgd": {}},
            result=object(),
            output_dir=tmp_path,
        )


def test_output_adapter_writes_into_supplied_directory(tmp_path) -> None:
    artifacts = write_output_adapters(
        {"dgd": _Adapter("dgd")},
        {"dgd": {"name": "example"}},
        result=object(),
        output_dir=tmp_path,
    )

    assert artifacts == {"dgd": (Path("artifact.txt"),)}
    assert (tmp_path / "artifact.txt").read_text() == "artifact\n"


def test_output_adapter_subscribes_to_live_recommendation_callbacks() -> None:
    received = []

    class SubscribedAdapter(_Adapter):
        def subscribe(self, config):
            assert config == {"channel": "private"}
            return RecommendationOutputCallbacks(
                on_candidate=lambda record: received.append(("candidate", record)),
                on_round=lambda round_no, candidates: received.append(("round", round_no, candidates)),
            )

    callbacks = resolve_output_callbacks(
        {"dgd": {"channel": "private"}},
        injected={"dgd": SubscribedAdapter("dgd")},
        entry_points=[],
    )
    record = CandidateRecord(
        candidate_id="candidate-000001",
        status=CandidateStatus.FEASIBLE,
        config={"backend": "vllm"},
        used_gpus=1,
        score=1.0,
        provenance=CandidateProvenance(model="example/model", hardware="h200_sxm"),
    )
    candidate = Candidate(config={"backend": "vllm"}, used_gpus=1, score=1.0, metrics={})

    assert callbacks.on_candidate is not None
    callbacks.on_candidate(record)
    assert callbacks.on_round is not None
    callbacks.on_round(2, [candidate])

    assert received == [("candidate", record), ("round", 2, [candidate])]


def test_output_adapter_callbacks_are_isolated_between_subscribers() -> None:
    observed = []

    class MutatingAdapter(_Adapter):
        def subscribe(self, config):
            del config

            def mutate(record):
                record.config["nested"]["value"] = 2

            return RecommendationOutputCallbacks(on_candidate=mutate)

    class ObservingAdapter(_Adapter):
        def subscribe(self, config):
            del config
            return RecommendationOutputCallbacks(
                on_candidate=lambda record: observed.append(record.config["nested"]["value"])
            )

    callbacks = resolve_output_callbacks(
        {"mutating": {}, "observing": {}},
        injected={
            "mutating": MutatingAdapter("mutating"),
            "observing": ObservingAdapter("observing"),
        },
        entry_points=[],
    )
    record = CandidateRecord(
        candidate_id="candidate-000001",
        status=CandidateStatus.FEASIBLE,
        config={"nested": {"value": 1}},
        used_gpus=1,
        score=1.0,
        provenance=CandidateProvenance(model="example/model", hardware="h200_sxm"),
    )

    assert callbacks.on_candidate is not None
    callbacks.on_candidate(record)

    assert observed == [1]
    assert record.config["nested"]["value"] == 1


def test_output_adapter_callback_failure_identifies_adapter_and_event() -> None:
    class FailingAdapter(_Adapter):
        def subscribe(self, config):
            del config

            def fail(record):
                del record
                raise ValueError("closed")

            return RecommendationOutputCallbacks(on_candidate=fail)

    callbacks = resolve_output_callbacks(
        {"dgd": {}},
        injected={"dgd": FailingAdapter("dgd")},
        entry_points=[],
    )
    record = CandidateRecord(
        candidate_id="candidate-000001",
        status=CandidateStatus.FEASIBLE,
        config={},
        used_gpus=1,
        score=1.0,
        provenance=CandidateProvenance(model="example/model", hardware="h200_sxm"),
    )

    assert callbacks.on_candidate is not None
    with pytest.raises(OutputAdapterExecutionError, match="dgd.*on_candidate.*ValueError: closed"):
        callbacks.on_candidate(record)
