# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Sweeper result schema, serialization, and search integration."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import aisimulate.sweeper.search as search_module
from aisimulate.sweeper import (
    RESULT_SCHEMA_VERSION,
    CandidateProvenance,
    CandidateRecord,
    CandidateRetention,
    CandidateStatus,
    OperationProvenance,
    ReasonCategory,
    ReplayReport,
    ResultViews,
    RunnerCapabilities,
    SearchStrategy,
    SmartSearchConfig,
    SweepCounts,
    Sweeper,
    SweepResult,
    SweepRunProvenance,
)
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.sampler import Suggestion
from aisimulate.sweeper.search_space import BranchSpace


def _config(*, target: str = "throughput") -> SmartSearchConfig:
    return SmartSearchConfig(
        search_space={
            "model_name": "example/model",
            "hardware_sku": "h200_sxm",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 8,
        },
        workload={
            "isl": 1024,
            "osl": 128,
            "concurrency": 8,
            "num_request_ratio": 10,
        },
        goal={"target": target},
        sweep={
            "max_rounds": 1,
            "candidates_per_round": 2,
            "parallel_evals": 1,
        },
    )


def _provenance() -> CandidateProvenance:
    return CandidateProvenance(
        model="example/model",
        hardware="h200_sxm",
        backend="trtllm",
        backend_version="1.0",
        performance_data=[{"source": "parquet", "revision": "abc123"}],
        topology={"deployment_mode": "agg", "tp": 4, "replicas": 2},
        workload={"concurrency": 8},
        objective={"target": "throughput"},
        power={"mean_power_w": 500.0},
        operations=[
            OperationProvenance(
                operation="attention",
                source="silicon",
                version="v1",
            )
        ],
        runner_metadata={"runner": "deterministic"},
    )


def _record(
    candidate_id: str,
    status: CandidateStatus,
    *,
    reason_category: ReasonCategory | None = None,
) -> CandidateRecord:
    feasible = status is CandidateStatus.FEASIBLE
    return CandidateRecord(
        candidate_id=candidate_id,
        status=status,
        config={"deployment_mode": "agg", "backend": "trtllm", "used_gpus": 8},
        used_gpus=8,
        score=100.0 if feasible else None,
        metrics={"output_throughput_tok_s": 100.0} if feasible else {},
        reason_category=reason_category,
        reason=None if feasible else f"example {status.value}",
        provenance=_provenance(),
    )


def _complete_result() -> SweepResult:
    records = [
        _record("candidate-000001", CandidateStatus.FEASIBLE),
        _record(
            "candidate-000002",
            CandidateStatus.INFEASIBLE,
            reason_category=ReasonCategory.GPU_BUDGET,
        ),
        _record(
            "candidate-000003",
            CandidateStatus.UNSUPPORTED,
            reason_category=ReasonCategory.BACKEND_TOPOLOGY,
        ),
        _record(
            "candidate-000004",
            CandidateStatus.TIMED_OUT,
            reason_category=ReasonCategory.RUNTIME_TIMEOUT,
        ),
        _record(
            "candidate-000005",
            CandidateStatus.FAILED,
            reason_category=ReasonCategory.REPLAY_RUNTIME,
        ),
    ]
    return SweepResult(
        counts=SweepCounts(
            evaluated=4,
            feasible=1,
            infeasible=1,
            unsupported=1,
            timed_out=1,
            failed=1,
            cache_hits=2,
        ),
        candidates=records,
        views=ResultViews(top_n=["candidate-000001"]),
        provenance=SweepRunProvenance(
            search_strategy=SearchStrategy.EXHAUSTIVE,
            implementation="legacy-aiconfigurator-adapter",
            implementation_version="0.12.0",
            run_id="test-run",
            created_at=datetime(2026, 8, 19, tzinfo=UTC),
            input_fingerprint="sha256:example",
            config=_config().model_dump(mode="json"),
        ),
    )


def test_result_json_round_trip_is_lossless_and_schema_versioned():
    result = _complete_result()

    payload = result.to_json(indent=None)
    decoded = json.loads(payload)

    assert decoded["schema_version"] == RESULT_SCHEMA_VERSION
    assert decoded["provenance"]["search_strategy"] == "exhaustive"
    assert SweepResult.from_json(payload) == result
    assert SweepResult.model_json_schema()["properties"]["schema_version"]["const"] == "1.0"


def test_result_rejects_unknown_schema_version_and_inconsistent_counts():
    payload = json.loads(_complete_result().to_json())
    payload["schema_version"] = "2.0"
    with pytest.raises(ValidationError, match="Input should be '1.0'"):
        SweepResult.model_validate(payload)

    payload = json.loads(_complete_result().to_json())
    payload["counts"]["failed"] = 0
    payload["counts"]["evaluated"] = 3
    with pytest.raises(ValidationError, match="retained failed candidates"):
        SweepResult.model_validate(payload)

    payload = json.loads(_complete_result().to_json())
    payload["views"]["top_n"] = ["candidate-000002"]
    with pytest.raises(ValidationError, match="non-feasible candidates"):
        SweepResult.model_validate(payload)


def test_flat_csv_is_one_row_per_candidate_with_canonical_json_cells():
    result = _complete_result()

    rows = list(csv.DictReader(io.StringIO(result.to_csv())))

    assert len(rows) == 5
    assert rows[0]["schema_version"] == "1.0"
    assert rows[0]["is_top_n"] == "True"
    assert json.loads(rows[0]["config_json"])["backend"] == "trtllm"
    assert json.loads(rows[0]["provenance_json"])["operations"][0]["source"] == "silicon"
    assert rows[3]["reason_category"] == "runtime_timeout"


class _Sampler:
    def __init__(self, branch, study_id, objectives=None):
        del study_id, objectives
        self.branch = branch

    def suggest(self, count):
        return [
            Suggestion(
                selection={
                    "deployment_mode": "agg",
                    "backend": "trtllm",
                    "agg_max_num_batched_tokens": 8192,
                    "agg_max_num_seqs": 256 * (index + 1),
                },
                parallel_config=self.branch.parallel_configs[0],
                handle=index,
            )
            for index in range(count)
        ]

    def observe(self, suggestion, metrics):
        del suggestion, metrics

    def observe_infeasible(self, suggestion, reason):
        del suggestion, reason


class _Runner:
    def run(self, spec):
        max_num_seqs = spec.backend_deployment.agg_engine_args["max_num_seqs"]
        return ReplayReport(
            metrics={
                "output_throughput_tok_s": float(max_num_seqs),
                "mean_power_w": 400.0,
            },
            metadata={
                "performance_data": [{"source": "parquet", "revision": "abc123"}],
                "operations": [
                    {
                        "operation": "attention",
                        "source": "silicon",
                        "version": "v1",
                    }
                ],
            },
        )

    def close(self):
        pass


class _RunnerFactory:
    def capabilities(self):
        return RunnerCapabilities(supported_backend_topologies=(("*", "*"),))

    def create(self, worker_id):
        del worker_id
        return _Runner()


class _UnsupportedThenFailedSampler:
    def __init__(self, branch, study_id, objectives=None):
        del study_id, objectives
        self.branch = branch
        self.calls = 0

    def suggest(self, count):
        del count
        self.calls += 1
        if self.calls > 2:
            return []
        backend = "vllm" if self.calls == 1 else "trtllm"
        selection = {
            "deployment_mode": "agg",
            "backend": backend,
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        }
        return [
            Suggestion(
                selection=selection,
                parallel_config=self.branch.parallel_configs[0],
                handle=self.calls,
            )
        ]

    def observe(self, suggestion, metrics):
        del suggestion, metrics

    def observe_infeasible(self, suggestion, reason):
        del suggestion, reason


class _FailingRunner:
    def run(self, spec):
        del spec
        raise RuntimeError("runtime unavailable")

    def close(self):
        pass


class _FailingRunnerFactory(_RunnerFactory):
    def create(self, worker_id):
        del worker_id
        return _FailingRunner()


def test_optimizer_guided_run_emits_complete_ledger_and_top_n(monkeypatch):
    parallel_config = ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2)
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(parallel_config,),
        supported_backends={parallel_config: frozenset({"trtllm"})},
        knob_choices={"backend": ["trtllm"]},
    )
    monkeypatch.setattr(
        search_module,
        "enumerate_branches",
        lambda config, *, max_seq_len=None, runner_capabilities=None: [branch],
    )
    monkeypatch.setattr(
        search_module,
        "resolve_backend_version",
        lambda hardware, backend: "1.0",
    )

    result = Sweeper(
        runner_factory=_RunnerFactory(),
        sampler_factory=_Sampler,
        show_progress=False,
    ).run_result(_config(), top_n=1)

    assert result.provenance.search_strategy is SearchStrategy.OPTIMIZER_GUIDED
    assert result.counts == SweepCounts(
        evaluated=2,
        feasible=2,
        infeasible=0,
        unsupported=0,
        timed_out=0,
        failed=0,
        cache_hits=0,
    )
    assert len(result.candidates) == 2
    assert len(result.views.top_n) == 1
    assert result.selected_candidates[0].score == 512.0
    assert result.candidates[0].provenance.performance_data[0]["source"] == "parquet"
    assert result.candidates[0].provenance.power["mean_power_w"] == 400.0

    views_only = Sweeper(
        runner_factory=_RunnerFactory(),
        sampler_factory=_Sampler,
        show_progress=False,
    ).run_result(
        _config(),
        top_n=1,
        candidate_retention=CandidateRetention.VIEWS,
    )
    assert views_only.counts.feasible == 2
    assert len(views_only.candidates) == 1
    assert views_only.selected_candidates[0].score == 512.0


def test_optimizer_guided_result_separates_unsupported_and_runtime_failure(monkeypatch):
    parallel_config = ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2)
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(parallel_config,),
        supported_backends={parallel_config: frozenset({"trtllm"})},
        knob_choices={"backend": ["trtllm"]},
    )
    monkeypatch.setattr(
        search_module,
        "enumerate_branches",
        lambda config, *, max_seq_len=None, runner_capabilities=None: [branch],
    )
    monkeypatch.setattr(
        search_module,
        "resolve_backend_version",
        lambda hardware, backend: "1.0",
    )

    result = Sweeper(
        runner_factory=_FailingRunnerFactory(),
        sampler_factory=_UnsupportedThenFailedSampler,
        show_progress=False,
    ).run_result(_config())

    assert result.counts == SweepCounts(
        evaluated=1,
        feasible=0,
        infeasible=0,
        unsupported=1,
        timed_out=0,
        failed=1,
        cache_hits=0,
    )
    assert [record.status for record in result.candidates] == [
        CandidateStatus.UNSUPPORTED,
        CandidateStatus.FAILED,
    ]
    assert result.candidates[0].reason_category is ReasonCategory.BACKEND_TOPOLOGY
    assert result.candidates[1].reason_category is ReasonCategory.REPLAY_RUNTIME
    assert result.selected_candidates == []
