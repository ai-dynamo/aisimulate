# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical Sweeper result schema, serialization, and search integration."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import aisimulate.sweeper.search as search_module
from aisimulate.sweeper import (
    RESULT_SCHEMA_VERSION,
    CandidateProvenance,
    CandidateRecord,
    CandidateRetention,
    CandidateStatus,
    LoadRecommendationRecord,
    LoadRecommendationView,
    LoadTarget,
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


class _StrictV1Envelope(BaseModel):
    """Approximate the original strict v1.0 envelope for compatibility checks."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"]
    candidate_retention: CandidateRetention
    counts: dict
    candidates: list[dict]
    views: dict
    provenance: dict


def _partial_load_payload() -> dict:
    payload = _complete_result().model_dump(mode="json")
    payload["load_recommendation"] = {
        "target": {"request_rate": 15, "max_gpus": 8, "allow_partial": True},
        "recommendations": [
            {
                "candidate_id": "candidate-000001",
                "capacity_per_replica": 10,
                "capacity_per_gpu": 1.25,
                "replicas_needed": 2,
                "total_gpus_needed": 16,
                "deployed_replicas": 1,
                "deployed_gpus": 8,
                "supported_load": 10,
                "load_served_pct": 1000 / 15,
                "limiting_role": "decode",
                "partial": True,
            }
        ],
    }
    return payload


def test_result_json_round_trip_is_lossless_and_schema_versioned():
    result = _complete_result()

    payload = result.to_json(indent=None)
    decoded = json.loads(payload)

    assert decoded["schema_version"] == RESULT_SCHEMA_VERSION
    assert decoded["provenance"]["search_strategy"] == "exhaustive"
    assert SweepResult.from_json(payload) == result
    version_schema = SweepResult.model_json_schema()["properties"]["schema_version"]
    assert version_schema["default"] == "1.1"
    assert version_schema["enum"] == ["1.0", "1.1"]


def test_v11_reader_round_trips_v10_without_breaking_a_strict_v10_reader():
    original = _complete_result().model_copy(update={"schema_version": "1.0"})

    payload = original.to_json(indent=None)
    decoded = json.loads(payload)

    assert "load_recommendation" not in decoded
    assert _StrictV1Envelope.model_validate_json(payload).schema_version == "1.0"
    assert SweepResult.from_json(payload) == original
    assert json.loads(SweepResult.from_json(payload).to_json(indent=None)) == decoded
    assert next(csv.reader(io.StringIO(original.to_csv()))) == [
        "schema_version",
        "candidate_id",
        "status",
        "reason_category",
        "reason",
        "used_gpus",
        "score",
        "config_json",
        "metrics_json",
        "objectives_json",
        "provenance_json",
        "is_top_n",
        "is_pareto",
    ]


def test_v10_payload_cannot_masquerade_with_v11_load_recommendation():
    payload = _complete_result().model_dump(mode="json")
    payload["schema_version"] = "1.0"
    payload["load_recommendation"] = {
        "target": {"request_rate": 10},
        "no_feasible_reasons": [{"reason": "capacity unavailable"}],
    }

    with pytest.raises(ValidationError, match="requires schema_version='1.1'"):
        SweepResult.model_validate(payload)


def test_result_json_round_trip_preserves_full_load_recommendation():
    payload = _complete_result().model_dump(mode="json")
    payload["load_recommendation"] = LoadRecommendationView(
        target=LoadTarget(request_rate=15, max_gpus=8, allow_partial=True),
        recommendations=[
            LoadRecommendationRecord(
                candidate_id="candidate-000001",
                capacity_per_replica=10,
                capacity_per_gpu=1.25,
                replicas_needed=2,
                total_gpus_needed=16,
                deployed_replicas=1,
                deployed_gpus=8,
                supported_load=10,
                load_served_pct=1000 / 15,
                limiting_role="decode",
                partial=True,
            )
        ],
    ).model_dump(mode="json")
    result = SweepResult.model_validate(payload)

    decoded = json.loads(result.to_json(indent=None))
    recommendation = decoded["load_recommendation"]["recommendations"][0]

    assert decoded["schema_version"] == RESULT_SCHEMA_VERSION
    assert recommendation == {
        "candidate_id": "candidate-000001",
        "capacity_per_replica": 10.0,
        "capacity_per_gpu": 1.25,
        "replicas_needed": 2,
        "total_gpus_needed": 16,
        "deployed_replicas": 1,
        "deployed_gpus": 8,
        "supported_load": 10.0,
        "load_served_pct": 1000 / 15,
        "limiting_role": "decode",
        "partial": True,
    }
    assert SweepResult.from_json(result.to_json()) == result


def test_result_rejects_unknown_schema_version_and_inconsistent_counts():
    payload = json.loads(_complete_result().to_json())
    payload["schema_version"] = "2.0"
    with pytest.raises(ValidationError, match="Input should be '1.0' or '1.1'"):
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


def test_retention_policies_enforce_status_counts_and_exact_view_membership():
    feasible_payload = _complete_result().model_dump(mode="json")
    feasible_payload["candidate_retention"] = "feasible"
    feasible_payload["candidates"] = feasible_payload["candidates"][:1]
    assert SweepResult.model_validate(feasible_payload).counts.feasible == 1

    failed_retained = json.loads(json.dumps(feasible_payload))
    failed_retained["candidates"].append(
        _record(
            "candidate-000005",
            CandidateStatus.FAILED,
            reason_category=ReasonCategory.REPLAY_RUNTIME,
        ).model_dump(mode="json")
    )
    with pytest.raises(ValidationError, match="can retain only feasible"):
        SweepResult.model_validate(failed_retained)

    missing_feasible = json.loads(json.dumps(feasible_payload))
    missing_feasible["candidates"] = []
    missing_feasible["views"] = {"top_n": [], "pareto_front": []}
    with pytest.raises(ValidationError, match="retain every feasible"):
        SweepResult.model_validate(missing_feasible)

    views_payload = _complete_result().model_dump(mode="json")
    views_payload["candidate_retention"] = "views"
    views_payload["candidates"] = views_payload["candidates"][:2]
    with pytest.raises(ValidationError, match="retained unreferenced"):
        SweepResult.model_validate(views_payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["load_recommendation"]["target"].update(
                allow_partial=False
            ),
            "requires target.allow_partial=true",
        ),
        (
            lambda payload: payload["load_recommendation"]["target"].pop(
                "max_gpus"
            ),
            "requires target.max_gpus",
        ),
        (
            lambda payload: payload["load_recommendation"]["target"].update(
                max_gpus=4
            ),
            "cannot exceed target.max_gpus",
        ),
        (
            lambda payload: payload["load_recommendation"]["recommendations"][
                0
            ].update(replicas_needed=3, total_gpus_needed=24),
            "uncapped minimum",
        ),
        (
            lambda payload: payload["load_recommendation"]["recommendations"][
                0
            ].update(partial=False),
            "partial must indicate",
        ),
    ],
)
def test_load_recommendation_rejects_inconsistent_partial_and_cap_payloads(
    mutate,
    message,
):
    payload = _partial_load_payload()
    mutate(payload)

    with pytest.raises(ValidationError, match=message):
        SweepResult.model_validate(payload)


def test_load_recommendation_failure_ids_must_be_retained_and_unique():
    payload = _complete_result().model_dump(mode="json")
    payload["load_recommendation"] = {
        "target": {"request_rate": 15},
        "no_feasible_reasons": [
            {
                "candidate_id": "candidate-000001",
                "reason": "capacity unavailable",
            }
        ],
    }
    assert SweepResult.model_validate(payload).load_recommendation is not None

    payload["load_recommendation"]["no_feasible_reasons"].append(
        {
            "candidate_id": "candidate-000001",
            "reason": "another reason",
        }
    )
    with pytest.raises(ValidationError, match="failure candidate IDs must be unique"):
        SweepResult.model_validate(payload)

    payload["load_recommendation"]["no_feasible_reasons"] = [
        {
            "candidate_id": "candidate-999999",
            "reason": "capacity unavailable",
        }
    ]
    with pytest.raises(ValidationError, match="not retained"):
        SweepResult.model_validate(payload)


def test_flat_csv_is_one_row_per_candidate_with_canonical_json_cells():
    result = _complete_result()

    rows = list(csv.DictReader(io.StringIO(result.to_csv())))

    assert len(rows) == 5
    assert rows[0]["schema_version"] == RESULT_SCHEMA_VERSION
    assert rows[0]["is_top_n"] == "True"
    assert json.loads(rows[0]["config_json"])["backend"] == "trtllm"
    assert json.loads(rows[0]["provenance_json"])["operations"][0]["source"] == "silicon"
    assert rows[3]["reason_category"] == "runtime_timeout"


def test_flat_csv_expands_load_target_and_recommendation_fields():
    payload = _complete_result().model_dump(mode="json")
    payload["load_recommendation"] = {
        "target": {"request_rate": 15, "max_gpus": 8, "allow_partial": True},
        "recommendations": [
            {
                "candidate_id": "candidate-000001",
                "capacity_per_replica": 10,
                "capacity_per_gpu": 1.25,
                "replicas_needed": 2,
                "total_gpus_needed": 16,
                "deployed_replicas": 1,
                "deployed_gpus": 8,
                "supported_load": 10,
                "load_served_pct": 1000 / 15,
                "limiting_role": "decode",
                "partial": True,
            }
        ],
    }

    rows = list(
        csv.DictReader(io.StringIO(SweepResult.model_validate(payload).to_csv()))
    )

    assert rows[0]["load_target_kind"] == "request_rate"
    assert rows[0]["load_target_value"] == "15.0"
    assert rows[0]["load_target_max_gpus"] == "8"
    assert rows[0]["load_target_allow_partial"] == "True"
    assert rows[0]["recommendation_total_gpus_needed"] == "16"
    assert rows[0]["recommendation_deployed_gpus"] == "8"
    assert rows[0]["recommendation_limiting_role"] == "decode"
    assert rows[0]["recommendation_partial"] == "True"
    assert rows[1]["recommendation_total_gpus_needed"] == ""


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
                "request_throughput_rps": float(max_num_seqs) / 32,
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


def test_run_result_preserves_mixed_full_and_partial_load_recommendations(monkeypatch):
    parallel_config = ReplicaParallelConfig(
        ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2
    )
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
    ).run_result(
        _config(),
        top_n=1,
        candidate_retention=CandidateRetention.VIEWS,
        load_target=LoadTarget(request_rate=20, max_gpus=16, allow_partial=True),
        recommendation_top_n=2,
    )

    assert result.counts.feasible == 2
    assert len(result.candidates) == 2
    assert result.load_recommendation is not None
    recommendations = result.load_recommendation.recommendations
    assert [item.partial for item in recommendations] == [False, True]
    assert recommendations[0].total_gpus_needed == 16
    assert recommendations[0].deployed_gpus == 16
    assert recommendations[0].load_served_pct == 100
    assert recommendations[1].total_gpus_needed == 24
    assert recommendations[1].deployed_gpus == 16
    assert recommendations[1].supported_load == 16
    assert recommendations[1].limiting_role == "agg"
    assert SweepResult.from_json(result.to_json()) == result


class _NoRequestCapacityRunner(_Runner):
    def run(self, spec):
        report = super().run(spec)
        report.metrics.pop("request_throughput_rps")
        return report


class _NoRequestCapacityRunnerFactory(_RunnerFactory):
    def create(self, worker_id):
        del worker_id
        return _NoRequestCapacityRunner()


def test_run_result_preserves_no_feasible_recommendation_reasons(monkeypatch):
    parallel_config = ReplicaParallelConfig(
        ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2
    )
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
        runner_factory=_NoRequestCapacityRunnerFactory(),
        sampler_factory=_Sampler,
        show_progress=False,
    ).run_result(
        _config(),
        candidate_retention=CandidateRetention.VIEWS,
        top_n=1,
        load_target=LoadTarget(request_rate=20),
    )

    assert result.counts.feasible == 2
    assert result.load_recommendation is not None
    assert result.load_recommendation.recommendations == []
    assert len(result.load_recommendation.no_feasible_reasons) == 2
    assert all(
        "does not expose positive finite request_rate capacity" in failure.reason
        for failure in result.load_recommendation.no_feasible_reasons
    )
    failure_ids = {
        failure.candidate_id
        for failure in result.load_recommendation.no_feasible_reasons
    }
    assert failure_ids == {candidate.candidate_id for candidate in result.candidates}
    rows = list(csv.DictReader(io.StringIO(result.to_csv())))
    assert json.loads(rows[0]["recommendation_no_feasible_reasons_json"]) == [
        failure.model_dump(mode="json")
        for failure in result.load_recommendation.no_feasible_reasons
    ]
    assert {
        row["candidate_id"]: row["recommendation_failure_reason"] for row in rows
    } == {
        failure.candidate_id: failure.reason
        for failure in result.load_recommendation.no_feasible_reasons
        if failure.candidate_id is not None
    }
    assert all(
        row["recommendation_no_feasible_reasons_json"]
        == rows[0]["recommendation_no_feasible_reasons_json"]
        for row in rows
    )
    assert SweepResult.from_json(result.to_json()) == result


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
    ).run_result(_config(), load_target=LoadTarget(request_rate=20))

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
    assert result.load_recommendation is not None
    assert [
        failure.candidate_id
        for failure in result.load_recommendation.no_feasible_reasons
    ] == ["candidate-000001", "candidate-000002"]
    assert "backend_topology" in result.load_recommendation.no_feasible_reasons[0].reason
    assert "runtime unavailable" in result.load_recommendation.no_feasible_reasons[1].reason
