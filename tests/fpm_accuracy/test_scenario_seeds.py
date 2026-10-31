# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve measured self-benchmark identities and rank coordinates for scenario seeds."""

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_hf_dataset import CONFIGURATION_PATH, _build_dataset, _fpm_payload

from scripts.fpm_accuracy.exceptions import DataError, DependencyError
from scripts.fpm_accuracy.models import scenario_predictors as adapters


@pytest.fixture
def case(tmp_path):
    records = [
        {
            **_fpm_payload(counter=index, wall_time=(index + 1) / 1000),
            "worker_id": worker,
        }
        for index, worker in enumerate(["worker-a", "worker-b", "worker-a", "worker-b"])
    ]
    source = "\n".join(json.dumps(record) for record in records).encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "traffic.jsonl", source)],
    )
    return dataset.measurement_case(CONFIGURATION_PATH)


def self_benchmark(case, *, schema=6, dp=1, coordinate_system=None, row_dp=None, **changes):
    artifact = case.fpm_artifacts[0]
    row = {
        "workload_kind": "decode",
        "batch_size": 7,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 2048,
        "latency_ms": 12.5,
        "dp": dp if row_dp is None else row_dp,
        **changes,
    }
    pq.write_table(pa.Table.from_pylist([row]), artifact.local_path)
    metadata = {
        "schema_version": schema,
        "parquet_sha256": hashlib.sha256(artifact.local_path.read_bytes()).hexdigest(),
        "configuration_selector": {"dp": dp},
        "row_count": 1,
        "coordinate_system": coordinate_system,
    }
    artifact.local_metadata_path.write_text(json.dumps(metadata))
    artifact = replace(
        artifact,
        sha256=hashlib.sha256(artifact.local_path.read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(artifact.local_metadata_path.read_bytes()).hexdigest(),
        row_count=1,
    )
    return replace(case, fpm_artifacts=(artifact,)), artifact


def test_self_benchmark_uses_actual_rows_and_seeds_each_worker_without_reading_targets(
    case,
):
    case, artifact = self_benchmark(case)
    seeds = adapters.self_benchmark_iterations(case, artifact)
    assert len(seeds) == 2
    assert {seed.ranks[0].worker_id for seed in seeds} == {"worker-a", "worker-b"}
    for seed in seeds:
        assert seed.observed_time_ms == 12.5
        assert seed.ranks[0].scheduled.num_decode_requests == 7
        assert seed.ranks[0].scheduled.sum_decode_kv_tokens == 2048
        assert "wall_time" not in seed.prediction_input().rank_payloads[0]
        assert seed.ranks[0].extra_metadata == {}


@pytest.mark.parametrize("dp", [4, 8, 16])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_balanced_self_benchmark_repeats_rank_workloads_and_preserves_one_group_target(case, dp, phase):
    case = replace(case, worker_role="aggregated")
    case, artifact = self_benchmark(
        case,
        dp=dp,
        coordinate_system="iteration_totals_balanced_v1",
        partition_policy="balanced_v1",
        workload_kind=phase,
        total_prefill_tokens=13 if phase == "prefill" else 0,
        total_kv_read_tokens=29,
    )
    seeds = adapters.self_benchmark_iterations(case, artifact)
    assert len(seeds) == 2  # One measured table row per logical worker, not per DP rank.
    assert {seed.ranks[0].worker_id for seed in seeds} == {"worker-a", "worker-b"}
    for seed in seeds:
        assert [rank.dp_rank for rank in seed.ranks] == list(range(dp))
        assert len({rank.worker_id for rank in seed.ranks}) == 1
        assert [rank.wall_time_s for rank in seed.ranks] == [0.0125] + [0.0] * (dp - 1)
        assert seed.observed_time_ms == 12.5
        assert all("wall_time" not in rank for rank in seed.prediction_input().rank_payloads)
        for rank in seed.ranks:
            scheduled = rank.scheduled
            assert scheduled.num_prefill_requests == (7 if phase == "prefill" else 0)
            assert scheduled.sum_prefill_tokens == (13 if phase == "prefill" else 0)
            assert scheduled.sum_prefill_kv_tokens == (29 if phase == "prefill" else 0)
            assert scheduled.num_decode_requests == (7 if phase == "decode" else 0)
            assert scheduled.sum_decode_kv_tokens == (29 if phase == "decode" else 0)
            assert rank.source_metadata() == {
                "coordinate_system": "iteration_totals_balanced_v1",
                "partition_policy": "balanced_v1",
                "rank_workload_reconstruction": "replicated_table_coordinates",
                "target_scope": "recorded_aggregate_latency",
                "aggregate_label_carrier_dp_rank": 0,
                "individual_rank_latencies_recorded": False,
            }
    changed_targets = replace(
        case,
        observations=tuple(
            replace(
                observation,
                iteration=replace(
                    observation.iteration,
                    ranks=tuple(replace(rank, wall_time_s=987.0) for rank in observation.iteration.ranks),
                ),
            )
            for observation in case.observations
        ),
    )
    assert adapters.self_benchmark_iterations(changed_targets, artifact) == seeds


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_balanced_self_benchmark_features_match_canonical_replicated_collection(case, phase):
    trainer = pytest.importorskip("aisimulate_core.sdk.fpm_learned")
    case = replace(case, worker_role="aggregated")
    dp = 16
    case, artifact = self_benchmark(
        case,
        dp=dp,
        coordinate_system="iteration_totals_balanced_v1",
        partition_policy="balanced_v1",
        workload_kind=phase,
        total_prefill_tokens=13 if phase == "prefill" else 0,
        total_kv_read_tokens=29,
    )
    seed = adapters.self_benchmark_iterations(case, artifact)[0]
    # The collector's expected_scheduled contract checks these same point
    # coordinates on every rank. Different observed rank latencies do not
    # change features; only the recorded aggregate label is transported.
    canonical = [
        {
            "version": 1,
            "worker_id": "worker-a",
            "counter_id": 0,
            "dp_rank": rank,
            "wall_time": 0.0125 if rank == dp - 1 else 0.001,
            "scheduled_requests": {
                "num_prefill_requests": 7 if phase == "prefill" else 0,
                "sum_prefill_tokens": 13 if phase == "prefill" else 0,
                "sum_prefill_kv_tokens": 29 if phase == "prefill" else 0,
                "num_decode_requests": 7 if phase == "decode" else 0,
                "sum_decode_kv_tokens": 29 if phase == "decode" else 0,
            },
        }
        for rank in range(dp)
    ]
    actual = seed.tuning_payload()
    features = trainer.FEATURE_PRESETS["fpm5"]
    assert trainer.compute_features(actual) == trainer.compute_features(canonical)
    assert trainer.iteration_wall_ms(actual) == trainer.iteration_wall_ms(canonical) == 12.5
    assert trainer.build_dataset([actual], "aggregated", features) == trainer.build_dataset(
        [canonical], "aggregated", features
    )
    values = trainer.compute_features(actual)
    assert values["num_active_ranks"] == dp
    assert values["sum_prefill_tokens"] == (dp * 13 if phase == "prefill" else 0)
    assert values["sum_decode_kv_tokens"] == (dp * 29 if phase == "decode" else 0)
    assert values["max_rank_prefill_tokens"] == (13 if phase == "prefill" else 0)
    assert values["max_rank_decode_kv_tokens"] == (29 if phase == "decode" else 0)


@pytest.mark.parametrize("coordinates", [None, "unknown", "iteration_totals"])
def test_multi_rank_self_benchmark_requires_known_coordinate_semantics(case, coordinates):
    case, artifact = self_benchmark(case, dp=4, coordinate_system=coordinates, partition_policy="balanced_v1")
    with pytest.raises(DependencyError, match="coordinates"):
        adapters.self_benchmark_iterations(case, artifact)


@pytest.mark.parametrize("policy", [None, "unknown", "uneven_v1"])
def test_multi_rank_self_benchmark_requires_declared_replication_policy(case, policy):
    case, artifact = self_benchmark(
        case, dp=4, coordinate_system="iteration_totals_balanced_v1", partition_policy=policy
    )
    with pytest.raises(DependencyError, match="partition policy"):
        adapters.self_benchmark_iterations(case, artifact)


@pytest.mark.parametrize("dp", [True, 0, -1, 4.0, "4"])
def test_self_benchmark_rejects_invalid_metadata_dp(case, dp):
    case, artifact = self_benchmark(case, dp=dp)
    with pytest.raises(DataError, match="invalid attention DP"):
        adapters.self_benchmark_iterations(case, artifact)


@pytest.mark.parametrize("row_dp", [True, 0, -1, 4.0, "4", 8])
def test_multi_rank_self_benchmark_rejects_invalid_or_mismatched_row_dp(case, row_dp):
    case, artifact = self_benchmark(
        case, dp=4, row_dp=row_dp, coordinate_system="iteration_totals_balanced_v1", partition_policy="balanced_v1"
    )
    with pytest.raises(DataError, match="dp|DP"):
        adapters.self_benchmark_iterations(case, artifact)


@pytest.mark.parametrize("options", [{"schema": 7}, {"dp": 2}])
def test_self_benchmark_rejects_unrepresented_execution_or_rank_identity(case, options):
    case, artifact = self_benchmark(case, **options)
    with pytest.raises(DependencyError):
        adapters.self_benchmark_iterations(case, artifact)


@pytest.mark.parametrize("changes", [{"latency_ms": 0.0}, {"batch_size": True}, {"total_prefill_tokens": 1}])
def test_self_benchmark_rejects_corrupt_training_targets_or_coordinates(case, changes):
    case, artifact = self_benchmark(case, **changes)
    with pytest.raises(DataError):
        adapters.self_benchmark_iterations(case, artifact)


def test_self_benchmark_rechecks_pinned_bytes(case):
    case, artifact = self_benchmark(case)
    artifact.local_path.write_bytes(b"different parquet")
    with pytest.raises(DataError, match="changed after"):
        adapters.self_benchmark_iterations(case, artifact)


def test_self_benchmark_rejects_substituted_metadata_pair(case):
    case, artifact = self_benchmark(case)
    replaced = replace(artifact, metadata_sha256="a" * 64)
    with pytest.raises(DataError, match="not a member"):
        adapters.self_benchmark_iterations(case, replaced)


def test_self_benchmark_metadata_must_bind_exact_table(case):
    case, artifact = self_benchmark(case)
    metadata = json.loads(artifact.local_metadata_path.read_text())
    metadata["parquet_sha256"] = "a" * 64
    artifact.local_metadata_path.write_text(json.dumps(metadata))
    artifact = replace(artifact, metadata_sha256=hashlib.sha256(artifact.local_metadata_path.read_bytes()).hexdigest())
    case = replace(case, fpm_artifacts=(artifact,))
    with pytest.raises(DataError, match="does not bind"):
        adapters.self_benchmark_iterations(case, artifact)


def test_single_rank_self_benchmark_rejects_mismatched_row_dp(case):
    case, artifact = self_benchmark(case, dp=1, row_dp=2)
    with pytest.raises(DataError, match="DP differs"):
        adapters.self_benchmark_iterations(case, artifact)
