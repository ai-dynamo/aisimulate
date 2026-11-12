# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve measured self-benchmark identities and rank coordinates for scenario seeds."""

import hashlib
import json
import random
from dataclasses import asdict, replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_hf_dataset import CONFIGURATION_PATH, _build_dataset, _fpm_payload

from scripts.fpm_accuracy.exceptions import DataError, DependencyError
from scripts.fpm_accuracy.models import scenario_predictors as adapters
from scripts.fpm_accuracy.scenarios import _seed_target_normalization_evidence


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
    pytest.importorskip("aisim._runtime")
    trainer = pytest.importorskip("aisim_core.sdk.fpm_learned")
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
        case,
        dp=4,
        coordinate_system="iteration_totals_balanced_v1",
        partition_policy=policy,
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
        case,
        dp=4,
        row_dp=row_dp,
        coordinate_system="iteration_totals_balanced_v1",
        partition_policy="balanced_v1",
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
    artifact = replace(
        artifact,
        metadata_sha256=hashlib.sha256(artifact.local_metadata_path.read_bytes()).hexdigest(),
    )
    case = replace(case, fpm_artifacts=(artifact,))
    with pytest.raises(DataError, match="does not bind"):
        adapters.self_benchmark_iterations(case, artifact)


def test_single_rank_self_benchmark_rejects_mismatched_row_dp(case):
    case, artifact = self_benchmark(case, dp=1, row_dp=2)
    with pytest.raises(DataError, match="DP differs"):
        adapters.self_benchmark_iterations(case, artifact)


def test_genuine_seed_targets_do_not_construct_a_normalization_model(case, monkeypatch):
    case, artifact = self_benchmark(case, kv_seed_regime="real_kv")

    def forbidden(*args, **kwargs):
        raise AssertionError("Genuine rows need no native target normalization.")

    monkeypatch.setattr(adapters, "create_scenario_predictor", forbidden)
    seeds = adapters.self_benchmark_iterations(case, artifact)
    assert all(seed.observed_time_ms == 12.5 for seed in seeds)
    assert _seed_target_normalization_evidence(seeds) is None


def two_phase_self_benchmark(case, *, prefill_fake=False):
    case, artifact = self_benchmark(case, kv_seed_regime="fake_fallback")
    decode = pq.read_table(artifact.local_path).to_pylist()[0]
    prefill = {
        **decode,
        "workload_kind": "prefill",
        "total_prefill_tokens": 128,
        "kv_seed_regime": "fake_fallback" if prefill_fake else "real_kv",
    }
    pq.write_table(pa.Table.from_pylist([prefill, decode]), artifact.local_path)
    digest = hashlib.sha256(artifact.local_path.read_bytes()).hexdigest()
    metadata = json.loads(artifact.local_metadata_path.read_text())
    metadata.update(parquet_sha256=digest, row_count=2)
    artifact.local_metadata_path.write_text(json.dumps(metadata))
    artifact = replace(
        artifact,
        sha256=digest,
        metadata_sha256=hashlib.sha256(artifact.local_metadata_path.read_bytes()).hexdigest(),
        row_count=2,
    )
    return replace(case, fpm_artifacts=(artifact,), worker_role="prefill"), artifact


def test_prefill_seeding_needs_no_native_model_for_discarded_fake_decode_rows(case, monkeypatch):
    case, artifact = two_phase_self_benchmark(case)
    original = artifact.local_path.read_bytes()

    def unavailable(*args, **kwargs):
        raise DependencyError("Native interpolation cannot prepare this profile.")

    monkeypatch.setattr(adapters, "create_scenario_predictor", unavailable)
    seeds = adapters.self_benchmark_iterations(case, artifact)
    assert len(seeds) == 2
    assert {seed.ranks[0].worker_id for seed in seeds} == {"worker-a", "worker-b"}
    for seed in seeds:
        assert seed.observed_time_ms == 12.5
        assert seed.ranks[0].scheduled.sum_prefill_tokens == 128
        assert seed.ranks[0].scheduled.num_decode_requests == 0
    assert _seed_target_normalization_evidence(seeds) is None
    assert artifact.local_path.read_bytes() == original


@pytest.mark.parametrize("omit_decode", [False, True])
def test_relevant_fake_prefill_still_validates_all_marked_source_rows(case, monkeypatch, omit_decode):
    from types import SimpleNamespace

    case, artifact = two_phase_self_benchmark(case, prefill_fake=True)
    rows = pq.read_table(artifact.local_path).to_pylist()
    records = [
        {
            "row_index": index,
            **{k: row[k] for k in ("workload_kind", "batch_size", "total_prefill_tokens", "total_kv_read_tokens")},
            "raw_latency_ms": row["latency_ms"],
            "effective_latency_ms": row["latency_ms"],
            "replaced": False,
        }
        for index, row in enumerate(rows)
        if not (omit_decode and row["workload_kind"] == "decode")
    ]
    report = {
        "policy": "same_cell_batch_real_kv_extrapolation_v1",
        "row_count": 2,
        "replacement_enabled": True,
        "replaced_count": 0,
        "kept_count": len(records),
        "rows": records,
    }
    closed = []
    monkeypatch.setattr(
        adapters,
        "create_scenario_predictor",
        lambda *a, **k: SimpleNamespace(
            fpm_seed_target_normalization=lambda: report, close=lambda: closed.append(True)
        ),
    )
    if omit_decode:
        with pytest.raises(DataError, match="omitted marked rows"):
            adapters.self_benchmark_iterations(case, artifact)
    else:
        seeds = adapters.self_benchmark_iterations(case, artifact)
        summary = _seed_target_normalization_evidence(seeds)
        assert summary["marked_source_rows"] == 1  # Only the selected phase becomes seed data.
        assert summary["kept_source_rows"] == 1
        assert summary["replicated_seed_rows"] == 2
    assert closed == [True]


@pytest.mark.parametrize(
    "enabled,replaced,effective",
    [(True, True, 21.0), (True, False, 12.5), (False, False, 12.5)],
)
@pytest.mark.parametrize("dp", [1, 4])
def test_fake_seed_targets_transport_native_labels_and_preserve_source(
    case, monkeypatch, enabled, replaced, effective, dp
):
    case, artifact = self_benchmark(
        case,
        kv_seed_regime="fake_fallback",
        dp=dp,
        coordinate_system="iteration_totals_balanced_v1",
        partition_policy="balanced_v1",
    )
    original = artifact.local_path.read_bytes()
    calls = []

    class NativeTargets:
        def fpm_seed_target_normalization(self):
            return {
                "policy": "same_cell_batch_real_kv_extrapolation_v1",
                "row_count": 1,
                "replacement_enabled": enabled,
                "replaced_count": int(replaced),
                "kept_count": int(not replaced),
                "rows": [
                    {
                        "row_index": 0,
                        "workload_kind": "decode",
                        "batch_size": 7,
                        "total_prefill_tokens": 0,
                        "total_kv_read_tokens": 2048,
                        "raw_latency_ms": 12.5,
                        "effective_latency_ms": effective,
                        "replaced": replaced,
                    }
                ],
            }

        def close(self):
            calls.append("close")

    def create(method, context):
        assert method == "warmup"
        assert context.fpm_artifact == artifact
        calls.append("construct")
        return NativeTargets()

    monkeypatch.setattr(adapters, "create_scenario_predictor", create)
    seeds = adapters.self_benchmark_iterations(case, artifact)
    assert calls == ["construct", "close"]  # Once per artifact, not per worker.
    assert [seed.observed_time_ms for seed in seeds] == [effective, effective]
    assert all([rank.wall_time_s for rank in seed.ranks] == [effective / 1000] + [0.0] * (dp - 1) for seed in seeds)
    assert artifact.local_path.read_bytes() == original
    target = seeds[0].ranks[0].extra_metadata["self_benchmark_target"]
    assert target["raw_latency_ms"] == 12.5
    assert target["source_parquet_sha256"] == artifact.sha256
    assert target["source_metadata_sha256"] == artifact.metadata_sha256
    assert target["replaced"] is replaced
    summary = _seed_target_normalization_evidence(seeds)
    assert summary["marked_source_rows"] == 1
    assert summary["replicated_seed_rows"] == 2
    assert summary["replaced_source_rows"] == int(replaced)
    assert summary["kept_source_rows"] == int(not replaced)
    assert summary["replacement_enabled"] is enabled
    assert len(summary["effective_targets_sha256"]) == 64


def test_seed_coordinates_use_shared_validated_metadata_completion(case, monkeypatch):
    from scripts.fpm_accuracy.models import aic_fpm_database

    case, artifact = self_benchmark(case, dp=4, partition_policy="balanced_v1")
    original = artifact.local_metadata_path.read_bytes()
    calls = []

    def complete(metadata, selected):
        assert selected == artifact
        assert metadata["coordinate_system"] is None
        calls.append(selected.metadata_sha256)
        return {**metadata, "coordinate_system": "iteration_totals_balanced_v1"}, {
            "coordinate_system": "iteration_totals_balanced_v1"
        }

    monkeypatch.setattr(aic_fpm_database, "_recover_reconstructed_metadata", complete)
    seeds = adapters.self_benchmark_iterations(case, artifact)
    assert calls == [artifact.metadata_sha256]
    assert all(len(seed.ranks) == 4 and seed.observed_time_ms == 12.5 for seed in seeds)
    assert artifact.local_metadata_path.read_bytes() == original


def test_mixed_fallback_keeps_effective_seed_provenance(case, monkeypatch):
    from types import SimpleNamespace

    from scripts.fpm_accuracy.models import learned_scenario, mixed_scenario
    from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext

    case, artifact = self_benchmark(replace(case, worker_role="aggregated"))
    seed = adapters.self_benchmark_iterations(case, artifact)[0]
    target = {
        "row_index": 0,
        "policy": "same_cell_batch_real_kv_extrapolation_v1",
        "source_parquet_sha256": artifact.sha256,
        "source_metadata_sha256": artifact.metadata_sha256,
        "raw_latency_ms": 12.5,
        "effective_latency_ms": 10.0,
        "replacement_enabled": True,
        "replaced": True,
    }
    seed = replace(
        seed, ranks=(replace(seed.ranks[0], wall_time_s=0.01, extra_metadata={"self_benchmark_target": target}),)
    )

    class Mixed:
        def __init__(self, engine_config, selected, scenario):
            assert selected == artifact and scenario == "a"
            self.provenance = {"grouping": {"target_scope": "recorded_aggregate_latency"}}

        def close(self):
            pass

    monkeypatch.setattr(mixed_scenario, "MixedOriginalA", Mixed)
    monkeypatch.setattr(adapters, "map_worker_config_to_aic", lambda *args: {})
    monkeypatch.setattr(
        adapters.aic_predictors,
        "_import_aisim_forward_pass_perf_model",
        lambda: SimpleNamespace(normalize_config=lambda x: x),
    )
    monkeypatch.setattr(
        adapters.aic_predictors,
        "_canonical_config_type",
        lambda: SimpleNamespace(from_legacy_engine_config=lambda *args: object()),
    )
    monkeypatch.setattr(adapters, "_import_gbdt_trainer", lambda: object())
    monkeypatch.setattr(
        learned_scenario, "LearnedScenarioPredictor", lambda *args, **kwargs: SimpleNamespace(mixed=kwargs["mixed"])
    )
    context = PredictorContext(
        worker=case.configuration.worker_config_record, worker_role="aggregated", fpm_artifact=artifact
    )
    model = adapters.create_scenario_predictor(
        "gbdt", context, training_iterations=(seed,), mixed_fallback=True, scenario="a"
    )
    assert model.mixed is not None
    assert model.mixed.provenance["grouping"]["target_scope"] == "native_normalized_aggregate_latency"
    assert model.mixed.provenance["seed_target_normalization"] == _seed_target_normalization_evidence((seed,))


def test_fake_seed_targets_fail_explicitly_when_native_normalization_is_unavailable(case, monkeypatch):
    from types import SimpleNamespace

    case, artifact = self_benchmark(case, kv_seed_regime="fake_fallback")
    closed = []
    monkeypatch.setattr(
        adapters,
        "create_scenario_predictor",
        lambda *a, **k: SimpleNamespace(close=lambda: closed.append(True)),
    )
    with pytest.raises(DependencyError, match="require native seed target normalization"):
        adapters.self_benchmark_iterations(case, artifact)
    assert closed == [True]


def test_offline_regression_order_is_label_and_source_order_independent_for_each_worker(case):
    case, artifact = self_benchmark(case)
    base = adapters.self_benchmark_iterations(case, artifact)[0].ranks[0]
    one_worker = tuple(
        adapters.ForwardPassIteration.single_rank(
            replace(
                base,
                fpm_id=index,
                counter_id=index,
                wall_time_s=(index + 1) / 1000,
                scheduled=replace(base.scheduled, sum_decode_kv_tokens=100 + index * 47),
            )
        )
        for index in range(8)
    )
    source = one_worker + tuple(replace(row, ranks=(replace(row.ranks[0], worker_id="replica"),)) for row in one_worker)
    ordered = adapters.order_offline_regression_seeds(source)
    assert adapters.order_offline_regression_seeds(tuple(reversed(source))) == ordered
    assert adapters.order_offline_regression_seeds(source[::2] + source[1::2]) == ordered

    def geometry(rows):
        return [(row.ranks[0].worker_id, row.ranks[0].scheduled.sum_decode_kv_tokens) for row in rows]

    changed = tuple(
        replace(
            row,
            ranks=(
                replace(
                    row.ranks[0],
                    fpm_id=1000 - index,
                    counter_id=2000 - index,
                    wall_time_s=row.ranks[0].wall_time_s * 7,
                    capture_id="different-source",
                    extra_metadata={"source_row_index": 1000 - index},
                ),
            ),
        )
        for index, row in enumerate(reversed(source))
    )
    assert geometry(adapters.order_offline_regression_seeds(changed)) == geometry(ordered)
    per_worker = {
        worker: [kv for identity, kv in geometry(ordered) if identity == worker]
        for worker in (base.worker_id, "replica")
    }
    assert per_worker[base.worker_id] == per_worker["replica"]
    assert per_worker[base.worker_id] != sorted(per_worker[base.worker_id])


def test_offline_regression_order_rejects_duplicate_features_without_label_tiebreak(case):
    case, artifact = self_benchmark(case)
    seed = adapters.self_benchmark_iterations(case, artifact)[0]
    duplicate = replace(seed, ranks=(replace(seed.ranks[0], wall_time_s=0.9, counter_id=99, fpm_id=99),))
    with pytest.raises(DataError, match="duplicate workload feature keys"):
        adapters.order_offline_regression_seeds((seed, duplicate))


@pytest.mark.parametrize("dp", [1, 16])
def test_compact_offline_order_preserves_original_named_feature_order(case, dp):
    case, artifact = self_benchmark(case)
    base = adapters.self_benchmark_iterations(case, artifact)[0].ranks[0]
    rows = tuple(
        adapters.ForwardPassIteration(
            ranks=tuple(
                replace(
                    base,
                    dp_rank=rank,
                    scheduled=replace(
                        base.scheduled,
                        num_decode_requests=1 + index % 7,
                        sum_decode_kv_tokens=500 - index * 3 + rank,
                        var_decode_kv_tokens=(index + rank) / 2,
                    ),
                    queued=replace(base.queued, num_decode_requests=index % 3),
                )
                for rank in reversed(range(dp))
            )
        )
        for index in range(40)
    )

    def original_key(iteration):
        return (
            iteration.configuration_id,
            tuple(
                (
                    rank.version,
                    rank.dp_rank,
                    str(rank.workload_kind),
                    tuple(sorted(asdict(rank.scheduled).items())),
                    tuple(sorted(asdict(rank.queued).items())),
                )
                for rank in sorted(iteration.ranks, key=lambda rank: rank.dp_rank)
            ),
        )

    reference = sorted(rows, key=original_key)
    random.Random(0).shuffle(reference)
    assert adapters.order_offline_regression_seeds(rows) == tuple(reference)
