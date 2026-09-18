# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AISim FPM Gym; see README.md for pinned source and modifications.

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fpm_accuracy.exceptions import ConfigurationError, DataError
from fpm_accuracy.hf import (
    SUPPORTED_EVIDENCE_FORMAT_IDS,
    SUPPORTED_PROTOCOL_IDS,
    CaseStatus,
    HfDataset,
    MeasurementState,
    OrderingKind,
    adapter_for,
)
from fpm_accuracy.hf import dataset as dataset_module

REVISION = "a" * 40
CONFIGURATION_PATH = "data/Org--Model/h100-sxm/vllm/1.0/single"
SNAPSHOT_ID = "aisim-commit-aaaaaaaa"
MEASUREMENT_POLICY_ID = "aisim-fpm/forward-pass-measurement-v1/max-rank-v3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_dataset(
    root: Path,
    *,
    protocol_id: str | None,
    files: list[tuple[str, str, bytes]],
    evidence_format_id: str | None = None,
    include_fpm: bool = True,
    dp: int = 1,
) -> HfDataset:
    measurement_entries = []
    for index, (role, suffix, content) in enumerate(files):
        relative = f"{CONFIGURATION_PATH}/measurements/{suffix}"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        measurement_entries.append(
            {
                "measurement_file_id": f"measurement-{index}",
                "path": relative,
                "sha256": _sha256(path),
                "role": role,
                "source_path": f"source/{suffix}",
                "source_sha256": _sha256(path),
                "source_revision": REVISION,
                "configuration_path": CONFIGURATION_PATH,
                "snapshot_id": SNAPSHOT_ID,
            }
        )

    measurement_manifest_path = f"{CONFIGURATION_PATH}/measurements/manifest.json"
    measurement_manifest = {
        "manifest_version": 4,
        "measurement_artifact_id": "measurements-1",
        "measurement_protocol_id": protocol_id,
        "configuration_path": CONFIGURATION_PATH,
        "snapshot_id": SNAPSHOT_ID,
        "snapshot_status": "current",
        "model_id": "Org/Model",
        "system": "h100_sxm",
        "framework": "vllm",
        "framework_version": "1.0",
        "parallelism": "single",
        "files": measurement_entries,
    }
    if evidence_format_id is not None:
        measurement_manifest["evidence_format_id"] = evidence_format_id
    _write_json(root / measurement_manifest_path, measurement_manifest)
    fpm_entries = []
    if include_fpm:
        configuration_selector = {
            "model_path": "Org/Model",
            "system": "h100_sxm",
            "backend": "vllm",
            "backend_version": "1.0",
            "weight_quantization": "bfloat16",
            "kv_cache_dtype": "bfloat16",
            "parallel_strategy": "single",
            "tp": 1,
            "pp": 1,
            "dp": dp,
            "moe_tp": 1,
            "moe_ep": 1,
            "cp": 1,
        }
        fpm_relative = f"{CONFIGURATION_PATH}/fpm/fpm.parquet"
        fpm_path = root / fpm_relative
        fpm_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {**{field: [value] for field, value in configuration_selector.items()}, "workload_kind": ["decode"]}
            ),
            fpm_path,
        )
        fpm_hash = _sha256(fpm_path)
        metadata_relative = f"{CONFIGURATION_PATH}/fpm/fpm.metadata.json"
        _write_json(
            root / metadata_relative,
            {
                "parquet_sha256": fpm_hash,
                "row_count": 1,
                "schema_name": "aic_fpm_forward_perf",
                "schema_version": 6,
                "configuration_selector": configuration_selector,
            },
        )
        fpm_entries.append(
            {
                "artifact_id": "fpm-1",
                "path": fpm_relative,
                "metadata_path": metadata_relative,
                "sha256": fpm_hash,
                "role": "primary",
                "phases": ["decode", "prefill"],
                "row_count": 1,
            }
        )
    config_manifest_path = f"{CONFIGURATION_PATH}/manifest.json"
    configuration_manifest = {
        "manifest_version": 3,
        "configuration_path": CONFIGURATION_PATH,
        "snapshot_id": SNAPSHOT_ID,
        "snapshot_status": "current",
        "model_id": "Org/Model",
        "model_revision": "model-sha",
        "system": "h100_sxm",
        "gpu_family": "H100",
        "framework": "vllm",
        "framework_version": "1.0",
        "parallelism": "single",
        "parallel_strategy": "single",
        "tp": 1,
        "pp": 1,
        "dp": dp,
        "moe_tp": 1,
        "moe_ep": 1,
        "cp": 1,
        "weight_quantization": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "aisim_commit": None,
        "fpm": fpm_entries,
        "measurements": {
            "artifact_id": "measurements-1",
            "protocol_id": protocol_id,
            "manifest_path": measurement_manifest_path,
            "manifest_sha256": _sha256(root / measurement_manifest_path),
            "file_count": len(measurement_entries),
        },
    }
    if evidence_format_id is not None:
        configuration_manifest["measurements"]["evidence_format_id"] = evidence_format_id
    _write_json(root / config_manifest_path, configuration_manifest)
    _write_json(
        root / "catalog/index.json",
        {
            "catalog_version": 5,
            "dataset_id": "nvidia/aisimulate-fpm-dataset",
            "configuration_manifests": [config_manifest_path],
            "history_manifests": [],
        },
    )
    return HfDataset.from_local(root, revision=REVISION)


@pytest.mark.parametrize(
    ("relative", "field"),
    [
        ("catalog/index.json", "dataset_id"),
        (f"{CONFIGURATION_PATH}/manifest.json", "snapshot_id"),
        (f"{CONFIGURATION_PATH}/measurements/manifest.json", "snapshot_id"),
        (f"{CONFIGURATION_PATH}/fpm/fpm.metadata.json", "parquet_sha256"),
        (f"{CONFIGURATION_PATH}/fpm/fpm.metadata.json", "model_path"),
    ],
)
def test_duplicate_hf_json_keys_fail_even_with_matching_hashes(tmp_path, relative, field):
    _build_dataset(tmp_path, protocol_id=None, files=[])
    path = tmp_path / relative
    original = path.read_text()
    path.write_text(original.replace(f'"{field}":', f'"{field}": "ambiguous", "{field}":', 1))
    # Keep provenance valid so only strict parsing can reject the ambiguous bytes.
    if relative.endswith("measurements/manifest.json"):
        configuration = tmp_path / CONFIGURATION_PATH / "manifest.json"
        manifest = json.loads(configuration.read_text())
        manifest["measurements"]["manifest_sha256"] = _sha256(path)
        _write_json(configuration, manifest)
    with pytest.raises(DataError, match="duplicate JSON key"):
        dataset = HfDataset.from_local(tmp_path, revision=REVISION)
        dataset.measurement_case(CONFIGURATION_PATH)


def _fpm_payload(*, rank: int = 0, counter: int = 1, wall_time: float = 0.01) -> dict[str, object]:
    return {
        "version": 1,
        "worker_id": "worker",
        "dp_rank": rank,
        "counter_id": counter,
        "wall_time": wall_time,
        "observed_at_unix_ms": 1000 + counter,
        "scheduled_requests": {
            "num_prefill_requests": 0,
            "sum_prefill_tokens": 0,
            "sum_prefill_kv_tokens": 0,
            "num_decode_requests": 2,
            "sum_decode_kv_tokens": 128,
        },
        "queued_requests": {},
    }


def _benchmark_point(benchmark_id: int = 9) -> dict[str, object]:
    return {
        "point_type": "decode",
        "benchmark_id": benchmark_id,
        "batch_size": 2,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 128,
    }


def _listener_iteration(*, complete: bool = True) -> dict[str, object]:
    rank_zero = _fpm_payload(rank=0, counter=11, wall_time=0.01)
    rank_zero["_recv_ts"] = "1000.001"
    rank_one = _fpm_payload(rank=1, counter=27, wall_time=0.013)
    rank_one["_recv_ts"] = "999.9995"
    return {
        "version": 1,
        "source_kind": "rank_event_stream",
        "iteration_id": "decode:1000.001",
        "producer": {"component": "instrumented-scheduler", "version": "1"},
        "collector": {"kind": "service", "component": "campaign-listener"},
        "grouping": {
            "method": "listener_receive_time_window",
            "authority": "derived",
            "performed_by": "campaign-importer-v1",
            "key": "decode:1000.001:8ms",
            "engine_role": "decode",
            "source_field": "_recv_ts",
            "source_unit": "s",
            "clock_correction": {
                "method": "campaign-clock-offset-v1",
                "reference_clock_id": "node-a",
                "offset_unit": "ms",
                "groups": [
                    {"clock_id": "node-a", "dp_ranks": [0], "offset": 0},
                    {"clock_id": "node-b", "dp_ranks": [1], "offset": 2},
                ],
            },
            "window": {
                "size": 8,
                "unit": "ms",
                "anchor": "earliest_corrected_receive_timestamp",
                "boundary": "half_open",
                "distinct_by": "dp_rank",
            },
            "window_start": "1000.001",
        },
        "expected_dp_ranks": [0, 1],
        "complete": complete,
        "max_rank_wall_time": 0.013 if complete else None,
        "rank_measurements": [rank_zero, rank_one] if complete else [rank_zero],
    }


def _build_pre_grouped_dataset(root: Path, groups: list[list[dict[str, object]]]) -> HfDataset:
    truth = gzip.compress(json.dumps(groups).encode())
    evidence = b'{"grouping":"reviewed"}'
    _build_dataset(
        root,
        protocol_id="forward-pass-measurement-v1",
        files=[
            ("truth", "fpm_iterations.json.gz", truth),
            ("supporting_evidence", "alignment_report.json", evidence),
        ],
        include_fpm=False,
        dp=2,
    )
    measurement_path = root / CONFIGURATION_PATH / "measurements/manifest.json"
    manifest = json.loads(measurement_path.read_text())
    truth_entry, evidence_entry = manifest["files"]
    truth_entry.update(
        {
            "representation": "pre_grouped_rank_lists",
            "iteration_count": len(groups),
            "rank_record_count": sum(len(group) for group in groups),
            "grouping": {
                "method": "pre_grouped",
                "authority": "derived",
                "producer": {"component": "uploader-reviewed-capture"},
                "expected_dp_ranks": [0, 1],
                "provenance_files": [{"path": evidence_entry["path"], "sha256": evidence_entry["sha256"]}],
            },
        }
    )
    _write_json(measurement_path, manifest)
    config_path = root / CONFIGURATION_PATH / "manifest.json"
    config = json.loads(config_path.read_text())
    config["measurements"]["manifest_sha256"] = _sha256(measurement_path)
    _write_json(config_path, config)
    return HfDataset.from_local(root, revision=REVISION)


def test_supported_measurement_protocols_and_evidence_formats_are_distinct() -> None:
    assert {
        "forward-pass-measurement-v1",
        "forward-pass-record-v1",
    } == SUPPORTED_PROTOCOL_IDS
    assert {
        "wave-summary-v1",
        "bucket-summary-v1",
    } == SUPPORTED_EVIDENCE_FORMAT_IDS
    assert adapter_for("wave-summary-v1") is None


def test_fpm_stream_is_target_blind_and_chronological(tmp_path: Path) -> None:
    rows = [
        _fpm_payload(counter=2),
        _fpm_payload(rank=1, counter=1),
        _fpm_payload(counter=1),
        _fpm_payload(counter=3, wall_time=0.0),
    ]
    content = gzip.compress("".join(f"{json.dumps(row)}\n" for row in rows).encode())
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl.gz", content), ("configuration", "config.json", b"{}")],
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.READY
    assert case.ordering is OrderingKind.CHRONOLOGICAL
    assert case.worker_role == "decode"
    assert case.parser_policy_id == MEASUREMENT_POLICY_ID
    assert [item.iteration.representative_rank.counter_id for item in case.observations] == [1, 2]
    assert all("wall_time" not in rank for item in case.observations for rank in item.prediction_input().rank_payloads)
    assert all("wall_time" in rank for item in case.observations for rank in item.tuning_payload())
    assert {(issue.reason, issue.count) for issue in case.issues} == {
        ("non_positive_latency", 1),
        ("unexpected_dp_rank", 1),
    }
    assert case.fpm_artifacts[0].local_path.is_file()
    assert case.fpm_artifacts[0].metadata_sha256 == _sha256(case.fpm_artifacts[0].local_metadata_path)
    assert case.fpm_artifacts[0].metadata_provenance_url in case.provenance_urls
    assert case.configuration.manifest_sha256 == _sha256(tmp_path / case.configuration.manifest_path)
    assert f"blob/{REVISION}/" in case.provenance_urls[0]
    without_fpm = dataset.measurement_case(CONFIGURATION_PATH, fpm_artifact_ids=())
    assert without_fpm.case_id == case.case_id
    assert without_fpm.measurement_membership_sha256 == case.measurement_membership_sha256


def test_raw_multi_rank_stream_is_not_reduced_to_rank_zero(tmp_path: Path) -> None:
    rows = [_fpm_payload(rank=0), _fpm_payload(rank=1)]
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", "".join(f"{json.dumps(row)}\n" for row in rows).encode())],
        include_fpm=False,
        dp=2,
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.NO_MEASUREMENTS
    assert case.observations == ()
    assert [(issue.state, issue.reason, issue.count) for issue in case.issues] == [
        (MeasurementState.MEASUREMENT_UNAVAILABLE, "unsynchronized_attention_dp_stream", 2)
    ]


def test_pre_grouped_rank_lists_preserve_membership_and_score_maximum(tmp_path: Path) -> None:
    groups = [
        [_fpm_payload(rank=0, counter=11, wall_time=0.01), _fpm_payload(rank=1, counter=27, wall_time=0.013)],
        [_fpm_payload(rank=0, counter=12, wall_time=0.02), _fpm_payload(rank=1, counter=28, wall_time=0.017)],
    ]
    dataset = _build_pre_grouped_dataset(tmp_path, groups)

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.READY
    assert [observation.actual_ms for observation in case.observations] == [13.0, 20.0]
    assert [[rank.dp_rank for rank in observation.iteration.ranks] for observation in case.observations] == [
        [0, 1],
        [0, 1],
    ]
    assert [observation.source_row for observation in case.observations] == [1, 2]
    assert all(
        "wall_time" not in rank
        for observation in case.observations
        for rank in observation.prediction_input().rank_payloads
    )


def test_pre_grouped_rank_lists_require_exact_declared_membership(tmp_path: Path) -> None:
    dataset = _build_pre_grouped_dataset(
        tmp_path,
        [[_fpm_payload(rank=1, counter=27), _fpm_payload(rank=0, counter=11)]],
    )

    with pytest.raises(DataError, match="ranks do not match expected_dp_ranks"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_canonical_listener_iteration_preserves_all_ranks_and_scores_maximum(tmp_path: Path) -> None:
    content = f"{json.dumps(_listener_iteration())}\n".encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
        include_fpm=False,
        dp=2,
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.READY
    assert case.ordering is OrderingKind.CHRONOLOGICAL
    assert case.parser_policy_id == MEASUREMENT_POLICY_ID
    assert case.issues == ()
    assert case.observations[0].event_time == "1000.001"
    assert [rank.dp_rank for rank in case.observations[0].iteration.ranks] == [0, 1]
    assert [rank.counter_id for rank in case.observations[0].iteration.ranks] == [11, 27]
    assert case.observations[0].actual_ms == pytest.approx(13.0)
    assert len(case.observations[0].prediction_input().rank_payloads) == 2
    assert all("wall_time" not in rank for rank in case.observations[0].prediction_input().rank_payloads)
    assert [rank["wall_time"] for rank in case.observations[0].tuning_payload()] == [0.01, 0.013]


def test_canonical_benchmark_iteration_normalizes_to_the_same_rank_shape(tmp_path: Path) -> None:
    rank_zero = _fpm_payload(rank=0, counter=9, wall_time=0.01)
    rank_one = _fpm_payload(rank=1, counter=9, wall_time=0.013)
    iteration = {
        "version": 1,
        "source_kind": "self_benchmark",
        "iteration_id": "benchmark:9",
        "producer": {"component": "self-benchmark"},
        "grouping": {
            "method": "benchmark_id",
            "authority": "producer",
            "key": "9",
            "benchmark_id": 9,
            "point": _benchmark_point(),
        },
        "expected_dp_ranks": [0, 1],
        "complete": True,
        "max_rank_wall_time": 0.013,
        "rank_measurements": [rank_zero, rank_one],
    }
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", f"{json.dumps(iteration)}\n".encode())],
        include_fpm=False,
        dp=2,
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.READY
    assert case.ordering is OrderingKind.FILE_ORDER_FALLBACK
    assert [rank.dp_rank for rank in case.observations[0].iteration.ranks] == [0, 1]
    assert case.observations[0].actual_ms == pytest.approx(13.0)


def test_canonical_single_rank_iteration_is_directly_evaluable(tmp_path: Path) -> None:
    rank = _fpm_payload(rank=0, counter=3, wall_time=0.007)
    iteration = {
        "version": 1,
        "source_kind": "rank_event_stream",
        "iteration_id": "rank:0:3",
        "producer": {"component": "instrumented-scheduler"},
        "grouping": {"method": "single_rank", "authority": "producer", "key": "rank:0:3"},
        "expected_dp_ranks": [0],
        "complete": True,
        "max_rank_wall_time": 0.007,
        "rank_measurements": [rank],
    }
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", f"{json.dumps(iteration)}\n".encode())],
        include_fpm=False,
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.READY
    assert case.ordering is OrderingKind.CHRONOLOGICAL
    assert len(case.observations[0].iteration.ranks) == 1
    assert case.observations[0].actual_ms == pytest.approx(7.0)


def test_mixed_canonical_groupings_share_millisecond_chronology(tmp_path):
    listener = _listener_iteration()
    listener["expected_dp_ranks"] = [0]
    listener["rank_measurements"] = listener["rank_measurements"][:1]
    listener["max_rank_wall_time"] = 0.01
    listener["grouping"]["clock_correction"]["groups"] = listener["grouping"]["clock_correction"]["groups"][:1]
    rows = []
    for timestamp in (1000002, 1000000):
        rank = _fpm_payload(counter=timestamp)
        rank["observed_at_unix_ms"] = timestamp
        rows.append(
            {
                "version": 1,
                "source_kind": "rank_event_stream",
                "iteration_id": str(timestamp),
                "producer": {"component": "instrumented-scheduler"},
                "grouping": {"method": "single_rank", "authority": "producer", "key": str(timestamp)},
                "expected_dp_ranks": [0],
                "complete": True,
                "max_rank_wall_time": 0.01,
                "rank_measurements": [rank],
            }
        )
    rows.insert(1, listener)
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        include_fpm=False,
        files=[("truth", "truth.jsonl", "\n".join(map(json.dumps, rows)).encode())],
    )
    case = dataset.measurement_case(CONFIGURATION_PATH)
    assert case.ordering is OrderingKind.CHRONOLOGICAL
    assert [item.iteration.ranks[0].counter_id for item in case.observations] == [1000000, 11, 1000002]


@pytest.mark.parametrize("field", ["moe_ep", "moe_tp"])
@pytest.mark.parametrize("value", [None, "bad", "2", True, 1.5, 0, -1])
def test_invalid_manifest_moe_parallelism_fails_with_data_error(tmp_path, field, value):
    dataset = _build_dataset(tmp_path, protocol_id="forward-pass-measurement-v1", files=[])
    path = tmp_path / CONFIGURATION_PATH / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    _write_json(path, manifest)
    with pytest.raises(DataError, match=field):
        dataset.configurations()


def test_missing_moe_parallelism_defaults_match_fpm_identity(tmp_path):
    content = json.dumps(_fpm_payload()).encode()
    dataset = _build_dataset(
        tmp_path, protocol_id="forward-pass-measurement-v1", files=[("truth", "truth.jsonl", content)]
    )
    path = tmp_path / CONFIGURATION_PATH / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["moe_ep"], manifest["moe_tp"]
    _write_json(path, manifest)
    case = dataset.measurement_case(CONFIGURATION_PATH)
    engine = case.configuration.worker_config_record.config.aic_engine_config
    assert engine["moe_ep_size"] == engine["moe_tp_size"] == 1
    assert case.status is CaseStatus.READY
    assert len(case.fpm_artifacts) == len(case.observations) == 1


def test_incomplete_canonical_iteration_is_retained_as_unavailable_evidence(tmp_path: Path) -> None:
    content = f"{json.dumps(_listener_iteration(complete=False))}\n".encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
        include_fpm=False,
        dp=2,
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.NO_MEASUREMENTS
    assert [(issue.state, issue.reason, issue.count) for issue in case.issues] == [
        (MeasurementState.MEASUREMENT_UNAVAILABLE, "incomplete_synchronized_iteration", 1)
    ]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_cached_max", "max_rank_wall_time does not match"),
        ("missing_receive_timestamp", "lacks join evidence"),
        ("bad_window_start", "is not the earliest corrected receive time"),
        ("bad_clock_partition", "must partition expected_dp_ranks"),
    ],
)
def test_canonical_listener_iteration_validates_materialized_join_evidence(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    iteration = _listener_iteration()
    if mutation == "wrong_cached_max":
        iteration["max_rank_wall_time"] = 0.014
    elif mutation == "missing_receive_timestamp":
        del iteration["rank_measurements"][1]["_recv_ts"]
    elif mutation == "bad_window_start":
        iteration["grouping"]["window_start"] = "1000.002"
    else:
        iteration["grouping"]["clock_correction"]["groups"][1]["dp_ranks"] = [2]
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", f"{json.dumps(iteration)}\n".encode())],
        include_fpm=False,
        dp=2,
    )

    with pytest.raises(DataError, match=message):
        dataset.measurement_case(CONFIGURATION_PATH)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dp_rank", 0.5),
        ("wall_time", "0.01"),
        ("scheduled_requests.num_decode_requests", True),
        ("scheduled_requests.var_decode_kv_tokens", -1),
    ],
)
def test_fpm_stream_rejects_lossy_or_boolean_integer_coercion(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = _fpm_payload()
    if "." not in field:
        payload[field] = value
    else:
        _, nested_field = field.split(".", maxsplit=1)
        scheduled = dict(payload["scheduled_requests"])
        scheduled[nested_field] = value
        payload["scheduled_requests"] = scheduled
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", f"{json.dumps(payload)}\n".encode())],
        include_fpm=False,
    )

    with pytest.raises(DataError, match="invalid"):
        dataset.measurement_case(CONFIGURATION_PATH)


@pytest.mark.parametrize(
    "field",
    ["version", "worker_id", "dp_rank", "counter_id", "wall_time", "scheduled_requests", "queued_requests"],
)
def test_fpm_stream_requires_every_rank_measurement_field(tmp_path: Path, field: str) -> None:
    payload = _fpm_payload()
    del payload[field]
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", f"{json.dumps(payload)}\n".encode())],
        include_fpm=False,
    )

    with pytest.raises(DataError, match="missing fields"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_forward_pass_measurement_rejects_an_undeclared_serialization(tmp_path: Path) -> None:
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.txt", f"{json.dumps(_fpm_payload())}\n".encode())],
        include_fpm=False,
    )

    with pytest.raises(DataError, match="does not support file"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_derived_truth_and_file_order_fallback_are_deterministic(tmp_path: Path) -> None:
    csv_data = (
        "measurement_id,phase,batch_size,total_prefill_tokens,total_kv_read_tokens,truth_latency_ms\n"
        "point-b,prefill,2,32,8,4.5\n"
        "point-a,decode,4,,128,7.5\n"
    )
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-record-v1",
        files=[("derived_truth", "truth.csv.gz", gzip.compress(csv_data.encode()))],
        include_fpm=False,
    )

    first = dataset.measurement_case(CONFIGURATION_PATH)
    second = dataset.measurement_case(CONFIGURATION_PATH)

    assert first.ordering is OrderingKind.FILE_ORDER_FALLBACK
    assert first.parser_policy_id == "aisim-fpm/forward-pass-record-v1"
    assert "file_order_fallback" in first.warnings[0]
    assert [item.workload_kind.value for item in first.observations] == ["prefill", "decode"]
    assert [item.actual_ms for item in first.observations] == [4.5, 7.5]
    assert [item.observation_id for item in first.observations] == [item.observation_id for item in second.observations]
    assert first.case_id == second.case_id
    assert first.measurement_membership_sha256 == second.measurement_membership_sha256
    assert len(first.measurement_membership_sha256) == 64


@pytest.mark.parametrize(
    "csv_data",
    [
        "measurement_id,phase,batch_size,truth_latency_ms\npoint,prefill,1,1.0\n",
        (
            "measurement_id,phase,batch_size,total_prefill_tokens,total_kv_read_tokens,truth_latency_ms\n"
            "point,Prefill,1,16,0,1.0\n"
        ),
        (
            "measurement_id,phase,batch_size,total_prefill_tokens,total_kv_read_tokens,truth_latency_ms\n"
            "point,prefill,1,0,0,1.0\n"
        ),
        (
            "measurement_id,phase,batch_size,total_prefill_tokens,total_kv_read_tokens,truth_latency_ms\n"
            "point,decode,1,1,16,1.0\n"
        ),
        (
            "measurement_id,phase,batch_size,total_prefill_tokens,total_kv_read_tokens,truth_latency_ms\n"
            "point,decode,1,0,16,0\n"
        ),
    ],
)
def test_forward_pass_record_enforces_the_typed_row_contract(tmp_path: Path, csv_data: str) -> None:
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-record-v1",
        files=[("derived_truth", "truth.csv", csv_data.encode())],
        include_fpm=False,
    )

    with pytest.raises(DataError):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_forward_pass_record_rejects_an_undeclared_serialization(tmp_path: Path) -> None:
    csv_data = (
        "measurement_id,phase,batch_size,total_prefill_tokens,total_kv_read_tokens,truth_latency_ms\n"
        "point,decode,1,0,16,1.0\n"
    )
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-record-v1",
        files=[("derived_truth", "truth.txt", csv_data.encode())],
        include_fpm=False,
    )

    with pytest.raises(DataError, match="does not support file"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_synchronized_benchmark_parser_preserves_rank_shape(tmp_path: Path) -> None:
    rank_zero = _fpm_payload(rank=0, counter=9, wall_time=0.0)
    rank_one = _fpm_payload(rank=1, counter=9, wall_time=0.013)
    benchmark = {
        "iteration_groups": [
            {
                "benchmark_id": 9,
                "point": _benchmark_point(),
                "complete": True,
                "expected_dp_ranks": [0, 1],
                "wall_time": 0.013,
                "rank_results": [{"dp_rank": 0, "fpms": [rank_zero]}, {"dp_rank": 1, "fpms": [rank_one]}],
            }
        ]
    }
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "benchmark.json", json.dumps(benchmark).encode())],
        include_fpm=False,
        dp=2,
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert len(case.observations) == 1
    assert case.parser_policy_id == MEASUREMENT_POLICY_ID
    assert [rank.dp_rank for rank in case.observations[0].iteration.ranks] == [0, 1]
    assert case.observations[0].actual_ms == pytest.approx(13.0)
    assert case.issues == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("incomplete", "complete must be true"),
        ("missing_point_coordinate", "missing fields"),
        ("zero_iteration_time", "non-positive wall_time"),
    ],
)
def test_synchronized_benchmark_enforces_the_published_record_shape(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    rank = _fpm_payload(counter=9)
    group = {
        "benchmark_id": 9,
        "point": _benchmark_point(),
        "complete": True,
        "expected_dp_ranks": [0],
        "wall_time": 0.01,
        "rank_results": [{"dp_rank": 0, "fpms": [rank]}],
    }
    if mutation == "incomplete":
        group["complete"] = False
    elif mutation == "missing_point_coordinate":
        del group["point"]["batch_size"]
    else:
        group["wall_time"] = 0
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "benchmark.json", json.dumps({"iteration_groups": [group]}).encode())],
        include_fpm=False,
    )

    with pytest.raises(DataError, match=message):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_complete_benchmark_requires_exact_consistent_ranks(tmp_path: Path) -> None:
    rank_zero = _fpm_payload(rank=0, counter=9, wall_time=0.01)
    rank_one = _fpm_payload(rank=1, counter=9, wall_time=0.013)
    point = _benchmark_point()
    benchmark = {
        "iteration_groups": [
            {
                "benchmark_id": 9,
                "point": point,
                "complete": True,
                "expected_dp_ranks": [0],
                "wall_time": 0.013,
                "rank_results": [{"dp_rank": 0, "fpms": [rank_zero]}, {"dp_rank": 1, "fpms": [rank_one]}],
            },
            {
                "benchmark_id": 9,
                "point": point,
                "complete": True,
                "expected_dp_ranks": [0, 1],
                "wall_time": 0.013,
                "rank_results": [{"dp_rank": 0, "fpms": [rank_zero]}, {"dp_rank": 0, "fpms": [rank_one]}],
            },
            {
                "benchmark_id": 9,
                "point": point,
                "complete": True,
                "expected_dp_ranks": [0, 1],
                "wall_time": 0.014,
                "rank_results": [{"dp_rank": 0, "fpms": [rank_zero]}, {"dp_rank": 1, "fpms": [rank_one]}],
            },
        ]
    }
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "benchmark.json", json.dumps(benchmark).encode())],
        include_fpm=False,
        dp=2,
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.NO_MEASUREMENTS
    assert case.observations == ()
    assert [(issue.reason, issue.count) for issue in case.issues] == [("incomplete_or_invalid_benchmark_point", 3)]


def test_membership_and_observation_identity_cover_every_rank_payload_and_target(tmp_path: Path) -> None:
    def build_case(root: Path, *, rank_one_tokens: int, rank_one_wall: float):
        rank_zero = _fpm_payload(rank=0, counter=9, wall_time=0.02)
        rank_zero["scheduled_requests"] = {
            **rank_zero["scheduled_requests"],
            "sum_decode_kv_tokens": 256,
        }
        rank_one = _fpm_payload(rank=1, counter=9, wall_time=rank_one_wall)
        rank_one["scheduled_requests"] = {
            **rank_one["scheduled_requests"],
            "sum_decode_kv_tokens": rank_one_tokens,
        }
        benchmark = {
            "iteration_groups": [
                {
                    "benchmark_id": 9,
                    "point": _benchmark_point(),
                    "complete": True,
                    "expected_dp_ranks": [0, 1],
                    "wall_time": 0.02,
                    "rank_results": [
                        {"dp_rank": 0, "fpms": [rank_zero]},
                        {"dp_rank": 1, "fpms": [rank_one]},
                    ],
                }
            ]
        }
        dataset = _build_dataset(
            root,
            protocol_id="forward-pass-measurement-v1",
            files=[("truth", "benchmark.json", json.dumps(benchmark).encode())],
            include_fpm=False,
            dp=2,
        )
        return dataset.measurement_case(CONFIGURATION_PATH)

    baseline = build_case(tmp_path / "baseline", rank_one_tokens=64, rank_one_wall=0.01)
    changed_payload = build_case(tmp_path / "payload", rank_one_tokens=32, rank_one_wall=0.01)
    changed_target = build_case(tmp_path / "target", rank_one_tokens=64, rank_one_wall=0.012)

    assert baseline.observations[0].iteration.representative_rank.dp_rank == 0
    assert {case.observations[0].actual_ms for case in (baseline, changed_payload, changed_target)} == {20.0}
    assert len({case.measurement_membership_sha256 for case in (baseline, changed_payload, changed_target)}) == 3
    assert len({case.observations[0].observation_id for case in (baseline, changed_payload, changed_target)}) == 3


def test_static_wave_is_supporting_evidence_without_a_measurement_protocol(tmp_path: Path) -> None:
    rows = [
        {"arm": "a", "c": 1, "wave": 0, "warmup": True, "round_ms_p50": 3.0, "counters": {}},
        {
            "arm": "a",
            "c": 4,
            "wave": 1,
            "warmup": False,
            "round_ms_p50": 4.0,
            "counters": {"vllm:iteration_tokens_total_count": 64},
        },
        {"arm": "a", "c": 4, "wave": 2, "warmup": False, "round_ms_p50": 5.0, "counters": {}},
    ]
    dataset = _build_dataset(
        tmp_path,
        protocol_id=None,
        evidence_format_id="wave-summary-v1",
        files=[
            (
                "supporting_evidence",
                "provenance/wave.jsonl",
                "".join(f"{json.dumps(row)}\n" for row in rows).encode(),
            )
        ],
        include_fpm=False,
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.SUPPORTING_EVIDENCE_ONLY
    assert case.protocol_id is None
    assert case.configuration.measurements.evidence_format_id == "wave-summary-v1"
    assert case.parser_policy_id is None
    assert case.observations == ()
    assert case.truth_files == ()
    assert len(case.helper_files) == 1
    assert "supporting evidence" in case.warnings[0]


@pytest.mark.parametrize(
    ("protocol_id", "message"),
    [(None, "require a measurement protocol")],
)
def test_truth_with_an_unknown_or_missing_protocol_fails_closed(
    tmp_path: Path,
    protocol_id: str | None,
    message: str,
) -> None:
    dataset = _build_dataset(
        tmp_path,
        protocol_id=protocol_id,
        files=[("truth", "truth.jsonl", b"{}\n")],
        include_fpm=False,
    )

    with pytest.raises(DataError, match=message):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_unknown_protocol_and_evidence_ids_fail_even_without_truth(tmp_path: Path) -> None:
    unknown_protocol = _build_dataset(
        tmp_path / "protocol",
        protocol_id="unknown-v1",
        files=[],
        include_fpm=False,
    )
    unknown_evidence = _build_dataset(
        tmp_path / "evidence",
        protocol_id=None,
        evidence_format_id="unknown-summary-v1",
        files=[],
        include_fpm=False,
    )

    assert unknown_protocol.measurement_case(CONFIGURATION_PATH).status == "unsupported_protocol"
    with pytest.raises(DataError, match="unsupported evidence format"):
        unknown_evidence.configurations()


def test_measurement_protocol_and_evidence_format_are_mutually_exclusive(tmp_path: Path) -> None:
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        evidence_format_id="wave-summary-v1",
        files=[],
        include_fpm=False,
    )

    with pytest.raises(DataError, match="may not declare both"):
        dataset.configurations()


def test_supporting_evidence_format_cannot_bind_truth_files(tmp_path: Path) -> None:
    dataset = _build_dataset(
        tmp_path,
        protocol_id=None,
        evidence_format_id="wave-summary-v1",
        files=[("truth", "truth.jsonl", f"{json.dumps(_fpm_payload())}\n".encode())],
        include_fpm=False,
    )

    with pytest.raises(DataError, match="may not declare truth"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_empty_measurement_bundle_is_explicit(tmp_path: Path) -> None:
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[],
        include_fpm=False,
    )

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert case.status is CaseStatus.NO_MEASUREMENTS
    assert case.observations == ()


def test_override_selects_stable_ids_and_rejects_unknown_ids(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    dataset_root = tmp_path / "dataset"
    _build_dataset(
        dataset_root,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "one.jsonl", content), ("truth", "two.jsonl", content)],
        include_fpm=False,
    )
    override = tmp_path / "overrides.yaml"
    override.write_text(
        f"""version: 1
overrides:
  - configuration_path: {CONFIGURATION_PATH}
    truth_file_ids: [measurement-1]
    worker_role: prefill
    ordering: intentional_sweep
""",
        encoding="utf-8",
    )
    dataset = HfDataset.from_local(dataset_root, revision=REVISION, overrides_path=override)

    case = dataset.measurement_case(CONFIGURATION_PATH)

    assert [file.measurement_file_id for file in case.truth_files] == ["measurement-1"]
    assert case.worker_role == "prefill"
    assert case.ordering is OrderingKind.INTENTIONAL_SWEEP
    assert case.override_applied is True
    assert case.override_sha256 == _sha256(override)
    assert case.override_effects == (
        'truth_file_ids=["measurement-1"]',
        'worker_role="prefill"',
        'ordering="intentional_sweep"',
    )
    original_case_id = case.case_id

    override.write_text(
        f"""version: 1
overrides:
  # Same binding, different override artifact.
  - configuration_path: {CONFIGURATION_PATH}
    truth_file_ids: [measurement-1]
    worker_role: prefill
    ordering: intentional_sweep
""",
        encoding="utf-8",
    )
    same_binding = HfDataset.from_local(dataset_root, revision=REVISION, overrides_path=override)
    assert same_binding.measurement_case(CONFIGURATION_PATH).case_id != original_case_id

    override.write_text(
        f"""version: 1
overrides:
  - configuration_path: {CONFIGURATION_PATH}
    truth_file_ids: [measurement-missing]
""",
        encoding="utf-8",
    )
    invalid = HfDataset.from_local(dataset_root, revision=REVISION, overrides_path=override)
    with pytest.raises(ConfigurationError, match="unknown stable artifact IDs"):
        invalid.measurement_case(CONFIGURATION_PATH)


def test_hash_mismatch_and_unsafe_paths_fail_closed(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
        include_fpm=False,
    )
    truth_path = tmp_path / CONFIGURATION_PATH / "measurements/truth.jsonl"
    truth_path.write_bytes(b"changed")
    with pytest.raises(DataError, match="hash mismatch"):
        dataset.measurement_case(CONFIGURATION_PATH)

    index = json.loads((tmp_path / "catalog/index.json").read_text())
    index["configuration_manifests"] = ["../outside.json"]
    _write_json(tmp_path / "catalog/index.json", index)
    unsafe = HfDataset.from_local(tmp_path, revision=REVISION)
    with pytest.raises(DataError, match="unsafe HF path"):
        unsafe.configurations()


def test_symlink_escape_is_rejected_even_when_content_hash_matches(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    dataset_root = tmp_path / "dataset"
    dataset = _build_dataset(
        dataset_root,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
        include_fpm=False,
    )
    truth_path = dataset_root / CONFIGURATION_PATH / "measurements/truth.jsonl"
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(content)
    truth_path.unlink()
    truth_path.symlink_to(outside)

    with pytest.raises(DataError, match="escapes its pinned root"):
        dataset.measurement_case(CONFIGURATION_PATH)


@pytest.mark.parametrize("shared", [False, True])
def test_hub_cache_blob_symlinks_are_allowed_only_for_the_resolved_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shared: bool,
) -> None:
    snapshot = tmp_path / "datasets--nvidia--aisimulate-fpm-dataset" / "snapshots" / REVISION
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    _build_dataset(
        snapshot,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
        include_fpm=False,
    )
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    shared_root = tmp_path / "blobs"
    if shared:
        shared_root.mkdir()
        (shared_root / ".huggingface-shared-blobs").write_text("1\n")
    for source in tuple(path for path in snapshot.rglob("*") if path.is_file()):
        blob = blobs / hashlib.sha256(source.read_bytes()).hexdigest()
        if not blob.exists():
            if shared:
                target = shared_root / blob.name[:2] / blob.name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(source.read_bytes())
                blob.symlink_to(os.path.relpath(target, blob.parent))
            else:
                blob.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(os.path.relpath(blob, source.parent))

    class FakeApi:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def repo_info(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sha=REVISION)

    monkeypatch.setattr(dataset_module, "HfApi", FakeApi)
    monkeypatch.setattr(dataset_module, "snapshot_download", lambda **_kwargs: str(snapshot))

    dataset = HfDataset.from_hub(revision="main")

    assert dataset.revision == REVISION
    assert len(dataset.measurement_case(CONFIGURATION_PATH).observations) == 1
    if shared:
        # An unmarked sibling directory is not a Hub-owned shared store.
        (shared_root / ".huggingface-shared-blobs").unlink()
        with pytest.raises(DataError, match="escapes its pinned root"):
            HfDataset.from_hub(revision="main")


def test_hub_cache_snapshot_rejects_symlink_outside_its_own_blobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "datasets--nvidia--aisimulate-fpm-dataset" / "snapshots" / REVISION
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    _build_dataset(
        snapshot,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
        include_fpm=False,
    )
    blobs = snapshot.parent.parent / "blobs"
    blobs.mkdir()
    outside = tmp_path / "other-cache" / "same-content"
    outside.parent.mkdir()
    outside.write_bytes(content)
    truth = snapshot / CONFIGURATION_PATH / "measurements/truth.jsonl"
    truth.unlink()
    truth.symlink_to(outside)

    class FakeApi:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def repo_info(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sha=REVISION)

    monkeypatch.setattr(dataset_module, "HfApi", FakeApi)
    monkeypatch.setattr(dataset_module, "snapshot_download", lambda **_kwargs: str(snapshot))

    dataset = HfDataset.from_hub(revision="main")

    with pytest.raises(DataError, match="escapes its pinned root"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_local_git_dataset_must_match_a_clean_pinned_checkout(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    _build_dataset(
        root,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
        include_fpm=False,
    )
    subprocess.run(["git", "init", "-q", "-b", "main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "FPM Gym Test"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "fpm-gym@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert HfDataset.from_local(root, revision="main").revision == revision
    (root / "untracked.txt").write_text("not pinned", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must be clean"):
        HfDataset.from_local(root, revision=revision)


def test_fpm_roles_phases_and_sidecar_bytes_are_validated(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
    )
    configuration = dataset.configurations()[0]
    artifact = configuration.fpm_artifacts[0]
    metadata = json.loads(artifact.local_metadata_path.read_text())
    metadata["note"] = "changed after discovery"
    _write_json(artifact.local_metadata_path, metadata)
    with pytest.raises(DataError, match="hash mismatch"):
        dataset.measurement_case(CONFIGURATION_PATH)

    for field, value, message in (
        ("role", "unreviewed", "unknown FPM role"),
        ("phases", ["decode", "generation"], "invalid FPM phases"),
    ):
        root = tmp_path / field
        invalid = _build_dataset(
            root,
            protocol_id="forward-pass-measurement-v1",
            files=[("truth", "truth.jsonl", content)],
        )
        manifest_path = root / CONFIGURATION_PATH / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["fpm"][0][field] = value
        _write_json(manifest_path, manifest)
        with pytest.raises(DataError, match=message):
            invalid.configurations()


def test_fpm_physical_row_count_is_verified(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
    )
    fpm_path = tmp_path / CONFIGURATION_PATH / "fpm/fpm.parquet"
    pq.write_table(pa.table({"workload_kind": ["decode", "prefill"]}), fpm_path)
    fpm_sha256 = _sha256(fpm_path)
    metadata_path = tmp_path / CONFIGURATION_PATH / "fpm/fpm.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["parquet_sha256"] = fpm_sha256
    _write_json(metadata_path, metadata)
    manifest_path = tmp_path / CONFIGURATION_PATH / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fpm"][0]["sha256"] = fpm_sha256
    _write_json(manifest_path, manifest)

    with pytest.raises(DataError, match="has 2 rows; expected 1"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_fpm_sidecar_selector_must_match_selected_configuration(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
    )
    metadata_path = tmp_path / CONFIGURATION_PATH / "fpm/fpm.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["configuration_selector"]["model_path"] = "Other/Model"
    _write_json(metadata_path, metadata)

    with pytest.raises(DataError, match="selects a different configuration"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_fpm_parquet_identity_rows_must_match_selected_configuration(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
    )
    fpm_path = tmp_path / CONFIGURATION_PATH / "fpm/fpm.parquet"
    table = pq.read_table(fpm_path)
    model_index = table.schema.get_field_index("model_path")
    pq.write_table(table.set_column(model_index, "model_path", pa.array(["Other/Model"])), fpm_path)
    fpm_sha256 = _sha256(fpm_path)
    metadata_path = tmp_path / CONFIGURATION_PATH / "fpm/fpm.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["parquet_sha256"] = fpm_sha256
    _write_json(metadata_path, metadata)
    manifest_path = tmp_path / CONFIGURATION_PATH / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fpm"][0]["sha256"] = fpm_sha256
    _write_json(manifest_path, manifest)
    dataset = HfDataset.from_local(tmp_path, revision=REVISION)

    with pytest.raises(DataError, match="contains rows for a different configuration"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_fpm_parquet_requires_configuration_identity_columns(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
    )
    fpm_path = tmp_path / CONFIGURATION_PATH / "fpm/fpm.parquet"
    table = pq.read_table(fpm_path).drop(["dp"])
    pq.write_table(table, fpm_path)
    fpm_sha256 = _sha256(fpm_path)
    metadata_path = tmp_path / CONFIGURATION_PATH / "fpm/fpm.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["parquet_sha256"] = fpm_sha256
    _write_json(metadata_path, metadata)
    manifest_path = tmp_path / CONFIGURATION_PATH / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fpm"][0]["sha256"] = fpm_sha256
    _write_json(manifest_path, manifest)
    dataset = HfDataset.from_local(tmp_path, revision=REVISION)

    with pytest.raises(DataError, match="missing configuration identity columns"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_current_snapshot_does_not_select_historical_fpm_by_default(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
    )
    manifest_path = tmp_path / CONFIGURATION_PATH / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fpm"][0]["role"] = "historical"
    _write_json(manifest_path, manifest)

    assert dataset.measurement_case(CONFIGURATION_PATH).fpm_artifacts == ()
    selected = dataset.measurement_case(CONFIGURATION_PATH, fpm_artifact_ids=("fpm-1",))
    assert [artifact.artifact_id for artifact in selected.fpm_artifacts] == ["fpm-1"]


def test_empty_override_is_rejected(tmp_path: Path) -> None:
    override = tmp_path / "overrides.yaml"
    override.write_text(
        f"version: 1\noverrides:\n  - configuration_path: {CONFIGURATION_PATH}\n",
        encoding="utf-8",
    )
    _build_dataset(
        tmp_path / "dataset",
        protocol_id="forward-pass-measurement-v1",
        files=[],
    )
    with pytest.raises(ConfigurationError, match="at least one binding or correction"):
        HfDataset.from_local(tmp_path / "dataset", revision=REVISION, overrides_path=override)


def test_unknown_override_selector_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _build_dataset(
        dataset_root,
        protocol_id="forward-pass-measurement-v1",
        files=[],
    )
    override = tmp_path / "overrides.yaml"
    override.write_text(
        "version: 1\noverrides:\n  - configuration_path: data/not-present\n    worker_role: decode\n",
        encoding="utf-8",
    )
    dataset = HfDataset.from_local(dataset_root, revision=REVISION, overrides_path=override)

    with pytest.raises(ConfigurationError, match="unknown configuration snapshot"):
        dataset.configurations()


def test_unknown_measurement_role_is_rejected(tmp_path: Path) -> None:
    content = f"{json.dumps(_fpm_payload())}\n".encode()
    _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "truth.jsonl", content)],
        include_fpm=False,
    )
    measurement_path = tmp_path / CONFIGURATION_PATH / "measurements/manifest.json"
    measurement_manifest = json.loads(measurement_path.read_text())
    measurement_manifest["files"][0]["role"] = "prediction"
    _write_json(measurement_path, measurement_manifest)
    config_path = tmp_path / CONFIGURATION_PATH / "manifest.json"
    config_manifest = json.loads(config_path.read_text())
    config_manifest["measurements"]["manifest_sha256"] = _sha256(measurement_path)
    _write_json(config_path, config_manifest)
    dataset = HfDataset.from_local(tmp_path, revision=REVISION)

    with pytest.raises(DataError, match="unknown role"):
        dataset.measurement_case(CONFIGURATION_PATH)


def test_unknown_protocol_with_truth_stays_visible_without_guessing_parser(tmp_path):
    dataset = _build_dataset(
        tmp_path, protocol_id="future-v2", files=[("truth", "truth.jsonl", b"{}\n")], include_fpm=False
    )
    case = dataset.measurement_case(CONFIGURATION_PATH)
    assert case.status == "unsupported_protocol" and not case.observations
    assert len(case.truth_files) == 1
