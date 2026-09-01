# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from tools.cuda_graph_profiles.common import GIB, profile_id
from tools.cuda_graph_profiles.infx import _extract_nested_tar
from tools.cuda_graph_profiles.parser import ProfileParseError, load_yaml, parse_log_text
from tools.cuda_graph_profiles.publish import (
    ProfileValidationError,
    _rank_range,
    _validate_duplicate_profiles,
    _validate_rows,
    validate_database,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parents[2] / "fixtures/cuda_graph_profiles"
DATABASE = Path(__file__).parents[3] / "src/aiconfigurator_core/systems/cuda_graph_profiles/v1"
REPORTS = Path(__file__).parents[3] / "tools/cuda_graph_profiles/reports/v1"
LOCK = Path(__file__).parents[3] / "tools/cuda_graph_profiles/infx_sources.lock.json"


def _parse(name: str):
    return parse_log_text((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("fixture", "expected_gib"),
    [
        ("deepseek_v4_h200_estimate.log", 1.47),
        ("minimax_m3_h100_pool_summary.log", 0.87),
        ("minimax_m3_b200_multiple_ranks.log", 1.99),
    ],
)
def test_reservation_parser_numerical_regressions(fixture: str, expected_gib: float) -> None:
    parsed = _parse(fixture)
    assert max(parsed.estimated_bytes_by_rank.values()) == round(expected_gib * GIB)


def test_pool_summary_preserves_actual_as_diagnostic() -> None:
    parsed = _parse("minimax_m3_h100_pool_summary.log")
    assert max(parsed.actual_bytes_by_rank.values()) == round(0.50 * GIB)
    assert max(parsed.estimated_bytes_by_rank.values()) == round(0.87 * GIB)


def test_actual_only_legacy_log_is_not_promoted_to_estimate() -> None:
    parsed = _parse("actual_only_legacy.log")
    assert parsed.estimated_bytes_by_rank == {}
    assert max(parsed.actual_bytes_by_rank.values()) == round(1.39 * GIB)


def test_explicit_disabled_log_records_zero() -> None:
    parsed = _parse("disabled.log")
    assert parsed.graph_disabled
    assert parsed.estimated_bytes_by_rank == {0: 0}


def test_multiple_rank_estimate_is_rank_local() -> None:
    parsed = _parse("minimax_m3_b200_multiple_ranks.log")
    assert len(parsed.estimated_bytes_by_rank) == 4
    assert max(parsed.estimated_bytes_by_rank.values()) == round(1.99 * GIB)


def test_malformed_log_fails() -> None:
    with pytest.raises(ProfileParseError, match="no CUDA graph reservation"):
        _parse("malformed.log")


def test_nested_multinode_archive_is_extracted_and_parsed(tmp_path: Path) -> None:
    archive = tmp_path / "nested.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for source in sorted((FIXTURES / "nested_multinode").iterdir()):
            data = source.read_bytes()
            member = tarfile.TarInfo(source.name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    extracted = tmp_path / "extracted"
    _extract_nested_tar(archive, extracted)
    parsed = parse_log_text(
        (extracted / "worker-0.out").read_text(encoding="utf-8"),
        config=load_yaml(extracted / "config.yaml"),
    )
    assert parsed.graph_disabled
    assert parsed.identity["model_revision"] == "0123456789012345678901234567890123456789"
    assert parsed.identity["system"] == "h200_sxm"


def test_profile_hash_is_deterministic_and_excludes_concurrency() -> None:
    row = {"model_id": "example/model", "system": "h200_sxm"}
    first = profile_id({**row, "concurrency": 1})
    second = profile_id({**row, "concurrency": 256})
    assert first == second


def test_duplicate_semantic_profile_above_five_percent_fails() -> None:
    rows = [
        {"profile_id": "same", "estimated_cuda_graph_bytes": 100},
        {"profile_id": "same", "estimated_cuda_graph_bytes": 106},
    ]
    with pytest.raises(ProfileValidationError, match="more than 5%"):
        _validate_duplicate_profiles(rows)


def test_rank_aggregation_uses_maximum_rank_local_reservation() -> None:
    minimum, maximum = _rank_range({0: 100, 1: 104}, "reservation", enforce_compatibility=True)
    assert (minimum, maximum) == (100, 104)


def test_incompatible_rank_reservations_fail() -> None:
    with pytest.raises(ProfileValidationError, match="incompatible rank-local reservation"):
        _rank_range({0: 100, 1: 106}, "reservation", enforce_compatibility=True)


def test_missing_provenance_fails_publication() -> None:
    row = pq.read_table(DATABASE / "cuda_graph_profiles.parquet").to_pylist()[0]
    row["system"] = None
    with pytest.raises(ProfileValidationError, match="missing provenance"):
        _validate_rows([row])


def test_packaged_database_and_reports_validate() -> None:
    result = validate_database(DATABASE)
    assert result["status"] == "valid"
    assert result["measurement_count"] == 7
    for report in (
        "source_mapping.report.json",
        "reconciliation.report.json",
        "exclusions.report.json",
        "validation.report.json",
    ):
        assert (REPORTS / report).is_file()


def test_database_has_required_measurement_classes_and_no_internal_paths() -> None:
    rows = pq.read_table(DATABASE / "cuda_graph_profiles.parquet").to_pylist()
    assert any(row["training_eligible"] for row in rows)
    assert any(row["exclusion_reason"] == "actual_only_legacy_log" for row in rows)
    assert any(row["graph_disabled"] and row["estimated_cuda_graph_bytes"] == 0 for row in rows)
    rendered = json.dumps(rows, sort_keys=True)
    for marker in ("/Users/", "/home/", "/tmp/", "/mnt/", "/scratch/", "/lustre/"):
        assert marker not in rendered


def test_source_lock_pins_run_attempt_artifact_and_extracted_files() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    assert lock["sources"]
    for source in lock["sources"]:
        assert source["run_id"] and source["run_attempt"] and len(source["head_sha"]) == 40
        for artifact in source["artifacts"]:
            assert artifact["artifact_id"] and artifact["artifact_name"] and artifact["files"]
            assert all(len(file["sha256"]) == 64 for file in artifact["files"])
