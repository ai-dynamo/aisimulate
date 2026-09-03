# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.cuda_graph_profiles.common import GIB, profile_id
from tools.cuda_graph_profiles.infx import _extract_nested_tar
from tools.cuda_graph_profiles.parser import ProfileParseError, load_yaml, parse_log_text
from tools.cuda_graph_profiles.publish import (
    ProfileValidationError,
    _rank_range,
    _validate_component_reconstruction,
    _validate_duplicate_profiles,
    _validate_rows,
    validate_database,
)
from tools.cuda_graph_profiles.train import train_model

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


def test_profiled_graph_sets_are_parsed() -> None:
    parsed = parse_log_text(
        "\n".join(
            (
                "(Worker_TP0 pid=10) Profiling CUDA graph memory: PIECEWISE=49 (largest=512), FULL=49 (largest=512)",
                "(Worker_TP0 pid=10) Estimated CUDA graph memory: 1.99 GiB total",
            )
        )
    )
    assert parsed.profiled_full_count == 49
    assert parsed.profiled_full_largest_capture_size == 512
    assert parsed.profiled_piecewise_count == 49
    assert parsed.profiled_piecewise_largest_capture_size == 512


def test_graph_memory_components_are_parsed_and_reconstruct_total() -> None:
    parsed = parse_log_text(
        "\n".join(
            (
                "(Worker_TP0 pid=10) Profiling CUDA graph memory: PIECEWISE=3 (largest=4), FULL=2 (largest=2)",
                "(Worker_TP0 pid=10) Estimated PIECEWISE CUDA graph memory: "
                "128 MiB first-capture + (3 - 1) x 16 MiB per-graph",
                "(Worker_TP0 pid=10) Estimated FULL CUDA graph memory: "
                "224 MiB first-capture + (2 - 1) x 16 MiB per-graph",
                "(Worker_TP0 pid=10) Estimated CUDA graph memory: 0.27 GiB total",
            )
        )
    )
    assert parsed.graph_components_by_rank[0]["piecewise"].first_capture_bytes == 128 << 20
    assert parsed.graph_components_by_rank[0]["full"].per_graph_bytes == 16 << 20
    assert _validate_component_reconstruction(parsed)


def test_incompatible_repeated_graph_component_fails() -> None:
    with pytest.raises(ProfileParseError, match="incompatible rank-local full"):
        parse_log_text(
            "\n".join(
                (
                    "Estimated FULL CUDA graph memory: 128 MiB first-capture + (2 - 1) x 16 MiB per-graph",
                    "Estimated FULL CUDA graph memory: 256 MiB first-capture + (2 - 1) x 16 MiB per-graph",
                    "Estimated CUDA graph memory: 0.16 GiB total",
                )
            )
        )


def test_component_total_mismatch_fails_validation() -> None:
    parsed = parse_log_text(
        "\n".join(
            (
                "Profiling CUDA graph memory: FULL=2 (largest=2)",
                "Estimated FULL CUDA graph memory: 128 MiB first-capture + (2-1) x 16 MiB per-graph",
                "Estimated CUDA graph memory: 1.00 GiB total",
            )
        )
    )
    with pytest.raises(ProfileValidationError, match="do not reconstruct"):
        _validate_component_reconstruction(parsed)


def test_encoder_graph_component_is_preserved() -> None:
    parsed = parse_log_text(
        "\n".join(
            (
                "Profiling CUDA graph memory: FULL=1 (largest=1), ENCODER=2 (largest=512)",
                "Estimated FULL CUDA graph memory: 128 MiB first-capture + (1-1) x 16 MiB per-graph",
                "Estimated encoder CUDA graph memory: 64 MiB for 2 graphs",
                "Estimated CUDA graph memory: 0.19 GiB total",
            )
        )
    )
    assert parsed.profiled_encoder_count == 2
    assert parsed.profiled_encoder_largest_capture_size == 512
    assert parsed.encoder_graph_bytes_by_rank == {0: 64 << 20}
    assert _validate_component_reconstruction(parsed)


def test_benchmark_expert_parallelism_uses_engine_rank_topology() -> None:
    parsed = parse_log_text(
        "\n".join(
            (
                "Initializing a V1 LLM engine (v0.25.1) with config: model='example/model', "
                "tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=8, "
                "decode_context_parallel_size=1, quantization=fp8, dtype=torch.bfloat16, "
                "kv_cache_dtype=fp8, compilation_config={'mode': <CompilationMode.NONE: 0>, "
                "'backend': 'inductor', 'cudagraph_mode': <CUDAGraphMode.FULL_DECODE_ONLY: (2, 0)>, "
                "'cudagraph_capture_sizes': [1]}, kernel_config=KernelConfig("
                "enable_flashinfer_autotune=False, moe_backend='auto', linear_backend='auto')",
                "Using FLASH_ATTN attention backend",
                "(Worker_DP0_EP0 pid=10) Estimated CUDA graph memory: 1.00 GiB total",
            )
        ),
        benchmark={"tp": 8, "ep": 8, "dp_attention": "true"},
    )
    assert parsed.identity["tp_size"] == 1
    assert parsed.identity["attention_dp_size"] == 8
    assert parsed.identity["moe_tp_size"] == 1
    assert parsed.identity["moe_ep_size"] == 8


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


def test_component_training_row_requires_reconstructable_measurements() -> None:
    row = next(
        row for row in pq.read_table(DATABASE / "cuda_graph_profiles.parquet").to_pylist() if row["training_eligible"]
    )
    row["component_training_eligible"] = True
    row["component_exclusion_reason"] = None
    with pytest.raises(ProfileValidationError, match="missing full_first_capture_bytes"):
        _validate_rows([row])


def test_packaged_database_and_reports_validate() -> None:
    result = validate_database(DATABASE)
    assert result["status"] == "valid"
    assert result["measurement_count"] == 9
    assert result["model_enabled"] is False
    assert result["model_version"] == "cuda-graph-component-interpolation-v3"
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
    assert not any(row["component_training_eligible"] for row in rows)
    assert any(row["exclusion_reason"] == "actual_only_legacy_log" for row in rows)
    assert any(row["graph_disabled"] and row["estimated_cuda_graph_bytes"] == 0 for row in rows)
    assert all(row["model_architecture_config_sha256"] for row in rows if row["training_eligible"])
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


def test_component_model_training_passes_with_dense_in_domain_sweeps(tmp_path: Path) -> None:
    rows = []
    for model_id, system in (
        ("MiniMaxAI/MiniMax-M2.7", "h200_sxm"),
        ("deepseek-ai/DeepSeek-V4-Pro", "b200_sxm"),
    ):
        for index in range(10):
            rows.append(
                {
                    "attention_backend": "FLASH_ATTN",
                    "attention_dp_size": 1,
                    "backend_version": "0.25.1",
                    "compilation_backend": "inductor",
                    "compilation_mode": "NONE",
                    "component_training_eligible": True,
                    "compute_dtype": "bfloat16",
                    "cuda_graph_capture_sizes": "[1,2,4]",
                    "cuda_graph_mode": "FULL_DECODE_ONLY",
                    "dcp_size": 1,
                    "estimated_cuda_graph_bytes": 1020,
                    "flashinfer_autotune": False,
                    "full_first_capture_bytes": 1000,
                    "full_per_graph_bytes": 10,
                    "graph_disabled": False,
                    "kv_cache_dtype": "fp8",
                    "linear_backend": "auto",
                    "max_model_len": 4096 + index,
                    "max_num_batched_tokens": 4096,
                    "max_num_seqs": 8,
                    "measurement_id": f"{model_id}-{index}",
                    "model_id": model_id,
                    "moe_backend": "auto",
                    "moe_ep_size": 1,
                    "moe_tp_size": 4,
                    "pcp_size": 1,
                    "piecewise_first_capture_bytes": None,
                    "piecewise_per_graph_bytes": None,
                    "pp_size": 1,
                    "profile_id": f"{model_id}-{index}",
                    "quantization": "fp8",
                    "speculative_method": "none",
                    "speculative_tokens": 0,
                    "system": system,
                    "tp_size": 4,
                }
            )
    parquet_path = tmp_path / "profiles.parquet"
    model_path = tmp_path / "model.json"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    model = train_model(parquet_path, model_path)
    assert model["enabled"] is True, model["gate_failures"]
    assert model["holdout_prediction_coverage"] == 0.8
    assert model["holdout_metrics"]["median_mape"] == 0
