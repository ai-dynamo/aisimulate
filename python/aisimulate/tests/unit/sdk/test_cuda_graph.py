# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from aiconfigurator_core.sdk._cuda_graph_component_model import (
    COMPONENT_CATEGORICAL_FEATURES,
    COMPONENT_NUMERIC_FEATURES,
    COMPONENT_TARGET_FIELDS,
    component_observation,
    required_components,
)
from aiconfigurator_core.sdk.cuda_graph import (
    CudaGraphProfileDatabaseError,
    CudaGraphReservationRequest,
    estimate_cuda_graph_reservation,
)

pytestmark = pytest.mark.unit

DATABASE = Path(__file__).parents[3] / "src/aiconfigurator_core/systems/cuda_graph_profiles/v1"


def _rows() -> list[dict[str, object]]:
    return pq.read_table(DATABASE / "cuda_graph_profiles.parquet").to_pylist()


def _request_from_row(row: dict[str, object]) -> CudaGraphReservationRequest:
    return CudaGraphReservationRequest(
        model_id=row["model_id"],
        model_revision=row["model_revision"],
        model_config_sha256=row["model_config_sha256"],
        system=row["system"],
        backend=row["backend"],
        backend_version=row["backend_version"],
        backend_build=row["backend_build"],
        tp_size=row["tp_size"],
        pp_size=row["pp_size"],
        attention_dp_size=row["attention_dp_size"],
        dcp_size=row["dcp_size"],
        pcp_size=row["pcp_size"],
        moe_tp_size=row["moe_tp_size"],
        moe_ep_size=row["moe_ep_size"],
        quantization=row["quantization"],
        compute_dtype=row["compute_dtype"],
        kv_cache_dtype=row["kv_cache_dtype"],
        cuda_graph_mode=row["cuda_graph_mode"],
        cuda_graph_capture_sizes=tuple(json.loads(row["cuda_graph_capture_sizes"])),
        compilation_mode=row["compilation_mode"],
        compilation_backend=row["compilation_backend"],
        moe_backend=row["moe_backend"],
        linear_backend=row["linear_backend"],
        flashinfer_autotune=row["flashinfer_autotune"],
        max_num_seqs=row["max_num_seqs"],
        max_num_batched_tokens=row["max_num_batched_tokens"],
        max_model_len=row["max_model_len"],
        attention_backend=row["attention_backend"],
        speculative_method=row["speculative_method"],
        speculative_tokens=row["speculative_tokens"],
    )


def _full_only_row() -> dict[str, object]:
    return next(
        row
        for row in _rows()
        if row["model_id"] == "deepseek-ai/DeepSeek-V4-Pro"
        and row["cuda_graph_mode"] == "FULL_DECODE_ONLY"
        and row["estimated_cuda_graph_bytes"] is not None
    )


def _external_database(tmp_path: Path) -> Path:
    destination = tmp_path / "cuda-graph-db"
    destination.mkdir()
    for name in (
        "cuda_graph_profiles.parquet",
        "cuda_graph_profiles.metadata.json",
        "cuda_graph_reservation_model.json",
    ):
        shutil.copy2(DATABASE / name, destination / name)
    return destination


def _enable_synthetic_model(database: Path, reference_row: dict[str, object]) -> None:
    model_path = database / "cuda_graph_reservation_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    components = {
        component_name: {
            "observation_count": 0,
            "observations": [],
            "target_field": target_field,
        }
        for component_name, target_field in COMPONENT_TARGET_FIELDS.items()
    }
    targets = {"full_first_capture": 1000, "full_per_graph": 10}
    for index in range(20):
        row = {**reference_row, "profile_id": f"synthetic-{index:02d}"}
        for component_name in required_components(row):
            observation = component_observation(row, component_name, target_bytes=targets[component_name])
            if index >= 10:
                observation["categorical"]["model_id"] = "second/model"
                observation["categorical"]["gpu_family"] = "b200"
            components[component_name]["observations"].append(observation)
            components[component_name]["observation_count"] += 1
    observations = [observation for component in components.values() for observation in component["observations"]]
    training_domain = {
        "categorical": {
            field: sorted({str(observation["categorical"][field]) for observation in observations})
            for field in COMPONENT_CATEGORICAL_FEATURES
        },
        "numeric": {
            field: [
                min(float(observation["numeric"][field]) for observation in observations),
                max(float(observation["numeric"][field]) for observation in observations),
            ]
            for field in COMPONENT_NUMERIC_FEATURES
        },
    }
    model.update(
        {
            "artifact_version": "cuda-graph-component-interpolation-v3",
            "enabled": True,
            "components": components,
            "training_domain": training_domain,
            "residual_log_interval": [0.0, math.log(1.2)],
            "training_profile_count": 20,
            "holdout_prediction_count": 20,
            "holdout_prediction_coverage": 1.0,
            "holdout_metrics": {
                "median_mape": 0.1,
                "p90_ape": 0.2,
                "upper_bound_coverage": 0.95,
                "maximum_underprediction": 0.1,
            },
            "gate_failures": [],
        }
    )
    model_path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_disabled_graph_returns_zero_without_loading_database(tmp_path: Path) -> None:
    request = CudaGraphReservationRequest(
        model_id="example/model",
        system="h200_sxm",
        cuda_graph_enabled=False,
    )
    estimate = estimate_cuda_graph_reservation(request, database_path=tmp_path / "missing")
    assert estimate.source == "disabled"
    assert estimate.reservation_bytes == 0


@pytest.mark.parametrize(
    ("model_id", "system", "tp_size", "attention_dp_size", "expected_gib"),
    [
        ("MiniMaxAI/MiniMax-M3-MXFP8", "h100_sxm", 8, 1, 0.87),
        ("nvidia/MiniMax-M3-NVFP4", "b200_sxm", 4, 1, 1.99),
        ("deepseek-ai/DeepSeek-V4-Pro", "h200_sxm", 8, 1, 1.47),
    ],
)
def test_exact_profile_lookup_numerical_regressions(
    model_id: str,
    system: str,
    tp_size: int,
    attention_dp_size: int,
    expected_gib: float,
) -> None:
    row = next(
        row
        for row in _rows()
        if row["model_id"] == model_id
        and row["system"] == system
        and row["tp_size"] == tp_size
        and row["attention_dp_size"] == attention_dp_size
        and row["estimated_cuda_graph_bytes"] not in {None, 0}
    )
    estimate = estimate_cuda_graph_reservation(_request_from_row(row))
    assert estimate.source == "profile"
    assert estimate.reservation_bytes == round(expected_gib * (1 << 30))


def test_unversioned_exact_profile_is_explicit() -> None:
    row = next(
        row for row in _rows() if row["model_revision"] is None and row["estimated_cuda_graph_bytes"] not in {None, 0}
    )
    estimate = estimate_cuda_graph_reservation(_request_from_row(row))
    assert estimate.source == "profile"
    assert estimate.identity_completeness == "unversioned_model"


def test_external_database_override(tmp_path: Path) -> None:
    database = _external_database(tmp_path)
    row = next(row for row in _rows() if row["estimated_cuda_graph_bytes"] not in {None, 0})
    estimate = estimate_cuda_graph_reservation(_request_from_row(row), database_path=database)
    assert estimate.source == "profile"


def test_modeled_lookup_returns_calibrated_upper_bound(tmp_path: Path) -> None:
    database = _external_database(tmp_path)
    row = _full_only_row()
    _enable_synthetic_model(database, row)
    request = replace(_request_from_row(row), backend_build="synthetic-build")
    estimate = estimate_cuda_graph_reservation(request, database_path=database)
    expected = 1000 + (int(row["cuda_graph_full_count"]) - 1) * 10
    assert estimate.source == "modeled"
    assert estimate.central_estimate_bytes == expected
    assert (
        estimate.reservation_bytes
        == estimate.interval_upper_bytes
        == round(math.expm1(math.log1p(expected) + math.log(1.2)))
    )


def test_out_of_domain_model_miss_is_unavailable(tmp_path: Path) -> None:
    database = _external_database(tmp_path)
    row = _full_only_row()
    _enable_synthetic_model(database, row)
    request = replace(
        _request_from_row(row),
        backend_build="synthetic-build",
        model_id="unobserved/model",
    )
    estimate = estimate_cuda_graph_reservation(request, database_path=database)
    assert estimate.source == "unavailable"
    assert estimate.miss_reason == "out_of_domain"


def test_out_of_numeric_domain_model_miss_is_unavailable(tmp_path: Path) -> None:
    database = _external_database(tmp_path)
    row = _full_only_row()
    _enable_synthetic_model(database, row)
    request = replace(
        _request_from_row(row),
        backend_build="synthetic-build",
        max_model_len=int(row["max_model_len"]) + 1,
    )
    estimate = estimate_cuda_graph_reservation(request, database_path=database)
    assert estimate.source == "unavailable"
    assert estimate.miss_reason == "out_of_domain"


def test_enabled_model_that_fails_declared_gates_is_rejected(tmp_path: Path) -> None:
    database = _external_database(tmp_path)
    row = _full_only_row()
    _enable_synthetic_model(database, row)
    model_path = database / "cuda_graph_reservation_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["training_profile_count"] = 19
    model_path.write_text(json.dumps(model), encoding="utf-8")
    request = replace(_request_from_row(row), backend_build="synthetic-build")
    with pytest.raises(CudaGraphProfileDatabaseError, match="does not pass"):
        estimate_cuda_graph_reservation(request, database_path=database)


def test_enabled_model_with_invalid_component_schema_is_rejected(tmp_path: Path) -> None:
    database = _external_database(tmp_path)
    row = _full_only_row()
    _enable_synthetic_model(database, row)
    model_path = database / "cuda_graph_reservation_model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["components"]["full_first_capture"]["observations"][0]["numeric"].pop("max_num_seqs")
    model_path.write_text(json.dumps(model), encoding="utf-8")
    request = replace(_request_from_row(row), backend_build="synthetic-build")
    with pytest.raises(CudaGraphProfileDatabaseError, match="safety-gate evidence"):
        estimate_cuda_graph_reservation(request, database_path=database)


def test_corrupt_database_fails_closed(tmp_path: Path) -> None:
    database = _external_database(tmp_path)
    metadata_path = database / "cuda_graph_profiles.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["parquet_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    request = CudaGraphReservationRequest(model_id="example/model", system="h200_sxm")
    with pytest.raises(CudaGraphProfileDatabaseError, match="checksum mismatch"):
        estimate_cuda_graph_reservation(request, database_path=database)
