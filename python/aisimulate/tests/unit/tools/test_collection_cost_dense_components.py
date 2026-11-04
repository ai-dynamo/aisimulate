# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conditional components require identification and independent validation."""

import copy

import pytest

from tools.perf_database import collection_cost_dense_components as fit

pytestmark = pytest.mark.unit


def design(operation="attention_context"):
    rows = []
    lengths = (16384, 32768) if operation == "attention_context" else (65535, 262143)
    for replicate in (1, 2, 3):
        for heads, kv in ((4, 1), (32, 2), (16, 1)):
            for batch in (1, 4):
                for length in lengths:
                    cache = max(131072, batch * length) * kv * 128 * 4
                    inputs = cache + batch * length * heads * 128 * 2
                    row = {
                        "sample_id": f"r{replicate}-h{heads}-b{batch}-l{length}",
                        "status": "passed",
                        "replicate": replicate,
                        "group_id": f"worker-{replicate}",
                        "native_pass_index": 2,
                        "geometry_role": "whole_geometry_holdout" if heads == 16 else "fit",
                        "system": "test_gpu",
                        "backend": "test_backend",
                        "operation": operation,
                        "source_commit": "native",
                        "audit_driver_sha256": "driver",
                        "observer_sha256": "observer",
                        "actual_kernel_source": "fixed-native-dispatch",
                        "runtime_protocol": {"image_sha256": "image"},
                        "case": {
                            "batch_size": batch,
                            "input_len": length,
                            "num_heads": heads,
                            "num_kv_heads": kv,
                            "head_dim": 128,
                            "attn_dtype": "bf16",
                            "kv_cache_dtype": "bf16",
                            "window_size": 0,
                            "dispatch": "planned",
                        },
                        "num_runs_executed": 6,
                        "requested_num_warmups": 5,
                        "repeat_n": 1,
                        "allocation_observations": [{"tensors": [{"storage_bytes": cache}]}],
                        "kv_unique_allocation_bytes": cache,
                        "forward_input_unique_storage_bytes": inputs,
                    }
                    preparation = 0.01 + 2e-9 * cache + 3e-9 * inputs
                    execution = 0.02 + 1e-14 * fit.features(row, "execution")[1]
                    row.update(
                        preparation_seconds=preparation,
                        execution_seconds=execution,
                        collection_seconds=preparation + execution,
                        phase_seconds={
                            key: execution * fraction
                            for key, fraction in zip(fit.PHASES, (0.2, 0.1, 0.1, 0.55, 0.05), strict=True)
                        },
                    )
                    rows.append(row)
    return rows


@pytest.mark.parametrize("operation", ["attention_context", "attention_generation"])
def test_identified_components_pass_whole_geometry_and_fresh_group_gates(operation):
    result = fit.fit_samples(design(operation))
    assert len(result["pools"]) == 1
    for component in ("preparation", "execution"):
        model = result["pools"][0][component]
        assert model["qualified"]
        assert model["max_whole_geometry_holdout_error"] < 1e-10
        assert model["independent_third_component_error"] < 1e-10
        assert all(c >= 0 for c in model["coefficients"])
    assert result["whole_worker_gpu_hours"] is result["cold_setup_gpu_hours"] is result["fleet_gpu_hours"] is None
    assert "proxy" in result["execution_feature"]


def test_holdout_geometry_does_not_leak_into_coefficient_fit():
    rows = design()
    original = fit.fit_samples(rows)["pools"][0]["preparation"]["coefficients"]
    for row in rows:
        if row["geometry_role"] == "whole_geometry_holdout":
            row["preparation_seconds"] *= 2
            row["collection_seconds"] = row["preparation_seconds"] + row["execution_seconds"]
    model = fit.fit_samples(rows)["pools"][0]["preparation"]
    assert model["coefficients"] == original
    assert not model["qualified"] and model["max_whole_geometry_holdout_error"] > 0.20


def test_independent_component_gate_fails_even_when_geometry_points_pass():
    rows = design()
    for row in rows:
        if row["replicate"] == 3:
            row["preparation_seconds"] *= 1.2
            row["collection_seconds"] = row["preparation_seconds"] + row["execution_seconds"]
    model = fit.fit_samples(rows)["pools"][0]["preparation"]
    assert model["max_whole_geometry_holdout_error"] < 0.20
    assert model["independent_third_component_error"] > 0.15
    assert not model["qualified"]


def test_singular_storage_design_is_not_identified():
    rows = design()
    for row in rows:
        row["forward_input_unique_storage_bytes"] = row["kv_unique_allocation_bytes"] * 2
    model = fit.fit_samples(rows)["pools"][0]["preparation"]
    assert not model["qualified"] and "rank-deficient" in model["reason"]


def test_nonphysical_negative_coefficient_is_rejected():
    rows = design()
    for row in rows:
        row["preparation_seconds"] = 50 - 1e-9 * row["kv_unique_allocation_bytes"]
        row["collection_seconds"] = row["preparation_seconds"] + row["execution_seconds"]
    model = fit.fit_samples(rows)["pools"][0]["preparation"]
    assert not model["qualified"] and "negative/nonphysical" in model["reason"]


def test_no_allocation_inference_or_overlapping_phase_addition():
    rows = design()
    missing = copy.deepcopy(rows[0])
    missing["sample_id"] = "missing"
    missing.pop("allocation_observations")
    overlap = copy.deepcopy(rows[0])
    overlap["sample_id"] = "overlap"
    overlap["execution_seconds"] *= 2
    result = fit.fit_samples([missing, overlap])
    assert not result["pools"] and len(result["excluded"]) == 2


def test_kernel_changes_split_and_unqualify_incomplete_geometry_design():
    rows = design()
    for row in rows:
        if row["geometry_role"] == "whole_geometry_holdout":
            row["actual_kernel_source"] = "different-kernel"
    result = fit.fit_samples(rows)
    assert len(result["pools"]) == 2
    assert all(not p["design_qualified"] for p in result["pools"])


def test_first_pass_is_not_compiler_warm_or_extra_training():
    rows = design()
    first = {**rows[0], "sample_id": "first-pass", "native_pass_index": 1}
    result = fit.fit_samples([*rows, first])
    assert result["excluded"] == [{"sample_id": "first-pass", "reason": "outside subsequent-in-worker pass-2 scope"}]
    assert "does not establish compiler-warm" in result["scope"]


def test_shared_worker_ids_and_holdout_role_leakage_reject_design():
    for mutate in (
        lambda r: r.update(group_id="one-worker"),
        lambda r: r.update(geometry_role="fit") if fit.geometry(r) == fit.HOLDOUT_GEOMETRY else None,
    ):
        rows = design()
        for row in rows:
            mutate(row)
        assert not fit.fit_samples(rows)["pools"][0]["design_qualified"]
