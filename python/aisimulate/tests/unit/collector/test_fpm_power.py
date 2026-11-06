# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPM power window: schema-3 rank artifacts and schema-8 publication (AISIM-1973)."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward.config import FPMCollectionOptions, add_fpm_arguments, reject_fpm_arguments_without_fpm
from collector.fpm_forward.database import (
    POWER_WINDOW_MEASUREMENT_POLICY,
    SINGLE_SAMPLE_MEASUREMENT_POLICY,
    aggregate_cell,
    validate_formal_database_commit,
    write_formal_database,
)
from collector.fpm_forward.native_artifact import select_native_measurements, validate_native_collection
from collector.fpm_forward.planner import BackendPolicy, FPMCell
from collector.fpm_forward.runtime_memory import saved_plan_identity
from collector.fpm_forward.types import ParallelTopology

from aisimulate.fpm_contract import FPM_POWER_WINDOW_REL_TOL
from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS

from .test_fpm_profile_collection import _argv, _plan, _profile, no_models_or_timing_data  # noqa: F401

pytestmark = pytest.mark.unit

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def _power(energy=(), *, window_s=2.5, iterations=200, drift=0):
    """A measured block in Dynamo's shape (_bench_power_block)."""
    return {
        "status": "measured",
        "unavailable_reason": None,
        "gpu_energy": [{"gpu_uuid": uuid, "energy_wms": value} for uuid, value in energy],
        "window_s": window_s,
        "window_iterations": iterations,
        "kv_drift_tokens": drift,
    }


def _unavailable(reason="nvml_permission"):
    """Dynamo's _bench_power_unavailable block."""
    return {
        "status": "unavailable",
        "unavailable_reason": reason,
        "gpu_energy": [],
        "window_s": None,
        "window_iterations": None,
        "kv_drift_tokens": None,
    }


def _point(phase, benchmark_id, *, batch=4, total_kv=1280, sample_reasons=("capture",)):
    return {
        "point_type": phase,
        "benchmark_id": benchmark_id,
        "total_prefill_tokens": 257 if phase == "prefill" else 0,
        "total_kv_read_tokens": total_kv,
        "batch_size": batch,
        "sample_reasons": list(sample_reasons),
    }


def _scheduled(point):
    prefill = point["point_type"] == "prefill"
    return {
        "num_prefill_requests": point["batch_size"] if prefill else 0,
        "sum_prefill_tokens": point["total_prefill_tokens"],
        "sum_prefill_kv_tokens": point["total_kv_read_tokens"] if prefill else 0,
        "num_decode_requests": 0 if prefill else point["batch_size"],
        "sum_decode_kv_tokens": 0 if prefill else point["total_kv_read_tokens"],
    }


def _write_cell(
    tmp_path,
    *,
    phase="decode",
    tp=1,
    dp=1,
    points=None,
    powers=None,
    schema_version=3,
    walls=None,
    cell=None,
    plan_sha256="plan-sha",
):
    """Write one rank artifact per DP rank in PR 1's envelope.

    ``powers[benchmark_id][rank]`` is that rank's power block (omitted for
    schema 2); ``walls[benchmark_id][rank]`` its latency-sample wall time.
    ``cell`` and ``plan_sha256`` bind the artifacts to a real saved plan.
    """
    points = points or [_point(phase, 1)]
    cell = cell or FPMCell(
        cell_id=f"fpm-power-{phase}",
        workload_kind=phase,
        topology=ParallelTopology(tp=tp, pp=1, dp=dp, moe_tp=1, moe_ep=tp * dp, cp=1),
        weight_quantization="fp8",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline_auto", {}, {}),
        parallel_strategy="tp",
        gemm_quant_mode="fp8",
        moe_quant_mode="fp8",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )
    plan = SimpleNamespace(
        sha256=plan_sha256,
        aic_revision="revision",
        model_path="org/model",
        system="h200_sxm",
        backend="vllm",
        cells=(cell,),
        options=SimpleNamespace(warmup_iterations=0, measure_power=True),
        capability=SimpleNamespace(
            support_level="exact",
            template_id="aic_exact:dense",
            template_version=1,
            aic_database_version="0.24.0",
        ),
    )
    fpms = {}
    for point in points:
        for rank in range(dp):
            wall = (walls or {}).get(point["benchmark_id"], {}).get(rank, 0.004 + 0.002 * rank)
            fpm = {
                "counter_id": point["benchmark_id"],
                "dp_rank": rank,
                "wall_time": wall,
                "scheduled_requests": _scheduled(point),
            }
            if schema_version == 3:
                fpm["power"] = powers[point["benchmark_id"]][rank]
            fpms[(point["benchmark_id"], rank)] = fpm
    groups = []
    for point in points:
        rank_fpms = [fpms[(point["benchmark_id"], rank)] for rank in range(dp)]
        groups.append(
            {
                "benchmark_id": point["benchmark_id"],
                "point": point,
                "expected_dp_ranks": list(range(dp)),
                "complete": True,
                "wall_time": max(fpm["wall_time"] for fpm in rank_fpms),
                "rank_results": [{"dp_rank": rank, "fpms": [fpm]} for rank, fpm in enumerate(rank_fpms)],
            }
        )
    measured = sum(group["wall_time"] for group in groups)
    cell_dir = tmp_path / "cell"
    for rank in range(dp):
        output = cell_dir / "raw" / f"pod-{rank}" / ("benchmark.json" if rank == 0 else f"benchmark_dp{rank}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        (output.parent / "collector-provenance.json").write_text(
            json.dumps(
                {
                    "schema_name": "aic_fpm_collector_provenance",
                    "schema_version": 1,
                    "cell_id": cell.cell_id,
                    "plan_sha256": plan.sha256,
                    "attempt_id": "attempt",
                    "runtime": {"backend": "vllm", "backend_version": "0.24.0"},
                }
            )
        )
        config = {"mode": phase, **({"power_window_s": 2.0} if schema_version == 3 else {})}
        output.write_text(
            json.dumps(
                {
                    "schema_version": schema_version,
                    "artifact_type": "rank",
                    "status": "complete",
                    "valid": True,
                    "usable": True,
                    "timing_valid": True,
                    "stop_reason": None,
                    "error": None,
                    "run_id": "run",
                    "grid_digest": "grid",
                    "config": config,
                    "coverage": {"expected_points": len(points), "completed_points": len(points), "skipped_points": 0},
                    "dp": {"rank": rank, "size": dp},
                    "results": [{"point": point, "fpms": [fpms[(point["benchmark_id"], rank)]]} for point in points],
                    "iteration_groups": groups,
                    "skipped_points": [],
                    "missing_phases": [],
                    # Power-window iterations are never FPM samples, so they
                    # stay out of measured_iteration_seconds.
                    "timing": {"benchmark_elapsed_seconds": 60.0, "measured_iteration_seconds": measured},
                }
            )
        )
    return plan, cell, cell_dir


def _rank_paths(cell_dir):
    return sorted((cell_dir / "raw").glob("*/benchmark*.json"))


def _edit_ranks(cell_dir, edit):
    for path in _rank_paths(cell_dir):
        payload = json.loads(path.read_text())
        edit(payload)
        path.write_text(json.dumps(payload))


def _tp2_decode(tmp_path, power=None, **kwargs):
    power = power or _power([("GPU-a", 600_000.0), ("GPU-b", 640_000.0)], window_s=2.0, drift=40)
    return _write_cell(tmp_path, phase="decode", tp=2, powers={1: {0: power}}, **kwargs)


def _dep2_prefill(tmp_path, rank1=None, **kwargs):
    rank0 = _power([("GPU-0", 500_000.0)], window_s=2.5)
    rank1 = rank1 or _power([("GPU-1", 520_000.0)], window_s=2.5)
    return _write_cell(tmp_path, phase="prefill", dp=2, powers={1: {0: rank0, 1: rank1}}, **kwargs)


def test_tp_power_averages_every_gpu_the_rank_drives(tmp_path):
    plan, cell, cell_dir = _tp2_decode(tmp_path)

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    # (600,000 + 640,000) W*ms / 2 GPUs / (2.0 s * 1000) = 310 W per GPU.
    assert rows[0]["power_w"] == pytest.approx(310.0)
    assert rows[0]["latency_ms"] == pytest.approx(4.0)
    assert rows[0]["measurement_policy"] == POWER_WINDOW_MEASUREMENT_POLICY
    assert rows[0]["measurement_repeats"] == 1


def test_dep_power_sums_energy_across_ranks(tmp_path):
    plan, cell, cell_dir = _dep2_prefill(tmp_path)

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    # (500,000 + 520,000) W*ms / 2 GPUs / (2.5 s * 1000) = 204 W per GPU.
    assert rows[0]["power_w"] == pytest.approx(204.0)
    # Latency stays the slowest rank's latency sample.
    assert rows[0]["latency_ms"] == pytest.approx(6.0)


@pytest.mark.parametrize("reason", ["nvml_permission", "prefill_cache_mismatch", "any free-form text"])
def test_unavailable_rank_keeps_latency_without_power(tmp_path, caplog, reason):
    plan, cell, cell_dir = _dep2_prefill(tmp_path, rank1=_unavailable(reason))

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert rows[0]["power_w"] is None
    assert rows[0]["latency_ms"] == pytest.approx(6.0)
    assert rows[0]["measurement_policy"] == POWER_WINDOW_MEASUREMENT_POLICY
    assert "1 of 2 ranks measured power" in caplog.text


def _mutate(power, mutation):
    if mutation == "unavailable-energy":
        power.update(_unavailable(), gpu_energy=[{"gpu_uuid": "GPU-a", "energy_wms": 1.0}])
    elif mutation == "unavailable-window":
        power.update(_unavailable(), window_s=1.5)
    elif mutation == "unavailable-iterations":
        power.update(_unavailable(), window_iterations=3)
    elif mutation == "unavailable-drift":
        power.update(_unavailable(), kv_drift_tokens=0)
    elif mutation == "unavailable-empty-reason":
        power.update(_unavailable(), unavailable_reason=" ")
    elif mutation == "measured-reason":
        power["unavailable_reason"] = "nvml_error"
    elif mutation == "unknown-status":
        power["status"] = "partial"
    elif mutation == "missing-key":
        del power["kv_drift_tokens"]
    elif mutation == "extra-key":
        power["source"] = "nvml"
    elif mutation == "no-energy":
        power["gpu_energy"] = []
    elif mutation in {"nan-energy", "zero-energy", "negative-energy", "bool-energy"}:
        value = {"nan-energy": float("nan"), "zero-energy": 0.0, "negative-energy": -1.0, "bool-energy": True}
        power["gpu_energy"][0]["energy_wms"] = value[mutation]
    elif mutation == "empty-uuid":
        power["gpu_energy"][0]["gpu_uuid"] = ""
    elif mutation == "short-window":
        # Dynamo reports window_below_minimum instead of a short measurement.
        power["window_s"] = 1.999
    elif mutation == "zero-iterations":
        power["window_iterations"] = 0
    elif mutation == "bool-iterations":
        power["window_iterations"] = True
    elif mutation == "negative-drift":
        power["kv_drift_tokens"] = -1
    elif mutation == "float-drift":
        power["kv_drift_tokens"] = 1.0
    else:
        raise AssertionError(mutation)


@pytest.mark.parametrize(
    "mutation",
    [
        "unavailable-energy",
        "unavailable-window",
        "unavailable-iterations",
        "unavailable-drift",
        "unavailable-empty-reason",
        "measured-reason",
        "unknown-status",
        "missing-key",
        "extra-key",
        "no-energy",
        "nan-energy",
        "zero-energy",
        "negative-energy",
        "bool-energy",
        "empty-uuid",
        "short-window",
        "zero-iterations",
        "bool-iterations",
        "negative-drift",
        "float-drift",
    ],
)
def test_malformed_power_block_raises(tmp_path, mutation):
    power = _power([("GPU-a", 600_000.0), ("GPU-b", 640_000.0)], window_s=2.0, drift=40)
    _mutate(power, mutation)
    plan, cell, cell_dir = _tp2_decode(tmp_path, power=power)

    with pytest.raises(ValueError, match="power"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_prefill_power_must_report_zero_drift(tmp_path):
    plan, cell, cell_dir = _dep2_prefill(tmp_path, rank1=_power([("GPU-1", 520_000.0)], window_s=2.5, drift=1))

    with pytest.raises(ValueError, match="kv_drift_tokens is invalid for prefill"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


@pytest.mark.parametrize(
    ("energy", "message"),
    [
        ([("GPU-0", 520_000.0)], "reports a GPU twice"),
        ([("GPU-1", 260_000.0), ("GPU-2", 260_000.0)], "covers 3 GPUs, expected 2"),
    ],
)
def test_cross_rank_gpu_identity_must_cover_the_topology(tmp_path, energy, message):
    plan, cell, cell_dir = _dep2_prefill(tmp_path, rank1=_power(energy, window_s=2.5))

    with pytest.raises(ValueError, match=message):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_tp_power_requires_every_gpu_of_the_rank(tmp_path):
    plan, cell, cell_dir = _tp2_decode(tmp_path, power=_power([("GPU-a", 600_000.0)], window_s=2.0))

    with pytest.raises(ValueError, match="covers 1 GPUs, expected 2"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_window_agreement_within_tolerance_uses_the_longest_window(tmp_path):
    shorter = 2.5 * (1 - 0.9 * FPM_POWER_WINDOW_REL_TOL)
    plan, cell, cell_dir = _dep2_prefill(tmp_path, rank1=_power([("GPU-1", 520_000.0)], window_s=shorter))

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    # The group window is the longest rank window, 2.5 s:
    # (500,000 + 520,000) W*ms / 2 GPUs / 2,500 ms = 204 W per GPU.
    assert rows[0]["power_w"] == pytest.approx(204.0)


@pytest.mark.parametrize(
    ("rank1", "message"),
    [
        (_power([("GPU-1", 520_000.0)], window_s=2.5 * (1 - 1.1 * FPM_POWER_WINDOW_REL_TOL)), "window_s differs"),
        (_power([("GPU-1", 520_000.0)], window_s=2.5, iterations=201), "window_iterations differ"),
    ],
)
def test_window_disagreement_across_ranks_raises(tmp_path, rank1, message):
    plan, cell, cell_dir = _dep2_prefill(tmp_path, rank1=rank1)

    with pytest.raises(ValueError, match=message):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


@pytest.mark.parametrize(("drift", "powered"), [(128, True), (129, False)])
def test_decode_kv_drift_gate_keeps_latency(tmp_path, caplog, drift, powered):
    # total_kv_read_tokens is 1280, so 10% allows 128 appended tokens.
    power = _power([("GPU-a", 600_000.0), ("GPU-b", 640_000.0)], window_s=2.0, drift=drift)
    plan, cell, cell_dir = _tp2_decode(tmp_path, power=power)

    with caplog.at_level("INFO"):
        rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert (rows[0]["power_w"] is not None) is powered
    assert rows[0]["latency_ms"] == pytest.approx(4.0)
    assert f"{0 if powered else 1} above the KV drift bound" in caplog.text


def test_local_power_must_match_the_synchronized_group_copy(tmp_path):
    plan, cell, cell_dir = _dep2_prefill(tmp_path)

    def edit(payload):
        if payload["dp"]["rank"] == 1:
            payload["results"][0]["fpms"][0]["power"]["gpu_energy"][0]["energy_wms"] = 521_000.0

    _edit_ranks(cell_dir, edit)

    with pytest.raises(ValueError, match="local result differs from synchronized group"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_power_window_time_stays_out_of_measured_iteration_seconds(tmp_path):
    plan, cell, cell_dir = _dep2_prefill(tmp_path)
    collection = validate_native_collection(cell, cell_dir / "raw", expected_attempt_id="attempt")
    assert collection.artifact_schema_version == 3

    def include_window(payload):
        payload["timing"]["measured_iteration_seconds"] += 2.5

    _edit_ranks(cell_dir, include_window)

    with pytest.raises(ValueError, match="measured timing mismatch"):
        validate_native_collection(cell, cell_dir / "raw", expected_attempt_id="attempt")


def test_rank_artifacts_must_share_one_schema(tmp_path):
    plan, cell, cell_dir = _dep2_prefill(tmp_path)

    def downgrade(payload):
        if payload["dp"]["rank"] == 1:
            payload["schema_version"] = 2

    _edit_ranks(cell_dir, downgrade)

    with pytest.raises(ValueError, match="different artifact schema versions"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_schema_three_requires_power_on_every_fpm(tmp_path):
    plan, cell, cell_dir = _dep2_prefill(tmp_path)
    _edit_ranks(cell_dir, lambda payload: payload["results"][0]["fpms"][0].pop("power"))

    with pytest.raises(ValueError, match="lacks its power block"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


@pytest.mark.parametrize("schema_version", [1, 4, True, 3.0, "3"])
def test_unknown_artifact_schema_is_rejected(tmp_path, schema_version):
    plan, cell, cell_dir = _dep2_prefill(tmp_path)
    _edit_ranks(cell_dir, lambda payload: payload.update(schema_version=schema_version))

    with pytest.raises(ValueError, match="not a PR11509 native rank artifact"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_schema_two_artifact_cannot_carry_power(tmp_path):
    plan, cell, cell_dir = _dep2_prefill(tmp_path)
    _edit_ranks(cell_dir, lambda payload: payload.update(schema_version=2))

    with pytest.raises(ValueError, match="carries power in a schema-2 artifact"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_schema_two_artifact_stays_latency_only(tmp_path):
    plan, cell, cell_dir = _write_cell(tmp_path, phase="prefill", dp=2, schema_version=2)

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    parquet, metadata, _skipped = write_formal_database(plan, rows, systems_root=tmp_path / "systems")

    assert "power_w" not in rows[0]
    assert rows[0]["measurement_policy"] == SINGLE_SAMPLE_MEASUREMENT_POLICY
    sidecar = json.loads(metadata.read_text())
    assert (sidecar["schema_version"], sidecar["measurement_policy"]) == (7, SINGLE_SAMPLE_MEASUREMENT_POLICY)
    assert "power_w" not in pq.read_table(parquet).column_names


def test_power_travels_with_the_selected_duplicate_sample(tmp_path):
    # Two requested points clamp onto one physical coordinate; the unclamped
    # native sample is published with its own power, never a mean.
    points = [_point("decode", 1, sample_reasons=("context_clamped",)), _point("decode", 2)]
    powers = {
        1: {0: _power([("GPU-a", 400_000.0), ("GPU-b", 400_000.0)], window_s=2.0)},
        2: {0: _power([("GPU-a", 600_000.0), ("GPU-b", 640_000.0)], window_s=2.0)},
    }
    walls = {1: {0: 0.009}, 2: {0: 0.004}}
    plan, cell, cell_dir = _write_cell(tmp_path, phase="decode", tp=2, points=points, powers=powers, walls=walls)

    collection = validate_native_collection(cell, cell_dir / "raw", expected_attempt_id="attempt")
    (selected,) = select_native_measurements(collection, cell_id=cell.cell_id)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert selected.point["benchmark_id"] == 2
    assert [row["power_w"] for row in rows] == [pytest.approx(310.0)]
    assert [row["latency_ms"] for row in rows] == [pytest.approx(4.0)]


def _seal_as_schema_6(metadata_path, parquet_path):
    """Rewrite a sealed schema-7 pair into the historical schema-6 layout."""
    pq.write_table(pq.read_table(parquet_path).drop(list(EXECUTION_COLUMNS)), parquet_path)
    sidecar = json.loads(metadata_path.read_text())
    sidecar.update(schema_version=6, parquet_sha256=hashlib.sha256(parquet_path.read_bytes()).hexdigest())
    metadata_path.write_text(json.dumps(sidecar))


@pytest.mark.parametrize("existing_schema", [6, 7])
def test_powered_run_republishes_an_existing_pair_as_schema_eight(tmp_path, existing_schema):
    systems_root = tmp_path / "systems"
    old_plan, old_cell, old_dir = _write_cell(tmp_path / "old", phase="prefill", dp=2, schema_version=2)
    old_rows = aggregate_cell(old_plan, old_cell, old_dir, expected_attempt_id="attempt")
    parquet, metadata, _ = write_formal_database(old_plan, old_rows, systems_root=systems_root)
    if existing_schema == 6:
        _seal_as_schema_6(metadata, parquet)

    plan, cell, cell_dir = _tp2_decode(tmp_path / "new")
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    # The same prefill cell republished by another attempt stays first-publisher-owned.
    rerun = [dict(old_rows[0], collector_attempt_id="attempt-2", runtime_run_id="run-2")]
    parquet, metadata, skipped = write_formal_database(plan, rows + rerun, systems_root=systems_root)

    assert skipped == (old_cell.cell_id,)
    sidecar = json.loads(metadata.read_text())
    assert sidecar["schema_version"] == 8
    assert sidecar["measurement_policy"] == POWER_WINDOW_MEASUREMENT_POLICY
    assert sidecar["measurement_repeats"] == 1
    table = pq.read_table(parquet)
    assert table.schema.field("power_w").type == pa.float64()
    published = {row["cell_id"]: row for row in table.to_pylist()}
    assert published[old_cell.cell_id]["power_w"] is None
    assert published[old_cell.cell_id]["measurement_policy"] == SINGLE_SAMPLE_MEASUREMENT_POLICY
    assert published[old_cell.cell_id]["collector_attempt_id"] == "attempt"
    assert published[cell.cell_id]["power_w"] == pytest.approx(310.0)

    # A latency-only run merged later keeps the pair at schema 8.
    later_plan, later_cell, later_dir = _write_cell(tmp_path / "later", phase="decode", schema_version=2)
    later_rows = [
        dict(row, cell_id="fpm-power-later")
        for row in aggregate_cell(later_plan, later_cell, later_dir, expected_attempt_id="attempt")
    ]
    parquet, metadata, _ = write_formal_database(later_plan, later_rows, systems_root=systems_root)
    assert json.loads(metadata.read_text())["schema_version"] == 8
    later = [row for row in pq.read_table(parquet).to_pylist() if row["cell_id"] == "fpm-power-later"]
    assert {row["power_w"] for row in later} == {None}


@pytest.mark.parametrize("plan_power", [False, None])
def test_power_artifacts_require_a_plan_that_requested_power(tmp_path, plan_power):
    # An image-level DYN_BENCHMARK_POWER_WINDOW_S can write schema 3 under a
    # plan whose identity says no power; never publish it as a v8 power run.
    plan, cell, cell_dir = _dep2_prefill(tmp_path)
    if plan_power is None:
        del plan.options.measure_power  # plans saved before the field
    else:
        plan.options.measure_power = plan_power

    with pytest.raises(ValueError, match="measure_power"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def _saved_tep_prefill(tmp_path, frozen):
    """Write schema-3 artifacts for a saved plan's single-rank prefill cell."""
    cell = next(cell for cell in frozen.cells if cell.workload_kind == "prefill" and cell.topology.dp == 1)
    power = _power([(f"GPU-{gpu}", 500_000.0) for gpu in range(cell.topology.tp)], window_s=2.5)
    _, _, cell_dir = _write_cell(
        tmp_path,
        phase="prefill",
        tp=cell.topology.tp,
        powers={1: {0: power}},
        cell=cell,
        plan_sha256=frozen.sha256,
    )
    return cell, cell_dir


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_finalization_aggregates_power_through_the_saved_plan(tmp_path):
    # Onboarding finalization aggregates against the identity reread from
    # collection-plan.json, not the live plan object.
    plan = _plan(_profile())
    assert plan.options.measure_power
    frozen = saved_plan_identity(plan.to_dict())
    cell, cell_dir = _saved_tep_prefill(tmp_path, frozen)

    rows = aggregate_cell(frozen, cell, cell_dir, expected_attempt_id="attempt")

    # 8 GPUs x 500,000 W*ms / 8 GPUs / (2.5 s * 1000) = 200 W per GPU.
    assert rows[0]["power_w"] == pytest.approx(200.0)
    assert rows[0]["measurement_policy"] == POWER_WINDOW_MEASUREMENT_POLICY
    assert rows[0]["source_plan_sha256"] == frozen.sha256
    assert frozen.options.measure_power is True


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_saved_plan_without_measure_power_reads_false(tmp_path):
    from collector.fpm_forward import cli

    profile = _profile()
    options = FPMCollectionOptions.from_args(cli._parser().parse_args([*_argv(profile), "--no-fpm-measure-power"]))
    payload = _plan(profile, options=options).to_dict()
    assert "measure_power" not in payload["options"]
    frozen = saved_plan_identity(payload)
    assert frozen.options.measure_power is False
    # A plan saved before the field existed has no key either.
    historical = json.loads((Path(__file__).parent / "fixtures/fpm_collection_plan_v10.json").read_text())
    assert saved_plan_identity(historical).options.measure_power is False

    cell, cell_dir = _saved_tep_prefill(tmp_path, frozen)
    with pytest.raises(ValueError, match="plan without measure_power"):
        aggregate_cell(frozen, cell, cell_dir, expected_attempt_id="attempt")


def test_all_unavailable_power_still_publishes_a_double_column(tmp_path):
    plan, cell, cell_dir = _tp2_decode(tmp_path, power=_unavailable("counter_wrap"))

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    parquet, metadata, _ = write_formal_database(plan, rows, systems_root=tmp_path / "systems")

    assert json.loads(metadata.read_text())["schema_version"] == 8
    table = pq.read_table(parquet)
    assert table.schema.field("power_w").type == pa.float64()
    assert table.column("power_w").to_pylist() == [None]


def test_commit_validation_accepts_schema_seven_and_eight(tmp_path):
    systems_root = tmp_path / "systems"
    old_plan, old_cell, old_dir = _write_cell(tmp_path / "old", phase="prefill", dp=2, schema_version=2)
    old_plan.cells = (old_cell,)
    parquet, metadata, _ = write_formal_database(
        old_plan, aggregate_cell(old_plan, old_cell, old_dir, expected_attempt_id="attempt"), systems_root=systems_root
    )
    sealed = validate_formal_database_commit(parquet, metadata, old_plan)
    assert sealed["schema_version"] == 7

    plan, cell, cell_dir = _tp2_decode(tmp_path / "new")
    parquet, metadata, _ = write_formal_database(
        plan, aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt"), systems_root=systems_root
    )
    assert validate_formal_database_commit(parquet, metadata, plan)["schema_version"] == 8
    # A checkpoint sealed against the schema-7 pair still validates after the
    # republish: its rows gained only a null power_w.
    resumed = validate_formal_database_commit(
        parquet,
        metadata,
        old_plan,
        expected_cell_rows=sealed["cell_rows"],
        expected_attempt_ids={old_cell.cell_id: "attempt"},
    )
    assert resumed["cell_rows"] == sealed["cell_rows"]


def test_commit_validation_requires_the_power_column_in_schema_eight(tmp_path):
    plan, cell, cell_dir = _tp2_decode(tmp_path)
    parquet, metadata, _ = write_formal_database(
        plan, aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt"), systems_root=tmp_path / "systems"
    )
    pq.write_table(pq.read_table(parquet).drop(["power_w"]), parquet)
    sidecar = json.loads(metadata.read_text())
    sidecar["parquet_sha256"] = hashlib.sha256(parquet.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match=r"missing schema-v8 columns: \['power_w'\]"):
        validate_formal_database_commit(parquet, metadata, plan)
    incoming = [dict(row, cell_id="other") for row in pq.read_table(parquet).to_pylist()]
    with pytest.raises(ValueError, match=r"missing columns: \['power_w'\]"):
        write_formal_database(plan, incoming, systems_root=tmp_path / "systems")


def _fpm_args(**overrides):
    parser = argparse.ArgumentParser()
    add_fpm_arguments(parser)
    args = parser.parse_args(overrides.pop("argv", []))
    values = {"fpm_max_gpus": 4, "fpm_gpu_counts": [4], **overrides}
    for name, value in values.items():
        setattr(args, name, value)
    return args


@pytest.mark.parametrize(
    ("argv", "smoke", "expected"),
    [
        ([], False, True),
        # Smoke shares the formal plan; the runner omits the window when rendering it.
        ([], True, True),
        (["--no-fpm-measure-power"], False, False),
        (["--fpm-measure-power"], False, True),
    ],
)
def test_measure_power_defaults_on_and_joins_the_plan_identity(argv, smoke, expected):
    options = FPMCollectionOptions.from_args(_fpm_args(argv=argv, smoke=smoke))

    assert options.measure_power is expected
    # Present only when enabled: plans saved before the field keep their hash.
    assert options.to_dict().get("measure_power") is (True if expected else None)
    assert replace(options, measure_power=not expected).to_dict() != options.to_dict()


def test_measure_power_requires_fpm_collection():
    args = _fpm_args(argv=["--no-fpm-measure-power"])
    args.ops = ["gemm"]

    with pytest.raises(ValueError, match="--fpm-measure-power"):
        reject_fpm_arguments_without_fpm(args)
