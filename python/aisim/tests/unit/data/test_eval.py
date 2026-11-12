# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aisim_core.data import Identity
from aisim_core.data.eval import (
    EvalRecord,
    GpuTime,
    PredictionIdentity,
    attest,
    compute_results,
    current_records,
    load_record,
    prediction_fingerprint,
    record_path,
    verify_record,
    write_record,
    write_truth_table,
)

pytestmark = pytest.mark.unit

SPEC = "a" * 64
PIN = "b" * 64


def _predict(inputs):
    return inputs["tokens"] * 1.1 + (5.0 if inputs["tokens"] == 40 else 0.0)


def _identity():
    return Identity(
        hardware="b200_sxm",
        framework="vllm",
        framework_version="0.11.0",
        operation_family=None,
        model="Qwen/Qwen3-32B",
        effective_settings={},
        unresolved_identity=None,
        parallelism={"aggregated": {"tp": 4, "ep": 1}},
        parallelism_complete=True,
        deployment="aggregated",
    )


@pytest.fixture
def built(tmp_path):
    rows = [(f"s{i}", {"tokens": 10 * i}, 10.0 * i) for i in range(1, 6)] + [("bad", {"tokens": 1}, None)]
    art = write_truth_table(tmp_path / "truth.parquet", rows)
    cases = [(c, t, _predict(i)) for c, i, t in rows]
    results = compute_results(cases)
    pred = PredictionIdentity.build("fpm_interpolation", {"profile": PIN}, SPEC)
    record = EvalRecord.model_validate(
        {
            "schema_version": "aisim-eval/v1",
            "prediction": pred.model_dump(),
            "truth": {
                "layer": "fpm",
                "snapshot_id": None,
                "configuration_ids": ["cfg-" + "c" * 64],
                "identity": _identity().model_dump(),
                "artifacts": [art.model_dump(by_alias=True)],
                "truth_table_artifact_id": "truth-table",
            },
            "results": results.model_dump(),
            "assumptions": [{"name": "kv_dtype", "value": "fp8", "status": "estimated", "kind": "calibration"}],
            "gpu_time": GpuTime(authority="unknown").model_dump(),
            "simulation_cost": {"wall_seconds": 1.5, "gpu_seconds": 0.0},
            "producer": {"kind": "agent", "name": "t", "evaluator_revision": "r1"},
            "tier": "submitted",
            "created_at": "2026-11-07T00:00:00Z",
        }
    )
    return tmp_path, record


def test_fingerprint_ignores_unrelated_and_tracks_spec():
    base = prediction_fingerprint("p", {"a": PIN}, SPEC)
    assert base == prediction_fingerprint("p", {"a": PIN}, SPEC)
    assert base != prediction_fingerprint("p", {"a": PIN}, "d" * 64)
    with pytest.raises(ValueError):
        PredictionIdentity(code_path="p", data_pins={"a": PIN}, spec_sha256=SPEC, fingerprint="e" * 64)


def test_unknown_gpu_time_cannot_be_zero():
    with pytest.raises(ValueError):
        GpuTime(authority="unknown", gpu_seconds=0.0)


def test_exclusion_and_metrics(built):
    _, record = built
    assert record.results.n_total == 6 and record.results.n_scored == 5
    assert [e.case_id for e in record.results.exclusions] == ["bad"]
    assert record.results.coverage == pytest.approx(5 / 6)


def test_write_verify_supersede_exit_gate(built):
    root, record = built
    path = write_record(root, record)
    assert path == record_path(root, record) and path.stem == record.record_id
    assert write_record(root, record) == path  # idempotent
    original = path.read_bytes()

    report = verify_record(load_record(path), root, _predict, record.prediction.fingerprint)
    assert report.ok, report.problems

    verified = attest(record, report)
    vpath = write_record(root, verified)
    assert vpath != path and path.read_bytes() == original  # original untouched
    assert verified.supersedes == record.record_id and verified.tier == "verified"
    assert [r.record_id for r in current_records(path.parent)] == [verified.record_id]


def test_verifier_rejects_wrong_claims_and_fingerprint(built):
    root, record = built
    report = verify_record(record, root, lambda i: _predict(i) * 1.5, record.prediction.fingerprint)
    assert not report.ok and any("mape_pct" in p for p in report.problems)
    assert not verify_record(record, root, _predict, "f" * 64).ok
    with pytest.raises(ValueError):
        attest(record, report)


def test_tampered_truth_table_fails(built):
    root, record = built
    (root / "truth.parquet").write_bytes(b"x")
    assert "hash" in verify_record(record, root, _predict, record.prediction.fingerprint).problems[0]


def test_immutability_and_supersede_rules(built):
    root, record = built
    path = write_record(root, record)
    path.write_text(path.read_text().replace("2026-11-07", "2027-01-01"))
    with pytest.raises(ValueError, match="content hash"):
        load_record(path)
    other = EvalRecord.model_validate({**record.model_dump(by_alias=True), "supersedes": "9" * 64})
    with pytest.raises(ValueError, match="not found"):
        write_record(root, other)
    with pytest.raises(ValueError):
        EvalRecord.model_validate({**record.model_dump(by_alias=True), "tier": "verified"})
    with pytest.raises(ValueError):
        EvalRecord.model_validate({**record.model_dump(by_alias=True), "status": "failed"})


def test_eval_uses_gpu_time_module_type():
    from aisim_core.data import gpu_time

    assert GpuTime is gpu_time.GpuTime


def test_record_gpu_time_archival_warns_qualified_errors(built):
    from aisim_core.data.eval import check_record_gpu_time

    _, record = built
    unknown = check_record_gpu_time(record)
    assert [(f.severity, f.code) for f in unknown] == [("warn", "gpu_time.missing")]
    absent = record.model_copy(update={"gpu_time": None})
    assert check_record_gpu_time(absent)[0].severity == "warn"
    doc = record.model_dump(by_alias=True)
    doc["gpu_time"] = None
    doc.update(
        tier="qualified",
        verification={
            "verifier_version": "1",
            "recomputed_fingerprint": record.prediction.fingerprint,
            "verified_record_id": record.record_id,
        },
    )
    with pytest.raises(ValueError, match="qualified record needs gpu_time"):
        EvalRecord.model_validate(doc)
    doc["gpu_time"] = GpuTime(authority="unknown").model_dump(by_alias=True)
    with pytest.raises(ValueError, match="qualified record needs gpu_time"):
        EvalRecord.model_validate(doc)
    doc["gpu_time"] = GpuTime(authority="measured", gpu_count=4, wall_seconds=0.0).model_dump(by_alias=True)
    assert EvalRecord.model_validate(doc).tier == "qualified"  # zero is recorded, not missing
