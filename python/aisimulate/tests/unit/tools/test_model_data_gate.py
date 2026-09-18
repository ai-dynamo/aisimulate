# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Git/Parquet fixtures for the portable shadow gate's failure contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

TOOL = Path(__file__).resolve().parents[3] / "tools/model_data_gate"
sys.path.insert(0, str(TOOL.parents[1]))
from tools.model_data_gate import run as gate
from tools.model_data_gate import stages

pytestmark = pytest.mark.unit
TABLE = gate.DATA + "h100_sxm/gemm/trtllm/1.0.0/gemm_perf.parquet"


def commit(repo):
    gate.git(repo, "add", "--all")
    gate.git(repo, "-c", "user.name=Gate Tests", "-c", "user.email=gate@example.invalid", "commit", "-m", "fixture")
    return gate.git(repo, "rev-parse", "HEAD").decode().strip()


def rows():
    # Synthetic measurements, not hardware accuracy evidence. The deliberate
    # curve below is flat; injecting a center peak must trigger the real detector.
    return [
        {
            "framework": "TRTLLM",
            "version": "1.0.0",
            "device": "NVIDIA H100",
            "op_name": "gemm",
            "kernel_source": "test",
            "gemm_dtype": "bfloat16",
            "m": m,
            "n": 128,
            "k": 128,
            "latency": 1.0,
        }
        for m in (1, 2, 4, 8, 16)
    ]


def write_table(repo, records=None, path=TABLE, **metadata_updates):
    records = rows() if records is None else records
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), target)
    metadata = {
        "schema_version": 1,
        "provenance": "collected",
        "runtime": {"framework": "trtllm", "version": "1.0.0"},
        "tables": {
            Path(path).stem: {
                "status": "complete",
                "rows": len(records),
                "collector_ref": "synthetic-fixture",
                "collector_hash": "sha256:" + "1" * 64,
                "case_plan_hash": "sha256:" + "2" * 64,
                "collected_at": "2026-09-17",
            }
        },
    }
    metadata.update(metadata_updates)
    target.with_name("collection_meta.yaml").write_text(yaml.safe_dump(metadata))
    return target


@pytest.fixture
def repo(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    gate.git(source, "init", "-q")
    catalog = source / gate.APP / "collector/op_backend_catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(yaml.safe_dump({"families": [{"family": "gemm", "op_files": ["gemm_perf"]}]}))
    spec_path = source / gate.SYSTEMS / "h100_sxm.yaml"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text(yaml.safe_dump({"gpu": {"bfloat16_tc_flops": 989e12, "mem_bw": 3.35e12}}))
    return source, commit(source)


def run(repo, base, tmp_path):
    head = commit(repo)
    out = tmp_path / "report"
    code = gate.run_gate(repo, base, head, out)
    report = json.loads((out / "report.json").read_text())
    assert report["base_sha"] == base and report["head_sha"] == head
    assert report["exit_code"] == code
    assert (out / "summary.md").is_file()
    assert set(report["stages"]) == set(gate.STAGES)
    return code, report


def test_valid_addition_does_not_pretend_unimplemented_stages_pass(repo, tmp_path):
    source, base = repo
    write_table(source)
    code, report = run(source, base, tmp_path)
    assert code == 1 and report["conclusion"] == "Failed"
    artifact = report["stages"]["Artifact integrity"]
    assert artifact["findings"] == []
    assert artifact["artifacts"][0]["row_diff"]["added_rows"] == 5
    assert artifact["artifacts"][0]["head"]["sha256"]
    assert report["stages"]["Production reachability"]["status"] == "INCOMPLETE"
    assert report["stages"]["Behavior and parity"]["status"] == "INCOMPLETE"


@pytest.mark.parametrize(
    "damage,rule",
    [
        ("unreadable", "readable_artifact"),
        ("lfs", "readable_artifact"),
        ("missing_column", "required_columns"),
        ("duplicate", "duplicate_physical_key"),
        ("nan", "finite_nonnull"),
        ("null", "finite_nonnull"),
        ("infinity", "finite_nonnull"),
        ("zero", "timing_boundary"),
        ("negative", "timing_boundary"),
        ("string_dimension", "shape_type"),
        ("identity", "row_identity"),
        ("sidecar", "collection_provenance"),
        ("metadata_identity", "sidecar_identity"),
        ("metadata_count", "sidecar_rows"),
        ("estimated", "collection_provenance"),
    ],
)
def test_real_artifact_failures_produce_reports(repo, tmp_path, damage, rule):
    source, base = repo
    records = rows()
    if damage == "missing_column":
        for row in records:
            del row["k"]
    if damage == "duplicate":
        records.append(dict(records[0], latency=2.0))
    if damage in {"nan", "null", "infinity", "zero", "negative"}:
        records[0]["latency"] = {
            "nan": float("nan"),
            "null": None,
            "infinity": float("inf"),
            "zero": 0.0,
            "negative": -1.0,
        }[damage]
    if damage == "string_dimension":
        for row in records:
            row["m"] = str(row["m"])
    if damage == "identity":
        records[0]["version"] = "other-version"
    target = write_table(source, records)
    if damage in {"unreadable", "lfs"}:
        target.write_bytes(b"broken" if damage == "unreadable" else stages.parquet_diff.LFS_POINTER_PREFIX + b"oid x")
    metadata_path = target.with_name("collection_meta.yaml")
    if damage == "sidecar":
        metadata_path.unlink()
    if damage.startswith("metadata_") or damage == "estimated":
        metadata = yaml.safe_load(metadata_path.read_text())
        if damage == "metadata_identity":
            metadata["runtime"]["version"] = "wrong"
        if damage == "metadata_count":
            metadata["tables"]["gemm_perf"]["rows"] = 900
        if damage == "estimated":
            metadata["provenance"] = "estimated"
        metadata_path.write_text(yaml.safe_dump(metadata))
    code, report = run(source, base, tmp_path)
    assert code == 1
    artifact = report["stages"]["Artifact integrity"]
    assert artifact["status"] == "FAIL"
    assert rule in {item["rule"] for item in artifact["findings"]}


def test_old_version_spike_is_checked_even_with_a_newer_version(repo, tmp_path):
    source, _base = repo
    write_table(source)
    newer = TABLE.replace("/1.0.0/", "/9.0.0/")
    write_table(
        source,
        [dict(row, version="9.0.0") for row in rows()],
        path=newer,
        runtime={"framework": "trtllm", "version": "9.0.0"},
    )
    base = commit(source)
    records = rows()
    records[2]["latency"] = 100.0
    write_table(source, records)
    code, report = run(source, base, tmp_path)
    numeric = report["stages"]["Numerical sanity"]
    assert code == 1 and numeric["status"] == "FAIL"
    assert any(item["kind"] == "spike_violation" and item["version"] == "1.0.0" for item in numeric["findings"])


def test_impossible_gemm_and_changed_gpu_spec_do_not_share_base_cache(repo, tmp_path):
    source, _base = repo
    records = [dict(row, m=row["m"] * 1000, n=1024, k=1024, latency=0.0001) for row in rows()]
    write_table(source, records)
    spec_path = source / gate.SYSTEMS / "h100_sxm.yaml"
    spec_path.write_text(yaml.safe_dump({"gpu": {"bfloat16_tc_flops": 1e30, "mem_bw": 1e30}}))
    base = commit(source)
    spec_path.write_text(yaml.safe_dump({"gpu": {"bfloat16_tc_flops": 989e12, "mem_bw": 3.35e12}}))
    _, report = run(source, base, tmp_path)
    numeric = report["stages"]["Numerical sanity"]
    assert any(item["kind"] == "below_sol" for item in numeric["findings"])
    assert not numeric["comparisons"][0]["base_findings"]


def test_delta_exemptions_are_finite_and_do_not_allow_negative_calibration():
    table = pa.Table.from_pylist([{"latency": -1.0, "m": 1}])
    for name in ("computescale", "dsv4_csa_topk_calib", "glm5_topk_module"):
        path = gate.DATA + f"h100_sxm/quantize/trtllm/1.0.0/{name}_perf.parquet"
        failures, _ = stages.validate_table(path, table, {f"{name}_perf": "quantize"})
        assert any(item["rule"] == "timing_boundary" for item in failures) == (name != "computescale")
    table = pa.Table.from_pylist([{"latency": float("nan"), "m": 1}])
    failures, _ = stages.validate_table(path, table, {"glm5_topk_module_perf": "quantize"})
    assert any(item["rule"] == "finite_nonnull" for item in failures)


def test_unrelated_is_explicit_na(repo, tmp_path):
    source, base = repo
    (source / "README.md").write_text("documentation")
    code, report = run(source, base, tmp_path)
    assert code == 0 and report["conclusion"] == "Not applicable"
    assert all(value["status"] == "SKIP" for value in report["stages"].values())


@pytest.mark.parametrize(
    "path",
    [
        "unknown.py",
        "docs/cli/prediction-details.schema.json",
        "examples/model.yaml",
        "crates/core/src/perfmodel/perf_database/gemm.rs",
        ".github/workflows/model-data-quality.yml",
        gate.APP + "/tools/model_data_gate/run.py",
        gate.APP + "/src/aiconfigurator_core/sdk/perf_database.py",
        gate.APP + "/collector/op_catalog.py",
    ],
)
def test_indirect_and_gate_changes_are_applicable(path):
    assert gate.applicable([path])


def test_rename_delete_and_newline_paths_cannot_disappear(repo, tmp_path):
    source, _base = repo
    target = write_table(source)
    base = commit(source)
    target.rename(target.with_name("renamed\ntable.parquet"))
    head = commit(source)
    assert TABLE in gate.changed_paths(source, base, head)
    assert TABLE.replace("gemm_perf.parquet", "renamed\ntable.parquet") in gate.changed_paths(source, base, head)
    assert gate.run_gate(source, base, head, tmp_path / "report") == 1
    report = json.loads((tmp_path / "report/report.json").read_text())
    artifact = report["stages"]["Artifact integrity"]
    assert any(item["row_diff"]["removed_rows"] == 5 for item in artifact["artifacts"])


@pytest.mark.parametrize("kind", ["crash", "no_result", "bad_status", "nonfinite"])
def test_required_stage_crash_or_missing_result_fails_closed(repo, tmp_path, monkeypatch, kind):
    source, base = repo
    write_table(source)

    def broken(*_args):
        if kind == "crash":
            raise RuntimeError("planted stage failure")
        return {
            "no_result": None,
            "bad_status": gate.result("SKIP", "not allowed"),
            "nonfinite": gate.result("PASS", "bad", observed=float("nan")),
        }.get(kind)

    monkeypatch.setattr(stages, "artifact_integrity", broken)
    code, report = run(source, base, tmp_path)
    assert code == 1 and report["stages"]["Artifact integrity"]["status"] == "FAIL"
    assert report["stages"]["Numerical sanity"]["status"] == "INCOMPLETE"


def test_invalid_sha_still_writes_failure_evidence(repo, tmp_path):
    source, base = repo
    out = tmp_path / "report"
    assert gate.run_gate(source, "main", base, out) == 1
    assert json.loads((out / "report.json").read_text())["conclusion"] == "Failed"
    assert (out / "summary.md").exists()


def test_snapshot_symlink_is_rejected(repo, tmp_path):
    source, base = repo
    (source / gate.SYSTEMS / "outside").symlink_to("/etc/passwd")
    _, report = run(source, base, tmp_path)
    assert "symlinks" in report["stages"]["Artifact integrity"]["error"]


def test_classification_only_is_never_passing_validation(repo, tmp_path):
    source, base = repo
    write_table(source)
    head = commit(source)
    out = tmp_path / "report"
    assert gate.run_gate(source, base, head, out, classify_only=True) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["conclusion"] == "Failed" and report["exit_code"] == 1


def test_workflow_uses_event_shas_hosted_runner_and_only_shadow_step_tolerance():
    root = TOOL.parents[3]
    text = (root / ".github/workflows/model-data-quality.yml").read_text()
    workflow = yaml.safe_load(text)
    job = workflow["jobs"]["gate"]
    assert job["runs-on"] == "ubuntu-latest"
    assert workflow["permissions"] == {"contents": "read"}
    assert job["env"] == {
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }
    assert "pull_request_target" not in text and "secrets." not in text
    steps = job["steps"]
    assert steps[0]["with"]["persist-credentials"] is False
    assert [step["id"] for step in steps if step.get("continue-on-error")] == ["shadow"]
    assert steps[-1]["with"]["if-no-files-found"] == "error"


def test_git_export_attributes_cannot_hide_or_rewrite_snapshot(repo, tmp_path):
    source, _base = repo
    write_table(source)
    (source / ".gitattributes").write_text("**/*.parquet export-ignore\n")
    head = commit(source)
    destination = tmp_path / "snapshot"
    gate.export_snapshot(source, head, destination)
    assert (destination / TABLE).read_bytes() == (source / TABLE).read_bytes()


def test_extra_columns_do_not_hide_duplicate_native_gemm_coordinates(repo, tmp_path):
    source, base = repo
    records = [dict(rows()[0], collection_note="first"), dict(rows()[0], latency=2.0, collection_note="second")]
    write_table(source, records)
    _, report = run(source, base, tmp_path)
    assert any(item["rule"] == "duplicate_physical_key" for item in report["stages"]["Artifact integrity"]["findings"])
