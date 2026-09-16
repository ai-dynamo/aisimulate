# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json

import pytest
import yaml
from collector.campaigns.slurm_collection_hook import case_hash, completion, validate_snapshot

pytestmark = pytest.mark.unit


def test_gemm_only_cannot_complete_all_model_campaign():
    assert not completion({"gemm": 10, "moe": 5}, {"gemm": 10}, [])
    assert not completion({"gemm": 10}, {"gemm": 10}, ["communication plan missing"])
    assert completion({"gemm": 10, "moe": 5}, {"gemm": 10, "moe": 5}, [])
    assert not completion({"gemm": 10}, {"gemm": 20}, [])
    assert not completion({}, {}, [])
    assert not completion({"gemm": 0}, {}, [])


def fixture(tmp_path):
    case = "['bfloat16', 1, 768, 7168]"
    spec = {
        "id": "gemm-bf16-00",
        "op": "gemm",
        "mode": "full",
        "planned_tasks": 1,
        "case_set_sha256": case_hash([case]),
    }
    data = tmp_path / "data"
    (data / "checkpoint/vllm").mkdir(parents=True)
    status = {
        "spec": spec,
        "status": "complete",
        "done": 1,
        "failed": 0,
        "collector_exit_code": 0,
        "source_commit": "a" * 40,
    }
    (tmp_path / "status.json").write_text(json.dumps(status))
    checkpoint = {
        "framework_version": "0.25.0",
        "sm_version": 100,
        "done": ["vllm.gemm:run_gemm:" + case],
        "failed": [],
    }
    (data / "checkpoint/vllm/gemm.json").write_text(json.dumps(checkpoint))
    meta = {
        "runtime": {"framework": "vllm", "version": "0.25.0"},
        "tables": {"gemm_perf": {"status": "complete", "rows": 1, "collector_ref": "a" * 40}},
    }
    (data / "collection_meta.yaml").write_text(yaml.safe_dump(meta))
    (data / "gemm_perf.parquet").write_bytes(b"fixture file; reader is injected")
    rows = [{"version": "0.25.0", "latency": 0.01, "gemm_dtype": "bfloat16", "m": 1, "n": 768, "k": 7168}]
    return spec, status, checkpoint, rows


def test_validates_exact_checkpoint_and_output_keys(tmp_path):
    spec, _, _, rows = fixture(tmp_path)
    assert validate_snapshot(tmp_path, spec, table_reader=lambda _: rows)["done"] == 1


@pytest.mark.parametrize("failure", ["missing_id", "failed_id", "wrong_version", "wrong_key", "nan", "no_parquet"])
def test_rejects_incomplete_or_mislabeled_data(tmp_path, failure):
    spec, _, checkpoint, rows = fixture(tmp_path)
    if failure == "missing_id":
        checkpoint["done"] = []
    elif failure == "failed_id":
        checkpoint["failed"] = ["failed-task"]
    elif failure == "wrong_version":
        rows[0]["version"] = "0.24.0"
    elif failure == "wrong_key":
        rows[0]["m"] = 2
    elif failure == "nan":
        rows[0]["latency"] = float("nan")
    elif failure == "no_parquet":
        (tmp_path / "data/gemm_perf.parquet").unlink()
    (tmp_path / "data/checkpoint/vllm/gemm.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError):
        validate_snapshot(tmp_path, spec, table_reader=lambda _: rows)


def test_cpu_hook_memory_limit_cannot_leak_into_gpu_step(tmp_path, monkeypatch):
    from collector.campaigns import slurm_collection_hook as hook

    monkeypatch.setenv("SLURM_MEM_PER_NODE", "4096")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("SLURM_CONF", "/etc/slurm/cluster.conf")
    captured = {}

    def fake_submit(command, **kwargs):
        captured.update(command=command, **kwargs)
        return "12345\n"

    monkeypatch.setattr(hook.subprocess, "check_output", fake_submit)
    campaign = {
        "root": str(tmp_path),
        "runner_revision": "test",
        "slurm": {"account": "test", "partition": "batch", "image": "/image.sqsh", "memory": "128G"},
    }
    campaign["slurm"]["full_qos"] = "normal"
    spec = {"id": "gemm-00", "op": "gemm", "mode": "full", "planned_tasks": 1}
    hook.submit({}, campaign, spec)
    assert "--mem=128G" in captured["command"]
    assert "--qos=normal" in captured["command"]
    wrapped = next(v for v in captured["command"] if v.startswith("--wrap="))
    assert "--mem=128G" in wrapped and "--cpus-per-task=32" in wrapped
    assert "SLURM_MEM_PER_NODE" not in captured["env"]
    assert "SLURM_CPUS_PER_TASK" not in captured["env"]
    assert captured["env"]["SLURM_CONF"] == "/etc/slurm/cluster.conf"


@pytest.mark.parametrize("mismatch", ["missing", "undeclared"])
def test_rejects_any_declared_observed_table_mismatch(tmp_path, mismatch):
    spec, _, _, rows = fixture(tmp_path)
    path = tmp_path / "data/collection_meta.yaml"
    meta = yaml.safe_load(path.read_text())
    if mismatch == "missing":
        meta["tables"]["second_perf"] = {"rows": 1, "status": "complete"}
        path.write_text(yaml.safe_dump(meta))
    else:
        (tmp_path / "data/second_perf.parquet").write_bytes(b"unclaimed")
    with pytest.raises(ValueError, match="table mismatch"):
        validate_snapshot(tmp_path, spec, table_reader=lambda _: rows)


@pytest.mark.parametrize("latency", [None, "0.01", True, [], {}, float("inf"), float("nan")])
def test_bad_latency_is_a_shard_validation_error(tmp_path, latency):
    spec, _, _, rows = fixture(tmp_path)
    rows[0]["latency"] = latency
    with pytest.raises(ValueError):
        validate_snapshot(tmp_path, spec, table_reader=lambda _: rows)


def test_bad_shard_does_not_prevent_later_ready_submission(tmp_path, monkeypatch):
    from collector.campaigns import slurm_collection_hook as hook

    bad_job = tmp_path / "results/gemm-bf16-00/job-1"
    spec, _, _, rows = fixture(bad_job)
    rows[0]["latency"] = None
    later = {"id": "later", "op": "gemm", "mode": "full", "planned_tasks": 1}
    (tmp_path / "plan.json").write_text(json.dumps({"smokes": [], "shards": [spec, later]}))
    original = hook.validate_snapshot
    monkeypatch.setattr(
        hook, "validate_snapshot", lambda path, selected: original(path, selected, table_reader=lambda _: rows)
    )
    monkeypatch.setattr(hook.subprocess, "check_output", lambda *args, **kwargs: "")
    submitted = []

    def submit(config, campaign, selected):
        submitted.append(selected["id"])
        return {"job": "2", "shard": selected["id"]}

    monkeypatch.setattr(hook, "submit", submit)
    report = hook.tick(
        {
            "user": "test",
            "max_active_jobs": 2,
            "max_infra_attempts_per_revision": 1,
            "expected_tasks": {"gemm": 2},
            "campaigns": [{"root": str(tmp_path), "plan": "plan.json", "runner_revision": "test"}],
        },
        apply=True,
    )
    assert report["shards"][0]["state"] == "validation_failed"
    assert submitted == ["later"]


@pytest.mark.parametrize("failure", ["document", "runtime", "table_name", "table_entry", "yaml"])
def test_corrupt_metadata_is_a_shard_validation_error(tmp_path, failure):
    spec, _, _, rows = fixture(tmp_path)
    path = tmp_path / "data/collection_meta.yaml"
    meta = yaml.safe_load(path.read_text())
    if failure == "document":
        meta = []
    elif failure == "runtime":
        meta["runtime"] = None
    elif failure == "table_name":
        meta["tables"][123] = {}
    elif failure == "table_entry":
        meta["tables"]["gemm_perf"] = []
    if failure == "yaml":
        path.write_text("runtime: [unterminated")
    else:
        path.write_text(yaml.safe_dump(meta))
    with pytest.raises(ValueError):
        validate_snapshot(tmp_path, spec, table_reader=lambda _: rows)
