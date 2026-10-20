# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from cryptography.exceptions import InvalidTag

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.e2e_accuracy import hf_measurements as hf
from scripts.e2e_accuracy import private_artifacts
from scripts.e2e_accuracy.source import hf_bindings
from scripts.e2e_accuracy.source.cohort import select_points
from scripts.e2e_accuracy.source.recipes import runtime_evidence
from scripts.e2e_accuracy.source.schema import SiliconRow


@pytest.fixture
def manifest():
    return dict(
        schema_version=2,
        repo_id=hf.REPO_ID,
        revision="a" * 40,
        release_tag="hf/" + "a" * 40,
        selection_policy=hf.POLICY,
        max_age_days=180,
    )


@pytest.fixture
def record():
    return dict(
        model="test-model",
        hardware="h100",
        framework="vllm",
        precision="fp8",
        spec_method="none",
        disagg=False,
        is_multinode=False,
        prefill_tp=1,
        prefill_ep=1,
        prefill_dp_attention=False,
        prefill_num_workers=0,
        decode_tp=8,
        decode_ep=1,
        decode_dp_attention=False,
        decode_num_workers=1,
        num_prefill_gpu=0,
        num_decode_gpu=8,
        id=123,
        date="2026-10-17",
        isl=1024,
        osl=128,
        conc=32,
        metrics={"mean_ttft": 0.1, "mean_tpot": 0.01},
        image="test-image",
        benchmark_type="single_turn",
        run_url="https://github.com/SemiAnalysisAI/InferenceX/actions/runs/987/attempts/2",
    )


def test_resolve_pins_private_commit(monkeypatch, manifest):
    info = SimpleNamespace(private=True, sha="b" * 40)
    monkeypatch.setattr(hf, "HfApi", lambda: SimpleNamespace(repo_info=lambda *a, **k: info))
    resolved = hf.resolve_latest_manifest(manifest)
    assert resolved["revision"] == "b" * 40 and manifest["revision"] == "a" * 40
    info.private = False
    with pytest.raises(ValueError, match="private"):
        hf.resolve_latest_manifest(manifest)


@pytest.mark.parametrize(
    "change",
    [
        dict(revision="main"),
        dict(repo_id="other/repo"),
        dict(selection_policy="gym-resolved-config-v2"),
        dict(schema_version=True),
        dict(max_age_days=True),
        dict(release_tag="hf/" + "b" * 40),
    ],
)
def test_invalid_hf_manifest(manifest, change):
    with pytest.raises(ValueError):
        hf.validate_manifest(manifest | change)


def test_exact_producer_attempt_only(monkeypatch, record):
    calls = []
    run = dict(id=987, run_attempt=2, head_sha="b" * 40)

    def get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: run)

    monkeypatch.setattr(hf.requests, "get", get)
    assert hf.producer_metadata(record["run_url"])["head_sha"] == "b" * 40
    assert calls == ["https://api.github.com/repos/SemiAnalysisAI/InferenceX/actions/runs/987/attempts/2"]
    assert hf.producer_metadata(record["run_url"].removesuffix("/attempts/2")) == {}
    run["run_attempt"] = 3
    with pytest.raises(ValueError, match="identity"):
        hf.producer_metadata(record["run_url"])


@pytest.mark.parametrize("status", [404, 410])
def test_deleted_producer_has_no_invented_revision(monkeypatch, record, status):
    monkeypatch.setattr(hf.requests, "get", lambda *a, **k: SimpleNamespace(status_code=status))
    assert "head_sha" not in hf.producer_metadata(record["run_url"])


@pytest.mark.parametrize(
    "failure", [502, 503, "connection", "timeout", "rate_limit", "rate_limit_date", "rate_limit_429"]
)
def test_producer_recovers_from_transient_failure(monkeypatch, record, failure):
    run = dict(id=987, run_attempt=2, head_sha="b" * 40)
    success = hf.requests.Response()
    success.status_code = 200
    success._content = json.dumps(run).encode()
    if failure in ("connection", "timeout"):
        first = hf.requests.ConnectionError() if failure == "connection" else hf.requests.Timeout()
    else:
        first = hf.requests.Response()
        first.status_code = 403 if str(failure).startswith("rate_limit") else failure
        if failure == "rate_limit_429":
            first.status_code = 429
            first.headers["Retry-After"] = "5"
        first._content = b""
        first._content_consumed = True
        if failure == "rate_limit":
            first.headers["Retry-After"] = "5"
        if failure == "rate_limit_date":
            first.headers["Retry-After"] = "Thu, 01 Jan 1970 00:00:05 GMT"
            monkeypatch.setattr(hf.time, "time", lambda: 0)
    responses = iter([first, success])
    calls, sleeps = [], []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(hf.requests, "get", get)
    monkeypatch.setattr(hf.time, "sleep", sleeps.append)
    assert hf.producer_metadata(record["run_url"])["head_sha"] == run["head_sha"]
    assert len(calls) == 2 and calls[0] == calls[1]
    assert sleeps == [5 if str(failure).startswith("rate_limit") else 1]


@pytest.mark.parametrize(
    "failure", [502, "connection", 403, 401, 429, "rate_limit", "rate_limit_429", "long_rate_limit", "bad_rate_limit"]
)
def test_producer_retries_are_bounded(monkeypatch, record, failure):
    calls, sleeps = [], []

    def get(*args, **kwargs):
        calls.append(1)
        if failure == "connection":
            raise hf.requests.ConnectionError("offline")
        response = hf.requests.Response()
        response.status_code = 403 if isinstance(failure, str) else failure
        response._content = b""
        response._content_consumed = True
        if failure in ("rate_limit", "rate_limit_429"):
            response.headers["Retry-After"] = "0"
        if failure == "rate_limit_429":
            response.status_code = 429
        if failure == "long_rate_limit":
            response.headers["Retry-After"] = "120"
        if failure == "bad_rate_limit":
            response.headers["Retry-After"] = "invalid"
        return response

    monkeypatch.setattr(hf.requests, "get", get)
    monkeypatch.setattr(hf.time, "sleep", sleeps.append)
    with pytest.raises(hf.requests.RequestException):
        hf.producer_metadata(record["run_url"])
    retried = failure in (502, "connection", "rate_limit", "rate_limit_429")
    assert len(calls) == (3 if retried else 1)
    assert sleeps == ([1, 2] if retried else [])


def test_local_configuration_identity_and_cohort(record):
    run = dict(head_sha="b" * 40, github_run_id="987", run_attempt=2)
    tables = hf.to_tables([record, record | dict(id=124, conc=64)], {record["run_url"]: run})
    assert len(tables["configs"]) == 1
    config_id = tables["configs"][0]["id"]
    assert config_id.startswith("hf:") and len(config_id) == 67
    assert hf.to_tables([dict(reversed(list(record.items())))], {})["configs"][0]["id"] == config_id
    assert hf.to_tables([record | dict(decode_tp=4)], {})["configs"][0]["id"] != config_id
    points, _ = select_points(tables)
    assert len(points) == 2
    assert all(p["source_row"]["head_sha"] == "b" * 40 for p in points)
    with pytest.raises(ValueError, match="duplicate HF benchmark"):
        hf.to_tables([record, record | dict(conc=64)], {})


@pytest.fixture
def cache(tmp_path, monkeypatch, record, request):
    artifacts = {}

    def register(path, payload, schema, rows=None):
        local = tmp_path / path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(payload)
        artifacts[path] = local
        return dict(
            path=path,
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
            schema=schema,
            **({"rows": rows} if rows is not None else {}),
        )

    row = dict(
        observation_id="inferencex:123",
        record_sha256=hashlib.sha256(hf.canonical(record)).hexdigest(),
        raw_record_json=hf.canonical(record).decode(),
    )
    count = getattr(request, "param", 1)
    table = pa.Table.from_pylist([row] * count).replace_schema_metadata({b"schema_id": b"inferencex-observations-v1"})
    output = pa.BufferOutputStream()
    pq.write_table(table, output)
    ref = register(
        "data/e2e/inferencex/observations.parquet", output.getvalue().to_pybytes(), "inferencex-observations-v1", count
    )
    partitions = [
        dict(source_model=record["model"], hardware=record["hardware"], framework=record["framework"], artifacts=[ref])
    ]
    snapshot = dict(
        schema_version="inferencex-input-snapshot/v1",
        snapshot_id="test",
        completeness="complete",
        partitions=partitions,
        counts={"observations_distinct": count},
        source_scope={"kind": "api-current-model-view"},
    )
    ref = register("data/e2e/inferencex/snapshot.json", hf.canonical(snapshot), "inferencex-input-snapshot/v1")
    index = dict(
        schema_version="e2e-source-index/v1",
        source_kind="inferencex",
        snapshot_id="test",
        snapshot=ref,
        partitions=partitions,
    )
    register(hf.INDEX_PATH, hf.canonical(index), "e2e-source-index/v1")
    calls = []

    def download(repo, path, **kwargs):
        calls.append((repo, path, kwargs))
        return artifacts[path]

    monkeypatch.setattr(hf, "hf_hub_download", download)
    monkeypatch.setattr(
        hf, "producer_metadata", lambda url: dict(head_sha="b" * 40, github_run_id="987", run_attempt=2)
    )
    return artifacts, calls


def test_fetch_only_verified_observations_at_exact_commit(cache, manifest, tmp_path):
    artifacts, calls = cache
    output = hf.fetch(manifest, tmp_path / "out")
    tables = json.loads(output.read_text())
    assert len(tables["benchmark_results"]) == 1
    assert {c[1] for c in calls} == set(artifacts)
    assert all(c[0] == hf.REPO_ID and c[2] == dict(repo_type="dataset", revision="a" * 40) for c in calls)
    artifacts["data/e2e/inferencex/observations.parquet"].write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        hf.fetch(manifest, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("path", ["/etc/passwd", "data/e2e/../x", "data/e2e/inferencex/../../x", "other/file"])
def test_unsafe_artifact_paths(manifest, path):
    with pytest.raises(ValueError, match="unsafe"):
        hf.download(manifest, path)


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "synthetic-test-secret")
    monkeypatch.setenv("GITHUB_REPOSITORY", "test/repo")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    source = tmp_path / "source"
    source.mkdir()
    (source / "rows.json").write_bytes(b"private-test-row" * 100000)
    encrypted = tmp_path / "encrypted"
    private_artifacts.transfer(source, encrypted, decrypt=False)
    return source, encrypted, tmp_path / "opened"


def test_private_artifact_streaming_roundtrip(sealed):
    source, encrypted, opened = sealed
    assert (encrypted / "rows.json.enc").read_bytes() != (source / "rows.json").read_bytes()
    private_artifacts.transfer(encrypted, opened, decrypt=True)
    assert (opened / "rows.json").read_bytes() == (source / "rows.json").read_bytes()


@pytest.mark.parametrize("damage", ["token", "run", "path", "bytes"])
def test_private_artifact_tamper_never_leaves_plaintext(sealed, monkeypatch, damage):
    _, encrypted, opened = sealed
    path = encrypted / "rows.json.enc"
    if damage == "token":
        monkeypatch.setenv("HF_TOKEN", "different-secret")
    elif damage == "run":
        monkeypatch.setenv("GITHUB_RUN_ID", "456")
    elif damage == "path":
        path.rename(encrypted / "other.json.enc")
    else:
        data = bytearray(path.read_bytes())
        data[-20] ^= 1
        path.write_bytes(data)
    with pytest.raises(InvalidTag):
        private_artifacts.transfer(encrypted, opened, decrypt=True)
    assert not list(opened.rglob("*"))


def test_private_artifact_rejects_symlinks_and_plaintext(sealed):
    source, encrypted, opened = sealed
    (encrypted / "link").symlink_to(source / "rows.json")
    with pytest.raises(ValueError, match="symlink"):
        private_artifacts.transfer(encrypted, opened, decrypt=True)
    (encrypted / "link").unlink()
    with pytest.raises(ValueError, match="unencrypted"):
        private_artifacts.transfer(source, opened, decrypt=True)


@pytest.mark.parametrize("cache", [2], indirect=True)
def test_duplicate_observations_with_valid_checksums_fail_before_output(cache, manifest, tmp_path):
    with pytest.raises(ValueError, match="duplicate HF benchmark identity"):
        hf.fetch(manifest, tmp_path / "duplicate-output")
    assert not (tmp_path / "duplicate-output").exists()


@pytest.mark.parametrize("archive_version", [1, 2])
def test_reviewed_hf_binding_preserves_archive_and_full_database_log_reuse(record, monkeypatch, archive_version):
    record = record | dict(spec_method="mtp")
    tables = hf.to_tables([record], {})
    point = select_points(tables, include_mtp=True)[0][0]
    row = SiliconRow(**point["source_row"])
    original = replace(
        row, config_id=7, workflow_run_id=9, server_log_id="11", infx_config={**row.infx_config, "id": 7}
    )
    binding = hf_bindings.bind_row(row, original, 2)
    monkeypatch.setattr(
        hf_bindings,
        "load_manifest",
        lambda _: {
            "schema_version": "reviewed-hf-evidence-bindings/1",
            "records": {hf_bindings.row_digest(row): binding},
        },
    )
    points, _ = select_points(tables, include_mtp=True)
    assert points[0]["server_log_uses"] == 2  # The HF snapshot contains only one of the shared log's rows.
    assert points[0]["config"]["id"].startswith("hf:")
    parsed = ["model", {}, "version", "vllm", {}, {}]
    monkeypatch.setattr(
        runtime_evidence,
        "load_manifest",
        lambda _: {
            "schema_version": f"reviewed-runtime-observations/{archive_version}",
            "records": {
                hf_bindings.row_digest(original)
                if archive_version == 1
                else runtime_evidence.observation_id(original): {
                    "benchmark_id": row.bench_id,
                    "parsed": parsed,
                    "measurement_sha256": hf_bindings.row_digest(original),
                }
            },
        },
    )
    assert runtime_evidence.archived_recipe(row)[:5] == tuple(parsed[:5])
    changed = replace(row, metrics={"mean_ttft": 99})
    assert hf_bindings.reviewed_binding(changed) is None
    with pytest.raises(ValueError, match="measurement identity mismatch"):
        runtime_evidence.archived_recipe(changed)
    assert hf_bindings.reviewed_binding(replace(row, run_started_at="2026-10-19")) == binding


@pytest.mark.parametrize(
    "changes",
    [
        {"metrics": {"mean_ttft": 99}},
        {"image": "different"},
        {"conc": 999},
        {"head_sha": "c" * 40},
        {"decode_tp": 4},
        {"run_attempt": 5},
    ],
)
def test_hf_binding_rejects_changed_measurements(record, changes):
    point = select_points(hf.to_tables([record], {}))[0][0]
    row = SiliconRow(**point["source_row"])
    with pytest.raises(ValueError, match="identity mismatch"):
        hf_bindings.bind_row(row, replace(row, **changes), 1)
