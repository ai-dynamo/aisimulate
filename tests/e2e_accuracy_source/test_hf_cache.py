# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
import requests

from scripts.e2e_accuracy import hf_cache as cache


def row(identifier="1", hardware="h200"):
    return {
        "id": identifier,
        "model": "example",
        "hardware": hardware,
        "framework": "sglang",
        "run_url": "https://github.com/example/bench/actions/runs/123/attempts/2",
        "date": "2026-10-01",
        "curve_workflow_run_id": "456",
        "metrics": {
            "mean_ttft": 0.4,
            "mean_tpot": 0,
            "p99_tpot": 0.025,
            "mean_itl": None,
            "output_tput_per_gpu": 70,
            "mean_intvty": 50,
        },
    }


def acquisition(value, time="2026-10-01T00:00:00Z"):
    return cache.Acquisition(
        "https://inferencex.semianalysis.com/api/v1/benchmarks?model=Example",
        cache.canonical(value),
        time,
    )


def prepare(root, rows):
    return cache.prepare_snapshot(
        root,
        acquisition({"models": [{"name": "Example", "dbKeys": ["example"]}]}),
        {"Example": acquisition(rows)},
        normalizer_revision="a" * 40,
        publication={"redistribution_status": "unresolved"},
    )


def test_metric_semantics_and_provenance():
    observations, metrics = cache.normalize("Example", [row()])
    assert observations[0]["source_config_id"] is None
    assert observations[0]["producer_run_url"].endswith("runs/123/attempts/2")
    assert "curve_workflow_run_id" not in observations[0]["raw_record_json"]
    fields = {m["source_metric"]: m for m in metrics}
    assert fields["mean_ttft"]["value"] == 400
    assert fields["mean_tpot"]["value"] == 0
    assert fields["mean_itl"]["value"] is None
    assert fields["mean_itl"]["status"] == "missing"
    assert fields["p99_tpot"]["percentile"] == 99
    assert fields["output_tput_per_gpu"]["normalization"] == "per_device"
    assert "mean_intvty" not in fields


def test_conflicting_duplicate_fails():
    changed = row()
    changed["metrics"]["mean_ttft"] = 10
    with pytest.raises(ValueError, match="Conflicting"):
        cache.normalize("Example", [row(), changed])


def test_delta_reuses_unchanged_partitions_and_snapshot(tmp_path):
    inputs = [row(), row("2", "b200")]
    first = prepare(tmp_path, inputs)
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert prepare(tmp_path, inputs) == first
    assert {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    changed = copy.deepcopy(inputs)
    changed[0]["metrics"]["mean_ttft"] = 0.5
    second = prepare(tmp_path, changed)
    assert second["snapshot_id"] != first["snapshot_id"]
    old_b200 = next(p for p in first["partitions"] if p["hardware"] == "b200")
    assert old_b200 in second["partitions"]
    cache.validate_snapshot(tmp_path, first)
    cache.validate_snapshot(tmp_path, second)


def test_failed_preparation_preserves_index(tmp_path):
    prepare(tmp_path, [row()])
    original = (tmp_path / cache.INDEX_PATH).read_bytes()
    with pytest.raises(ValueError, match="Missing metrics"):
        prepare(tmp_path, [{"id": "3"}])
    assert (tmp_path / cache.INDEX_PATH).read_bytes() == original


def test_hash_corruption_and_unsafe_paths(tmp_path):
    index = prepare(tmp_path, [row()])
    snapshot = json.loads((tmp_path / index["snapshot"]["path"]).read_bytes())
    raw = tmp_path / snapshot["acquisitions"][0]["artifact"]["path"]
    raw.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        cache.validate_snapshot(tmp_path, index)
    for name in ("../secret", "/tmp/secret", "data/../../secret", "data\\secret"):
        with pytest.raises(ValueError, match="Unsafe"):
            cache.safe_path(tmp_path, name)


def test_curve_context_does_not_change_observation_identity():
    first = cache.normalize("Example", [row()])
    inherited = row()
    inherited["curve_workflow_run_id"] = "999"
    assert cache.normalize("Example", [inherited]) == first


def test_empty_response_is_valid_but_not_empty_plan(tmp_path):
    index = prepare(tmp_path, [])
    assert not index["partitions"]
    with pytest.raises(ValueError, match="Empty acquisition"):
        cache.prepare_snapshot(
            tmp_path,
            acquisition({"models": []}),
            {},
            normalizer_revision="a" * 40,
            publication={},
        )


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def response(status, content=b"[]", **headers):
    r = requests.Response()
    r.status_code = status
    r._content = content
    r.headers.update(headers)
    return r


def test_conditional_cache_and_corruption(tmp_path):
    session = Session([response(200, ETag='"v1"'), response(304)])
    client = cache.InferenceXClient(tmp_path, session)
    assert client.get("benchmarks", model="Example").body == b"[]"
    assert client.get("benchmarks", model="Example").body == b"[]"
    assert session.calls[1][1]["headers"] == {"If-None-Match": '"v1"'}
    next(tmp_path.glob("*.body")).write_bytes(b"corrupt")
    client.session = Session([response(304)])
    with pytest.raises(ValueError, match="verified cached"):
        client.get("benchmarks", model="Example")


def test_bounded_retry_and_invalid_run_id(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(cache.time, "sleep", sleeps.append)
    session = Session([response(429, **{"Retry-After": "2"}), response(500), response(200)])
    client = cache.InferenceXClient(tmp_path, session)
    client.get("benchmarks", model="Example")
    assert len(session.calls) == 3 and sleeps == [2, 2]
    with pytest.raises(ValueError, match="positive integer"):
        client.get("benchmarks", runId="latest")
    client.session = Session([response(503)] * 4)
    with pytest.raises(requests.HTTPError):
        client.get("benchmarks", model="Example")


def test_unrelated_source_commit_does_not_create_snapshot(tmp_path):
    first = prepare(tmp_path, [row()])
    second = cache.prepare_snapshot(
        tmp_path,
        acquisition({"models": [{"name": "Example", "dbKeys": ["example"]}]}),
        {"Example": acquisition([row()])},
        normalizer_revision="b" * 40,
        publication={"redistribution_status": "unresolved"},
    )
    assert second == first
