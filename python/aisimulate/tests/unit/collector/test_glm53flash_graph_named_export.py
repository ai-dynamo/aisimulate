# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TEST_ONLY named analysis semantics and original trace rederivation."""

import copy

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from collector import glm53flash_graph_export as graph
from collector import glm53flash_validation as native
from collector.glm53flash_jsonl import file_sha256

from ..sdk.test_glm53flash_graph_named_consumer import authored_proof
from .test_glm53flash_graph_export import control_fixture, fixture

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
def test_same_geometry_distinct_named_collectives_keep_actual_medians(backend):
    proof = authored_proof(backend)
    legacy, selection = graph.aggregate_graph(proof, evidence_sha256="a" * 64)
    rows, named_selection = graph.aggregate_graph(proof, evidence_sha256="a" * 64, lookup_contract=graph.NAMED_CONTRACT)
    assert selection == named_selection
    assert len(rows) == (367 if backend == "sglang" else 278) * 2
    embedding = next(r for r in rows if r["operation_name"] == "embedding_allreduce" and r["prefix"] == 128)
    pooled = next(r for r in legacy if (r["geometry"], r["prefix"]) == (embedding["geometry"], 128))
    assert embedding["latency"] == 0.8
    assert pooled["latency"] == 0.004
    for group in proof["forwards"].values():
        for row in group.values():
            row["binding"]["embedding_allreduce"]["latency"] *= 2
            row["binding"]["embedding_allreduce"]["dispatch"] = "c" * 64
    changed, _ = graph.aggregate_graph(proof, evidence_sha256="a" * 64, lookup_contract=graph.NAMED_CONTRACT)
    for old, new in zip(rows, changed, strict=True):
        if old["operation_name"] == "embedding_allreduce":
            assert new["latency"] == 2 * old["latency"]
            assert new["dispatch_fingerprint"] == "c" * 64
        else:
            assert new == old
    with pytest.raises(ValueError, match="mixes dispatch"):
        graph.aggregate_graph(proof, evidence_sha256="a" * 64)


@pytest.mark.parametrize(
    "defect",
    [
        "manifest",
        "geometry",
        "binding",
        "warmup",
        "measure",
        "role",
        "duplicate_point",
        "within_unit_dispatch",
        "contract",
    ],
)
def test_named_reducer_rejects_incomplete_or_mixed_originals(defect):
    proof = authored_proof()
    contract = graph.NAMED_CONTRACT
    if defect == "manifest":
        proof["manifest"]["phases"]["generation"].pop()
    elif defect == "geometry":
        proof["manifest"]["phases"]["generation"][1]["geometry"] = "{}"
    elif defect == "binding":
        proof["forwards"][1, 5][1]["binding"].pop("embedding_allreduce")
    elif defect == "warmup":
        proof["forwards"].pop((1, 0))
    elif defect == "measure":
        proof["forwards"].pop((1, 14))
    elif defect == "role":
        proof["forwards"][1, 5][0]["sampling_role"] = "warmup"
    elif defect == "duplicate_point":
        for rep in range(15):
            proof["forwards"][3, rep] = copy.deepcopy(proof["forwards"][1, rep])
    elif defect == "within_unit_dispatch":
        proof["forwards"][1, 5][1]["binding"]["embedding_allreduce"]["dispatch"] = "d" * 64
    else:
        contract = "unknown"
    with pytest.raises(ValueError):
        graph.aggregate_graph(proof, evidence_sha256="a" * 64, lookup_contract=contract)


def exported_original(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    original = tmp_path / "original"
    original.mkdir()
    graph.export_graph(root, run, original / graph.BASENAME, **control_fixture(tmp_path, monkeypatch))
    return run, root, original


def test_named_republish_rederives_original_traces_and_preserves_every_original(tmp_path, monkeypatch):
    run, root, original = exported_original(tmp_path, monkeypatch)
    before = {p.name: file_sha256(p) for p in root.iterdir() if p.is_file()}
    output = tmp_path / "named"
    output.mkdir()
    graph.republish_named_graph(root, run, output / graph.BASENAME)
    assert {p.name: file_sha256(p) for p in root.iterdir() if p.is_file()} == before
    loaded = native._load_native(run, root, calibration_evidence=False)
    bound = graph.bind_calibration([output / graph.BASENAME], run, loaded)
    assert bound["lookup_contract"] == graph.NAMED_CONTRACT
    assert graph.bind_calibration([original / graph.BASENAME], run, loaded).get("lookup_contract") is None
    nullable = tmp_path / "nullable"
    nullable.mkdir()
    legacy_rows = pq.read_table(original / graph.BASENAME).to_pylist()
    for row in legacy_rows:
        row.update(graph_lookup_contract=None, operation_name=None)
    pq.write_table(pa.Table.from_pylist(legacy_rows), nullable / graph.BASENAME)
    assert graph.bind_calibration([nullable / graph.BASENAME], run, loaded).get("lookup_contract") is None
    rows = pq.read_table(output / graph.BASENAME).to_pylist()
    rows[0]["latency"] += 1
    pq.write_table(pa.Table.from_pylist(rows), output / graph.BASENAME)
    with pytest.raises(ValueError, match="differs from original"):
        graph.bind_calibration([output / graph.BASENAME], run, loaded)


@pytest.mark.parametrize("defect", ["trace", "name", "contract", "pooled"])
def test_named_republish_never_recovers_from_changed_or_pooled_data(tmp_path, monkeypatch, defect):
    run, root, original = exported_original(tmp_path, monkeypatch)
    output = tmp_path / "named"
    output.mkdir()
    if defect == "trace":
        trace = next(root.glob("graph-profile-*.json"))
        trace.write_bytes(trace.read_bytes() + b" ")
        with pytest.raises(ValueError):
            graph.republish_named_graph(root, run, output / graph.BASENAME)
        return
    graph.republish_named_graph(root, run, output / graph.BASENAME)
    loaded = native._load_native(run, root, calibration_evidence=False)
    rows = pq.read_table(output / graph.BASENAME).to_pylist()
    if defect == "name":
        rows[0]["operation_name"] = "not_captured"
    elif defect == "contract":
        rows[0]["graph_lookup_contract"] = None
    else:
        rows = pq.read_table(original / graph.BASENAME).to_pylist()
        for row in rows:
            row.update(operation_name="embedding_allreduce", graph_lookup_contract=graph.NAMED_CONTRACT)
    pq.write_table(pa.Table.from_pylist(rows), output / graph.BASENAME)
    with pytest.raises(ValueError):
        graph.bind_calibration([output / graph.BASENAME], run, loaded)


def test_initial_explicit_named_export_uses_same_original_evidence_contract(tmp_path, monkeypatch):
    run, root = fixture(tmp_path, monkeypatch)
    output = tmp_path / graph.BASENAME
    graph.export_graph(
        root, run, output, lookup_contract=graph.NAMED_CONTRACT, **control_fixture(tmp_path, monkeypatch)
    )
    loaded = native._load_native(run, root, calibration_evidence=True)
    bound = graph.bind_calibration([output], run, loaded)
    assert bound["lookup_contract"] == graph.NAMED_CONTRACT
    assert bound["native_runtime_run_id"] == loaded["runtime_run_id"]
    assert len(pq.read_table(output).to_pylist()) == 3
