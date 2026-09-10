# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow.parquet as pq
import pytest
from collector.sglang.collect_dsv41_module import aggregate_rank_records, get_dsv41_module_test_cases
from collector.sglang.dsv41_contract import build_manifest, operation_geometry, validate_row, write_parquet

pytestmark = pytest.mark.unit


def row():
    manifest = build_manifest(4, False)
    attention = next(e for e in manifest["phases"]["generation"] if e["component"] == "attention")
    return {
        "component": "attention",
        "geometry": attention["geometry"],
        "batch_size": 1,
        "prefix": 0,
        "x": 129,
        "latency": 0.125,
        "kernel_source": "test.native_dispatch",
        "measurement_scope": "local_compute",
        "source_sha256": "a" * 64,
        "config_sha256": manifest["config_sha256"],
        "runtime_digest": "sha256:" + "b" * 64,
        "used_cuda_graph": False,
        "sample_count": 5,
        "kv_seed_regime": "real_kv",
        "execution_profile": "full",
    }


def test_manifest_matches_native_graph_and_profiles():
    full, replay = build_manifest(4, False), build_manifest(4, True)
    assert full["config_sha256"] == replay["config_sha256"]
    for entry in replay["phases"]["context"]:
        if entry["component"] == "attention":
            assert json.loads(entry["geometry"])["bounded_prefill"] == (entry["layer"] >= 21)
    assert operation_geometry({"name": "display", "n": 32, "k": 64}) == '{"k":64,"n":32}'
    assert len(get_dsv41_module_test_cases()) == 8


@pytest.mark.parametrize(
    "mutation",
    [
        {"kv_seed_regime": "zero_filled"},
        {"measurement_scope": "whole_module_includes_collective"},
        {"runtime_digest": "latest"},
        {"prefix": 3},
        {"batch_size": 1.5},
        {"latency": float("nan")},
    ],
)
def test_rejects_unqualified_measurements(mutation):
    with pytest.raises(ValueError):
        validate_row(row() | mutation)


def test_parquet_contract_rejects_duplicate_and_mixed_provenance(tmp_path):
    target = tmp_path / "dsv41_module_perf.parquet"
    write_parquet([row()], target)
    schema = pq.read_schema(target)
    assert str(schema.field("x").type) == "int64"
    assert str(schema.field("used_cuda_graph").type) == "bool"
    with pytest.raises(ValueError, match="duplicate"):
        write_parquet([row(), row()], target)
    with pytest.raises(ValueError, match="one immutable"):
        write_parquet([row(), row() | {"x": 130, "source_sha256": "c" * 64}], target)


def test_bounded_context_requires_actual_tail_and_real_prefix():
    manifest = build_manifest(4, True)
    entry = next(e for e in manifest["phases"]["context"] if e["component"] == "attention" and e["layer"] == 21)
    point = row() | {"geometry": entry["geometry"], "prefix": 1152, "x": 128, "execution_profile": "decoder_bounded"}
    validate_row(point)
    for change in ({"x": 129}, {"execution_profile": "full"}, {"kv_seed_regime": "n/a"}):
        with pytest.raises(ValueError):
            validate_row(point | change)


def test_rank_aggregation_requires_complete_distinct_invocations(tmp_path):
    paths = [tmp_path / f"rank-{rank}.jsonl" for rank in range(2)]
    for rank, path in enumerate(paths):
        path.write_text(
            "\n".join(
                json.dumps(row() | {"sample": 1, "invocation": invocation, "tp_rank": rank, "latency": value + rank})
                for invocation, value in ((1, 1.0), (2, 3.0))
            )
        )
    result = aggregate_rank_records(paths, 2)
    assert result[0]["latency"] == 3.0  # median(max(1,2), max(3,4))
    assert result[0]["sample_count"] == 2
    with pytest.raises(ValueError, match="rank files"):
        aggregate_rank_records(paths[:1], 2)
    paths[0].write_text(paths[0].read_text() + "\n" + paths[0].read_text().splitlines()[0])
    with pytest.raises(ValueError, match="duplicate TP rank"):
        aggregate_rank_records(paths, 2)
