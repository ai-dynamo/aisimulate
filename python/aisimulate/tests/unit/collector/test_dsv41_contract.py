# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path

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
    assert full["config_sha256"] == "d7637228d27528f6bd259781b5a27258068f50bf637c9c83aab784d81579669d"
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


def test_writer_to_native_silicon_query_roundtrip(tmp_path):
    import aiconfigurator_core._aiconfigurator_core as core
    from aiconfigurator_core.sdk.engine import _evaluate_single_op
    from aiconfigurator_core.sdk.perf_database import PerfDatabase

    package = Path(__file__).resolve().parents[3]
    shutil.copy(package / "src/aiconfigurator_core/systems/gb300.yaml", tmp_path / "gb300.yaml")
    data = tmp_path / "data/gb300/sglang/0.0.0.dev0"
    data.mkdir(parents=True)
    point = row()
    write_parquet([point], data / "dsv41_module_perf.parquet")
    operation = core.op_from_spec_json(
        json.dumps({"Dsv41Attention": json.loads(point["geometry"]) | {"name": "generation_attention"}})
    )
    database = PerfDatabase(
        "gb300", "sglang", "0.0.0.dev0", str(tmp_path), database_mode="SILICON", strict_provenance=False
    )
    result = _evaluate_single_op(database, operation, is_context=False, batch_size=1, s=129, x=1)
    assert float(result) == pytest.approx(0.125)
    assert result.source == "silicon"


def test_baseline_rank_admission_converts_nccl_bytes_to_elements(tmp_path):
    from collector.sglang.collect_dsv41_module import aggregate_baseline_records

    paths = []
    for rank in range(2):
        path = tmp_path / f"baseline-rank-{rank}.jsonl"
        point = {
            "kind": "nccl",
            "nccl_dtype": "half",
            "num_gpus": 2,
            "op_name": "all_reduce",
            "message_size": 10240,
            "latency": rank + 1.0,
            "sample": 2,
            "tp_rank": rank,
            "source_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "runtime_digest": "sha256:" + "c" * 64,
            "used_cuda_graph": False,
            "kernel_source": "torch.distributed.nccl.all_reduce",
            "execution_profile": "decoder_bounded",
        }
        path.write_text(json.dumps(point) + "\n")
        paths.append(path)
    result = aggregate_baseline_records(paths, 2)["nccl"][0]
    assert result["message_size"] == 5120
    assert result["wire_dtype"] == "bfloat16"
    assert result["latency"] == 2.0
    paths[0].write_text(paths[0].read_text() * 2)
    with pytest.raises(ValueError, match="duplicate baseline rank/sample"):
        aggregate_baseline_records(paths, 2)
