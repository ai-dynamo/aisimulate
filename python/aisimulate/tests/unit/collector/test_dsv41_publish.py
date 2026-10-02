# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from collector.sglang.dsv41_publish import BASELINE_COLUMNS, IDENTITY, _rank_maxima, pool_runs

pytestmark = pytest.mark.unit

IDENT = dict(source_sha256="s" * 64, config_sha256="c" * 64, runtime_digest="sha256:" + "d" * 64, used_cuda_graph=False, execution_profile="full", kernel_source="k")


def _write_run(tmp_path: Path, name: str, tp: int, latencies: dict[int, list[float]]):
    raw = tmp_path / name
    raw.mkdir()
    for rank in range(tp):
        lines = []
        for sample, values in latencies.items():
            lines.append(json.dumps(dict(component="mhc", geometry="{}", batch_size=1, prefix=0, x=8, latency=values[rank], sample=sample, invocation=0, tp_rank=rank, **IDENT)))
        (raw / f"rank-{rank}.jsonl").write_text("\n".join(lines) + "\n")
    return raw


def test_rank_maxima_takes_the_slowest_rank_per_sample(tmp_path):
    raw = _write_run(tmp_path, "tp2", 2, {0: [1.0, 3.0], 1: [2.0, 0.5]})
    out = _rank_maxima(raw, "rank-{rank}.jsonl", 2, lambda r: (r["component"], r["x"]), ("sample", "invocation", "tp_rank"))
    template, maxima = out[("mhc", 8)]
    assert sorted(maxima) == [2.0, 3.0]
    assert "tp_rank" not in template and template["latency"] == 1.0  # template is the first raw row minus run-local fields


def test_pool_runs_concatenates_samples_of_shared_keys(tmp_path):
    a = _rank_maxima(_write_run(tmp_path, "tp2", 2, {0: [1.0, 1.0], 1: [3.0, 3.0]}), "rank-{rank}.jsonl", 2, lambda r: r["x"], ("sample", "invocation", "tp_rank"))
    b = _rank_maxima(_write_run(tmp_path, "tp4", 4, {0: [2.0] * 4}), "rank-{rank}.jsonl", 4, lambda r: r["x"], ("sample", "invocation", "tp_rank"))
    rows = pool_runs([a, b], (*IDENTITY, "kernel_source"))
    assert len(rows) == 1 and rows[0]["sample_count"] == 3 and rows[0]["latency"] == 2.0  # median of [1, 3, 2]


def test_pool_runs_refuses_mixed_identity(tmp_path):
    a = _rank_maxima(_write_run(tmp_path, "tp2", 2, {0: [1.0, 1.0]}), "rank-{rank}.jsonl", 2, lambda r: r["x"], ("sample", "invocation", "tp_rank"))
    b = _rank_maxima(_write_run(tmp_path, "tp4", 4, {0: [2.0] * 4}), "rank-{rank}.jsonl", 4, lambda r: r["x"], ("sample", "invocation", "tp_rank"))
    next(iter(b.values()))[0]["kernel_source"] = "other"
    with pytest.raises(ValueError, match="measurement identity"):
        pool_runs([a, b], (*IDENTITY, "kernel_source"))


def test_rank_maxima_rejects_incomplete_rank_set(tmp_path):
    raw = _write_run(tmp_path, "tp2", 2, {0: [1.0, 1.0]})
    (raw / "rank-1.jsonl").write_text("")
    with pytest.raises(ValueError, match="incomplete rank set"):
        _rank_maxima(raw, "rank-{rank}.jsonl", 2, lambda r: r["x"], ("sample", "invocation", "tp_rank"))


def test_baseline_columns_match_the_published_table_schemas():
    # keyed exactly like collect_dsv41_module.aggregate_baseline_records; the published parquet
    # columns are these plus latency/kernel_source/sample_count (+ wire_dtype for nccl)
    assert BASELINE_COLUMNS["gemm"] == ("gemm_dtype", "m", "n", "k")
    assert BASELINE_COLUMNS["nccl"] == ("op_name", "nccl_dtype", "num_gpus", "message_size")
    assert BASELINE_COLUMNS["moe"][-1] == "distribution"
