# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rebuild measured tables from rank evidence and verify strict native queries."""

import gzip
import json
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from collector.sglang.collect_dsv41_module import aggregate_baseline_records, aggregate_rank_records
from collector.sglang.dsv41_contract import write_parquet

import aiconfigurator_core._aiconfigurator_core as c
from aiconfigurator_core.sdk.engine import EngineHandle, _evaluate_single_op
from aiconfigurator_core.sdk.perf_database import PerfDatabase

root = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="dsv41-tables-") as temporary:
    raw = Path(temporary)
    for profile in ("full", "decoder_bounded"):
        stage = raw / profile
        stage.mkdir()
        for file in (root / profile / "evidence").glob("*.jsonl.gz"):
            (stage / file.name.removesuffix(".gz")).write_bytes(gzip.decompress(file.read_bytes()))
    baselines = aggregate_baseline_records(sorted((raw / "decoder_bounded").glob("baseline-rank-*.jsonl")), 4)
    for profile in ("full", "decoder_bounded"):
        rows = aggregate_rank_records(sorted((raw / profile).glob("rank-*.jsonl")), 4)
        rebuilt = raw / profile / "module.parquet"
        write_parquet(rows, rebuilt)
        systems = root / profile / "systems/data/gb300"
        assert pq.read_table(rebuilt).equals(
            pq.read_table(systems / "dsv41/sglang/0.0.0.dev0/dsv41_module_perf.parquet")
        )
        for kind in ("gemm", "moe", "nccl"):
            path = systems / (
                f"{kind}/sglang/0.0.0.dev0/{kind}_perf.parquet"
                if kind != "nccl"
                else "comm/nccl/2.29.7/nccl_perf.parquet"
            )
            assert pq.read_table(path).to_pylist() == baselines[kind]
summary = {}
for profile in ["full", "decoder_bounded"]:
    p = (root / profile / "systems").resolve()
    db = PerfDatabase(
        "gb300", "sglang", "0.0.0.dev0", str(p), database_mode="SILICON", shared_layer=False, strict_provenance=True
    )
    points = pq.read_table(p / "data/gb300/dsv41/sglang/0.0.0.dev0/dsv41_module_perf.parquet").to_pylist()
    bad = []
    for r in points:
        body = json.loads(r["geometry"])
        op = c.op_from_spec_json(
            json.dumps(
                {
                    "Dsv41"
                    + {"attention": "Attention", "mhc": "Mhc", "linear": "Linear", "engram": "Engram"}[
                        r["component"]
                    ]: body | {"name": "point_check"}
                }
            )
        )
        res = _evaluate_single_op(
            db,
            op,
            is_context=body.get("is_context", True),
            batch_size=r["batch_size"],
            s=r["x"],
            prefix=r["prefix"],
            x=r["x"],
        )
        if abs(float(res) - r["latency"]) > 1e-6 or res.source != "silicon":
            bad.append((r["component"], r["x"], r["prefix"], float(res), r["latency"], res.source))
    print(profile, "rows", len(points), "bad", bad[:5])
    assert not bad
    h = EngineHandle.compile(
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "gb300",
        "sglang",
        backend_version="0.0.0.dev0",
        tp_size=4,
        moe_tp_size=4,
        moe_ep_size=1,
        decoder_replay=profile == "decoder_bounded",
        systems_path=str(p),
        database_mode="SILICON",
        shared_layer=False,
        strict_provenance=True,
    )
    cases = []
    for batch in [1, 2]:
        for prefix in [0, 256]:
            for q in [3, 128, 129, 256]:
                ctx, gen = h.run_static_per_op(batch_size=batch, isl=q + prefix, osl=2, prefix=prefix)
                case = {
                    "batch": batch,
                    "prefix": prefix,
                    "query": q,
                    "prefill_ms": sum(x[1] for x in ctx),
                    "decode_ms": sum(x[1] for x in gen),
                    "context_sources": dict(Counter(x[3] for x in ctx)),
                    "generation_sources": dict(Counter(x[3] for x in gen)),
                }
                cases.append(case)
    summary[profile] = {"point_checks": len(points), "workload_cases": cases}
expected = json.loads((root / "prediction_grid.json").read_text())
assert summary == expected, "recorded predictions differ from current engine"
print("618 strict native point checks and 32 complete workload estimates passed")
