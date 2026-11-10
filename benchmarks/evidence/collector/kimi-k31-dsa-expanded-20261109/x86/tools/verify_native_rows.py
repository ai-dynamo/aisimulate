# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate every exported row using an isolated strict native consumer."""

import argparse
import ast
import hashlib
import json
import math
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from aisimulate_core.sdk import common
from aisimulate_core.sdk.engine import EngineHandle, _ops_json
from aisimulate_core.sdk.engine_table_view import fetch_table_view
from aisimulate_core.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule
from aisimulate_core.sdk.perf_database import PerfDatabase

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--evidence", type=Path, required=True)
parser.add_argument("--system-yaml", type=Path, required=True)
parser.add_argument("--plan", type=Path, required=True)
parser.add_argument("--raw-root", type=Path, required=True)
args = parser.parse_args()
system = args.system_yaml.stem
version = "0.0.0.dev1+g41208373c.d20261104"
root = args.evidence / "consumer-systems"
leaf = root / "data" / system / "sparse_attention" / "sglang" / version
leaf.mkdir(parents=True, exist_ok=False)
shutil.copy2(args.system_yaml, root / args.system_yaml.name)
source_files = [*sorted(args.source.glob("*.parquet")), args.source / "collection_meta.yaml"]
for file in source_files:
    shutil.copy2(file, leaf / file.name)
db = PerfDatabase(system, "sglang", version, str(root), database_mode="HYBRID", strict_provenance=True)
handle = EngineHandle.for_database(db, systems_path=str(root))
seen = set()
results = []
for path in sorted(leaf.glob("*.parquet")):
    for row in pq.read_table(path).to_pylist():
        phase = "context" if row["op_name"] == "dsa_context_module" else "generation"
        assert row["architecture"] == "KimiK31ForConditionalGeneration"
        heads, batch, length, step = (int(row[k]) for k in ("num_heads", "batch_size", "isl", "step"))
        key = (phase, heads, batch, length, step)
        assert key not in seen, key
        seen.add(key)
        assert int(row["tp_size"]) * heads == 96
        view = fetch_table_view(db, f"_{phase}_dsa_module_data")
        if phase == "context":
            view = view[common.FMHAQuantMode.bfloat16]
        buckets = view[common.KVCacheQuantMode.fp8][common.GEMMQuantMode.bfloat16][row["architecture"]]
        assert set(buckets) == {"flashmla_kv"}, buckets.keys()
        constructor_args = ("k31_native", 1.0, heads, common.KVCacheQuantMode.fp8)
        if phase == "context":
            op = ContextDSAModule(
                *constructor_args,
                common.FMHAQuantMode.bfloat16,
                common.GEMMQuantMode.bfloat16,
                architecture=row["architecture"],
                attention_backend="flashmla",
            )
        else:
            op = GenerationDSAModule(
                *constructor_args,
                common.GEMMQuantMode.bfloat16,
                architecture=row["architecture"],
                attention_backend="flashmla",
            )
        result = handle.evaluate_ops_json(
            _ops_json([op]),
            is_context=phase == "context",
            batch_size=batch,
            s=length if phase == "context" else length + step,
            prefix=step if phase == "context" else 0,
        )
        assert math.isfinite(float(row["latency"])) and float(row["latency"]) > 0
        assert result[0][1] == float(row["latency"]), (row, result)
        assert result[0][3] == "silicon", (row, result)
        results.append(
            dict(
                phase=phase,
                batch=batch,
                isl=length,
                step=step,
                tp=int(row["tp_size"]),
                kernel_source=row["kernel_source"],
                latency_ms=result[0][1],
            )
        )
expected = set()
for op, cases in json.loads(args.plan.read_text()).items():
    phase = "context" if "context" in op else "generation"
    for item in cases:
        seq, batch, heads, _kv, _compute, _gemm, _model, prefix = item["case"]
        expected.add((phase, heads, batch, seq if phase == "context" else 1, prefix if phase == "context" else seq - 1))
failed = set()
errors = []
for path in args.raw_root.rglob("collection_summary_sglang.json"):
    for error in json.loads(path.read_text())["errors"]:
        if error.get("error_type") != "FullNodeCaseFailure":
            continue
        phase = "context" if "context" in error["module"] else "generation"
        seq, batch, heads, _kv, _compute, _gemm, _model, prefix = ast.literal_eval(error["task_params"])
        key = (phase, heads, batch, seq if phase == "context" else 1, prefix if phase == "context" else seq - 1)
        if key not in expected:
            continue
        failed.add(key)
        errors.append(error)
assert not seen & failed, "failed cases produced rows"
assert seen | failed == expected, {
    "unaccounted": sorted(expected - seen - failed),
    "unexpected": sorted((seen | failed) - expected),
}
report = {
    "system": system,
    "strict_provenance": True,
    "rows_verified": len(results),
    "attempted_cases": len(expected),
    "failed_cases": sorted(failed),
    "failed_events": errors,
    "source_files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in source_files],
    "results": results,
    "scope": "Exact native SILICON retrieval validates exported data integrity; this is not an accuracy score.",
}
(args.evidence / "consumer-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {k: report[k] for k in ["system", "strict_provenance", "rows_verified", "attempted_cases", "failed_cases"]}
    )
)
