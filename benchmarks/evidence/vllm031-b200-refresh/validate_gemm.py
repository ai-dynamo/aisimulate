# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate canonical GEMM identities and exact Python/native consumer values."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from aisimulate_core.sdk import common
from aisimulate_core.sdk.engine import _probe_handle_for, build_ops_json
from aisimulate_core.sdk.operations.gemm import GEMM
from aisimulate_core.sdk.perf_database import PerfDatabase


def main():
    evidence = Path(__file__).resolve().parent
    systems = evidence.parents[2] / "python/aisimulate/src/aisimulate_core/systems"
    table = systems / "data/b200_sxm/gemm/vllm/0.31.0/gemm_perf.parquet"
    digest = hashlib.sha256(table.read_bytes()).hexdigest()
    publication = json.loads((evidence / "gemm-publication-validation.json").read_text())
    assert digest == publication["merged_parquet_sha256"]
    rows = pq.read_table(table).to_pylist()
    offset = 0
    for event in publication["events"]:
        event_rows = rows[offset : offset + event["rows"]]
        fingerprint = hashlib.sha256(json.dumps(event_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert fingerprint == event["ordered_rows_sha256"]
        offset += event["rows"]
    assert offset == len(rows)
    spec = yaml.safe_load((systems / "b200_sxm.yaml").read_text())["gpu"]
    database = PerfDatabase("b200_sxm", "vllm", "0.31.0", systems_root=str(systems), shared_layer=False)
    handle = _probe_handle_for(database, None)
    older_fp8 = {tuple(case) for case in json.loads((evidence / "fp8-older-allowed-cases.json").read_text())["cases"]}
    assert len(older_fp8) == 1606
    identities = set()
    counts, floors, sources = Counter(), Counter(), Counter()
    examples = []
    for row in rows:
        assert (row["framework"], row["version"], row["device"], row["op_name"]) == (
            "VLLM",
            "0.31.0",
            "NVIDIA B200",
            "gemm",
        ), row
        quant, m, n, k = (row[key] for key in ("gemm_dtype", "m", "n", "k"))
        identity = (quant, m, n, k)
        assert identity not in identities, identity
        identities.add(identity)
        assert all(isinstance(v, int) and v > 0 for v in (m, n, k))
        raw = row["latency"]
        assert math.isfinite(raw) and raw > 0, row
        assert quant in ("bfloat16", "fp8", "fp8_block", "nvfp4"), row
        assert quant != "fp8" or m == 1 or identity in older_fp8, row
        counts[f"{quant}/{row['kernel_source']}"] += 1
        # Mirrors gemm.rs::gemm_sol_with_flops; no fp8_static overhead
        # subtraction applies to these measured quant modes. NVFP4's
        # 9/16 bytes includes its scale overhead (common/enums.rs).
        width, flops_key = {
            "bfloat16": (2, "bfloat16_tc_flops"),
            "fp8": (1, "fp8_tc_flops"),
            "fp8_block": (1, "fp8_tc_flops"),
            "nvfp4": (9 / 16, "fp4_tc_flops"),
        }[quant]
        flops = spec[flops_key]
        sol = max(2 * m * n * k / flops, width * (m * n + m * k + n * k) / spec["mem_bw"]) * 1000
        expected = max(raw, sol)
        operation = GEMM("gemm", 1.0, n, k, getattr(common.GEMMQuantMode, quant))
        python_value = float(operation._engine_query(database, x=m))
        native = handle.evaluate_ops_json(build_ops_json([operation]), is_context=False, batch_size=m, s=1)[0]
        native_value = float(native[1])
        assert math.isclose(python_value, expected, rel_tol=1e-6), (
            identity,
            raw,
            sol,
            python_value,
        )
        assert math.isclose(native_value, expected, rel_tol=1e-6), (
            identity,
            raw,
            sol,
            native,
        )
        assert str(native[3]) == "silicon", (identity, native)
        sources[str(native[3])] += 1
        if sol > raw:
            floors[quant] += 1
            if len(examples) < 8:
                examples.append(
                    {
                        "identity": identity,
                        "measured_ms": raw,
                        "sol_ms": sol,
                        "query_ms": native_value,
                    }
                )
    assert older_fp8 <= identities
    assert len(rows) == len(identities) == 113557
    assert counts == {
        "bfloat16/torch.nn.functional.linear": 37518,
        "fp8/CutlassFP8ScaledMMLinearKernel": 2113,
        "fp8_block/CutlassFp8BlockScaledMMKernel": 5328,
        "fp8_block/DeepGemmFp8BlockScaledMMKernel": 31080,
        "nvfp4/FlashInferCuteDslNvFp4LinearKernel": 37518,
    }, counts
    result = {
        "system": "b200_sxm",
        "framework": "vllm",
        "version": "0.31.0",
        "parquet_sha256": digest,
        "shared_layer": False,
        "passed": len(rows),
        "failed": 0,
        "unique_consumer_identities": len(identities),
        "counts": dict(counts),
        "sources": dict(sources),
        "sol_floor_counts": dict(floors),
        "sol_floor_examples": examples,
        "expected_value": "max(measured latency, mathematical SOL, memory SOL); raw measurements remain unchanged",
        "loader_contract": (
            "First source wins; all published dtype/M/N/K identities are unique; no fp8_static correction applies."
        ),
        "interfaces": [
            "Python orchestration _engine_query",
            "native evaluate_ops_json",
        ],
        "legacy_note": (
            "Public legacy query was removed; the retained Python orchestration delegates to the native engine."
        ),
        "coverage": {
            "selected": 114667,
            "measured": 113557,
            "historical_framework_failures": 1110,
            "unattempted": 0,
        },
        "table_status": (
            "partial: 35405 distinct FP8 reuse keys and explicitly excluded GLM-specific cases "
            "remain outside fresh publication; 1110 historical block-FP8 gaps remain"
        ),
    }
    (evidence / "gemm-consumer-validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
