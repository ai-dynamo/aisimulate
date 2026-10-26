# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate paired quantization measurements and their native consumer contract."""

import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from aisimulate_core.sdk import common
from aisimulate_core.sdk.engine import _probe_handle_for, build_ops_json
from aisimulate_core.sdk.engine_table_view import fetch_table_view
from aisimulate_core.sdk.operations.gemm import GEMM
from aisimulate_core.sdk.perf_database import PerfDatabase


def main():
    evidence = Path(__file__).resolve().parent
    repo = evidence.parents[2]
    systems = repo / "python/aisimulate/src/aisimulate_core/systems"
    data = systems / "data/b200_sxm/quantize/vllm/0.31.0"
    publication = json.loads(
        (evidence / "compute-scale-publication-validation.json").read_text()
    )
    for name, digest in publication["original_files_sha256"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == digest, name
    for name, digest in publication["unchanged_prior_files_sha256"].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest, name
    database = PerfDatabase(
        "b200_sxm", "vllm", "0.31.0", systems_root=str(systems), shared_layer=False
    )
    handle = _probe_handle_for(database, None)
    components, zeros = {}, []
    for op, filename, kernel in (
        (
            "compute_scale",
            "computescale_perf.parquet",
            "dynamic_per_token_scaled_fp8_quant_minus_static_scaled_fp8_quant",
        ),
        ("scale_matrix", "scale_matrix_perf.parquet", "static_scaled_fp8_quant"),
    ):
        rows = pq.read_table(data / filename).to_pylist()
        view = fetch_table_view(database, f"_{op}_data")
        native = json.loads(handle._engine.table_view_json(f"_{op}_data"))
        assert set(native) == {"fp8"}
        keys = {(int(m), int(k)) for m, by_k in native["fp8"].items() for k in by_k}
        measured = {}
        for row in rows:
            assert tuple(
                row[k]
                for k in (
                    "framework",
                    "version",
                    "device",
                    "op_name",
                    "kernel_source",
                    "quant_dtype",
                )
            ) == ("VLLM", "0.31.0", "NVIDIA B200", op, kernel, "fp8"), row
            m, k, latency = row["m"], row["k"], row["latency"]
            assert isinstance(m, int) and isinstance(k, int) and min(m, k) > 0
            assert (m, k) not in measured
            assert math.isfinite(latency) and (
                latency >= 0 if op == "compute_scale" else latency > 0
            )
            measured[m, k] = latency
            assert view[common.GEMMQuantMode.fp8][m][k]["latency"] == latency
            assert native["fp8"][str(m)][str(k)]["latency"] == latency
            if latency == 0:
                zeros.append([m, k])
        assert len(rows) == len(measured) == 1628 and keys == set(measured)
        components[op] = measured
    assert set(components["compute_scale"]) == set(components["scale_matrix"])
    assert sorted(zeros) == [[4353, 32], [8192, 32], [16384, 32], [32768, 32]]

    # These supporting tables have no standalone public query. Exercise their
    # real consumer at every M/K through fp8_static GEMM, with both input modes.
    # A fixed N=256 is a consumer probe, not a new measured GEMM identity.
    spec = yaml.safe_load((systems / "b200_sxm.yaml").read_text())["gpu"]
    floors, queries = 0, 0
    for (m, k), overhead in components["compute_scale"].items():
        n = 256
        dynamic = GEMM("gemm", 1.0, n, k, common.GEMMQuantMode.fp8)
        baseline = float(dynamic._engine_query(database, x=m))
        sol = (
            max(
                2 * m * n * k / spec["fp8_tc_flops"],
                (m * n + m * k + n * k) / spec["mem_bw"],
            )
            * 1000
        )
        for low_precision in (False, True):
            subtracted = (
                baseline
                - overhead
                - (components["scale_matrix"][m, k] if low_precision else 0)
            )
            expected = max(subtracted, sol)
            operation = GEMM(
                "gemm",
                1.0,
                n,
                k,
                common.GEMMQuantMode.fp8_static,
                low_precision_input=low_precision,
            )
            python_value = float(operation._engine_query(database, x=m))
            native = handle.evaluate_ops_json(
                build_ops_json([operation]), is_context=False, batch_size=m, s=1
            )[0]
            assert math.isclose(python_value, expected, rel_tol=1e-6, abs_tol=1e-12), (
                m,
                k,
                low_precision,
                python_value,
                expected,
            )
            assert math.isclose(
                float(native[1]), expected, rel_tol=1e-6, abs_tol=1e-12
            ), (m, k, low_precision, native, expected)
            assert str(native[3]) == "estimated", native
            floors += subtracted < sol
            queries += 1
    result = {
        "system": "b200_sxm",
        "framework": "vllm",
        "version": "0.31.0",
        "shared_layer": False,
        "paired_shapes": 1628,
        "table_identities_checked": 3256,
        "failed": 0,
        "python_and_native_table_views_exact": True,
        "preserved_zero_overhead_keys": sorted(zeros),
        "static_quantization_rows_strictly_positive": 1628,
        "fp8_static_consumer_queries_per_interface": queries,
        "consumer_n": 256,
        "consumer_source": "estimated",
        "consumer_sol_floor_count": floors,
        "prior_gemm_bmm_rows_byte_unchanged": 114193,
        "interfaces": [
            "Python fetch_table_view",
            "native table_view_json",
            "Python GEMM._engine_query",
            "native evaluate_ops_json",
        ],
        "legacy_note": "Public query_compute_scale/query_scale_matrix APIs are retired. Both retained interfaces use the compiled engine; this is not an independent legacy implementation.",
        "contract": "compute_scale is max(0, dynamic/5 - static/5); scale_matrix is static/5. fp8_static subtracts overhead, also static quantization for low-precision inputs, and applies the existing GEMM SOL floor.",
        "limits": "No new GPU measurements, numerical proof for every shape, or end-to-end accuracy claim. The consumer probes do not turn estimated fp8_static GEMM into measured data.",
    }
    (evidence / "compute-scale-consumer-validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
