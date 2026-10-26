# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check frozen GDN coverage, published bytes, and every exact consumer key."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from aisimulate_core.sdk.engine import _probe_handle_for, build_ops_json
from aisimulate_core.sdk.operations.mamba import GDNKernel
from aisimulate_core.sdk.perf_database import PerfDatabase


def main():
    evidence = Path(__file__).resolve().parent
    repo = evidence.parents[2]
    systems = repo / "python/aisimulate/src/aisimulate_core/systems"
    data = systems / "data/b200_sxm/linear_attention/vllm/0.31.0"
    publication = json.loads((evidence / "gdn-publication-validation.json").read_text())
    baseline = json.loads((evidence / "gdn-acceptance-inventory.json").read_text())
    callbacks = json.loads((evidence / "gdn-frozen-callbacks.json").read_text())
    for name, digest in publication["original_files_sha256"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == digest
    for name, digest in publication["unchanged_prior_files_sha256"].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
    metadata = yaml.safe_load((data / "collection_meta.yaml").read_text())
    assert metadata["runtime"] == publication["runtime"]
    case_ids = ["vllm.gdn:run_gdn_torch:" + str(item["args"]) for item in callbacks]
    assert len(case_ids) == len(set(case_ids)) == 84
    plan_hash = (
        "sha256:" + hashlib.sha256("\n".join(sorted(case_ids)).encode()).hexdigest()
    )
    assert metadata["tables"]["gdn_perf"]["case_plan_hash"] == plan_hash
    assert (
        metadata["tables"]["gdn_perf"]["collector_hash"]
        == publication["collector_hash"]
    )
    fields = baseline["expected"]["identity_fields"]
    expected = set()
    for item in callbacks:
        phase, dm, dc, hk, dk, hv, dv, batches, lengths, model = item["args"]
        kernels = (
            ("causal_conv1d_fn", "chunk_gated_delta_rule_flashinfer")
            if phase == "context"
            else (
                "causal_conv1d_update",
                "fused_recurrent_gated_delta_rule_packed_decode",
            )
        )
        for batch in batches:
            for seq in lengths if phase == "context" else [1]:
                for kernel in kernels:
                    values = [
                        phase,
                        batch,
                        seq,
                        batch * seq,
                        dm,
                        dc,
                        hk,
                        dk,
                        hv,
                        dv,
                        model,
                        kernel,
                    ]
                    expected.add(tuple(values))
    rows = pq.read_table(data / "gdn_perf.parquet").to_pylist()
    actual = [tuple(row[field] for field in fields) for row in rows]
    assert len(rows) == len(set(actual)) == len(expected) == 10332
    assert set(actual) == expected
    assert (
        Counter(row["kernel_source"] for row in rows)
        == baseline["expected"]["per_kernel"]
    )
    database = PerfDatabase(
        "b200_sxm", "vllm", "0.31.0", systems_root=str(systems), shared_layer=False
    )
    handle = _probe_handle_for(database, None)
    consumer_keys = set()
    python_queries = native_queries = logical_queries = 0
    for row in rows:
        assert (row["framework"], row["version"], row["device"], row["op_name"]) == (
            "vLLM",
            "0.31.0",
            "NVIDIA B200",
            "gdn",
        )
        latency = row["latency"]
        assert math.isfinite(latency) and latency > 0
        kernel = row["kernel_source"]
        # Packed decode normalization is the existing on-disk loading contract.
        canonical = (
            "fused_sigmoid_gating_delta_rule_update"
            if kernel == "fused_recurrent_gated_delta_rule_packed_decode"
            else kernel
        )
        geometry = tuple(
            row[k]
            for k in (
                "d_model",
                "num_k_heads",
                "head_k_dim",
                "num_v_heads",
                "head_v_dim",
                "d_conv",
            )
        )
        key = (canonical, row["phase"], geometry, row["batch_size"], row["seq_len"])
        assert key not in consumer_keys, (
            "Distinct model labels collide in the consumer key"
        )
        consumer_keys.add(key)
        sources = [canonical]
        if kernel == "chunk_gated_delta_rule_flashinfer":
            sources.append("chunk_gated_delta_rule")
        for source in sources:
            operation = GDNKernel("gdn", 1.0, source, row["phase"], *geometry)
            python_value = operation._engine_query(
                database, batch_size=row["batch_size"], s=row["seq_len"]
            )
            native = handle.evaluate_ops_json(
                build_ops_json([operation]),
                is_context=row["phase"] == "context",
                batch_size=row["batch_size"],
                s=row["seq_len"],
            )[0]
            assert python_value.source == "silicon" and str(native[3]) == "silicon", (
                key,
                python_value,
                native,
            )
            assert math.isclose(
                float(python_value), latency, rel_tol=1e-9, abs_tol=1e-12
            ), (key, python_value, latency)
            assert math.isclose(
                float(native[1]), latency, rel_tol=1e-9, abs_tol=1e-12
            ), (key, native, latency)
            python_queries += 1
            native_queries += 1
            logical_queries += source in {
                "chunk_gated_delta_rule",
                "fused_sigmoid_gating_delta_rule_update",
            }
    report = {
        "system": "b200_sxm",
        "framework": "vllm",
        "version": "0.31.0",
        "shared_layer": False,
        "callbacks": 84,
        "exact_measurement_identities": len(actual),
        "unique_consumer_keys": len(consumer_keys),
        "python_queries": python_queries,
        "native_queries": native_queries,
        "logical_recurrence_queries_per_interface": logical_queries,
        "silicon_queries_per_interface": native_queries,
        "failed": 0,
        "prior_rows_byte_unchanged": publication["prior_rows_byte_unchanged"],
        "limits": "Python orchestration and native evaluation share the same compiled engine. No standalone retired legacy API, independent second implementation, new GPU measurement, reuse approval, or end-to-end accuracy claim.",
    }
    (evidence / "gdn-consumer-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
