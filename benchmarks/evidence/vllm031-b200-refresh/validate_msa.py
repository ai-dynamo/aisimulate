# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate exact MSA coverage, measurement sources and Python/native consumers."""

import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from aisimulate_core.sdk import common
from aisimulate_core.sdk.engine import _probe_handle_for, build_ops_json
from aisimulate_core.sdk.operations.msa import ContextMSAModule
from aisimulate_core.sdk.perf_database import PerfDatabase


def main():
    evidence = Path(__file__).resolve().parent
    repo = evidence.parents[2]
    systems = repo / "python/aisimulate/src/aisimulate_core/systems"
    data = systems / "data/b200_sxm/msa/vllm/0.31.0"
    publication = json.loads((evidence / "msa-publication-validation.json").read_text())
    for name, digest in publication["published_files_sha256"].items():
        assert hashlib.sha256((data / name).read_bytes()).hexdigest() == digest
    for name, digest in publication["unchanged_prior_files_sha256"].items():
        assert hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest
    raw = yaml.safe_load((evidence / "msa-raw-collection-meta.yaml").read_text())
    published = yaml.safe_load((data / "collection_meta.yaml").read_text())
    assert raw["tables"]["msa_context_module_perf"]["status"] == "complete"
    raw["tables"]["msa_context_module_perf"]["status"] = "partial"
    assert published == raw
    checkpoint = json.loads(
        (evidence / "msa-context-final-checkpoint.json").read_text()
    )
    done, failed = set(checkpoint["done"]), set(checkpoint["failed"])
    assert len(done) == len(checkpoint["done"]) == 13992
    assert len(failed) == len(checkpoint["failed"]) == 1400
    assert not done & failed and not checkpoint["attempted"]
    seed = json.loads((evidence / "msa_context_seed_provenance.json").read_text())
    groups = [set(group["keys"]) for group in seed["measured_source_groups"]]
    assert list(map(len, groups)) == [184, 6330]
    assert not groups[0] & groups[1]
    inherited = groups[0] | groups[1]
    assert inherited <= done and len(done - inherited) == 7478
    assert (
        set().union(*(set(g["keys"]) for g in seed["failure_source_groups"])) == failed
    )
    excluded = json.loads(
        (evidence / "msa_context_csr_exclusion_filters.json").read_text()
    )
    assert len(excluded) == len(set(excluded)) == 298
    prefix = "vllm.msa_context_module:run_msa_module_worker:"
    inventory = evidence / "msa-original-case-ids.json"
    assert (
        hashlib.sha256(inventory.read_bytes()).hexdigest()
        == publication["original_case_ids_sha256"]
    )
    original = json.loads(inventory.read_text())
    assert len(original) == len(set(original)) == 17568
    excluded_ids = {
        key for key in original if any(fragment in key for fragment in excluded)
    }
    assert len(excluded_ids) == 3576
    assert failed <= excluded_ids and not done & excluded_ids
    assert done | excluded_ids == set(original)
    for name, keys in {
        "original": original,
        "selected": done,
        "excluded": excluded_ids,
        "seed_done": inherited,
        "seed_failed": failed,
        "remaining": done - inherited,
        "blocked_unattempted": excluded_ids - failed,
    }.items():
        digest = hashlib.sha256(
            json.dumps(sorted(keys), separators=(",", ":")).encode()
        ).hexdigest()
        assert digest == publication["expected_key_set_hashes_json_sorted"][name], name
    rows = pq.read_table(data / "msa_context_module_perf.parquet").to_pylist()
    database = PerfDatabase(
        "b200_sxm", "vllm", "0.31.0", systems_root=str(systems), shared_layer=False
    )
    handle = _probe_handle_for(database, None)
    identities = set()
    for row in rows:
        assert (
            row["framework"],
            row["version"],
            row["device"],
            row["op_name"],
            row["kernel_source"],
            row["architecture"],
            row["tp_size"],
        ) == (
            "VLLM",
            "0.31.0",
            "NVIDIA B200",
            "msa_context_module",
            "MiniMaxM3SparseMSAImpl",
            "MiniMaxM3ForCausalLM",
            1,
        )
        args = [
            row[k]
            for k in (
                "isl",
                "batch_size",
                "num_heads",
                "kv_cache_dtype",
                "mla_dtype",
                "gemm_type",
                "model",
                "step",
            )
        ]
        identity = prefix + str(args)
        assert identity in done and identity not in identities
        identities.add(identity)
        latency = row["latency"]
        assert math.isfinite(latency) and latency > 0
        operation = ContextMSAModule(
            "msa",
            1.0,
            num_heads=row["num_heads"],
            num_kv_heads=max(1, row["num_heads"] // 16),
            hidden_size=6144,
            head_dim=128,
            v_head_dim=128,
            index_n_heads=4,
            index_head_dim=128,
            index_topk=16,
            block_size=128,
            kvcache_quant_mode=getattr(common.KVCacheQuantMode, row["kv_cache_dtype"]),
            fmha_quant_mode=getattr(common.FMHAQuantMode, row["mla_dtype"]),
            gemm_quant_mode=getattr(common.GEMMQuantMode, row["gemm_type"]),
        )
        python_value = operation._engine_query(
            database, batch_size=row["batch_size"], s=row["isl"], prefix=row["step"]
        )
        native = handle.evaluate_ops_json(
            build_ops_json([operation]),
            is_context=True,
            batch_size=row["batch_size"],
            s=row["isl"],
            prefix=row["step"],
        )[0]
        assert python_value.source == "silicon" and str(native[3]) == "silicon", (
            identity
        )
        assert math.isclose(
            float(python_value), latency, rel_tol=1e-9, abs_tol=1e-12
        ), identity
        assert math.isclose(float(native[1]), latency, rel_tol=1e-9, abs_tol=1e-12), (
            identity
        )
    assert identities == done and len(rows) == 13992
    report = {
        "system": "b200_sxm",
        "framework": "vllm",
        "version": "0.31.0",
        "shared_layer": False,
        "measured": len(rows),
        "original_cases": 17568,
        "explained_failed": len(failed),
        "source_blocked_unattempted": 2176,
        "measurement_source_rows": [184, 6330, 7478],
        "python_queries": len(rows),
        "native_queries": len(rows),
        "scope": "Exact op-level consumer checks; both interfaces share the compiled engine.",
    }
    (evidence / "msa-consumer-validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
