# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate measured ARM DSA rows through the normal public current query slot."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from aisimulate import _runtime
from aisimulate_core.sdk import common
from aisimulate_core.sdk.engine import EngineHandle, _ops_json
from aisimulate_core.sdk.operations.dsa import ContextDSAModule, GenerationDSAModule
from aisimulate_core.sdk.perf_database import get_database, resolve_query_version


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=["gb200", "gb300"])
    parser.add_argument("--source-leaf", type=Path, required=True)
    parser.add_argument("--systems-root", type=Path, required=True)
    parser.add_argument("--snapshot-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not os.environ.get("AIC_ALLOW_UNLISTED_VERSIONS")
    assert args.source_leaf.name == "0.0.0.dev1+ge79de62d2.d20261104"
    assert len(args.snapshot_head) == 40
    assert not args.output.exists(), "Preserve existing receipt rather than overwrite"
    expected = {
        "dsa_context_module_perf.parquet",
        "dsa_generation_module_perf.parquet",
        "collection_meta.yaml",
    }
    assert {path.name for path in args.source_leaf.iterdir()} == expected
    target = args.systems_root / "data" / args.system / "sparse_attention" / "sglang" / args.source_leaf.name
    target.mkdir(parents=True, exist_ok=False)
    for name in sorted(expected):
        shutil.copy2(args.source_leaf / name, target / name)
        assert sha(target / name) == sha(args.source_leaf / name)
    version = resolve_query_version(args.system, "sglang", "current", systems_paths=str(args.systems_root))
    db = get_database(
        args.system,
        "sglang",
        "current",
        systems_paths=str(args.systems_root),
        database_mode="SILICON",
        strict_provenance=True,
    )
    assert db is not None and db.enable_shared_layer and db.version == version
    engine = EngineHandle.for_database(db, systems_path=str(args.systems_root))
    checks = []
    for phase in ["context", "generation"]:
        path = target / f"dsa_{phase}_module_perf.parquet"
        candidates = [row for row in pq.read_table(path).to_pylist() if row["tp_size"] == 8 and row["batch_size"] == 1]
        row = min(candidates, key=lambda item: (item["isl"], item["step"]))
        assert row["architecture"] == "KimiK31ForConditionalGeneration"
        positional = ("k31", 1.0, row["num_heads"], common.KVCacheQuantMode.fp8)
        kwargs = {"architecture": row["architecture"], "attention_backend": "flashmla"}
        op = (
            ContextDSAModule(*positional, common.FMHAQuantMode.bfloat16, common.GEMMQuantMode.bfloat16, **kwargs)
            if phase == "context"
            else GenerationDSAModule(*positional, common.GEMMQuantMode.bfloat16, **kwargs)
        )
        result = engine.evaluate_ops_json(
            _ops_json([op]),
            is_context=phase == "context",
            batch_size=row["batch_size"],
            s=row["isl"] if phase == "context" else row["isl"] + row["step"],
            prefix=row["step"] if phase == "context" else 0,
        )[0]
        assert result[3] == "silicon" and result[1] == row["latency"], (phase, row, result)
        checks.append(
            {
                "phase": phase,
                "tp": 8,
                "batch_size": row["batch_size"],
                "isl": row["isl"],
                "step": row["step"],
                "num_heads": row["num_heads"],
                "latency_ms": result[1],
                "source": result[3],
                "source_table_sha256": sha(path),
            }
        )
    receipt = {
        "system": args.system,
        "base_systems_snapshot_head": args.snapshot_head,
        "query_slot": "current",
        "resolved_query_version": version,
        "measured_source_version": args.source_leaf.name,
        "strict_provenance": True,
        "database_mode": "SILICON",
        "shared_layer_argument": "omitted (public default)",
        "effective_shared_layer": bool(db.enable_shared_layer),
        "native_sha256": sha(Path(_runtime.__file__)),
        "verification_script_sha256": sha(Path(__file__)),
        "query_versions_sha256": sha(args.systems_root / "query_versions.yaml"),
        "system_config_sha256": sha(args.systems_root / f"{args.system}.yaml"),
        "overlay_source_files": {name: sha(target / name) for name in sorted(expected)},
        "checks": checks,
        "scope": (
            "Normal public current query-slot backward-fill returns measured native source rows with strict provenance "
            "and exact SILICON latency. This does not qualify the resolved newer serving runtime or establish "
            "prediction accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {"system": args.system, "passed": len(checks), "current": version, "receipt_sha256": sha(args.output)}
        )
    )


if __name__ == "__main__":
    main()
