#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the PR665 normalization against the archived, unmodified append."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from aisimulate_core.sdk.engine import EngineHandle
from aisimulate_core.sdk.perf_database import get_database_view

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TABLE = (
    ROOT
    / "python/aisimulate/src/aisimulate_core/systems/data/b300_sxm/sparse_attention/sglang/0.5.21"
    / "dsa_generation_module_perf.parquet"
)


def main():
    report = json.loads((HERE / "normalization.json").read_text())
    archived = HERE / "original-append.parquet"
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == report["original_sha256"]
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == report["normalized_sha256"]
    original = pq.read_table(archived).to_pandas()
    shipped = pq.read_table(TABLE).to_pandas()
    keys = report["physical_key"]
    assert len(original) == 6620 and len(shipped) == 6312
    assert not shipped.duplicated(keys).any()
    expected = original.drop_duplicates(keys, keep="last").reset_index(drop=True)
    pd.testing.assert_frame_equal(shipped.reset_index(drop=True), expected)
    new = original[original.model == "nvidia/GLM-5.2-NVFP4"]
    pd.testing.assert_frame_equal(
        shipped[shipped.model == "nvidia/GLM-5.2-NVFP4"].reset_index(drop=True),
        new.reset_index(drop=True),
    )
    assert len(new) == 500
    assert len(original) - len(shipped) == 308
    db = get_database_view(
        "b300_sxm",
        "sglang",
        "0.5.21",
        systems_paths=str(ROOT / "python/aisimulate/src/aisimulate_core/systems"),
        database_mode="SILICON",
        shared_layer=False,
        strict_provenance=True,
    )
    handle = EngineHandle.for_database(db, systems_path=db.systems_root)
    checked = 0
    for model in ["nvidia/GLM-5.2-NVFP4", "zai-org/GLM-5.2"]:
        group = shipped[(shipped.model == model) & (shipped.num_heads == 8)]
        for _, row in group.groupby("op_name", sort=True).head(2).iterrows():
            full_fraction = 0.0 if "skip_indexer" in row.op_name else 1.0
            op = {
                "DsaGeneration": {
                    "name": "physical coordinate regression",
                    "scale_factor": 1.0,
                    "num_heads": int(row.num_heads),
                    "kv_cache_dtype": row.kv_cache_dtype,
                    "fmha_quant_mode": row.mla_dtype,
                    "gemm_quant_mode": row.gemm_type,
                    "architecture": row.architecture,
                    "index_topk": 2048,
                    "cp_size": 1,
                    "dcp_size": 1,
                    "full_frac": full_fraction,
                }
            }
            result = handle.evaluate_ops_json(
                json.dumps([op]),
                is_context=False,
                batch_size=int(row.batch_size),
                s=int(row.isl + row.step),
                x=int(row.batch_size),
            )[0]
            assert abs(result[1] - row.latency) < 1e-9, (row.to_dict(), result)
            checked += 1
    assert checked >= 4
    print(f"PASS: Rust consumer reproduced {checked} exact full/skip added and unchanged coordinates")
    print("PASS: 6312 unique physical rows; 500 new rows unchanged; 308 superseded rows archived")


if __name__ == "__main__":
    main()
