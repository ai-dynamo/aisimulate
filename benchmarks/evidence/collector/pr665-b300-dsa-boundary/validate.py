#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the PR665 normalization against the archived, unmodified append."""

import hashlib
import json
import zipfile
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
    for hashes_path in [HERE / "artifact-hashes.json", *HERE.glob("recollection/artifact-hashes.json")]:
        artifact_hashes = json.loads(hashes_path.read_text())
        with zipfile.ZipFile(hashes_path.parent / artifact_hashes["archive"]) as archive:
            for name, hashes in artifact_hashes["files"].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == hashes["executed_sha256"]
                assert (
                    hashlib.sha256((hashes_path.parent / name).read_bytes()).hexdigest() == hashes["published_sha256"]
                )
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
    replacements = pd.DataFrame(columns=original.columns)
    if report.get("recollected_tables"):
        replacements = pd.concat([pd.read_csv(HERE / path) for path in report["recollected_tables"]], ignore_index=True)
        assert len(replacements) == report["recollected_rows"] == 44
        assert replacements.batch_size.eq(52).all()
        assert not replacements.duplicated(keys).any()
        replace_keys = pd.MultiIndex.from_frame(replacements[keys])
        expected = expected[~pd.MultiIndex.from_frame(expected[keys]).isin(replace_keys)]
        expected = pd.concat([expected, replacements], ignore_index=True)
    pd.testing.assert_frame_equal(
        shipped.sort_values(keys).reset_index(drop=True), expected.sort_values(keys).reset_index(drop=True)
    )
    new = original[original.model == "nvidia/GLM-5.2-NVFP4"]
    retained = shipped[shipped.model == "nvidia/GLM-5.2-NVFP4"]
    if len(replacements):
        retained = retained[retained.batch_size != 52]
        new = new[new.batch_size != 52]
    pd.testing.assert_frame_equal(
        retained.sort_values(keys).reset_index(drop=True), new.sort_values(keys).reset_index(drop=True)
    )
    assert len(new) == 500 - len(replacements)
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
        samples = pd.concat(
            [group.groupby("op_name", sort=True).head(2), group[group.batch_size == 52].groupby("op_name").tail(2)]
        ).drop_duplicates(keys)
        for _, row in samples.iterrows():
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
    print(
        f"PASS: 6312 unique physical rows; {len(new)} historical additions unchanged; "
        f"{len(replacements)} B52 rows recollected; 308 superseded rows archived"
    )


if __name__ == "__main__":
    main()
