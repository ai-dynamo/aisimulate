# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolate measured decode rows and an anchor-only copy for SDK validation.

These are private diagnostic systems roots. All generation DSA donor files are
removed before inserting the measured slice, so shared fallback cannot reveal
withheld observations. Input rows, column types and framework identities are
preserved; no latency model or production-data installation occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--systems-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-history", type=int, default=8192)
    parser.add_argument("--heldout-history", nargs="+", type=int, default=[131072, 1048575])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Use a new output directory to preserve earlier receipts")
    histories = {args.anchor_history, *args.heldout_history}
    if len(histories) != 1 + len(args.heldout_history):
        raise ValueError("Anchor and held-out histories must be distinct")
    source = pq.read_table(args.source_parquet)
    variants = {"dsa_generation_module", "dsa_generation_module_skip_indexer"}
    rows = [
        row
        for row in source.to_pylist()
        if row["model"] == "nvidia/GLM-5.2-NVFP4"
        and row["architecture"] == "GlmMoeDsaForCausalLM"
        and row["mla_dtype"] == "bfloat16"
        and row["kv_cache_dtype"] == "fp8"
        and row["gemm_type"] == "bfloat16"
        and row["num_heads"] == 8
        and row["batch_size"] == 1
        and row["isl"] == 1
        and row["step"] in histories
        and row["op_name"] in variants
    ]
    coordinates = {(row["step"], row["op_name"]) for row in rows}
    expected = {(history, variant) for history in histories for variant in variants}
    if coordinates != expected or len(rows) != len(expected):
        raise ValueError("Expected exactly one full and one reuse observation at every requested history")
    if any(row["framework"] != "SGLang" or row["version"] != "0.5.14" for row in rows):
        raise ValueError("This validation requires actual SGLang 0.5.14 observations")
    manifest = {
        "source_sha256": hashlib.sha256(args.source_parquet.read_bytes()).hexdigest(),
        "source_rows": rows,
        "anchor_history": args.anchor_history,
        "heldout_history": args.heldout_history,
        "overlays": {},
    }
    for name, selected in [("exact", rows), ("anchor", [r for r in rows if r["step"] == args.anchor_history])]:
        destination = args.output_dir / name / "systems"
        shutil.copytree(args.systems_root, destination)
        removed = []
        for donor in sorted(destination.rglob("dsa_generation_module_perf.parquet")):
            removed.append(str(donor.relative_to(destination)))
            donor.unlink()
        target = destination / "data/b200_sxm/sparse_attention/sglang/0.5.14/dsa_generation_module_perf.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(selected, schema=source.schema), target)
        if pq.read_table(target).to_pylist() != selected:
            raise RuntimeError("Parquet roundtrip changed an observation")
        manifest["overlays"][name] = {
            "removed_generation_sources": removed,
            "rows": len(selected),
            "parquet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    (args.output_dir / "overlay-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
