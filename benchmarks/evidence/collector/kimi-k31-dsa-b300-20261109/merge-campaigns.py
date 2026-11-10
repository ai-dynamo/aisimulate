# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combine disjoint native measurements without changing rows or collection events."""

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from collector.provenance import append_collection_event, render_collection_meta, validate_collection_meta_for_update

original, chunked, output = (Path(arg).resolve() for arg in sys.argv[1:4])
output.mkdir(parents=True, exist_ok=True)
metas = [yaml.safe_load((path / "collection_meta.yaml").read_text()) for path in (original, chunked)]
for meta in metas:
    validate_collection_meta_for_update(meta, tables_to_update={"dsa_generation_module_perf"})
assert metas[0]["runtime"] == metas[1]["runtime"]
name = "dsa_generation_module_perf"
parts = [pq.read_table(path / f"{name}.parquet") for path in (original, chunked)]
assert parts[0].schema == parts[1].schema
keys = []
for part in parts:
    keys.append(
        {
            tuple(
                row[k]
                for k in (
                    "architecture",
                    "num_heads",
                    "batch_size",
                    "isl",
                    "step",
                    "tp_size",
                    "kv_cache_dtype",
                    "mla_dtype",
                    "gemm_type",
                    "kernel_source",
                )
            )
            for row in part.to_pylist()
        }
    )
assert len(keys[0]) == parts[0].num_rows and len(keys[1]) == parts[1].num_rows
assert not keys[0] & keys[1], "measurement keys overlap; no implicit replacement permitted"
combined = pa.concat_tables(parts)
pq.write_table(combined, output / f"{name}.parquet", compression="zstd")
roundtrip = pq.read_table(output / f"{name}.parquet")
canon = lambda rows: Counter(json.dumps(row, sort_keys=True) for row in rows)
assert canon(roundtrip.to_pylist()) == canon(parts[0].to_pylist() + parts[1].to_pylist()), (
    "not an exact all-column row union"
)
tables = dict(metas[0]["tables"])
count = parts[0].num_rows
source_events = metas[1]["tables"][name].get("collections", [metas[1]["tables"][name]])
for event in source_events:
    count += event["rows"]
    tables[name] = append_collection_event(tables[name], event, table=name, merged_rows=count)
assert count == combined.num_rows
(output / "collection_meta.yaml").write_bytes(render_collection_meta(metas[0]["runtime"], tables))
shutil.copy2(original / "dsa_context_module_perf.parquet", output / "dsa_context_module_perf.parquet")
manifest = {
    "schema": "disjoint-native-op-union/v1",
    "operation": (
        "Arrow concatenation of disjoint immutable native rows; "
        "provenance.append_collection_event/render_collection_meta preserve all source events."
    ),
    "original_git_commit": "e6b7c36d5c998037599544d75c644d5d690f3ab1",
    "derivation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "invocation": ["python", "merge-campaigns.py", "<original60>", "<chunked12>", "<combined>"],
    "inputs": [
        {
            "campaign": label,
            "files": [
                {"name": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
                for file in sorted(path.glob("*.parquet"))
            ]
            + [
                {
                    "name": "collection_meta.yaml",
                    "sha256": hashlib.sha256((path / "collection_meta.yaml").read_bytes()).hexdigest(),
                }
            ],
        }
        for label, path in (("original60", original), ("chunked12", chunked))
    ],
    "outputs": [
        {"name": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        for file in sorted(output.glob("*.parquet"))
    ]
    + [
        {
            "name": "collection_meta.yaml",
            "sha256": hashlib.sha256((output / "collection_meta.yaml").read_bytes()).hexdigest(),
        }
    ],
    "validation": {
        "disjoint_consumer_keys": True,
        "exact_all_column_row_union": True,
        "generation_rows": combined.num_rows,
        "context_rows": pq.read_metadata(output / "dsa_context_module_perf.parquet").num_rows,
        "original_events_preserved": True,
        "new_events_preserved": True,
    },
}
(output / "derivation.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest["validation"]))
