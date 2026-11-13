# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact union of disjoint canonical TP tables; preserve source collection events."""

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from collector.provenance import append_collection_event, render_collection_meta, validate_collection_meta_for_update

from aisimulate_core.sdk.perf_database import _load_collection_meta_yaml

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--receipt", type=Path, required=True)
parser.add_argument("sources", type=Path, nargs="+")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
metas = [yaml.safe_load((p / "collection_meta.yaml").read_text()) for p in args.sources]
raw_metas = copy.deepcopy(metas)
migrations = []
for source_index, meta in enumerate(metas):
    source_path = args.sources[source_index] / "collection_meta.yaml"
    # The strict SDK schema reader is deliberately not replaced by the writer's validator.
    assert _load_collection_meta_yaml(str(source_path)) == raw_metas[source_index]
    for table_name, entry in meta["tables"].items():
        if meta["schema_version"] != 1 or "classified_failures" not in entry:
            continue
        value = entry["classified_failures"]
        if type(value) is not int or value != 0:
            raise ValueError("v1-to-v2 migration only supports an explicit integer zero classified_failures")
        original = copy.deepcopy(entry)
        del entry["classified_failures"]
        assert {**entry, "classified_failures": value} == original
        migrations.append(
            {
                "source_index": source_index,
                "source_sidecar_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "table": table_name,
                "original_v1_event": original,
                "projected_v2_event": copy.deepcopy(entry),
                "omitted_v1_field": {"classified_failures": 0},
                "reason": (
                    "Schema-v2 event contract has no classified_failures field or replacement. "
                    "Original v1 bytes retain explicit zero; every legal event field is unchanged."
                ),
            }
        )
assert all(m["runtime"] == metas[0]["runtime"] for m in metas)
names = sorted(set.union(*(set(m["tables"]) for m in metas)))
for m in metas:
    validate_collection_meta_for_update(m, tables_to_update=set(m["tables"]))
output_tables = {}
counts = {}
for name in names:
    indexes = [i for i, m in enumerate(metas) if name in m["tables"]]
    parts = [pq.read_table(args.sources[i] / f"{name}.parquet") for i in indexes]
    assert all(t.schema == parts[0].schema for t in parts)
    seen = set()
    for part in parts:
        keys = {
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
        assert len(keys) == part.num_rows and not keys & seen, "overlapping measurement keys"
        seen.update(keys)
    combined = pa.concat_tables(parts)
    target = args.output / f"{name}.parquet"
    pq.write_table(combined, target, compression="zstd")
    canonical = lambda rows: Counter(json.dumps(row, sort_keys=True) for row in rows)
    assert canonical(pq.read_table(target).to_pylist()) == canonical([r for p in parts for r in p.to_pylist()])
    table = copy.deepcopy(metas[indexes[0]]["tables"][name])
    count = parts[0].num_rows
    for index in indexes[1:]:
        source = metas[index]["tables"][name]
        for event in source.get("collections", [source]):
            count += event["rows"]
            table = append_collection_event(table, event, table=name, merged_rows=count)
    assert count == combined.num_rows, (name, count, combined.num_rows)
    output_tables[name] = table
    counts[name] = combined.num_rows
(args.output / "collection_meta.yaml").write_bytes(render_collection_meta(metas[0]["runtime"], output_tables))
rendered = yaml.safe_load((args.output / "collection_meta.yaml").read_text())
assert _load_collection_meta_yaml(str(args.output / "collection_meta.yaml")) == rendered
for name in names:
    expected = [
        event
        for meta in metas
        if name in meta["tables"]
        for event in meta["tables"][name].get("collections", [meta["tables"][name]])
    ]
    actual = rendered["tables"][name].get("collections", [rendered["tables"][name]])
    assert Counter(json.dumps(e, sort_keys=True) for e in expected) == Counter(
        json.dumps(e, sort_keys=True) for e in actual
    )
    for migration in [item for item in migrations if item["table"] == name]:
        assert migration["projected_v2_event"] in actual
        assert {**migration["projected_v2_event"], **migration["omitted_v1_field"]} == migration["original_v1_event"]


def files(p):
    return {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(p.iterdir())
        if f.name == "collection_meta.yaml" or f.suffix == ".parquet"
    }


receipt = {
    "schema": "disjoint-canonical-tp-union-v2",
    "event_schema_migrations": migrations,
    "sources": [{"path": str(p.resolve()), "files": files(p)} for p in args.sources],
    "output": str(args.output.resolve()),
    "output_files": files(args.output),
    "counts": counts,
    "validation": {
        "all_column_rows_preserved": True,
        "source_collection_event_semantics_preserved": True,
        "original_v1_sidecar_bytes_preserved": True,
        "strict_sdk_schema_reader_passed": True,
        "disjoint_measurement_keys": True,
    },
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"counts": counts, "output": str(args.output)}))
