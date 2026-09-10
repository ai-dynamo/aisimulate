# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the versioned correction of six historical V4.1 indexer labels.

This changes metadata, not measurements. Original files are read-only inputs.
The actual indexer source is identified in source-proof.json and the adjacent
README; no upstream implementation is copied into this derivation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def corrected_row(original):
    row = original.copy()
    if row["component"] == "attention":
        geometry = json.loads(row["geometry"])
        require(canonical(geometry) == row["geometry"], "noncanonical original geometry")
        require(
            type(geometry["index_n_heads"]) is int and geometry["index_n_heads"] == 8, "unexpected original index heads"
        )
        require(geometry["num_heads"] == 16, "derivation is limited to the recorded TP4 geometry")
        geometry["index_n_heads"] = 32
        row["geometry"] = canonical(geometry)
    return row


def derive(historical_root, output, indexer_source):
    script = Path(__file__).resolve()
    proof_path = script.with_name("source-proof.json")
    proof = json.loads(proof_path.read_text())
    require(file_sha(indexer_source) == proof["actual_indexer_source_sha256"], "actual indexer source hash mismatch")
    require(len(proof["tables"]) == 6, "expected six frozen input tables")
    packages = set()
    source_receipts = []
    # Verify every archived source manifest, including the package digest
    # recorded on each historical row, before writing any derived dataset.
    for item in proof["source_files"]:
        path = historical_root / item["path"]
        require(file_sha(path) == item["sha256"], "archived source receipt changed")
        sources = json.loads(path.read_text())
        require(
            sources["srt/layers/attention/dsv4/dsv41_sparse.py"] == proof["actual_indexer_source_sha256"],
            "indexer source differs between runs",
        )
        require(sources["srt/models/deepseek_v4.py"] == item["native_model_sha256"], "actual model source hash drift")
        packages.add(digest(sources))
        source_receipts.append(item)
    require(len(packages) == 1, "source package identities differ")
    inventory = []
    for item in proof["tables"]:
        path = historical_root / item["path"]
        require(file_sha(path) == item["sha256"], "historical table hash changed")
        table = pq.read_table(path)
        old = table.to_pylist()
        require(len(old) == item["rows"], "historical row count changed")
        require({r["source_sha256"] for r in old} == packages, "table is not bound to the recorded source package")
        require({r["config_sha256"] for r in old} == set(item["configs"]), "config identity changed")
        require({r["runtime_digest"] for r in old} == set(item["runtime_digests"]), "runtime identity changed")
        new = [corrected_row(row) for row in old]
        require(
            sum(a != b for a, b in zip(old, new, strict=True)) == item["attention_rows"], "unexpected changed-row count"
        )
        keys = [(r["component"], r["geometry"], r["batch_size"], r["prefix"], r["x"]) for r in new]
        require(len(keys) == len(set(keys)), "corrected keys collide")
        source_systems = path.parents[5]
        require(source_systems.name == "systems", "unexpected input layout")
        relative = source_systems.relative_to(historical_root)
        if relative.parts[0] in ("full", "decoder_bounded"):
            destination = Path("initial") / relative
        elif relative.parts[0] == "prefix-refinement-v1":
            destination = Path("prefix-refinement") / Path(*relative.parts[1:])
        else:
            require(relative.parts[0] == "study", "unknown input cohort")
            destination = relative
        require(not (output / destination).exists(), "refusing to overwrite an existing derived dataset")
        inventory.append((item, path, table, old, new, source_systems, destination))
    manifests = []
    for profile in ("full", "decoder_bounded"):
        original = historical_root / "prefix-refinement-v1" / f"{profile}-manifest.json"
        manifest = json.loads(original.read_text())
        require(manifest["tp_size"] == 4 and manifest["execution_profile"] == profile, "unexpected original manifest")
        for entries in manifest["phases"].values():
            for entry in entries:
                entry["geometry"] = corrected_row(entry)["geometry"]
        target = output / f"{profile}-manifest.json"
        require(not target.exists(), "refusing to overwrite a derived manifest")
        manifests.append((original, manifest, target))
    receipts = []
    for item, original_path, table, old, new, source_systems, destination in inventory:
        target_systems = output / destination
        shutil.copytree(source_systems, target_systems)
        relative_table = original_path.relative_to(source_systems)
        target = target_systems / relative_table
        geometry = pa.array([r["geometry"] for r in new], type=table.schema.field("geometry").type)
        corrected = table.set_column(table.schema.get_field_index("geometry"), table.schema.field("geometry"), geometry)
        pq.write_table(corrected, target)
        loaded = pq.read_table(target)
        require(loaded.schema == table.schema, "Arrow schema changed")
        for column in table.column_names:
            if column != "geometry":
                require(table[column].equals(loaded[column]), f"column {column} changed")
        mappings = []
        for i, (before, after) in enumerate(zip(old, loaded.to_pylist(), strict=True)):
            require(after == new[i], "unexpected derived row content")
            bits = struct.pack("!d", before["latency"]).hex()
            require(bits == struct.pack("!d", after["latency"]).hex(), "latency bits changed")
            mappings.append(
                {
                    "row": i,
                    "original_sha256": digest(before),
                    "derived_sha256": digest(after),
                    "latency_f64_bits": bits,
                    "geometry_changed": before["geometry"] != after["geometry"],
                }
            )
        meta_path = target.parent / "collection_meta.yaml"
        metadata = yaml.safe_load(meta_path.read_text())
        require(
            metadata["tables"]["dsv41_module_perf"]["data_sha256"] == item["sha256"],
            "original sidecar did not bind original data",
        )
        metadata["tables"]["dsv41_module_perf"]["data_sha256"] = file_sha(target)
        meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
        copied = []
        for source in sorted(source_systems.rglob("*")):
            if not source.is_file() or source in (original_path, original_path.parent / "collection_meta.yaml"):
                continue
            dest = target_systems / source.relative_to(source_systems)
            require(file_sha(source) == file_sha(dest), "unrelated copied artifact changed")
            copied.append({"path": str(source.relative_to(historical_root)), "sha256": file_sha(source)})
        receipts.append(
            {
                "original_path": item["path"],
                "original_sha256": item["sha256"],
                "derived_path": str(target.relative_to(output)),
                "derived_sha256": file_sha(target),
                "rows": len(old),
                "changed_attention_rows": item["attention_rows"],
                "all_other_columns_unchanged": True,
                "latency_bits_unchanged": True,
                "rows_mapping": mappings,
                "unchanged_artifacts": copied,
            }
        )
    # Recheck originals after all output writes; never modify historical data.
    for item in proof["tables"]:
        require(
            file_sha(historical_root / item["path"]) == item["sha256"], "historical input changed during derivation"
        )
    manifest_receipts = []
    for original, manifest, target in manifests:
        with target.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2) + "\n")
        manifest_receipts.append(
            {
                "original_path": str(original.relative_to(historical_root)),
                "original_sha256": file_sha(original),
                "derived_path": target.name,
                "derived_sha256": file_sha(target),
            }
        )
    receipt = {
        "schema": "dsv41.indexer-identity-derivation.v2",
        "kind": "metadata_correction_no_new_measurements",
        "derivation_source_sha256": file_sha(script),
        "source_proof_sha256": file_sha(proof_path),
        "actual_indexer_source_sha256": proof["actual_indexer_source_sha256"],
        "change": {"column": "geometry.index_n_heads", "from": 8, "to": 32},
        "original_source_receipts": source_receipts,
        "manifests": manifest_receipts,
        "datasets": receipts,
    }
    with (output / "derivation.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--indexer-source", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = derive(args.historical_root.resolve(), args.output_dir.resolve(), args.indexer_source)
    print(json.dumps({"datasets": len(result["datasets"]), "rows": sum(d["rows"] for d in result["datasets"])}))
