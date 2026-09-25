# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage all eight accepted GLM FPM configurations for dataset ingestion.

This command reruns native/installed-consumer acceptance. It creates local
configuration partitions and their exact source-row lineage; it does not upload,
modify an existing dataset catalog, or invent an immutable Hub revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS
from collector.glm53flash_jsonl import file_sha256

from . import glm53flash_validation as validation

SCHEMA = "glm53flash_fpm_publication_stage_v1"
REPO_ID = "nvidia/aisimulate-fpm-dataset"


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def partition_table(source: Path, metadata: Path, destination: Path) -> list[dict]:
    """Preserve Arrow columns and every source row while splitting config leaves."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    original = json.loads(metadata.read_text())
    source_sha = file_sha256(source)
    table = pq.read_table(source)
    if (
        original.get("schema_name") != "aic_fpm_forward_perf"
        or original.get("schema_version") != 7
        or original.get("parquet_sha256") != source_sha
        or original.get("row_count") != table.num_rows
        or table.num_rows == 0
    ):
        raise ValueError("publication source is not a committed schema-v7 FPM table")
    partitions = {}
    for index, row in enumerate(table.to_pylist()):
        model, backend, version, tp = (row.get(key) for key in ("model_path", "backend", "backend_version", "tp"))
        expected_quant = "nvfp4" if model == "nvidia/GLM-5.3-Flash-NVFP4" else "fp8_block"
        if (
            model not in MODEL_REVISIONS
            or backend not in {"vllm", "sglang"}
            or type(tp) is not int
            or tp not in (2, 4)
            or row.get("system") != "gb300"
            or original.get("system") != "gb300"
            or original.get("backend") != backend
            or original.get("backend_version") != version
            or row.get("weight_quantization") != expected_quant
            or row.get("input_tokenizer_revision") != MODEL_REVISIONS[model]
            or row.get("workload_kind") not in {"prefill", "decode"}
            or row.get("moe_tp") != tp
            or any(row.get(key) != 1 for key in ("pp", "dp", "cp", "moe_ep"))
            or not isinstance(version, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", version)
        ):
            raise ValueError("publication table contains a row outside the GLM GB300 matrix")
        partitions.setdefault((model, backend, version, tp), []).append(index)
    outputs = []
    for (model, backend, version, tp), indices in sorted(partitions.items()):
        selected = table.take(pa.array(indices, type=pa.int64()))
        phases = sorted(set(selected["workload_kind"].to_pylist()))
        if phases != ["decode", "prefill"]:
            raise ValueError("each published configuration must contain both native phases")
        leaf = Path("data") / model.replace("/", "--") / "gb300" / backend / version / f"pure-tp{tp}"
        target = destination / leaf / "fpm/fpm_forward_perf.parquet"
        target.parent.mkdir(parents=True, exist_ok=False)
        pq.write_table(selected, target, compression="zstd")
        # Re-serialization changes file bytes; the Arrow values/schema and exact
        # original row indices, rather than a byte-copy claim, prove this split.
        if not pq.read_table(target).equals(selected):
            raise ValueError("published partition changed Arrow row values or schema")
        info = dict(original)
        # Database aic_revision is the plan renderer identity, including its
        # installed RECORD identity. It does not identify a separate worker wheel.
        planner_revision = info.get("aic_revision")
        info.update(
            parquet_sha256=file_sha256(target),
            row_count=len(indices),
            model_paths=[model],
            revision_identity_schema="glm53flash_revision_identity_v1",
            planner_revision=planner_revision,
            producer_revision=planner_revision,
            producer_revision_semantics="legacy_planner_revision_alias",
            source_partition={
                "parquet_sha256": source_sha,
                "metadata_sha256": file_sha256(metadata),
                "row_indices": indices,
                "original_row_count": table.num_rows,
            },
        )
        for field, column in (
            ("source_plan_sha256", "source_plan_sha256"),
            ("collector_attempt_ids", "collector_attempt_id"),
            ("runtime_run_ids", "runtime_run_id"),
            ("runtime_grid_digests", "runtime_grid_digest"),
        ):
            info[field] = sorted(set(selected[column].to_pylist()))
        metadata_target = target.with_suffix(".metadata.json")
        _write_json(metadata_target, info)
        outputs.append(
            {
                "model_id": model,
                "model_revision": MODEL_REVISIONS[model],
                "backend": backend,
                "backend_version": version,
                "tp": tp,
                "weight_quantization": "nvfp4" if "NVFP4" in model else "fp8",
                "phases": phases,
                "configuration_path": leaf.as_posix(),
                "parquet": {"path": target.relative_to(destination).as_posix(), "sha256": file_sha256(target)},
                "metadata": {
                    "path": metadata_target.relative_to(destination).as_posix(),
                    "sha256": file_sha256(metadata_target),
                },
                "rows": len(indices),
                "source_partition": info["source_partition"],
            }
        )
    return outputs


def stage(manifest_path: Path, destination: Path) -> dict:
    """Require fresh full-matrix acceptance before writing any publishable rows."""
    manifest_path = manifest_path.resolve()
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest.get("mode") != "fpm":
        raise ValueError("FPM publication cannot export Ops or unspecified measurements")
    destination.mkdir(parents=True, exist_ok=False)
    report = validation.evaluate(manifest, manifest_path.parent)
    _write_json(destination / "validation/acceptance.json", report)
    if report["acceptance"] != "PASSED":
        _write_json(destination / "stage.json", {"schema": SCHEMA, "status": "NOT_QUALIFIED", "files": []})
        raise ValueError("all sixteen phase cells must pass fresh installed-consumer acceptance before publication")
    sources, origins = {}, {}
    for entry in manifest["entries"]:
        for receipt in entry["consumer_data"]:
            path = (manifest_path.parent / receipt["path"]).resolve()
            digest = file_sha256(path)
            if digest != receipt["sha256"]:
                raise ValueError("consumer data changed after independent acceptance")
            if path in sources and sources[path] != digest:
                raise ValueError("consumer source has inconsistent receipts")
            sources[path] = digest
            origins.setdefault(path, set()).add(receipt["path"])
    tables = [path for path in sources if path.name == "fpm_forward_perf.parquet"]
    if not tables:
        raise ValueError("accepted manifest has no native FPM tables")
    configurations = []
    for table in sorted(tables):
        metadata = table.with_suffix(".metadata.json")
        if metadata not in sources:
            raise ValueError("accepted source table lacks a metadata receipt")
        configurations.extend(partition_table(table, metadata, destination))
    keys = [(row["backend"], row["weight_quantization"], row["tp"]) for row in configurations]
    if len(keys) != 8 or set(keys) != {key[:3] for key in validation.REQUIRED}:
        raise ValueError("publication partitions must cover exactly the eight accepted configurations")
    # Archive source bytes once even when sixteen phase entries share them.
    source_receipts = {}
    for path, digest in sorted(sources.items()):
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("consumer source changed during partition export")
        if digest in source_receipts:
            receipt = source_receipts[digest]
            receipt["original_consumer_paths"] = sorted(set(receipt["original_consumer_paths"]) | origins[path])
            continue
        target = destination / "sources" / f"{digest}{path.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(raw)
        source_receipts[digest] = {
            "path": target.relative_to(destination).as_posix(),
            "sha256": digest,
            "original_consumer_paths": sorted(origins[path]),
        }
    if manifest_path.read_bytes() != manifest_raw:
        raise ValueError("acceptance input manifest changed during publication staging")
    input_target = destination / "validation/input-manifest.json"
    with input_target.open("xb") as stream:
        stream.write(manifest_raw)
    index = {
        "schema": SCHEMA,
        "status": "STAGED_NOT_PUBLISHED",
        "repo_id": REPO_ID,
        "input_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "input_manifest": {
            "path": "validation/input-manifest.json",
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
        },
        "acceptance": {
            "path": "validation/acceptance.json",
            "sha256": file_sha256(destination / "validation/acceptance.json"),
        },
        "sources": [source_receipts[key] for key in sorted(source_receipts)],
        "configurations": configurations,
        "remaining": [
            "canonical dataset policy and catalogs",
            "Hub immutable commit",
            "consumer pin",
            "installed offline validation",
        ],
    }
    _write_json(destination / "stage.json", index)
    return index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    stage(args.manifest, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
