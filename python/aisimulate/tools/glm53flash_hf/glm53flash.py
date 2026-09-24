# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Original GLM campaign policy. No native results are manufactured here.

This module consumes AISimulate publication-stage-v1 and the public dataset's
manifest/catalog interfaces. It does not include upstream dataset source code.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

CAMPAIGN = "glm53flash-pr324"
POLICY = "glm53flash-accepted-arrow-partitions-v1"
POLICY_MODULES = (
    "glm53flash.py",
    "raw_campaign.py",
    "raw_archive.py",
    "external_control.py",
    "external_control_vllm.py",
)
MODELS = {
    "zai-org/GLM-5.3-Flash": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    "nvidia/GLM-5.3-Flash-NVFP4": "09b04e5e74bca08ca8549fc736d4cdd8624bfde3",
}
KEYS = {
    (b, q, t, p) for b in ("vllm", "sglang") for q in ("fp8", "nvfp4") for t in (2, 4) for p in ("prefill", "decode")
}
IDENTITY = [
    "model_path",
    "system",
    "backend",
    "backend_version",
    "weight_quantization",
    "kv_cache_dtype",
    "parallel_strategy",
    "tp",
    "pp",
    "dp",
    "moe_tp",
    "moe_ep",
    "cp",
]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked(root, receipt):
    rel = PurePosixPath(receipt["path"])
    require(not rel.is_absolute() and ".." not in rel.parts, "escaping receipt path")
    target = root / str(rel)
    require(target.resolve().is_relative_to(root.resolve()), "escaping receipt symlink")
    require(sha(target) == receipt["sha256"], "receipt digest mismatch")
    return target


def validate_acceptance(report):
    require(
        report.get("schema") == "glm53flash_independent_holdout_v1"
        and report.get("mode") == "fpm"
        and report.get("acceptance") == "PASSED"
        and report.get("threshold_mape_pct") == 10
        and not report.get("errors")
        and not report.get("test_only"),
        "not an accepted FPM report",
    )
    consumer = report.get("consumer", {})
    require(
        consumer.get("distribution") == "aisimulate"
        and consumer.get("api") == "RustForwardPassPerfModel.best_available"
        and re.fullmatch(r"[0-9a-f]{64}", consumer.get("payload_sha256", "")),
        "installed consumer identity missing",
    )
    seen = set()
    requested = 0
    for cell in report["cells"]:
        key = tuple(cell[k] for k in ("backend", "weight_quantization", "tp", "phase"))
        require(key in KEYS and key not in seen, "unknown or duplicate acceptance cell")
        seen.add(key)
        require(
            cell.get("acceptance") == "PASSED" and not cell.get("errors"),
            "failed phase cell",
        )
        points = cell["points"]
        require(
            points and all(p.get("status") == "MEASURED_AND_PREDICTED" for p in points),
            "incomplete holdout predictions",
        )
        ape = []
        groups = set()
        for point in points:
            observed, predicted = point["measured_ms"], point["prediction_ms"]
            require(
                all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (observed, predicted)),
                "invalid holdout latency",
            )
            ape.append(100 * abs(predicted - observed) / observed)
            groups.add(point["context_group"])
        mape = sum(ape) / len(ape)
        require(
            mape <= 10
            and math.isclose(cell["metrics"]["mape_pct"], mape, abs_tol=1e-9)
            and cell["metrics"]["coverage"] == 1
            and cell["metrics"]["requested"] == cell["metrics"]["compared"] == len(points)
            and {"1K-32K", "64K", "128K"} <= groups,
            "holdout coverage or accuracy mismatch",
        )
        require(
            all(cell.get(role + "_evidence") for role in ("calibration", "holdout")),
            "native evidence receipts missing",
        )
        requested += len(points)
    require(seen == KEYS, "all sixteen phase cells are required")
    require(
        report["coverage"]
        == {
            "required_configurations": 8,
            "required_phase_cells": 16,
            "passed_phase_cells": 16,
            "requested_points": requested,
            "compared_points": requested,
        },
        "coverage summary mismatch",
    )


def validate_external_receipts(records, *, stage_root=None, evidence_root=None):
    require(
        stage_root is not None and evidence_root is not None, "bound raw evidence requires stage and evidence roots"
    )
    if __package__:
        from . import raw_campaign
    else:
        import raw_campaign
    return raw_campaign.validate(records, Path(stage_root), Path(evidence_root))


def validate_stage(root):
    stage = read(root / "stage.json")
    require(
        stage.get("schema") == "glm53flash_fpm_publication_stage_v1"
        and stage.get("status") == "STAGED_NOT_PUBLISHED"
        and stage.get("repo_id") == "nvidia/aisimulate-fpm-dataset",
        "invalid publication stage",
    )
    acceptance = read(checked(root, stage["acceptance"]))
    validate_acceptance(acceptance)
    require(isinstance(stage.get("input_manifest"), dict), "original acceptance manifest receipt missing")
    original_manifest = checked(root, stage["input_manifest"])
    canonical_manifest_sha = hashlib.sha256(
        json.dumps(
            read(original_manifest), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()
    require(
        stage["input_manifest"].get("sha256") == stage.get("input_manifest_sha256")
        and acceptance.get("input_manifest_sha256") == canonical_manifest_sha,
        "acceptance report is not bound to the exact original input manifest",
    )
    accepted_cells = {
        tuple(cell[k] for k in ("backend", "weight_quantization", "tp", "phase")): cell for cell in acceptance["cells"]
    }
    sources = {}
    for receipt in stage["sources"]:
        require(receipt["sha256"] not in sources, "duplicate source receipt")
        sources[receipt["sha256"]] = checked(root, receipt)
    cells, rows_by_source = set(), {}
    for part in stage["configurations"]:
        key = (part["backend"], part["weight_quantization"], part["tp"])
        require(
            key in {k[:3] for k in KEYS} and key not in cells,
            "invalid partition matrix",
        )
        cells.add(key)
        require(
            MODELS.get(part["model_id"]) == part["model_revision"]
            and part["model_id"] == ("nvidia/GLM-5.3-Flash-NVFP4" if key[1] == "nvfp4" else "zai-org/GLM-5.3-Flash"),
            "wrong model revision",
        )
        require(
            all(
                accepted_cells[(*key, phase)][role + "_evidence"].get("backend_version") == part["backend_version"]
                for phase in ("prefill", "decode")
                for role in ("calibration", "holdout")
            ),
            "accepted native runtime differs from partition runtime",
        )
        table = pq.read_table(checked(root, part["parquet"]))
        meta = read(checked(root, part["metadata"]))
        lineage = part["source_partition"]
        indices = lineage["row_indices"]
        require(
            indices and all(type(i) is int and i >= 0 for i in indices) and len(indices) == len(set(indices)),
            "duplicate or invalid source row indices",
        )
        original = pq.read_table(sources[lineage["parquet_sha256"]])
        original_meta = read(sources[lineage["metadata_sha256"]])
        require(
            original.num_rows == lineage["original_row_count"]
            and max(indices) < original.num_rows
            and table.equals(original.take(pa.array(indices, type=pa.int64())), check_metadata=True),
            "partition changed original Arrow rows or schema",
        )
        require(
            original_meta.get("parquet_sha256") == lineage["parquet_sha256"]
            and original_meta.get("row_count") == original.num_rows
            and meta.get("source_partition") == lineage
            and meta.get("parquet_sha256") == part["parquet"]["sha256"]
            and meta.get("row_count") == part["rows"] == table.num_rows
            and meta.get("schema_name") == "aic_fpm_forward_perf"
            and meta.get("schema_version") == 7,
            "partition metadata mismatch",
        )
        rows_by_source.setdefault(lineage["parquet_sha256"], []).extend(indices)
        first = table.to_pylist()[0]
        identity = {field: first[field] for field in IDENTITY}
        quant = "nvfp4" if key[1] == "nvfp4" else "fp8_block"
        require(
            identity["model_path"] == part["model_id"]
            and identity["system"] == "gb300"
            and identity["backend"] == key[0]
            and identity["backend_version"] == part["backend_version"]
            and identity["tp"] == identity["moe_tp"] == key[2]
            and all(identity[k] == 1 for k in ("pp", "dp", "cp", "moe_ep"))
            and identity["weight_quantization"] == quant
            and identity["kv_cache_dtype"] == "fp8",
            "partition execution identity mismatch",
        )
        rows = table.to_pylist()
        require(
            all(
                all(row[field] == value for field, value in identity.items())
                and row["input_tokenizer_revision"] == part["model_revision"]
                and type(row["latency_ms"]) in (int, float)
                and math.isfinite(row["latency_ms"])
                and row["latency_ms"] > 0
                for row in rows
            )
            and {row["workload_kind"] for row in rows} == {"prefill", "decode"},
            "mixed or invalid calibration rows",
        )
    require(cells == {k[:3] for k in KEYS}, "all eight partitions are required")
    for digest, indices in rows_by_source.items():
        require(
            sorted(indices) == list(range(pq.read_table(sources[digest]).num_rows)),
            "source row union omits or duplicates calibration rows",
        )
    return stage


def validate_snapshot(root, manifest):
    require(manifest["provenance"]["source_campaign_id"] == CAMPAIGN, "wrong GLM campaign")
    policy_path = root / manifest["provenance"]["import_receipt"]
    require(policy_path.resolve().is_relative_to(root.resolve()), "escaping import receipt")
    require(
        sha(policy_path) == manifest["provenance"]["import_receipt_sha256"],
        "import receipt changed",
    )
    policy = read(policy_path)
    require(
        policy["policy"] == POLICY and re.fullmatch(r"[0-9a-f]{40}", policy["source_revision"]),
        "unreviewed GLM import policy or source revision",
    )
    stage_path = checked(root, policy["stage"])
    stage = validate_stage(stage_path.parent)
    external_path = checked(root, policy["external_raw_evidence"])
    validate_external_receipts(read(external_path), stage_root=stage_path.parent, evidence_root=external_path.parent)
    parts = [
        p
        for p in stage["configurations"]
        if (p["model_id"], p["backend"], p["backend_version"], p["tp"])
        == tuple(manifest[k] for k in ("model_id", "framework", "framework_version", "tp"))
    ]
    require(
        len(parts) == 1 and len(manifest["fpm"]) == 1,
        "manifest does not select one accepted partition",
    )
    part = parts[0]
    entry = manifest["fpm"][0]
    require(
        entry["sha256"] == part["parquet"]["sha256"]
        and entry["row_count"] == part["rows"]
        and manifest["model_revision"] == part["model_revision"],
        "canonical partition changed",
    )
    meta = read(root / entry["metadata_path"])
    original = read(stage_path.parent / part["metadata"]["path"])
    first = pq.read_table(stage_path.parent / part["parquet"]["path"]).to_pylist()[0]
    require(
        all(
            manifest[field] == first[field]
            for field in (
                "system",
                "tp",
                "pp",
                "dp",
                "moe_tp",
                "moe_ep",
                "cp",
                "parallel_strategy",
                "weight_quantization",
                "kv_cache_dtype",
            )
        )
        and manifest["parallelism"] == f"pure-tp{first['tp']}"
        and manifest["provenance"]["source_revision"] == policy["source_revision"],
        "manifest differs from accepted execution identity",
    )
    require(
        {k: v for k, v in meta.items() if k not in ("import_policy", "supporting_files")} == original
        and meta["import_policy"] == POLICY,
        "canonical metadata discarded original fields",
    )
    paths = set()
    for receipt in meta["supporting_files"]:
        checked(root, receipt)
        paths.add(receipt["path"])
    require(
        manifest["provenance"]["import_receipt"] in paths,
        "import provenance not cataloged",
    )
    return paths


def validate_catalog_record(root, record):
    manifest = read(root / record["configuration_path"] / "manifest.json")
    validate_snapshot(root, manifest)
    require(
        record["schema_name"] == "aic_fpm_forward_perf" and record["schema_version"] == 7,
        "GLM requires schema-v7 FPM",
    )
    for key in (
        "model_id",
        "model_revision",
        "system",
        "framework",
        "framework_version",
        "tp",
        "pp",
        "dp",
        "moe_tp",
        "moe_ep",
        "cp",
        "weight_quantization",
        "kv_cache_dtype",
        "snapshot_id",
    ):
        require(record[key] == manifest[key], "catalog identity mismatch: " + key)
    entry = manifest["fpm"][0]
    require(
        all(record[k] == entry[k] for k in ("path", "metadata_path", "sha256", "row_count")),
        "catalog calibration differs from accepted manifest",
    )
    policy = read(root / manifest["provenance"]["import_receipt"])
    part = next(
        p for p in read(root / policy["stage"]["path"])["configurations"] if p["parquet"]["sha256"] == record["sha256"]
    )
    lineage = part["source_partition"]
    require(
        record["source_sha256"] == lineage["parquet_sha256"]
        and record["source_row_count"] == lineage["original_row_count"]
        and record["source_revision"] == policy["source_revision"],
        "catalog source lineage mismatch",
    )
    return 7
