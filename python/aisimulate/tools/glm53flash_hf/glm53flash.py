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

if __package__:
    from . import (
        external_control,
        external_control_current,
        external_control_sglang_mixed,
        portable_history,
        raw_campaign,
    )
else:
    # The embedding dataset loader temporarily adds scripts/ only while loading
    # this module. Resolve the complete portable closure before it restores paths.
    import external_control
    import external_control_current
    import external_control_sglang_mixed  # noqa: F401
    import portable_history
    import raw_campaign

CAMPAIGN = "glm53flash-pr324"
POLICY = "glm53flash-accepted-arrow-partitions-v1"
HISTORY_POLICY = "glm53flash-accepted-arrow-partitions-history-v2"
REVISION_SCHEMA = "glm53flash_revision_identity_v1"
PLANNER_ALIAS = "legacy_planner_revision_alias"
POLICY_MODULES = (
    "glm53flash.py",
    "raw_campaign.py",
    "raw_archive.py",
    "native_roots.py",
    "external_control.py",
    "external_control_vllm.py",
    "external_control_current.py",
    "external_control_sglang_mixed.py",
    "closed_history.py",
    "cleanup_reconciliation.py",
    "cleanup_executor.py",
    "accounting_termination.py",
    "portable_history.py",
    "import_glm53flash.py",
    "profile.py",
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
    return raw_campaign.validate(records, Path(stage_root), Path(evidence_root))


def requires_attempt_history(records, evidence_root):
    """Current external controls cannot be downgraded to the unchanged legacy policy."""
    return any(
        read(checked(Path(evidence_root), record["external_control"])).get("schema") in external_control_current.SCHEMAS
        for record in records
        if "external_control" in record
    )


def validate_attempt_history(import_policy, provenance, records, evidence_root, stage_sha256):
    """Mandatory portable evidence check; this function never requests raw tar."""
    required = requires_attempt_history(records, evidence_root)
    selected = import_policy["policy"]
    require(not required or selected == HISTORY_POLICY, "current campaign requires immutable attempt history policy")
    if selected != HISTORY_POLICY:
        require(
            "external_raw_history" not in import_policy and "external_raw_history" not in provenance,
            "legacy policy cannot silently ignore attempt history",
        )
        return {}
    entry = import_policy.get("external_raw_history")
    require(isinstance(entry, dict) and provenance.get("external_raw_history") == entry, "history provenance differs")
    verified = portable_history.verify_portable_history(
        Path(evidence_root), entry, records, expected_stage_sha256=stage_sha256, archive=raw_campaign.archive
    )
    return verified["files"]


def validate_planner_revision(meta, original):
    """Recognize legacy partitions; new partitions must preserve the original value."""
    fields = {"revision_identity_schema", "planner_revision", "producer_revision_semantics"}
    if not fields.intersection(meta):
        return False
    value = original.get("aic_revision")
    require(
        meta.get("revision_identity_schema") == REVISION_SCHEMA
        and isinstance(value, str)
        and bool(value.strip())
        and all(meta.get(k) == value for k in ("aic_revision", "planner_revision", "producer_revision"))
        and meta.get("producer_revision_semantics") == PLANNER_ALIAS,
        "partition planner revision is missing, mislabeled, or changed",
    )
    return True


def publication_revisions(meta, report, records, evidence_root, source_revision, part):
    """Re-derive explicit revision names from validated original controls.

    Callers first validate the complete stage and raw campaign. Legacy stages
    remain readable without invented native identities. New stages require all
    four phase/role controls, never a planner or analysis revision substitution.
    """
    key = tuple(part[k] for k in ("backend", "weight_quantization", "tp"))
    selected = [r for r in records if tuple(r[k] for k in ("backend", "weight_quantization", "tp")) == key]
    if not validate_planner_revision(meta, meta):
        require(
            all(
                "external_control" not in r
                or read(checked(Path(evidence_root), r["external_control"])).get("schema")
                not in external_control_current.SCHEMAS
                for r in selected
            ),
            "current external controls require explicit revision metadata",
        )
        return None
    require(
        len(selected) == 4
        and {(r["phase"], r["role"]) for r in selected}
        == {(p, r) for p in ("prefill", "decode") for r in ("calibration", "holdout")},
        "revision identity needs exactly four original phase/role records",
    )
    identities, controls, cached = [], [], {}
    for record in selected:
        ref = record.get("external_control")
        require(isinstance(ref, dict), "native producer revision requires original external controls")
        path = checked(Path(evidence_root), ref)
        cache_key = str(path), ref["sha256"]
        if cache_key not in cached:
            document = read(path)
            require(
                document.get("schema") in external_control_current.SCHEMAS,
                "new revision identity requires current controls",
            )
            _, get, admission = external_control.validate(path.parent, document)
            cached[cache_key] = external_control_current.configuration_revisions(
                document, get, admission, *key, meta["planner_revision"]
            )
        identities.append(cached[cache_key])
        controls.append(ref["sha256"])
    require(all(i == identities[0] for i in identities), "phase/role native producer or planner identities differ")
    consumer = report.get("consumer", {})
    require(
        consumer.get("distribution") == "aisimulate"
        and consumer.get("api") == "RustForwardPassPerfModel.best_available"
        and isinstance(consumer.get("version"), str)
        and bool(consumer["version"])
        and re.fullmatch(r"[0-9a-f]{64}", consumer.get("payload_sha256", ""))
        and re.fullmatch(r"[0-9a-f]{40}", source_revision),
        "analysis installed consumer or publication tool identity is missing",
    )
    return {
        "schema": REVISION_SCHEMA,
        "planner_revision": meta["planner_revision"],
        "producer_revision": meta["producer_revision"],
        "producer_revision_semantics": PLANNER_ALIAS,
        **identities[0],
        "analysis_revision": {
            "installed_consumer": consumer,
            "publication_tool_revision": source_revision,
        },
        "external_control_sha256": sorted(set(controls)),
    }


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
        validate_planner_revision(meta, original_meta)
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
        policy["policy"] in {POLICY, HISTORY_POLICY} and re.fullmatch(r"[0-9a-f]{40}", policy["source_revision"]),
        "unreviewed GLM import policy or source revision",
    )
    stage_path = checked(root, policy["stage"])
    stage = validate_stage(stage_path.parent)
    external_path = checked(root, policy["external_raw_evidence"])
    external = read(external_path)
    validate_external_receipts(external, stage_root=stage_path.parent, evidence_root=external_path.parent)
    history_files = validate_attempt_history(
        policy, manifest["provenance"], external, external_path.parent, sha(stage_path)
    )
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
    revisions = publication_revisions(
        original,
        read(checked(stage_path.parent, stage["acceptance"])),
        external,
        external_path.parent,
        policy["source_revision"],
        part,
    )
    require(
        policy.get("revision_identity") == manifest["provenance"].get("revision_identity") == revisions,
        "canonical revision identity differs from original controls or installed analysis",
    )
    if revisions is not None:
        require(
            manifest.get("aisim_commit") == revisions["native_producer_revision"]
            and manifest.get("aisim_commit_status") == "recorded"
            and manifest.get("aisim_commit_semantics") == "native_producer_revision"
            and manifest["provenance"].get("producer_revisions") == [original["producer_revision"]]
            and manifest["provenance"].get("producer_revisions_semantics") == PLANNER_ALIAS
            and manifest["provenance"].get("producer_identity_missing") is False,
            "canonical revision aliases are inconsistent",
        )
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
        and meta["import_policy"] == policy["policy"],
        "canonical metadata discarded original fields",
    )
    paths = {str((external_path.parent / name).relative_to(root)) for name in history_files}
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
    if manifest["provenance"].get("revision_identity") is not None:
        require(
            all(
                record.get(k) == manifest.get(k)
                for k in ("aisim_commit", "aisim_commit_status", "aisim_commit_semantics")
            ),
            "catalog native producer revision differs from manifest",
        )
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
