# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare a NEW local canonical dataset copy; this command never uploads.

The existing dataset validator receives four explicit GLM policy routing edits.
Existing catalog/manifests are preserved and the existing public write_catalogs
and validate_dataset APIs perform the final canonical validation.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

if __package__:
    from . import glm53flash as policy
else:
    import glm53flash as policy


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


MANAGER_SHA256 = "9686df97a10496e06e346ee2749ba6e823c24dff18f69cf6f9be72b824d11b52"


def _name(node, value):
    return isinstance(node, ast.Name) and node.id == value


def _campaign_comparison(node, operation, symbol):
    return (
        isinstance(node, ast.Compare)
        and _name(node.left, "campaign_id")
        and len(node.ops) == len(node.comparators) == 1
        and isinstance(node.ops[0], operation)
        and _name(node.comparators[0], symbol)
    )


def _glm_comparison(operation=ast.Eq):
    return ast.Compare(
        left=ast.Name(id="campaign_id", ctx=ast.Load()),
        ops=[operation()],
        comparators=[ast.Name(id="GLM53FLASH_CAMPAIGN", ctx=ast.Load())],
    )


def _call(name, *arguments):
    return ast.Call(
        func=ast.Name(id=name, ctx=ast.Load()),
        args=[ast.Name(id=value, ctx=ast.Load()) for value in arguments],
        keywords=[],
    )


def _render(node):
    return ast.unparse(ast.fix_missing_locations(node))


def route_policy(path):
    """Transform only AST-selected hooks in the pinned external validator copy.

    Source text is supplied by the external snapshot at runtime. No upstream
    source fragments are embedded in this implementation. Span edits preserve
    all unrelated original bytes, including copyright and license comments.
    """
    policy.require(policy.sha(path) == MANAGER_SHA256, "dataset validator API drift; inspect before adapting")
    source = path.read_text()
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_dataset"]
    policy.require(len(functions) == 1, "dataset validator function is ambiguous")
    validator = functions[0]
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def offset(line, byte_column):
        # CPython AST columns are UTF-8 byte offsets; source slicing uses Unicode.
        return offsets[line - 1] + len(lines[line - 1].encode()[:byte_column].decode())

    edits = []

    def replace(node, replacement):
        edits.append(
            (offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset), _render(replacement))
        )

    # Add our imports after the existing import block, keeping future imports first.
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    policy.require(bool(imports), "dataset validator imports missing")
    import_node = ast.ImportFrom(
        module="glm53flash",
        level=0,
        names=[
            ast.alias(name="CAMPAIGN", asname="GLM53FLASH_CAMPAIGN"),
            ast.alias(name="validate_snapshot", asname="validate_glm53flash_snapshot"),
            ast.alias(name="validate_catalog_record", asname="validate_glm53flash_catalog_record"),
        ],
    )
    insertion = offsets[max(node.end_lineno for node in imports)]
    edits.append((insertion, insertion, _render(import_node) + "\n"))

    snapshots = [
        node
        for node in ast.walk(validator)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and _name(node.targets[0], "fpm_entries")
    ]
    policy.require(len(snapshots) == 1, "dataset snapshot hook is ambiguous")
    hook = snapshots[0]
    snapshot_call = ast.Expr(
        value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="external_fpm_provenance_paths", ctx=ast.Load()), attr="update", ctx=ast.Load()
            ),
            args=[_call("validate_glm53flash_snapshot", "root", "manifest")],
            keywords=[],
        )
    )
    block = ast.If(test=_glm_comparison(), body=[snapshot_call], orelse=[])
    text = "".join(" " * hook.col_offset + line + "\n" for line in _render(block).splitlines())
    edits.append((offsets[hook.lineno - 1], offsets[hook.lineno - 1], text))

    guards = [
        node
        for node in ast.walk(validator)
        if isinstance(node, ast.If)
        and any(_campaign_comparison(test, ast.NotIn, "DSV41_CAMPAIGNS") for test in ast.walk(node.test))
        and any(_name(test, "external_campaign") for test in ast.walk(node.test))
    ]
    policy.require(len(guards) == 2, "dataset external-campaign guards changed")
    for guard in guards:
        replace(guard.test, ast.BoolOp(op=ast.And(), values=[guard.test, _glm_comparison(ast.NotEq)]))

    catalog_hooks = [
        node
        for node in ast.walk(validator)
        if isinstance(node, ast.If)
        and _campaign_comparison(node.test, ast.Eq, "SELF_BENCHMARK_CAMPAIGN")
        and len(node.body) == 1
        and isinstance(node.body[0], ast.Assign)
        and len(node.body[0].targets) == 1
        and _name(node.body[0].targets[0], "expected_schema_version")
    ]
    policy.require(len(catalog_hooks) == 1, "dataset catalog hook is ambiguous")
    catalog = catalog_hooks[0]
    replace(catalog.test, ast.BoolOp(op=ast.Or(), values=[_glm_comparison(), catalog.test]))
    replace(
        catalog.body[0].value,
        ast.IfExp(
            test=_glm_comparison(),
            body=_call("validate_glm53flash_catalog_record", "root", "record"),
            orelse=catalog.body[0].value,
        ),
    )

    # Apply backwards so original AST spans remain valid; reject overlap.
    previous = len(source) + 1
    for start, end, replacement in sorted(edits, reverse=True):
        policy.require(end <= previous, "dataset routing edits overlap")
        source = source[:start] + replacement + source[end:]
        previous = start
    ast.parse(source)
    path.write_text(source)


def load_manager(root):
    old_paths = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root / "scripts"))
        spec = importlib.util.spec_from_file_location("glm_dataset_manager", root / "scripts/manage_dataset.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_paths
        sys.dont_write_bytecode = old_bytecode


def copy_policy(destination):
    """Copy the complete standalone validator closure into a dataset snapshot."""
    for module in policy.POLICY_MODULES:
        shutil.copyfile(Path(__file__).with_name(module), destination / "scripts" / module)


def prepare(base, stage_root, destination, external_receipts, source_revision, evidence_date):
    base, stage_root, destination = (Path(p).resolve() for p in (base, stage_root, destination))
    policy.require(not destination.exists(), "destination must not exist")
    policy.require(
        not destination.is_relative_to(base) and not destination.is_relative_to(stage_root),
        "destination cannot be inside an immutable input",
    )
    policy.require(
        re.fullmatch(r"[0-9a-f]{40}", source_revision),
        "source revision must be immutable Git SHA",
    )
    policy.require(
        re.fullmatch(r"\d{4}-\d{2}-\d{2}", evidence_date),
        "evidence date must be explicit",
    )
    stage = policy.validate_stage(stage_root)
    acceptance = policy.read(policy.checked(stage_root, stage["acceptance"]))
    external = policy.read(external_receipts)
    external_files = policy.validate_external_receipts(
        external, stage_root=stage_root, evidence_root=Path(external_receipts).resolve().parent
    )
    stage_sha = policy.sha(stage_root / "stage.json")
    policy.require(
        policy.sha(base / "scripts/manage_dataset.py") == MANAGER_SHA256,
        "dataset validator API drift; inspect before adapting",
    )
    index = policy.read(base / "catalog/index.json")
    # No state is changed until both original input sets pass their validators.
    manager = load_manager(base)
    manager.validate_dataset(base, write_report=False)
    shutil.copytree(
        base,
        destination,
        ignore=shutil.ignore_patterns(".cache", "__pycache__", ".git"),
    )
    copy_policy(destination)
    route_policy(destination / "scripts/manage_dataset.py")
    manager = load_manager(destination)
    campaign = Path("campaigns") / policy.CAMPAIGN / stage_sha
    shutil.copytree(stage_root, destination / campaign / "stage")
    external_path = campaign / "external-raw-evidence.json"
    # Keep exact external-receipt source bytes, including order and whitespace.
    shutil.copyfile(external_receipts, destination / external_path)
    for rel, sha256 in external_files.items():
        source = policy.checked(Path(external_receipts).resolve().parent, {"path": rel, "sha256": sha256})
        target = destination / campaign / rel
        policy.require(not target.exists(), "external receipt collides with existing campaign evidence")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    fpm_records = policy.read(base / "catalog/fpm.json")["records"]
    measurement_records = policy.read(base / "catalog/measurements.json")["records"]
    manifests = list(index["configuration_manifests"])
    added = []
    for part in stage["configurations"]:
        source_table = stage_root / part["parquet"]["path"]
        original_meta = policy.read(stage_root / part["metadata"]["path"])
        row = pq.read_table(source_table).to_pylist()[0]
        producer = original_meta.get("producer_revision")
        producer_commit = producer if isinstance(producer, str) and re.fullmatch(r"[0-9a-f]{40}", producer) else None
        revisions = policy.publication_revisions(
            original_meta, acceptance, external, Path(external_receipts).resolve().parent, source_revision, part
        )
        if revisions is not None:
            producer_commit = revisions["native_producer_revision"]
        identity = {
            "model_id": part["model_id"],
            "model_revision": part["model_revision"],
            "system": "gb300",
            "gpu_family": "GB300",
            "framework": part["backend"],
            "framework_version": part["backend_version"],
            "parallelism": f"pure-tp{part['tp']}",
            **{
                key: row[key]
                for key in (
                    "parallel_strategy",
                    "tp",
                    "pp",
                    "dp",
                    "moe_tp",
                    "moe_ep",
                    "cp",
                    "weight_quantization",
                    "kv_cache_dtype",
                )
            },
            "snapshot_id": "accepted-" + stage_sha[:16],
            "snapshot_status": "current",
            "evidence_date": evidence_date,
            "aisim_commit": producer_commit,
            "aisim_commit_status": "recorded" if producer_commit else "unknown",
        }
        if revisions is not None:
            identity["aisim_commit_semantics"] = "native_producer_revision"
        identity["configuration_path"] = manager.configuration_path(identity)
        leaf = Path(identity["configuration_path"])
        policy.require(
            not (destination / leaf).exists(),
            "existing GLM leaf needs explicit history migration",
        )
        target = leaf / "fpm/fpm_forward_perf.parquet"
        metadata_path = target.with_suffix(".metadata.json")
        (destination / target).parent.mkdir(parents=True)
        shutil.copyfile(source_table, destination / target)
        import_path = leaf / "fpm/provenance/import.json"
        receipt = {
            "policy": policy.POLICY,
            "source_revision": source_revision,
            "stage": {"path": str(campaign / "stage/stage.json"), "sha256": stage_sha},
            "external_raw_evidence": {
                "path": str(external_path),
                "sha256": policy.sha(external_receipts),
            },
            "status": "CANONICAL_LOCAL_NOT_PUBLISHED",
        }
        if revisions is not None:
            receipt["revision_identity"] = revisions
        write(destination / import_path, receipt)
        meta = dict(
            original_meta,
            import_policy=policy.POLICY,
            supporting_files=[
                {
                    "path": str(import_path),
                    "sha256": policy.sha(destination / import_path),
                }
            ],
        )
        write(destination / metadata_path, meta)
        provenance = {
            "source_campaign_id": policy.CAMPAIGN,
            "source_repository": "https://github.com/ai-dynamo/aisimulate",
            "source_revision": source_revision,
            "source_campaign_path": str(campaign),
            "producer_revisions": [producer] if producer else [],
            "producer_identity_missing": not bool(producer),
            "import_receipt": str(import_path),
            "import_receipt_sha256": policy.sha(destination / import_path),
        }
        if revisions is not None:
            provenance.update(revision_identity=revisions, producer_revisions_semantics=policy.PLANNER_ALIAS)
        artifact = "fpm-" + part["parquet"]["sha256"][:16]
        entry = {
            "artifact_id": artifact,
            "path": str(target),
            "metadata_path": str(metadata_path),
            "sha256": part["parquet"]["sha256"],
            "row_count": part["rows"],
            "role": "primary",
            "phases": ["decode", "prefill"],
        }
        measurement_path = leaf / "measurements/manifest.json"
        measurement_id = "measurements-" + policy.sha(destination / import_path)[:16]
        write(
            destination / measurement_path,
            dict(
                identity,
                manifest_version=4,
                provenance=provenance,
                measurement_artifact_id=measurement_id,
                measurement_protocol_id=None,
                files=[],
            ),
        )
        manifest_path = leaf / "manifest.json"
        manifest = dict(
            identity,
            manifest_version=3,
            provenance=provenance,
            fpm=[entry],
            measurements={
                "artifact_id": measurement_id,
                "manifest_path": str(measurement_path),
                "manifest_sha256": policy.sha(destination / measurement_path),
                "protocol_id": None,
                "file_count": 0,
            },
        )
        write(destination / manifest_path, manifest)
        manifests.append(str(manifest_path))
        lineage = part["source_partition"]
        source_path = next(r["path"] for r in stage["sources"] if r["sha256"] == lineage["parquet_sha256"])
        fpm_records.append(
            dict(
                identity,
                fpm_artifact_id=artifact,
                path=str(target),
                metadata_path=str(metadata_path),
                sha256=entry["sha256"],
                row_count=part["rows"],
                bytes=(destination / target).stat().st_size,
                schema_name="aic_fpm_forward_perf",
                schema_version=7,
                schema_fingerprint=manager.schema_fingerprint(destination / target),
                source_campaign_id=policy.CAMPAIGN,
                source_revision=source_revision,
                source_kind="accepted_native_calibration",
                source_path=str(campaign / "stage" / source_path),
                source_sha256=lineage["parquet_sha256"],
                source_row_count=lineage["original_row_count"],
                source_partition_id=entry["sha256"],
                base_occurrence=True,
                role="primary",
                status_reason=None,
                timing_scope="native_forward_pass",
                phases=["decode", "prefill"],
            )
        )
        added.append(str(manifest_path))
    manager.write_catalogs(
        destination,
        fpm_records,
        measurement_records,
        sorted(manifests),
        index["history_manifests"],
        index["source"]["commit_time"],
    )
    counts = manager.validate_dataset(destination, write_report=False)
    result = {
        "status": "CANONICAL_LOCAL_NOT_PUBLISHED",
        "policy": policy.POLICY,
        "stage_sha256": stage_sha,
        "base_index_sha256": policy.sha(base / "catalog/index.json"),
        "added_manifests": added,
        "counts": counts,
        "remaining": [
            "review exact upload diff against current Hub parent revision",
            "Hub commit",
            "immutable consumer pin",
            "installed offline prediction acceptance",
        ],
    }
    write(destination / campaign / "import-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "stage", "destination", "external-receipts"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--evidence-date", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.base,
                args.stage,
                args.destination,
                args.external_receipts,
                args.source_revision,
                args.evidence_date,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
