# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen Rubin pilot through AISimulate's existing collector executor.

The orchestration follows this repository's collector/collect.py. Kernel
execution, checkpoints, error classification, parquet publication, and the
collection_meta.yaml transaction remain owned by that shared implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import multiprocessing as mp
import os
import sys
from contextlib import chdir
from datetime import UTC, datetime
from pathlib import Path

from collector.sglang_rubin import _shared_helper, _worker_helper_imports
from collector.sglang_rubin.registry import (
    CHECKPOINT_METADATA_SHA256,
    MODEL_PATH,
    REGISTRY,
    SGLANG_COMMIT,
    SGLANG_DISTRIBUTION_VERSION,
    SGLANG_REPOSITORY,
    SM_VERSION,
)
from collector.sglang_rubin.runtime import (
    IMAGE_ARM64_DIGEST,
    IMAGE_REF,
    IMAGE_REPOSITORY,
    REQUIRED_SERVING_ENV,
    collect_inventory,
    declared_serving_configuration,
    validate_runtime,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_MODEL_CONFIG = _PACKAGE_ROOT / "src/aisimulate_core/model_configs/nvidia--GLM-5.2-NVFP4_config.json"
_SHAPE_ENV = tuple(
    f"AIC_DSA_{phase}_{dimension}"
    for phase in ("CONTEXT", "GENERATION")
    for dimension in ("PREFIX_LENS", "SEQ_LENS", "BATCH_SIZES")
)


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", choices=[MODEL_PATH], default=MODEL_PATH)
    parser.add_argument("--ops", nargs="+", choices=[entry.op for entry in REGISTRY])
    parser.add_argument("--plan-only", action="store_true", help="Print the CPU-only plan; does not qualify the pilot")
    parser.add_argument("--checkpoint-dir", type=Path, help="Local NVFP4 model checkpoint (metadata only is read)")
    parser.add_argument("--launcher-image", help="Digest-pinned launcher reference; not attested inside the container")
    parser.add_argument("--output-dir", type=Path, help="Fresh dataset directory, or matching directory with --resume")
    parser.add_argument("--resume", action="store_true", help="Reuse this dataset's standard collector checkpoints")
    parser.add_argument("--resume-retry-failed", action="store_true")
    parser.add_argument("--limit", type=_positive_int, help="Maximum outer cases per op")
    parser.add_argument("--smoke", action="store_true", help="Shuffle and sample one outer case per op by default")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle outer cases with the existing seed 42")
    parser.add_argument(
        "--case-filter",
        action="append",
        dest="gemm_case_filters",
        metavar="SUBSTR",
        help="Existing executor substring filter (repeatable OR); requires --ops gemm",
    )
    workers = parser.add_mutually_exclusive_group()
    workers.add_argument("--processes", type=_positive_int, default=1, help="GPU worker count (default: 1)")
    workers.add_argument("--sequential", action="store_true", help="Run on device 0 in the parent process")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    if args.ops and len(args.ops) != len(set(args.ops)):
        parser.error("duplicate --ops entries")
    if args.gemm_case_filters and args.ops != ["gemm"]:
        parser.error("--case-filter requires collecting --ops gemm alone")
    if args.gemm_case_filters and any(not value for value in args.gemm_case_filters):
        parser.error("--case-filter must not be empty")
    if args.resume_retry_failed and not args.resume:
        parser.error("--resume-retry-failed requires --resume")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    if not args.plan_only:
        if args.checkpoint_dir is None:
            parser.error("--checkpoint-dir is required for runtime collection")
        if args.launcher_image not in (IMAGE_REF, f"{IMAGE_REPOSITORY}@{IMAGE_ARM64_DIGEST}"):
            parser.error("--launcher-image must be the pinned runtime index or ARM64 manifest reference")
    args.limit = args.limit if args.limit is not None else (1 if args.smoke else None)
    args.shuffle = args.shuffle or args.smoke
    return args


def _case_filters(op: str, gemm_case_filters: list[str] | None = None) -> list[str] | None:
    # Use the executor's existing, visible runtime selection. These fragments
    # match the stock case tuple contracts, not a new generator or YAML rule.
    if op == "moe":
        return [f", 4, 1, '{MODEL_PATH}'"]
    if op.startswith("dsa_"):
        return ["'dsa', None, 4, None]"]
    if op == "gemm":
        return gemm_case_filters
    return None


def _build_plan(args):
    from collector.model_cases import build_collection_case_plan

    plan = build_collection_case_plan(backend="sglang", model_path=args.model_path, sm_version=SM_VERSION)
    registered = {entry.op for entry in REGISTRY}
    if set(plan.ops) != registered:
        raise RuntimeError(
            f"Pilot registry differs from the canonical model plan: {sorted(set(plan.ops) ^ registered)}"
        )
    ops = args.ops or plan.ops
    document = {
        **plan.to_log_dict(),
        "ops": ops,
        "qualification": "unqualified; planning does not run kernels or validate predictions",
        "declared_image": IMAGE_REF,
        "audited_sglang_commit": SGLANG_COMMIT,
        "serving_scope": {"tensor_parallel_size": 4, "expert_parallel_size": 1, "data_parallel_attention": False},
        "declared_serving_configuration": declared_serving_configuration(),
        "observed_serving_environment": {name: os.environ.get(name) for name in REQUIRED_SERVING_ENV},
        "runtime_case_filters": {entry.op: _case_filters(entry.op, args.gemm_case_filters) for entry in REGISTRY},
        "dsa_inner_shape_environment": {name: os.environ.get(name) for name in _SHAPE_ENV},
        "limit_per_op": args.limit,
        "shuffle": args.shuffle,
        "smoke": args.smoke,
    }
    return plan, document


def _checkpoint_errors(directory: Path, inventory: dict) -> list[str]:
    """Bind observed metadata and cached case shapes to the same frozen snapshot."""
    files = inventory.get("observed", {}).get("checkpoint", {}).get("files", {})
    errors = []
    for name, expected_hash in CHECKPOINT_METADATA_SHA256.items():
        cached_path = _MODEL_CONFIG.with_name(f"nvidia--GLM-5.2-NVFP4_{name}")
        for label, path in (("Checkpoint", directory / name), ("Cached model", cached_path)):
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as error:
                errors.append(f"Cannot read {label.lower()} {name}: {error}")
                continue
            if digest != expected_hash:
                errors.append(f"{label} {name} SHA-256 differs from the Hecate NVFP4 snapshot")
        if files.get(name, {}).get("sha256") != expected_hash:
            errors.append(f"Observed checkpoint {name} SHA-256 differs from the Hecate NVFP4 snapshot")
    return errors


def _load_executor():
    """Share one helper module with legacy `from helper` imports and workers."""
    _shared_helper()
    collector_dir = str(_PACKAGE_ROOT / "collector")
    if collector_dir not in sys.path:
        sys.path.insert(0, collector_dir)
    return importlib.import_module("collector.collect")


def _runtime_context(inventory: dict, ops: list[str]):
    from collector.framework_manifest import CollectorRuntime
    from collector.version_resolver import build_collections

    version = inventory["observed"]["package_versions"]["sglang"]["version"]
    runtime = CollectorRuntime(
        framework="sglang",
        version=version,
        images={"default": IMAGE_REF},
        source_commit=SGLANG_COMMIT,
        source_repo=SGLANG_REPOSITORY,
        collector_dir="collector/sglang_rubin",
        data_backend="sglang",
    )
    return {
        "framework": "sglang",
        "installed_version": version,
        "runtime": runtime,
        "sm_version": SM_VERSION,
        "collections": build_collections(REGISTRY, "sglang", version, ops),
    }


def _dataset_identity(inventory: dict, plan: dict) -> dict:
    from collector.provenance import collector_hash, load_closures

    closures = load_closures(_PACKAGE_ROOT / "collector/hash_closures.yaml")
    observed = inventory["observed"]
    return {
        "schema": "sglang-rubin-pilot-v1",
        "model_path": MODEL_PATH,
        "sm_version": SM_VERSION,
        "declared_image": IMAGE_REF,
        "audited_sglang_commit": SGLANG_COMMIT,
        "declared_serving_configuration": plan["declared_serving_configuration"],
        "observed_serving_environment": observed["serving_environment"],
        "package_versions": observed["package_versions"],
        "reported_build_environment": observed["reported_build_environment"],
        "torch_cuda_version": observed["cuda"]["torch_cuda_version"],
        "checkpoint_files": observed["checkpoint"]["files"],
        "collector_hashes": {
            module: collector_hash(module, _PACKAGE_ROOT, closures)
            for module in sorted({entry.module for entry in REGISTRY})
        },
        "dsa_inner_shape_environment": plan["dsa_inner_shape_environment"],
        "smoke": plan["smoke"],
        "runtime_case_filters": plan["runtime_case_filters"],
    }


def _write_new_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")


def _prepare_output(output: Path, *, resume: bool) -> None:
    if output.is_symlink():
        raise RuntimeError("Pilot output directory must not be a symlink")
    if resume:
        identity_path = output / "pilot_identity.json"
        if not identity_path.is_file() or identity_path.is_symlink():
            raise RuntimeError("--resume requires an existing pilot_identity.json in --output-dir")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory is not empty; use a fresh directory or --resume")
    output.mkdir(parents=True, exist_ok=True)


def _execute(args, plan, document, inventory, output: Path) -> int:
    import pyarrow.parquet as pq

    executor = _load_executor()
    context = _runtime_context(inventory, document["ops"])
    identity = _dataset_identity(inventory, document)
    identity_path = output / "pilot_identity.json"
    if args.resume:
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Pilot dataset identity changed; use a fresh output directory")
    else:
        _write_new_json(identity_path, identity)

    checkpoint_root = str(output / ".collector_checkpoint")
    resume_options = {
        "resume": args.resume,
        "retry_failed": args.resume_retry_failed,
        "checkpoint_dir": checkpoint_root,
    }
    processes = 0 if args.sequential else args.processes
    if processes > len(inventory["observed"]["cuda"]["devices"]):
        raise RuntimeError("--processes exceeds the observed visible CUDA device count")
    if processes:
        method = mp.get_start_method(allow_none=True)
        if method not in (None, "spawn"):
            raise RuntimeError("GPU collection requires a fresh Python process using multiprocessing spawn")
        if method is None:
            mp.set_start_method("spawn")

    with _worker_helper_imports(), chdir(output):
        os.environ["COLLECTOR_MODEL_PATH"] = MODEL_PATH
        executor.logger = executor.setup_logging(scope=["sglang_rubin"], debug=args.debug)
        executor._recover_collector_provenance_transaction(output, backend="sglang", checkpoint_dir=checkpoint_root)
        executor._preflight_collector_provenance(output, context)
        existing = {path.resolve(): path.stat().st_mtime_ns for path in executor.find_perf_csv_outputs(output)}
        errors = []
        for collection in context["collections"]:
            op = collection["type"]
            case_filters = document["runtime_case_filters"][op]
            executor.logger.info("Pilot scope for %s: %s", op, case_filters)
            op_errors = executor.collect_ops(
                processes,
                [collection],
                context["installed_version"],
                limit=args.limit,
                shuffle=args.shuffle,
                backend="sglang",
                resume_options=resume_options,
                model_path=MODEL_PATH,
                case_plan=plan,
                sm_version=SM_VERSION,
                case_filters=case_filters,
            )
            errors.extend(op_errors)
            if not op_errors:
                tracker = executor._resume_tracker_for_collection(
                    collection, context, backend="sglang", checkpoint_dir=checkpoint_root, sm_version=SM_VERSION
                )
                tracker.load_existing()
                if not tracker._done and not tracker._failed:
                    errors.append(
                        {
                            "module": f"sglang.{op}",
                            "error_type": "EmptyCollection",
                            "error_message": "No cases executed in the requested pilot scope",
                        }
                    )
        executor.generate_collection_summary(errors, "sglang", context["installed_version"])
        touched = {
            path
            for path in executor.find_perf_csv_outputs(output)
            if path.resolve() not in existing or path.stat().st_mtime_ns != existing[path.resolve()]
        }
        if args.resume:
            touched.update(
                executor._pending_resume_perf_outputs(
                    output, context, backend="sglang", checkpoint_dir=checkpoint_root, sm_version=SM_VERSION
                )
            )
        if touched:
            executor._finalize_collector_outputs_transaction(
                output,
                sorted(touched),
                context,
                errors,
                backend="sglang",
                checkpoint_dir=checkpoint_root,
                sm_version=SM_VERSION,
            )
        metadata, _snapshot = executor._load_clean_collector_sidecar(output)
        tables = (metadata or {}).get("tables", {})
        table_op_names = {}
        for collection in context["collections"]:
            table = Path(collection["perf_filename"]).stem
            parquet_path = output / f"{table}.parquet"
            error_message = None
            if table not in tables or not parquet_path.is_file():
                error_message = f"No finalized parquet and provenance for {table}"
            elif collection["type"].startswith("dsa_"):
                if table not in table_op_names:
                    with pq.ParquetFile(parquet_path) as parquet:
                        table_op_names[table] = (
                            set(parquet.read(columns=["op_name"])["op_name"].to_pylist())
                            if "op_name" in parquet.schema_arrow.names
                            else set()
                        )
                if collection["type"] not in table_op_names[table]:
                    error_message = f"No {collection['type']} rows in finalized parquet for {table}"
            if error_message is not None:
                errors.append(
                    {
                        "module": f"sglang.{collection['type']}",
                        "error_type": "MissingPerfOutput",
                        "error_message": error_message,
                    }
                )
        executor.generate_collection_summary(errors, "sglang", context["installed_version"])
        return int(bool(errors))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan, document = _build_plan(args)
    if args.plan_only:
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    invocation = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output_dir or Path(f"sglang-rubin-{invocation}")).absolute()
    try:
        _prepare_output(output, resume=args.resume)
        inventory = collect_inventory(
            checkpoint_dir=args.checkpoint_dir, launcher_image=args.launcher_image, check_imports=True
        )
        errors = validate_runtime(inventory) + _checkpoint_errors(args.checkpoint_dir, inventory)
        installed = inventory.get("observed", {}).get("package_versions", {}).get("sglang", {}).get("version")
        if installed != SGLANG_DISTRIBUTION_VERSION:
            errors.append(f"Expected SGLang distribution {SGLANG_DISTRIBUTION_VERSION!r}, observed {installed!r}")
        inventory["validation"] = {"requested": True, "errors": errors}
        _write_new_json(output / f"inventory-{invocation}.json", inventory)
        _write_new_json(output / f"plan-{invocation}.json", document)
        if errors:
            raise RuntimeError("Pilot preflight failed: " + "; ".join(errors))
        status = _execute(args, plan, document, inventory, output)
        print(f"Pilot collector artifacts: {output}")
        return status
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        print(f"Rubin pilot collection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
