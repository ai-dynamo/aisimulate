# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent GLM holdout acceptance from frozen native receipts and public SDK.

No supplied latency scores are accepted. Native readers reconstruct measured
latencies; the installed Rust consumer predicts each frozen point without tuning.
See README.glm53flash-validation.md for the campaign manifest contract.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS
from aisimulate_core.sdk.glm53flash import MODEL_REVISIONS
from collector.glm53flash_jsonl import file_sha256, iter_records
from collector.glm53flash_protocol import PROTOCOL, TIMING_BOUNDARIES

from .native_artifact import _expected_scheduled, validate_native_collection

SCHEMA = "glm53flash_independent_holdout_v1"
PHASES = ("prefill", "decode")
GROUPS = ("1K-32K", "64K", "128K")
REQUIRED = tuple(
    (backend, quant, tp, phase)
    for backend in ("vllm", "sglang")
    for quant in ("fp8", "nvfp4")
    for tp in (2, 4)
    for phase in PHASES
)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _sha(value) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256 identity")
    return value


def _read_json_receipt(receipt: dict, base: Path):
    path = (base / receipt["path"]).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != _sha(receipt["sha256"]):
        raise ValueError(f"receipt digest mismatch: {path.name}")
    return json.loads(raw)


def _geometry(point: dict) -> tuple:
    _expected_scheduled(point)
    batch = point["batch_size"]
    query, prefix = point["total_prefill_tokens"], point["total_kv_read_tokens"]
    if query % batch or prefix % batch or batch > 32 or _context_length(point) > 131072:
        raise ValueError("holdout point must be homogeneous and within batch32/context128K")
    return point["point_type"], batch, query, prefix


def _context_length(point: dict) -> int:
    # Decode keeps the database's total_prefill_tokens=0 feature, but each
    # request computes one current token in addition to its existing KV.
    scheduled = point["batch_size"] if point["point_type"] == "decode" else point["total_prefill_tokens"]
    return (scheduled + point["total_kv_read_tokens"]) // point["batch_size"]


def _group(point: dict) -> str:
    context = _context_length(point)
    if context < 1024:
        return "below1K"
    if context <= 32768:
        return "1K-32K"
    return "64K" if context <= 65536 else "128K"


def _plan_run(spec: dict, base: Path, role: str) -> dict:
    plan = _read_json_receipt(spec["plan"], base)
    if plan.get("schema_name") != "aic_fpm_collection_plan" or plan.get("system") != "gb300":
        raise ValueError("acceptance requires a frozen GB300 FPM collection plan")
    _sha(plan["sha256"])
    options = plan["options"]
    if options.get("dataset_role", "calibration") != role:
        raise ValueError(f"expected frozen {role} plan")
    cells = [cell for cell in plan["cells"] if cell["cell_id"] == spec["cell_id"]]
    if len(cells) != 1:
        raise ValueError("plan must contain exactly one referenced cell")
    cell = cells[0]
    topology = cell["topology"]
    checkpoint_format = {"fp8_block": "fp8", "fp8": "fp8", "nvfp4": "nvfp4"}.get(cell["weight_quantization"])
    key = (plan["backend"], checkpoint_format, topology["tp"], cell["workload_kind"])
    if key not in REQUIRED or cell.get("backend") != key[0] or cell.get("state_protocol") != PROTOCOL:
        raise ValueError("cell is outside the required GLM native matrix")
    if any(topology[k] != 1 for k in ("pp", "dp", "cp", "moe_ep")) or topology["moe_tp"] != topology["tp"]:
        raise ValueError("holdout requires pure TP2/TP4")
    model = "zai-org/GLM-5.3-Flash" if key[1] == "fp8" else "nvidia/GLM-5.3-Flash-NVFP4"
    if plan["model_path"] != model:
        raise ValueError("checkpoint name and precision disagree")
    corpus = _sha(cell["input_text_sha256"])
    if options.get("input_text_sha256") != corpus:
        raise ValueError("frozen corpus differs between plan and cell")
    frozen = options["benchmark_points"]
    if digest(frozen["payload"]) != _sha(frozen["sha256"]):
        raise ValueError("frozen point manifest digest mismatch")
    points = [
        dict(point, point_type=key[3], benchmark_id=index, total_prefill_tokens=point.get("total_prefill_tokens", 0))
        for index, point in enumerate(frozen["payload"][key[3]], 1)
    ]
    geometries = [_geometry(point) for point in points]
    if not points or len(set(geometries)) != len(points):
        raise ValueError("frozen holdout geometry must be nonempty and unique")
    identity = cell["execution_identity"]
    if not identity.get("model_config_sha256") or tuple(identity[k] for k in EXECUTION_COLUMNS[1:]) != (
        "full",
        "none",
        "text",
    ):
        raise ValueError("holdout requires full text execution identity")
    runtime_cell = SimpleNamespace(
        **{
            k: cell[k]
            for k in ("cell_id", "workload_kind", "parallel_strategy", "input_text_sha256", "backend", "state_protocol")
        },
        topology=SimpleNamespace(**topology),
        execution_identity=tuple(identity[k] for k in EXECUTION_COLUMNS),
    )
    run = {
        "key": key,
        "plan": plan,
        "cell": cell,
        "runtime_cell": runtime_cell,
        "points": points,
        "corpus": corpus,
        "geometries": set(geometries),
        "spec": spec,
        "role": role,
    }
    if "shards" in spec:
        from collector.glm53flash_shard_contract import validate_point_union

        shard_manifest = _read_json_receipt(spec["shard_manifest"], base)
        children = [_plan_run(child, base, role) for child in spec["shards"]]
        if any("children" in child or child["key"] != key or child["corpus"] != corpus for child in children):
            raise ValueError("acceptance shards changed parent execution identity or corpus")
        by_id = {child["cell"]["cell_id"]: child for child in children}
        if len(by_id) != len(children):
            raise ValueError("duplicate acceptance shard")
        validate_point_union(
            plan, shard_manifest, {cid: child["plan"] for cid, child in by_id.items()}, parent_cell_id=cell["cell_id"]
        )
        for shard in shard_manifest["shards"]:
            if shard["parent_cell_id"] == cell["cell_id"]:
                child = by_id[shard["child_cell_id"]]
                child["original_point_ids"] = {
                    entry["native_benchmark_id"]: entry["original_point_id"] for entry in shard["point_map"]
                }
        run.update(children=children, shard_manifest=shard_manifest, parent_cell_id=cell["cell_id"])
    return run


def _native_run(run: dict, base: Path) -> dict:
    spec = run["spec"]
    if not spec.get("raw_root"):
        raise FileNotFoundError("native raw collection is not supplied")
    root = (base / spec["raw_root"]).resolve()
    attempt = spec.get("attempt_id")
    if not isinstance(attempt, str) or not attempt:
        raise ValueError("a frozen collector attempt_id is required")
    native = validate_native_collection(
        run["runtime_cell"], root, expected_plan_sha256=run["plan"]["sha256"], expected_attempt_id=attempt
    )
    from collector.glm53flash_runtime_identity import validate_backend_version

    validate_backend_version(run["key"][0], native.backend_version)
    observed = {point.point["benchmark_id"]: point for point in native.points}
    expected = {point["benchmark_id"]: _geometry(point) for point in run["points"]}
    if {bid: _geometry(point.point) for bid, point in observed.items()} != expected:
        raise ValueError("native collection differs from frozen requested points")
    if native.input_provenance["text_sha256"] != run["corpus"]:
        raise ValueError("native collection differs from frozen corpus")
    if native.input_provenance["tokenizer_revision"] != MODEL_REVISIONS[run["plan"]["model_path"]]:
        raise ValueError("native tokenizer revision differs from checkpoint pin")
    request_ids = set()
    receipts = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            receipts.append({"path": str(path.relative_to(root)), "sha256": file_sha256(path)})
    for path in sorted(root.rglob("*.token-streams.jsonl")):
        for observation in iter_records(path):
            request_ids.update(row["request_id"] for row in observation["requests"])
    for path in sorted(root.rglob("benchmark*.json")):
        payload = json.loads(path.read_bytes())
        if payload.get("artifact_type") != "rank":
            continue
        provenance = payload.get("input_provenance", {})
        manifest = provenance.get("native_forward_manifest")
        if manifest:
            from .sglang_artifact import read_receipt

            requests = json.loads(read_receipt(path.parent, manifest["requests"]))
            if requests.get("dataset_role") != run["role"]:
                raise ValueError("native SGLang request role differs from plan")
            request_ids.update(requests["requests"])
    if not request_ids:
        raise ValueError("native request identities are missing")
    return {
        "values": {bid: max(value for _, value in point.rank_wall_times) * 1000 for bid, point in observed.items()},
        "request_ids": request_ids,
        "receipts": receipts,
        "runtime_run_id": native.runtime_run_id,
        "runtime_grid_digest": native.runtime_grid_digest,
        "input_provenance": native.input_provenance,
        "backend_version": native.backend_version,
    }


def installed_consumer_identity() -> dict:
    """Verify loaded public Python/native files belong to a noneditable wheel."""
    import aisimulate
    import aisimulate_core
    from aisimulate import _runtime
    from aisimulate_core import _native
    from aisimulate_core.sdk import rust_engine_step

    distribution = importlib.metadata.distribution("aisimulate")
    files = distribution.files or []
    owned = {
        str(path): path for path in files if str(path).startswith(("aisimulate/", "aisimulate_core/")) and path.hash
    }
    native = [name for name in owned if name.startswith("aisimulate/_runtime") and name.endswith((".so", ".pyd"))]
    sdk = "aisimulate_core/sdk/rust_engine_step.py"
    if (
        not native
        or sdk not in owned
        or Path(distribution.locate_file(sdk)).resolve() != Path(rust_engine_step.__file__).resolve()
    ):
        raise ValueError("acceptance requires the installed wheel public SDK and Rust extension")
    if (
        Path(distribution.locate_file("aisimulate_core/__init__.py")).resolve()
        != Path(aisimulate_core.__file__).resolve()
    ):
        raise ValueError("loaded consumer is not owned by the installed wheel")
    if Path(_runtime.__file__).resolve() not in {
        Path(distribution.locate_file(owned[name])).resolve() for name in native
    }:
        raise ValueError("loaded Rust extension is not owned by the installed wheel")
    for package, module in (("aisimulate", aisimulate), ("aisimulate_core/_native", _native)):
        filename = "aisimulate/__init__.py" if package == "aisimulate" else "aisimulate_core/_native.py"
        if (
            filename not in owned
            or Path(distribution.locate_file(filename)).resolve() != Path(module.__file__).resolve()
        ):
            raise ValueError("loaded unified package or compatibility shim is not owned by the installed wheel")
    if _native.RustForwardPassPerfModel is not _runtime.RustForwardPassPerfModel:
        raise ValueError("compatibility shim does not expose the canonical native binding")
    import base64

    actual = []
    for name, item in sorted(owned.items()):
        if item.hash.mode != "sha256":
            raise ValueError("consumer RECORD must use SHA256")
        raw = Path(distribution.locate_file(item)).read_bytes()
        sha = hashlib.sha256(raw).digest()
        if base64.urlsafe_b64encode(sha).decode().rstrip("=") != item.hash.value or len(raw) != item.size:
            raise ValueError(f"installed consumer differs from wheel RECORD: {name}")
        actual.append((name, sha.hex()))
    return {
        "distribution": "aisimulate",
        "version": distribution.version,
        "payload_sha256": digest(actual),
        "api": "RustForwardPassPerfModel.best_available",
    }


def _load_native(run: dict, base: Path, mode: str) -> dict:
    if "children" in run:
        values, request_ids, children, receipts = {}, set(), {}, []
        boundaries, versions = set(), set()
        for child in run["children"]:
            native = _load_native(child, base, mode)
            cid = child["cell"]["cell_id"]
            if request_ids & native["request_ids"]:
                raise ValueError("native requests were reused across independent shard attempts")
            request_ids.update(native["request_ids"])
            mapped = {child["original_point_ids"][bid]: value for bid, value in native.get("values", {}).items()}
            if values.keys() & mapped.keys():
                raise ValueError("native shard values overlap original frozen point IDs")
            values.update(mapped)
            children[cid] = native
            receipts.append(
                {
                    "child_cell_id": cid,
                    "source_plan_sha256": child["plan"]["sha256"],
                    "original_point_ids": child["original_point_ids"],
                    **{key: value for key, value in native.items() if key not in {"values", "request_ids"}},
                }
            )
            boundaries.add(native.get("timing_boundary", TIMING_BOUNDARIES[run["key"][0]]))
            versions.add(native.get("backend_version"))
        if len(boundaries) != 1:
            raise ValueError("native shard timing boundaries differ")
        from collector.glm53flash_runtime_identity import validate_backend_version

        if len(versions) != 1:
            raise ValueError("native shard runtime versions differ")
        version = validate_backend_version(run["key"][0], versions.pop())
        if run["role"] == "holdout" and set(values) != {point["benchmark_id"] for point in run["points"]}:
            raise ValueError("native shards omit original frozen holdout point IDs")
        return {
            "values": values,
            "request_ids": request_ids,
            "shards": receipts,
            "timing_boundary": boundaries.pop(),
            "backend_version": version,
            "_children": children,
        }
    if mode == "fpm":
        return _native_run(run, base)
    try:
        from collector.glm53flash_validation import load_native
    except ImportError as error:
        raise FileNotFoundError(
            "Ops native calibration/holdout receipt adapter is required before acceptance"
        ) from error
    native = load_native(run, base)
    if not isinstance(native.get("timing_boundary"), str) or not native["timing_boundary"]:
        raise ValueError("Ops evidence must declare its independent GPU timing boundary")
    if run["role"] == "holdout":
        expected = {point["benchmark_id"] for point in run["points"]}
        if set(native.get("values", {})) != expected:
            raise ValueError("Ops holdout omits frozen requested points")
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in native["values"].values()
        ):
            raise ValueError("Ops holdout has invalid native GPU latency")
    return native


def _bind_fpm_rows(paths: list[Path], run: dict, native: dict, *, _allowed_cells: set[str] | None = None) -> dict:
    """Verify the consumer table is exactly this observed calibration cell."""
    import pyarrow.parquet as pq

    if "children" in run:
        allowed = {child["cell"]["cell_id"] for child in run["children"]}
        return {
            "parent_plan_sha256": run["plan"]["sha256"],
            "shards": [
                _bind_fpm_rows(paths, child, native["_children"][child["cell"]["cell_id"]], _allowed_cells=allowed)
                for child in run["children"]
            ],
        }

    backend, quant, tp, phase = run["key"]
    selected = []
    for path in paths:
        if path.name != "fpm_forward_perf.parquet":
            continue
        for row in pq.read_table(path).to_pylist():
            if (
                row.get("model_path"),
                row.get("backend"),
                row.get("weight_quantization"),
                row.get("tp"),
                row.get("workload_kind"),
            ) == (run["plan"]["model_path"], backend, run["cell"]["weight_quantization"], tp, phase):
                if _allowed_cells is not None:
                    if row.get("cell_id") not in _allowed_cells:
                        raise ValueError("consumer FPM contains a donor cell outside the frozen shard union")
                    if row["cell_id"] != run["cell"]["cell_id"]:
                        continue
                selected.append(row)
    if not selected:
        raise FileNotFoundError("consumer has no receipted FPM calibration rows for this cell")
    expected_identity = {
        "cell_id": run["cell"]["cell_id"],
        "source_plan_sha256": run["plan"]["sha256"],
        "collector_attempt_id": run["spec"]["attempt_id"],
        "runtime_run_id": native["runtime_run_id"],
        "runtime_grid_digest": native["runtime_grid_digest"],
        "input_text_sha256": run["corpus"],
        "input_token_ids_sha256": native["input_provenance"]["token_ids_sha256"],
        "input_tokenizer_revision": MODEL_REVISIONS[run["plan"]["model_path"]],
        "state_protocol": PROTOCOL,
        "timing_boundary": TIMING_BOUNDARIES[backend],
        "backend_version": native["backend_version"],
        "pp": 1,
        "dp": 1,
        "cp": 1,
        "moe_tp": tp,
        "moe_ep": 1,
        **run["cell"]["execution_identity"],
    }
    expected = {_geometry(point): native["values"][point["benchmark_id"]] for point in run["points"]}
    observed = {}
    for row in selected:
        if any(row.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("consumer FPM row is not bound to the native calibration receipt")
        geometry = _geometry(dict(row, point_type=row["workload_kind"]))
        if geometry in observed or geometry not in expected:
            raise ValueError("consumer FPM rows contain duplicate or unfrozen calibration geometry")
        value = row.get("latency_ms")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(value, expected[geometry], rel_tol=1e-10, abs_tol=1e-10)
        ):
            raise ValueError("consumer FPM latency differs from native calibration measurement")
        observed[geometry] = value
    if observed.keys() != expected.keys():
        raise ValueError("consumer FPM rows omit frozen calibration points")
    return {
        "source_plan_sha256": run["plan"]["sha256"],
        "rows": len(observed),
        "native_runtime_run_id": native["runtime_run_id"],
    }


def _predict(run: dict, entry: dict, mode: str, base: Path, *, calibration: dict, calibration_native: dict) -> dict:
    from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig, RustForwardPassPerfModel

    config = dict(entry["consumer_config"])
    if mode == "ops" and calibration["spec"].get("ops_execution_mode", "eager") != run["spec"].get(
        "ops_execution_mode", "eager"
    ):
        raise ValueError("Ops calibration and holdout use different native execution modes")
    backend, _quant, tp, _phase = run["key"]
    required = {
        "model": run["plan"]["model_path"],
        "system": "gb300",
        "backend": backend,
        "worker_type": "aggregated",
        "tp": tp,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp_size": tp,
        "moe_ep_size": 1,
        "nextn": 0,
        "decoder_replay": False,
        "enable_eplb": False,
        "fallback_policy": "deny",
        "database_mode": "SILICON",
        "strict_provenance": True,
        "enable_shared_layer": False,
        "estimation_mode": "fpm_interpolation" if mode == "fpm" else "op_level",
        "kvcache_quant_mode": "fp8",
    }
    if mode == "fpm":
        required["fpm_fmha_quant_mode"] = "fp8"
    for key, expected in required.items():
        if key in config and config[key] != expected:
            raise ValueError(f"consumer {key} differs from holdout contract")
        config[key] = expected
    from collector.glm53flash_runtime_identity import validate_backend_version

    version = validate_backend_version(backend, calibration_native.get("backend_version"))
    if config.get("backend_version") != version:
        raise ValueError("consumer backend revision differs from pinned runtime")
    if config.get("speculation") or config.get("estimator_config"):
        raise ValueError("holdout prediction forbids speculation or online correction/tuning")
    roots = tuple(str((base / path).resolve()) for path in config.get("systems_paths", ()))
    if not roots or not entry.get("consumer_data"):
        raise ValueError("consumer needs explicit calibration data roots and SHA256 receipts")
    receipts = entry["consumer_data"]
    for receipt in receipts:
        path = (base / receipt["path"]).resolve()
        if not any(path.is_relative_to(Path(root)) for root in roots):
            raise ValueError("consumer data receipt is outside the selected roots")
        if hashlib.sha256(path.read_bytes()).hexdigest() != _sha(receipt["sha256"]):
            raise ValueError("consumer calibration data digest mismatch")
    # Bind all selected data/config bytes, preventing an unreceipted holdout
    # table or donor inside the roots from becoming a calibration source.
    actual = {
        str(path.resolve())
        for root in roots
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix in {".parquet", ".yaml", ".yml", ".json", ".txt"}
    }
    if actual != {str((base / receipt["path"]).resolve()) for receipt in receipts}:
        raise ValueError("consumer data receipts do not cover every selected data/config file")
    paths = [(base / receipt["path"]).resolve() for receipt in receipts]
    if mode == "fpm":
        binding = _bind_fpm_rows(paths, calibration, calibration_native)
    else:
        try:
            from collector.glm53flash_validation import bind_calibration
        except ImportError as error:
            raise FileNotFoundError("Ops calibration-evidence receipt adapter is required before acceptance") from error
        if "children" in calibration:
            from collector.glm53flash_validation import bind_sharded_calibration

            binding = bind_sharded_calibration(
                paths,
                [
                    (child, calibration_native["_children"][child["cell"]["cell_id"]])
                    for child in calibration["children"]
                ],
                calibration["shard_manifest"],
            )
        else:
            binding = bind_calibration(paths, calibration, calibration_native)
    config["systems_paths"] = roots
    if mode == "ops" and calibration["spec"].get("ops_execution_mode") == "native_full_graph":
        from collector.glm53flash_graph_export import predict_homogeneous

        prediction = predict_homogeneous(run, base, config, calibration_native)
        return {**prediction, "config": config, "data_receipts": receipts, "calibration_binding": binding}
    model = RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**config))
    rows = {}
    try:
        for point in run["points"]:
            payload = {
                "version": 1,
                "wall_time": 0.0,
                "scheduled_requests": dict(
                    _expected_scheduled(point), var_prefill_length=0.0, var_decode_kv_tokens=0.0
                ),
            }
            try:
                value = model.estimate_forward_pass_time_ms(payload)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("public consumer returned no finite positive prediction")
                rows[point["benchmark_id"]] = {"prediction_ms": value}
            except Exception as error:
                rows[point["benchmark_id"]] = {"error": f"{type(error).__name__}: {error}"}
        return {
            "rows": rows,
            "config": config,
            "diagnostics": dict(model.diagnostics()),
            "data_receipts": receipts,
            "calibration_binding": binding,
        }
    finally:
        model.close()


def metrics(rows: list[dict]) -> dict:
    paired = [row for row in rows if row.get("status") == "MEASURED_AND_PREDICTED"]
    errors = [abs(row["prediction_ms"] - row["measured_ms"]) for row in paired]
    ape = sorted(100 * error / row["measured_ms"] for error, row in zip(errors, paired, strict=True))
    return {
        "requested": len(rows),
        "compared": len(paired),
        "coverage": len(paired) / len(rows) if rows else 0.0,
        "mape_pct": statistics.mean(ape) if ape else None,
        "wape_pct": 100 * sum(errors) / sum(row["measured_ms"] for row in paired) if paired else None,
        "p95_ape_pct": ape[math.ceil(0.95 * len(ape)) - 1] if ape else None,
        "max_ape_pct": max(ape) if ape else None,
    }


def evaluate(manifest: dict, base: Path) -> dict:
    if manifest.get("schema") != SCHEMA or manifest.get("mode") not in ("fpm", "ops"):
        raise ValueError("expected explicit glm53flash_independent_holdout_v1 mode fpm or ops")
    mode = manifest["mode"]
    threshold = 10.0 if mode == "fpm" else 20.0
    report = {
        "schema": SCHEMA,
        "mode": mode,
        "threshold_mape_pct": threshold,
        "acceptance": "NOT_EVALUATED",
        "input_manifest_sha256": digest(manifest),
        "consumer": None,
        "errors": [],
        "cells": [],
        "http_metrics": {
            "acceptance": "NOT_EVALUATED",
            "receipts": manifest.get("http_metrics", []),
            "boundary": "HTTP end-to-end; never included in native-forward error metrics",
        },
    }
    for receipt in manifest.get("http_metrics", []):
        _read_json_receipt(receipt, base)
    prepared = {}
    calibration_corpora, holdout_corpora = set(), set()
    calibration_geometries, holdout_geometries = set(), set()
    calibration_requests, holdout_requests = set(), set()
    for entry in manifest.get("entries", []):
        calibration = _plan_run(entry["calibration"], base, "calibration")
        holdout = _plan_run(entry["holdout"], base, "holdout")
        if calibration["key"] != holdout["key"] or holdout["key"] in prepared:
            raise ValueError("calibration/holdout cell mismatch or duplicate acceptance cell")
        key = holdout["key"]
        record = {"entry": entry, "holdout": holdout, "calibration": calibration, "errors": []}
        prepared[key] = record
        calibration_corpora.add(calibration["corpus"])
        holdout_corpora.add(holdout["corpus"])
        calibration_geometries.update(calibration["geometries"])
        holdout_geometries.update(holdout["geometries"])
        for name, run, identities in (
            ("calibration", calibration, calibration_requests),
            ("holdout", holdout, holdout_requests),
        ):
            try:
                record[name + "_native"] = _load_native(run, base, mode)
                identities.update(record[name + "_native"]["request_ids"])
            except FileNotFoundError as error:
                record["errors"].append({"role": name, "status": "NOT_EVALUATED", "error": str(error)})
            except Exception as error:
                record["errors"].append({"role": name, "status": "FAILED", "error": f"{type(error).__name__}: {error}"})
    for label, left, right in (
        ("corpus", calibration_corpora, holdout_corpora),
        ("geometry", calibration_geometries, holdout_geometries),
        ("request_id", calibration_requests, holdout_requests),
    ):
        if left & right:
            report["errors"].append(
                {"status": "FAILED", "error": f"calibration/holdout {label} overlap", "count": len(left & right)}
            )
    if prepared and not report["errors"]:
        try:
            report["consumer"] = installed_consumer_identity()
        except Exception as error:
            report["errors"].append({"status": "NOT_EVALUATED", "error": f"installed consumer: {error}"})
    for key in REQUIRED:
        result = {
            "backend": key[0],
            "weight_quantization": key[1],
            "tp": key[2],
            "phase": key[3],
            "timing_boundary": TIMING_BOUNDARIES[key[0]] if mode == "fpm" else None,
            "acceptance": "NOT_EVALUATED",
            "errors": [],
            "points": [],
        }
        record = prepared.get(key)
        if record:
            if mode == "ops" and record.get("holdout_native"):
                result["timing_boundary"] = record["holdout_native"]["timing_boundary"]
            result["errors"] = record["errors"]
            for point in record["holdout"]["points"]:
                row = {"point": point, "context_group": _group(point), "status": "NOT_EVALUATED"}
                if record.get("holdout_native"):
                    row.update(
                        measured_ms=record["holdout_native"]["values"][point["benchmark_id"]],
                        status="MEASURED_NO_PREDICTION",
                    )
                elif any(error.get("role") == "holdout" and error["status"] == "FAILED" for error in result["errors"]):
                    row["status"] = "FAILED_NATIVE_EVIDENCE"
                result["points"].append(row)
            for role in ("calibration", "holdout"):
                native = record.get(role + "_native")
                if native:
                    result[role + "_evidence"] = {
                        k: v for k, v in native.items() if k not in {"values", "request_ids"} and not k.startswith("_")
                    }
            if not result["errors"] and not report["errors"]:
                try:
                    from collector.glm53flash_runtime_identity import validate_runtime_pair

                    validate_runtime_pair(key[0], record["calibration_native"], record["holdout_native"])
                    prediction = _predict(
                        record["holdout"],
                        record["entry"],
                        mode,
                        base,
                        calibration=record["calibration"],
                        calibration_native=record["calibration_native"],
                    )
                    result["prediction_provenance"] = {k: v for k, v in prediction.items() if k != "rows"}
                    for row in result["points"]:
                        bid = row["point"]["benchmark_id"]
                        row["measured_ms"] = record["holdout_native"]["values"][bid]
                        output = prediction["rows"][bid]
                        row.update(output)
                        row["status"] = "FAILED_PREDICTION" if "error" in output else "MEASURED_AND_PREDICTED"
                except FileNotFoundError as error:
                    result["errors"].append({"status": "NOT_EVALUATED", "error": f"public consumer: {error}"})
                except Exception as error:
                    result["errors"].append(
                        {"status": "FAILED", "error": f"public consumer: {type(error).__name__}: {error}"}
                    )
        result["metrics"] = metrics(result["points"])
        result["groups"] = {
            group: metrics([row for row in result["points"] if row["context_group"] == group])
            for group in (*GROUPS, "below1K")
        }
        failed = any(error["status"] == "FAILED" for error in result["errors"] + report["errors"])
        if failed or any(row["status"] == "FAILED_PREDICTION" for row in result["points"]):
            result["acceptance"] = "FAILED"
        elif result["metrics"]["compared"]:
            complete = result["metrics"]["coverage"] == 1 and all(result["groups"][g]["requested"] for g in GROUPS)
            result["acceptance"] = "PASSED" if complete and result["metrics"]["mape_pct"] <= threshold else "FAILED"
        report["cells"].append(result)
    statuses = {cell["acceptance"] for cell in report["cells"]}
    report["acceptance"] = "FAILED" if "FAILED" in statuses else "PASSED" if statuses == {"PASSED"} else "NOT_EVALUATED"
    report["coverage"] = {
        "required_configurations": 8,
        "required_phase_cells": 16,
        "passed_phase_cells": sum(cell["acceptance"] == "PASSED" for cell in report["cells"]),
        "requested_points": sum(cell["metrics"]["requested"] for cell in report["cells"]),
        "compared_points": sum(cell["metrics"]["compared"] for cell in report["cells"]),
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = evaluate(json.loads(args.manifest.read_text()), args.manifest.resolve().parent)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["acceptance"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
