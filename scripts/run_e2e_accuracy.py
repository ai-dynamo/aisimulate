#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run both bundled predictors over a fixed public measurement cohort.

All raw rows, child output, and partial shards remain local. Only a complete,
validated summary and its qualification record are eligible for upload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, date, datetime
from numbers import Real
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_e2e_accuracy_overview import GPUS_PER_NODE_BY_FAMILY, build_summary
from build_pages_site import _accuracy_summary
from fetch_accuracy_measurements import digest

REPOSITORY = "https://github.com/ai-dynamo/aisimulate"
POLICY = "latest-complete-config-run-v1"


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(value) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def positive(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def select_points(tables: dict, max_age_days: int) -> tuple[list[dict], dict]:
    configs = {row["id"]: row for row in tables["configs"]}
    runs = {row["id"]: row for row in tables["workflow_runs"]}
    if len(configs) != len(tables["configs"]) or len(runs) != len(tables["workflow_runs"]):
        raise ValueError("duplicate measurement config/run IDs")
    cutoff = max(date.fromisoformat(row["date"]) for row in tables["benchmark_results"])

    def family(bench):
        config = configs[bench["config_id"]]
        return tuple(
            config.get(key)
            for key in (
                "model",
                "hardware",
                "framework",
                "precision",
                "disagg",
                "spec_method",
            )
        ) + (bench["isl"], bench["osl"])

    latest_by_family = {}
    for bench in tables["benchmark_results"]:
        run = runs[bench["workflow_run_id"]]
        if (
            bench["benchmark_type"] == "single_turn"
            and bench.get("error") is None
            and run.get("status") == "completed"
            and run.get("conclusion") == "success"
        ):
            key = family(bench)
            latest_by_family[key] = max(latest_by_family.get(key, date.min), date.fromisoformat(bench["date"]))
    groups = {}
    excluded = Counter()
    ids = set()
    for bench in tables["benchmark_results"]:
        if bench["id"] in ids:
            raise ValueError("duplicate benchmark ID")
        ids.add(bench["id"])
        config, run = configs[bench["config_id"]], runs[bench["workflow_run_id"]]
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            excluded["incomplete_measurement_run"] += 1
            continue
        metrics = bench["metrics"]
        if (
            bench["benchmark_type"] != "single_turn"
            or bench.get("error") is not None
            or bench.get("offload_mode", "off") != "off"
        ):
            excluded["nonstandard_or_error"] += 1
            continue
        per_node = GPUS_PER_NODE_BY_FAMILY.get(config.get("hardware"))
        total_gpus = config.get("num_decode_gpu", 0) + (config.get("num_prefill_gpu", 0) if config.get("disagg") else 0)
        if config["is_multinode"] or (per_node is not None and total_gpus > per_node):
            excluded["multinode"] += 1
            continue
        if (latest_by_family[family(bench)] - date.fromisoformat(bench["date"])).days > max_age_days:
            excluded["stale"] += 1
            continue
        if not all(positive(metrics.get(key)) for key in ("mean_ttft", "mean_tpot")):
            excluded["missing_mean_latency"] += 1
            continue
        if not all(type(bench.get(key)) is int and bench[key] > 0 for key in ("isl", "osl", "conc")):
            raise ValueError("invalid benchmark workload")
        # Select one complete run for each topology/workload, retaining all its
        # concurrency points. Never splice curves from different images/runs.
        key = (
            bench["config_id"],
            bench["isl"],
            bench["osl"],
            bench.get("recipe_fingerprint"),
        )
        rank = (bench["date"], run.get("run_started_at") or "", run["id"])
        previous = groups.get(key)
        if previous is None or rank > previous[0]:
            groups[key] = (rank, [bench])
        elif rank == previous[0]:
            previous[1].append(bench)
    points = []
    for _, benches in groups.values():
        if len({row.get("image") for row in benches}) != 1:
            excluded["mixed_image_curve"] += len(benches)
            continue
        concs = [row["conc"] for row in benches]
        if len(concs) != len(set(concs)):
            raise ValueError("duplicate concurrency in a selected curve")
        for bench in benches:
            points.append(
                {
                    "id": sha([bench["id"], bench["config_id"]]),
                    "config": configs[bench["config_id"]],
                    "benchmark": bench,
                }
            )
    if not points:
        raise ValueError("measurement selection is empty")
    points.sort(key=lambda point: point["id"])
    return points, {
        "measurement_date_through": cutoff.isoformat(),
        "selected": len(points),
        "excluded": dict(sorted(excluded.items())),
    }


def wheel_identity(wheel: Path) -> dict:
    """Check installed runtime AND legacy CLI bytes against the single wheel."""
    dist = importlib.metadata.distribution("aisimulate")
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if member.endswith("/") or not member.startswith(
                ("aisimulate/", "aiconfigurator/", "aiconfigurator_core/")
            ):
                continue
            installed = Path(dist.locate_file(member))
            if not installed.is_file() or archive.read(member) != installed.read_bytes():
                raise ValueError("installed package differs from qualified wheel")
            checked += 1
    if checked < 3:
        raise ValueError("wheel does not contain the unified package")
    for name in ("aisimulate._runtime", "aisimulate.runner", "aiconfigurator.cli.api"):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(Path(dist.locate_file("")).resolve()):
            raise ValueError("import resolved outside installed wheel")
    return {"wheel_sha256": digest(wheel), "packages": {"aisimulate": dist.version}}


def replay_spec(request, backend_version: str):
    from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec

    def engine(worker, system):
        args = {
            "engine_type": request.backend.name,
            "aic_model_path": request.model.path,
            "aic_system": system,
            "aic_backend_version": backend_version,
            "aic_tp_size": worker.tp_size,
            "aic_pp_size": worker.pp_size,
            "aic_attention_dp_size": worker.attention_dp_size,
            "max_num_seqs": max(256, request.workload.concurrency),
            "max_num_batched_tokens": 8192,
            "enable_prefix_caching": False,
            "aic_forward_model": "op_level",
        }
        for name in ("moe_tp_size", "moe_ep_size"):
            if getattr(worker, name) is not None:
                args["aic_" + name] = getattr(worker, name)
        for field, name in (
            ("gemm", "gemm_dtype"),
            ("moe", "moe_dtype"),
            ("fmha", "fmha_dtype"),
            ("kvcache", "kv_cache_dtype"),
            ("communication", "comm_dtype"),
        ):
            if getattr(request.quantization, field) is not None:
                args["aic_" + name] = getattr(request.quantization, field)
        return args

    topology = request.topology
    kwargs = {}
    if topology.kind == "agg":
        kwargs.update(
            agg_engine_args=engine(topology.worker, request.systems.prefill),
            num_workers=topology.worker.replicas,
        )
    else:
        kwargs.update(
            prefill_engine_args=engine(topology.prefill, request.systems.prefill),
            decode_engine_args=engine(topology.decode, request.systems.decode),
            num_prefill_workers=topology.prefill.replicas,
            num_decode_workers=topology.decode.replicas,
        )
    return ReplaySpec(
        backend_deployment=BackendDeploymentSpec(
            deployment_mode=topology.kind,
            backend=request.backend.name,
            backend_version=backend_version,
            **kwargs,
        ),
        workload={
            "isl": request.workload.isl,
            "osl": request.workload.osl,
            "request_count": request.workload.concurrency * 10,
            "random_range_ratio": 0.8,
            "random_seed": 0,
        },
        goal={},
        concurrency=request.workload.concurrency,
    )


def predict_point(point: dict) -> dict:
    from aiconfigurator.cli.api import cli_estimate
    from aiconfigurator.sdk.config_adapter import (
        InferenceXSource,
        adapt_config,
        to_cli_estimate_kwargs,
    )
    from aisimulate.runner import EngineReplayRunnerFactory

    config, bench = point["config"], point["benchmark"]
    # A fingerprint alone cannot reconstruct non-normalized recipe knobs.
    # Keep those points visible in campaign exclusions until a reviewed recipe
    # adapter can resolve them; speculative acceptance is never guessed.
    if bench.get("recipe_fingerprint"):
        return {
            "id": point["id"],
            "outcome": "unsupported",
            "reason": "recipe_required",
        }
    adaptation = adapt_config(InferenceXSource(config=config, benchmark=bench))
    if len(adaptation.requests) != 1:
        return {
            "id": point["id"],
            "outcome": "unsupported",
            "reason": "adapter_unsupported",
        }
    request = adaptation.requests[0]
    try:
        baseline = cli_estimate(**to_cli_estimate_kwargs(request))
        if not all(positive(value) for value in (baseline.ttft, baseline.tpot)):
            raise ValueError("invalid baseline latency")
    except Exception:
        return {
            "id": point["id"],
            "outcome": "baseline_failed",
            "reason": "baseline_failed",
        }
    worker = request.topology.worker if request.topology.kind == "agg" else request.topology.decode
    row = {
        "silicon_model": config["model"],
        "display_name": config["model"],
        "hf_model_path": request.model.path,
        "hardware": config["hardware"],
        "framework": config["framework"],
        "precision": config["precision"],
        "spec_method": config["spec_method"],
        "disagg": config["disagg"],
        "config_id": sha([config["id"], bench.get("recipe_fingerprint")]),
        "isl": bench["isl"],
        "osl": bench["osl"],
        "conc": bench["conc"],
        "is_multinode": config["is_multinode"],
        "silicon_ttft_ms": bench["metrics"]["mean_ttft"] * 1000,
        "silicon_tpot_ms": bench["metrics"]["mean_tpot"] * 1000,
        "aic_ttft_ms": float(baseline.ttft),
        "aic_tpot_ms": float(baseline.tpot),
        "aisimulate_total_gpus": config["num_decode_gpu"] + (config["num_prefill_gpu"] if config["disagg"] else 0),
        **{
            name: getattr(worker, name)
            for name in (
                "tp_size",
                "pp_size",
                "attention_dp_size",
                "moe_tp_size",
                "moe_ep_size",
            )
        },
    }
    try:
        spec = replay_spec(request, baseline.backend_version)
        metrics = EngineReplayRunnerFactory().create(0).run(spec).metrics
        if metrics.get("completed_requests") != bench["conc"] * 10:
            raise ValueError("replay did not complete every request")
        ttft, tpot = metrics.get("mean_ttft_ms"), metrics.get("mean_tpot_ms")
        if not all(positive(value) for value in (ttft, tpot)):
            raise ValueError("replay returned invalid latency")
        row.update(
            aisimulate_status="success",
            dynamo_ttft_ms=ttft,
            dynamo_tpot_ms=tpot,
            aisimulate_runner="aisimulate.engine_replay",
        )
    except Exception:
        row.update(aisimulate_status="failed")
    return {
        "id": point["id"],
        "outcome": "evaluated",
        "row": row,
        "backend_version": baseline.backend_version,
    }


def run_child(point: dict, timeout: int) -> dict:
    # Each child is bounded independently; native aborts and hangs cannot erase
    # the remaining cohort. The parent retains one outcome per selected point.
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "point"],
            input=encoded(point),
            capture_output=True,
            timeout=timeout,
            check=True,
        )
        value = json.loads(result.stdout)
        if value.get("id") != point["id"]:
            raise ValueError("child returned a different point")
        return value
    except (subprocess.SubprocessError, ValueError):
        return {
            "id": point["id"],
            "outcome": "worker_failed",
            "reason": "worker_failed",
        }


def qualify_results(points: list[dict], results: list[dict]) -> list[dict]:
    expected = {point["id"] for point in points}
    actual = [result["id"] for result in results]
    if len(expected) != len(points) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("campaign is incomplete or has overlapping points")
    if any(result.get("outcome") not in {"evaluated", "unsupported", "baseline_failed"} for result in results):
        raise ValueError("campaign child failed before recording both predictor outcomes")
    rows = [result["row"] for result in results if result["outcome"] == "evaluated"]
    if not rows or not any(row.get("aisimulate_status") == "success" for row in rows):
        raise ValueError("campaign has no successful matched predictions")
    return rows


def campaign(args) -> None:
    if args.branch != "main" and not re.fullmatch(r"release/[A-Za-z0-9][A-Za-z0-9._/-]*", args.branch):
        raise ValueError("expected main or release/* branch")
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("expected full source commit")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in 1..8")
    identity = wheel_identity(args.wheel)
    manifest = json.loads(args.manifest.read_text())
    if manifest["selection_policy"] != POLICY:
        raise ValueError("unknown cohort selection policy")
    points, selection = select_points(json.loads(args.tables.read_text()), manifest["max_age_days"])
    started = datetime.now(UTC).isoformat()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda point: run_child(point, args.point_timeout), points):
            results.append(result)
            if len(results) % 50 == 0:
                print(f"Recorded {len(results)}/{len(points)} point outcomes", flush=True)
    rows = qualify_results(points, results)
    if wheel_identity(args.wheel) != identity:
        raise ValueError("runtime changed during campaign")
    source = {"branch": args.branch, "commit_sha": args.commit, "clean": True}
    counts = Counter(row["aisimulate_status"] for row in rows)
    run = {
        "status": "complete",
        "selected": len(rows),
        **{status: counts[status] for status in ("success", "failed", "unsupported")},
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "method": "randomized_synthetic_engine_replay",
        "runtime": {**identity, "source_checkout": source},
    }
    baseline = {
        "status": "complete",
        "runtime": {
            **identity,
            "source_checkout": {**source, "repository": REPOSITORY},
            "cli_entry_point": "aiconfigurator.main:main",
        },
    }
    common = {
        "release_tag": manifest["release_tag"],
        "aic_commit_sha": args.commit,
        "aisimulate_run": run,
        "aic_run": baseline,
    }
    predictions = {**common, "rows": rows, "generated_at": started}
    summary = build_summary(
        predictions,
        {**common, "point_count": len(rows), "sha256": sha(results)},
        {
            **common,
            "final_unique_groups": len(rows),
            "dump_max_date": selection["measurement_date_through"],
        },
        predictions_sha256=sha(predictions),
        source_url=REPOSITORY.replace("ai-dynamo/aisimulate", "SemiAnalysisAI/InferenceX-app")
        + "/releases/tag/"
        + manifest["release_tag"],
        branch=args.branch,
    )
    campaign_info = {
        "schema_version": 1,
        "branch": args.branch,
        "commit_sha": args.commit,
        "wheel_sha256": identity["wheel_sha256"],
        "dataset_sha256": digest(args.manifest),
        "measurement_sha256": digest(args.tables),
        "cohort_sha256": sha([point["id"] for point in points]),
        "driver_sha256": digest(Path(__file__)),
        "run_id": args.run_id,
        "run_attempt": args.run_attempt,
        "selection_policy": POLICY,
        "measurement_filter_counts": selection["excluded"],
        "selected": len(points),
        "published": summary["totals"]["rows"],
        "outcomes": dict(sorted(Counter(result["outcome"] for result in results).items())),
        "exclusion_reasons": dict(
            sorted(Counter(result["reason"] for result in results if "reason" in result).items())
        ),
        "backend_versions": sorted({result["backend_version"] for result in results if "backend_version" in result}),
        "release_tag": manifest["release_tag"],
        "started_at": started,
        "completed_at": run["completed_at"],
        "status": "complete",
        "advisory": True,
    }
    summary["snapshot"]["campaign"] = campaign_info
    _accuracy_summary(encoded(summary).decode())
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("public artifact directory must be empty")
    (args.output / "summary.json").write_bytes(encoded(summary))
    (args.output / "qualification.json").write_bytes(
        encoded({**campaign_info, "summary_sha256": digest(args.output / "summary.json")})
    )
    print(json.dumps({"selected": len(points), "published": len(rows), "replay": dict(counts)}))


def main():
    if sys.argv[1:] == ["point"]:
        point = json.load(sys.stdin)
        # Native Rust logging bypasses redirect_stdout; redirect the descriptors
        # too so raw per-point metrics cannot leak into Actions logs.
        saved = os.dup(1)
        with tempfile.TemporaryFile(mode="w+") as logs:
            os.dup2(logs.fileno(), 1)
            os.dup2(logs.fileno(), 2)
            with redirect_stdout(logs), redirect_stderr(logs):
                result = predict_point(point)
            sys.stdout.flush()
            os.dup2(saved, 1)
        os.close(saved)
        sys.stdout.buffer.write(encoded(result))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tables", "manifest", "wheel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("branch", "commit", "run-id", "run-attempt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--point-timeout", type=int, default=180)
    campaign(parser.parse_args())


if __name__ == "__main__":
    main()
