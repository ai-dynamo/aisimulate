#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run both bundled predictors over a fixed public measurement cohort.

Per-point checkpoints and shard results are internal workflow artifacts. Only a
complete, validated summary and qualification record are eligible for Pages.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from datetime import UTC, date, datetime
from numbers import Real
from pathlib import Path

# Support direct execution as well as package imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.e2e_accuracy.build_e2e_accuracy_overview import GPUS_PER_NODE_BY_FAMILY, build_summary
from scripts.e2e_accuracy.fetch_accuracy_measurements import RESOLVED_POLICY, digest, validate_manifest
from scripts.e2e_accuracy.source.cohort import select_points as select_resolved_points
from scripts.e2e_accuracy.source.recipes.runtime_evidence import prepare_runtime_evidence
from scripts.pages.build_pages_site import _accuracy_summary

REPOSITORY = "https://github.com/ai-dynamo/aisimulate"


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(value) -> str:
    return hashlib.sha256(encoded(value)).hexdigest()


def positive(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def measurement_gpu_count(config: dict) -> int:
    """Keep legacy TP*EP candidates for the adapter to validate before prediction.

    This is a selection rule, not a substitute for the wheel's adapter. The
    adapted deployment must confirm the physical count in ``predict_point``.
    """
    decode = config.get("num_decode_gpu", 0)
    prefill = config.get("num_prefill_gpu", 0) if config.get("disagg") else 0
    if any(type(count) is not int or count < 0 for count in (decode, prefill)):
        raise ValueError("invalid GPU count")
    tp, ep = config.get("decode_tp"), config.get("decode_ep")
    if (
        config.get("is_multinode") is False
        and not config.get("disagg")
        and type(config.get("decode_num_workers")) is int
        and config["decode_num_workers"] in (0, 1)
        and type(tp) is int
        and type(ep) is int
        and tp > 0
        and ep > 1
        and tp % ep == 0
        and decode == tp * ep
    ):
        return tp
    return decode + prefill


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

    def eligibility_exclusion(bench):
        config, run = configs[bench["config_id"]], runs[bench["workflow_run_id"]]
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            return "incomplete_measurement_run"
        if (
            bench["benchmark_type"] != "single_turn"
            or bench.get("error") is not None
            or bench.get("offload_mode", "off") != "off"
        ):
            return "nonstandard_or_error"
        try:
            total_gpus = measurement_gpu_count(config)
        except ValueError:
            return "invalid_gpu_count"
        per_node = GPUS_PER_NODE_BY_FAMILY.get(config.get("hardware"))
        if config["is_multinode"] or (per_node is not None and total_gpus > per_node):
            return "multinode"
        if not all(positive(bench["metrics"].get(key)) for key in ("mean_ttft", "mean_tpot")):
            return "missing_mean_latency"
        return None

    latest_by_family = {}
    for bench in tables["benchmark_results"]:
        if eligibility_exclusion(bench) is None:
            key = family(bench)
            latest_by_family[key] = max(latest_by_family.get(key, date.min), date.fromisoformat(bench["date"]))
    groups = {}
    excluded = Counter()
    ids = set()
    for bench in tables["benchmark_results"]:
        if bench["id"] in ids:
            raise ValueError("duplicate benchmark ID")
        ids.add(bench["id"])
        run = runs[bench["workflow_run_id"]]
        reason = eligibility_exclusion(bench)
        if reason is not None:
            excluded[reason] += 1
            continue
        if (latest_by_family[family(bench)] - date.fromisoformat(bench["date"])).days > max_age_days:
            excluded["stale"] += 1
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
            if previous is not None:
                excluded["superseded_curve"] += len(previous[1])
            groups[key] = (rank, [bench])
        elif rank == previous[0]:
            previous[1].append(bench)
        else:
            excluded["superseded_curve"] += 1
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
                    "github_run_id": runs[bench["workflow_run_id"]].get("github_run_id"),
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


def predictor_module_names(files) -> tuple[str, str]:
    """Select the namespace shipped by the evaluated wheel, not the evaluator."""
    members = {str(path) for path in files}
    layouts = [
        (api, adapter)
        for api, adapter in (
            ("aisimulate.legacy_cli.api", "aisimulate.sdk.config_adapter"),
            ("aiconfigurator.cli.api", "aiconfigurator.sdk.config_adapter"),
        )
        if api.replace(".", "/") + ".py" in members
    ]
    if not layouts:
        raise ValueError("wheel has no supported baseline CLI layout")
    if len(layouts) != 1:
        raise ValueError("wheel contains ambiguous baseline CLI layouts")
    api, adapter = layouts[0]
    if adapter.replace(".", "/") + "/__init__.py" not in members:
        raise ValueError("wheel is missing its matching config adapter")
    return api, adapter


def wheel_identity(wheel: Path) -> dict:
    """Check installed runtime AND legacy CLI bytes against the single wheel."""
    dist = importlib.metadata.distribution("aisimulate")
    checked = 0
    with zipfile.ZipFile(wheel) as archive:
        api, adapter = predictor_module_names(archive.namelist())
        for member in archive.namelist():
            if member.endswith("/") or not member.startswith(("aisimulate/", "aiconfigurator/", "aisimulate_core/")):
                continue
            installed = Path(dist.locate_file(member))
            if not installed.is_file() or archive.read(member) != installed.read_bytes():
                raise ValueError("installed package differs from qualified wheel")
            checked += 1
    if checked < 3:
        raise ValueError("wheel does not contain the unified package")
    for name in ("aisimulate._runtime", "aisimulate.runner", api, adapter):
        module = importlib.import_module(name)
        if not Path(module.__file__).resolve().is_relative_to(Path(dist.locate_file("")).resolve()):
            raise ValueError("import resolved outside installed wheel")
    return {
        "wheel_sha256": digest(wheel),
        "packages": {"aisimulate": dist.version},
        "baseline_api": api,
        "config_adapter": adapter,
        "cli_entry_point": "aiconfigurator.main:main"
        if api.startswith("aiconfigurator.")
        else "aisimulate.legacy_cli.entrypoint:main",
    }


def replay_spec(request, backend_version: str):
    from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec

    def engine(worker, system, role):
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
            # Both legacy releases and current runners default to op-level timing.
            # Legacy native engines reject the newer aic_forward_model selector.
        }
        for field, target in (
            ("max_seq_len", "max_model_len"),
            (
                "free_gpu_memory_fraction",
                {
                    "vllm": "gpu_memory_utilization",
                    "sglang": "mem_fraction_static",
                    "trtllm": "free_gpu_memory_fraction",
                }[request.backend.name],
            ),
        ):
            value = getattr(request.runtime, f"{role}_{field}", None) if role != "agg" else None
            if value is None:
                value = getattr(request.runtime, field)
            if value is not None:
                args[target] = value
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
            agg_engine_args=engine(topology.worker, request.systems.prefill, "agg"),
            num_workers=topology.worker.replicas,
        )
    else:
        kwargs.update(
            prefill_engine_args=engine(topology.prefill, request.systems.prefill, "prefill"),
            decode_engine_args=engine(topology.decode, request.systems.decode or request.systems.prefill, "decode"),
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
    if "source_row" in point:
        return predict_resolved_point(point)
    from aisimulate.runner import EngineReplayRunnerFactory

    api_name, adapter_name = predictor_module_names(importlib.metadata.distribution("aisimulate").files or ())
    cli_estimate = importlib.import_module(api_name).cli_estimate
    adapter = importlib.import_module(adapter_name)

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
    adaptation = adapter.adapt_config(adapter.InferenceXSource(config=config, benchmark=bench))
    if len(adaptation.requests) != 1:
        return {
            "id": point["id"],
            "outcome": "unsupported",
            "reason": "adapter_unsupported",
        }
    request = adaptation.requests[0]
    topology = request.topology
    workers = (
        (("decode", topology.worker),)
        if topology.kind == "agg"
        else (("prefill", topology.prefill), ("decode", topology.decode))
    )
    total_gpus = sum(worker.replicas * worker.gpus_per_replica for _, worker in workers)
    if total_gpus != measurement_gpu_count(config) or any(
        config.get(f"{role}_dp_attention") is True and worker.tp_size != 1 for role, worker in workers
    ):
        # Historical wheels may inflate TP*EP or ignore attention DP. Keep
        # their unsupported coverage visible rather than publish wrong scores.
        return {"id": point["id"], "outcome": "unsupported", "reason": "adapter_topology_mismatch"}
    try:
        baseline = cli_estimate(**adapter.to_cli_estimate_kwargs(request))
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
        "silicon_github_run_id": point.get("github_run_id"),
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
        "aisimulate_total_gpus": total_gpus,
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
    row.update(measurement_chart_metrics(bench["metrics"], total_gpus))
    row.update(baseline_chart_metrics(baseline, total_gpus))
    try:
        spec = replay_spec(request, baseline.backend_version)
        runner = EngineReplayRunnerFactory().create(0)
        try:
            metrics = runner.run(spec).metrics
        finally:
            runner.close()
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
            **replay_chart_metrics(metrics, total_gpus),
        )
    except Exception:
        row.update(aisimulate_status="failed")
    return {
        "id": point["id"],
        "outcome": "evaluated",
        "row": row,
        "backend_version": baseline.backend_version,
    }


def optional_positive(value):
    return float(value) if positive(value) else None


def measurement_chart_metrics(metrics, total_gpus):
    e2e = optional_positive(metrics.get("mean_e2el"))
    total = metrics.get("tput_per_gpu")
    if not positive(total) and positive(metrics.get("total_token_throughput")):
        total = metrics["total_token_throughput"] / total_gpus
    output = metrics.get("output_tput_per_gpu")
    if not positive(output) and positive(metrics.get("output_throughput")):
        output = metrics["output_throughput"] / total_gpus
    return {
        "silicon_e2e_ms": e2e * 1000 if e2e is not None else None,
        "silicon_total_per_gpu": optional_positive(total),
        "silicon_output_per_gpu": optional_positive(output),
    }


def baseline_chart_metrics(baseline, total_gpus):
    output = optional_positive(getattr(baseline, "tokens_per_second", None))
    output = output / total_gpus if output is not None else None
    return {
        "aic_e2e_ms": optional_positive(getattr(baseline, "request_latency", None)),
        "aic_output_per_gpu": output,
        "aic_total_per_gpu": None,  # The baseline API has no native total-token throughput.
    }


def replay_chart_metrics(metrics, total_gpus):
    return {
        "dynamo_e2e_ms": optional_positive(metrics.get("mean_e2e_latency_ms")),
        **{
            f"dynamo_{kind}_per_gpu": float(metrics[key]) / total_gpus if positive(metrics.get(key)) else None
            for kind, key in (
                ("output", "output_throughput_tok_s"),
                ("total", "total_throughput_tok_s"),
            )
        },
    }


def predict_resolved_point(point):
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper.replay import BackendDeploymentSpec, ReplaySpec
    from scripts.e2e_accuracy.source.estimate import estimate_kwargs
    from scripts.e2e_accuracy.source.model_config_snapshot import materialize_model_config
    from scripts.e2e_accuracy.source.replay import replay_spec as resolved_replay_spec
    from scripts.e2e_accuracy.source.schema import SiliconRow

    if point.get("resolution_error"):
        return {"id": point["id"], "outcome": "unsupported", "reason": "source_unresolved"}
    deployment = point["deployment"]
    config, bench = point["config"], point["benchmark"]
    api_name, adapter_name = predictor_module_names(importlib.metadata.distribution("aisimulate").files or ())
    namespace = "aiconfigurator" if api_name.startswith("aiconfigurator.") else "aisimulate"
    try:
        version = importlib.import_module(namespace + ".sdk.perf_database").get_latest_database_version(
            system=deployment["system"], backend=deployment["backend"]
        )
        if not version:
            raise ValueError("no performance-data version")
    except Exception:
        return {"id": point["id"], "outcome": "unsupported", "reason": "database_unavailable"}
    roles = deployment["roles"]
    shape = roles["decode" if config["disagg"] else "aggregated"]["topology"]
    row = {
        "silicon_github_run_id": point["source_row"].get("github_run_id"),
        "silicon_model": config["model"],
        "display_name": config["model"],
        "hf_model_path": deployment["model_path"],
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
        "configuration_quality": deployment["configuration_quality"],
        "aic_status": "failed",
        "aic_ttft_ms": None,
        "aic_tpot_ms": None,
        "aisimulate_status": "failed",
        "aisimulate_total_gpus": sum(
            spec["topology"]["tp"]
            * spec["topology"]["pp"]
            * spec["topology"]["attention_dp"]
            * spec["topology"]["workers"]
            for spec in roles.values()
        ),
        **{
            field: shape[key]
            for field, key in (
                ("tp_size", "tp"),
                ("pp_size", "pp"),
                ("attention_dp_size", "attention_dp"),
                ("moe_tp_size", "moe_tp"),
                ("moe_ep_size", "moe_ep"),
            )
        },
    }
    total_gpus = row["aisimulate_total_gpus"]
    row.update(measurement_chart_metrics(bench["metrics"], total_gpus))
    try:
        adapter = importlib.import_module(adapter_name)
        model_path = (
            materialize_model_config(deployment["checkpoint_config"])
            if deployment.get("checkpoint_config")
            else deployment["model_path"]
        )
        if hasattr(adapter, "ResolvedInferenceXSource"):
            source = adapter.ResolvedInferenceXSource(
                deployment,
                config,
                bench,
                "https://github.com/SemiAnalysisAI/InferenceX/tree/" + point["source_row"]["head_sha"],
            )
            report = adapter.adapt_config(
                source,
                adapter.AdapterOverrides(model_path=model_path, backend_version=version),
            )
            if not report.requests:
                raise ValueError("resolved source cannot be represented by estimate API")
            kwargs = adapter.to_cli_estimate_kwargs(report.requests[0])
        else:
            # Historical wheels predate the resolved adapter. Use the same
            # source-derived kwargs as gym, never the alias-only DB adapter.
            kwargs = estimate_kwargs(SiliconRow(**point["source_row"]), deployment).to_call_kwargs()
            kwargs.update(model_path=model_path, backend_version=version)
        baseline = importlib.import_module(api_name).cli_estimate(**kwargs)
        if not all(positive(value) for value in (baseline.ttft, baseline.tpot)):
            raise ValueError("invalid baseline latency")
        row.update(
            aic_status="success",
            aic_ttft_ms=float(baseline.ttft),
            aic_tpot_ms=float(baseline.tpot),
        )
        row.update(baseline_chart_metrics(baseline, total_gpus))
    except Exception as error:
        row["aic_error"] = str(error)
    try:
        spec = resolved_replay_spec(deployment, version, BackendDeploymentSpec, ReplaySpec)
        runner = EngineReplayRunnerFactory().create(0)
        try:
            metrics = runner.run(spec).metrics
        finally:
            runner.close()
        if metrics.get("completed_requests") != deployment["workload"]["request_count"]:
            raise ValueError("replay did not complete the source workload")
        ttft, tpot = metrics.get("mean_ttft_ms"), metrics.get("mean_tpot_ms")
        if not all(positive(value) for value in (ttft, tpot)):
            raise ValueError("invalid replay latency")
        row.update(
            aisimulate_status="success",
            dynamo_ttft_ms=ttft,
            dynamo_tpot_ms=tpot,
            aisimulate_runner="aisimulate.engine_replay",
            **replay_chart_metrics(metrics, total_gpus),
        )
    except Exception as error:
        row["aisimulate_error"] = str(error)
    return {"id": point["id"], "outcome": "evaluated", "row": row, "backend_version": version}


def resolve_points(points, cache_dir, workers, *, allow_estimated_defaults=False):
    from scripts.e2e_accuracy.source.deployment import inspect_deployment
    from scripts.e2e_accuracy.source.recipes.inferencex_recipe import GitHubRecipeSource
    from scripts.e2e_accuracy.source.schema import SiliconRow

    source = GitHubRecipeSource(cache_dir=cache_dir, archived_runtime=True)

    def resolve(point):
        try:
            deployment, evidence, issues = inspect_deployment(
                SiliconRow(**point["source_row"]),
                source,
                allow_estimated_defaults=allow_estimated_defaults,
            )
            return {**point, "deployment": deployment, "evidence": evidence, "resolution_error": issues}
        except Exception as error:
            return {**point, "resolution_error": [{"message": str(error)}]}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        resolved = []
        for point in pool.map(resolve, points):
            resolved.append(point)
            if len(resolved) % 50 == 0:
                print(f"Resolved source evidence for {len(resolved)}/{len(points)} points", flush=True)
        return resolved


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


def validate_outcomes(points: list[dict], results: list[dict]) -> None:
    expected = {point["id"] for point in points}
    actual = [result["id"] for result in results]
    if len(expected) != len(points) or len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("campaign is incomplete or has overlapping points")
    if any(result.get("outcome") not in {"evaluated", "unsupported", "baseline_failed"} for result in results):
        raise ValueError("campaign child failed before recording both predictor outcomes")


def qualify_results(points: list[dict], results: list[dict]) -> list[dict]:
    validate_outcomes(points, results)
    rows = [result["row"] for result in results if result["outcome"] == "evaluated"]
    if not rows or not any(row.get("aisimulate_status") == "success" for row in rows):
        raise ValueError("campaign has no successful matched predictions")
    return rows


def campaign(args) -> None:
    preview = getattr(args, "preview", False)
    branch_pattern = r"[A-Za-z0-9][A-Za-z0-9._/-]*" if preview else r"release/[A-Za-z0-9][A-Za-z0-9._/-]*"
    if args.branch.endswith("/") or (args.branch != "main" and not re.fullmatch(branch_pattern, args.branch)):
        raise ValueError("expected main or release/* branch")
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise ValueError("expected full source commit")
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be in 1..8")
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    identity = wheel_identity(args.wheel)
    tables = json.loads(args.tables.read_text())
    resolved = manifest["selection_policy"] == RESOLVED_POLICY
    if resolved:
        points, selection = select_resolved_points(tables, manifest["max_age_days"])
        for point in points:
            point["id"] = sha([point["benchmark"]["id"], point["config"]["id"]])
    else:
        points, selection = select_points(tables, manifest["max_age_days"])
    del tables
    points.sort(key=lambda point: point["id"])
    point_ids = [point["id"] for point in points]
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("duplicate selected point IDs")
    shard_count = getattr(args, "shard_count", 1)
    shard_index = getattr(args, "shard_index", 0)
    if not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    if shard_count > 1 and args.output.exists() and any(args.output.iterdir()):
        raise ValueError("shard output directory must be empty")
    metadata = {
        "manifest": manifest,
        "identity": identity,
        "selection": selection,
        "point_ids": point_ids,
        "branch": args.branch,
        "commit": args.commit,
        "run_id": args.run_id,
        "preview": preview,
        "configuration_mode": getattr(args, "configuration_mode", "verified"),
        "point_timeout": args.point_timeout,
        "dataset_sha256": digest(args.manifest),
        "measurement_sha256": digest(args.tables),
        "driver_sha256": driver_digest(),
    }
    points = points[shard_index::shard_count]
    started = datetime.now(UTC).isoformat()
    if resolved:
        args.evidence.mkdir(parents=True, exist_ok=True)
        if getattr(args, "fetch_runtime_evidence", False):
            acquired = prepare_runtime_evidence(points, args.source_cache)
            (args.evidence / "runtime-fetch.json").write_bytes(encoded(acquired))
            print("Runtime evidence: " + json.dumps(acquired["counts"]), flush=True)
        points = resolve_points(
            points,
            args.source_cache,
            args.workers,
            allow_estimated_defaults=getattr(args, "configuration_mode", "verified") == "estimated",
        )
        args.evidence.mkdir(parents=True, exist_ok=True)
        (args.evidence / "resolved-points.json").write_bytes(encoded(points))
        print(
            f"Resolved {sum(not p.get('resolution_error') for p in points)}/{len(points)} source deployments",
            flush=True,
        )
    results = []
    evidence = getattr(args, "evidence", None)
    if evidence:
        evidence.mkdir(parents=True, exist_ok=True)
    # Completion order preserves progress even when an earlier point stalls.
    checkpoint = (evidence / "results.jsonl").open("wb") if evidence else nullcontext()
    with checkpoint as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_child, point, args.point_timeout) for point in points]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if stream:
                stream.write(encoded(result))
                stream.flush()
            if len(results) % 50 == 0:
                print(f"Recorded {len(results)}/{len(points)} point outcomes", flush=True)
    results.sort(key=lambda result: result["id"])
    if evidence:
        (evidence / "results.json").write_bytes(encoded(results))
    validate_outcomes(points, results)
    if wheel_identity(args.wheel) != identity:
        raise ValueError("runtime changed during campaign")
    cohort_sha = sha(points if resolved else [point["id"] for point in points])
    if shard_count > 1:
        args.output.mkdir(parents=True, exist_ok=True)
        bundle = {
            "schema_version": 1,
            "metadata": metadata,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "run_attempt": args.run_attempt,
            "cohort_sha256": cohort_sha,
            "started_at": started,
            "results": results,
        }
        (args.output / f"shard-{shard_index}.json").write_bytes(encoded(bundle))
    else:
        publish_campaign(metadata, results, cohort_sha, started, args.output, args.run_attempt)


def driver_digest() -> str:
    return sha(
        {
            str(path.relative_to(Path(__file__).parent)): digest(path)
            for path in [
                Path(__file__),
                *sorted(Path(__file__).with_name("source").rglob("*.py")),
                *sorted(Path(__file__).with_name("source").rglob("*.json")),
            ]
        }
    )


def merge_shards(args) -> None:
    """Reuse complete siblings from earlier attempts of this same workflow run."""
    if args.shard_count < 1:
        raise ValueError("shard count must be positive")
    paths = sorted(args.shards.glob("*.json"))
    if {path.name for path in paths} != {f"shard-{i}.json" for i in range(args.shard_count)}:
        raise ValueError("missing or unexpected shard files")
    bundles = [json.loads((args.shards / f"shard-{i}.json").read_text()) for i in range(args.shard_count)]
    metadata = bundles[0]["metadata"]
    expected = {
        "branch": args.branch,
        "commit": args.commit,
        "run_id": args.run_id,
        "preview": args.preview,
        "configuration_mode": args.configuration_mode,
        "dataset_sha256": digest(args.manifest),
        "driver_sha256": driver_digest(),
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("shard provenance does not match this campaign")
    if metadata["identity"]["wheel_sha256"] != args.wheel_sha256:
        raise ValueError("shard wheel does not match this campaign")
    if metadata["manifest"] != json.loads(args.manifest.read_text()):
        raise ValueError("shard dataset manifest mismatch")
    ids = metadata["point_ids"]
    if ids != sorted(set(ids)):
        raise ValueError("selected point IDs must be unique and sorted")
    results = []
    for index, bundle in enumerate(bundles):
        if (
            bundle["schema_version"] != 1
            or bundle["metadata"] != metadata
            or bundle["shard_count"] != args.shard_count
            or bundle["shard_index"] != index
            or not 1 <= int(bundle["run_attempt"]) <= int(args.run_attempt)
            or not re.fullmatch(r"[0-9a-f]{64}", bundle["cohort_sha256"])
        ):
            raise ValueError("inconsistent shard provenance")
        validate_outcomes([{"id": id_} for id_ in ids[index :: args.shard_count]], bundle["results"])
        results.extend(bundle["results"])
    results.sort(key=lambda result: result["id"])
    publish_campaign(
        metadata,
        results,
        sha([bundle["cohort_sha256"] for bundle in bundles]),
        min(bundle["started_at"] for bundle in bundles),
        args.output,
        args.run_attempt,
    )


def publish_campaign(metadata, results, cohort_sha, started, output, run_attempt) -> None:
    manifest, identity, selection = (metadata[key] for key in ("manifest", "identity", "selection"))
    resolved = manifest["selection_policy"] == RESOLVED_POLICY
    preview = metadata["preview"]
    points = [{"id": id_} for id_ in metadata["point_ids"]]
    rows = qualify_results(points, results)
    source = {"branch": metadata["branch"], "commit_sha": metadata["commit"], "clean": True}
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
        },
    }
    common = {
        "release_tag": manifest["release_tag"],
        "aic_commit_sha": metadata["commit"],
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
        branch=metadata["branch"],
        preview=preview,
        exclude_multinode=not resolved,
    )
    campaign_info = {
        "metric_contract": "serving-metrics-v1",
        "schema_version": 1,
        "branch": metadata["branch"],
        "commit_sha": metadata["commit"],
        "wheel_sha256": identity["wheel_sha256"],
        "dataset_sha256": metadata["dataset_sha256"],
        "measurement_sha256": metadata["measurement_sha256"],
        "cohort_sha256": cohort_sha,
        "driver_sha256": metadata["driver_sha256"],
        "run_id": metadata["run_id"],
        "run_attempt": run_attempt,
        "selection_policy": manifest["selection_policy"],
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
    if resolved:
        campaign_info["configuration"] = {
            "profile": "coverage-experiment/1" if metadata["configuration_mode"] == "estimated" else "verified",
            "counts": dict(Counter(row["configuration_quality"] for row in rows)),
        }
    summary["snapshot"]["campaign"] = campaign_info
    if preview:
        campaign_info["preview"] = True
    _accuracy_summary(encoded(summary).decode(), allow_preview=preview)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("public artifact directory must be empty")
    (output / "summary.json").write_bytes(encoded(summary))
    (output / "qualification.json").write_bytes(
        encoded({**campaign_info, "summary_sha256": digest(output / "summary.json")})
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
    if sys.argv[1:2] == ["merge"]:
        for name in ("shards", "manifest", "output"):
            parser.add_argument("--" + name, type=Path, required=True)
        for name in ("branch", "commit", "run-id", "run-attempt", "wheel-sha256"):
            parser.add_argument("--" + name, required=True)
        parser.add_argument("--shard-count", type=int, required=True)
        parser.add_argument("--configuration-mode", choices=("verified", "estimated"), default="verified")
        parser.add_argument("--preview", action="store_true")
        merge_shards(parser.parse_args(sys.argv[2:]))
        return
    for name in ("tables", "manifest", "wheel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("branch", "commit", "run-id", "run-attempt"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--configuration-mode", choices=("verified", "estimated"), default="verified")
    parser.add_argument("--fetch-runtime-evidence", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--point-timeout", type=int, default=180)
    parser.add_argument("--source-cache", type=Path, default=Path(".cache/e2e-accuracy/source"))
    parser.add_argument("--evidence", type=Path, default=Path(".cache/e2e-accuracy/evidence"))
    parser.add_argument(
        "--preview", action="store_true", help="Produce a branch preview excluded from public publication"
    )
    campaign(parser.parse_args())


if __name__ == "__main__":
    main()
