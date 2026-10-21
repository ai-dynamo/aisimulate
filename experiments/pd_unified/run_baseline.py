"""Prepare and run static DynoSim experiments with the installed WSL environment.

The parent process prepares trace inputs and captures the SDK child's complete
stdout/stderr. Simulator source and original trace files are never edited.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import yaml


BASE = Path(__file__).resolve().parent
CASES = ("agg1", "agg4", "p1d3", "p2d2", "p3d1")
STATIC_CASES = CASES[1:]
TRACE_NAMES = {
    "conversation": "conversation_trace.jsonl",
    "toolagent": "toolagent_trace.jsonl",
    "synthetic": "synthetic_trace.jsonl",
}
EXPECTED_VERSIONS = {"aisimulate": "0.13.0", "ai-dynamo": "1.6.0", "ai-dynamo-runtime": "1.6.0"}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def absolute_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def select_rows(rows: list[dict], max_requests: int, context_limit: int, policy: str) -> tuple[list[dict], dict]:
    """Keep a chronological prefix including the complete boundary timestamp."""
    if max_requests < 0 or context_limit <= 0 or policy not in {"filter", "error"}:
        raise ValueError("Invalid trace selection settings")
    timestamps = [r["timestamp"] for r in rows]
    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Trace timestamps must be nondecreasing; no automatic sorting is performed")
    if max_requests and len(rows) > max_requests:
        boundary = rows[max_requests - 1]["timestamp"]
        candidates = [r for r in rows if r["timestamp"] <= boundary]
    else:
        candidates = rows
    too_long = [
        r for r in candidates
        if r["input_length"] >= context_limit
        or r["input_length"] + r["output_length"] > context_limit
    ]
    if too_long and policy == "error":
        raise ValueError(
            f"{len(too_long)} requests in the selected prefix exceed context={context_limit}. "
            "Choose a suitable model/context or explicitly use --context-policy filter."
        )
    selected = [
        r for r in candidates
        if r["input_length"] < context_limit
        and r["input_length"] + r["output_length"] <= context_limit
    ]
    if not selected:
        raise ValueError("No requests remain after selection")
    return selected, {
        "source_request_count": len(rows),
        "requested_prefix_limit": max_requests,
        "prefix_rows_including_boundary_timestamp": len(candidates),
        "excluded_for_context": len(too_long),
        "selected_request_count": len(selected),
        "first_timestamp_ms": selected[0]["timestamp"],
        "last_timestamp_ms": selected[-1]["timestamp"],
        "context_limit": context_limit,
        "context_policy": policy,
        "timestamp_policy": "Original milliseconds, order, ties and gaps preserved.",
        "cache_initial_state": "cold",
        "warmup_requests": 0,
    }


def prepare_trace(trace_root: Path, name: str, max_requests: int, context_limit: int, policy: str) -> tuple[Path, dict]:
    dataset = trace_root / "Mooncake_FAST25"
    source = dataset / "upstream" / "FAST25-release" / "traces" / TRACE_NAMES[name]
    manifest_path = dataset / "upstream_manifest.json"
    if not source.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Trace or manifest missing below {dataset}. "
            "Copy the complete Mooncake_FAST25 directory and check PD_TRACE_ROOT."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = source.relative_to(dataset).as_posix()
    entry = next((e for e in manifest["files"] if e["local_path"] == relative), None)
    source_digest = sha256(source)
    if entry is None or entry["sha256"] != source_digest or source.stat().st_size != entry["size"]:
        raise ValueError(f"Original trace does not match its preserved manifest: {source}")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows, 1):
        if (
            type(row.get("input_length")) is not int or row["input_length"] <= 0
            or type(row.get("output_length")) is not int or row["output_length"] < 0
            or type(row.get("timestamp")) not in (int, float)
            or not math.isfinite(row["timestamp"]) or row["timestamp"] < 0
            or not isinstance(row.get("hash_ids"), list)
            or row["input_length"] > 512 * len(row["hash_ids"])
            or any(type(x) is not int or not 0 <= x < 2**64 for x in row["hash_ids"])
        ):
            raise ValueError(f"Invalid Mooncake record on line {index}")
    selected, selection = select_rows(rows, max_requests, context_limit, policy)
    recipe = {
        "recipe_version": 1,
        "source_sha256": source_digest,
        "max_requests": max_requests,
        "context_limit": context_limit,
        "context_policy": policy,
        "include_complete_boundary_timestamp": True,
    }
    recipe_id = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:20]
    derived = dataset / "derived" / "dynosim-static-v1" / recipe_id
    derived.mkdir(parents=True, exist_ok=True)
    output = derived / "requests.jsonl"
    payload = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in selected).encode()
    output_digest = hashlib.sha256(payload).hexdigest()
    if output.exists():
        if sha256(output) != output_digest:
            raise ValueError(f"Existing derived trace was modified: {output}")
    else:
        output.write_bytes(payload)
    details = {
        "recipe": recipe,
        "selection": selection,
        "source_path": str(source),
        "source_revision": manifest["revision"],
        "source_sha256": source_digest,
        "derived_path": str(output),
        "derived_sha256": output_digest,
        "trace_block_size": 512,
        "selector_script_sha256": sha256(Path(__file__)),
    }
    write_json(derived / "selection.json", details)
    return output, details


def config_gpu_count(config: dict) -> int:
    # All shipped cases use dense TP/PP/attention-DP groups, never extra MoE ranks.
    return sum(
        p["replicas"] * p["tensor"] * p["pipeline"] * p["attention_data"]
        for worker in config["engine"]["workers"].values()
        for p in [worker["parallelism"]]
    )


def validate_experiment_contract(raw: dict, case: str) -> None:
    engine = raw["engine"]
    if raw.get("planner") or raw["traffic"].get("stop"):
        raise ValueError("This baseline requires static pools and full draining of the selected trace")
    if engine["estimation_mode"] != "op_level" or engine["fallback_policy"] != "deny":
        raise ValueError("This baseline requires explicit op_level timing with fallback_policy=deny")
    for worker in engine["workers"].values():
        if worker["timing"]["type"] != "default" or worker.get("startup_seconds", 0) != 0:
            raise ValueError("No fixed/polynomial timing or startup transitions in the static baseline")
        if worker["kv_cache"].get("host_offload") or worker["kv_cache"].get("g3_offload"):
            raise ValueError("The initial experiment is HBM-only")
    expected = 1 if case == "agg1" else 4
    if config_gpu_count(raw) != expected:
        raise ValueError(f"Case {case} must use {expected} GPUs")


def package_metadata() -> dict:
    packages = {}
    for name, expected in EXPECTED_VERSIONS.items():
        actual = importlib.metadata.version(name)
        packages[name] = {"installed": actual, "prepared_for": expected}
        if actual != expected:
            print(f"NOTE: {name}={actual}; experiment was prepared for {expected}", flush=True)
    for module in ("aisimulate", "dynamo"):
        spec = importlib.util.find_spec(module)
        packages[module + "_location"] = (
            {"origin": spec.origin, "search_locations": list(spec.submodule_search_locations or [])}
            if spec else None
        )
        location = spec.origin if spec else None
        if not location and spec and spec.submodule_search_locations:
            location = next(iter(spec.submodule_search_locations))
        if location:
            start = Path(location).resolve()
            for candidate in (start, *start.parents):
                if not (candidate / ".git").exists():
                    continue
                try:
                    head = subprocess.run(["git", "-C", str(candidate), "rev-parse", "HEAD"],
                                          check=True, capture_output=True, text=True, timeout=5).stdout.strip()
                    dirty = subprocess.run(["git", "-C", str(candidate), "status", "--porcelain", "--untracked-files=no"],
                                           check=True, capture_output=True, text=True, timeout=5).stdout.strip()
                    packages[module + "_git"] = {"root": str(candidate), "head": head, "tracked_changes": bool(dirty)}
                except (OSError, subprocess.SubprocessError) as exc:
                    packages[module + "_git"] = {"unavailable": type(exc).__name__}
                break
    return packages


def preflight_engine_launch_configs(spec, run_dir: Path) -> None:
    """Exercise the same launch materialization as Dynamo, without Replay."""
    from aisimulate.runner import materialize_engine_launch_config
    from dynamo.mocker.config import normalize_mocker_config

    deployment = spec.backend_deployment
    roles = ("aggregated",) if deployment.deployment_mode == "agg" else ("prefill", "decode")
    launches = {}
    status = {
        "status": "running",
        "scope": "Performance model construction, KV capacity and native launch validation; no forward queries or Replay.",
        "roles": {},
    }
    status_path = run_dir / "engine-preflight.json"
    write_json(status_path, status)
    for role in roles:
        field = "agg_engine_args" if role == "aggregated" else f"{role}_engine_args"
        status["roles"][role] = {"status": "loading"}
        write_json(status_path, status)
        print(f"Preflight {role}: loading performance data and resolving KV capacity...", flush=True)
        try:
            # This is DynamoReplayRunner._engine_args' existing public helper path.
            launch = normalize_mocker_config(
                materialize_engine_launch_config(
                    deployment.backend, deployment.backend_version, deployment.parallel_config,
                    getattr(deployment, field), role,
                )
            )
            launches[role] = launch
            write_json(run_dir / "engine-launch-configs.json", launches)
            engine = launch["engine"]
            timing_identity = engine.get("timing_model", {}).get("config", {})
            blocks = engine["num_gpu_blocks"]
            block_size = engine["block_size"]
            status["roles"][role] = {
                "status": "ready",
                "backend_version": timing_identity.get("backend_version"),
                "tensor_parallel_size": launch.get("tensor_parallel_size"),
                "attention_dp_size": launch.get("dp_size"),
                "num_gpu_blocks_per_rank": blocks,
                "block_size_tokens": block_size,
                "kv_token_slots_per_rank": blocks * block_size,
                "max_model_len": engine.get("max_model_len"),
            }
            print(f"Preflight {role}: ready, version={timing_identity.get('backend_version')}, "
                  f"KV slots/rank={blocks * block_size}", flush=True)
        except Exception as exc:
            status["status"] = "failed"
            status["roles"][role] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            write_json(status_path, status)
            raise
        write_json(status_path, status)
    status["status"] = "ready"
    write_json(status_path, status)


def run_sdk_child(run_dir: Path, check_only: bool) -> int:
    # Import only in the selected Linux venv, never during Windows preparation.
    print(f"Python: {sys.executable}\nWorking directory: {Path.cwd()}", flush=True)
    print(f"UID: {os.geteuid() if hasattr(os, 'geteuid') else 'not-posix'}", flush=True)
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.config.cli import CorePredictionConfig
    from aisimulate.config.common import split_config_sections
    from aisimulate.config_adapter import PredictionAdapterContext, resolve_config_adapters
    from aisimulate.resources import GuardedRunnerFactory, build_plan, require_plan
    from aisimulate.stack import resolve_runner_factory
    from aisimulate.sweeper.replay import ReplayOutputRequirements, canonical_json

    packages = package_metadata()
    print(json.dumps(packages, ensure_ascii=False, indent=2), flush=True)
    write_json(run_dir / "runtime.json", {"packages": packages, "python": sys.executable, "started_utc": utc_now()})
    raw = yaml.safe_load((run_dir / "resolved.yaml").read_text(encoding="utf-8"))
    core_raw, adapter_raw = split_config_sections(raw, command="predict")
    config = CorePredictionConfig.model_validate(core_raw)
    context = PredictionAdapterContext(
        engine=config.engine.model_dump(mode="json", exclude_none=True),
        traffic=config.traffic.model_dump(mode="json", exclude_none=True),
        evaluation=config.evaluation.model_dump(mode="json", exclude_none=True),
    )
    adapters = resolve_config_adapters(f"dynamo.{section}" for section in adapter_raw)
    adapter_specs = {
        f"dynamo.{section}": adapters[f"dynamo.{section}"].compile_prediction(value, context)
        for section, value in adapter_raw.items()
    }
    factory = resolve_runner_factory("dynamo")
    print("Compiling ReplaySpec and checking runner capabilities...", flush=True)
    spec = prediction_to_replay_spec(config, adapter_specs=adapter_specs, execution_mode="offline")
    # Retain the compiled input and declared contract even if compatibility fails.
    (run_dir / "replay-spec.json").write_text(canonical_json(spec) + "\n", encoding="utf-8")
    capabilities = factory.capabilities()
    (run_dir / "runner-capabilities.json").write_text(canonical_json(capabilities) + "\n", encoding="utf-8")
    capabilities.require_compatible(spec)
    write_json(run_dir / "performance-model-config.json", spec.backend_deployment.performance_model_metadata)
    plan = build_plan(spec.workload, stack="dynamo", policy=config.execution.resources, factory=factory, concurrency=spec.concurrency)
    write_json(run_dir / "resource-plan.json", plan)
    require_plan(plan)
    if check_only:
        preflight_engine_launch_configs(spec, run_dir)
        print("SDK and engine-launch preflight passed. No Replay executed; "
              "actual trace query coverage still requires a full run.", flush=True)
        return 0
    metadata = json.loads((run_dir / "run-metadata.json").read_text(encoding="utf-8"))
    guarded = GuardedRunnerFactory(factory, "dynamo", config.execution.resources)
    runner = guarded.create(0)
    print("Starting offline replay setup and execution (including model/capacity loading); "
          "simulation time and wall-clock runtime are different.", flush=True)
    try:
        report = runner.run(
            spec,
            output_requirements=ReplayOutputRequirements(
                include_raw_report=True,
                capture_per_request=True,
                capture_telemetry=True,
                telemetry_sample_interval_ms=metadata["telemetry_ms"],
            ),
        )
    finally:
        runner.close()
    native = report.metadata["native_report"]
    if not isinstance(native.get("per_request"), list):
        raise RuntimeError("Installed Dynamo did not return per-request records")
    telemetry = native.get("telemetry") or report.metadata.get("telemetry")
    if not isinstance(telemetry, dict) or not isinstance(telemetry.get("samples"), list):
        raise RuntimeError("Installed Dynamo did not return the requested telemetry")
    write_json(run_dir / "prediction.json", native)
    for filename, records in (("requests.jsonl", native["per_request"]), ("telemetry.jsonl", telemetry["samples"])):
        with (run_dir / filename).open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    from analyze_results import analyze_run

    analysis = analyze_run(run_dir)
    print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)
    return 0 if analysis["valid"] else 2


def run_child_process(run_dir: Path, check_only: bool) -> int:
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-dir", str(run_dir)]
    if check_only:
        command.append("--check-only")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with (run_dir / "run.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", env=env, cwd=BASE)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def write_comparison(batch_dir: Path) -> None:
    import csv

    rows = []
    for folder in sorted(batch_dir.iterdir()):
        if not folder.is_dir() or not (folder / "analysis.json").is_file():
            continue
        meta = json.loads((folder / "run-metadata.json").read_text(encoding="utf-8"))
        analysis = json.loads((folder / "analysis.json").read_text(encoding="utf-8"))
        rows.append({
            "case": meta["case"], "gpu_count": meta["num_gpus"], "speedup": meta["speedup"],
            "trace_sha256": meta["trace_sha256"],
            **{key: analysis.get(key) for key in (
                "valid", "num_input_requests", "num_completed_full_output", "p99_ttft_ms",
                "p99_request_mean_itl_ms", "p99_first_admit_wait_ms", "goodput_request_rps",
            )},
        })
    if rows:
        with (batch_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--case", choices=CASES)
    group.add_argument("--suite", choices=("static",))
    parser.add_argument("--trace", choices=TRACE_NAMES, default="conversation")
    parser.add_argument("--trace-root", default=os.environ.get("PD_TRACE_ROOT") or (
        r"C:\projects\AI_Data_Trace_Weights\Traces" if os.name == "nt" else "/home/qingya/AI_Data_Trace_Weights/Traces"))
    parser.add_argument("--output-root", default=os.environ.get("PD_OUTPUT_ROOT", str(BASE / "results")))
    parser.add_argument("--max-requests", type=int, default=200, help="Raw prefix limit; keep all tied arrivals. 0=full file.")
    parser.add_argument("--context-policy", choices=("filter", "error"), default="filter")
    parser.add_argument("--speedup", type=float, default=1.0, help="Arrival-rate multiplier, not compute acceleration")
    parser.add_argument("--telemetry-ms", type=float, default=1000.0)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare trace/config only; works on Windows")
    parser.add_argument("--check-only", action="store_true", help="Validate SDK, performance-data loading and engine launch; no Replay")
    parser.add_argument("--worker-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_dir:
        return run_sdk_child(absolute_path(args.worker_dir), args.check_only)
    if args.max_requests < 0 or any(not math.isfinite(x) or x <= 0 for x in (args.speedup, args.telemetry_ms)):
        parser.error("max-requests must be nonnegative; speedup and telemetry-ms must be finite and positive")
    if args.prepare_only and args.check_only:
        parser.error("Choose only one of --prepare-only and --check-only")
    if not args.prepare_only and not sys.platform.startswith("linux"):
        parser.error("Run the simulation using the installed WSL/Linux venv. Windows supports --prepare-only.")
    case_names = STATIC_CASES if args.suite else (args.case or "agg1",)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    batch_dir = absolute_path(args.output_root) / f"{stamp}_{args.suite or case_names[0]}"
    batch_dir.mkdir(parents=True, exist_ok=False)
    print(f"Output: {batch_dir}", flush=True)
    for case in case_names:
        run_dir = batch_dir / case
        run_dir.mkdir()
        raw = yaml.safe_load((BASE / "configs" / f"{case}.yaml").read_text(encoding="utf-8"))
        validate_experiment_contract(raw, case)
        trace, trace_info = prepare_trace(absolute_path(args.trace_root), args.trace, args.max_requests,
                                          raw["engine"]["context_length"], args.context_policy)
        raw["traffic"]["source"]["paths"] = [str(trace)]
        raw["traffic"]["load"]["speedup"] = args.speedup
        (run_dir / "resolved.yaml").write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8")
        write_json(run_dir / "trace-selection.json", trace_info)
        metadata = {
            "case": case, "engine_mode": raw["engine"]["mode"], "num_gpus": config_gpu_count(raw),
            "speedup": args.speedup, "telemetry_ms": args.telemetry_ms,
            "sla": raw.get("evaluation", {}).get("sla", {}),
            "sla_status": "Exploratory thresholds, not confirmed business SLOs.",
            "input_requests": trace_info["selection"]["selected_request_count"],
            "trace_sha256": trace_info["derived_sha256"], "created_utc": utc_now(),
            "python": sys.executable, "script_sha256": sha256(Path(__file__)),
            "status": "prepared",
        }
        write_json(run_dir / "run-metadata.json", metadata)
        print(f"{case}: {metadata['num_gpus']} GPUs, {metadata['input_requests']} requests, "
              f"{trace_info['selection']['excluded_for_context']} context exclusions, speedup={args.speedup}", flush=True)
        print(f"Trace: {trace}", flush=True)
        if args.prepare_only:
            continue
        try:
            code = run_child_process(run_dir, args.check_only)
        except BaseException:
            metadata.update(status="interrupted", finished_utc=utc_now())
            write_json(run_dir / "run-metadata.json", metadata)
            raise
        metadata.update(status=("checked" if args.check_only else "completed") if code == 0 else "failed",
                        exit_code=code, finished_utc=utc_now())
        write_json(run_dir / "run-metadata.json", metadata)
        write_comparison(batch_dir)
        if code:
            print(f"Run did not pass validity checks. Inspect {run_dir / 'run.log'}", flush=True)
            return code
    print(f"Saved: {batch_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
