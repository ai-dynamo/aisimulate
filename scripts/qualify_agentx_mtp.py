#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualify installed CLI MTP behavior on a supplied, reproducible Weka corpus.

Runs functional/cost checks only. Does not allocate GPUs or establish accuracy.
Engine cases use duration profiles; Dynamo cases replay complete traces through
its existing round-robin/KV routes. Requires matching installed builds; never
alters imports. Affinity and Dynamo duration profiles require separate routing
integration and are not qualified by this script.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import distribution
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "tests/e2e/configs/unified_cli/predict/engine"


def require(condition, message):
    """Qualification checks must also execute with Python optimization enabled."""
    if not condition:
        raise ValueError(message)


def unpack_report(payload):
    """Read the Engine flat report or Dynamo's existing summary envelope."""
    if "summary" not in payload:
        return payload
    require(isinstance(payload["summary"], dict), "report summary must be an object")
    return {
        **payload["summary"],
        **{
            key: payload[key]
            for key in ("speculation", "agentic_qualification", "agentic_input_format", "agentic_lanes")
            if key in payload
        },
    }


def package_receipt(name):
    package = distribution(name)
    direct_url = json.loads(package.read_text("direct_url.json") or "null")
    wheel_sha256 = None
    if direct_url:
        source = urlparse(direct_url.get("url", ""))
        if source.scheme == "file":
            wheel = Path(unquote(source.path))
            if wheel.suffix == ".whl" and wheel.is_file():
                wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
        else:
            wheel_sha256 = direct_url.get("archive_info", {}).get("hashes", {}).get("sha256")
    return {
        "version": package.version,
        "direct_url": direct_url,
        "wheel_sha256": wheel_sha256,
        "record_sha256": hashlib.sha256((package.read_text("RECORD") or "").encode()).hexdigest(),
    }


def validate_report(report, raw, sd, duration):
    acceptance = report["speculative_acceptance"]
    require(report["agentic_qualification"] == "functional_only", "unexpected qualification")
    require(
        report["agentic_model_projection"]["target_model"] == raw["engine"]["model"],
        "target model changed",
    )
    require(report["completed_requests"] > 0, "no requests completed")
    forwards = acceptance["decode_forwards"]
    require(forwards > 0, "no measured decode forwards")
    mean = acceptance["mean_accept_length"]
    require(
        isinstance(mean, (int, float)) and math.isfinite(mean),
        "missing or nonfinite mean acceptance",
    )
    require(
        acceptance["sampling_population"] == "measurement_completed_decode_passes",
        "unexpected sampling population",
    )
    depth = raw["engine"]["speculation"]["num_speculative_tokens"] if sd else 0
    require(1 <= mean <= depth + 1, "acceptance outside configured depth")
    if sd:
        assumptions = raw["engine"]["speculation"]
        require(set(report["speculation"]) == set(raw["engine"]["workers"]), "missing or unexpected SD role metadata")
        for role in report["speculation"].values():
            require(role["resolved_method"] == "mtp", "method changed")
            require(role["target_model"] == raw["engine"]["model"], "SD target model changed")
            require(role["num_speculative_tokens"] == depth, "SD depth changed")
            require(
                role["expected_accepted_draft_tokens"] == assumptions["expected_accepted_tokens"],
                "SD expectation changed",
            )
            require(role["seed"] == assumptions["seed"], "SD seed changed")
            require(role["capacity_source"] == "explicit_fixed", "KV capacity provenance changed")
        expected = 1 + raw["engine"]["speculation"]["expected_accepted_tokens"]
        # A bounded [1, K+1] sample has variance at most K²/4. Six standard
        # errors avoid treating ordinary sampling noise as a qualification bug.
        tolerance = max(0.02, 3 * depth / math.sqrt(forwards))
        require(
            abs(mean - expected) <= tolerance,
            "sampled AL does not match configured expectation",
        )
    else:
        require(not report.get("speculation"), "SD-off case reports active speculation")
    if duration is None:
        require(not report.get("agentic_profile"), "complete-trace case unexpectedly used a profile")
        outcomes = report["agentic_play_outcomes"]
        require(len(outcomes) == report["agentic_graph"]["play_count"] > 0, "missing play outcomes")
        require(
            all(
                item["status"] == "completed"
                and isinstance(item.get("settled_at_ms"), (int, float))
                and math.isfinite(item["settled_at_ms"])
                for item in outcomes
            ),
            "complete-trace replay left an unsuccessful or unsettled play",
        )
        return
    profile = report["agentic_profile"]
    require(not profile["cancel_drain_timed_out"], "cancellation drain timed out")
    require(
        profile["client_in_flight_requests"] == profile["server_unsettled_requests"] == 0,
        "unsettled requests",
    )
    require(
        abs(profile["admission_cutoff_ms"] - profile["profile_start_ms"] - duration * 1000) < 1e-5,
        "admission window differs from requested duration",
    )
    require(
        profile["issued_requests"] == profile["successful_responses"] + profile["canceled_requests"],
        "request terminal counts do not balance",
    )


def run_case(case, *, cli, output, duration, trace_sha256):
    name, raw, stack, sd = case
    directory = output / name
    directory.mkdir(exist_ok=True)
    config = directory / "config.yaml"
    command = [
        cli,
        "predict",
        "--stack",
        stack,
        "--config",
        str(config),
        "--output-dir",
        str(directory / "result"),
    ]
    receipt = {
        "case": name,
        "command": command,
        "exit_code": None,
        "status": "failed",
        "trace_sha256": trace_sha256,
        "qualification": "functional_only",
        "hardware_accuracy": "NOT_EVALUATED",
        "execution_scope": "duration_profile" if stack == "engine" else "complete_trace",
    }
    try:
        config.write_text(yaml.safe_dump(raw, sort_keys=False))
        receipt["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        with (directory / "cli.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        receipt["exit_code"] = result.returncode
        require(result.returncode == 0, f"CLI exited with status {result.returncode}")
        report = unpack_report(json.loads((directory / "result/prediction.json").read_text()))
        validate_report(report, raw, sd, duration if stack == "engine" else None)
        receipt.update(
            status="passed",
            completed_requests=report["completed_requests"],
            duration_ms=report["duration_ms"],
            acceptance=report["speculative_acceptance"],
            profile=report.get("agentic_profile"),
            play_outcomes=report.get("agentic_play_outcomes"),
            phases=report.get("agentic_phases"),
            speculation=report.get("speculation"),
            graph_digest=report["agentic_graph"]["graph_digest"],
            lifecycle_digest=report.get("agentic_lifecycle_digest"),
            lifecycle_event_count=report.get("agentic_lifecycle_event_count"),
        )
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{name}: {receipt['status']} (exit={receipt['exit_code']})", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", default="aisimulate")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--family", choices=("glm", "dsv4", "all"), default="all")
    parser.add_argument("--scope", choices=("engine", "dynamo", "all"), default="all")
    args = parser.parse_args()
    require(args.jobs > 0, "jobs must be positive")
    require(
        math.isfinite(args.duration) and args.duration > 0,
        "duration must be positive and finite",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    trace = args.trace.resolve()
    trace_sha256 = hashlib.sha256(trace.read_bytes()).hexdigest()
    script = Path(__file__).resolve()
    archived_script = args.output / "qualify_agentx_mtp.py"
    if script != archived_script.resolve():
        shutil.copyfile(script, archived_script)
    package_names = ("aisimulate",) if args.scope == "engine" else ("aisimulate", "ai-dynamo", "ai-dynamo-runtime")
    packages = {name: package_receipt(name) for name in package_names}
    manifest = {
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "argv": sys.argv,
        "trace_sha256": trace_sha256,
        "duration_seconds": args.duration,
        "scope": args.scope,
        "duration_applies_to": "engine_duration_profiles_only",
        "packages": packages,
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "qualification": "functional_only",
        "hardware_accuracy": "NOT_EVALUATED",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cases = []
    for family, start in (("glm", 13), ("dsv4", 15)):
        if args.family not in (family, "all"):
            continue
        for offset, topology in enumerate(("agg", "disagg")):
            sample = next(EXAMPLES.glob(f"{start + offset:02d}-*.yaml"))
            backends = ("vllm", "sglang") if family == "glm" else ("sglang",)
            for backend in backends:
                variants = [
                    (stack, route, sd)
                    for stack, route in (("engine", "default"), ("dynamo", "default"), ("dynamo", "kv"))
                    if args.scope in (stack, "all")
                    for sd in ((False, True) if family == "dsv4" else (True,))
                ]
                for stack, route, sd in variants:
                    raw = yaml.safe_load(sample.read_text())
                    raw["engine"]["backend"] = backend
                    raw["engine"]["backend_version"] = "0.24.0" if backend == "vllm" else "0.5.14"
                    if not sd:
                        del raw["engine"]["speculation"]
                    raw["traffic"]["source"].update(paths=[str(trace)], block_size=64)
                    raw["traffic"]["load"].update(agentic_lanes=2)
                    if stack == "engine":
                        raw["traffic"]["load"].update(
                            agentic_snapshot={"seed": 42},
                            agentic_warmup=True,
                            agentic_profile={"duration_seconds": args.duration},
                        )
                    for worker in raw["engine"]["workers"].values():
                        worker["parallelism"]["replicas"] = 2
                    raw["execution"] = {"resources": {"memory_limit_gb": 4}}
                    if route != "default":
                        raw["router"] = {"policy": "kv_router"}
                    name = f"{family}-{backend}-{topology}-{stack}-{route}-sd{int(sd)}"
                    cases.append((name, raw, stack, sd))

    # Preserve the requested matrix even if construction fails before replay.
    (args.output / "matrix.json").write_text(
        json.dumps(
            [{"case": case[0], "status": "not_run"} for case in cases],
            indent=2,
        )
        + "\n"
    )

    from aisimulate_core.sdk import RustForwardPassPerfModel

    costs = []
    identities = {}
    for _, raw, _, _ in cases:
        engine = raw["engine"]
        identities[(engine["model"], engine["backend"])] = engine
    for (model_name, backend), engine in identities.items():
        base = {
            "model": model_name,
            "backend": backend,
            "backend_version": engine["backend_version"],
            "system": engine["hardware"],
            "tp": 8,
            "moe_tp_size": 8,
            "moe_ep_size": 1,
            "worker_type": "decode",
            "estimation_mode": "op_level",
            "fallback_policy": "deny",
        }
        values = {}
        for method, controls in (
            ("off", {}),
            ("legacy_nextn", {"nextn": 3}),
            (
                "explicit_mtp",
                {
                    "speculation": {
                        "kind": "mtp",
                        "params": {"num_speculative_tokens": 3},
                    }
                },
            ),
        ):
            model = RustForwardPassPerfModel.best_available({**base, **controls})
            try:
                values[method] = model.predict_decode_latency_total(1, 1024)
                require(
                    values[method] is not None and math.isfinite(values[method]) and values[method] > 0,
                    f"invalid {method} cost",
                )
            finally:
                model.close()
        require(
            values["legacy_nextn"] == values["explicit_mtp"],
            "explicit and legacy iteration costs differ",
        )
        costs.append(
            {
                "config": base,
                "batch_size": 1,
                "context_tokens": 1024,
                "decode_forward_ms": values,
            }
        )
    (args.output / "cost-comparison.json").write_text(json.dumps(costs, indent=2) + "\n")

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        receipts = list(
            pool.map(
                lambda case: run_case(
                    case,
                    cli=args.cli,
                    output=args.output,
                    duration=args.duration,
                    trace_sha256=trace_sha256,
                ),
                cases,
            )
        )
    (args.output / "matrix.json").write_text(json.dumps(receipts, indent=2) + "\n")
    raise SystemExit(int(any(receipt["status"] != "passed" for receipt in receipts)))


if __name__ == "__main__":
    main()
