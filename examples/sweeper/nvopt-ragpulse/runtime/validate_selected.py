# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh full-cohort replay of a native terminal winner using its immutable bundle.

Mount this reporting helper separately; it does not change a running bundle.
KV routing can be stochastic, so measured differences are reported, not hidden.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--telemetry", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    bundle = args.bundle.resolve()
    completion = read(run / "completion.json")
    supervisor = read(run / "supervisor.json")
    accounting = read(run / "optimizer-accounting.json")
    if completion["status"] != "completed" or supervisor["returncode"] != 0:
        raise ValueError("Validation requires a clean native terminal search")
    if not accounting["all_suggestions_receipted"]:
        raise ValueError("Native suggestion feedback is incomplete")
    started = read(run / "run-start.json")
    for name, expected in started["code_sha256"].items():
        if hashlib.sha256((bundle / "runtime" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen runtime helper changed: {name}")
    result = read(run / "sweep-result.json")
    candidates = [candidate for candidate in result["candidates"] if candidate["status"] == "feasible"]
    if not candidates:
        raise ValueError("No native feasible winner exists")
    winner = min(candidates, key=lambda c: (-c["score"], c["used_gpus"], c["candidate_id"]))
    evidence = winner["provenance"]["runner_metadata"]["scenario"]
    original = run / "attempts" / Path(evidence["attempt_path"]).name
    recorded = read(original / "attempt.json")
    if recorded["status"] != "completed":
        raise ValueError("Selected native candidate lacks a completed replay")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "candidate_id": winner["candidate_id"], "reference_score": winner["score"],
               "selection": winner, "native_termination": accounting["termination_reason"],
               "scope": "fresh_full_day_reproduction_not_new_optimization_or_heldout_accuracy"}
    write(args.output / "validation.json", summary)
    clock = time.perf_counter()
    try:
        sys.path.insert(0, str(bundle / "runtime"))
        from deserialize import replay_spec_from_dict
        from scenario_runner import ScenarioRunnerFactory
        from aisimulate.sweeper.replay import ReplayOutputRequirements, canonical_json
        import dynamo._core

        with Path(dynamo._core.__file__).open("rb") as binary:
            if hashlib.file_digest(binary, "sha256").hexdigest() != started["binding_sha256"]:
                raise ValueError("Fresh validation native binary differs")
        if version("aisimulate") != started["runner_contract"]["runtime"]["aisimulate_python"]:
            raise ValueError("Fresh validation AIS version differs")
        spec = replay_spec_from_dict(read(original / "requested-spec.json"))
        if hashlib.sha256(canonical_json(spec).encode()).hexdigest() != evidence["requested_spec_sha256"]:
            raise ValueError("Requested spec does not match the native winner")
        history = None
        for hook in spec.runtime_hooks:
            if hook.provider == "dynamo.planner":
                history = hook.config["planner_config"]["load_predictor_warmup_trace"]
        factory = ScenarioRunnerFactory(
            attempts_dir=str(args.output / "attempts"), scenario=started["scenario"],
            trace_path=spec.workload["trace_paths"][0],
            dynamo_sha=started["runner_contract"]["runtime"]["dynamo_commit"],
            history_trace=history, history_sha256=started.get("history_sha256"),
            max_address_space_gib=96)
        runner = factory.create(0)
        try:
            if hashlib.sha256(canonical_json(runner.prepare_spec(spec)).encode()).hexdigest() != evidence["effective_spec_sha256"]:
                raise ValueError("Fresh effective spec differs from selected execution")
            report = runner.run(spec, output_requirements=ReplayOutputRequirements(
                include_raw_report=args.telemetry, capture_telemetry=args.telemetry))
        finally:
            runner.close()
        actual = report.metadata["scenario"]
        expected_metrics = recorded["summary"]
        actual_metrics = actual["native_summary"]
        ignored = {"wall_time_ms", "processed_tokens_per_s", "processed_output_tokens_per_s"}
        differences = {}
        for key in (expected_metrics.keys() | actual_metrics.keys()) - ignored:
            before, after = expected_metrics.get(key), actual_metrics.get(key)
            equal = (math.isclose(before, after, rel_tol=1e-8, abs_tol=1e-6)
                     if isinstance(before, (int, float)) and isinstance(after, (int, float))
                     else before == after)
            if not equal:
                differences[key] = {"reference": before, "fresh": after}
        score = actual["derived"]["fixed_day_goodput_per_gpu"]
        summary.update(status="functional_pass", actual_score=score,
                       score_relative_change=(score / winner["score"] - 1) if winner["score"] else None,
                       modeled_metrics_match=not differences, modeled_differences=differences,
                       actual_derived=actual["derived"], actual_summary=actual_metrics,
                       detail_attempt_path=actual["attempt_path"], telemetry_captured=args.telemetry,
                       note="Native routing may be stochastic; differences are retained. Full cohort/spec checks passed.")
    except BaseException as error:
        summary.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        summary.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - clock)
        write(args.output / "validation.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "candidate_id", "reference_score", "actual_score", "modeled_metrics_match", "wall_seconds")}))


if __name__ == "__main__":
    main()
