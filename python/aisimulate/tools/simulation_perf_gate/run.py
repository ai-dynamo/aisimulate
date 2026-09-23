# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Alternate exact revision-local processes, checkpointing every paired case."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.forward_perf_gate.run import THREAD_ENV
from tools.simulation_perf_gate import PROTOCOL_VERSION, digest
from tools.simulation_perf_gate.cases import expand_cases
from tools.simulation_perf_gate.compare import difference, validate, write_report


def invoke(
    *, python: Path, worker: Path, case: dict, revision: str, phase: str, cpu: int, timeout: float, log: Path
) -> dict:
    request = {"protocol_version": PROTOCOL_VERSION, "revision": revision, "case": case, "phase": phase}
    response = {
        "protocol_version": PROTOCOL_VERSION,
        "revision": revision,
        "case_id": case["case_id"],
        "case_hash": digest(case),
        "phase": phase,
        "status": "ERROR",
    }
    environment = {
        **os.environ,
        **THREAD_ENV,
        "AIC_ALLOW_UNLISTED_VERSIONS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    # Never allow an inherited Python path to substitute another revision's package.
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    start = time.perf_counter()
    try:
        with log.open("w") as stderr:
            completed = subprocess.run(
                ["taskset", "--cpu-list", str(cpu), str(python), str(worker)],
                input=json.dumps(request),
                text=True,
                stdout=subprocess.PIPE,
                stderr=stderr,
                cwd=worker.parents[4],
                env=environment,
                timeout=timeout,
                check=True,
            )
        parsed = json.loads(completed.stdout)
        if not isinstance(parsed, dict):
            raise ValueError("worker response must be a JSON object")
        json.dumps(parsed, allow_nan=False)
        response = parsed
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        response["error"] = {"type": type(error).__name__, "message": str(error)}
    response["worker_elapsed_ms"] = (time.perf_counter() - start) * 1000
    return response


def retain_request_artifacts(pair: dict, case: dict, revisions: dict, output: Path) -> None:
    """Compare full records once; avoid copying large traces into every checkpoint."""
    rows = {}
    for side in ("base", "head"):
        response = pair[side]
        try:
            validate(response, case, revisions[side], "availability")
        except (KeyError, TypeError, ValueError):
            continue
        rows[side] = response["behavior"].pop("per_request")
        name = f"{case['case_id']}-{side}-requests.json.gz"
        encoded = json.dumps(rows[side], separators=(",", ":"), allow_nan=False).encode()
        with gzip.open(output / name, "wb", compresslevel=1) as destination:
            destination.write(encoded)
        response["per_request_artifact"] = {
            "path": name,
            "records": len(rows[side]),
            "sha256": digest(rows[side]),
            "complete": True,
        }
    if len(rows) == 2:
        pair["per_request_difference"] = difference(rows["base"], rows["head"], "/per_request")


def checkpoint(raw: dict, output: Path, started: float) -> None:
    raw["elapsed_seconds"] = time.monotonic() - started
    pending = output / "raw.json.tmp"
    pending.write_text(json.dumps(raw, indent=2, allow_nan=False) + "\n")
    pending.replace(output / "raw.json")


def qualification_errors(raw: dict, results: list[dict]) -> list[str]:
    errors = []
    if raw["elapsed_seconds"] > 900:
        errors.append("benchmark exceeded the 15-minute qualification budget")
    for case, result in zip(raw["cases"], results, strict=True):
        if (
            not case.get("trace_sha256")
            and result["classification"] == "PASS"
            and min(result["base_median_ms"], result["head_median_ms"]) < 2000
        ):
            errors.append(f"{case['case_id']}: median replay below the 2-second qualification target")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for side in ("base", "head"):
        parser.add_argument(f"--{side}-python", type=Path, required=True)
        parser.add_argument(f"--{side}-worker", type=Path, required=True)
        parser.add_argument(f"--{side}-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--worker-timeout", type=float, default=120)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument(
        "--qualification", action="store_true", help="Require the full same-revision suite within 15 min"
    )
    args = parser.parse_args()
    if args.rounds < 1 or args.worker_timeout <= 0:
        parser.error("rounds and worker timeout must be positive")
    if shutil.which("taskset") is None:
        parser.error("taskset is required")
    if args.qualification and (args.rounds != 5 or args.case_ids or args.base_revision != args.head_revision):
        parser.error("qualification requires the full suite, five rounds, and identical revisions")
    cases = expand_cases()
    if args.case_ids:
        known = {item["case_id"] for item in cases}
        if set(args.case_ids) - known:
            parser.error("unknown case IDs: " + ", ".join(sorted(set(args.case_ids) - known)))
        cases = [item for item in cases if item["case_id"] in args.case_ids]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    cpu = min(os.sched_getaffinity(0))
    revisions = {side: getattr(args, f"{side}_revision") for side in ("base", "head")}
    raw = {
        "protocol_version": PROTOCOL_VERSION,
        "revisions": revisions,
        "cases": cases,
        "round_count": args.rounds,
        "samples": {},
        "host": {"platform": platform.platform(), "cpu": cpu, "thread_env": THREAD_ENV},
        "workers": {
            side: {
                "python": str(getattr(args, f"{side}_python").absolute()),
                "worker": str(getattr(args, f"{side}_worker").resolve()),
            }
            for side in ("base", "head")
        },
    }
    started = time.monotonic()
    try:
        for round_index in range(args.rounds + 1):
            phase = "availability" if round_index == 0 else "measure"
            order = cases if round_index % 2 == 0 else list(reversed(cases))
            sides = ("base", "head") if round_index % 2 == 0 else ("head", "base")
            for item in order:
                samples = raw["samples"].setdefault(item["case_id"], {"rounds": []})
                if round_index and any(samples["availability"][s].get("status") != "OK" for s in sides):
                    continue  # The missing baseline remains an invalid result, never a pass.
                pair = {"round": round_index}
                for side in sides:
                    print(f"{phase} {round_index}/{args.rounds} {item['case_id']} {side}", flush=True)
                    pair[side] = invoke(
                        python=getattr(args, f"{side}_python").absolute(),
                        worker=getattr(args, f"{side}_worker").resolve(),
                        revision=revisions[side],
                        case=item,
                        phase=phase,
                        cpu=cpu,
                        timeout=args.worker_timeout,
                        log=logs / f"{item['case_id']}-{round_index}-{side}.log",
                    )
                if round_index == 0:
                    retain_request_artifacts(pair, item, revisions, output)
                    samples["availability"] = pair
                else:
                    samples["rounds"].append(pair)
                checkpoint(raw, output, started)
    finally:
        checkpoint(raw, output, started)
        results = write_report(raw, output)
    if args.qualification:
        raw["qualification_errors"] = qualification_errors(raw, results)
        checkpoint(raw, output, started)
        write_report(raw, output)
    return int(bool(raw.get("qualification_errors")) or any(result["classification"] != "PASS" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
