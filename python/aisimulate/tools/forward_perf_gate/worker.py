#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one or more revision-local forward-prediction benchmark requests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from dataclasses import replace
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PYTHON_ROOT))

from aiconfigurator.sdk.errors import PerfDataNotAvailableError
from tools.forward_perf_gate import PROTOCOL_VERSION
from tools.forward_perf_gate.measurement import (
    BenchmarkCase,
    clear_caches,
    ensure_rust_library_present,
    measure_cold_and_warm,
    measure_session_setup_ms,
    phase_call,
    priming_runtime_config,
    redirect_output,
)

CASE_KEYS = {
    "case_id",
    "model_id",
    "model_path",
    "system_name",
    "backend_name",
    "backend_version",
    "database_mode",
    "phase",
    "batch_size",
    "isl",
    "osl",
    "prefix",
    "stride",
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "moe_tp_size",
    "moe_ep_size",
}
GROUP_KEYS = (
    "model_id",
    "model_path",
    "system_name",
    "backend_name",
    "backend_version",
    "database_mode",
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "moe_tp_size",
    "moe_ep_size",
)


def canonical_case_hash(case: dict) -> str:
    payload = json.dumps(case, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_case(case: object) -> dict:
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    missing = CASE_KEYS - case.keys()
    unknown = case.keys() - CASE_KEYS
    if missing or unknown:
        raise ValueError(f"invalid case fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError("case_id must be a non-empty string")
    if case["database_mode"] not in {"SILICON", "EMPIRICAL"}:
        raise ValueError(f"unsupported database_mode: {case['database_mode']!r}")
    if case["phase"] not in {"context", "generation"}:
        raise ValueError(f"unsupported phase: {case['phase']!r}")
    for name in (
        "batch_size",
        "isl",
        "osl",
        "stride",
        "tp_size",
        "pp_size",
        "attention_dp_size",
        "moe_tp_size",
        "moe_ep_size",
    ):
        if not isinstance(case[name], int) or case[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(case["prefix"], int) or case["prefix"] < 0:
        raise ValueError("prefix must be a non-negative integer")
    return case


def _validate_common(payload: dict) -> tuple[int, int, str]:
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol_version: {payload.get('protocol_version')!r}")
    warmup = payload.get("warmup", 10)
    iterations = payload.get("iterations", 100)
    if not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    revision = payload.get("revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("revision must be a non-empty string")
    return warmup, iterations, revision


def validate_request(payload: dict) -> tuple[dict, int, int, str]:
    warmup, iterations, revision = _validate_common(payload)
    if "cases" in payload:
        raise ValueError("case and cases are mutually exclusive")
    return _validate_case(payload.get("case")), warmup, iterations, revision


def validate_batch_request(payload: dict) -> tuple[list[dict], int, int, str]:
    warmup, iterations, revision = _validate_common(payload)
    if "case" in payload:
        raise ValueError("case and cases are mutually exclusive")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("cases must be a non-empty array")
    validated = [_validate_case(case) for case in cases]
    case_ids = [case["case_id"] for case in validated]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case IDs must be unique")
    return validated, warmup, iterations, revision


def _environment() -> dict:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    thread_vars = {
        name: os.environ.get(name, "")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_affinity": affinity,
        "thread_env": thread_vars,
    }


def _response(case: dict, revision: str) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "revision": revision,
        "case_id": case["case_id"],
        "case_hash": canonical_case_hash(case),
        "case": case,
        "environment": _environment(),
    }


def _benchmark_case(case: dict) -> BenchmarkCase:
    return BenchmarkCase(
        model_path=case["model_path"],
        system_name=case["system_name"],
        backend_name=case["backend_name"],
        backend_version=case["backend_version"],
        batch_size=case["batch_size"],
        isl=case["isl"],
        osl=case["osl"],
        prefix=case["prefix"],
        tp_size=case["tp_size"],
        pp_size=case["pp_size"],
        attention_dp_size=case["attention_dp_size"],
        moe_tp_size=case["moe_tp_size"],
        moe_ep_size=case["moe_ep_size"],
    )


def _record_error(response: dict, exc: Exception) -> None:
    if isinstance(exc, PerfDataNotAvailableError):
        response.update(
            {
                "status": "DATA_MISS",
                "error": {"type": type(exc).__name__, "message": str(exc).splitlines()[0][:500]},
            }
        )
        return
    response.update(
        {
            "status": "INVALID",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc).splitlines()[0][:500],
                "traceback": traceback.format_exc(),
            },
        }
    )


def _group_key(case: dict) -> tuple:
    return tuple(case[name] for name in GROUP_KEYS)


def _run_case_group(cases: list[dict], *, warmup: int, iterations: int, revision: str) -> list[dict]:
    responses = [_response(case, revision) for case in cases]
    representative = _benchmark_case(cases[0])
    try:
        clear_caches(representative)
        ensure_rust_library_present()
        session_setup_ms, session, group_runtime_config = measure_session_setup_ms(
            representative,
            suppress_loader_output=True,
            database_mode=cases[0]["database_mode"],
            shared_layer=False,
        )
        phases = list(dict.fromkeys(case["phase"] for case in cases))
        with redirect_output(True):
            for phase in phases:
                phase_case = next(case for case in cases if case["phase"] == phase)
                phase_call(
                    session,
                    priming_runtime_config(group_runtime_config, phase=phase),
                    phase=phase,
                    stride=phase_case["stride"],
                )()
    except Exception as exc:
        for response in responses:
            _record_error(response, exc)
        return responses

    for case, response in zip(cases, responses, strict=True):
        try:
            runtime_config = replace(
                group_runtime_config,
                batch_size=case["batch_size"],
                isl=case["isl"],
                osl=case["osl"],
                prefix=case["prefix"],
            )
            call = phase_call(
                session,
                runtime_config,
                phase=case["phase"],
                stride=case["stride"],
            )
            with redirect_output(True):
                predicted_value, cold_us, warm_samples, warm_stats = measure_cold_and_warm(
                    call,
                    warmup=warmup,
                    iterations=iterations,
                )
            response.update(
                {
                    "status": "OK",
                    "predicted_value_ms": predicted_value,
                    "session_setup_ms": session_setup_ms,
                    "cold_definition": "steady_state_unseen_query",
                    "steady_state_setup_queries": len(phases),
                    "group_case_count": len(cases),
                    "cold_us": cold_us,
                    "warm_samples_us": warm_samples,
                    "warm": warm_stats,
                }
            )
        except Exception as exc:
            _record_error(response, exc)
    return responses


def _run_cases(cases: list[dict], *, warmup: int, iterations: int, revision: str) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for case in cases:
        groups.setdefault(_group_key(case), []).append(case)

    results_by_id = {}
    for group in groups.values():
        for result in _run_case_group(group, warmup=warmup, iterations=iterations, revision=revision):
            results_by_id[result["case_id"]] = result
    return [results_by_id[case["case_id"]] for case in cases]


def run_request(payload: dict) -> dict:
    case, warmup, iterations, revision = validate_request(payload)
    return _run_case_group([case], warmup=warmup, iterations=iterations, revision=revision)[0]


def run_batch_request(payload: dict) -> dict:
    cases, warmup, iterations, revision = validate_batch_request(payload)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "revision": revision,
        "results": _run_cases(cases, warmup=warmup, iterations=iterations, revision=revision),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, help="Read the request from a JSON file instead of stdin.")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    try:
        text = args.request.read_text() if args.request else sys.stdin.read()
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        response = run_batch_request(payload) if "cases" in payload else run_request(payload)
    except Exception as exc:
        print(f"invalid worker request: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(response, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
