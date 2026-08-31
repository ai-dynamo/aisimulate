# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward-prediction setup and timing helpers for the advisory gate."""

from __future__ import annotations

import contextlib
import io
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, replace

from aiconfigurator.sdk import config, perf_database
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.models import get_model
from tools.prediction_regression_gate import grid

_perf_counter_ns = time.perf_counter_ns


@dataclass(frozen=True)
class BenchmarkCase:
    model_path: str
    system_name: str = "b200_sxm"
    backend_name: str = "vllm"
    backend_version: str = "0.24.0"
    batch_size: int = 1
    isl: int = 1024
    osl: int = 2
    prefix: int = 0
    tp_size: int = 8
    pp_size: int = 1
    attention_dp_size: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 8


@contextlib.contextmanager
def redirect_output(enabled: bool):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def build_session(
    case: BenchmarkCase,
    *,
    suppress_loader_output: bool,
    database_mode: str | None = None,
    shared_layer: bool | None = None,
) -> tuple[InferenceSession, config.RuntimeConfig]:
    with redirect_output(suppress_loader_output):
        if database_mode is None:
            database = perf_database.get_database(case.system_name, case.backend_name, case.backend_version)
        else:
            database = perf_database.get_database_view(
                case.system_name,
                case.backend_name,
                case.backend_version,
                database_mode=database_mode,
                shared_layer=shared_layer,
            )
        if database is None:
            raise RuntimeError(
                f"failed to load perf database for {case.system_name}/{case.backend_name}/{case.backend_version}"
            )
        backend = get_backend(case.backend_name)
        model_config = config.ModelConfig(
            tp_size=case.tp_size,
            pp_size=case.pp_size,
            attention_dp_size=case.attention_dp_size,
            moe_tp_size=case.moe_tp_size,
            moe_ep_size=case.moe_ep_size,
        )
        model = get_model(case.model_path, model_config, case.backend_name)
    runtime_config = config.RuntimeConfig(
        batch_size=case.batch_size,
        beam_width=1,
        isl=case.isl,
        osl=case.osl,
        prefix=case.prefix,
    )
    return InferenceSession(model, database, backend), runtime_config


def clear_caches(case: BenchmarkCase) -> None:
    """Reset prediction state through the public database eviction contract."""
    perf_database.unload_database(case.system_name, case.backend_name, case.backend_version)


def ensure_rust_library_present() -> None:
    # The compiled engine ships as the maturin-built ``aiconfigurator_core``
    # extension; importing it is the availability check.
    import aiconfigurator_core  # noqa: F401


def percentile(samples: list[float], value: float) -> float:
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * value / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("at least one timed sample is required")
    return {
        "call_mean_us": statistics.fmean(samples),
        "call_median_us": statistics.median(samples),
        "call_p50_us": percentile(samples, 50),
        "call_p90_us": percentile(samples, 90),
        "call_p99_us": percentile(samples, 99),
        "call_min_us": min(samples),
        "call_max_us": max(samples),
        "iterations": float(len(samples)),
    }


def time_calls(
    call: Callable[[], float],
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        call()

    samples = []
    for _ in range(iterations):
        start = _perf_counter_ns()
        call()
        samples.append((_perf_counter_ns() - start) / 1000.0)
    return samples


def measure_cold_and_warm(
    call: Callable[[], float],
    *,
    warmup: int,
    iterations: int,
) -> tuple[float, float, list[float], dict[str, float]]:
    """Time one unseen query after reusable state is initialized, then repeats."""
    start = _perf_counter_ns()
    predicted_value = call()
    cold_us = (_perf_counter_ns() - start) / 1000.0
    warm_samples = time_calls(call, warmup=warmup, iterations=iterations)
    return predicted_value, cold_us, warm_samples, summarize_samples(warm_samples)


def priming_runtime_config(
    runtime_config: config.RuntimeConfig,
    *,
    phase: str,
) -> config.RuntimeConfig:
    """Return one supported off-matrix query that initializes a prediction phase."""
    return replace(
        runtime_config,
        batch_size=2,
        isl=2048,
        osl={"context": grid.CTX_OSL, "generation": grid.GEN_OSL}[phase],
        prefix=0,
    )


def phase_call(
    session: InferenceSession,
    runtime_config: config.RuntimeConfig,
    *,
    phase: str,
    stride: int = 1,
) -> Callable[[], float]:
    mode = {"context": "static_ctx", "generation": "static_gen"}[phase]
    runtime = replace(runtime_config, engine_step_backend="rust")

    def call() -> float:
        return session.run_static_latency_only(runtime, mode=mode, stride=stride)

    return call


def measure_session_setup_ms(
    case: BenchmarkCase,
    *,
    suppress_loader_output: bool,
    database_mode: str | None = None,
    shared_layer: bool | None = None,
) -> tuple[float, InferenceSession, config.RuntimeConfig]:
    start = _perf_counter_ns()
    session, runtime_config = build_session(
        case,
        suppress_loader_output=suppress_loader_output,
        database_mode=database_mode,
        shared_layer=shared_layer,
    )
    return (_perf_counter_ns() - start) / 1_000_000.0, session, runtime_config
