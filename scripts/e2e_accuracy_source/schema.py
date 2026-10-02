# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SiliconRow:
    """One InferenceX (config, benchmark_row) pair joined on config_id.

    A single row corresponds to one measured (config_id, isl, osl, conc)
    tuple at one (date, image) point. After dedupe + coherence, exactly
    one SiliconRow remains per (config_id, isl, osl, conc).
    """

    # joined key
    config_id: int
    isl: int
    osl: int
    conc: int

    # config fields
    hardware: str
    framework: str
    silicon_model: str
    precision: str
    spec_method: str
    disagg: bool
    is_multinode: bool
    prefill_tp: int
    prefill_ep: int
    prefill_dp_attention: bool
    prefill_num_workers: int
    decode_tp: int
    decode_ep: int
    decode_dp_attention: bool
    decode_num_workers: int
    num_prefill_gpu: int
    num_decode_gpu: int

    # benchmark fields
    bench_id: str
    workflow_run_id: str
    date: str
    image: str | None
    metrics: dict[str, float]
    benchmark_type: str = "single_turn"
    server_log_id: str | None = None
    recipe_fingerprint: str | None = None

    # workflow_runs join. These fields identify the immutable InferenceX
    # source revision that produced the benchmark.
    github_run_id: str | None = None
    run_attempt: int | None = None
    head_sha: str | None = None
    head_branch: str | None = None
    workflow_url: str | None = None
    infx_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DropRecord:
    """A row dropped at some pipeline stage; aggregated into coverage.json."""

    stage: str
    reason: str
    config_id: int
    isl: int | None = None
    osl: int | None = None
    conc: int | None = None


@dataclass
class CliEstimateKwargs:
    """The kwargs passed to aiconfigurator.cli.api.cli_estimate().

    Mirrors the API surface; only the fields we actually populate are
    enumerated. The agg / disagg branches share the shared fields and
    populate the matching subset.
    """

    # shared
    model_path: str
    system_name: str
    backend_name: str
    mode: str  # "agg" | "disagg"
    isl: int
    osl: int
    database_mode: str = "SILICON"
    backend_version: str | None = None

    # Internal metadata, materialized by the runner and never forwarded as a CLI argument.
    model_config_snapshot: dict[str, Any] | None = None

    # quant overrides (None means let AISim auto-infer)
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    comm_quant_mode: str | None = None

    # Exact source capacity controls supported by cli_estimate.
    free_gpu_memory_fraction: float | None = None
    prefill_free_gpu_memory_fraction: float | None = None
    decode_free_gpu_memory_fraction: float | None = None
    max_seq_len: int | None = None
    prefill_max_seq_len: int | None = None
    decode_max_seq_len: int | None = None

    # agg-only
    batch_size: int | None = None
    tp_size: int | None = None
    pp_size: int | None = None
    attention_dp_size: int | None = None
    moe_tp_size: int | None = None
    moe_ep_size: int | None = None

    # disagg-only
    decode_system_name: str | None = None
    prefill_tp_size: int | None = None
    prefill_pp_size: int | None = None
    prefill_attention_dp_size: int | None = None
    prefill_moe_tp_size: int | None = None
    prefill_moe_ep_size: int | None = None
    prefill_batch_size: int | None = None
    prefill_num_workers: int | None = None
    decode_tp_size: int | None = None
    decode_pp_size: int | None = None
    decode_attention_dp_size: int | None = None
    decode_moe_tp_size: int | None = None
    decode_moe_ep_size: int | None = None
    decode_batch_size: int | None = None
    decode_num_workers: int | None = None

    def to_call_kwargs(self) -> dict[str, Any]:
        """Drop None fields so cli_estimate sees only what we actually set."""
        return {k: v for k, v in self.__dict__.items() if v is not None and k != "model_config_snapshot"}
