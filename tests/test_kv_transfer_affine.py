# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Affine payload geometry and canonical materialization (CPU-only)."""

import json

import pytest

from aisimulate.capacity import materialize_aic_num_gpu_blocks
from aisimulate.config.engine import KvTransferConfig
from aisimulate_core.sdk import RustForwardPassPerfModel
from aisimulate_core.sdk.config_builders import build_model_config
from aisimulate_core.sdk.models import get_model

MODEL = "deepseek-ai/DeepSeek-V4-Pro"


def _config(tp=1, dp=8, ep=8, dtype="fp8"):
    return {
        "model": MODEL,
        "system": "b300_sxm",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "worker_type": "prefill",
        "tp": tp,
        "pp": 1,
        "attention_dp": dp,
        "moe_tp_size": tp if dp == 1 else 1,
        "moe_ep_size": ep,
        "kvcache_quant_mode": dtype,
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
    }


@pytest.mark.parametrize("tp,dp,ep", [(1, 8, 8), (4, 1, 1), (8, 1, 1)])
@pytest.mark.parametrize("dtype,expected", [("fp8", (4444, 21479424)), ("bfloat16", (8408, 25477120))])
def test_v4_affine_envelope_and_precision(tp, dp, ep, dtype, expected):
    model = get_model(
        MODEL,
        build_model_config(
            tp_size=tp,
            pp_size=1,
            attention_dp_size=dp,
            moe_tp_size=tp if dp == 1 else 1,
            moe_ep_size=ep,
            kvcache_quant_mode=dtype,
        ),
        "vllm",
    )
    a, b = model.get_kv_transfer_affine_bytes()
    assert (a, b) == expected
    for length in [0, 1, 64, 127, 128, 129, 511, 512, 4096, 32768]:
        resident = model.get_kvcache_rank_bytes_per_sequence(length)
        assert a * length + b >= resident
        if length >= 128 and length % 128 == 0:
            assert a * length + b == resident
    assert model.get_kvcache_rank_bytes_per_sequence(0) == 17481728


@pytest.mark.parametrize("tp,dp,ep", [(1, 8, 8), (4, 1, 1), (8, 1, 1)])
def test_canonical_model_exposes_v4_geometry(tp, dp, ep):
    model = RustForwardPassPerfModel.best_available(_config(tp, dp, ep))
    try:
        assert model.kv_transfer_geometry() == {
            "bytes_per_token": 4444,
            "bytes_per_request": 21479424,
            "source": "deepseek_v4_affine_upper_envelope",
        }
    finally:
        model.close()


def test_explicit_capacity_still_materializes_transfer_and_round_trips():
    raw = {
        "num_gpu_blocks": 8192,
        "kv_transfer_bandwidth": 100.0,
        "timing_model": {"type": "external", "provider": "aic", "config": _config()},
    }
    resolved = materialize_aic_num_gpu_blocks(raw)
    assert resolved["num_gpu_blocks"] == 8192
    assert resolved["kv_transfer_bytes_per_token"] == 4444
    assert resolved["kv_transfer_bytes_per_request"] == 21479424
    assert materialize_aic_num_gpu_blocks(json.loads(json.dumps(resolved))) == resolved
    overridden = materialize_aic_num_gpu_blocks({**raw, "kv_transfer_bytes_per_request": 7})
    assert overridden["kv_transfer_bytes_per_token"] == 4444
    assert overridden["kv_transfer_bytes_per_request"] == 7
    legacy = materialize_aic_num_gpu_blocks({**raw, "kv_transfer_bytes_per_token": 123})
    assert legacy["kv_transfer_bytes_per_token"] == 123
    assert legacy.get("kv_transfer_bytes_per_request") is None
    alias = materialize_aic_num_gpu_blocks({**raw, "kv_bytes_per_token": 123})
    assert alias["kv_transfer_bytes_per_token"] == 123
    assert alias.get("kv_transfer_bytes_per_request") is None
    assert "kv_bytes_per_token" not in alias
    with pytest.raises(ValueError, match="duplicate field"):
        materialize_aic_num_gpu_blocks({**raw, "kv_bytes_per_token": 123, "kv_transfer_bytes_per_token": 456})
    omitted = materialize_aic_num_gpu_blocks({**raw, "kv_transfer_bandwidth": None})
    assert omitted.get("kv_transfer_bytes_per_token") is None
    assert omitted.get("kv_transfer_bytes_per_request") is None


def test_public_affine_override_is_strict_and_round_trips():
    transfer = KvTransferConfig(bytes_per_token=123, bytes_per_request=456, bandwidth_gb_per_second=100)
    assert KvTransferConfig.model_validate_json(transfer.model_dump_json()) == transfer
    for bad in [-1, True, 1.5, 1 << 64]:
        with pytest.raises(ValueError):
            KvTransferConfig(bytes_per_request=bad)
    for bad in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            KvTransferConfig(bandwidth_gb_per_second=bad)


def _public_config(trace, regime, bandwidth, *, custom_decode=False):
    from aisimulate.config.cli import CorePredictionConfig

    def worker(role):
        tp, dp, ep = (1, 8, 8) if regime == "HT" else (4 if role == "prefill" else 8, 1, 1)
        return {
            "parallelism": {
                "replicas": 1,
                "tensor": tp,
                "attention_data": dp,
                "moe_tensor": tp if dp == 1 else 1,
                "moe_expert": ep,
            },
            "scheduler": {"max_batched_tokens": 8192, "max_sequences": 256},
            "kv_cache": {
                "block_size": 64,
                "capacity": {"type": "fixed", "blocks": 8192},
            },
            "timing": (
                {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
                if custom_decode and role == "decode"
                else {"kvcache_quant_mode": "fp8"}
            ),
        }

    engine = {
        "mode": "disaggregated",
        "model": MODEL,
        "hardware": "b300_sxm",
        "backend": "vllm",
        "backend_version": "0.24.0",
        "context_length": 65536,
        "workers": {role: worker(role) for role in ("prefill", "decode")},
    }
    if bandwidth is not None:
        engine["kv_transfer"] = {
            "bandwidth_gb_per_second": bandwidth,
            "timing_mode": "full_prompt",
        }
    return CorePredictionConfig.model_validate(
        {
            "engine": engine,
            "traffic": {
                "source": {
                    "type": "trace",
                    "format": "mooncake",
                    "paths": [str(trace)],
                },
                "load": {"type": "trace_timestamps"},
            },
        }
    )


@pytest.mark.parametrize("regime,custom_decode", [("HT", False), ("LC", False), ("LC", True)])
@pytest.mark.parametrize("native_auto", [False, True])
def test_real_v4_replay_bandwidth_scaling_and_native_auto(tmp_path, regime, custom_decode, native_auto):
    from aisimulate import _runtime
    from aisimulate.compiler import prediction_to_replay_spec
    from aisimulate.runner import EngineReplayRunnerFactory
    from aisimulate.sweeper import ReplayOutputRequirements

    lengths = [64, 512, 4096, 32768]
    trace = tmp_path / "four.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": i * 1_000_000,
                    "input_length": length,
                    "output_length": 4,
                    "hash_ids": [i * 1000 + k for k in range((length + 63) // 64)],
                }
            )
            for i, length in enumerate(lengths)
        )
        + "\n"
    )

    class NativeAuto:
        def run_replay_json(self, raw):
            execution = json.loads(raw)
            if native_auto:
                # Bypass Python-derived bytes to exercise direct native AIS
                # model resolution as a second independent integration path.
                rank = execution.get("spec", execution)["engine"]["prefill"]["rank"]
                rank.pop("kv_transfer_bytes_per_token", None)
                rank.pop("kv_transfer_bytes_per_request", None)
            return _runtime.run_replay_json(json.dumps(execution))

    observed = {}
    for bandwidth in [None, 100.0, 1.0]:
        spec = prediction_to_replay_spec(_public_config(trace, regime, bandwidth, custom_decode=custom_decode))
        if bandwidth is not None:
            assert spec.backend_deployment.decode_engine_args.get("kv_transfer_bandwidth") is None
        report = (
            EngineReplayRunnerFactory(runtime=NativeAuto())
            .create(0)
            .run(
                spec,
                output_requirements=ReplayOutputRequirements(capture_per_request=True),
            )
        )
        rows = sorted(
            report.metadata["native_report"]["per_request"],
            key=lambda row: row["arrival_time_ms"],
        )
        assert len(rows) == 4
        observed[bandwidth] = rows
        for length, row in zip(lengths, rows, strict=True):
            expected_ms = 0 if bandwidth is None else (length * 4444 + 21479424) / (bandwidth * 1_000_000)
            span = row["destination_activated_ms"] - row["destination_reserved_ms"]
            assert span == pytest.approx(expected_ms, abs=1e-7)
    for baseline, fast, slow in zip(observed[None], observed[100.0], observed[1.0], strict=True):
        fast_delta = fast["ttft_ms"] - baseline["ttft_ms"]
        slow_delta = slow["ttft_ms"] - baseline["ttft_ms"]
        assert fast_delta > 0
        assert fast_delta == pytest.approx(fast["destination_activated_ms"] - fast["destination_reserved_ms"], abs=1e-6)
        assert slow_delta == pytest.approx(fast_delta * 100, abs=1e-6)
