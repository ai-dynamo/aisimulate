# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MTP through real Weka import and the installed native execution boundary."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from aisimulate import EngineReplayRunnerFactory, ReplayOutputRequirements, _runtime
from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig

pytestmark = [pytest.mark.integration, pytest.mark.pre_merge, pytest.mark.gpu_0]

_WEKA = Path(__file__).parent / "e2e/configs/unified_cli/fixtures/traces/weka-relative.json"
_MODEL = "nvidia/GLM-5.2-NVFP4"


@pytest.fixture
def burst_weka(tmp_path):
    source = json.loads(_WEKA.read_text())

    def extend(requests):
        for request in requests:
            if request["type"] == "s":
                request["out"] = 8
            elif "requests" in request:
                extend(request["requests"])

    extend(source["requests"])
    path = tmp_path / "burst-weka.json"
    path.write_text(json.dumps(source))
    return path


def _prediction(path, backend, topology):
    worker = {
        "parallelism": {"tensor": 8, "moe_tensor": 8},
        "scheduler": {"max_batched_tokens": 8192, "max_sequences": 8},
        "kv_cache": {"block_size": 64, "capacity": {"type": "fixed", "blocks": 4096}},
    }
    return {
        "engine": {
            "mode": topology,
            "model": _MODEL,
            "backend": backend,
            "hardware": "b300_sxm",
            "backend_version": "0.5.14" if backend == "sglang" else "0.24.0",
            "context_length": 262144,
            "speculation": {
                "kind": "mtp",
                "num_speculative_tokens": 2,
                "expected_accepted_tokens": 2,
                "seed": 73,
            },
            "workers": {
                role: deepcopy(worker)
                for role in (("aggregated",) if topology == "aggregated" else ("prefill", "decode"))
            },
        },
        "traffic": {
            "source": {
                "type": "trace",
                "format": "weka",
                "paths": [str(path)],
                "block_size": 4,
            },
            "load": {"type": "trace_timestamps", "agentic_lanes": 1},
        },
    }


def _run_prediction(raw):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
    runner = EngineReplayRunnerFactory().create(0)
    try:
        return runner.run(spec, output_requirements=ReplayOutputRequirements(include_raw_report=True))
    finally:
        runner.close()


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
def test_real_weka_mtp_preserves_dag_outputs_and_target_identity(burst_weka, backend, topology):
    report = _run_prediction(_prediction(burst_weka, backend, topology))
    assert report.metrics["completed_requests"] == 4
    assert report.metrics["total_output_tokens"] == 32
    assert report.metadata["agentic_model_projection"]["target_model"] == _MODEL
    assert report.metadata["agentic_qualification"] == "functional_only"
    assert report.metadata["speculative_acceptance"]["mean_accept_length"] == 3
    assert report.metadata["speculative_acceptance"]["sampling_population"] == "measurement_completed_decode_passes"
    assert all(role["resolved_method"] == "mtp" for role in report.metadata["speculation"].values())


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("topology", ["aggregated", "disaggregated"])
def test_real_weka_one_token_tails_keep_prefill_outside_acceptance(backend, topology):
    report = _run_prediction(_prediction(_WEKA, backend, topology))
    assert report.metrics["completed_requests"] == report.metrics["total_output_tokens"] == 4
    acceptance = report.metadata["speculative_acceptance"]
    if topology == "aggregated":
        # Aggregated prefill supplies the only visible token without decode.
        assert acceptance["decode_forwards"] == acceptance["accepted_tokens_including_base"] == 0
        assert acceptance["mean_accept_length"] is None
    else:
        # P/D hands off prompt state, then decode verifies the first token.
        assert acceptance["decode_forwards"] == 4
        assert acceptance["mean_accept_length"] == 3
        assert acceptance["accepted_tokens_including_base"] == 12


def _native_payload(path, *, model=_MODEL, external=False):
    rank = {
        "backend": "sglang" if external else "vllm",
        "block_size": 4,
        "num_gpu_blocks": 128,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 64,
        "aic_nextn": 2,
        "aic_nextn_accept_rates": "1,1",
        "aic_mtp_seed": 73,
        "timing_model": {"type": "fixed", "prefill_ms": 1, "decode_ms": 1},
    }
    if external:
        rank["timing_model"] = {
            "type": "external",
            "provider": "aic",
            "config": {
                "model": model,
                "backend": "sglang",
                "system": "b300_sxm",
                "backend_version": "0.5.14",
                "tp": 8,
                "moe_tp_size": 8,
                "moe_ep_size": 1,
                "nextn": 2,
            },
        }
    return {
        "spec": {
            "version": 1,
            "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
            "engine": {"rank": rank, "tensor_parallel_size": 8 if external else 1},
            "requests": [],
        },
        "traffic": {
            "source_type": "trace",
            "load_type": "trace_timestamps",
            "trace_format": "weka",
            "trace_path": str(path),
            "trace_block_size": 4,
            "agentic_lanes": 1,
            "execution_model": model,
        },
    }


def test_native_legacy_nextn_accepts_glm_and_rejects_dspark_target(burst_weka):
    report = json.loads(_runtime.run_replay_json(json.dumps(_native_payload(burst_weka, external=True))))
    assert report["completed_requests"] == 4
    assert report["speculative_acceptance"]["mean_accept_length"] == 3
    payload = _native_payload(burst_weka, external=True, model="moonshotai/Kimi-K3")
    with pytest.raises(RuntimeError, match="DSPARK"):
        _runtime.run_replay_json(json.dumps(payload))


@pytest.mark.parametrize(
    "override,error",
    [
        (
            {
                "native_host_offload": {"num_host_blocks": 8},
                "kv_cache_bytes_per_token": 16,
            },
            "HBM-only",
        ),
        (
            {
                "prefix_match_unit": 4,
                "state_cache": {"bytes_per_request": 16},
                "kv_cache_bytes_per_token": 16,
            },
            "prefix_match_unit does not support speculative decoding",
        ),
        (
            {
                "enable_prefix_caching": False,
                "kv_cache_capacity_bytes": 1024,
                "kv_cache_groups": [
                    {
                        "name": "window",
                        "kind": "attention",
                        "num_layers": 4,
                        "block_size_tokens": 4,
                        "page_size_bytes": 64,
                        "sliding_window": 8,
                    }
                ],
            },
            "grouped cache",
        ),
    ],
)
def test_native_agentic_mtp_retains_cache_restrictions(override, error):
    payload = _native_payload(_WEKA, external="kv_cache_groups" in override)
    payload["spec"]["engine"]["rank"].update(override)
    with pytest.raises(RuntimeError, match=error):
        _runtime.run_replay_json(json.dumps(payload))


def test_native_agentic_mtp_requires_explicit_capacity():
    payload = _native_payload(_WEKA)
    payload["spec"]["engine"]["num_gpu_blocks_is_explicit"] = False
    with pytest.raises(RuntimeError, match="explicit fixed KV capacity"):
        _runtime.run_replay_json(json.dumps(payload))


def test_native_explicit_mtp_rejects_fpm_before_profile_lookup():
    payload = _native_payload(_WEKA, external=True)
    config = payload["spec"]["engine"]["rank"]["timing_model"]["config"]
    config.pop("nextn")
    config.update(
        forward_model="fpm",
        speculation={"kind": "mtp", "params": {"num_speculative_tokens": 2}},
    )
    with pytest.raises(RuntimeError, match="op_level"):
        _runtime.run_replay_json(json.dumps(payload))


@pytest.mark.parametrize("rates", [None, "", "  "])
def test_native_agentic_requires_authored_acceptance(rates):
    payload = _native_payload(_WEKA, external=True)
    payload["spec"]["engine"]["rank"]["aic_nextn_accept_rates"] = rates
    with pytest.raises(RuntimeError, match="explicit acceptance"):
        _runtime.run_replay_json(json.dumps(payload))


@pytest.mark.parametrize(
    "timing",
    [{"type": "fixed", "prefill_ms": 1, "decode_ms": 1}, {"type": "polynomial"}],
)
def test_native_agentic_mtp_requires_modeled_draft_cost(timing):
    payload = _native_payload(_WEKA)
    payload["spec"]["engine"]["rank"]["timing_model"] = timing
    with pytest.raises(RuntimeError, match="AIC op_level timing"):
        _runtime.run_replay_json(json.dumps(payload))


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
def test_fractional_acceptance_and_seed_through_public_replay(tmp_path, backend):
    source = json.loads(_WEKA.read_text())

    def extend(requests):
        for request in requests:
            if request["type"] == "s":
                request["out"] = 1024
            elif "requests" in request:
                extend(request["requests"])

    extend(source["requests"])
    path = tmp_path / "long-weka.json"
    path.write_text(json.dumps(source))
    raw = _prediction(path, backend, "aggregated")
    raw["engine"]["speculation"]["expected_accepted_tokens"] = 1.5
    reports = []
    for seed in (73, 73, 74):
        raw["engine"]["speculation"]["seed"] = seed
        report = _run_prediction(raw)
        acceptance = report.metadata["speculative_acceptance"]
        assert report.metrics["completed_requests"] == 4
        assert report.metrics["total_output_tokens"] == 4096
        assert acceptance["decode_forwards"] > 1500
        # Draft count alternates between one and two; with >1500 samples,
        # 0.05 is wider than 3.8 standard errors of the Bernoulli mean.
        assert acceptance["mean_accept_length"] == pytest.approx(2.5, abs=0.05)
        reports.append((acceptance, report.metrics["duration_ms"]))
    assert reports[0] == reports[1]
    assert reports[0] != reports[2]
