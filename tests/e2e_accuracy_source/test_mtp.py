# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from scripts.e2e_accuracy.source.filter import apply_filter_rules
from scripts.e2e_accuracy.source.mtp import (
    acceptance,
    parse_runtime,
    runtime_inputs,
    validate_mtp,
)
from scripts.e2e_accuracy.source.replay import engine_args


def test_mtp_is_opt_in_without_mutating_source():
    rows = [
        SimpleNamespace(
            hardware="h200",
            framework="sglang",
            benchmark_type="single_turn",
            spec_method=method,
            metrics={"mean_tpot": 1},
            config_id=i,
            isl=1,
            osl=2,
            conc=3,
        )
        for i, method in enumerate(("none", "mtp", "draft_model"))
    ]
    assert apply_filter_rules(rows)[0] == rows[:1]
    assert apply_filter_rules(rows, spec_method="mtp")[0] == rows[1:2]
    assert rows[1].spec_method == "mtp"


@pytest.mark.parametrize("depth,accepted", [(0, 0), (6, 2), (True, 1), (2, -1), (2, 3), (2, float("nan"))])
def test_invalid_mtp_inputs(depth, accepted):
    with pytest.raises(ValueError):
        validate_mtp(dict(depth=depth, accepted_draft_tokens=accepted, kind="observed_counters"))


def test_synthetic_length_excludes_base_token():
    spec = dict(
        method="mtp",
        num_speculative_tokens=3,
        rejection_sample_method="synthetic",
        synthetic_acceptance_length=2.49,
    )
    result = acceptance({"speculative_config": spec}, "", "vllm")
    assert result["accepted_draft_tokens"] == pytest.approx(1.49)
    assert result["kind"] == "configured_synthetic"


def test_unconditional_rates_sum_and_counters_are_weighted():
    spec = dict(
        method="mtp",
        num_speculative_tokens=2,
        rejection_sample_method="synthetic",
        synthetic_acceptance_rates=[0.8, 0.2],
    )
    assert acceptance({"speculative_config": spec}, "", "vllm")["accepted_draft_tokens"] == 1
    spec.pop("rejection_sample_method")
    text = "Accepted: 9 tokens, Drafted: 10 tokens\nAccepted: 10 tokens, Drafted: 100 tokens"
    assert acceptance({"speculative_config": spec}, text, "vllm")["accepted_draft_tokens"] == pytest.approx(38 / 110)


def test_sglang_reconstruction_is_labeled():
    args = dict(
        speculative_algorithm="EAGLE",
        speculative_eagle_topk=1,
        speculative_num_steps=2,
        speculative_num_draft_tokens=3,
    )
    result = acceptance(args, "accept len: 2.0\naccept len: 2.5", "sglang")
    assert result["accepted_draft_tokens"] == 1.25
    assert result["kind"] == "observed_log_samples"
    assert result["samples"] == 2
    args["speculative_eagle_topk"] = 2
    with pytest.raises(ValueError, match="tree"):
        acceptance(args, "", "sglang")


def test_threshold_is_not_acceptance_and_external_eagle_is_not_mtp():
    with pytest.raises(ValueError, match="missing acceptance"):
        acceptance(
            {
                "speculative_config": dict(
                    method="mtp",
                    num_speculative_tokens=2,
                    acceptance_length_threshold=1,
                )
            },
            "",
            "vllm",
        )
    with pytest.raises(ValueError, match="unsupported speculation scheme"):
        acceptance(
            {"speculative_config": dict(method="eagle3", num_speculative_tokens=2)},
            "",
            "vllm",
        )


def test_runtime_parser_does_not_execute_objects(tmp_path):
    marker = tmp_path / "not-created"
    text = (
        "non-default args: {'tensor_parallel_size': 4, "
        f"'extra': __import__('pathlib').Path('{marker}').touch(), "
        "'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 2}, "
        "'attention': Config(backend=<Backend.FOO: 'foo'>)}"
    )
    parsed = parse_runtime(text, "vllm")
    assert parsed["tensor_parallel_size"] == 4
    assert parsed["speculative_config"]["num_speculative_tokens"] == 2
    assert not marker.exists()


def test_trt_nested_dtype_is_not_target_dtype():
    text = (
        "model='org/model' dtype='auto' kv_cache_config=KvCacheConfig(dtype='fp8', tokens_per_block=32) "
        "speculative_config=MTPDecodingConfig(max_draft_len=2, decoding_type='MTP') "
        "telemetry_config=TelemetryConfig(disabled=False) garbage_collection_gen0_threshold=20000 max_batch_size=8"
    )
    result = parse_runtime(text, "trtllm")
    assert result["dtype"] == "auto"
    assert result["kv_cache_config"]["dtype"] == "fp8"
    assert result["speculative_config"]["max_draft_len"] == 2


def test_disaggregated_prefill_preserves_mtp_identity_from_same_workload():
    config = (
        "server_args=ServerArgs(model_path='org/model', speculative_algorithm='EAGLE', "
        "speculative_eagle_topk=1, speculative_num_steps=2, speculative_num_draft_tokens=3)"
    )
    parsed = ("org/model", {"prefill": {}, "decode": {}}, "0.5.14", "sglang", {}, {})
    logs = {"prefill": [(config, {})], "decode": [(config + "\naccept len: 2.25", {})]}
    _, mtp = runtime_inputs({}, parsed, logs)
    assert mtp["prefill"]["depth"] == mtp["decode"]["depth"] == 2
    assert mtp["prefill"]["accepted_draft_tokens"] == 1.25
    assert mtp["prefill"]["acceptance_source_role"] == "decode"
    logs["prefill"] = [
        (
            config.replace("num_steps=2", "num_steps=1").replace("num_draft_tokens=3", "num_draft_tokens=2"),
            {},
        )
    ]
    with pytest.raises(ValueError, match="depths differ"):
        runtime_inputs({}, parsed, logs)


def test_mtp_controls_reach_both_scheduler_and_timing_inputs():
    role = dict(
        topology=dict(tp=1, pp=1, attention_dp=1, moe_tp=1, moe_ep=1),
        quantization=dict(gemm="bf16", moe="bf16"),
        args=dict(kv_cache_dtype="bf16"),
        mtp=dict(depth=2, accepted_draft_tokens=1.25),
    )
    deployment = dict(
        backend="sglang",
        system="h200_sxm",
        model_path="test/model",
        roles={"aggregated": role},
    )
    result = engine_args(deployment, "aggregated", "0.5.14")
    assert result["aic_nextn"] == 2
    assert result["aic_nextn_accepted"] == 1.25
    assert result["aic_forward_model"] == "op_level"


@pytest.mark.parametrize("names,wrong_id", [(["server.log", "other.log"], False), (["server.log"], True)])
def test_logs_require_unambiguous_bundle_and_matching_benchmark(monkeypatch, names, wrong_id):
    from scripts.e2e_accuracy.source import mtp

    def read(route, params, cache):
        return (names if route == "server-log-files" else {"id": 99, "serverLog": "text"}), "https://example.com"

    monkeypatch.setattr(mtp, "read_api", read)
    with pytest.raises(ValueError, match="identity mismatch" if wrong_id else "ambiguous"):
        mtp.log_files({"benchmark": {"id": 1}, "config": {"disagg": False}}, None)


def test_worker_speculation_must_agree():
    parsed = ("org/model", {"decode": {}, "prefill": {}}, "1.0", "sglang", {}, {})
    config = (
        "server_args=ServerArgs(model_path='org/model', speculative_algorithm='EAGLE', "
        "speculative_eagle_topk=1, speculative_num_steps=2, speculative_num_draft_tokens=3)"
    )
    with pytest.raises(ValueError, match="workers disagree"):
        runtime_inputs({}, parsed, {"decode": [(config, {}), (config.replace("EAGLE", "NEXTN"), {})]})
