# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from compare_e2e import (
    checked_requests,
    compare_cohort,
    observed_metrics,
    paired_summary,
    predicted_metrics,
    replay_spec,
    timing_provider,
)
from compare_forward import canonical


def test_fpm_selector_reaches_replay_without_becoming_an_arithmetic_override():
    config = dict(
        model_name="model",
        backend="vllm",
        system_name="gb200",
        backend_version="pinned",
        decoder_replay=False,
        database_mode="SILICON",
        systems_path="systems",
        forward_model="fpm",
        fpm_fmha_dtype="fp8",
    )
    provider = timing_provider(config)
    assert provider["fpm_fmha_dtype"] == "fp8"
    assert "fmha_dtype" not in provider
    assert timing_provider(config | {"activation_dtype": "bfloat16"})["fmha_dtype"] == "bfloat16"


def case(name="short", *, start=1000000000):
    tokens = [1, 2, 3]
    digest = hashlib.sha256(canonical(tokens).encode()).hexdigest()
    request = {
        "request_id": name,
        "input_token_ids": tokens,
        "input_token_ids_sha256": digest,
        "output_tokens": 2,
    }
    plan = {
        "cohort_id": name,
        "purpose": name,
        "trial_index": 0,
        "trial_seed": 123,
        "requests": [request],
    }
    client = {
        "cohort_id": name,
        "purpose": name,
        "trial_index": 0,
        "trial_seed": 123,
        "valid": True,
        "started": {"monotonic_ns": start},
        "finished": {"monotonic_ns": start + 10000000},
        "cache_control": {
            "policy": "native-clear-before-cohort",
            "acknowledgements": [{"status": "success"}],
        },
        "output_tokens_per_second": 200,
        "requests": [
            {
                "request_id": name,
                "valid": True,
                "input_token_ids_sha256": digest,
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "started_monotonic_ns": start + 100000,
                "ttft_ms": 4,
                "average_tpot_ms": 6,
                "exact_itl_available": True,
                "itl_ms": [6],
            }
        ],
    }
    return plan, client


def fixed_engine():
    return {
        "tensor_parallel_size": 4,
        "num_gpu_blocks_is_explicit": True,
        "rank": {
            "backend": "sglang",
            "num_gpu_blocks": 16,
            "block_size": 256,
            "max_num_seqs": 4,
            "max_num_batched_tokens": 2048,
            "timing_model": {"type": "fixed", "prefill_ms": 2, "decode_ms": 0.6},
        },
    }


def test_materializes_real_tokens_and_submit_offsets_without_observed_timing():
    plan, client = case()
    spec, start = replay_spec(fixed_engine(), plan, client)
    assert start == 0
    assert spec["requests"] == [
        {
            "id": "short",
            "input_tokens": 3,
            "input_token_ids": [1, 2, 3],
            "output_tokens": 2,
            "arrival_time_ms": 0.1,
        }
    ]
    client["requests"][0]["ttft_ms"] = 9999
    assert replay_spec(fixed_engine(), plan, client)[0] == spec


@pytest.mark.parametrize("mutation", ["hash", "missing_tokens", "usage", "duplicate", "bool_token"])
def test_client_and_real_token_mismatch_fail(mutation):
    plan, client = case()
    if mutation == "hash":
        client["requests"][0]["input_token_ids_sha256"] = "0" * 64
    elif mutation == "missing_tokens":
        plan["requests"][0]["input_token_ids"] = []
    elif mutation == "usage":
        client["requests"][0]["completion_tokens"] = 1
    elif mutation == "duplicate":
        client["requests"] *= 2
    else:
        plan["requests"][0]["input_token_ids"] = [True, 2, 3]
    with pytest.raises(ValueError):
        checked_requests(plan, client)


def test_prefix_seed_is_same_engine_and_excluded_from_target_metrics():
    seed, seed_client = case("prefix-warm-A")
    plan, client = case("prefix-reuse-B", start=2000000000)
    plan["seed_cohort_id"] = seed["cohort_id"]
    client["cache_control"] = {"policy": "preserve-A-prefix"}
    spec, offset = replay_spec(fixed_engine(), plan, client, seed_cohort=seed, seed_observed=seed_client)
    assert [r["id"] for r in spec["requests"]] == ["prefix-warm-A", "prefix-reuse-B"]
    assert offset == 1000
    assert spec["requests"][1]["arrival_time_ms"] == 1000.1
    with pytest.raises(ValueError, match="seed cohort"):
        replay_spec(fixed_engine(), plan, client)
    seed["trial_index"] = 1
    with pytest.raises(ValueError, match="seed cohort"):
        replay_spec(fixed_engine(), plan, client, seed_cohort=seed, seed_observed=seed_client)


def test_cold_requires_actual_acknowledgement():
    plan, client = case()
    client["cache_control"]["acknowledgements"] = [{"status": "failed"}]
    with pytest.raises(ValueError, match="acknowledgements"):
        replay_spec(fixed_engine(), plan, client)


def test_missing_native_prediction_remains_in_coverage():
    plan, client = case()

    def missing(_):
        raise ValueError("missing prefix table")

    result = compare_cohort(fixed_engine(), plan, client, missing)
    assert result["status"] == "prediction_unavailable"
    assert paired_summary([result], final=True)["short"] == {
        "coverage": {"planned_trials": 1, "predicted_trials": 0},
        "metrics": {},
    }


def test_native_submillisecond_provider_and_real_request_adapter():
    import aisimulate._runtime as native

    plan, client = case()
    result = compare_cohort(fixed_engine(), plan, client, native.run_replay_json)
    assert result["status"] == "predicted"
    # Existing SGLang replay charges one first-output decode after prefill.
    assert result["prediction"]["ttft_ms"] == pytest.approx(2.6)
    assert result["prediction"]["exact_itl_ms"] == pytest.approx(0.6)
    assert result["prediction"]["output_tokens_per_second"] == pytest.approx(2000 / 3.3)
    assert result["signed_error_percent"]["ttft_ms"] == pytest.approx(-35)


def test_actual_external_aic_provider_compiles_and_runs_sol(tmp_path):
    import shutil
    from pathlib import Path

    from compare_e2e import sglang_engine

    import aiconfigurator_core
    import aisimulate._runtime as native

    # Explicit empty overlay admits the pinned preview version without an
    # environment override. SOL needs the system specification, not timings.
    shutil.copyfile(
        Path(aiconfigurator_core.__file__).parent / "systems/gb300.yaml",
        tmp_path / "gb300.yaml",
    )
    (tmp_path / "data/gb300").mkdir(parents=True)
    config = {
        "model_name": "deepseek-ai/DeepSeek-V4.1-Flash",
        "system_name": "gb300",
        "backend": "sglang",
        "backend_version": "0.0.0.dev0",
        "database_mode": "SOL",
        "decoder_replay": False,
        "systems_path": str(tmp_path),
    }
    receipt = {
        "schema": "dsv41.serving.scheduler.receipt.v1",
        "actual_native_startup": {
            "page_size": 256,
            "total_kv_blocks": 16,
            "max_total_num_tokens": 4096,
            "max_running_requests": 4,
            "max_prefill_tokens": 16384,
            "chunked_prefill_size": 2048,
            "model_context_len": 1048576,
        },
        "configured_scheduler": {
            "tp_size": 4,
            "dp_size": 1,
            "moe_dp_size": 1,
            "pp_size": 1,
            "attn_cp_size": 1,
            "dcp_size": 1,
            "disable_radix_cache": False,
            "enable_mixed_chunk": False,
            "schedule_policy": "fcfs",
            "speculative_algorithm": None,
            "enable_decoder_swa_bounded_replay": False,
            "schedule_conservativeness": 1.0,
        },
    }
    plan, client = case()
    result = compare_cohort(sglang_engine(receipt, config), plan, client, native.run_replay_json)
    assert result["status"] == "predicted", result.get("failure")
    assert 0 < result["prediction"]["ttft_ms"] < 5
    assert 0 < result["prediction"]["average_tpot_ms"] < 1


@pytest.mark.parametrize("past_kv", [127, 128, 129])
def test_actual_fpm_replay_selector_preserves_decode_past_kv_boundary(past_kv):
    from pathlib import Path

    import aisimulate._runtime as native
    from aiconfigurator_core.sdk.rust_engine_step import RustForwardPassPerfModel

    calibration = Path(__file__).parent.parent / "gb200-fpm/calibration-v1"
    config = json.loads((calibration / "prediction-config.json").read_text())
    config["fpm_fmha_dtype"] = config.pop("activation_dtype")
    config["systems_path"] = str((calibration / "systems").resolve())
    query = past_kv + 1
    engine = fixed_engine()
    engine["rank"].update(
        backend="vllm",
        timing_model={"type": "external", "provider": "aic", "config": timing_provider(config)},
    )
    spec = {
        "version": 1,
        "topology": {"kind": "aggregated", "workers": {"initial_workers": 1}},
        "record_per_request": True,
        "engine": engine,
        "requests": [
            {
                "id": "boundary",
                "input_tokens": query,
                "input_token_ids": list(range(query)),
                "output_tokens": 1,
                "arrival_time_ms": 0.0,
            }
        ],
    }
    report = json.loads(native.run_replay_json(json.dumps(spec)))
    model = RustForwardPassPerfModel.from_native(config)
    pref = dict(num_prefill_requests=1, sum_prefill_tokens=query, sum_prefill_kv_tokens=0)
    dec = dict(num_decode_requests=1, sum_decode_kv_tokens=past_kv)
    expected = sum(
        model.estimate_forward_pass_time_ms({"version": 1, "wall_time": 1.0, "scheduled_requests": scheduled})
        for scheduled in (pref, dec)
    )
    assert report["completed_requests"] == 1
    # The current replay emits its first output after a decode iteration;
    # preserve that documented scheduler boundary while testing its KV axis.
    assert report["per_request"][0]["ttft_ms"] == pytest.approx(expected, abs=0.000001, rel=0)


def test_no_interval_for_partial_or_too_few_trials_and_paired_resampling():
    rows = [
        {
            "purpose": "short",
            "trial_index": i,
            "status": "predicted",
            "observed": {"ttft_ms": i + 1},
            "prediction": {"ttft_ms": 2 * (i + 1)},
        }
        for i in range(20)
    ]
    diagnostic = paired_summary(rows, final=False)["short"]["metrics"]["ttft_ms"]
    assert not any("ci95" in key for key in diagnostic)
    small = paired_summary(rows[:19], final=True)["short"]["metrics"]["ttft_ms"]
    assert not any("ci95" in key for key in small)
    final = paired_summary(rows, final=True)["short"]["metrics"]["ttft_ms"]
    assert final["paired_ratio_error_percent_bootstrap_ci95"] == [100, 100]
    with pytest.raises(ValueError, match="duplicate"):
        paired_summary(rows + rows[:1], final=True)


def test_cache_disagreement_is_reported_without_discarding_timing():
    from compare_e2e import attach_cache_comparison

    result = {
        "status": "predicted",
        "prediction": {"ttft_ms": 100},
        "requests": [{"request_id": "r", "reused_input_tokens": 256}],
    }
    audited = {"native_request_proof": {"requests": [{"request_id": "r", "initial_cross_request_cached_tokens": 0}]}}
    attach_cache_comparison(result, audited)
    assert result["cache_semantics_match"] is False
    assert result["prediction"] == {"ttft_ms": 100}
    assert result["requests"][0]["native_initial_cached_tokens"] == 0


def test_native_nonterminal_output_cannot_be_a_comparison():
    with pytest.raises(ValueError, match="complete"):
        predicted_metrics(
            {"per_request": [{"request_id": "r", "terminal_status": "cancelled"}]},
            ["r"],
            0,
        )


def test_no_seed_timing_injected_into_provider():
    plan, client = case()
    seen = []

    def capture(payload):
        seen.append(json.loads(payload))
        raise ValueError("probe only")

    compare_cohort(fixed_engine(), plan, client, capture)
    assert seen[0]["engine"]["rank"]["timing_model"] == fixed_engine()["rank"]["timing_model"]
    assert "ttft_ms" not in canonical(seen[0])


def test_actual_native_prefix_seed_reuses_two_pages():
    import aisimulate._runtime as native

    seed, seed_client = case("prefix-warm-A")
    plan, client = case("prefix-reuse-B", start=2000000000)
    for planned, observed, tokens in [
        (seed, seed_client, list(range(512))),
        (plan, client, list(range(768))),
    ]:
        digest = hashlib.sha256(canonical(tokens).encode()).hexdigest()
        planned["requests"][0].update(input_token_ids=tokens, input_token_ids_sha256=digest)
        observed["requests"][0].update(prompt_tokens=len(tokens), input_token_ids_sha256=digest)
    plan["seed_cohort_id"] = seed["cohort_id"]
    client["cache_control"] = {"policy": "preserve-A-prefix"}
    result = compare_cohort(
        fixed_engine(),
        plan,
        client,
        native.run_replay_json,
        seed_cohort=seed,
        seed_observed=seed_client,
    )
    assert result["status"] == "predicted"
    assert result["requests"][0]["reused_input_tokens"] == 512
    seed_client["cache_control"]["acknowledgements"] = [{"status": "failed"}]
    with pytest.raises(ValueError, match="acknowledged"):
        replay_spec(fixed_engine(), plan, client, seed_cohort=seed, seed_observed=seed_client)


def test_closed_measurement_binds_original_http_and_scheduler_files(tmp_path):
    from compare_e2e import qualify_e2e_sources
    from compare_forward import file_hash

    client, scheduler = tmp_path / "client.json", tmp_path / "scheduler.json"
    client.write_text('{"run_id":"same", "ttft_ms": 10}')
    scheduler.write_text('{"run_id":"same", "capacity": 16}')
    measurement = {
        "source_bindings": {
            "client_summary_file_sha256": file_hash(client),
            "scheduler_receipt_file_sha256": file_hash(scheduler),
        }
    }
    qualify_e2e_sources(measurement, client, scheduler)
    client.write_text('{"run_id":"same", "ttft_ms": 5}')
    with pytest.raises(ValueError, match="client_summary"):
        qualify_e2e_sources(measurement, client, scheduler)
    measurement["source_bindings"]["client_summary_file_sha256"] = file_hash(client)
    scheduler.write_text('{"run_id":"same", "capacity": 1000}')
    with pytest.raises(ValueError, match="scheduler_receipt"):
        qualify_e2e_sources(measurement, client, scheduler)


def pressure_audit():
    state = {
        "schema": "dsv41.cache.pressure.v1",
        "run_id": "run",
        "epoch_id": "epoch",
        "allocation_attempts": 3,
        "allocation_refusals": 0,
        "allocation_exceptions": 0,
        "preemptions": 0,
        "minimum_free_blocks": 50,
        "watermark_blocks": 5,
    }
    return {
        "producer_audits": [
            {
                "run_id": "run",
                "dispatch_audit": {
                    "cache_pressure": dict(state),
                    "records": [{"dispatch_id": 1, "cache_pressure": dict(state)}],
                },
            }
        ],
        "active_rows": [{"file": "trace", "line": 1, "dispatch_id": 1}],
    }, {"rows": [{"file": "trace", "line": 1}]}


@pytest.mark.parametrize(
    "field,value",
    [
        ("allocation_refusals", 1),
        ("allocation_exceptions", 1),
        ("preemptions", 1),
        ("minimum_free_blocks", 5),
        ("epoch_id", "other"),
        ("allocation_attempts", 0),
    ],
)
def test_native_pressure_rejects_unconstrained_prediction(field, value):
    from compare_e2e import capacity_eligibility

    audit, cohort = pressure_audit()
    assert capacity_eligibility(audit, cohort) is None
    audit["producer_audits"][0]["dispatch_audit"]["records"][0]["cache_pressure"][field] = value
    assert capacity_eligibility(audit, cohort) is not None


def test_missing_pressure_dispatch_does_not_become_zero_pressure():
    from compare_e2e import capacity_eligibility

    audit, cohort = pressure_audit()
    del audit["producer_audits"][0]["dispatch_audit"]["records"][0]["cache_pressure"]
    assert "missing" in capacity_eligibility(audit, cohort)


def test_vllm_capacity_is_frozen_workload_envelope_not_sum_of_physical_groups():
    from compare_e2e import vllm_engine

    receipt = {
        "schema": "dsv41.scheduler.resolved.v1",
        "scheduler_block_size_tokens": 256,
        "max_running_requests": 2,
        "max_scheduled_tokens": 512,
        "model": {"max_model_len": 2050, "enforce_eager": True},
        "speculative_config": None,
        "parallel": {
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "enable_expert_parallel": False,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1,
        },
        "scheduler": {"policy": "fcfs", "enable_chunked_prefill": True},
        "cache": {"enable_prefix_caching": True},
        "kv_pool": {"num_blocks": 50000, "groups": []},
    }
    config = {
        "model_name": "deepseek-ai/DeepSeek-V4.1-Flash",
        "system_name": "gb200",
        "backend": "vllm",
        "backend_version": "pinned",
        "decoder_replay": False,
        "database_mode": "SILICON",
        "systems_path": "overlay",
    }
    plan = {"cohorts": [case()[0]]}
    first = vllm_engine(receipt, config, plan)
    receipt["kv_pool"] = {
        "num_blocks": 99999,
        "groups": [{"block_size": 128}, {"block_size": 512}],
    }
    assert vllm_engine(receipt, config, plan) == first
    assert first["rank"]["max_num_seqs"] == 2 and first["rank"]["max_num_batched_tokens"] == 512
    assert first["rank"]["num_gpu_blocks"] == 3
    receipt["parallel"]["enable_expert_parallel"] = True
    with pytest.raises(ValueError, match="topology"):
        vllm_engine(receipt, config, plan)


def test_request_completion_and_last_token_are_distinct_from_cohort_duration():
    import aisimulate._runtime as native

    plan, client = case()
    client["requests"][0].update(latency_ms=11, last_token_latency_ms=10)
    assert observed_metrics(client)["request_latency_ms"] == 11
    assert observed_metrics(client)["last_token_latency_ms"] == 10
    result = compare_cohort(fixed_engine(), plan, client, native.run_replay_json)
    # Arrival is 0.1 ms after cohort start; latency excludes that offset.
    assert result["prediction"]["request_latency_ms"] == pytest.approx(3.2)
    assert result["prediction"]["last_token_latency_ms"] == pytest.approx(3.2)
    assert result["signed_error_percent"]["request_latency_ms"] == pytest.approx(100 * (3.2 / 11 - 1))
    assert result["signed_error_percent"]["last_token_latency_ms"] == pytest.approx(-68)
