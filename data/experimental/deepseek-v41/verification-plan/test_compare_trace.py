# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from compare_trace import (
    CHECKPOINT_SHA,
    PRODUCERS,
    RUNTIMES,
    compare_cohorts,
    compare_interval,
    digest,
    independent_trial_summary,
    qualify_inputs,
    qualify_prediction_config,
    systems_identity,
    trace_summary,
)


def trial_rows(count=20):
    return [
        {
            "purpose": "short",
            "trial_index": i,
            "trial_seed": 9000 + i,
            "intervals": [{"status": "predicted", "observed_ms": 10.0, "predicted_ms": 11.0 if i % 2 else 9.0}],
        }
        for i in range(count)
    ]


def test_trace_uncertainty_keeps_correlated_intervals_in_their_independent_trial():
    rows = trial_rows()
    first = independent_trial_summary(rows, final=True, resamples=200)["short"]
    for row in rows:
        row["intervals"] *= 100
    repeated = independent_trial_summary(rows, final=True, resamples=200)["short"]
    assert repeated == first
    assert first["fully_predicted_trials"] == 20
    assert first["interval_wape_percent"] == pytest.approx(10)
    assert first["whole_trial_bootstrap_ci95"]["mean_trial_total_forward_signed_error_percent"][0] < 0
    assert first["whole_trial_bootstrap_ci95"]["mean_trial_total_forward_signed_error_percent"][1] > 0


@pytest.mark.parametrize("reason", ["diagnostic", "short", "missing_prediction"])
def test_trace_final_interval_requires_complete_independent_trials(reason):
    rows = trial_rows(19 if reason == "short" else 20)
    if reason == "missing_prediction":
        rows[0]["intervals"].append({"status": "prediction_unavailable", "observed_ms": 20.0})
    result = independent_trial_summary(rows, final=reason != "diagnostic")["short"]
    assert "whole_trial_bootstrap_ci95" not in result
    if reason == "missing_prediction":
        assert result["fully_predicted_trials"] == 19
        assert result["observed_trials"] == 20
        assert result["missing_prediction_trial_indices"] == [0]


@pytest.mark.parametrize("field", ["trial_index", "trial_seed"])
def test_trace_duplicate_trial_identity_cannot_inflate_confidence(field):
    rows = trial_rows()
    rows[1][field] = rows[0][field]
    with pytest.raises(ValueError, match="duplicate independent trace"):
        independent_trial_summary(rows, final=True)


def test_trace_interval_wape_does_not_cancel_opposite_errors_within_a_trial():
    rows = trial_rows()
    for row in rows:
        row["intervals"] = [
            {"status": "predicted", "observed_ms": 10.0, "predicted_ms": 12.0},
            {"status": "predicted", "observed_ms": 10.0, "predicted_ms": 8.0},
        ]
    result = independent_trial_summary(rows, final=True, resamples=200)["short"]
    assert result["mean_trial_total_forward_signed_error_percent"] == 0
    assert result["interval_wape_percent"] == 20


def fixture(*, backend="sglang", replay=False, diagnostic=False):
    system, version, semantics = RUNTIMES[backend]
    plan = {
        "sampling_role": "main",
        "dataset_role": "verification",
        "run_id": "public-run",
        "requested_trials": 1,
        "cohorts": [],
    }
    active, cohorts = [], []
    for index in range(2):
        case = {
            "cohort_id": f"c{index}",
            "trial_index": 0,
            "trial_seed": 9000,
            "purpose": f"decode-{index}",
            "comparison_role": "primary",
            "requests": [
                {
                    "request_id": f"request-{index}",
                    "input_token_ids": [1, 2, 3],
                    "input_token_ids_sha256": digest([1, 2, 3]),
                    "output_tokens": 1,
                }
            ],
        }
        plan["cohorts"].append(case)
        if index == 1 and diagnostic:
            continue
        rid = f"native-{index}"
        row = {
            "file": "trace.jsonl.gz",
            "line": index + 1,
            "dispatch_id": index + 1,
            "fpm": {
                "version": 1,
                "worker_id": "worker",
                "dp_rank": 0,
                "counter_id": index,
                "wall_time": 0.01,
                "scheduled_requests": {
                    "num_prefill_requests": 0,
                    "num_decode_requests": 1,
                    "sum_prefill_tokens": 0,
                    "sum_prefill_kv_tokens": 0,
                    "sum_decode_kv_tokens": 4 if backend == "sglang" else 3,
                    "var_prefill_length": 0.0,
                    "var_decode_kv_tokens": 0.0,
                },
            },
            "dispatch_requests": [
                {
                    "rid": rid,
                    "phase": "decode",
                    "inclusive_context_tokens": 4,
                    "past_kv_tokens": 3,
                    "input_token_ids_sha256": digest([1, 2, 3]),
                    "prompt_tokens": 3,
                    "max_new_tokens": 1,
                }
            ],
        }
        active.append(row)
        cohorts.append(
            {
                "cohort_id": case["cohort_id"],
                "rows": [{k: row[k] for k in ("file", "line")}],
                "native_request_proof": {
                    "requests": [
                        {
                            "request_id": f"request-{index}",
                            "native_rid": rid,
                            "prefill_new_tokens": 3,
                            "initial_cross_request_cached_tokens": 0,
                            "required_decode_iterations": 0,
                        }
                    ],
                    "iteration_roles": {
                        str(index + 1): "unreturned_overlap_output" if index else "useful_request_work"
                    },
                },
            }
        )
    count = len(active)
    producer = {
        "run_id": plan["run_id"],
        "worker_id": "worker",
        "dp_rank": 0,
        "module": PRODUCERS[backend][0],
        "source_sha256": PRODUCERS[backend][1],
        "observer_sha256": "a" * 64,
        "errors": [],
        "thread_alive_at_shutdown": False,
        **dict.fromkeys(("complete", "run_exited", "shutdown_called", "dispatch_audit_required"), True),
        **dict.fromkeys(("queue_full", "send_again", "send_error", "publish_suppressed"), 0),
        **dict.fromkeys(
            ("attempted", "enqueued", "dequeued", "active_sent", "sent", "sequence_allocated", "send_attempted"), count
        ),
        "dispatch_audit": {
            "started": count,
            "completed": count,
            "pending": [],
            "records": [
                {"dispatch_id": r["dispatch_id"], "requests": deepcopy(r["dispatch_requests"])} for r in active
            ],
        },
    }
    audit = {
        "valid": True,
        "errors": [],
        "producer_audits": [producer],
        "active_rows": active,
        "cohorts": cohorts,
        "files": [{"file": "trace.jsonl.gz", "sha256": "b" * 64}],
    }
    if diagnostic:
        audit.update(
            complete_requested_study=False,
            qualification_scope="closed physical lifecycle",
            original_combined_summary_sha256="c" * 64,
        )
    measurement = {
        "run_id": plan["run_id"],
        "backend": backend,
        "system_name": system,
        "backend_version": version,
        "model_name": "deepseek-ai/DeepSeek-V4.1-Flash",
        "tp_size": 4,
        "moe_ep_size": 1,
        "decoder_replay": replay,
        "execution_mode": "eager",
        "producer_semantics": semantics,
        "runtime_digest": "sha256:" + "d" * 64,
        "model_config_canonical_sha256": CHECKPOINT_SHA,
        "producer_source_sha256": PRODUCERS[backend][1],
        "observer_source_sha256": "a" * 64,
        "source_bindings": dict.fromkeys(
            ("execution_file_sha256", "worker_config_file_sha256", "measurement_builder_sha256", "audit_tool_sha256"),
            "e" * 64,
        ),
    }
    bind(audit, plan, measurement)
    return audit, plan, measurement


def bind(audit, plan, measurement):
    measurement["source_bindings"].update(audit_canonical_sha256=digest(audit), plan_canonical_sha256=digest(plan))


def config(measurement):
    return {
        k: measurement[k] for k in ("model_name", "system_name", "backend", "backend_version", "tp_size", "moe_ep_size")
    } | {
        "moe_tp_size": 4,
        "decoder_replay": measurement["decoder_replay"],
        "database_mode": "SILICON",
        "enable_shared_layer": False,
        "strict_provenance": True,
        "systems_path": "systems",
    }


def test_complete_main_and_explicit_diagnostic_have_different_coverage():
    audit, plan, measurement = fixture()
    assert set(qualify_inputs(audit, plan, measurement)) == {"c0", "c1"}
    with pytest.raises(ValueError, match="explicit closed-segment"):
        qualify_inputs(audit, plan, measurement, diagnostic=True)
    audit, plan, measurement = fixture(diagnostic=True)
    with pytest.raises(ValueError, match="incomplete closed segment"):
        qualify_inputs(audit, plan, measurement)
    assert set(qualify_inputs(audit, plan, measurement, diagnostic=True)) == {"c0"}
    assert len(plan["cohorts"]) == 2 and plan["requested_trials"] == 1


@pytest.mark.parametrize(
    "mutation", ["audit", "plan", "missing_source", "observer", "version", "pp", "precision", "dp_bool"]
)
def test_source_and_execution_mutations_reject(mutation):
    audit, plan, measurement = fixture()
    if mutation == "audit":
        audit["active_rows"][0]["fpm"]["wall_time"] = 0.02
    elif mutation == "plan":
        plan["cohorts"][0]["trial_seed"] += 1
    elif mutation == "missing_source":
        measurement["source_bindings"].pop("audit_tool_sha256")
    elif mutation == "observer":
        measurement["observer_source_sha256"] = "0" * 64
    elif mutation == "version":
        measurement["backend_version"] = "0.5.14"
    elif mutation == "pp":
        measurement["pp_size"] = 2
    elif mutation == "precision":
        measurement["model_config_canonical_sha256"] = "0" * 64
    else:
        measurement["attention_dp_size"] = True
    with pytest.raises(ValueError):
        qualify_inputs(audit, plan, measurement)


@pytest.mark.parametrize(
    "mutation", ["drop", "pending", "missing_row", "duplicate_row", "unbound_trace", "bad_request", "budget"]
)
def test_closure_reconciliation_rejects_even_with_refreshed_payload_hash(mutation):
    audit, plan, measurement = fixture()
    if mutation == "drop":
        audit["producer_audits"][0]["queue_full"] = 1
    elif mutation == "pending":
        audit["producer_audits"][0]["dispatch_audit"]["pending"] = [3]
    elif mutation == "missing_row":
        audit["active_rows"].pop()
    elif mutation == "duplicate_row":
        audit["active_rows"][1] = deepcopy(audit["active_rows"][0])
    elif mutation == "unbound_trace":
        audit["files"][0]["file"] = "another.gz"
    elif mutation == "bad_request":
        audit["cohorts"][0]["native_request_proof"]["requests"][0]["prefill_new_tokens"] += 1
    else:
        plan["requested_trials"] = 2
    bind(audit, plan, measurement)
    with pytest.raises(ValueError):
        qualify_inputs(audit, plan, measurement)


@pytest.mark.parametrize("backend", ["sglang", "vllm"])
@pytest.mark.parametrize("mode", ["SOL", "HYBRID", "SILICON"])
def test_qualified_prediction_contract(backend, mode):
    _, _, measurement = fixture(backend=backend)
    qualify_prediction_config(config(measurement) | {"database_mode": mode}, measurement)


@pytest.mark.parametrize(
    "mutation", [None, "missing_override", "wrong_override", "missing_receipt", "wrong_native_identity", "op_level"]
)
def test_gb200_fpm_fp8_identity_requires_independent_native_receipt(mutation):
    _, _, measurement = fixture(backend="vllm")
    measurement["fmha_quant_mode"] = "fp8"
    measurement["source_bindings"]["fmha_identity_receipt_sha256"] = "f" * 64
    cfg = config(measurement) | {"forward_model": "fpm", "activation_dtype": "fp8"}
    if mutation == "missing_override":
        cfg.pop("activation_dtype")
    elif mutation == "wrong_override":
        cfg["activation_dtype"] = "bf16"
    elif mutation == "missing_receipt":
        measurement["source_bindings"].pop("fmha_identity_receipt_sha256")
    elif mutation == "wrong_native_identity":
        measurement["fmha_quant_mode"] = "bfloat16"
    elif mutation == "op_level":
        cfg["forward_model"] = "op_level"
    if mutation:
        with pytest.raises(ValueError):
            qualify_prediction_config(cfg, measurement)
    else:
        qualify_prediction_config(cfg, measurement)


@pytest.mark.parametrize("mutation", ["short_scenario", "duplicate_index", "duplicate_seed", "mixed_corpus"])
def test_each_trace_scenario_requires_its_own_complete_independent_budget(mutation):
    audit, plan, measurement = fixture()
    if mutation == "mixed_corpus":
        plan["corpus_role"] = "primary"
        plan["cohorts"][0]["corpus_role"] = "primary"
        plan["cohorts"][1]["corpus_role"] = "field-notes"
    elif mutation == "duplicate_index":
        plan["cohorts"][1]["purpose"] = plan["cohorts"][0]["purpose"]
    else:
        plan["requested_trials"] = 2
        plan["cohorts"][1]["trial_index"] = 1
        if mutation == "duplicate_seed":
            plan["cohorts"][1]["purpose"] = plan["cohorts"][0]["purpose"]
    bind(audit, plan, measurement)
    with pytest.raises(ValueError, match="(independent trial budget|corpus stratum)"):
        qualify_inputs(audit, plan, measurement)


@pytest.mark.parametrize(
    "mutation",
    [
        {"decoder_replay": True},
        {"kv_cache_dtype": "fp8"},
        {"cp_size": 2},
        {"nextn": 3},
        {"enable_shared_layer": True},
        {"strict_provenance": False},
        {"backend_version": "latest"},
        {"extra_ignored_knob": True},
    ],
)
def test_prediction_does_not_change_profile_precision_or_lookup_policy(mutation):
    _, _, measurement = fixture()
    with pytest.raises(ValueError):
        qualify_prediction_config(config(measurement) | mutation, measurement)


def test_vllm_replay_is_not_silently_generalized():
    audit, plan, measurement = fixture(backend="vllm", replay=True)
    with pytest.raises(ValueError, match="decoder replay"):
        qualify_inputs(audit, plan, measurement)


@pytest.mark.parametrize(
    "backend,target,expected",
    [("sglang", "op_level", 4), ("sglang", "fpm", 3), ("vllm", "op_level", 4), ("vllm", "fpm", 3)],
)
def test_native_axes_are_normalized_once_without_observed_timing(backend, target, expected):
    audit, _, _ = fixture(backend=backend)
    original = deepcopy(audit["active_rows"][0])

    def predictor(metrics):
        assert "wall_time" not in metrics
        assert metrics["scheduled_requests"]["sum_decode_kv_tokens"] == expected
        return 10.0

    row = compare_interval(original, predictor, backend=backend, forward_model=target, decoder_replay=False)
    assert row["predicted_ms"] == 10 and row["signed_error_percent"] == 0
    assert original == audit["active_rows"][0]


def test_equal_query_with_different_prefix_is_missing_for_bounded_even_when_variance_zero():
    audit, _, _ = fixture()
    row = audit["active_rows"][0]
    row["dispatch_requests"] = [
        {"rid": "a", "phase": "prefill", "query_tokens": 128, "prefix_tokens": 0},
        {"rid": "b", "phase": "prefill", "query_tokens": 128, "prefix_tokens": 512},
    ]
    row["fpm"]["scheduled_requests"].update(
        num_decode_requests=0,
        sum_decode_kv_tokens=0,
        num_prefill_requests=2,
        sum_prefill_tokens=256,
        sum_prefill_kv_tokens=512,
    )

    def must_not_predict(_):
        raise AssertionError("heterogeneous bounded inputs must not reach native averaging")

    result = compare_interval(row, must_not_predict, backend="sglang", forward_model="op_level", decoder_replay=True)
    assert result["status"] == "prediction_unavailable" and result["failure_type"] == "ValueError"
    assert trace_summary([result])["planned_points"] == 1


def test_all_attributed_overlap_work_is_retained_and_missing_predictions_not_zeroed():
    audit, plan, measurement = fixture()
    cohorts = qualify_inputs(audit, plan, measurement)

    def predictor(_):
        raise ValueError("missing table /private/owned/table.parquet")

    rows = compare_cohorts(
        audit, plan, cohorts, predictor, backend="sglang", forward_model="op_level", decoder_replay=False
    )
    assert len(rows) == 2 and rows[1]["intervals"][0]["native_work_role"] == "unreturned_overlap_output"
    intervals = [r for cohort in rows for r in cohort["intervals"]]
    assert trace_summary(intervals) == {"planned_points": 2, "predicted_points": 0}
    assert all("predicted_ms" not in row and "/private" not in row["failure"] for row in intervals)


def test_invalid_native_axis_and_aggregate_are_data_errors_not_missing_predictions():
    audit, _, _ = fixture()
    row = audit["active_rows"][0]
    row["dispatch_requests"][0]["inclusive_context_tokens"] = 5
    with pytest.raises(ValueError, match="decode contexts"):
        compare_interval(row, lambda _: 1.0, backend="sglang", forward_model="op_level", decoder_replay=False)


def test_system_inventory_hashes_actual_files_and_rejects_escape(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "gb200.yaml").write_text("data_dir: data\n")
    table = tmp_path / "data" / "table.parquet"
    table.write_bytes(b"first")
    cfg = {"system_name": "gb200", "systems_path": str(tmp_path)}
    first = systems_identity(cfg)
    table.write_bytes(b"changed")
    assert systems_identity(cfg)["inventory_sha256"] != first["inventory_sha256"]
    table.unlink()
    table.symlink_to(tmp_path / "gb200.yaml")
    with pytest.raises(ValueError, match="symlink"):
        systems_identity(cfg)


@pytest.mark.parametrize("forward_model,prediction_variance", [("op_level", 0.0), ("fpm", 65536.0)])
def test_vllm_original_prompt_variance_is_preserved_and_explicitly_bridged(forward_model, prediction_variance):
    audit, _, _ = fixture(backend="vllm")
    row = audit["active_rows"][0]
    row["dispatch_requests"] = [
        {"rid": "a", "phase": "prefill", "query_tokens": 128, "prefix_tokens": 0, "prompt_tokens": 128},
        {"rid": "b", "phase": "prefill", "query_tokens": 128, "prefix_tokens": 512, "prompt_tokens": 640},
    ]
    row["fpm"]["scheduled_requests"].update(
        num_decode_requests=0,
        sum_decode_kv_tokens=0,
        num_prefill_requests=2,
        sum_prefill_tokens=256,
        sum_prefill_kv_tokens=512,
        var_prefill_length=65536.0,
    )
    original = deepcopy(row)

    def predict(metrics):
        assert metrics["scheduled_requests"]["var_prefill_length"] == prediction_variance
        return 10.0

    result = compare_interval(row, predict, backend="vllm", forward_model=forward_model, decoder_replay=False)
    assert result["predicted_ms"] == 10.0
    assert result["native_scheduled_requests"]["var_prefill_length"] == 65536.0
    assert result["variance_bridge"]["current_query_variance"] == 0.0
    assert result["variance_bridge"]["native_prefill_axis"] == "original_prompt_tokens"
    assert row == original
    row["fpm"]["scheduled_requests"]["var_prefill_length"] = 0.0
    with pytest.raises(ValueError, match="variance differ"):
        compare_interval(row, predict, backend="vllm", forward_model=forward_model, decoder_replay=False)


@pytest.mark.parametrize("mutation", ["duplicate", "geometry", "missing"])
def test_closed_native_snapshots_bind_every_dispatch(mutation):
    audit, plan, measurement = fixture()
    records = audit["producer_audits"][0]["dispatch_audit"]["records"]
    if mutation == "duplicate":
        records[1] = deepcopy(records[0])
    elif mutation == "geometry":
        records[0]["requests"][0]["inclusive_context_tokens"] += 1
    else:
        records[0]["requests"] = []
    bind(audit, plan, measurement)
    with pytest.raises(ValueError, match="dispatch"):
        qualify_inputs(audit, plan, measurement)


@pytest.mark.parametrize("mutation", ["same_length_content", "native_token_hash", "native_rid", "output_limit"])
def test_frozen_token_content_is_bound_to_every_native_dispatch(mutation):
    audit, plan, measurement = fixture()
    row = audit["active_rows"][0]
    if mutation == "same_length_content":
        request = plan["cohorts"][0]["requests"][0]
        request["input_token_ids"] = [3, 2, 1]
        request["input_token_ids_sha256"] = digest(request["input_token_ids"])
    elif mutation == "native_token_hash":
        row["dispatch_requests"][0]["input_token_ids_sha256"] = "f" * 64
    elif mutation == "native_rid":
        audit["cohorts"][0]["native_request_proof"]["requests"][0]["native_rid"] = "another-request"
    else:
        row["dispatch_requests"][0]["max_new_tokens"] += 1
    audit["producer_audits"][0]["dispatch_audit"]["records"][0]["requests"] = deepcopy(row["dispatch_requests"])
    bind(audit, plan, measurement)
    with pytest.raises(ValueError, match="token content"):
        qualify_inputs(audit, plan, measurement)
