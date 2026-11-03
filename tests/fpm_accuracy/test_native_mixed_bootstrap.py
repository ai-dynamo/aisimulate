# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduled mixed bootstrap survives unavailable native prior timing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import test_scenarios as scenario_fixtures

from scripts.fpm_accuracy.models.learned_scenario import LearnedScenarioPredictor
from scripts.fpm_accuracy.models.mixed_scenario import MixedOriginalA
from scripts.fpm_accuracy.scenarios import score_scenario

# Pages installs only the Gym dependencies. Native-wheel jobs exercise this test.
pytest.importorskip("aisimulate_core.sdk")
phase_fixtures = import_module("python.aisimulate.tests.unit.sdk.test_fpm_phase_readiness")
training_support = import_module("python.aisimulate.tests.unit.sdk.fpm_training_support")
training_support.import_sklearn()
phase_roots = phase_fixtures.phase_roots
scenario_case = scenario_fixtures.scenario_case
isolate_training_dependencies = training_support.isolate_training_dependencies

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("method", ["gbdt", "selector"])
def test_scheduled_missing_prior_errors_preserve_native_mixed_bootstrap(
    phase_roots, scenario_case, monkeypatch, method
):
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, fpm_learned
    from aisimulate_core.sdk.errors import PerfDataNotAvailableError

    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    config, make_root = phase_roots
    root = Path(make_root("original-a"))
    config.pop("fpm_profile")
    config.update(model="deepseek-ai/DeepSeek-V3")
    config["estimator_config"]["fpm_interpolation"] = {"method": "sol", "text_only": True}
    parquet = root / "data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet"
    rows = pq.read_table(parquet).to_pylist()
    for row in rows:
        row.update(
            model_path=config["model"],
            pp=1,
            cp=1,
            attention_backend="auto",
            moe_backend="auto",
            enable_wideep=False,
            enable_eplb=False,
            gemm_quant_mode="bfloat16",
            moe_quant_mode="bfloat16",
            fmha_quant_mode="bfloat16",
            kv_cache_dtype="bfloat16",
        )
    rows.append({**rows[1], "cell_id": "decode-upper", "batch_size": 4, "total_kv_read_tokens": 128, "latency_ms": 4.0})
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    sidecar = parquet.with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text())
    metadata["parquet_sha256"] = hashlib.sha256(parquet.read_bytes()).hexdigest()
    metadata["row_count"] = len(rows)
    sidecar.write_text(json.dumps(metadata))
    artifact = replace(
        scenario_case.fpm_artifacts[0],
        local_path=parquet,
        local_metadata_path=sidecar,
        sha256=metadata["parquet_sha256"],
        metadata_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        row_count=len(rows),
    )

    # Only pure stores exist in A. The prior can price their exact coordinates,
    # but the mixed stream needs a native MLA roofline missing from this runtime.
    seeds = []
    for kind in ("prefill", "decode"):
        for index in range(20):
            original = scenario_fixtures.observation(index, kind, actual=2.0 if kind == "prefill" else 3.0).iteration
            rank = original.ranks[0]
            scheduled = replace(
                rank.scheduled,
                sum_prefill_tokens=int(kind == "prefill"),
                sum_decode_kv_tokens=int(kind == "decode"),
            )
            seeds.append(replace(original, ranks=(replace(rank, scheduled=scheduled),)))
    stream = tuple(
        replace(scenario_fixtures.observation(index, "mixed", actual=8.0), event_time_ms=index * 2500)
        for index in range(43)
    )
    stream = tuple(
        replace(
            item,
            iteration=replace(
                item.iteration,
                ranks=(
                    replace(
                        item.iteration.ranks[0],
                        scheduled=replace(
                            item.iteration.ranks[0].scheduled, num_decode_requests=2, sum_prefill_tokens=1
                        ),
                    ),
                ),
            ),
        )
        for item in stream
    )
    case = replace(scenario_case, observations=stream, fpm_artifacts=(artifact,))
    updates, errors, appends, predictions = [], [], [], []
    native_tune = RustForwardPassPerfModel.tune_with_fpms
    native_append = RustForwardPassPerfModel.append_mixed_learned_checkpoint
    native_predict = RustForwardPassPerfModel.estimate_forward_pass_with_route

    def record_tune(model, iterations):
        updates.append(iterations[0][0]["counter_id"])
        try:
            return native_tune(model, iterations)
        except Exception as error:
            errors.append(error)
            raise

    def record_append(model, path):
        checkpoint = json.loads((Path(path) / "model.json").read_text())
        appends.append((len(updates), updates[-1], checkpoint["metadata"]["train_rows"]))
        return native_append(model, path)

    def record_predict(model, ranks):
        predictions.append((ranks[0]["counter_id"], len(appends)))
        return native_predict(model, ranks)

    monkeypatch.setattr(RustForwardPassPerfModel, "tune_with_fpms", record_tune)
    monkeypatch.setattr(RustForwardPassPerfModel, "append_mixed_learned_checkpoint", record_append)
    monkeypatch.setattr(RustForwardPassPerfModel, "estimate_forward_pass_with_route", record_predict)

    def factory(selected_method, context, *, training_iterations, online, mixed_fallback, scenario):
        assert (selected_method, online, mixed_fallback, scenario) == (method, True, True, "ac")
        engine_config = {"system_name": "h200_sxm", "backend": "vllm", "backend_version": "0.25.1"}
        mixed = MixedOriginalA(engine_config, context.fpm_artifact, scenario)
        request = ForwardPassPerfModelConfig(**{**config, "systems_paths": (str(mixed.prepared.systems_root),)})
        probe = RustForwardPassPerfModel.best_available(mixed.configure_request(request))
        try:
            assert probe.estimate_forward_pass_time_ms(seeds[0].prediction_input().aic_payload()) == 2.0
            assert probe.estimate_forward_pass_time_ms(seeds[-1].prediction_input().aic_payload()) == 3.0
            with pytest.raises(PerfDataNotAvailableError, match="SOL roofline unavailable"):
                probe.estimate_forward_pass_time_ms(stream[0].prediction_input().aic_payload())
        finally:
            probe.close()
        return LearnedScenarioPredictor(
            method, RustForwardPassPerfModel, request, fpm_learned, training_iterations, online=True, mixed=mixed
        )

    result = score_scenario(case, method, "ac", factory=factory, bootstrap=lambda *_: seeds)

    assert result["status"] == "evaluated"
    assert updates == list(range(2, 43, 2))  # Every five seconds; intervening labels never enter bootstrap.
    assert len(errors) == 21
    assert all(isinstance(error, PerfDataNotAvailableError) for error in errors), [str(error) for error in errors]
    assert all("SOL roofline unavailable" in str(error) for error in errors)
    assert appends == [(20, 40, {"contains_locally_mixed": 20})]
    assert predictions == [(index, int(index > 40)) for index in range(43)]
    schedule = result["online_schedule"]
    assert schedule["scheduled_update_count"] == schedule["submitted_observation_count"] == 21
    assert schedule["update_error_count"] == 21
    assert schedule["skipped_observation_count"] == 22
    assert schedule["warmup_update_count"] == result["online_update_count"] == 0
    assert result["metrics"]["all"]["tuning_error_count"] == result["metrics"]["mixed"]["tuning_error_count"] == 21
    assert result["metrics"]["all"]["predicted_count"] == 0  # Missing-prior prediction errors remain visible.
    assert result["metrics"]["all"]["error_count"] == 43
    scope = result["online_model"]["diagnostics"]["scopes"][0]
    assert scope["stores"] == ["contains_locally_mixed", "pure_decode", "pure_prefill"]
    assert scope["offline_observations"] == 40
    assert scope["submitted_observations"] == 21
    assert scope["bootstrap_observations"] == 20
    assert scope["store_additions"] == 1
    assert scope["pending_bootstrap"] == {}
    assert scope["feedback_observations"] == 0
    assert scope["online"]["stores"]["contains_locally_mixed"]["accepted_observations"] == 1
