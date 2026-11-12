# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical interpolation transport, exact selected sources, and cleanup."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.fpm_accuracy.exceptions import ConfigurationError, DependencyError
from scripts.fpm_accuracy.models import aic_fpm_database, scenario_interpolation, scenario_predictors
from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext


@pytest.fixture
def backend(monkeypatch, tmp_path):
    artifacts = {name: SimpleNamespace(artifact_id=name, sha256=name + "-hash") for name in ("warmup", "nowarmup")}
    events = []
    prepared = SimpleNamespace(
        systems_root=tmp_path,
        diagnostics={"fpm_parquet_sha256": "source-hash"},
        close=lambda: events.append("close-prepared"),
    )
    model = SimpleNamespace(
        diagnostics=lambda: {
            "provenance": {
                "config": {
                    "systems_paths": ["/private/tmp/native-profile"],
                    "estimator_config": {
                        "fpm_interpolation": {
                            "method": "sol",
                            "text_only": True,
                            "fpm_parquet_path": "/tmp/table.parquet",
                        }
                    },
                }
            }
        },
        close=lambda: events.append("close-native"),
    )

    def migrate(engine, role, options):
        events.append(("migrate", deepcopy(engine), role, deepcopy(options)))
        return SimpleNamespace(estimator_config={})

    def configure(request, artifact, database):
        assert database is prepared
        events.append(("configure", artifact))
        request.estimator_config = {"fpm_interpolation": {"method": "sol", "text_only": True}}
        return request

    def construct(request):
        events.append(("best_available", request))
        return model

    native = SimpleNamespace(normalize_config=lambda request: request, best_available=construct)
    config_type = SimpleNamespace(from_legacy_engine_config=migrate)
    worker = SimpleNamespace(config=SimpleNamespace(parallelism=SimpleNamespace(decode_context_parallel_size=1)))
    context = PredictorContext(worker=worker, worker_role="aggregated", options={"min_observations": 7})
    monkeypatch.setattr(scenario_predictors.aic_predictors, "_import_aisim_forward_pass_perf_model", lambda: native)
    monkeypatch.setattr(scenario_predictors.aic_predictors, "_canonical_config_type", lambda: config_type)
    monkeypatch.setattr(
        scenario_predictors,
        "map_worker_config_to_aic",
        lambda worker, overrides: {"model_name": "Org/Model", "extra": {}},
    )
    monkeypatch.setattr(aic_fpm_database, "prepare_aic_fpm_database", lambda engine, artifact: prepared)
    monkeypatch.setattr(scenario_interpolation, "configure_scenario_interpolation", configure)
    return SimpleNamespace(
        context=context, artifacts=artifacts, native=native, model=model, events=events, prepared=prepared
    )


@pytest.mark.parametrize("method", ["warmup", "nowarmup"])
def test_interpolation_uses_exact_selected_library_and_canonical_native_model(backend, method):
    from dataclasses import replace

    context = replace(backend.context, fpm_artifact=backend.artifacts[method])
    predictor = scenario_predictors.create_scenario_predictor(method, context)
    migrate, configure, construct = backend.events
    assert migrate[1]["systems_path"] == str(backend.prepared.systems_root)
    assert migrate[1]["forward_model"] == "fpm"
    assert migrate[2:] == ("aggregated", {"min_observations": 7})
    assert configure == ("configure", backend.artifacts[method])
    assert construct[0] == "best_available"
    assert construct[1].estimator_config["fpm_interpolation"] == {"method": "sol", "text_only": True}
    assert context.options == {"min_observations": 7}
    diagnostics = predictor.diagnostics()
    assert diagnostics["interpolation"] == {
        "method": "sol",
        "timing_only": False,
        "text_only": True,
        "metadata_recovered": False,
    }
    assert "/private/tmp" not in str(diagnostics) and "/tmp/table.parquet" not in str(diagnostics)
    predictor.close()
    assert backend.events[-2:] == ["close-native", "close-prepared"]


@pytest.mark.parametrize("stage", ["identity", "native"])
def test_interpolation_construction_failure_preserves_error_and_cleans_up(backend, monkeypatch, stage):
    from dataclasses import replace

    def fail(*args):
        raise ValueError("source identity or native model rejected")

    if stage == "identity":
        monkeypatch.setattr(scenario_interpolation, "configure_scenario_interpolation", fail)
    else:
        backend.native.best_available = fail
    context = replace(backend.context, fpm_artifact=backend.artifacts["warmup"])
    with pytest.raises(ValueError, match="source identity or native model rejected"):
        scenario_predictors.create_scenario_predictor("warmup", context)
    assert backend.events[-1] == "close-prepared"
    assert "close-native" not in backend.events


def test_interpolation_rejects_missing_source_or_unsupported_dcp_before_staging(backend):
    with pytest.raises(ConfigurationError, match="explicit HF FPM artifact"):
        scenario_predictors.create_scenario_predictor("warmup", backend.context)
    backend.context.worker.config.parallelism.decode_context_parallel_size = 2
    with pytest.raises(DependencyError, match="dcp"):
        scenario_predictors.create_scenario_predictor("warmup", backend.context)
    assert not backend.events


@pytest.mark.parametrize(
    "message",
    [
        "perf database error: perf_interp: no data to anchor query {batch_size=4} (no site within max_site_distance)",
        "FPM prefill query outside the collected domain",
        "FPM direct prefill: no same-batch, two-sided KV bracket whose token curves cover this query",
        "FPM direct prefill: no two-sided covered batch bracket within one capture regime",
    ],
)
def test_interpolation_support_misses_are_unavailable_without_changing_estimator(backend, monkeypatch, message):
    import sys
    from dataclasses import replace

    class PerfDataNotAvailableError(RuntimeError):
        pass

    monkeypatch.setitem(
        sys.modules, "aisimulate_core.sdk.errors", SimpleNamespace(PerfDataNotAvailableError=PerfDataNotAvailableError)
    )

    def estimate(rows):
        raise PerfDataNotAvailableError(message)

    backend.model.estimate_forward_pass_time_ms = estimate
    context = replace(backend.context, fpm_artifact=backend.artifacts["warmup"])
    predictor = scenario_predictors.create_scenario_predictor("warmup", context)
    try:
        result = predictor.predict(SimpleNamespace(aic_payload=lambda: []))
        assert result.value_ms is None
        assert result.metadata["reason"] == "unsupported_interpolation_query"
        assert len([event for event in backend.events if event[0] == "best_available"]) == 1
    finally:
        predictor.close()


@pytest.mark.parametrize(
    "typed,message",
    [
        (True, "FPM parquet hash mismatch"),
        (False, "perf_interp: no data to anchor query"),
        (
            True,
            "perf_interp: no data to anchor query; SOL roofline unavailable: "
            "no Rust implementation for context_kda_conv1d",
        ),
        (True, "perf_interp: no data to anchor query; operation not implemented"),
    ],
)
def test_interpolation_does_not_hide_corruption_capability_or_untyped_failures(backend, monkeypatch, typed, message):
    import sys
    from dataclasses import replace

    class PerfDataNotAvailableError(RuntimeError):
        pass

    monkeypatch.setitem(
        sys.modules, "aisimulate_core.sdk.errors", SimpleNamespace(PerfDataNotAvailableError=PerfDataNotAvailableError)
    )
    kind = PerfDataNotAvailableError if typed else ValueError

    def estimate(rows):
        raise kind(message)

    backend.model.estimate_forward_pass_time_ms = estimate
    context = replace(backend.context, fpm_artifact=backend.artifacts["warmup"])
    predictor = scenario_predictors.create_scenario_predictor("warmup", context)
    try:
        with pytest.raises(kind, match=message):
            predictor.predict(SimpleNamespace(aic_payload=lambda: []))
    finally:
        predictor.close()
