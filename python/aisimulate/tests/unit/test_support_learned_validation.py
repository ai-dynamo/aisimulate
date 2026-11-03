# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Learned validation is opt-in and preserves the accepted deployment."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from aisimulate.support import serving_validation as serving
from aisimulate.support import timing, validation

from .sdk.fpm_training_support import import_sklearn
from .sdk.fpm_training_support import isolate_training_dependencies as isolate_training_dependencies
from .test_support_serving_validation import materialized_workload  # noqa: F401
from .test_support_serving_validation import serving_case as _serving_case  # noqa: F401
from .test_support_validation import validation_case as _replay_validation_case  # noqa: F401

pytestmark = pytest.mark.unit


@pytest.fixture
def validation_case(request):
    return request.getfixturevalue("_replay_validation_case")


def _learned_config(plan, output):
    prediction = yaml.safe_load((plan / "predict/pilot.yaml").read_text())
    prediction["engine"]["workers"]["aggregated"]["timing"].update(
        estimation_mode="fpm_learned",
        fallback_policy="deny",
        estimator_config={
            "correction": {"enabled": False},
            "fpm_learned": {"checkpoint_path": str(output.parent / "seed")},
        },
    )
    output.write_text(yaml.safe_dump(prediction))
    return prediction


def test_optional_config_preserves_learned_controls_and_original_plan(validation_case, tmp_path):
    _args, request, plan, trace, _output = validation_case
    original = (plan / "predict/pilot.yaml").read_bytes()
    selected = tmp_path / "learned.yaml"
    authored = _learned_config(plan, selected)
    learned = validation._prediction_config(request, plan, trace, selected)
    assert learned["engine"] == authored["engine"]
    assert timing.canonical_timing_config(learned)["estimation_mode"] == "fpm_learned"
    assert serving._scope(learned, trace)["estimation_mode"] == "fpm_learned"
    direct = validation._prediction_config(request, plan, trace)
    assert timing.selected_timing_mode(direct) == "fpm_interpolation"
    assert (plan / "predict/pilot.yaml").read_bytes() == original


@pytest.mark.parametrize("omit_timing", [False, True], ids=["default-timing", "omitted-timing"])
@pytest.mark.parametrize("omit_mode", [False, True], ids=["explicit-mode", "default-mode"])
def test_optional_config_preserves_engine_controls_when_worker_inherits(
    validation_case, tmp_path, omit_timing, omit_mode
):
    _args, request, plan, trace, _output = validation_case
    selected = tmp_path / "learned.yaml"
    authored = _learned_config(plan, selected)
    engine = authored["engine"]
    worker_timing = engine["workers"]["aggregated"]["timing"]
    for field in ("estimation_mode", "fallback_policy", "estimator_config"):
        engine[field] = worker_timing.pop(field)
    if omit_timing:
        del engine["workers"]["aggregated"]["timing"]
    if omit_mode:
        del engine["mode"]
    selected.write_text(yaml.safe_dump(authored))
    learned = validation._prediction_config(request, plan, trace, selected)
    assert learned["engine"] == engine
    assert timing.canonical_timing_config(learned)["estimation_mode"] == "fpm_learned"
    assert serving._scope(learned, trace)["estimation_mode"] == "fpm_learned"


@pytest.mark.parametrize("field", ["hardware", "scheduler", "profile", "precision", "estimator"])
def test_optional_config_rejects_changed_deployment_or_implicit_mode(validation_case, tmp_path, field):
    _args, request, plan, trace, _output = validation_case
    selected = tmp_path / "learned.yaml"
    authored = _learned_config(plan, selected)
    engine = authored["engine"]
    worker = engine["workers"]["aggregated"]
    if field == "hardware":
        engine["hardware"] = "h100_sxm"
    elif field == "scheduler":
        worker["scheduler"]["max_sequences"] += 1
    elif field == "profile":
        engine["fpm_profile"]["provenance"] = "changed"
    elif field == "precision":
        worker["timing"]["gemm_quant_mode"] = "float16"
    else:
        worker["timing"]["estimation_mode"] = "auto"
    selected.write_text(yaml.safe_dump(authored))
    with pytest.raises(ValueError):
        validation._prediction_config(request, plan, trace, selected)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("inherited", [False, True])
def test_learned_evidence_retains_native_dependency_closure_and_detects_drift(tmp_path, monkeypatch, legacy, inherited):
    source = tmp_path / "seed.bin"
    source.write_bytes(b"accepted seed")
    dependency = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "byte_length": source.stat().st_size,
        "purpose": "training checkpoint",
    }
    resolved = {
        "estimation_mode": "fpm_learned",
        "fallback_policy": "deny",
        "estimator_config": {"fpm_learned": {"resolved_source": {"kind": "bundle", "dependencies": [dependency]}}},
    }
    if legacy:
        resolved["estimator_config"]["fpm_learned"] = {"artifact_path": str(source)}
    state = tmp_path / "snapshot-state.bin"
    state.write_bytes(b"complete mutable state")
    state_dependency = {
        "path": str(state),
        "sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
        "byte_length": state.stat().st_size,
        "purpose": "snapshot state",
    }
    dependencies = {
        "systems_paths": [],
        "files": [dependency, state_dependency],
        "fallback": None,
        "owned_systems_paths": [],
        "external_systems_paths": [],
    }
    calls = []

    class Model:
        def diagnostics(self):
            return {"provenance": {"config": resolved}}

        def readiness_report(self):
            return {
                "stores": [{"workload_kind": "pure_decode", "ready": True}],
                "execution": {"ready": True, "required_stores": ["pure_decode"], "unavailable": []},
            }

        def dependency_info(self):
            return dependencies

        def verify_dependencies(self):
            calls.append("verified")

        def close(self):
            calls.append("closed")

    from aisimulate_core.sdk import RustForwardPassPerfModel

    monkeypatch.setattr(timing, "canonical_timing_config", lambda _prediction: {"selected": "identity"})
    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", lambda config: Model())
    prediction = {"engine": {"workers": {"aggregated": {"timing": {"estimation_mode": "fpm_learned"}}}}}
    if inherited:
        prediction["engine"]["estimation_mode"] = "fpm_learned"
        del prediction["engine"]["workers"]["aggregated"]["timing"]
    evidence = timing.learned_timing_evidence(prediction, pin=True)
    assert prediction["engine"]["workers"]["aggregated"]["timing"]["estimator_config"] == resolved["estimator_config"]
    assert evidence["config"] == resolved
    assert evidence["dependencies"] == [dependency, state_dependency]
    assert evidence["dependency_info"] == dependencies
    assert calls == ["closed"]
    timing.verify_timing_dependencies(evidence)
    assert calls == ["closed", "verified", "closed"]
    state.write_bytes(b"changed snapshot state!")
    with pytest.raises(ValueError, match="dependency changed"):
        timing.verify_timing_dependencies(evidence)


@pytest.mark.parametrize("reason", ["untrained_forest_store", "unexpected_failure"])
def test_unready_learned_validation_saves_incomplete_evidence_without_predict(
    validation_case, tmp_path, monkeypatch, reason
):
    from aisimulate import main as cli
    from aisimulate import supervision

    args, _request, plan, _trace, output = validation_case
    selected = tmp_path / "learned.yaml"
    _learned_config(plan, selected)
    unavailable = {"workload_kind": "pure_decode", "ready": False, "not_ready_reason": reason}
    readiness = {
        "ready": True,
        "stores": [{"workload_kind": "pure_prefill", "ready": True}, unavailable],
        "execution": {"ready": False, "required_stores": ["pure_prefill", "pure_decode"], "unavailable": [unavailable]},
    }
    monkeypatch.setattr(
        validation,
        "learned_timing_evidence",
        lambda *args, **kwargs: {
            "config": {"estimation_mode": "fpm_learned"},
            "dependencies": [],
            "readiness": readiness,
        },
    )
    monkeypatch.setattr(supervision, "main", lambda *_args: pytest.fail("unready model must not reach prediction"))
    assert cli.main([*args, "--prediction-config", str(selected)]) == 1
    report = json.loads((output / "validation.json").read_text())
    assert report["status"] == "incomplete"
    assert report["accuracy"] == "not_assessed"
    assert report["timing_model"]["readiness"] == readiness
    assert reason in report["issues"][0]
    assert json.loads((output / "learned-coverage.json").read_text())["status"] == "incomplete"
    assert "prediction_exit_code" not in report


def test_empty_explicit_learned_source_is_invalid_without_replay(validation_case, tmp_path, monkeypatch):
    from aisimulate import main as cli
    from aisimulate import supervision

    args, _request, plan, _trace, output = validation_case
    selected = tmp_path / "missing-learned.yaml"
    prediction = _learned_config(plan, selected)
    prediction["engine"]["workers"]["aggregated"]["timing"]["estimator_config"] = {
        "fpm_learned": {"source_policy": "explicit_only"}
    }
    selected.write_text(yaml.safe_dump(prediction))
    monkeypatch.setattr(supervision, "main", lambda *_args: pytest.fail("missing source must not replay"))
    with pytest.raises(SystemExit) as error:
        cli.main([*args, "--prediction-config", str(selected)])
    assert error.value.code == 2
    assert not (output / "validation.json").exists()


def test_serving_preparation_preserves_unready_learned_evidence(request, tmp_path, monkeypatch):
    case = request.getfixturevalue("_serving_case")
    selected = tmp_path / "cold-serving.yaml"
    prediction = yaml.safe_load(case.prediction_config.read_text())
    prediction["engine"]["workers"]["aggregated"]["timing"]["estimation_mode"] = "fpm_learned"
    selected.write_text(yaml.safe_dump(prediction))
    readiness = {
        "execution": {
            "ready": False,
            "unavailable": [{"workload_kind": "contains_locally_mixed", "not_ready_reason": "untrained_forest_store"}],
        }
    }
    monkeypatch.setattr(
        serving,
        "learned_timing_evidence",
        lambda *args, **kwargs: {
            "config": {"estimation_mode": "fpm_learned"},
            "readiness": readiness,
            "dependencies": [],
        },
    )
    monkeypatch.setattr(
        serving, "_prepare_matched_workload", lambda *args: pytest.fail("must not replay unready model")
    )
    path = serving.prepare_serving_validation(
        trace=case.trace,
        prediction_config=selected,
        prediction_report=case.prediction_report,
        output=tmp_path / "cold-serving",
        endpoint="http://127.0.0.1:8000",
        tokenizer=case.tokenizer,
        expected_execution=case.recipe["expected_execution"],
        aiperf_python=Path("/benchmark/bin/python"),
    )
    recipe = json.loads(path.read_text())
    assert recipe["status"] == "incomplete"
    assert recipe["timing_model"]["readiness"] == readiness
    assert any("contains_locally_mixed: untrained_forest_store" in issue for issue in recipe["issues"])
    assert "matched_workload" not in recipe


def test_serving_forward_validation_preserves_selected_model_and_rank_iteration(tmp_path, monkeypatch):
    config_path = tmp_path / "predict.yaml"
    config_path.write_text("engine: {}")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text("{}")
    canonical = {"estimation_mode": "fpm_learned", "attention_dp": 2, "estimator_config": {"custom": "retained"}}
    recipe = {"prediction_config": {"path": str(config_path)}, "timing_model": {"config": canonical}}
    ranks = [
        {"dp_rank": rank, "scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 64 + rank}}
        for rank in range(2)
    ]
    source = tmp_path / "native.json"
    source.write_text("{}")
    samples = tmp_path / "forward.json"
    samples.write_text(
        json.dumps(
            {
                "recipe": serving._identity(recipe_path),
                "sources": [serving._identity(source)],
                "samples": {"decode": [{"metrics_by_rank": ranks, "unit": "ms", "measured_ms": 7.0}]},
            }
        )
    )
    calls = []

    class Model:
        def estimate_forward_pass_time_ms(self, metrics):
            calls.append(copy.deepcopy(metrics))
            return 7.0

        def close(self):
            pass

    from aisimulate_core.sdk import RustForwardPassPerfModel

    def construct(config):
        assert config == canonical
        return Model()

    monkeypatch.setattr(serving, "_checked_recipe", lambda _: recipe)
    monkeypatch.setattr(RustForwardPassPerfModel, "best_available", construct)
    report = serving._forward_validation(samples, recipe_path, 0.1)
    assert report["phases"]["decode"]["status"] == "passed"
    assert calls == [ranks]


@pytest.mark.parametrize("snapshot", [False, True], ids=["bundle", "moved-snapshot"])
@pytest.mark.parametrize("inherited", [False, True], ids=["worker-controls", "inherited-controls"])
def test_seeded_bundle_replay_and_matched_serving_keep_source_pins(
    validation_case, request, tmp_path, snapshot, inherited
):
    import_sklearn()
    from aisimulate import main as cli
    from aisimulate_core.sdk import RustForwardPassPerfModel, fpm_learned

    args, _request, plan, trace, output = validation_case
    case = request.getfixturevalue("_serving_case")
    original = (plan / "predict/pilot.yaml").read_bytes()
    prediction = yaml.safe_load(original)
    canonical = timing.canonical_timing_config(prediction)
    canonical.update(estimation_mode="fpm_learned", estimator_config={"correction": {"enabled": False}})
    # Synthetic complete native rank groups exercise producer -> canonical
    # construction -> ordinary replay; they make no silicon accuracy claim.
    iterations = []
    for index in range(24):
        for phase in ("prefill", "decode", "mixed"):
            scheduled = {}
            if phase in {"prefill", "mixed"}:
                scheduled.update(num_prefill_requests=1, sum_prefill_tokens=64 + index, sum_prefill_kv_tokens=index)
            if phase in {"decode", "mixed"}:
                scheduled.update(num_decode_requests=1, sum_decode_kv_tokens=64 + index)
            iterations.append(
                [
                    {
                        "version": 1,
                        "dp_rank": 0,
                        "worker_id": "w0",
                        "counter_id": len(iterations),
                        "wall_time": 0.002 + index / 100000,
                        "scheduled_requests": scheduled,
                        "queued_requests": {},
                    }
                ]
            )
    source = tmp_path / "training.json"
    source.write_text(json.dumps(iterations))
    resolved = fpm_learned.train_bundle(
        iterations,
        "aggregated",
        output_dir=tmp_path / "bundle",
        identity=canonical,
        config=canonical,
        provenance={
            "anchor_kind": "b",
            "sources": [{"reference": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
            "evaluation_sources": [{"reference": str(trace), "sha256": hashlib.sha256(trace.read_bytes()).hexdigest()}],
        },
        max_iter=150,
        max_leaf_nodes=15,
        learning_rate=0.2,
    )
    if snapshot:
        model = RustForwardPassPerfModel.best_available(resolved)
        try:
            model.save_state_snapshot(tmp_path / "state")
        finally:
            model.close()
        shutil.move(tmp_path / "state", tmp_path / "moved-state")
        shutil.rmtree(tmp_path / "bundle")
        source.unlink()
        restored = RustForwardPassPerfModel.best_available(state_snapshot=tmp_path / "moved-state")
        try:
            resolved = restored.diagnostics()["provenance"]["config"]
        finally:
            restored.close()
    selected = tmp_path / "learned-predict.yaml"
    prediction["engine"]["workers"]["aggregated"]["timing"].update(
        estimation_mode="fpm_learned", fallback_policy="deny", estimator_config=resolved["estimator_config"]
    )
    if inherited:
        engine = prediction["engine"]
        worker_timing = engine["workers"]["aggregated"].pop("timing")
        for field in ("estimation_mode", "fallback_policy", "estimator_config"):
            engine[field] = worker_timing[field]
        del engine["mode"]
    selected.write_text(yaml.safe_dump(prediction))
    learned_output = tmp_path / "learned-replay"
    learned_args = [str(learned_output) if value == str(output) else value for value in args]
    assert cli.main([*learned_args, "--prediction-config", str(selected)]) == 0
    report = json.loads((learned_output / "validation.json").read_text())
    assert report["status"] == "covered"
    assert report["accuracy"] == "not_assessed"
    assert report["timing_model"]["dependencies"]
    if snapshot:
        identities = report["timing_model"]["dependencies"]
        assert str(tmp_path / "moved-state" / "manifest.json") in {item["path"] for item in identities}
        assert all(Path(item["path"]).is_relative_to(tmp_path / "moved-state") for item in identities)
        assert report["timing_model"]["config"]["estimator_config"]["state_snapshot"]["on_tune"] == "reject"
    assert report["fpm_query_coverage"]["kind"] == "learned_replay"
    assert "queries" not in report["fpm_query_coverage"]
    assert (plan / "predict/pilot.yaml").read_bytes() == original
    serving_prediction = learned_output / "predict.yaml"
    if inherited:
        # Replay pinning writes a role-local timing block. Exercise serving's
        # own selection and pinning on the equivalent inherited form as well.
        inherited_prediction = yaml.safe_load(serving_prediction.read_text())
        engine = inherited_prediction["engine"]
        worker_timing = engine["workers"]["aggregated"].pop("timing")
        for field in ("estimation_mode", "fallback_policy", "estimator_config"):
            engine[field] = worker_timing[field]
        assert "mode" not in engine
        serving_prediction = learned_output / "inherited-serving.yaml"
        serving_prediction.write_text(yaml.safe_dump(inherited_prediction))
    recipe_path = serving.prepare_serving_validation(
        prediction_config=serving_prediction,
        trace=trace,
        output=tmp_path / "learned-serving",
        endpoint="http://127.0.0.1:8000",
        tokenizer=case.tokenizer,
        expected_execution=case.recipe["expected_execution"],
        aiperf_python=Path("/benchmark/bin/python"),
    )
    recipe = serving._checked_recipe(recipe_path)
    assert recipe["status"] == "ready"
    assert recipe["timing_model"]["config"] == report["timing_model"]["config"]
    matched = yaml.safe_load(Path(recipe["matched_workload"]["prediction_config"]["path"]).read_text())
    assert timing.selected_timing_mode(matched) == "fpm_learned"
    coverage = json.loads(Path(recipe["matched_workload"]["coverage_report"]["path"]).read_text())
    assert coverage["kind"] == "learned_replay"
    assert coverage["status"] == "covered"
    dependency = Path(report["timing_model"]["dependencies"][-1]["path"])
    dependency.write_bytes(dependency.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        serving._checked_recipe(recipe_path)


@pytest.fixture
def learned_prior(tmp_path, monkeypatch):
    # Reuse the existing native A-prior fixture without importing its optional
    # training dependencies while collecting unrelated onboarding tests.
    from .sdk import test_fpm_mixed_composition_integration as fixture

    roots = fixture.phase_roots.__wrapped__(tmp_path, monkeypatch)
    source = fixture.mixed_seed.__wrapped__(tmp_path, roots)
    return source, fixture._config(source)


def _native_timing_evidence(model):
    dependencies = model.dependency_info()
    return {
        "config": model.diagnostics()["provenance"]["config"],
        "readiness": model.readiness_report(),
        "dependencies": dependencies["files"],
        "dependency_info": dependencies,
    }


@pytest.mark.parametrize("change", ["directory-added", "file-removed"])
def test_validation_rechecks_native_prior_closure(learned_prior, change):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    source, config = learned_prior
    model = RustForwardPassPerfModel.best_available(config)
    try:
        evidence = _native_timing_evidence(model)
    finally:
        model.close()
    timing.verify_timing_dependencies(evidence)
    data = Path(source[0]["systems_paths"][0]) / "data/h200_sxm"
    if change == "directory-added":
        # No pinned file changed: only native fingerprint validation can catch this.
        (data / "new-empty-version").mkdir()
    else:
        next(data.rglob("*.parquet")).unlink()
    with pytest.raises(timing.LearnedTimingDependencyError, match="unavailable or changed"):
        timing.verify_timing_dependencies(evidence)


def test_validation_rechecks_relocated_prior_after_original_sources_are_removed(learned_prior, tmp_path):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    source, config = learned_prior
    model = RustForwardPassPerfModel.best_available(config)
    try:
        model.save_state_snapshot(tmp_path / "state")
    finally:
        model.close()
    moved = tmp_path / "relocated"
    shutil.move(tmp_path / "state", moved)
    shutil.rmtree(source[0]["systems_paths"][0])
    shutil.rmtree(source[1])
    Path(source[2]["manifest_path"]).unlink()
    model = RustForwardPassPerfModel.best_available(state_snapshot=moved)
    try:
        evidence = _native_timing_evidence(model)
    finally:
        model.close()
    assert evidence["dependency_info"]["owned_systems_paths"]
    assert evidence["dependency_info"]["external_systems_paths"] == []
    assert all(Path(item["path"]).is_relative_to(moved) for item in evidence["dependencies"])
    timing.verify_timing_dependencies(evidence)


@pytest.mark.parametrize(
    "failure", ["prior directory added", "prior file removed", "pure_decode: untrained_forest_store"]
)
def test_serving_assessment_retains_incomplete_native_dependency_failure(request, tmp_path, monkeypatch, failure):
    case = request.getfixturevalue("_serving_case")
    recipe = json.loads(case.recipe_path.read_text())
    recipe["timing_model"] = {"config": {"estimation_mode": "fpm_learned"}, "dependencies": []}
    case.recipe_path.write_text(json.dumps(recipe))

    def unavailable(_evidence):
        raise timing.LearnedTimingDependencyError(failure)

    monkeypatch.setattr(serving, "verify_timing_dependencies", unavailable)
    output = tmp_path / "unready-assessment.json"
    report = serving.validate_serving_measurements(
        case.recipe_path,
        prediction_report=case.prediction_report,
        execution_evidence_path=case.observed_path,
        output_report=output,
    )
    assert report["status"] == "incomplete"
    assert report["timing_model"] == {"status": "incomplete", "issues": [failure]}
    assert json.loads(output.read_text()) == report
