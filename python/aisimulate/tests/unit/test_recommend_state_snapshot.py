# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recommendation roles timed by a calibrated forward-pass model state snapshot.

A role whose timing names only ``estimator_config.state_snapshot`` is pinned to
the snapshot's saved engine identity; replicas and scheduler knobs are still
searched, and Core's restoration rejects any other candidate.
"""

from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.recommend import recommendation_to_sweeper, run_recommendation
from aisimulate.runner import EngineReplayRunnerFactory
from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.forward_pass_estimator import ForwardPassEstimatorResolutionError, ForwardPassEstimatorResolver
from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel

pytestmark = pytest.mark.unit

_MODEL = "Qwen/Qwen3-0.6B"
_SYSTEM = "h200_sxm"
_BLOCK = 16
_SLOWER = 1.5


def _rank(prefill: int, decode: int, kv: int) -> dict:
    return {
        "version": 1,
        "wall_time": 0.0,
        "scheduled_requests": {
            "num_prefill_requests": int(prefill > 0),
            "sum_prefill_tokens": prefill,
            "num_decode_requests": decode,
            "sum_decode_kv_tokens": kv if decode else 0,
        },
    }


def _native(worker: str, **overrides) -> RustForwardPassPerfModel:
    fields = {
        "model": _MODEL,
        "system": _SYSTEM,
        "backend": "vllm",
        "worker_type": worker,
        "estimation_mode": "op_level",
        "kv_block_size": _BLOCK,
        "estimator_config": {"correction": {"mode": "fixed", "min_observations": 3}},
        **overrides,
    }
    return RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**fields))


def _calibrate(model: RustForwardPassPerfModel, worker: str) -> RustForwardPassPerfModel:
    """Observe every iteration `_SLOWER` times slower than the native estimate."""
    iterations = []
    for index in range(60):
        prefill = worker != "decode" and (worker == "prefill" or index % 2 == 1)
        rank = _rank(256 * (1 + index % 8), 0, 0) if prefill else _rank(0, 1 + index % 32, 2048 * (1 + index % 16))
        rank["wall_time"] = model.estimate_forward_pass_time_ms([rank]) * _SLOWER / 1000.0
        iterations.append([rank])
    model.tune_with_fpms(iterations)
    return model


def _moved(model: RustForwardPassPerfModel, root: Path, name: str, *, include_perf_db: bool) -> Path:
    """Save as the planner would, then expose only a copy at another mount."""
    saved = root / "planner" / name
    model.save_state_snapshot(saved, include_perf_db=include_perf_db)
    moved = root / "mount" / name
    shutil.copytree(saved, moved)
    shutil.rmtree(saved)
    return moved


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("snapshots")
    regression = RustForwardPassPerfModel.best_available(
        ForwardPassPerfModelConfig(
            model="dynamo-planner:test",
            system="unprofiled",
            backend="vllm",
            worker_type="aggregated",
            estimation_mode="fpm_regression",
        )
    )
    return {
        "packed": _moved(_calibrate(_native("aggregated"), "aggregated"), root, "packed", include_perf_db=True),
        "referenced": _moved(
            _calibrate(_native("aggregated"), "aggregated"), root, "referenced", include_perf_db=False
        ),
        "decode": _moved(_calibrate(_native("decode"), "decode"), root, "decode", include_perf_db=True),
        "uncalibrated": _moved(_native("aggregated"), root, "uncalibrated", include_perf_db=False),
        "regression": _moved(regression, root, "regression", include_perf_db=False),
        # Equivalent spellings of fields the sweeper never searches.
        "spelled": _moved(
            _calibrate(
                _native("aggregated", dcp=1, cp_size=1, moe_tp_size=1, moe_ep_size=1, moe_backend="default"),
                "aggregated",
            ),
            root,
            "spelled",
            include_perf_db=False,
        ),
        "strict": _moved(
            _calibrate(_native("aggregated", strict_provenance=True), "aggregated"),
            root,
            "strict",
            include_perf_db=True,
        ),
        "trtllm": _moved(
            _calibrate(_native("aggregated", backend="trtllm", kv_block_size=32), "aggregated"),
            root,
            "trtllm",
            include_perf_db=False,
        ),
        "no-block-size": _moved(
            _calibrate(_native("aggregated", kv_block_size=None), "aggregated"),
            root,
            "no-block-size",
            include_perf_db=False,
        ),
        "pipeline": _moved(
            _calibrate(_native("aggregated", pp=2), "aggregated"), root, "pipeline", include_perf_db=False
        ),
    }


def _request(snapshot: Path | None, *, mode: str = "aggregated", role: str = "aggregated") -> dict:
    workers = {name: {} for name in (("aggregated",) if mode == "aggregated" else ("prefill", "decode"))}
    if snapshot is not None:
        workers[role]["timing"] = {"estimator_config": {"state_snapshot": {"path": str(snapshot)}}}
    return {
        "engine": {"mode": mode, "model": _MODEL, "hardware": _SYSTEM, "context_length": 256, "workers": workers},
        "traffic": {
            "source": {"type": "synthetic", "input_tokens": 128, "output_tokens": 16},
            "load": {"type": "concurrency", "concurrency": 4},
            "stop": {"requests": 8},
        },
        "optimization": {"target": "throughput", "constraints": {"max_candidate_gpus": 2}},
        "optimizer": {"algorithm": "random", "max_trials": 1, "parallelism": 1},
    }


def _space(raw: dict) -> SearchSpace:
    return recommendation_to_sweeper(CoreRecommendationConfig.model_validate(raw)).search_space


@pytest.mark.parametrize("name", ["packed", "referenced"])
def test_a_snapshot_reference_alone_pins_the_role_to_the_saved_engine(snapshots, name):
    space = _space(_request(snapshots[name]))

    assert space.backend == ["vllm"]
    assert (
        space.backend_version
        == RustForwardPassPerfModel.inspect_state_snapshot(snapshots[name])["identity"]["backend_version"]
    )
    assert space.agg_block_size == _BLOCK
    # Replicas stay free; the role's shape is the saved one.
    assert space.parallel_independent_by_mode["agg"] == {
        "replicas": None,
        "tp": [1],
        "pp": [1],
        "attention_dp": [1],
        "moe_tp": [1],
        "moe_ep": [1],
    }
    assert space.role_estimator_controls["agg"] == {
        "estimation_mode": "op_level",
        "fallback_policy": "deny",
        "estimator_config": {"state_snapshot": {"path": str(snapshots[name])}},
    }
    packed_root = str(snapshots[name] / "perf_db" / "systems")
    assert (space.systems_paths_for("agg") == [packed_root]) == (name == "packed")
    # Derived values are ordinary settings: saving and reloading is a fixed point.
    saved = space.model_dump(mode="json")
    reloaded = SearchSpace.model_validate(deepcopy(saved))
    assert reloaded.model_dump(mode="json") == saved
    assert reloaded.systems_paths_for("agg") == space.systems_paths_for("agg")


def test_a_snapshot_pins_only_its_own_role(snapshots):
    space = _space(_request(snapshots["decode"], mode="disaggregated", role="decode"))

    assert space.decode_block_size == _BLOCK
    assert space.prefill_block_size is None
    assert space.parallel_independent_by_mode["disagg"] == {
        "decode_replicas": None,
        "decode_tp": [1],
        "decode_pp": [1],
        "decode_attention_dp": [1],
        "decode_moe_tp": [1],
        "decode_moe_ep": [1],
    }
    assert space.role_estimator_controls.get("prefill", {}) == {}
    assert space.systems_paths_for("decode") == [str(snapshots["decode"] / "perf_db" / "systems")]
    assert space.systems_paths_for("prefill") is None


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (
            lambda raw, w: w.update(parallelism={"preset": False, "tensor": 2}),
            r"agg tp choices=\[2\] conflicts with .*saved with 1",
        ),
        (lambda raw, w: w.update(kv_cache={"block_size": 64}), r"agg block_size=64 conflicts .*saved with 16"),
        (lambda raw, w: raw["engine"].update(backend="sglang"), r"engine.backend=\['sglang'\] conflicts"),
        (lambda raw, w: raw["engine"].update(backend_version="0.10.0"), r"engine.backend_version='0.10.0' conflicts"),
        (lambda raw, w: raw["engine"].update(gemm_quant_mode="fp8"), r"engine.gemm_quant_mode='fp8' conflicts"),
        (lambda raw, w: raw["engine"].update(nextn=1, nextn_accepted=0.5), r"engine.nextn=1 conflicts"),
        (lambda raw, w: raw["engine"].update(hardware="b200_sxm"), r"hardware='b200_sxm' conflicts"),
        (
            lambda raw, w: w["timing"].update(estimation_mode="fpm_interpolation"),
            r"estimation_mode='fpm_interpolation' conflicts",
        ),
        (lambda raw, w: w["timing"].update(database_mode="HYBRID"), r"database_mode conflicts .*pins it"),
        (lambda raw, w: w["timing"].update(systems_paths=["default"]), r"carries its own performance data"),
        (
            lambda raw, w: w["timing"]["estimator_config"].update(correction={"enabled": False}),
            r"estimator_config may only name state_snapshot",
        ),
        (
            lambda raw, w: w["timing"]["estimator_config"]["state_snapshot"].update(sha256="0" * 64),
            r"sha256 does not match the snapshot manifest",
        ),
        (
            lambda raw, w: raw["engine"].update(estimator_config=w.pop("timing")["estimator_config"]),
            r"applies to one worker role",
        ),
    ],
)
def test_explicit_settings_that_conflict_with_the_snapshot_fail_validation(snapshots, edit, message):
    raw = _request(snapshots["packed"])
    edit(raw, raw["engine"]["workers"]["aggregated"])

    with pytest.raises((ValidationError, ValueError), match=message):
        _space(raw)


def test_settings_that_agree_with_the_snapshot_are_accepted(snapshots):
    raw = _request(snapshots["packed"])
    worker = raw["engine"]["workers"]["aggregated"]
    worker["parallelism"] = {"preset": False, "tensor": {"choices": [1, 2]}, "replicas": {"choices": [1, 2]}}
    worker["kv_cache"] = {"block_size": {"choices": [16, 64]}}
    worker["timing"].update(estimation_mode="op_level", fallback_policy="deny")
    raw["engine"]["backend"] = {"choices": ["vllm", "sglang"]}

    space = _space(raw)

    assert space.backend == ["vllm"]
    assert space.agg_block_size == _BLOCK
    assert space.parallel_independent_by_mode["agg"]["tp"] == [1]
    assert space.parallel_independent_by_mode["agg"]["replicas"] == [1, 2]


@pytest.mark.parametrize(
    ("name", "role", "message"),
    [
        ("regression", "aggregated", r"records no kv_block_size"),
        ("uncalibrated", "aggregated", r"no enabled, ready correction store"),
        ("no-block-size", "aggregated", r"records no kv_block_size"),
        ("pipeline", "aggregated", r"records pp=2; engine replay has no pipeline-stage model"),
        ("decode", "aggregated", r"calibrated for a decode worker, not aggregated"),
    ],
)
def test_snapshots_replay_cannot_use_are_rejected(snapshots, name, role, message):
    with pytest.raises((ValidationError, ValueError), match=message):
        _space(_request(snapshots[name], role=role))


def test_a_conflicting_candidate_is_rejected_by_restoration(snapshots):
    space = _space(_request(snapshots["packed"]))
    candidate = {
        "deployment_mode": "agg",
        "backend": "vllm",
        "hardware_sku": _SYSTEM,
        "tp": 2,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "agg_block_size": _BLOCK,
        "agg_timing_model": None,
    }
    resolver = ForwardPassEstimatorResolver(space)
    resolved = resolver.resolve_candidate({**candidate, "tp": 1})
    assert resolved["agg"].config["estimator_config"]["state_snapshot"]["path"] == str(snapshots["packed"])

    with pytest.raises(ForwardPassEstimatorResolutionError, match=r"pinned identity at tp"):
        resolver.resolve_candidate(candidate)


def _throughput_run(raw: dict):
    worker = raw["engine"]["workers"]["aggregated"]
    worker["parallelism"] = {"preset": False, "replicas": 1}
    worker.setdefault("kv_cache", {})
    return run_recommendation(
        CoreRecommendationConfig.model_validate(raw),
        stack="engine",
        runner_factory=EngineReplayRunnerFactory(),
        show_progress=False,
    )


@pytest.mark.parametrize("name", ["packed", "referenced"])
def test_recommendation_replays_the_calibrated_model_and_exports_a_replayable_prediction(snapshots, name):
    calibrated = _throughput_run(_request(snapshots[name]))
    default = _request(None)
    default["engine"]["workers"]["aggregated"]["kv_cache"] = {"block_size": _BLOCK}
    default["engine"]["backend"] = "vllm"
    uncalibrated = _throughput_run(default)

    assert calibrated.counts.feasible == 1, [record.reason for record in calibrated.candidates]
    [candidate] = calibrated.selected_candidates
    [baseline] = uncalibrated.selected_candidates
    estimator = candidate.config["forward_pass_estimators"]["agg"]["config"]
    assert estimator["estimator_config"]["state_snapshot"]["path"] == str(snapshots[name])
    # Every observation was slower than the native estimate, so the calibrated
    # role serves the same traffic more slowly than the default estimator.
    assert candidate.metrics["mean_ttft_ms"] > baseline.metrics["mean_ttft_ms"] * 1.2
    # The exported prediction replays the same calibrated model.
    assert _replay(candidate.prediction_config).metrics["mean_ttft_ms"] == candidate.metrics["mean_ttft_ms"]


def test_equivalent_saved_spellings_resolve_and_the_default_backend_follows_the_snapshot(snapshots):
    space = _space(_request(snapshots["spelled"]))
    candidate = {
        "deployment_mode": "agg",
        "backend": "vllm",
        "hardware_sku": _SYSTEM,
        "tp": 1,
        "pp": 1,
        "attention_dp": 1,
        "moe_tp": 1,
        "moe_ep": 1,
        "agg_block_size": _BLOCK,
        "agg_timing_model": None,
    }
    resolved = ForwardPassEstimatorResolver(space).resolve_candidate(candidate)["agg"].config
    assert (resolved["dcp"], resolved["cp_size"], resolved["moe_tp_size"], resolved["moe_backend"]) == (
        1,
        1,
        1,
        "default",
    )
    # Recommend's default backend domain does not override a snapshot's backend.
    trtllm = _space(_request(snapshots["trtllm"]))
    assert (trtllm.backend, trtllm.agg_block_size) == (["trtllm"], 32)


def test_disaggregated_presets_keep_the_saved_shape_and_sdk_modes_may_be_tuples(snapshots):
    def shape(tensor: int) -> dict:
        return {"replicas": 1, "tensor": tensor, "pipeline": 1, "attention_data": 1, "moe_tensor": 1, "moe_expert": 1}

    raw = _request(snapshots["decode"], mode="disaggregated", role="decode")
    raw["engine"]["workers"]["decode"]["parallelism"] = {"preset": [shape(2), shape(1)]}
    space = _space(raw)
    assert [entry["tp"] for entry in space.parallel_custom_configs_by_mode["disagg"]["decode"]] == [1]
    raw["engine"]["workers"]["prefill"]["parallelism"] = {"preset": [shape(1), shape(2)]}
    space = _space(raw)
    assert {(entry["prefill"]["tp"], entry["decode"]["tp"]) for entry in space.parallel_configs_by_mode["disagg"]} == {
        (1, 1),
        (2, 1),
    }
    raw["engine"]["workers"]["decode"]["parallelism"] = {"preset": [shape(2)]}
    with pytest.raises((ValidationError, ValueError), match="every disagg parallel preset"):
        _space(raw)

    sdk = SearchSpace(
        model_name=_MODEL,
        hardware_sku=_SYSTEM,
        deployment_mode=("agg",),
        role_estimator_controls={"agg": {"estimator_config": {"state_snapshot": {"path": str(snapshots["packed"])}}}},
    )
    assert (sdk.agg_block_size, sdk.parallel_independent_by_mode["agg"]["tp"]) == (_BLOCK, [1])


def _replay(prediction_config: dict):
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(prediction_config))
    runner = EngineReplayRunnerFactory().create(0)
    try:
        return runner.run(spec)
    finally:
        runner.close()


def test_exported_predictions_restore_pinned_controls_and_a_moved_packed_snapshot(snapshots, tmp_path):
    shutil.copytree(snapshots["strict"], tmp_path / "strict")
    result = _throughput_run(_request(tmp_path / "strict"))
    [candidate] = result.selected_candidates
    assert candidate.config["forward_pass_estimators"]["agg"]["config"]["strict_provenance"] is True
    # Prediction YAML has no per-role strict_provenance; predict takes it from the snapshot.
    assert "strict_provenance" not in candidate.prediction_config["engine"]
    assert _replay(candidate.prediction_config).metrics["mean_ttft_ms"] == candidate.metrics["mean_ttft_ms"]
    # The packed snapshot moves; only its path in the prediction changes.
    shutil.move(tmp_path / "strict", tmp_path / "moved")
    moved = deepcopy(candidate.prediction_config)
    moved["engine"]["workers"]["aggregated"]["timing"]["estimator_config"]["state_snapshot"]["path"] = str(
        tmp_path / "moved"
    )
    assert _replay(moved).metrics["mean_ttft_ms"] == candidate.metrics["mean_ttft_ms"]


def test_exported_predictions_keep_the_snapshots_saved_spellings(snapshots):
    result = _throughput_run(_request(snapshots["spelled"]))
    [candidate] = result.selected_candidates
    assert _replay(candidate.prediction_config).metrics["mean_ttft_ms"] == candidate.metrics["mean_ttft_ms"]


def _disaggregated_candidate(snapshot: Path | None):
    raw = _request(snapshot, mode="disaggregated", role="decode")
    raw["engine"]["backend"] = "vllm"
    for worker in raw["engine"]["workers"].values():
        worker["parallelism"] = {"preset": False, "replicas": 1}
        worker["kv_cache"] = {"block_size": _BLOCK}
    result = run_recommendation(
        CoreRecommendationConfig.model_validate(raw),
        stack="engine",
        runner_factory=EngineReplayRunnerFactory(),
        show_progress=False,
    )
    [candidate] = result.selected_candidates
    return candidate


def test_exported_disaggregated_predictions_follow_a_moved_packed_snapshot(snapshots, tmp_path):
    shutil.copytree(snapshots["decode"], tmp_path / "decode")
    candidate = _disaggregated_candidate(tmp_path / "decode")
    baseline = _disaggregated_candidate(None)
    # Every decode observation was slower than the native estimate.
    assert candidate.metrics["mean_tpot_ms"] > baseline.metrics["mean_tpot_ms"] * 1.2
    timing = candidate.prediction_config["engine"]["workers"]["decode"]["timing"]
    # The packed root belongs to the snapshot, so the prediction names only the snapshot.
    assert "systems_paths" not in timing
    shutil.move(tmp_path / "decode", tmp_path / "moved")
    moved = deepcopy(candidate.prediction_config)
    moved["engine"]["workers"]["decode"]["timing"]["estimator_config"]["state_snapshot"]["path"] = str(
        tmp_path / "moved"
    )
    replayed = _replay(moved).metrics
    assert (replayed["mean_ttft_ms"], replayed["mean_tpot_ms"]) == (
        candidate.metrics["mean_ttft_ms"],
        candidate.metrics["mean_tpot_ms"],
    )


def test_predictions_take_decoder_replay_from_the_snapshot_unless_the_engine_sets_it(snapshots, monkeypatch):
    from aisimulate.sweeper import state_snapshot

    [candidate] = _throughput_run(_request(snapshots["referenced"])).selected_candidates
    exported = candidate.prediction_config
    assert "decoder_replay" not in exported["engine"]
    saved = state_snapshot.saved_snapshot_config
    # Only a DeepSeek-V4.1 SGLang model can save decoder replay; pretend this one did.
    monkeypatch.setattr(
        state_snapshot, "saved_snapshot_config", lambda config: {**saved(config), "decoder_replay": True}
    )

    def compiled(**engine) -> bool:
        raw = _request(snapshots["referenced"])
        del raw["optimization"], raw["optimizer"]
        raw["engine"].update(backend="vllm", **engine)
        raw["engine"]["workers"]["aggregated"]["kv_cache"] = {"block_size": _BLOCK}
        spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(raw))
        return spec.backend_deployment.agg_engine_args["timing_model"]["config"]["decoder_replay"]

    assert compiled() is True
    # An explicit value is kept, so restoration rejects a conflicting one.
    assert compiled(decoder_replay=False) is False
    # A recommendation's exported prediction leaves the value to the snapshot.
    spec = prediction_to_replay_spec(CorePredictionConfig.model_validate(exported))
    assert spec.backend_deployment.agg_engine_args["timing_model"]["config"]["decoder_replay"] is True


def test_capacity_lowering_checks_a_snapshot_configuration_without_restoring_it(snapshots):
    from aisimulate.capacity import materialize_aic_num_gpu_blocks
    from aisimulate_core.sdk.rust_engine_step import _snapshot_config

    def lower(config: dict) -> dict:
        return materialize_aic_num_gpu_blocks(
            {"timing_model": {"type": "external", "provider": "aic", "config": config}, "gpu_memory_utilization": 0.9}
        )

    config = _snapshot_config(snapshots["referenced"])
    lowered = lower(config)
    assert lowered["num_gpu_blocks"] > 0
    assert lowered["timing_model"]["config"] == {
        **RustForwardPassPerfModel.normalize_config(config),
        "gpu_memory_utilization": 0.9,
    }
    with pytest.raises(ValueError, match="pinned identity at tp"):
        lower({**config, "tp": 2})
    with pytest.raises(ValueError, match="is not a directory; a snapshot saved without include_perf_db"):
        lower({**config, "systems_paths": ["/nonexistent/systems"]})
    malformed = deepcopy(config)
    del malformed["estimator_config"]["state_snapshot"]["path"]
    with pytest.raises(ValueError, match="state_snapshot"):
        lower(malformed)


# ---------------------------------------------------------------------------
# Regression and learned snapshots, as a Dynamo planner saves them
# ---------------------------------------------------------------------------

_LIMITS = {
    "num_gpus": 1,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 256,
    "max_kv_tokens": 16384,
    "context_length": "unknown",
}
_PLANNER_SLOWER = 2.0


def _scheduled(prefills: int, prefill_tokens: int, decodes: int, kv: int) -> dict:
    return {
        "version": 1,
        "wall_time": 0.0,
        "scheduled_requests": {
            "num_prefill_requests": prefills,
            "sum_prefill_tokens": prefill_tokens,
            "num_decode_requests": decodes,
            "sum_decode_kv_tokens": kv,
        },
    }


def _planner(worker: str, kinds=("prefill", "decode", "mixed"), **overrides) -> RustForwardPassPerfModel:
    """A planner's regression for an unknown deployment: hardware, version and
    shape unknown, its limits recorded. It observes every forward pass
    `_PLANNER_SLOWER` times slower than the op-level estimate."""
    fields = {
        "model": _MODEL,
        "system": "unknown",
        "backend": "vllm",
        "backend_version": "unknown",
        "worker_type": worker,
        "tp": "unknown",
        "pp": "unknown",
        "attention_dp": "unknown",
        "kv_block_size": _BLOCK,
        "estimation_mode": "fpm_regression",
        "deployment_limits": _LIMITS,
        **overrides,
    }
    model = RustForwardPassPerfModel.best_available(ForwardPassPerfModelConfig(**fields))
    reference = _native(worker, estimator_config={})
    iterations = []
    for index in range(48):
        prefills, decodes = 1 + index % 4, 1 + index % 8
        shapes = {
            "prefill": (prefills, 128 * prefills * (1 + index % 3), 0, 0),
            "decode": (0, 0, decodes, decodes * (128 + 37 * (index % 11))),
            "mixed": (1, 32 * (1 + index % 6), decodes, decodes * (128 + 41 * (index % 7))),
        }
        for kind in kinds:
            rank = _scheduled(*shapes[kind])
            rank["wall_time"] = reference.estimate_forward_pass_time_ms([rank]) * _PLANNER_SLOWER / 1000.0
            iterations.append([rank])
    model.tune_with_fpms(iterations)
    return model


@pytest.fixture(scope="module")
def planner_snapshots(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("planner")

    def save(name: str, model: RustForwardPassPerfModel) -> Path:
        return _moved(model, root, name, include_perf_db=False)

    return {
        "decode": save("decode", _planner("decode", kinds=("decode",))),
        "aggregated": save("aggregated", _planner("aggregated")),
        "unready": save("unready", _planner("aggregated", kinds=("decode",))),
        "nextn": save("nextn", _planner("decode", kinds=("decode",), nextn=1)),
        "no-kv": save(
            "no-kv", _planner("decode", kinds=("decode",), deployment_limits={**_LIMITS, "max_kv_tokens": "unknown"})
        ),
        "no-limits": save("no-limits", _planner("decode", kinds=("decode",), deployment_limits=None)),
        "no-sequences": save(
            "no-sequences",
            _planner("decode", kinds=("decode",), deployment_limits={**_LIMITS, "max_num_seqs": "unknown"}),
        ),
        # A regression whose identity is known pins it like a native snapshot.
        "known": save(
            "known",
            _planner(
                "decode",
                kinds=("decode",),
                system=_SYSTEM,
                backend_version="0.19.0",
                tp=1,
                pp=1,
                attention_dp=1,
            ),
        ),
    }


_DECODE_SHAPE = {"preset": False, "replicas": 1, "tensor": 1, "pipeline": 1, "attention_data": 1}


def _planner_request(snapshot: Path | None, *, decode_parallelism=None, mode: str = "disaggregated") -> dict:
    role = "decode" if mode == "disaggregated" else "aggregated"
    raw = _request(snapshot, mode=mode, role=role)
    raw["engine"]["backend"] = "vllm"
    for worker in raw["engine"]["workers"].values():
        worker["parallelism"] = {"preset": False, "replicas": 1}
        worker["kv_cache"] = {"block_size": _BLOCK}
    raw["engine"]["workers"][role]["parallelism"] = deepcopy(decode_parallelism or _DECODE_SHAPE)
    return raw


def _planner_run(raw: dict):
    return run_recommendation(
        CoreRecommendationConfig.model_validate(raw),
        stack="engine",
        runner_factory=EngineReplayRunnerFactory(),
        show_progress=False,
    )


def test_a_planner_regression_snapshot_times_its_role_in_its_explicit_shape(planner_snapshots):
    space = _space(_planner_request(planner_snapshots["decode"]))
    # Unknown hardware and version take the engine's; the shape is the role's.
    assert space.hardware_sku == _SYSTEM
    assert space.backend_version is None
    assert space.parallel_independent_by_mode["disagg"]["decode_tp"] == [1]
    # KV capacity is the recorded one; scheduler limits are capped.
    assert space.decode_num_gpu_blocks == _LIMITS["max_kv_tokens"] // _BLOCK
    assert space.decode_max_num_seqs == [256]
    assert space.decode_max_num_batched_tokens == [8192]
    # Derived values are ordinary settings: saving and reloading is a fixed point.
    saved = space.model_dump(mode="json")
    assert SearchSpace.model_validate(deepcopy(saved)).model_dump(mode="json") == saved

    result = _planner_run(_planner_request(planner_snapshots["decode"]))
    baseline = _planner_run(_planner_request(None))
    assert result.counts.feasible == 1, [record.reason for record in result.candidates]
    [candidate] = result.selected_candidates
    [reference] = baseline.selected_candidates
    estimator = candidate.config["forward_pass_estimators"]["decode"]["config"]
    assert (estimator["estimation_mode"], estimator["system"], estimator["tp"]) == (
        "fpm_regression",
        "unknown",
        "unknown",
    )
    assert estimator["deployment_limits"] == _LIMITS
    # Every observed pass was twice the op-level estimate, so the regression
    # role decodes the same traffic more slowly than the op-level baseline.
    assert candidate.metrics["mean_tpot_ms"] > reference.metrics["mean_tpot_ms"] * 1.5
    decode = candidate.prediction_config["engine"]["workers"]["decode"]
    capacity = decode["kv_cache"]["capacity"]
    assert (capacity["type"], capacity["blocks"]) == ("fixed", _LIMITS["max_kv_tokens"] // _BLOCK)
    replayed = _replay(candidate.prediction_config).metrics
    assert (replayed["mean_ttft_ms"], replayed["mean_tpot_ms"]) == (
        candidate.metrics["mean_ttft_ms"],
        candidate.metrics["mean_tpot_ms"],
    )
    # Latency-only: the regression role leaves the deployment without
    # operation power, as other latency-only timing does.
    assert candidate.metrics["power_w"] is None


def test_an_aggregated_planner_regression_prices_prefill_decode_and_mixed_passes(planner_snapshots):
    raw = _planner_request(planner_snapshots["aggregated"], mode="aggregated")
    result = _planner_run(raw)
    baseline = _planner_run(_planner_request(None, mode="aggregated"))
    assert result.counts.feasible == 1, [record.reason for record in result.candidates]
    [candidate] = result.selected_candidates
    [reference] = baseline.selected_candidates
    assert candidate.config["forward_pass_estimators"]["agg"]["config"]["estimation_mode"] == "fpm_regression"
    for metric in ("mean_ttft_ms", "mean_tpot_ms"):
        assert candidate.metrics[metric] > reference.metrics[metric] * 1.5, metric
    assert _replay(candidate.prediction_config).metrics["mean_tpot_ms"] == candidate.metrics["mean_tpot_ms"]


@pytest.mark.parametrize(
    ("name", "edit", "message"),
    [
        (
            "decode",
            lambda raw, w: w.update(parallelism={"preset": False, "replicas": 1}),
            r"does not know the deployment's tp, pp, attention_dp.*parallelism.tensor.*single values",
        ),
        (
            "decode",
            lambda raw, w: w["parallelism"].update(tensor={"choices": [1, 2]}),
            r"needs one explicit shape.*tp domain is \[1, 2\]",
        ),
        ("decode", lambda raw, w: w["parallelism"].update(tensor=2), r"uses 2 GPUs per replica.*num_gpus=1"),
        (
            "decode",
            lambda raw, w: w["kv_cache"].update(capacity={"type": "fixed", "blocks": 100}),
            r"decode KV capacity blocks=100 conflicts .*saved with 1024",
        ),
        (
            "decode",
            lambda raw, w: w["kv_cache"].update(capacity={"memory_fraction": {"choices": [0.8, 0.9]}}),
            r"KV memory is not searched",
        ),
        (
            "decode",
            lambda raw, w: w.update(scheduler={"max_sequences": {"choices": [512, 1024]}}),
            r"scheduler max_sequences=\[512, 1024\] exceeds .*deployment_limits.max_num_seqs=256",
        ),
        (
            "decode",
            lambda raw, w: w.update(scheduler={"max_batched_tokens": 16384}),
            r"max_batched_tokens=\[16384\] exceeds .*max_num_batched_tokens=8192",
        ),
        ("nextn", lambda raw, w: None, r"records nextn=1; an fpm_regression model prices no speculative decoding"),
        ("no-kv", lambda raw, w: None, r"does not record deployment_limits.max_kv_tokens"),
        ("no-limits", lambda raw, w: None, r"does not record deployment_limits.max_kv_tokens"),
        ("no-sequences", lambda raw, w: None, r"does not record deployment_limits.max_num_seqs"),
        ("known", lambda raw, w: raw["engine"].update(hardware="b200_sxm"), r"hardware='b200_sxm' conflicts"),
        (
            "known",
            lambda raw, w: raw["engine"].update(backend_version="0.10.0"),
            r"engine.backend_version='0.10.0' conflicts .*saved with '0.19.0'",
        ),
        ("known", lambda raw, w: w["parallelism"].update(tensor=2), r"decode tp choices=\[2\] conflicts .*with 1"),
    ],
)
def test_planner_snapshot_rules_fail_validation(planner_snapshots, name, edit, message):
    raw = _planner_request(planner_snapshots[name])
    edit(raw, raw["engine"]["workers"]["decode"])

    with pytest.raises((ValidationError, ValueError), match=message):
        _space(raw)


def test_planner_snapshots_without_the_role_stores_are_rejected(planner_snapshots):
    raw = _planner_request(planner_snapshots["unready"], mode="aggregated")
    with pytest.raises((ValidationError, ValueError), match=r"without a ready contains_locally_mixed, pure_prefill"):
        _space(raw)
    raw = _planner_request(planner_snapshots["aggregated"], mode="aggregated")
    raw["engine"]["workers"]["aggregated"]["parallelism"]["attention_data"] = 2
    raw["engine"]["workers"]["aggregated"]["parallelism"]["tensor"] = 1
    with pytest.raises((ValidationError, ValueError), match=r"uses 2 GPUs per replica"):
        _space(raw)


def test_unknown_hardware_takes_the_engine_and_known_regression_identity_is_pinned(planner_snapshots):
    raw = _planner_request(planner_snapshots["decode"])
    raw["engine"]["hardware"] = "b200_sxm"
    raw["engine"]["backend_version"] = "0.10.0"
    space = _space(raw)
    assert (space.hardware_sku, space.backend_version) == ("b200_sxm", "0.10.0")
    known = _space(_planner_request(planner_snapshots["known"], decode_parallelism={"preset": False, "replicas": 1}))
    assert known.backend_version == "0.19.0"
    assert known.parallel_independent_by_mode["disagg"]["decode_tp"] == [1]
