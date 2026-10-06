# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_hf_dataset import CONFIGURATION_PATH, _build_dataset, _fpm_payload, _sha256, _write_json

from scripts.fpm_accuracy.evaluate import Metric, choose_variant, evaluate_case
from scripts.fpm_accuracy.exceptions import ConfigurationError, DependencyError
from scripts.fpm_accuracy.models import aic_predictors
from scripts.fpm_accuracy.models.aic_config import map_worker_config_to_aic
from scripts.fpm_accuracy.models.aic_fpm_database import prepare_aic_fpm_database
from scripts.fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, Prediction
from scripts.fpm_accuracy.models.worker_regression import (
    WorkerRegressionPredictor,
    infer_worker_roles,
    regression_buckets,
)
from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, RequestMetrics
from scripts.fpm_accuracy.types.worker_config import WorkerConfig
from scripts.notifications.accuracy_digest import decode_points


@pytest.fixture
def case(tmp_path):
    content = "\n".join(json.dumps(_fpm_payload(counter=index, wall_time=0.01)) for index in range(12)).encode()
    dataset = _build_dataset(
        tmp_path, protocol_id="forward-pass-measurement-v1", files=[("truth", "traffic.jsonl", content)]
    )
    return dataset.measurement_case(CONFIGURATION_PATH)


class Predictor(ForwardPassTimePredictor):
    def __init__(self, method, context, events):
        self.method, self.context, self.events = method, context, events
        self.count = 0
        self.closed = False

    @property
    def id(self):
        return self.method

    def predict(self, features):
        assert all("wall_time" not in rank for rank in features.rank_payloads)
        self.events.append((self, "predict", features.iteration_id))
        return Prediction(None if self.method == "regression" and self.count < 5 else 12)

    def tune(self, observations):
        for observation in observations:
            self.events.append((self, "tune", observation.iteration_id))
            self.count += 1

    def diagnostics(self):
        return {
            "regression_stores": [
                {"workload_kind": bucket, "ready": self.count >= 5, "retained_observations": self.count}
                for bucket in regression_buckets(self.context.worker_role)
            ]
        }

    def close(self):
        self.closed = True


def test_membership_cold_start_and_predict_before_tune(case):
    events, instances = [], []

    def factory(method, context):
        predictor = Predictor(method, context, events)
        instances.append(predictor)
        return predictor

    result = evaluate_case(case, factory=factory)
    regression = result["results"]["regression"]["metrics"]["all"]
    assert regression["measured_count"] == 12
    assert regression["predicted_count"] == 7
    assert regression["unavailable_count"] == 5
    assert regression["mape_pct"] == pytest.approx(20)
    warmup = result["results"]["warmup"]["metrics"]["all"]
    assert warmup["measured_count"] == warmup["predicted_count"] == 12
    assert result["results"]["nowarmup"]["metrics"]["all"]["unavailable_count"] == 12
    assert all(predictor.closed for predictor in instances)
    streams = [
        [event[2] for event in events if event[0] is predictor and event[1] == "predict"] for predictor in instances
    ]
    assert streams[0] == streams[1]
    regression_events = [
        (action, identity) for predictor, action, identity in events if predictor.method == "regression"
    ]
    for offset in range(0, len(regression_events), 2):
        predict, tune = regression_events[offset : offset + 2]
        assert predict[0] == "predict" and tune == ("tune", predict[1])


@pytest.mark.parametrize("execution_profile", ["full", "decoder_bounded"])
def test_v7_keeps_measurements_and_regression_without_staging_incomplete_identity(
    tmp_path, monkeypatch, execution_profile
):
    content = "\n".join(json.dumps(_fpm_payload(counter=index)) for index in range(12)).encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "traffic.jsonl", content)],
        fpm_schema_version=7,
    )
    parquet = tmp_path / CONFIGURATION_PATH / "fpm/fpm.parquet"
    table = pq.read_table(parquet)
    table = table.set_column(
        table.schema.get_field_index("execution_profile"), "execution_profile", pa.array([execution_profile])
    )
    pq.write_table(table, parquet)
    metadata_path = tmp_path / CONFIGURATION_PATH / "fpm/fpm.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["configuration_selector"]["execution_profile"] = execution_profile
    metadata["parquet_sha256"] = _sha256(parquet)
    _write_json(metadata_path, metadata)
    manifest_path = tmp_path / CONFIGURATION_PATH / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert "model_config_sha256" not in manifest and "execution_profile" not in manifest
    manifest["fpm"][0]["sha256"] = _sha256(parquet)
    _write_json(manifest_path, manifest)
    # Exercise the real FPM adapter and staging path without an installed SDK.
    # Reaching native model construction would fail on this sentinel.
    monkeypatch.setattr(aic_predictors, "_import_aisim_forward_pass_perf_model", lambda: SimpleNamespace())
    case = dataset.measurement_case(CONFIGURATION_PATH)
    with pytest.raises(DependencyError, match="schema v7"):
        prepare_aic_fpm_database({}, case.fpm_artifacts[0])

    def factory(method, context):
        if method == "aic-fpm":
            # No native dependency: the real staging boundary must refuse v7
            # before dropping its execution identity or building an overlay.
            return aic_predictors.AicFpmPredictor.create(context)
        return Predictor(method, context, [])

    result = evaluate_case(case, factory=factory)
    warmup = result["results"]["warmup"]
    assert warmup["status"] == "unsupported_predictor"
    assert warmup["metrics"]["all"]["measured_count"] == warmup["metrics"]["all"]["unavailable_count"] == 12
    assert warmup["metrics"]["all"]["error_count"] == 0
    assert result["results"]["regression"]["metrics"]["all"]["predicted_count"] == 7


def test_worker_state_is_isolated_and_full_rank_roles_are_used(case):
    observations = tuple(
        replace(
            item,
            iteration=ForwardPassIteration(
                tuple(replace(rank, worker_id=f"worker-{index % 2}") for rank in item.iteration.ranks)
            ),
        )
        for index, item in enumerate(case.observations)
    )
    instances = []

    def factory(method, context):
        predictor = Predictor(method, context, [])
        instances.append(predictor)
        return predictor

    result = evaluate_case(replace(case, observations=observations), factory=factory)
    assert result["results"]["regression"]["metrics"]["all"]["unavailable_count"] == 10
    assert len([item for item in instances if item.method == "regression"]) == 2
    base = observations[0].iteration.ranks[0]
    ranks = (
        replace(base, dp_rank=0),
        replace(base, dp_rank=1, scheduled=RequestMetrics(num_prefill_requests=1, sum_prefill_tokens=32)),
    )
    assert infer_worker_roles([ForwardPassIteration(ranks)]) == {base.worker_id: "aggregated"}


def test_predictor_failures_remain_in_denominator(case):
    def factory(*args):
        raise ValueError("unsupported model on this release")

    result = evaluate_case(case, factory=factory)
    metric = result["results"]["warmup"]["metrics"]["all"]
    assert metric["error_count"] == metric["measured_count"] == 12
    assert metric["mape_pct"] is None
    assert "unsupported model" not in json.dumps(result)


@pytest.mark.parametrize("canonical", [True, False])
def test_dcp_keeps_regression_and_reports_native_fpm_as_unsupported(tmp_path, monkeypatch, canonical):
    content = "\n".join(json.dumps(_fpm_payload(counter=index)) for index in range(12)).encode()
    dataset = _build_dataset(
        tmp_path,
        protocol_id="forward-pass-measurement-v1",
        files=[("truth", "traffic.jsonl", content)],
        tp=8,
        dcp=8,
    )
    case = dataset.measurement_case(CONFIGURATION_PATH)
    requests = []

    @dataclass(frozen=True)
    class Config:
        worker_type: str
        estimation_mode: str = "auto"
        fallback_policy: str = "deny"
        estimator_config: dict = field(default_factory=lambda: {"fpm_regression": {"sampling": {}, "fit": {}}})

        def to_dict(self):
            return asdict(self)

        @classmethod
        def from_legacy_engine_config(cls, config, worker_type, options):
            assert config["tp_size"] == 8
            assert config["cp_size"] == 1
            return cls(worker_type)

    class Model:
        def __init__(self, worker_type):
            self.worker_type = worker_type
            self.count = 0

        @classmethod
        def normalize_config(cls, config):
            return config.to_dict()

        @classmethod
        def best_available(cls, config):
            assert canonical
            assert config["estimation_mode"] == "fpm_regression"
            requests.append(config)
            return cls(config["worker_type"])

        @classmethod
        def from_regression(cls, worker_type, options):
            pytest.fail("older wheels must not silently run a different regression policy")

        @classmethod
        def from_native(cls, *args):
            pytest.fail("native FPM must reject unsupported DCP before construction")

        def regression_store_diagnostics(self):
            return [
                {"workload_kind": bucket, "ready": self.count >= 5, "retained_observations": self.count}
                for bucket in regression_buckets(self.worker_type)
            ]

        def estimate_forward_pass_time_ms(self, payload):
            return 10 if self.count >= 5 else None

        def tune_with_fpms(self, observations):
            self.count += len(observations)

        def diagnostics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(aic_predictors, "_import_aisim_forward_pass_perf_model", lambda: Model)
    monkeypatch.setattr(aic_predictors, "_canonical_config_type", lambda: Config if canonical else None)
    result = evaluate_case(case)
    warmup = result["results"]["warmup"]
    assert warmup["status"] == "unsupported_predictor"
    assert warmup["metrics"]["all"]["measured_count"] == warmup["metrics"]["all"]["unavailable_count"] == 12
    assert warmup["metrics"]["all"]["error_count"] == 0
    assert result["results"]["regression"]["status"] == ("evaluated" if canonical else "unsupported_predictor")
    regression = result["results"]["regression"]["metrics"]["all"]
    assert regression["measured_count"] == 12
    assert regression["predicted_count"] == (7 if canonical else 0)
    assert regression["unavailable_count"] == (5 if canonical else 12)
    assert regression["error_count"] == regression["tuning_error_count"] == 0
    assert len(requests) == int(canonical)
    assert case.configuration.worker_config_record.config.parallelism.decode_context_parallel_size == 8


def test_missing_worker_stores_are_logged_before_scoring(case, capsys):
    instances = []

    class MissingStore(Predictor):
        def diagnostics(self):
            return {"regression_stores": [{"workload_kind": "unexpected"}]}

    def factory(method, context):
        predictor = MissingStore(method, context, []) if method == "regression" else Predictor(method, context, [])
        instances.append(predictor)
        return predictor

    result = evaluate_case(case, factory=factory)
    metric = result["results"]["regression"]["metrics"]["all"]
    assert metric["error_count"] == metric["measured_count"] == 12
    assert result["results"]["regression"]["status"] == "predictor_error"
    assert all(predictor.closed for predictor in instances)
    assert "missing ['pure_decode']; available: ['unexpected']" in capsys.readouterr().out
    assert "unexpected" not in json.dumps(result)


@pytest.mark.parametrize("source", ["schema", "embedded", "override"])
@pytest.mark.parametrize(
    "field,precision",
    [
        ("weight_dtype", "weights"),
        ("moe_dtype", "experts"),
        ("activation_dtype", "activations"),
        ("kv_cache_dtype", "kv_cache"),
    ],
)
def test_worker_precision_never_silently_falls_back(case, source, field, precision):
    record = case.configuration.worker_config_record
    payload = record.config.model_dump()
    overrides = None
    if source == "schema":
        payload["aic_engine_config"] = None
        payload["precision"][precision] = "unrecognized-precision"
    elif source == "embedded":
        payload["aic_engine_config"][field] = "unrecognized-precision"
    else:
        overrides = {field: "unrecognized-precision"}
    record = replace(record, config=WorkerConfig.model_validate(payload))
    with pytest.raises(ConfigurationError, match="unsupported AISim dtype"):
        map_worker_config_to_aic(record, overrides)


def test_known_precision_aliases_and_absence_are_preserved(case):
    result = map_worker_config_to_aic(
        case.configuration.worker_config_record,
        {
            "weight_dtype": "BF16",
            "moe_dtype": "w4a16_mxfp4",
            "activation_dtype": None,
            "kv_cache_dtype": "fp16",
        },
    )
    assert result["weight_dtype"] == "bfloat16"
    assert result["moe_dtype"] == "w4a16_mxfp4"
    assert result["activation_dtype"] is None
    assert result["kv_cache_dtype"] == "float16"


@pytest.mark.parametrize("alias", ["fp8_e4m3", "fp8_e5m2"])
def test_recorded_fp8_cache_aliases_only_apply_to_kv(case, alias):
    record = case.configuration.worker_config_record
    assert map_worker_config_to_aic(record, {"kv_cache_dtype": alias})["kv_cache_dtype"] == "fp8"
    with pytest.raises(ConfigurationError, match="unsupported AISim dtype"):
        map_worker_config_to_aic(record, {"weight_dtype": alias})


def test_legacy_regression_is_unavailable_without_changing_measurement_membership(case, monkeypatch):
    class LegacyModel:
        @classmethod
        def from_regression(cls, options=None):
            raise AssertionError("must not use the legacy shared regression store")

    monkeypatch.setattr(aic_predictors, "_import_aisim_forward_pass_perf_model", lambda: LegacyModel)

    def factory(method, context):
        if method == "regression":
            with pytest.raises(DependencyError, match="worker-scoped"):
                aic_predictors.AicRegressionPredictor.create(context)
            return aic_predictors.AicRegressionPredictor.create(context)
        return Predictor(method, context, [])

    result = evaluate_case(case, factory=factory)["results"]
    assert result["regression"]["status"] == "unsupported_predictor"
    metric = result["regression"]["metrics"]["all"]
    assert metric["measured_count"] == metric["unavailable_count"] == 12
    assert metric["error_count"] == metric["predicted_count"] == 0
    assert result["warmup"]["metrics"]["all"]["predicted_count"] == 12


@pytest.fixture
def recommended_regression_config():
    report = Path(__file__).resolve().parents[2] / "docs/fpm-lazy-gym-results.json"
    return json.loads(report.read_text())["signed_lazy_estimator_config"]["fpm_regression"]


@pytest.mark.parametrize("dcp", [1, 8])
@pytest.mark.parametrize("role", ["prefill", "decode", "aggregated"])
def test_real_regression_uses_recommended_gym_config_only(tmp_path, dcp, role, recommended_regression_config):
    sdk = pytest.importorskip("aisimulate_core.sdk")
    from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext

    case = _build_dataset(tmp_path, protocol_id=None, files=[], tp=8, dcp=dcp).measurement_case(CONFIGURATION_PATH)
    context = PredictorContext(
        worker=case.configuration.worker_config_record,
        worker_role=role,
    )
    predictor = aic_predictors.AicRegressionPredictor.create(context)
    try:
        config = predictor.diagnostics()["provenance"]["config"]
        assert config["tp"] == 8
        assert config["worker_type"] == role
        assert config["estimation_mode"] == "fpm_regression"
        assert config["fallback_policy"] == "deny"
        actual = config["estimator_config"]["fpm_regression"]
        expected = recommended_regression_config
        assert {"axes": ["attention", "moe"], **actual["sampling"]} == expected["sampling"]
        assert actual["min_observations"] == expected["min_observations"]
        for key, value in expected["fit"].items():
            assert actual["fit"][key] == value

        shared = sdk.RustForwardPassPerfModel.normalize_config({**config, "estimator_config": {}})
        shared_regression = shared["estimator_config"]["fpm_regression"]
        assert shared_regression["sampling"]["bins_per_axis"] == [4, 4]
        shared_linear = shared_regression["fit"].get("linear", {})
        assert shared_linear.get("non_negative", True) is True
        assert shared_linear.get("update_policy", {"kind": "always"}) == {"kind": "always"}
    finally:
        predictor.close()


@pytest.mark.parametrize(
    "grid_options,bins", [({}, [4, 1]), ({"bucket_count": 9}, [3, 3]), ({"bucket_shape": [2, 3]}, [2, 3])]
)
def test_real_regression_preserves_explicit_controls(case, grid_options, bins, recommended_regression_config):
    pytest.importorskip("aisimulate_core.sdk")
    from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext

    context = PredictorContext(
        worker=case.configuration.worker_config_record,
        worker_role="decode",
        options={"max_observations": 32, "min_observations": 6, "regression_ridge_scale": 0.125, **grid_options},
        engine_config_overrides={
            "extra": {"estimator_config": json.dumps({"fpm_regression": {"fit": {"rebuild_interval": 128}}})}
        },
    )
    predictor = aic_predictors.AicRegressionPredictor.create(context)
    try:
        config = predictor.diagnostics()["provenance"]["config"]["estimator_config"]["fpm_regression"]
        assert {"axes": ["attention", "moe"], **config["sampling"]} == {
            "axes": ["attention", "moe"],
            "bins_per_axis": bins,
            "max_observations": 32,
        }
        assert config["min_observations"] == 6
        assert config["fit"]["linear"] == recommended_regression_config["fit"]["linear"]
        assert config["fit"]["singular_ridge_scale"] == 0.125
        assert config["fit"]["rebuild_interval"] == 128
    finally:
        predictor.close()


@pytest.mark.parametrize(
    "sampling",
    [
        pytest.param({"bins_per_axis": [4, 4]}, id="explicit-default-grid"),
        pytest.param({"bins_per_axis": [2, 7]}, id="custom-grid"),
        pytest.param({"axes": ["n", "moe"], "bins_per_axis": [2, 3]}, id="custom-axes"),
        pytest.param({"max_observations": 32}, id="capacity-only"),
        pytest.param({}, id="empty-block"),
    ],
)
def test_real_regression_preserves_explicit_canonical_sampling(case, sampling, recommended_regression_config):
    sdk = pytest.importorskip("aisimulate_core.sdk")
    from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext

    context = PredictorContext(
        worker=case.configuration.worker_config_record,
        worker_role="decode",
        engine_config_overrides={"extra": {"estimator_config": json.dumps({"fpm_regression": {"sampling": sampling}})}},
    )
    # The Rust migration owns default resolution within an explicit sampling
    # block, including when the caller supplied only capacity or an empty block.
    migrated = sdk.ForwardPassPerfModelConfig.from_legacy_engine_config(
        map_worker_config_to_aic(context.worker, context.engine_config_overrides), context.worker_role, context.options
    )
    predictor = aic_predictors.AicRegressionPredictor.create(context)
    try:
        config = predictor.diagnostics()["provenance"]["config"]["estimator_config"]["fpm_regression"]
        assert config["sampling"] == migrated.estimator_config["fpm_regression"]["sampling"]
        assert config["fit"]["linear"] == recommended_regression_config["fit"]["linear"]
    finally:
        predictor.close()


@pytest.mark.parametrize("options", [{"bucket_count": 9}, {"bucket_shape": [3, 2]}])
def test_real_regression_rejects_conflicting_canonical_and_legacy_grids(case, options):
    pytest.importorskip("aisimulate_core.sdk")
    from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext

    context = PredictorContext(
        worker=case.configuration.worker_config_record,
        worker_role="decode",
        options=options,
        engine_config_overrides={
            "extra": {"estimator_config": json.dumps({"fpm_regression": {"sampling": {"bins_per_axis": [2, 3]}}})}
        },
    )
    with pytest.raises(ValueError, match="conflict"):
        aic_predictors.AicRegressionPredictor.create(context)


def test_worker_regression_reports_resolved_native_grid(case):
    pytest.importorskip("aisimulate_core.sdk")
    from scripts.fpm_accuracy.evaluate import create_predictor
    from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext

    roles = infer_worker_roles(item.iteration for item in case.observations)
    context = PredictorContext(worker=case.configuration.worker_config_record, worker_role=case.worker_role)
    predictor = WorkerRegressionPredictor(context, roles, create_predictor)
    try:
        assert predictor.diagnostics()["spatial_bucket_count"] == 4
        assert predictor.diagnostics()["max_observations_per_store"] == 64
    finally:
        predictor.close()


def test_worker_regression_preserves_canonical_sampling_and_explicit_options(case):
    pytest.importorskip("aisimulate_core.sdk")
    from scripts.fpm_accuracy.evaluate import create_predictor
    from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext

    sampling = {"axes": ["n", "moe"], "bins_per_axis": [2, 3], "max_observations": 32}
    context = PredictorContext(
        worker=case.configuration.worker_config_record,
        worker_role=case.worker_role,
        options={"min_observations": 6},
        engine_config_overrides={"extra": {"estimator_config": json.dumps({"fpm_regression": {"sampling": sampling}})}},
    )
    calls = []

    def factory(method, child_context):
        calls.append(dict(child_context.options))
        return create_predictor(method, child_context)

    roles = infer_worker_roles(item.iteration for item in case.observations)
    predictor = WorkerRegressionPredictor(context, roles, factory)
    try:
        assert calls == [dict(context.options)] * len(roles)
        diagnostics = predictor.diagnostics()
        assert diagnostics["max_observations_per_store"] == 32
        assert diagnostics["spatial_bucket_count"] == 6
        for child in diagnostics["worker_diagnostics"].values():
            regression = child["provenance"]["config"]["estimator_config"]["fpm_regression"]
            assert regression["sampling"] == sampling
            assert regression["min_observations"] == 6
    finally:
        predictor.close()


@pytest.mark.parametrize("mode", ["fpm", "regression"])
@pytest.mark.parametrize("canonical", [True, False])
def test_predictor_uses_canonical_api_or_older_wheel_adapter(case, tmp_path, monkeypatch, mode, canonical):
    from scripts.fpm_accuracy.models.fpt_predictor import PredictorContext

    if canonical:
        pytest.importorskip("aisimulate_core.sdk")
    else:
        monkeypatch.setattr(aic_predictors, "_canonical_config_type", lambda: None)
    calls = []
    normalizations = []

    class Model:
        @classmethod
        def normalize_config(cls, config):
            normalizations.append(config)
            return config.to_dict()

        @classmethod
        def best_available(cls, config):
            assert canonical
            calls.append(config.to_dict() if hasattr(config, "to_dict") else config)
            return cls()

        @classmethod
        def from_native(cls, config, options):
            assert not canonical
            calls.append((config, options))
            return cls()

        @classmethod
        def from_regression(cls, worker_type, options):
            pytest.fail("older wheels must not silently run a different regression policy")

        def regression_store_diagnostics(self):
            return []

        def close(self):
            pass

    monkeypatch.setattr(aic_predictors, "_import_aisim_forward_pass_perf_model", lambda: Model)
    monkeypatch.setattr(
        aic_predictors,
        "prepare_aic_fpm_database",
        lambda *args: SimpleNamespace(
            systems_root=tmp_path,
            close=lambda: None,
        ),
    )
    context = PredictorContext(
        worker=case.configuration.worker_config_record,
        worker_role="decode",
        options={"max_observations": 32},
        fpm_artifact=object(),
    )
    cls = aic_predictors.AicFpmPredictor if mode == "fpm" else aic_predictors.AicRegressionPredictor
    if mode == "regression" and not canonical:
        with pytest.raises(DependencyError):
            cls.create(context)
        assert not calls and not normalizations
        return
    predictor = cls.create(context)
    predictor.close()
    assert len(calls) == 1
    assert len(normalizations) == int(mode == "regression")
    if canonical:
        assert calls[0]["worker_type"] == "decode"
        assert calls[0]["estimation_mode"] == ("fpm_interpolation" if mode == "fpm" else "fpm_regression")
        assert calls[0]["fallback_policy"] == "deny"
        assert calls[0]["estimator_config"]["fpm_regression"]["sampling"]["max_observations"] == 32
        if mode == "fpm":
            assert calls[0]["systems_paths"] == [str(tmp_path)]
            regression = calls[0]["estimator_config"]["fpm_regression"]
            assert regression["sampling"]["bins_per_axis"] == [4, 4]
            linear = regression["fit"].get("linear", {})
            assert linear.get("non_negative", True) is True
            assert linear.get("update_policy", {"kind": "always"}) == {"kind": "always"}


@pytest.mark.parametrize("support", ["missing", "reject", "ignore_grid", "ignore_linear"])
def test_regression_never_falls_back_when_wheel_cannot_apply_gym_policy(case, monkeypatch, support):
    @dataclass(frozen=True)
    class Config:
        worker_type: str
        estimation_mode: str = "auto"
        fallback_policy: str = "deny"
        estimator_config: dict = field(default_factory=lambda: {"fpm_regression": {"sampling": {}, "fit": {}}})

        @classmethod
        def from_legacy_engine_config(cls, config, worker_type, options):
            return cls(worker_type)

        def to_dict(self):
            return asdict(self)

    class Model:
        @staticmethod
        def regression_store_diagnostics():
            return []

        @staticmethod
        def normalize_config(config):
            if support == "reject":
                raise ValueError("unknown field linear")
            result = deepcopy(config.to_dict())
            regression = result["estimator_config"]["fpm_regression"]
            if support == "ignore_grid":
                regression["sampling"]["bins_per_axis"] = [4, 4]
            elif support == "ignore_linear":
                regression["fit"]["linear"] = {
                    "feature_axes": ["attention", "moe"],
                    "non_negative": True,
                    "update_policy": {"kind": "always"},
                }
            return result

        @staticmethod
        def best_available(config):
            pytest.fail("incompatible controls must be rejected before model construction")

        @staticmethod
        def from_regression(*args):
            pytest.fail("legacy regression must not silently replace the recommended policy")

    if support == "missing":
        monkeypatch.setattr(Model, "normalize_config", None)
    monkeypatch.setattr(aic_predictors, "_import_aisim_forward_pass_perf_model", lambda: Model)
    monkeypatch.setattr(aic_predictors, "_canonical_config_type", lambda: Config)

    def factory(method, context):
        if method == "regression":
            return aic_predictors.AicRegressionPredictor.create(context)
        return Predictor(method, context, [])

    results = evaluate_case(case, factory=factory)["results"]
    assert results["regression"]["status"] == "unsupported_predictor"
    metric = results["regression"]["metrics"]["all"]
    assert metric["measured_count"] == metric["unavailable_count"] == 12
    assert metric["predicted_count"] == metric["error_count"] == metric["tuning_error_count"] == 0
    assert results["warmup"]["metrics"]["all"]["predicted_count"] == 12


def test_micro_mape_and_variant_order():
    metric = Metric()
    metric.add(10, 20, False, False)
    metric.add(100, 100, False, False)
    metric.add(10, None, False, False)
    assert metric.export()["mape_pct"] == 50

    def result(identity, predicted, mape):
        return {
            "artifact": {"id": identity},
            "metrics": {"all": {"measured_count": 10, "predicted_count": predicted, "mape_pct": mape}},
        }

    assert (
        choose_variant([result("low-coverage", 5, 0), result("b", 10, 20), result("a", 10, 20)])["artifact"]["id"]
        == "a"
    )
    assert choose_variant([result("b", 10, 20), result("c", 10, 10)])["artifact"]["id"] == "c"


def test_bad_measurement_fails_campaign(case):
    with pytest.raises(ValueError, match="unique IDs"):
        evaluate_case(replace(case, observations=(case.observations[0], case.observations[0])))


def test_notification_evidence_keeps_point_failures_without_changing_public_result(case):
    """The notification sidecar is opt-in and must not change prediction/tuning order."""

    def factory(method, context):
        return Predictor(method, context, [])

    expected = evaluate_case(case, factory=factory)
    evidence = {}
    actual = evaluate_case(case, factory=factory, comparison=evidence)
    assert actual == expected
    points = evidence[case.configuration_id + "/" + case.configuration.snapshot_id]["methods"]
    assert set(points) == {"warmup", "nowarmup", "regression"}
    for method, packed in points.items():
        samples = decode_points(packed)
        metric = actual["results"][method]["metrics"]["all"]
        assert len(samples) == metric["measured_count"]
        values = [v for v in samples if v >= 0]
        assert len(values) == metric["predicted_count"]
        if values:
            assert sum(values) / len(values) == pytest.approx(metric["mape_pct"])
    assert "_points" not in json.dumps(actual)
