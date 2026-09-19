# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from fpm_accuracy.evaluate import Metric, choose_variant, evaluate_case
from fpm_accuracy.exceptions import ConfigurationError, DependencyError
from fpm_accuracy.models import aic_predictors
from fpm_accuracy.models.aic_config import map_worker_config_to_aic
from fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, Prediction
from fpm_accuracy.models.worker_regression import infer_worker_roles, regression_buckets
from fpm_accuracy.types.forward_pass import ForwardPassIteration, RequestMetrics
from fpm_accuracy.types.worker_config import WorkerConfig
from test_hf_dataset import CONFIGURATION_PATH, _build_dataset, _fpm_payload


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
        def best_available(cls, config):
            assert canonical
            assert config.estimation_mode == "fpm_regression"
            requests.append(config)
            return cls(config.worker_type)

        @classmethod
        def from_regression(cls, worker_type, options):
            assert not canonical
            requests.append(worker_type)
            return cls(worker_type)

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
    assert result["results"]["regression"]["status"] == "evaluated"
    regression = result["results"]["regression"]["metrics"]["all"]
    assert regression["measured_count"] == 12
    assert regression["predicted_count"] == 7
    assert regression["unavailable_count"] == 5
    assert regression["error_count"] == regression["tuning_error_count"] == 0
    assert len(requests) == 1
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


@pytest.mark.parametrize("dcp", [1, 8])
def test_real_regression_uses_canonical_identity_and_options(tmp_path, dcp):
    pytest.importorskip("aisimulate_core.sdk")
    from fpm_accuracy.models.fpt_predictor import PredictorContext

    case = _build_dataset(tmp_path, protocol_id=None, files=[], tp=8, dcp=dcp).measurement_case(CONFIGURATION_PATH)
    context = PredictorContext(
        worker=case.configuration.worker_config_record,
        worker_role="decode",
        options={"max_observations": 32, "min_observations": 6},
    )
    predictor = aic_predictors.AicRegressionPredictor.create(context)
    try:
        config = predictor.diagnostics()["provenance"]["config"]
        assert config["tp"] == 8
        assert config["worker_type"] == "decode"
        assert config["estimation_mode"] == "fpm_regression"
        assert config["fallback_policy"] == "deny"
        assert config["estimator_config"]["fpm_regression"]["sampling"]["max_observations"] == 32
        assert config["estimator_config"]["fpm_regression"]["min_observations"] == 6
    finally:
        predictor.close()


@pytest.mark.parametrize("mode", ["fpm", "regression"])
@pytest.mark.parametrize("canonical", [True, False])
def test_predictor_uses_canonical_api_or_older_wheel_adapter(case, tmp_path, monkeypatch, mode, canonical):
    from fpm_accuracy.models.fpt_predictor import PredictorContext

    if canonical:
        pytest.importorskip("aisimulate_core.sdk")
    else:
        monkeypatch.setattr(aic_predictors, "_canonical_config_type", lambda: None)
    calls = []

    class Model:
        @classmethod
        def best_available(cls, config):
            assert canonical
            calls.append(config.to_dict())
            return cls()

        @classmethod
        def from_native(cls, config, options):
            assert not canonical
            calls.append((config, options))
            return cls()

        @classmethod
        def from_regression(cls, worker_type, options):
            assert not canonical
            calls.append((worker_type, options))
            return cls()

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
    predictor = cls.create(context)
    predictor.close()
    assert len(calls) == 1
    if canonical:
        assert calls[0]["worker_type"] == "decode"
        assert calls[0]["estimation_mode"] == ("fpm_interpolation" if mode == "fpm" else "fpm_regression")
        assert calls[0]["fallback_policy"] == "deny"
        assert calls[0]["estimator_config"]["fpm_regression"]["sampling"]["max_observations"] == 32
        if mode == "fpm":
            assert calls[0]["systems_paths"] == [str(tmp_path)]


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
