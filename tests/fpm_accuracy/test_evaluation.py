# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
from fpm_accuracy.evaluate import Metric, choose_variant, evaluate_case
from fpm_accuracy.exceptions import DependencyError
from fpm_accuracy.models import aic_predictors
from fpm_accuracy.models.fpt_predictor import ForwardPassTimePredictor, Prediction
from fpm_accuracy.models.worker_regression import infer_worker_roles, regression_buckets
from fpm_accuracy.types.forward_pass import ForwardPassIteration, RequestMetrics
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
