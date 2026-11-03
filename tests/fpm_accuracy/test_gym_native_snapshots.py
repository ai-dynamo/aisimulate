# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mid-evaluation snapshots replace only the Gym scope's native model.

The real scorer retains its split and five-second tick state. Native models own
seeded prediction/update state; the wrapper owns first fitting. The update_every=4 cases are explicit test variants
for pending rows; production Gym uses update_every=1 and is not relabeled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_evaluation import case as source_case_fixture
from test_learned_scenario import observation
from test_native_learned_scenario import native_backend as native_backend_fixture

from scripts.fpm_accuracy.models.learned_scenario import LearnedScenarioPredictor
from scripts.fpm_accuracy.scenarios import score_scenario, split_observations
from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, RequestMetrics

source_case = source_case_fixture
native_backend = native_backend_fixture


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bits(value):
    return None if value is None else float(value).hex()


def _online_semantics(diagnostics):
    value = copy.deepcopy(diagnostics)
    value.pop("training_state_bytes", None)
    for store in value["stores"].values():
        store.pop("history_grouping_ms", None)
        store.pop("training_state_bytes", None)
    return value


@pytest.fixture
def snapshot_case(source_case):
    rows = []
    for index in range(360):
        iteration = observation(index, actual=4 + (index % 17) / 5)
        rank = replace(
            iteration.ranks[0],
            configuration_id=source_case.configuration_id,
            scheduled=RequestMetrics(
                num_decode_requests=1 + index % 4,
                sum_decode_kv_tokens=(1 + index % 4) * (64 + 32 * (index % 7)),
            ),
        )
        rows.append(
            replace(
                source_case.observations[0],
                observation_id=f"snapshot-{index}",
                order=index,
                source_row=index + 1,
                iteration=ForwardPassIteration.single_rank(rank),
                event_time_ms=index * 1000,
            )
        )
    return replace(source_case, observations=tuple(rows), worker_role="aggregated")


class SnapshotProbe:
    """Record the real scorer's calls and replace exactly one owned model."""

    def __init__(self, inner, native, destination, *, after, on_tune, remove_seed):
        self.inner, self.native, self.destination = inner, native, destination
        self.after, self.on_tune, self.remove_seed = after, on_tune, remove_seed
        self.predictions, self.attempts, self.accepted, self.errors = [], [], [], []
        self.replaced = False
        self.fixed_oracle = None
        self.final_online = None

    def predict(self, features):
        assert all("wall_time" not in row for row in features.aic_payload())
        value = self.inner.predict(features)
        self.predictions.append((features.iteration_id, _bits(value.value_ms)))
        if self.fixed_oracle is not None:
            expected = self.fixed_oracle.estimate_forward_pass_time_ms(features.aic_payload())
            assert _bits(value.value_ms) == _bits(expected)
        return value

    def tune(self, observations):
        assert len(observations) == 1
        row = observations[0]
        assert self.predictions[-1][0] == row.iteration_id
        record = (row.iteration_id, _digest(row.tuning_payload()))
        self.attempts.append(record)
        try:
            self.inner.tune(observations)
        except Exception as error:
            self.errors.append(str(error))
            raise
        self.accepted.append(record)
        if self.after is not None and len(self.accepted) == self.after:
            self._replace_native()

    def _replace_native(self):
        assert not self.replaced and len(self.inner._scopes) == 1
        scope = next(iter(self.inner._scopes.values()))
        wrapper_state = copy.deepcopy({key: value for key, value in vars(scope).items() if key != "model"})
        self.saved_online = _online_semantics(scope.model.learned_online_diagnostics())
        self.predictions_at_snapshot = len(self.predictions)
        saved = scope.model.save_state_snapshot(self.destination)
        assert _online_semantics(scope.model.learned_online_diagnostics()) == self.saved_online
        relocated = self.destination.with_name(self.destination.name + "-moved")
        self.destination.rename(relocated)
        saved["estimator_config"]["state_snapshot"].update(path=str(relocated), on_tune=self.on_tune)
        if self.remove_seed:
            # This root belongs exclusively to this wrapper fixture. The
            # snapshot lives under pytest's other directory, outside it.
            owned = Path(self.inner._temporary.name).resolve()
            originals = [
                Path(config["estimator_config"]["fpm_learned"]["checkpoint_path"])
                for config in self.inner.seed_configurations().values()
            ]
            assert originals and all(path.resolve().is_relative_to(owned) for path in originals)
            shutil.rmtree(owned)
            assert all(not path.exists() for path in originals)
        previous = scope.model
        restored = self.native.best_available(saved)
        assert _online_semantics(restored.learned_online_diagnostics()) == self.saved_online
        # Deliberately keep all wrapper fields, source bookkeeping, and counters.
        # The scorer's active split and clock are never reconstructed.
        scope.model = restored
        previous.close()
        assert {key: value for key, value in vars(scope).items() if key != "model"} == wrapper_state
        if self.on_tune == "reject":
            self.fixed_oracle = self.native.best_available(state_snapshot=relocated)
        self.replaced = True

    def diagnostics(self):
        scope = next(iter(self.inner._scopes.values()))
        self.final_online = _online_semantics(scope.model.learned_online_diagnostics())
        return self.inner.diagnostics()

    def close(self):
        try:
            self.inner.close()
        finally:
            if self.fixed_oracle is not None:
                self.fixed_oracle.close()


def _run(case, backend, output, *, scenario, after=None, on_tune="continue", update_every=1, remove_seed=False):
    native, request, trainer = backend
    instances = []

    class TestConfiguredPredictor(LearnedScenarioPredictor):
        def _configuration(self, source):
            config = super()._configuration(source)
            if update_every != 1 and source is not None:
                # This test variant exercises the public pending-row contract.
                # It is not a production recipe or a published Gym result.
                config["estimator_config"]["fpm_learned"]["update_every"] = update_every
                config = native.normalize_config(config)
            return config

    def factory(method, context, *, training_iterations=(), online=False):
        assert method == "gbdt" and online
        inner = TestConfiguredPredictor("gbdt", native, request, trainer, training_iterations, online=True)
        probe = SnapshotProbe(inner, native, output, after=after, on_tune=on_tune, remove_seed=remove_seed)
        instances.append(probe)
        return probe

    result = score_scenario(case, "gbdt", scenario, factory=factory)
    assert result["status"] == "evaluated" and len(instances) == 1
    assert result["metrics"]["all"]["error_count"] == 0
    return result, instances[0]


def _scoring_semantics(result):
    return {
        key: result[key]
        for key in (
            "status",
            "training_count",
            "test_count",
            "training_population_sha256",
            "scoring_population_sha256",
            "online_update_count",
            "metrics",
            "cold_start",
            "online_schedule",
        )
    }


@pytest.mark.parametrize("on_tune", ["continue", "reject"])
@pytest.mark.parametrize(
    "scenario,after,update_every,remove_seed",
    [
        pytest.param("c", 23, 1, False, id="ready-production-recipe"),
        pytest.param("c", 23, 4, False, id="ready-pending-test-variant"),
        pytest.param("bc", 3, 4, True, id="seeded-pending-original-checkpoint-removed"),
    ],
)
def test_mid_evaluation_native_snapshot_preserves_gym_transport(
    snapshot_case, native_backend, tmp_path, scenario, after, update_every, remove_seed, on_tune
):
    baseline, uninterrupted = _run(
        snapshot_case, native_backend, tmp_path / "unused", scenario=scenario, update_every=update_every
    )
    candidate, restored = _run(
        snapshot_case,
        native_backend,
        tmp_path / "snapshot",
        scenario=scenario,
        after=after,
        on_tune=on_tune,
        update_every=update_every,
        remove_seed=remove_seed,
    )
    assert restored.replaced
    testing = split_observations(snapshot_case.observations, scenario)[1]
    expected_ids = [row.iteration.iteration_id for row in testing[5::5]]
    assert [row[0] for row in uninterrupted.attempts] == expected_ids
    assert restored.attempts == uninterrupted.attempts
    assert [row[0] for row in restored.predictions] == [row.iteration.iteration_id for row in testing]
    boundary = restored.predictions_at_snapshot
    assert restored.predictions[:boundary] == uninterrupted.predictions[:boundary]
    if update_every == 4:
        assert restored.saved_online["stores"]["pure_decode"]["pending_rows"] == 3
    else:
        assert restored.saved_online["stores"]["pure_decode"]["pending_rows"] == 0
    if on_tune == "continue":
        assert not restored.errors and not uninterrupted.errors
        assert restored.predictions == uninterrupted.predictions
        assert restored.accepted == uninterrupted.accepted
        assert restored.final_online == uninterrupted.final_online
        assert _scoring_semantics(candidate) == _scoring_semantics(baseline)
    else:
        assert restored.accepted == uninterrupted.accepted[:after]
        assert len(restored.errors) == len(restored.attempts) - after
        assert all("on_tune" in error for error in restored.errors)
        assert restored.final_online == restored.saved_online
        for key in ("training_count", "test_count", "training_population_sha256", "scoring_population_sha256"):
            assert candidate[key] == baseline[key]
        schedule = candidate["online_schedule"]
        assert schedule["submitted_observation_count"] == baseline["online_schedule"]["submitted_observation_count"]
        assert schedule["update_error_count"] == len(restored.errors)
