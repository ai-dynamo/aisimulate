# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small structural fits checked against raw-row regularized-loss minimization.

This deliberately slow test oracle does not use native gradients, histograms,
checkpoint statistics, or sklearn's split finder. Candidate and allocation rules
are the public online contract; losses are evaluated directly on row residuals.
"""

from __future__ import annotations

import copy
import math
import sys
from collections import deque
from dataclasses import dataclass

import pytest

from aisimulate_core.sdk import fpm_learned
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from .fpm_training_support import import_sklearn
from .fpm_training_support import isolate_training_dependencies as isolate_training_dependencies
from .test_fpm_online_integration import IDENTITY

pytestmark = pytest.mark.unit
np = pytest.importorskip("numpy")
import_sklearn()
pytest.importorskip("pyarrow")


@dataclass(frozen=True, order=True)
class Split:
    feature: int
    threshold: float
    missing_left: bool

    def goes_left(self, x):
        value = x[self.feature]
        return self.missing_left if math.isnan(value) else value <= self.threshold


def _better(gain, other):
    """Treat only the documented relative-gain margin as a strict improvement."""
    return gain - other > 1e-12 * max(abs(gain), abs(other), 1.0)


def _current_split(tree, node):
    if tree["left"][node] < 0:
        return None
    return Split(tree["feature"][node], tree["threshold"][node], tree["missing_left"][node])


def _path(tree, x):
    node = 0
    while True:
        yield node
        split = _current_split(tree, node)
        if split is None:
            break
        node = tree["left" if split.goes_left(x) else "right"][node]


def _descendants(tree, node):
    children = {node}
    if tree["left"][node] >= 0:
        children.update(_descendants(tree, tree["left"][node]))
        children.update(_descendants(tree, tree["right"][node]))
    return children


class RawLossForest:
    """Independent oracle retaining raw rows and each node's reserved slot set."""

    def __init__(self, artifact, metadata, *, rank_tolerance=0.0):
        self.artifact = copy.deepcopy(artifact)
        self.store = self.artifact["stores"]["pure_decode"]
        self.rate = metadata["learning_rate"]
        self.ridge = metadata["l2_regularization"]
        self.minimum = metadata["min_samples_leaf"]
        self.max_depth = metadata["max_depth"]
        self.rank_tolerance = rank_tolerance
        self.candidates = sorted(
            {
                Split(feature, threshold, missing_left)
                for feature, bounds in enumerate(metadata["bin_thresholds"])
                for threshold in bounds
                for missing_left in (False, True)
            }
            | {Split(feature, sys.float_info.max, False) for feature in range(len(metadata["bin_thresholds"]))}
        )
        self.regions = [
            [_descendants(tree, node) for node in range(len(tree["value"]))] for tree in self.store["trees"]
        ]
        self.rebuilds = []

    def _minimum_loss(self, residuals, members):
        # Minimize .5*sum((residual-adjustment)^2) + .5*lambda*adjustment^2.
        # This intentionally uses residual errors, not the native G^2/H gain formula.
        if not members:
            return 0.0, 0.0
        adjustment = math.fsum(residuals[index] for index in members) / (len(members) + self.ridge)
        squared_error = math.fsum((residuals[index] - adjustment) ** 2 for index in members)
        return 0.5 * (squared_error + self.ridge * adjustment**2), adjustment

    def _scores(self, x, residuals, members):
        parent_loss = self._minimum_loss(residuals, members)[0]
        scores = []
        for split in self.candidates:
            left = [index for index in members if split.goes_left(x[index])]
            right = [index for index in members if not split.goes_left(x[index])]
            if min(len(left), len(right)) < self.minimum:
                continue
            child_loss = self._minimum_loss(residuals, left)[0] + self._minimum_loss(residuals, right)[0]
            scores.append((split, parent_loss - child_loss))
        return scores

    @staticmethod
    def _best(scores):
        # Candidates are in feature/threshold/missing-direction order. Keep the
        # first candidate for tied gains when rebuilding a fresh subtree.
        winner = None
        for split, gain in scores:
            if winner is None or _better(gain, winner[1]):
                winner = split, gain
        return winner

    def _must_rebuild(self, tree, node, scores):
        incumbent = _current_split(tree, node)
        if incumbent is None:
            best = self._best(scores)
            return best is not None and _better(best[1], 0.0)
        incumbent_gain = next((gain for split, gain in scores if split == incumbent), None)
        if incumbent_gain is None:
            return True
        rank = 1 + sum(_better(gain, incumbent_gain) for _, gain in scores)
        return rank > max(1, math.ceil(self.rank_tolerance * len(scores)))

    def _rebuild(self, tree_index, root, depth, members, x, residuals):
        tree = self.store["trees"][tree_index]
        regions = self.regions[tree_index]
        reserved = set(regions[root])
        available = sorted(reserved - {root})
        for node in reserved:
            for key in ("left", "right", "feature"):
                tree[key][node] = -1
            tree["threshold"][node] = tree["value"][node] = 0.0
            tree["missing_left"][node] = True
            regions[node] = {node}

        def frontier_leaf(node_depth, node_members):
            best = None
            if self.max_depth is None or node_depth < self.max_depth:
                best = self._best(self._scores(x, residuals, node_members))
            if best is not None and not _better(best[1], 0.0):
                best = None
            return node_depth, node_members, best

        frontier = {root: frontier_leaf(depth, members)}
        while len(available) >= 2:
            winner = None
            # Sorted node IDs implement the documented tie between frontier leaves.
            for node in sorted(frontier):
                best = frontier[node][2]
                if best is not None and (winner is None or _better(best[1], frontier[winner][2][1])):
                    winner = node
            if winner is None:
                break
            node_depth, node_members, (split, _) = frontier.pop(winner)
            left, right, *available = available
            tree["left"][winner], tree["right"][winner] = left, right
            tree["feature"][winner] = split.feature
            tree["threshold"][winner] = split.threshold
            tree["missing_left"][winner] = split.missing_left
            left_members = [index for index in node_members if split.goes_left(x[index])]
            right_members = [index for index in node_members if not split.goes_left(x[index])]
            frontier[left] = frontier_leaf(node_depth + 1, left_members)
            frontier[right] = frontier_leaf(node_depth + 1, right_members)
        active = _descendants(tree, root)
        for node in active:
            regions[node] = _descendants(tree, node)
        regions[root] = reserved  # Unused capacity survives a collapse.
        self.rebuilds.append((tree_index, root, depth, len(active), len(reserved)))

    def update(self, rows):
        x = [row[0] for row in rows]
        y = [row[1] for row in rows]
        prefix = [self.store["baseline"]] * len(rows)
        for tree_index, tree in enumerate(self.store["trees"]):
            residuals = [target - prediction for target, prediction in zip(y, prefix, strict=True)]
            pending = deque([(0, 0)])
            while pending:
                node, depth = pending.popleft()
                members = [index for index, values in enumerate(x) if node in _path(tree, values)]
                if len(self.regions[tree_index][node]) > 1:
                    scores = self._scores(x, residuals, members)
                    if self._must_rebuild(tree, node, scores):
                        self._rebuild(tree_index, node, depth, members, x, residuals)
                        continue
                if tree["left"][node] >= 0:
                    pending.extend(((tree["left"][node], depth + 1), (tree["right"][node], depth + 1)))
            memberships = {}
            for index, values in enumerate(x):
                memberships.setdefault(list(_path(tree, values))[-1], []).append(index)
            for node, child in enumerate(tree["left"]):
                if child < 0:
                    members = memberships.get(node, [])
                    tree["value"][node] = self.rate * self._minimum_loss(residuals, members)[1] if members else 0.0
            for node, members in memberships.items():
                for index in members:
                    prefix[index] += tree["value"][node]

    def predict(self, x):
        raw = self.store["baseline"] + math.fsum(
            tree["value"][list(_path(tree, x))[-1]] for tree in self.store["trees"]
        )
        return max(raw, 1e-6)  # The public ms-target prediction floor.


def _iteration(x, y):
    return [
        {
            "version": 1,
            "wall_time": y / 1000.0,
            "scheduled_requests": {"num_decode_requests": int(x[0]), "sum_decode_kv_tokens": int(x[1])},
        }
    ]


@pytest.mark.parametrize("tree_count,ridge", [(2, 0.0), (3, 1.5)])
def test_deep_structural_updates_match_raw_loss_reference_through_fifo(tmp_path, tree_count, ridge):
    from sklearn.ensemble import HistGradientBoostingRegressor

    anchors = [
        ([float(count), float(kv)], float(20 + 2 * count + kv // 50 + (8 if count >= 4 and kv >= 350 else 0)))
        for count in range(1, 7)
        for kv in range(100, 601, 100)
    ]
    x = np.asarray([row[0] for row in anchors])
    y = np.asarray([row[1] for row in anchors])
    fitted = HistGradientBoostingRegressor(
        max_iter=tree_count,
        max_leaf_nodes=7,
        max_depth=4,
        min_samples_leaf=2,
        learning_rate=0.35,
        l2_regularization=ridge,
        early_stopping=False,
        random_state=0,
    ).fit(x, y)
    artifact = {
        "schema": fpm_learned.SCHEMA_NAME,
        "schema_version": 1,
        "worker_type": "decode",
        "features": ["num_decode_requests", "sum_decode_kv_tokens"],
        "target": "ms",
        "stores": {"pure_decode": fpm_learned._export_hgb(fitted)},
        "metadata": {"identity": fpm_learned._checkpoint_identity(IDENTITY, "decode")},
    }
    metadata = fpm_learned._online_store_metadata(fitted, 2)
    assert max(len(tree["value"]) for tree in artifact["stores"]["pure_decode"]["trees"]) >= 7
    checkpoint = tmp_path / "seed"
    fpm_learned._write_training_checkpoint(
        checkpoint,
        artifact,
        fpm_learned._checkpoint_identity(IDENTITY, "decode"),
        {"pure_decode": (x.tolist(), y.tolist())},
        {"pure_decode": [(index, "[]") for index in range(len(anchors))]},
        {"pure_decode": metadata},
    )
    model = RustForwardPassPerfModel.best_available(
        {
            **IDENTITY,
            "estimation_mode": "fpm_learned",
            "estimator_config": {
                "correction": {"enabled": False},
                "fpm_learned": {
                    "checkpoint_path": str(checkpoint),
                    "adaptation": "online",
                    "update_mode": "structural",
                    "derivative_refresh": "eager",
                    "recent_window": 8,
                    "update_every": 4,
                    "split_sampling_rate": 1.0,
                    "split_rank_tolerance": 0.0,
                    # RawLossForest is an independent, unit-weight reference.
                    "history_weight": None,
                    "history_weight_ramp_observations": 0,
                    "history_bin_grouping": False,
                },
            },
        }
    )
    assert model.export_learned_artifact() == artifact
    reference = RawLossForest(artifact, metadata)
    seed_capacities = [len(tree["value"]) for tree in reference.store["trees"]]
    recent = []
    coordinates = [(1, 100), (1, 600), (6, 100), (6, 600)]
    for batch_index in range(6):
        incoming = []
        for count, kv in coordinates:
            phase = batch_index // 2
            elevated = (count == 1) if phase == 0 else ((kv == 100) if phase == 1 else ((count == 1) == (kv == 100)))
            label = float(600 + 100 * batch_index if elevated else 30 + 5 * batch_index)
            observation = _iteration((count, kv), label)
            # Match the API's seconds-to-ms target conversion before reference fitting.
            incoming.append(([float(count), float(kv)], observation[0]["wall_time"] * 1000.0))
        recent = (recent + incoming)[-8:]
        model.tune_with_fpms([_iteration(*row) for row in incoming])
        reference.update(anchors + recent)
        actual = model.export_learned_artifact()["stores"]["pure_decode"]
        assert actual["baseline"] == reference.store["baseline"]
        assert len(actual["trees"]) == tree_count
        for index, (observed, expected) in enumerate(zip(actual["trees"], reference.store["trees"], strict=True)):
            assert len(observed["value"]) == seed_capacities[index]
            for field in ("left", "right", "feature", "threshold", "missing_left"):
                assert observed[field] == expected[field], (batch_index, index, field)
            assert observed["value"] == pytest.approx(expected["value"], rel=2e-10, abs=2e-10)
        for count in range(1, 7):
            for kv in (50, 150, 350, 650):
                prediction = model.estimate_forward_pass_time_ms(_iteration((count, kv), 0.0))
                assert prediction == pytest.approx(reference.predict((count, kv)), rel=2e-10, abs=2e-10)

    # This test must exercise real deep rebuilding, not merely repeat stump/leaf-only cases.
    assert any(active >= 7 for _, _, _, active, _ in reference.rebuilds)
    assert {tree_index for tree_index, *_ in reference.rebuilds} == set(range(tree_count))
    diagnostics = model.learned_online_diagnostics()["stores"]["pure_decode"]
    assert diagnostics["anchor_rows"] == 36
    assert diagnostics["recent_rows"] == 8
    assert diagnostics["evicted_observations"] == 16
    assert diagnostics["eager_updates"] == 6
    assert diagnostics["rebuilt_nodes"] == sum(active for _, _, _, active, _ in reference.rebuilds)
