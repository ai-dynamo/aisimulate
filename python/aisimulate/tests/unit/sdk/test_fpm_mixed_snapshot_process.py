# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native priors and continuation survive removal of all learned/timing sources."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from aisimulate_core.sdk import RustForwardPassPerfModel, fpm_learned

from . import test_fpm_mixed_composition_integration as source_fixture

mixed_seed = source_fixture.mixed_seed
phase_roots = source_fixture.phase_roots
pytestmark = pytest.mark.unit


def _online_semantics(value):
    """Compare admission IDs, counters and pending state, excluding host costs."""
    value = json.loads(json.dumps(value))
    value.pop("training_state_bytes", None)
    for store in value["stores"].values():
        store.pop("training_state_bytes", None)
        store.pop("history_grouping_ms", None)
    return value


_CHILD = r"""
import json
from pathlib import Path
import sys
from aisimulate_core.sdk import RustForwardPassPerfModel

request = json.loads(Path(sys.argv[1]).read_text())
model = RustForwardPassPerfModel.best_available(state_snapshot=request["snapshot"])
before = [model.estimate_forward_pass_time_ms(query) for query in request["queries"]]
try:
    model.tune_with_fpms(request["later"][:1])
except ValueError as error:
    assert "on_tune" in str(error)
else:
    raise AssertionError("fixed snapshot accepted feedback")
config = model.diagnostics()["provenance"]["config"]
model.close()
config["estimator_config"]["state_snapshot"]["on_tune"] = "continue"
model = RustForwardPassPerfModel.best_available(config)
trajectory = []
for iteration in request["later"]:
    trajectory.append(model.estimate_forward_pass_time_ms(iteration))
    model.tune_with_fpms([iteration])
mixed = model.mixed_composition_state()
mixed.pop("binding")  # Relocated authenticated paths differ; state must not.
result = {
    "before": before, "trajectory": trajectory,
    "predictions": [model.estimate_forward_pass_time_ms(query) for query in request["queries"]],
    "forest": (model.learned_expert_artifacts() if request["selector"] else model.export_learned_artifact()),
    "selector": model.learned_selector_diagnostics(),
    "readiness": model.readiness_report(),
    "online": model.learned_online_diagnostics(),
    "mixed": mixed,
}
Path(request["result"]).write_text(json.dumps(result))
model.close()
"""


@pytest.mark.parametrize("adaptation", ["online", "causal_selector", "causal_selector_appended"])
@pytest.mark.parametrize("source_kind", ["checkpoint", "bundle"])
@pytest.mark.parametrize("admitted", [19, 21])
def test_mixed_snapshot_restores_and_continues_after_sources_are_removed(
    mixed_seed, tmp_path, adaptation, source_kind, admitted
):
    manual_append = adaptation == "causal_selector_appended"
    if manual_append:
        adaptation = "causal_selector"
    authored = source_fixture._config(mixed_seed, adaptation=adaptation)
    if adaptation == "online":
        authored["estimator_config"]["temporary_mixed_composition"]["quality_gate"] = {}
    bundle = tmp_path / "bundle"
    if source_kind == "bundle":
        policy = authored["estimator_config"].pop("temporary_mixed_composition")
        prior = {key: policy[key] for key in ("original_a", "calibration", "quality_gate") if key in policy}
        authored = fpm_learned.export_bundle(
            mixed_seed[1],
            bundle,
            config=authored,
            prior=prior,
            provenance={
                "anchor_kind": "a",
                "sources": [{"reference": "synthetic:mixed-fixture", "sha256": mixed_seed[2]["source_sha256"]}],
                "evaluation_sources": [],
            },
        )
    model = RustForwardPassPerfModel.best_available(authored)
    rows = [source_fixture._iteration("mixed", wall_ms=8 + index % 3, index=index) for index in range(28)]
    for row in rows[:admitted]:
        model.estimate_forward_pass_time_ms(row)
        model.tune_with_fpms([row])
    appended = tmp_path / "appended-mixed"
    if manual_append:
        fpm_learned.train(
            rows[:20],
            "aggregated",
            features=fpm_learned.FEATURE_PRESETS["fpm5"],
            max_iter=3,
            max_leaf_nodes=3,
            min_samples_leaf=2,
            min_store_rows=20,
            online_checkpoint_dir=appended,
            identity=mixed_seed[0],
        )
        assert model.append_mixed_learned_checkpoint(appended) == 1
        for row in rows[admitted : admitted + 2]:
            model.estimate_forward_pass_time_ms(row)
            model.tune_with_fpms([row])
        admitted += 2
        # The stable original-A expert must still authenticate the prior after
        # the adaptive pure store has changed as well as the appended store.
        for index in range(3):
            pure = source_fixture._iteration("decode", wall_ms=12.0, index=100 + index)
            model.estimate_forward_pass_time_ms(pure)
            model.tune_with_fpms([pure])
    before_online = model.learned_online_diagnostics()
    before_state = model.mixed_composition_state()
    saved = tmp_path / "snapshot"
    model.save_state_snapshot(saved, include_perf_db=False)
    assert model.learned_online_diagnostics() == before_online
    assert model.mixed_composition_state() == before_state
    moved = tmp_path / "relocated" / "snapshot"
    moved.parent.mkdir()
    shutil.move(saved, moved)
    summary = RustForwardPassPerfModel.inspect_state_snapshot(moved)
    assert summary["readiness"] == model.readiness_report()
    assert summary["dependencies"]["owned_systems_paths"]
    assert summary["dependencies"]["external_systems_paths"] == []
    for dependency in summary["dependencies"]["files"]:
        assert Path(dependency["path"]).is_relative_to(moved)
    original_manifest = Path(mixed_seed[2]["manifest_path"])
    for root in mixed_seed[0]["systems_paths"]:
        shutil.rmtree(root)
    shutil.rmtree(mixed_seed[1])
    if bundle.exists():
        shutil.rmtree(bundle)
    original_manifest.unlink()
    if appended.exists():
        shutil.rmtree(appended)
    # The independent checkpoint architecture/resource preflight is retained.
    # The snapshot owns timing inputs; it does not replace model metadata checks.
    queries = [source_fixture._iteration(kind) for kind in ("prefill", "decode", "mixed")]
    expected_before = [model.estimate_forward_pass_time_ms(query) for query in queries]
    result = tmp_path / "child-result.json"
    request = tmp_path / "child-request.json"
    request.write_text(
        json.dumps(
            {
                "snapshot": str(moved),
                "later": rows[admitted:],
                "queries": queries,
                "result": str(result),
                "selector": adaptation == "causal_selector",
            }
        )
    )
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, str(request)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert child.returncode == 0, child.stdout + child.stderr
    actual = json.loads(result.read_text())
    trajectory = []
    for row in rows[admitted:]:
        trajectory.append(model.estimate_forward_pass_time_ms(row))
        model.tune_with_fpms([row])
    state = model.mixed_composition_state()
    state.pop("binding")
    assert actual["before"] == expected_before
    assert actual["trajectory"] == trajectory
    expected_forest = (
        model.learned_expert_artifacts() if adaptation == "causal_selector" else model.export_learned_artifact()
    )
    assert actual["forest"] == expected_forest
    assert actual["selector"] == model.learned_selector_diagnostics()
    assert actual["readiness"] == model.readiness_report()
    assert _online_semantics(actual["online"]) == _online_semantics(model.learned_online_diagnostics())
    assert actual["mixed"] == state
    assert actual["predictions"] == [model.estimate_forward_pass_time_ms(query) for query in queries]
    prior_table = next((moved / "mixed" / "perf_db").rglob("*.parquet"))
    prior_table.write_bytes(prior_table.read_bytes() + b"tampered")
    with pytest.raises((ValueError, RuntimeError), match="sha256|SHA-256|digest|changed|length|fingerprint"):
        RustForwardPassPerfModel.best_available(state_snapshot=moved)
    model.close()


@pytest.mark.parametrize("omitted", ["mixed_metadata", "mixed_state_member"])
def test_mixed_snapshot_rejects_repinned_incomplete_closure(mixed_seed, tmp_path, omitted):
    from .test_fpm_state_snapshot import _rewrite_manifest

    model = RustForwardPassPerfModel.best_available(source_fixture._config(mixed_seed, adaptation="online"))
    snapshot = tmp_path / "snapshot"
    try:
        model.save_state_snapshot(snapshot)
    finally:
        model.close()

    def remove_member(manifest):
        if omitted == "mixed_metadata":
            manifest.pop("mixed")
        else:
            state_path = manifest["mixed"]["state"]
            manifest["files"] = [record for record in manifest["files"] if record["path"] != state_path]

    _rewrite_manifest(snapshot, remove_member)
    with pytest.raises(ValueError, match="mixed|member"):
        RustForwardPassPerfModel.best_available(state_snapshot=snapshot)
