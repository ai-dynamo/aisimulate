# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Complete learned state crosses a process boundary without source discovery."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from aisimulate_core.sdk import fpm_learned
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

from .fpm_training_support import import_sklearn
from .fpm_training_support import isolate_training_dependencies as isolate_training_dependencies

pytestmark = pytest.mark.unit


def _online_semantics(value):
    """Compare admission IDs, counters and pending state, excluding host costs."""
    value = json.loads(json.dumps(value))
    value.pop("training_state_bytes", None)
    for store in value["stores"].values():
        store.pop("training_state_bytes", None)
        store.pop("history_grouping_ms", None)
    return value


import_sklearn()


def observations(count, *, offset=0):
    return [
        [
            {
                "version": 1,
                "counter_id": index + offset,
                "wall_time": (10 + index % 8) / 1000,
                "scheduled_requests": {
                    "num_decode_requests": 1 + index % 8,
                    "sum_decode_kv_tokens": 32 * (1 + index % 8),
                },
            }
        ]
        for index in range(count)
    ]


def seeded_config(path):
    identity = {
        "model": "test/model",
        "system": "test-gpu",
        "backend": "vllm",
        "backend_version": "test-version",
        "worker_type": "decode",
    }
    fpm_learned.train(
        observations(40),
        "decode",
        identity=identity,
        online_checkpoint_dir=path,
        max_iter=4,
        max_leaf_nodes=3,
        min_samples_leaf=2,
        min_store_rows=20,
    )
    return {
        **identity,
        "estimation_mode": "fpm_learned",
        "estimator_config": {"fpm_learned": {"checkpoint_path": str(path), "update_every": 4}},
    }


_CHILD = r"""
import json
from pathlib import Path
import sys
from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

request = json.loads(Path(sys.argv[1]).read_text())
model = RustForwardPassPerfModel.best_available(state_snapshot=request["snapshot"])
fixed_report = model.readiness_report()
try:
    model.tune_with_fpms(request["later"][:1])
except ValueError as error:
    assert "on_tune" in str(error)
else:
    raise AssertionError("fixed snapshot accepted feedback")
assert model.readiness_report() == fixed_report
settings = model.diagnostics()["provenance"]["config"]
settings["estimator_config"]["state_snapshot"]["on_tune"] = "continue"
model = RustForwardPassPerfModel.best_available(settings)
model.tune_with_fpms(request["later"])
result = {
    "predictions": [model.estimate_forward_pass_time_ms(query) for query in request["queries"]],
    "readiness": model.readiness_report(),
    "forest": model.export_learned_artifact(),
    "online": model.learned_online_diagnostics(),
    "dependencies": model.dependency_info(),
}
Path(request["result"]).write_text(json.dumps(result))
"""


@pytest.mark.parametrize("admitted", [0, 3, 4, 5])
def test_seeded_ready_and_pending_snapshots_continue_in_a_new_process(tmp_path, admitted):
    source = tmp_path / "offline-seed"
    authored = seeded_config(source)
    model = RustForwardPassPerfModel.best_available(authored)
    stream = observations(28, offset=100)
    model.tune_with_fpms(stream[:admitted])
    before = model.readiness_report()
    before_online = model.learned_online_diagnostics()
    original = tmp_path / "original"
    model.save_state_snapshot(original)
    assert model.readiness_report() == before
    assert model.learned_online_diagnostics() == before_online
    moved = tmp_path / "other-process" / "moved"
    moved.parent.mkdir()
    shutil.move(original, moved)
    assert not original.exists()
    shutil.rmtree(source)
    inspection = RustForwardPassPerfModel.inspect_state_snapshot(moved)
    assert inspection["readiness"] == before
    assert inspection["resolved_config"]["estimator_config"]["fpm_learned"]["resolved_source"]["kind"] == "checkpoint"
    result = tmp_path / "result.json"
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"snapshot": str(moved), "later": stream[admitted:], "queries": stream[:3], "result": str(result)})
    )
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, str(request)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert child.returncode == 0, child.stdout + child.stderr
    restored = json.loads(result.read_text())
    model.tune_with_fpms(stream[admitted:])
    assert restored["forest"] == model.export_learned_artifact()
    assert restored["readiness"] == model.readiness_report()
    assert _online_semantics(restored["online"]) == _online_semantics(model.learned_online_diagnostics())
    assert restored["online"]["stores"]["pure_decode"]["accepted_observations"] == 28
    assert restored["predictions"] == [model.estimate_forward_pass_time_ms(query) for query in stream[:3]]
    model.close()
