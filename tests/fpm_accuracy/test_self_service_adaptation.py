# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the runnable adaptation caller, keeping fitting behind an SDK double."""

import hashlib
import json
import os
import sys
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.fpm_self_service import adapt_gbdt

# Pages has no SDK; the wheel contracts shard must exercise these caller tests.
if "AISIM_TEST_WHEEL" in os.environ:
    sdk = import_module("aisim_core.sdk")
else:
    pytest.importorskip("aisim._runtime")
    sdk = pytest.importorskip("aisim_core.sdk")


@pytest.fixture
def adaptation(tmp_path, monkeypatch, request):
    role = getattr(request, "param", "decode")
    run = tmp_path / "run"
    seed = run / "models" / f"{role}.online-seed"
    seed.mkdir(parents=True)
    (seed / "original.txt").write_text("keep the original seed")
    source = run / "selected" / f"{role}.online.jsonl"
    source.parent.mkdir()
    rows = [
        {
            "version": 1,
            "dp_rank": 0,
            "counter_id": index,
            "wall_time": 0.01 + index / 1000,
            "scheduled_requests": {
                "sum_prefill_tokens": 128 if role == "prefill" else 0,
                "num_decode_requests": 0 if role == "prefill" else index + 1,
                "sum_decode_kv_tokens": 0 if role == "prefill" else 128 * (index + 1),
            },
        }
        for index in range(2)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    identity = run / f"{role}.identity.json"
    identity.write_text(json.dumps({"model": "fixture/model", "system": "fixture_system", "worker_type": role}))
    case = SimpleNamespace(
        run=run,
        role=role,
        seed=seed,
        source=source,
        rows=rows,
        identity=identity,
        output=tmp_path / "adapted",
        models=[],
        prediction=10.0,
        restore_delta=0,
        on_tune=lambda: None,
    )

    class Model:
        def __init__(self, config, state, *, restored):
            self.config = deepcopy(config)
            self.rows = state["rows"]
            self.flushed = state["flushed"]
            self.restored = restored
            self.events = []

        def estimate_forward_pass_time_ms(self, query):
            self.events.append(("predict", deepcopy(query), len(self.rows)))
            if case.prediction is None:
                return None
            return case.prediction + len(self.rows) + self.flushed + (case.restore_delta if self.restored else 0)

        def tune_with_fpms(self, iterations):
            self.events.append(("tune", deepcopy(iterations), len(self.rows)))
            self.rows.extend(deepcopy(iterations))
            case.on_tune()

        def flush_updates(self):
            self.flushed = True

        def save_online_checkpoint(self, path):
            path.mkdir()
            (path / "state.json").write_text(json.dumps({"rows": self.rows, "flushed": self.flushed}))

        def export_learned_artifact(self):
            return {"observations": len(self.rows), "flushed": self.flushed}

        def diagnostics(self):
            return {"provenance": {"config": deepcopy(self.config)}}

    def best_available(config):
        checkpoint = Path(config["estimator_config"]["fpm_learned"]["checkpoint_path"])
        restored = checkpoint != seed
        state = json.loads((checkpoint / "state.json").read_text()) if restored else {"rows": [], "flushed": False}
        model = Model(config, state, restored=restored)
        case.models.append(model)
        return model

    # Patch the lazy SDK export without loading the optional compiled model.
    monkeypatch.setitem(sdk.__dict__, "RustForwardPassPerfModel", SimpleNamespace(best_available=best_available))

    def invoke(*options):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "adapt_gbdt.py",
                "--run-dir",
                str(run),
                "--role",
                role,
                "--output-dir",
                str(case.output),
                *options,
            ],
        )
        adapt_gbdt.main()

    case.invoke = invoke
    return case


@pytest.mark.parametrize("adaptation", ["aggregated", "prefill", "decode"], indirect=True)
@pytest.mark.parametrize("flush", [False, True])
def test_adaptation_predicts_before_feedback_and_persists_restorable_outputs(adaptation, flush, capsys):
    case = adaptation
    original_feedback = case.source.read_bytes()
    case.invoke(*(["--flush-updates"] if flush else []))

    online, restored = case.models
    # Each measurement is predicted using only earlier observations. Labels in
    # seconds reach admission unchanged, and never enter a prediction query.
    for index, row in enumerate(case.rows):
        prediction, admission = online.events[2 * index : 2 * index + 2]
        assert prediction[0] == "predict"
        assert prediction[2] == index
        assert "wall_time" not in prediction[1][0]
        assert prediction[1][0]["scheduled_requests"] == row["scheduled_requests"]
        assert admission[0] == "tune"
        assert admission[2] == index
        assert admission[1][0][0]["wall_time"] == row["wall_time"]
    assert restored.rows == online.rows
    assert online.flushed is restored.flushed is flush
    assert json.loads((case.output / "model.json").read_text()) == {"observations": 2, "flushed": flush}
    restore_config = json.loads((case.output / "restore-config.json").read_text())
    assert restore_config == restored.config
    assert restore_config["model"] == "fixture/model"
    assert restore_config["system"] == "fixture_system"
    assert restore_config["worker_type"] == case.role
    assert restore_config["estimation_mode"] == "fpm_learned"
    assert restore_config["fallback_policy"] == "deny"
    assert restore_config["estimator_config"]["fpm_learned"]["checkpoint_path"] == str(case.output / "checkpoint")
    assert restore_config["estimator_config"]["fpm_learned"]["adaptation"] == "online"
    assert restore_config["estimator_config"]["correction"] == {"enabled": False}

    receipt = json.loads((case.output / "adaptation.json").read_text())
    assert receipt["role"] == case.role
    assert receipt["feedback_sha256"] == hashlib.sha256(original_feedback).hexdigest()
    assert receipt["admitted_observations"] == 2
    assert receipt["last_admitted_observation_index"] == 1
    assert receipt["flushed_updates"] is flush
    assert receipt["restored_prediction_ms"] == 12.0 + flush
    assert json.loads(capsys.readouterr().out) == {"output_dir": str(case.output), **receipt}
    assert case.source.read_bytes() == original_feedback
    assert list(case.seed.iterdir()) == [case.seed / "original.txt"]
    assert (case.seed / "original.txt").read_text() == "keep the original seed"


@pytest.mark.parametrize("problem", ["role", "missing_seed", "inside_seed", "existing_output", "empty_feedback"])
def test_adaptation_rejects_invalid_run_before_loading_or_admitting(adaptation, problem):
    case = adaptation
    error = ValueError
    if problem == "role":
        case.identity.write_text(json.dumps({"worker_type": "prefill"}))
    elif problem == "missing_seed":
        case.seed.rename(case.seed.with_name("unavailable-seed"))
    elif problem == "inside_seed":
        case.output = case.seed / "adapted"
    elif problem == "existing_output":
        case.output.mkdir()
        (case.output / "existing.txt").write_text("preserve existing output")
        error = FileExistsError
    else:
        case.source.write_text("\n")

    with pytest.raises(error):
        case.invoke()
    assert not case.models
    assert not (case.output / "adaptation.json").exists()
    if problem == "existing_output":
        assert (case.output / "existing.txt").read_text() == "preserve existing output"


@pytest.mark.parametrize("problem", ["opposite_role", "idle", "invalid_duration", "uncovered"])
def test_invalid_feedback_is_not_admitted_or_published(adaptation, problem):
    case = adaptation
    row = case.rows[0]
    if problem == "opposite_role":
        row["scheduled_requests"].update(sum_prefill_tokens=128, num_decode_requests=0)
    elif problem == "idle":
        row["scheduled_requests"]["num_decode_requests"] = 0
    elif problem == "invalid_duration":
        row["wall_time"] = 0
    else:
        case.prediction = None
    case.source.write_text(json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="role-compatible busy work|No valid prediction"):
        case.invoke()
    assert not case.models[0].rows
    assert not list(case.output.iterdir())


@pytest.mark.parametrize("problem", ["skipped_record", "source_mutation"])
def test_feedback_integrity_failure_prevents_checkpoint_and_success_receipt(adaptation, problem):
    case = adaptation
    if problem == "skipped_record":
        with case.source.open("a") as output:
            output.write("not a JSON record\n")
    else:
        # A blank line changes the hash without changing the admitted row count.
        def mutate_source():
            with case.source.open("a") as output:
                output.write("\n")

        case.on_tune = mutate_source

    with pytest.raises(ValueError, match="Feedback changed or the parser skipped records"):
        case.invoke()
    assert len(case.models[0].rows) == 2
    assert not list(case.output.iterdir())


def test_restore_mismatch_leaves_partial_outputs_without_success_receipt(adaptation, capsys):
    case = adaptation
    case.restore_delta = 1
    with pytest.raises(RuntimeError, match="Restored prediction does not match"):
        case.invoke()
    assert (case.output / "checkpoint" / "state.json").is_file()
    assert (case.output / "model.json").is_file()
    assert not (case.output / "restore-config.json").exists()
    assert not (case.output / "adaptation.json").exists()
    assert not capsys.readouterr().out
