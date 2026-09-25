# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public onboarding from synthetic, independently validated runtime evidence."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from collector.fpm_forward.runtime_probe import normalize_probe_launch

import aisimulate.main as cli
from aisimulate.support.checkpoint import _load, report, save_checkpoint
from aisimulate.support.schema import SupportRequest

from .collector.test_runtime_observations import observation_fixture

pytestmark = pytest.mark.unit


def _write(path, value):
    path.write_text(json.dumps(value))


def _campaign(tmp_path, *, capacity_multiplier=1, context_length=4096, tp=2):
    index, launches = observation_fixture(tmp_path / "probe", dense=True, tp=tp)
    launch = launches["tp2"]
    model_path = Path(launch["model_config"]["path"])
    _write(
        model_path,
        {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "max_position_embeddings": 4096,
            "torch_dtype": "bfloat16",
            "hidden_size": 128,
            "vocab_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": max(4, tp),
            "num_key_value_heads": max(4, tp),
            "intermediate_size": 256,
            "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "kv_cache_quant_algo": "none"},
        },
    )
    launch["model_config"]["sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    launch["collection"]["max_model_len"] = context_length
    launch = normalize_probe_launch(launch)
    document = json.loads(index.read_text())
    document["configurations"]["tp2"]["launch"] = launch
    for phase in document["configurations"]["tp2"]["attempts"][0]["phases"].values():
        for ref in [phase["launch_manifest"], *phase["artifacts"]]:
            path = index.parent / ref["path"]
            value = json.loads(path.read_text())
            value["launch"] = launch
            if ref.get("kind") == "observation":
                value["model_config_sha256"] = launch["model_config"]["sha256"]
                value["resolved_config"]["model_config"]["max_model_len"] = context_length
                cache = value["cache"]
                cache["num_blocks"] *= capacity_multiplier
                if value["kind"] == "worker":
                    for field in ("available_cache_bytes", "allocated_cache_bytes"):
                        cache[field] *= capacity_multiplier
                    for storage in cache["storages"]:
                        storage["size_bytes"] *= capacity_multiplier
                    for allocation in cache["tensor_allocations"]:
                        allocation["size"] *= capacity_multiplier
                    for view in cache["layer_tensors"].values():
                        view["shape"][view["block_axis"]] *= capacity_multiplier
                else:
                    cache["initial_free_blocks"] = cache["num_blocks"] - cache["reserved_blocks"]
            _write(path, value)
            ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(index, document)
    precision = dict(launch["precision"])
    precision["kv_cache_dtype"] = precision.pop("kvcache_quant_mode")
    draft = {
        "identity": launch["identity"],
        "workload": {"input_tokens": 64, "output_tokens": 16, "concurrency": 1, "request_count": 1},
        "search": {
            "tensor_parallel": tp,
            "attention_data_parallel": 1,
            "moe_tensor_parallel": 1,
            "moe_expert_parallel": 1,
            "context_length": context_length,
        },
        "collection": {
            "max_num_tokens": 1024,
            "max_batch_size": 64,
            "gpu_memory_utilization": 0.9,
            "prefill_cudagraph_policy": "runtime",
        },
    }
    checkpoint = tmp_path / "checkpoint.json"
    save_checkpoint(
        checkpoint,
        patch={
            "inputs": {"model_config": str(model_path)},
            "configurations": {
                name: {
                    "draft_request": copy.deepcopy(draft),
                    "inputs": {"precision": precision, "collection_deployment": launch["deployment"]},
                }
                for name in ("tp2", "incomplete")
            },
        },
        expected_revision=None,
        accept=[],
    )
    return checkpoint, index, launch


def _import(checkpoint, index, output, *, configurations=()):
    args = [
        "onboard",
        "import-observations",
        "--checkpoint",
        str(checkpoint),
        "--observations",
        str(index),
        "--output-dir",
        str(output),
    ]
    for name in configurations:
        args.extend(("--configuration", name))
    return cli.main(args)


def test_import_keeps_partial_results_and_memory_is_independent_of_timing(tmp_path, capsys):
    checkpoint, index, _ = _campaign(tmp_path)
    assert _import(checkpoint, index, tmp_path / "drafts") == 1
    result = json.loads(capsys.readouterr().out)
    assert result["configurations"]["tp2"]["status"] == "complete"
    assert result["configurations"]["incomplete"]["status"] == "incomplete"
    state = _load(checkpoint)
    request = SupportRequest.model_validate(state.configurations["tp2"].draft_request)
    assert request.profile_deployment().resources.runtime_memory.kv_cache_bytes == 93 * 128
    assert state.configurations["tp2"].acceptance is None
    assert not state.configurations["incomplete"].draft_request.get("fpm_profile")
    assert state.configurations["incomplete"].progress.status == "blocked"
    assert state.configurations["incomplete"].progress.blockers == result["configurations"]["incomplete"]["diagnostics"]
    assert not list((tmp_path / "drafts").rglob("*.parquet"))
    assert not report(state, checkpoint)["integrity_issues"]
    previous = next(item for item in state.configurations["tp2"].history if item["event"] == "draft_replaced")
    assert previous["draft_request"]["search"]["tensor_parallel"] == 2


def test_snapshot_acceptance_survives_unrelated_probe_progress_but_detects_raw_tamper(tmp_path, capsys):
    checkpoint, index, _ = _campaign(tmp_path)
    assert _import(checkpoint, index, tmp_path / "drafts", configurations=["tp2"]) == 0
    capsys.readouterr()
    state = _load(checkpoint)
    state, _ = save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    document = json.loads(index.read_text())
    document["configurations"]["unrelated"] = {"attempts": []}
    _write(index, document)
    assert report(state, checkpoint)["configurations"]["tp2"]["profile_accepted"]
    snapshot = next((tmp_path / "drafts").rglob("worker-0-0.json"))
    snapshot.write_text("{}")
    status = report(state, checkpoint)
    assert not status["configurations"]["tp2"]["profile_accepted"]
    assert status["integrity_issues"]


def test_probe_public_preview_does_not_need_geometry_or_execute_bundle(tmp_path, capsys, monkeypatch):
    checkpoint, index, _ = _campaign(tmp_path)
    from collector.fpm_forward import runtime_probe

    calls = []

    def probe(configurations, **kwargs):
        calls.append((configurations, kwargs))
        return {"status": "preview", "configurations": {key: {"status": "preview"} for key in configurations}}

    monkeypatch.setattr(runtime_probe, "probe_runtime", probe)
    assert (
        cli.main(
            [
                "onboard",
                "probe-runtime",
                "--checkpoint",
                str(checkpoint),
                "--instrumentation",
                str(index.parent / "manifest.yaml"),
                "--output-dir",
                str(tmp_path / "preview"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    facts, options = calls[0]
    assert set(facts) == {"tp2", "incomplete"}
    assert all(value["topology"]["tp"] == 2 for value in facts.values())
    assert all("cache_groups" not in str(value) for value in facts.values())
    assert options["execute"] is False
    assert facts["tp2"]["collection"]["prefill_cudagraph_policy"] == "runtime"


def test_real_public_preview_parses_bundle_without_executing_python(tmp_path, capsys):
    checkpoint, index, _ = _campaign(tmp_path)
    assert (
        cli.main(
            [
                "onboard",
                "probe-runtime",
                "--checkpoint",
                str(checkpoint),
                "--configuration",
                "tp2",
                "--instrumentation",
                str(index.parent / "manifest.yaml"),
                "--output-dir",
                str(tmp_path / "preview"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "preview"
    assert set(result["configurations"]["tp2"]["phases"]) == {"prefill", "decode"}


@pytest.mark.parametrize("workload", [None, {"input_tokens": 1024, "output_tokens": 128}])
def test_launch_only_preview_preserves_workload_without_validating_it(tmp_path, capsys, workload):
    checkpoint, index, _ = _campaign(tmp_path, context_length=512)
    state = _load(checkpoint)
    state, _ = save_checkpoint(
        checkpoint,
        patch={"configurations": {"tp2": {"draft_request": {"workload": workload, "search": {"context_length": 512}}}}},
        expected_revision=state.revision,
        accept=[],
    )
    draft = copy.deepcopy(state.configurations["tp2"].draft_request)
    assert (
        cli.main(
            [
                "onboard",
                "probe-runtime",
                "--checkpoint",
                str(checkpoint),
                "--configuration",
                "tp2",
                "--instrumentation",
                str(index.parent / "manifest.yaml"),
                "--output-dir",
                str(tmp_path / "preview"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert set(result["configurations"]["tp2"]["phases"]) == {"prefill", "decode"}
    assert _load(checkpoint).configurations["tp2"].draft_request == draft
    with pytest.raises(ValueError, match="input and output tokens"):
        SupportRequest.model_validate(draft)
    # Import reaches real simulation validation, retaining the user's pending
    # workload rather than inventing a smaller synthetic request.
    assert _import(checkpoint, index, tmp_path / "drafts", configurations=["tp2"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert "input and output tokens" in str(result["configurations"]["tp2"]["diagnostics"])
    assert _load(checkpoint).configurations["tp2"].draft_request == draft


@pytest.mark.parametrize(
    ("section", "field", "value", "diagnostic"),
    [
        ("search", "context_length", 0, "context_length"),
        ("search", "attention_data_parallel", 2, "topology"),
        ("collection", "max_num_tokens", 1, "max_num_tokens"),
        ("identity", "model_revision", "main", "pinned revision"),
    ],
)
def test_launch_only_preview_still_rejects_invalid_launch_choices(tmp_path, capsys, section, field, value, diagnostic):
    checkpoint, index, _ = _campaign(tmp_path)
    state = _load(checkpoint)
    save_checkpoint(
        checkpoint,
        patch={"configurations": {"tp2": {"draft_request": {"workload": None, section: {field: value}}}}},
        expected_revision=state.revision,
        accept=[],
    )
    assert (
        cli.main(
            [
                "onboard",
                "probe-runtime",
                "--checkpoint",
                str(checkpoint),
                "--configuration",
                "tp2",
                "--instrumentation",
                str(index.parent / "manifest.yaml"),
                "--output-dir",
                str(tmp_path / "preview"),
            ]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert diagnostic in str(result["configurations"]["tp2"])


def test_probe_preserves_symlinked_config_directory_for_adjacent_sources(tmp_path, capsys):
    from aisimulate.support.runtime import checkpoint_launch

    checkpoint, index, launch = _campaign(tmp_path)
    snapshot = tmp_path / "hf-snapshot"
    snapshot.mkdir()
    model = snapshot / "config.json"
    model.symlink_to(launch["model_config"]["path"])
    blob = tmp_path / "sidecar-blob"
    _write(blob, {"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "none"}})
    (snapshot / "hf_quant_config.json").symlink_to(blob)
    state = _load(checkpoint)
    state, _ = save_checkpoint(
        checkpoint, patch={"inputs": {"model_config": str(model)}}, expected_revision=state.revision, accept=[]
    )
    _, facts, _ = checkpoint_launch(state, "tp2", checkpoint)
    assert facts["model_config"]["path"] == str(model)
    assert facts["model_config"]["sha256"] == launch["model_config"]["sha256"]
    assert facts["model_config"]["source_files"] == {
        "hf_quant_config.json": hashlib.sha256(blob.read_bytes()).hexdigest()
    }
    assert (
        cli.main(
            [
                "onboard",
                "probe-runtime",
                "--checkpoint",
                str(checkpoint),
                "--configuration",
                "tp2",
                "--instrumentation",
                str(index.parent / "manifest.yaml"),
                "--output-dir",
                str(tmp_path / "preview"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert set(result["configurations"]["tp2"]["phases"]) == {"prefill", "decode"}


def test_incomplete_legacy_profile_can_be_saved_incrementally(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    state, _ = save_checkpoint(
        checkpoint,
        patch={"configurations": {"worker": {"draft_request": {"fpm_profile": {"model": "example/model"}}}}},
        expected_revision=None,
        accept=[],
    )
    state, _ = save_checkpoint(
        checkpoint,
        patch={"configurations": {"worker": {"draft_request": {"fpm_profile": {"model_revision": "abc123"}}}}},
        expected_revision=state.revision,
        accept=[],
    )
    assert state.revision == 2
    assert state.configurations["worker"].draft_request == {
        "fpm_profile": {"model": "example/model", "model_revision": "abc123"}
    }
    assert state.configurations["worker"].acceptance is None


@pytest.mark.parametrize("remove_profile", [False, True])
def test_changed_scheduler_can_be_saved_and_probed_before_fresh_import(tmp_path, capsys, remove_profile):
    checkpoint, index, request, _ = _imported(tmp_path, capsys)
    state = _load(checkpoint)
    state, _ = save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    accepted = state.configurations["tp2"].acceptance.model_dump(mode="json")
    draft = {"collection": {"max_num_tokens": 2048}}
    if remove_profile:
        draft["fpm_profile"] = None
    state, _ = save_checkpoint(
        checkpoint,
        patch={"configurations": {"tp2": {"draft_request": draft}}},
        expected_revision=state.revision,
        accept=[],
    )
    saved = state.configurations["tp2"]
    assert saved.draft_request["collection"]["max_num_tokens"] == 2048
    assert saved.acceptance is None
    assert saved.history[-1]["acceptance"] == accepted
    if not remove_profile:
        assert saved.draft_request["fpm_profile"] == request.fpm_profile.model_dump(mode="json", exclude_none=True)
        # A second edit to an already incomplete request cannot rewrite the
        # observations or discard source hashes in the imported provenance.
        edited = copy.deepcopy(saved.draft_request)
        runtime = edited["fpm_profile"]["deployments"][0]["resources"]["runtime_memory"]
        provenance = json.loads(runtime["provenance"])
        provenance["observed_resources"]["runtime_memory"]["kv_cache_bytes"] = 1
        provenance["source_artifacts"] = []
        runtime["provenance"] = json.dumps(provenance)
        state, _ = save_checkpoint(
            checkpoint,
            patch={"configurations": {"tp2": {"draft_request": edited}}},
            expected_revision=state.revision,
            accept=[],
        )
        assert state.configurations["tp2"].draft_request["fpm_profile"] == saved.draft_request["fpm_profile"]
    with pytest.raises(ValueError):
        save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    assert (
        cli.main(
            [
                "onboard",
                "probe-runtime",
                "--checkpoint",
                str(checkpoint),
                "--configuration",
                "tp2",
                "--instrumentation",
                str(index.parent / "manifest.yaml"),
                "--output-dir",
                str(tmp_path / "preview"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    # A separate synthetic probe captures the selected scheduler envelope.
    fresh = tmp_path / "fresh-probe"
    shutil.copytree(index.parent, fresh)
    fresh_index = fresh / index.name
    document = json.loads(fresh_index.read_text())
    entry = document["configurations"]["tp2"]
    entry["launch"]["collection"]["max_num_batched_tokens"] = 2048
    for phase in entry["attempts"][0]["phases"].values():
        for ref in [phase["launch_manifest"], *phase["artifacts"]]:
            path = fresh / ref["path"]
            record = json.loads(path.read_text())
            record["launch"] = entry["launch"]
            if "resolved_config" in record:
                record["resolved_config"]["scheduler_config"]["max_num_batched_tokens"] = 2048
            _write(path, record)
            ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(fresh_index, document)
    assert _import(checkpoint, fresh_index, tmp_path / "new-drafts", configurations=["tp2"]) == 0
    capsys.readouterr()
    state = _load(checkpoint)
    assert state.configurations["tp2"].acceptance is None
    state, _ = save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    assert report(state, checkpoint)["configurations"]["tp2"]["profile_accepted"]


def test_failed_reimport_keeps_accepted_profile_and_historical_error_evidence(tmp_path, capsys):
    checkpoint, index, _, _ = _imported(tmp_path, capsys)
    state = _load(checkpoint)
    state, _ = save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    previous = copy.deepcopy(state.configurations["tp2"])
    valid = index.read_bytes()
    document = json.loads(valid)
    document["configurations"]["tp2"]["attempts"] = []
    _write(index, document)
    failed = tmp_path / "failed-drafts"
    assert _import(checkpoint, index, failed, configurations=["tp2"]) == 1
    result = json.loads(capsys.readouterr().out)["configurations"]["tp2"]
    state = _load(checkpoint)
    saved = state.configurations["tp2"]
    assert saved.acceptance == previous.acceptance
    assert saved.draft_request == previous.draft_request
    assert saved.progress.status == "blocked"
    assert saved.progress.blockers == result["diagnostics"]
    assert result["diagnostics"]
    refs = {key: ref for key, ref in saved.artifacts.items() if key not in previous.artifacts}
    assert refs and all(ref.archived and ref.sha256 for ref in refs.values())
    assert saved.history[-1]["validation"] == result["validation"]
    assert saved.history[-1]["sha256"] == hashlib.sha256(Path(result["validation"]).read_bytes()).hexdigest()
    assert cli.main(["onboard", "resume", "--checkpoint", str(checkpoint)]) == 0
    resumed = json.loads(capsys.readouterr().out)
    assert resumed["configurations"]["tp2"]["profile_accepted"]
    assert resumed["configurations"]["tp2"]["reported_progress"]["blockers"] == result["diagnostics"]
    # Changing failed-attempt files cannot invalidate the independently accepted snapshot.
    for ref in refs.values():
        (checkpoint.parent / ref.path).write_text("changed failed historical evidence")
    assert report(state, checkpoint)["configurations"]["tp2"]["profile_accepted"]
    assert report(state, checkpoint)["configurations"]["tp2"]["reported_progress"]["blockers"] == result["diagnostics"]
    index.write_bytes(valid)
    assert _import(checkpoint, index, tmp_path / "replacement-drafts", configurations=["tp2"]) == 0
    capsys.readouterr()
    saved = _load(checkpoint).configurations["tp2"]
    assert saved.acceptance is None
    assert any(item.get("acceptance") == previous.acceptance.model_dump(mode="json") for item in saved.history)


def test_changed_inputs_archive_acceptance_even_when_draft_is_unchanged(tmp_path, capsys):
    checkpoint, _, _, _ = _imported(tmp_path, capsys)
    state = _load(checkpoint)
    state, _ = save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    previous = copy.deepcopy(state.configurations["tp2"])
    state, _ = save_checkpoint(
        checkpoint,
        patch={
            "configurations": {
                "tp2": {
                    "inputs": {"collection_deployment": {"image": "changed@sha256:" + "d" * 64}},
                    "progress": {
                        "status": "blocked",
                        "blockers": ["Fresh image needs compatible observations."],
                        "next_action": "Run the selected image's probe.",
                    },
                }
            }
        },
        expected_revision=state.revision,
        accept=[],
    )
    saved = state.configurations["tp2"]
    assert saved.draft_request == previous.draft_request
    assert saved.acceptance is None
    history = saved.history[-1]
    assert history["acceptance"] == previous.acceptance.model_dump(mode="json")
    assert history["draft_request"] == previous.draft_request
    assert history["inputs"] == previous.inputs
    assert saved.progress.status == "blocked"
    assert saved.progress.blockers == ["Fresh image needs compatible observations."]


def test_unknown_geometry_does_not_mask_missing_explicit_topology(tmp_path, capsys):
    checkpoint, index, _ = _campaign(tmp_path)
    state = _load(checkpoint)
    save_checkpoint(
        checkpoint,
        patch={"configurations": {"tp2": {"draft_request": {"search": {"tensor_parallel": None}}}}},
        expected_revision=state.revision,
        accept=[],
    )
    assert (
        cli.main(
            [
                "onboard",
                "probe-runtime",
                "--checkpoint",
                str(checkpoint),
                "--configuration",
                "tp2",
                "--instrumentation",
                str(index.parent / "manifest.yaml"),
                "--output-dir",
                str(tmp_path / "preview"),
            ]
        )
        == 1
    )
    result = json.loads(capsys.readouterr().out)
    assert "tensor_parallel" in str(result["configurations"]["tp2"]["diagnostics"])


def _imported(tmp_path, capsys, *, capacity_multiplier=1, tp=2):
    checkpoint, index, _ = _campaign(tmp_path, capacity_multiplier=capacity_multiplier, tp=tp)
    assert _import(checkpoint, index, tmp_path / "drafts", configurations=["tp2"]) == 0
    result = json.loads(capsys.readouterr().out)
    state = _load(checkpoint)
    request = SupportRequest.model_validate(state.configurations["tp2"].draft_request)
    return checkpoint, index, request, result["configurations"]["tp2"]


def test_profile_edit_preserves_measured_values_and_requires_fresh_probe_for_settings(tmp_path, capsys):
    from aisimulate.support.runtime import runtime_probe_manifest, verify_runtime_profile

    checkpoint, _, request, _ = _imported(tmp_path, capsys)
    state = _load(checkpoint)
    state, _ = save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    payload = request.model_dump(mode="json", exclude_none=True)
    resources = payload["fpm_profile"]["deployments"][0]["resources"]
    observed = resources["runtime_memory"]["kv_cache_bytes"]
    resources["runtime_memory"]["kv_cache_bytes"] = observed - 128
    state, _ = save_checkpoint(
        checkpoint,
        patch={"configurations": {"tp2": {"draft_request": payload}}},
        expected_revision=state.revision,
        accept=[],
    )
    changed = SupportRequest.model_validate(state.configurations["tp2"].draft_request)
    manifest = runtime_probe_manifest(changed)
    assert manifest["observed_resources"]["runtime_memory"]["kv_cache_bytes"] == observed
    assert manifest["user_overrides"]["runtime_memory.kv_cache_bytes"]["value"] == observed - 128
    assert state.configurations["tp2"].acceptance is None
    assert state.configurations["tp2"].history[-1]["acceptance"] is not None
    verify_runtime_profile(changed)
    payload = changed.model_dump(mode="json", exclude_none=True)
    payload["collection"]["prefill_cudagraph_policy"] = "explicit"
    state, _ = save_checkpoint(
        checkpoint,
        patch={"configurations": {"tp2": {"draft_request": payload}}},
        expected_revision=state.revision,
        accept=[],
    )
    with pytest.raises(ValueError, match="fresh probe"):
        save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])


def test_plan_carries_explicit_formal_observer_flags_and_execution_requires_acceptance(tmp_path, capsys, monkeypatch):
    from aisimulate.support.runtime import runtime_probe_manifest

    checkpoint, _, request, files = _imported(tmp_path, capsys, capacity_multiplier=10)
    root = tmp_path / "collection"
    assert (
        cli.main(["onboard", "plan", "--config", files["request"], "--output-dir", str(root), "--format", "json"]) == 0
    )
    capsys.readouterr()
    commands = json.loads((root / "commands.json").read_text())
    preview = commands["fpm_plan_local"]
    assert "--fpm-runtime-instrumentation" in preview
    assert "--fpm-runtime-launch" in preview
    assert preview[preview.index("--fpm-runtime-configuration") + 1] == "tp2"
    execute = commands["fpm_run_local"][1:]
    assert "--executor" in execute and "slurm" in execute
    with pytest.raises(SystemExit, match="2"):
        cli.main(execute)
    assert "acceptance" in capsys.readouterr().err
    state = _load(checkpoint)
    save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    from types import SimpleNamespace

    from collector.fpm_forward import cli as collector_cli
    from collector.fpm_forward import entry

    from aisimulate.support import collection_readiness, fpm

    commands = []

    def resolve(command):
        commands.append(command[3:])
        return collector_cli._parser().parse_args(command[3:]), (SimpleNamespace(cells=()), {})

    monkeypatch.setattr(fpm, "_resolve_execution", resolve)
    monkeypatch.setattr(collection_readiness, "assess_readiness", lambda *a, **k: {"ready_for_full_collection": True})

    def fake_collect(argv):
        assert "--fpm-runtime-instrumentation" in argv
        manifest = runtime_probe_manifest(request)
        target = root / "observed"
        shutil.copytree(Path(manifest["observations_index"]).parent, target)
        (root / "fpm-checkpoint").mkdir(exist_ok=True)
        bindings = _formal_bindings(target / "observations.json")
        _write(
            root / "fpm-checkpoint" / "fpm_forward.json",
            {"runtime_observations": str(target / "observations.json"), **bindings},
        )
        return 0

    monkeypatch.setattr(entry, "run_resolved", lambda *_: fake_collect(commands[-1]) or [])
    assert cli.main(execute) == 0
    comparison = json.loads((root / "runtime-compatibility.json").read_text())
    assert comparison["compatibility"]["status"] == "compatible"
    assert comparison["compatibility"]["accepted_capacity_bytes"] == 948 * 128


def _formal_bindings(index, cells=None):
    """Synthetic formal contexts bound to distinct native cell attempts."""
    document = json.loads(index.read_text())
    entry = document["configurations"]["tp2"]
    entry["active_attempt_id"] = "formal-parent"
    attempt = entry["attempts"][0]
    attempt["attempt_id"] = "formal-parent"
    entries = {}
    for phase, result in attempt["phases"].items():
        cell_id = cells[phase] if cells else f"cell-{phase}"
        attempt_id = f"native-{phase}"
        result.update(cell_id=cell_id, collector_attempt_id=attempt_id)
        entries[cell_id] = {"status": "passed", "attempt_id": attempt_id}
        for ref in [result["launch_manifest"], *result["artifacts"]]:
            path = index.parent / ref["path"]
            value = json.loads(path.read_text())
            value.update(attempt_id="formal-parent", cell_id=cell_id, collector_attempt_id=attempt_id)
            _write(path, value)
            ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(index, document)
    return {"runtime_observation_attempt_id": "formal-parent", "cells": entries}


def _completed_runtime_collection(tmp_path, capsys, *, tp=2):
    from collector.fpm_forward.cli import _parser
    from collector.fpm_forward.config import FPMCollectionOptions
    from collector.fpm_forward.database import aggregate_cell, write_formal_database
    from collector.fpm_forward.entry import _load_generator_overrides
    from collector.fpm_forward.planner import build_collection_plan
    from collector.fpm_forward.runner import CHECKPOINT_SCHEMA

    from aisimulate.support.runtime import runtime_probe_manifest

    from .test_onboard_finalization import _native

    checkpoint, _, request, files = _imported(tmp_path, capsys, capacity_multiplier=10, tp=tp)
    state = _load(checkpoint)
    save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    root = tmp_path / "collection"
    assert cli.main(["onboard", "plan", "--config", files["request"], "--output-dir", str(root)]) == 0
    capsys.readouterr()
    args = _parser().parse_args(json.loads((root / "commands.json").read_text())["fpm_plan_local"][3:])
    manifest = runtime_probe_manifest(request)
    snapshot = Path(manifest["observations_index"])
    launch = json.loads(snapshot.read_text())["configurations"]["tp2"]["launch"]
    plan = build_collection_plan(
        backend="vllm",
        model_path=request.identity.model,
        model_architecture=request.fpm_profile.architecture,
        model_config_path=launch["model_config"]["path"],
        system=request.identity.gpu,
        selected_ops={"attention_context", "attention_generation"},
        options=FPMCollectionOptions.from_args(args),
        fpm_profile=request.fpm_profile,
        generator_overrides=_load_generator_overrides(args),
        runtime_instrumentation=args.fpm_runtime_instrumentation,
        runtime_launch=launch,
        runtime_configuration="tp2",
    )
    artifact_root = root / "fpm-artifacts" / plan.sha256[:16]
    artifact_root.mkdir(parents=True)
    _write(artifact_root / "collection-plan.json", plan.to_dict())
    observed = artifact_root / "observations"
    shutil.copytree(snapshot.parent, observed)
    bindings = _formal_bindings(
        observed / "observations.json", {cell.workload_kind: cell.cell_id for cell in plan.cells}
    )
    rows = []
    for cell in plan.cells:
        directory = artifact_root / "cells" / cell.cell_id
        raw = directory / "raw/pod-0"
        raw.mkdir(parents=True)
        _write(directory / "cell.json", cell.to_dict())
        provenance = {
            "schema_name": "aic_fpm_collector_provenance",
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "plan_sha256": plan.sha256,
            "attempt_id": f"native-{cell.workload_kind}",
            "runtime": {"backend": "vllm", "backend_version": "0.28.0"},
        }
        _write(raw / "collector-provenance.json", provenance)
        _write(raw / "benchmark.json", _native(cell.workload_kind))
        rows.extend(aggregate_cell(plan, cell, directory, expected_attempt_id=provenance["attempt_id"]))
    parquet, metadata, skipped = write_formal_database(plan, rows, systems_root=root / "systems/data")
    assert not skipped
    (root / "fpm-checkpoint").mkdir(exist_ok=True)
    _write(
        root / "fpm-checkpoint/fpm_forward.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "plan_sha256": plan.sha256,
            **bindings,
            "runtime_observations": str(observed / "observations.json"),
            "database": {
                "status": "passed",
                "parquet": str(parquet),
                "metadata": str(metadata),
                "published_cells": len(plan.cells),
                "plan_cells": len(plan.cells),
                "missing_cells": [],
                "skipped_first_publisher_wins": [],
            },
        },
    )
    return checkpoint, request, files, root


def test_formal_finalize_validates_new_observations_and_preserves_verified_timing_pair(tmp_path, capsys):
    from aisimulate.support.finalization import finalization_manifest

    checkpoint, request, files, root = _completed_runtime_collection(tmp_path, capsys)
    target = tmp_path / "finalized"
    assert (
        cli.main(
            [
                "onboard",
                "finalize",
                "--config",
                files["request"],
                "--output-dir",
                str(root),
                "--resolved-output-dir",
                str(target),
            ]
        )
        == 0
    )
    capsys.readouterr()
    resolved = SupportRequest.from_yaml(target / "request.yaml")
    evidence = finalization_manifest(resolved)
    assert evidence["runtime_compatibility"]["status"] == "compatible"
    assert len(evidence["formal_data"]) == 2
    assert list((target / "systems/data").rglob("fpm_forward_perf.parquet"))
    assert _load(checkpoint).configurations["tp2"].acceptance is not None
    # Finalization produces another draft. Replacing the imported request keeps
    # the accepted probe in history and does not treat the new profile as accepted.
    state = _load(checkpoint)
    artifacts = {key: {"archived": True} for key in state.configurations["tp2"].artifacts}
    artifacts["final-request"] = {"path": str(target / "request.yaml"), "kind": "request"}
    state, _ = save_checkpoint(
        checkpoint,
        patch={
            "configurations": {
                "tp2": {"draft_request": resolved.model_dump(mode="json", exclude_none=True), "artifacts": artifacts}
            }
        },
        expected_revision=state.revision,
        accept=[],
    )
    assert state.configurations["tp2"].acceptance is None
    state, _ = save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    assert report(state, checkpoint)["configurations"]["tp2"]["profile_accepted"]


def _reviewed_capacity_revision(tmp_path, capsys):
    checkpoint, original, files, root = _completed_runtime_collection(tmp_path, capsys)
    collection = json.loads((root / "fpm-checkpoint/fpm_forward.json").read_text())
    index = Path(collection["runtime_observations"])
    document = json.loads(index.read_text())
    for phase in document["configurations"]["tp2"]["attempts"][0]["phases"].values():
        for ref in phase["artifacts"]:
            path = index.parent / ref["path"]
            record = json.loads(path.read_text())
            if record["kind"] == "scheduler":
                record["cache"]["initial_free_blocks"] -= 1
                record["cache"]["reserved_blocks"] += 1
                record["cache"]["permanent_reserved_block_ids"].append(2)
                _write(path, record)
                ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(index, document)
    assert _import(checkpoint, index, tmp_path / "revised", configurations=["tp2"]) == 0
    revised = Path(json.loads(capsys.readouterr().out)["configurations"]["tp2"]["request"])
    state = _load(checkpoint)
    save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    return checkpoint, original, files, root, revised


def test_finalization_reuses_original_collection_with_accepted_observed_capacity_revision(tmp_path, capsys):
    from aisimulate.support.finalization import finalization_manifest, finalize
    from aisimulate.support.plan import check_plan, request_id
    from aisimulate.support.runtime import verify_runtime_profile

    checkpoint, original, files, root, memory_config = _reviewed_capacity_revision(tmp_path, capsys)
    revised = SupportRequest.from_yaml(memory_config)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    accepted = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="smaller than the accepted bound"):
        finalize(original, root, tmp_path / "unreviewed")
    with pytest.raises(ValueError, match="different request identity"):
        check_plan(revised, root)
    target = tmp_path / "resolved-revision"
    assert (
        cli.main(
            [
                "onboard",
                "finalize",
                "--config",
                files["request"],
                "--output-dir",
                str(root),
                "--memory-config",
                str(memory_config),
                "--resolved-output-dir",
                str(target),
            ]
        )
        == 0
    )
    capsys.readouterr()
    resolved = SupportRequest.from_yaml(target / "request.yaml")
    manifest = finalization_manifest(resolved)
    assert manifest["source_request_id"] == request_id(original)
    assert manifest["memory_revision"]["request_id"] == request_id(revised)
    assert resolved.profile_deployment().resources.runtime_memory.kv_cache_bytes == (
        revised.profile_deployment().resources.runtime_memory.kv_cache_bytes
    )
    assert before == {path: path.read_bytes() for path in before}
    assert checkpoint.read_bytes() == accepted
    check_plan(resolved, target)
    verify_runtime_profile(resolved)
    state = _load(checkpoint)
    artifacts = {key: {"archived": True} for key in state.configurations["tp2"].artifacts}
    artifacts["resolved-request"] = {"path": str(target / "request.yaml"), "kind": "request"}
    state, _ = save_checkpoint(
        checkpoint,
        patch={
            "configurations": {
                "tp2": {
                    "draft_request": resolved.model_dump(mode="json", exclude_none=True),
                    "artifacts": artifacts,
                }
            }
        },
        expected_revision=state.revision,
        accept=[],
    )
    assert state.configurations["tp2"].acceptance is None
    state, _ = save_checkpoint(checkpoint, patch={}, expected_revision=state.revision, accept=["tp2"])
    assert report(state, checkpoint)["configurations"]["tp2"]["profile_accepted"]
    memory_config.write_text(memory_config.read_text() + "\n")
    with pytest.raises(ValueError, match="memory revision.*changed"):
        verify_runtime_profile(resolved)


@pytest.mark.parametrize(
    ("change", "diagnostic"),
    [
        ("runtime", "non-capacity"),
        ("layout", "non-capacity"),
        ("workload", "non-capacity"),
        ("capacity_increase", "must lower"),
        ("capacity_override", "without capacity overrides"),
        ("forged_observation", "saved observed resources differ"),
        ("missing_attempts", "complete formal collection attempt bindings"),
        ("wrong_attempt", "different collection attempt"),
    ],
)
def test_capacity_revision_rejects_unrelated_changes_and_unverified_evidence(tmp_path, capsys, change, diagnostic):
    from aisimulate.support.runtime import verify_collection_runtime

    _, original, _, root, memory_config = _reviewed_capacity_revision(tmp_path, capsys)
    revised = SupportRequest.from_yaml(memory_config)
    resources = revised.profile_deployment().resources
    evidence = json.loads(resources.runtime_memory.provenance)
    checkpoint = json.loads((root / "fpm-checkpoint/fpm_forward.json").read_text())
    index = Path(checkpoint["runtime_observations"])
    if change == "runtime":
        revised.collection.gpu_memory_utilization = 0.8
    elif change == "layout":
        resources.cache_groups[0].page_size_bytes += 128
    elif change == "workload":
        revised.workload.request_count += 1
    elif change == "capacity_increase":
        resources.runtime_memory.kv_cache_bytes = (
            original.profile_deployment().resources.runtime_memory.kv_cache_bytes + 128
        )
    elif change == "capacity_override":
        resources.runtime_memory.kv_cache_bytes -= 128
        evidence["user_overrides"] = {
            "runtime_memory.kv_cache_bytes": {"value": resources.runtime_memory.kv_cache_bytes}
        }
    elif change == "forged_observation":
        evidence["observed_resources"]["runtime_memory"]["kv_cache_bytes"] -= 128
    elif change == "missing_attempts":
        checkpoint = None
    elif change == "wrong_attempt":
        checkpoint["runtime_observation_attempt_id"] = "unrelated"
    resources.runtime_memory.provenance = json.dumps(evidence)
    with pytest.raises(ValueError, match=diagnostic):
        verify_collection_runtime(original, index, collection_checkpoint=checkpoint, memory_request=revised)


def test_capacity_revision_requires_acceptance_and_exact_formal_observations(tmp_path, capsys):
    from aisimulate.support.finalization import finalize
    from aisimulate.support.runtime import verify_collection_runtime

    checkpoint, original, _, root, memory_config = _reviewed_capacity_revision(tmp_path, capsys)
    formal = json.loads((root / "fpm-checkpoint/fpm_forward.json").read_text())
    source = Path(formal["runtime_observations"])
    unrelated = tmp_path / "other-attempt"
    shutil.copytree(source.parent, unrelated)
    index = unrelated / source.name
    document = json.loads(index.read_text())
    attempt = document["configurations"]["tp2"]["attempts"][0]
    attempt["attempt_id"] = "unrelated-parent"
    document["configurations"]["tp2"]["active_attempt_id"] = attempt["attempt_id"]
    for phase in attempt["phases"].values():
        for ref in [phase["launch_manifest"], *phase["artifacts"]]:
            path = index.parent / ref["path"]
            record = json.loads(path.read_text())
            record["attempt_id"] = attempt["attempt_id"]
            _write(path, record)
            ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(index, document)
    assert _import(checkpoint, index, tmp_path / "unrelated-revision", configurations=["tp2"]) == 0
    revised_path = json.loads(capsys.readouterr().out)["configurations"]["tp2"]["request"]
    revised = SupportRequest.from_yaml(revised_path)
    with pytest.raises(ValueError, match="explicit acceptance"):
        finalize(original, root, tmp_path / "unaccepted", memory_config=revised_path)
    assert not (tmp_path / "unaccepted").exists()
    with pytest.raises(ValueError, match="same complete formal collection observations"):
        verify_collection_runtime(original, source, collection_checkpoint=formal, memory_request=revised)
    # The previously reviewed request is no longer the checkpoint's accepted
    # draft, even though its immutable evidence is still available.
    with pytest.raises(ValueError, match="explicit acceptance"):
        finalize(original, root, tmp_path / "superseded", memory_config=memory_config)


def test_large_runtime_provenance_reaches_supervised_recommendation_and_keeps_sources(tmp_path, capsys):
    from aisimulate.support.finalization import _merge_resources, _verify_collection, finalization_manifest
    from aisimulate.support.runtime import verify_runtime_profile

    # The fixture's arbitrary configuration label is independent of topology.
    checkpoint, original, files, root = _completed_runtime_collection(tmp_path, capsys, tp=8)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    accepted = checkpoint.read_bytes()
    target = tmp_path / "resolved"
    assert (
        cli.main(
            [
                "onboard",
                "finalize",
                "--config",
                files["request"],
                "--output-dir",
                str(root),
                "--resolved-output-dir",
                str(target),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert before == {path: path.read_bytes() for path in before}
    assert checkpoint.read_bytes() == accepted
    resolved = SupportRequest.from_yaml(target / "request.yaml")
    assert resolved.profile_deployment().tp == 8
    manifest = finalization_manifest(resolved)
    assert manifest == json.loads((target / "finalization.json").read_text())
    assert manifest["observation_provenance"] == "source_references"
    observation = json.loads(manifest["cell_observations"][0]["runtime_memory"]["provenance"])
    assert len(observation["rank_capacities"]) == 16
    assert len(observation["artifacts"]) == 18
    assert all(set(reference) == {"path", "sha256"} for reference in observation["artifacts"])
    for reference in manifest["source_artifacts"]:
        assert hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest() == reference["sha256"]
    output = tmp_path / "recommendation"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aisimulate",
            "recommend",
            "--config",
            str(target / "recommend/pilot.yaml"),
            "--output-dir",
            str(output),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    recommendation = json.loads((output / "recommendation.json").read_text())
    assert len(recommendation["candidates"]) == 1
    candidate = recommendation["candidates"][0]
    assert candidate["status"] == "failed"
    assert "FPM direct" in candidate["reason"] and "unsupported" in candidate["reason"]
    assert "bounded checkpoint budget" not in result.stderr
    events = [json.loads(line) for line in (output / "execution-events.jsonl").read_text().splitlines()]
    assert any(event["event"] == "candidate_completed" for event in events)
    verify_runtime_profile(resolved)
    observations, legacy_manifest, _, _ = _verify_collection(original, root)
    legacy = resolved.model_dump(mode="json")
    legacy["fpm_profile"]["deployments"][0]["resources"] = _merge_resources(observations, legacy_manifest)
    review = tmp_path / "resolved-checkpoint.json"
    state, _ = save_checkpoint(
        review,
        patch={
            "configurations": {
                "legacy": {"draft_request": legacy},
                "references": {"draft_request": resolved.model_dump(mode="json")},
            }
        },
        expected_revision=None,
        accept=["legacy", "references"],
    )
    assert all(config["profile_accepted"] for config in report(state, review)["configurations"].values())
    assert SupportRequest.model_validate(state.configurations["legacy"].draft_request).model_dump(mode="json") == legacy
    changed = resolved.model_dump(mode="json")
    resources = changed["fpm_profile"]["deployments"][0]["resources"]
    forged = copy.deepcopy(manifest)
    forged["source_artifacts"][0]["sha256"] = "0" * 64
    resources["runtime_memory"]["provenance"] = json.dumps(forged, sort_keys=True, separators=(",", ":"))
    with pytest.raises(ValueError, match="resources differ"):
        verify_runtime_profile(SupportRequest.model_validate(changed))
    for value in (None, "future-format", {}):
        forged["observation_provenance"] = value
        resources["runtime_memory"]["provenance"] = json.dumps(forged)
        with pytest.raises(ValueError, match="unsupported finalized observation provenance format"):
            verify_runtime_profile(SupportRequest.model_validate(changed))
    # The compact representation must still bind every immutable raw record.
    source = next(path for path in before if path.name == "worker-0-0.json")
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="changed|hash|incomplete"):
        verify_runtime_profile(resolved)
    assert all(not config["profile_accepted"] for config in report(state, review)["configurations"].values())


def test_formal_capacity_drop_and_geometry_change_are_rejected(tmp_path, capsys):
    from aisimulate.support.runtime import verify_collection_runtime

    _, index, request, _ = _imported(tmp_path, capsys)
    result = verify_collection_runtime(request, index)
    assert result["compatibility"]["status"] == "compatible"
    document = json.loads(index.read_text())
    for phase in document["configurations"]["tp2"]["attempts"][0]["phases"].values():
        for ref in phase["artifacts"]:
            path = index.parent / ref["path"]
            record = json.loads(path.read_text())
            if record["kind"] == "scheduler":
                record["cache"]["initial_free_blocks"] -= 1
                record["cache"]["reserved_blocks"] += 1
                record["cache"]["permanent_reserved_block_ids"].append(2)
            _write(path, record)
            ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(index, document)
    with pytest.raises(ValueError, match="smaller than the accepted bound"):
        verify_collection_runtime(request, index)
    for phase in document["configurations"]["tp2"]["attempts"][0]["phases"].values():
        for ref in phase["artifacts"]:
            path = index.parent / ref["path"]
            record = json.loads(path.read_text())
            cache = record["cache"]
            if record["kind"] == "scheduler":
                cache["initial_free_blocks"] += 1
                cache["reserved_blocks"] -= 1
                cache["permanent_reserved_block_ids"].remove(2)
            else:
                cache["layer_tensors"]["changed_layer"] = cache["layer_tensors"].pop("layer0")
                for allocation in cache["tensor_allocations"]:
                    allocation["shared_by"] = [
                        "changed_layer" if name == "layer0" else name for name in allocation["shared_by"]
                    ]
            for group in cache["groups"]:
                group["layer_names"] = ["changed_layer" if name == "layer0" else name for name in group["layer_names"]]
                if "layer_classes" in group and "layer0" in group["layer_classes"]:
                    group["layer_classes"]["changed_layer"] = group["layer_classes"].pop("layer0")
            _write(path, record)
            ref["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write(index, document)
    with pytest.raises(ValueError, match="geometry differs"):
        verify_collection_runtime(request, index)


def test_malformed_configuration_does_not_discard_an_independent_valid_draft(tmp_path, capsys):
    checkpoint, index, _ = _campaign(tmp_path)
    document = json.loads(index.read_text())
    document["configurations"]["incomplete"] = {"attempts": "malformed"}
    _write(index, document)
    assert _import(checkpoint, index, tmp_path / "drafts") == 1
    result = json.loads(capsys.readouterr().out)
    assert result["configurations"]["tp2"]["status"] == "complete"
    assert "list of objects" in str(result["configurations"]["incomplete"]["diagnostics"])


def test_import_optimistic_lock_keeps_concurrent_checkpoint_update(tmp_path, capsys, monkeypatch):
    import aisimulate.support.runtime as runtime

    checkpoint, index, _ = _campaign(tmp_path)
    actual_draft = runtime._draft

    def concurrent_edit(*args, **kwargs):
        result = actual_draft(*args, **kwargs)
        state = _load(checkpoint)
        save_checkpoint(
            checkpoint, patch={"research": {"concurrent": "preserved"}}, expected_revision=state.revision, accept=[]
        )
        return result

    monkeypatch.setattr(runtime, "_draft", concurrent_edit)
    with pytest.raises(SystemExit, match="2"):
        _import(checkpoint, index, tmp_path / "drafts", configurations=["tp2"])
    assert "checkpoint revision" in capsys.readouterr().err
    assert _load(checkpoint).research["concurrent"] == "preserved"
    assert not _load(checkpoint).configurations["tp2"].draft_request.get("fpm_profile")
