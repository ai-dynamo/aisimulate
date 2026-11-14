# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic source-bound snapshot identity tests; no GPU qualification."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from collector.fpm_forward import execution_evidence

from .test_k31_planner_precision import inputs  # noqa: F401
from .test_runtime_observations import _write

pytestmark = pytest.mark.unit


@pytest.fixture
def snapshot(request):
    bound_inputs = request.getfixturevalue("inputs")
    memory = bound_inputs["profile"]["deployments"][0]["resources"]["runtime_memory"]
    proof = json.loads(memory["provenance"])
    index_path = Path(proof["observations_index"]["path"])
    root = index_path.parent
    # The canonical collector archive lives beside cells/<cell>/raw.
    target = root / "runtime-observations.json"
    index = json.loads(index_path.read_text())
    entry = index["configurations"]["tp2"]
    attempt = next(a for a in entry["attempts"] if a["attempt_id"] == entry["active_attempt_id"])
    phase = attempt["phases"]["decode"]
    phase.update(cell_id="native-cell", collector_attempt_id="native-attempt")
    context_path = root / phase["launch_manifest"]["path"]
    context = json.loads(context_path.read_text())
    context.update(cell_id="native-cell", collector_attempt_id="native-attempt")
    phase["launch_manifest"]["sha256"] = _write(context_path, context)["sha256"]
    target.write_text(json.dumps(index))
    raw = root / "cells/native-cell/raw"
    raw.mkdir(parents=True)
    plan = SimpleNamespace(
        runtime_configuration="tp2",
        runtime_launch=bound_inputs["runtime_launch"],
        runtime_instrumentation=bound_inputs["runtime_instrumentation"],
    )
    cell = SimpleNamespace(cell_id="native-cell", workload_kind="decode")
    collection = SimpleNamespace(collector_attempt_id="native-attempt")
    return plan, cell, raw, collection, target


def test_snapshot_identity_uses_revalidated_same_attempt_worker(snapshot):
    plan, cell, raw, collection, index = snapshot
    models, source = execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)
    assert set(models) == {(0, 0, 0), (0, 1, 0)}
    assert source["sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()
    revision = "a" * 40
    public = plan.runtime_launch["identity"]["model"]
    model = {"model": f"/cache/models--{public.replace('/', '--')}/snapshots/{revision}", "revision": revision}
    native = dict(model)
    assert not execution_evidence._matches_pinned_hub_snapshot(public, model, {revision})
    assert execution_evidence._matches_pinned_hub_snapshot(public, model, {revision}, native)


@pytest.mark.parametrize("mutation", ["other_attempt", "other_cell", "wrong_context", "bad_config_hash", "symlink"])
def test_snapshot_identity_rejects_unbound_or_changed_native_evidence(snapshot, mutation):
    plan, cell, raw, collection, index_path = snapshot
    index = json.loads(index_path.read_text())
    entry = index["configurations"]["tp2"]
    attempt = next(a for a in entry["attempts"] if a["attempt_id"] == entry["active_attempt_id"])
    phase = attempt["phases"]["decode"]
    if mutation == "other_attempt":
        phase["collector_attempt_id"] = "other"
    elif mutation == "other_cell":
        phase["cell_id"] = "other"
    elif mutation == "wrong_context":
        path = index_path.parent / phase["launch_manifest"]["path"]
        context = json.loads(path.read_text())
        context["collector_attempt_id"] = "other"
        phase["launch_manifest"]["sha256"] = _write(path, context)["sha256"]
    elif mutation == "bad_config_hash":
        for ref in phase["artifacts"]:
            path = index_path.parent / ref["path"]
            record = json.loads(path.read_text())
            if record["kind"] == "worker":
                record["model_config_sha256"] = "0" * 64
                ref["sha256"] = _write(path, record)["sha256"]
                break
    index_path.write_text(json.dumps(index))
    if mutation == "symlink":
        moved = index_path.with_suffix(".original")
        index_path.rename(moved)
        index_path.symlink_to(moved)
    with pytest.raises((ValueError, KeyError)):
        execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)


@pytest.mark.parametrize(
    "mutation", ["wrong_native_model", "wrong_native_dtype", "wrong_commit", "wrong_loaded_hash", "missing_none_field"]
)
def test_snapshot_alias_fallback_does_not_weaken_identity(snapshot, mutation):
    plan, _, _, _, _ = snapshot
    revision = "a" * 40
    public = plan.runtime_launch["identity"]["model"]
    model = {
        "model": f"/cache/models--{public.replace('/', '--')}/snapshots/{revision}",
        "revision": revision,
        "dtype": "torch.bfloat16",
    }
    native = copy.deepcopy(model)
    if mutation == "wrong_native_model":
        native["model"] += "-other"
    elif mutation == "wrong_native_dtype":
        native["dtype"] = "torch.float16"
    elif mutation == "wrong_commit":
        model["model"] = model["model"].replace(revision, "0" * 40)
    elif mutation == "missing_none_field":
        model["loaded_config_commit_hash"] = None
    else:
        model["loaded_config_commit_hash"] = "0" * 40
        native["loaded_config_commit_hash"] = "0" * 40
    assert not execution_evidence._matches_pinned_hub_snapshot(public, model, {revision}, native)


@pytest.mark.parametrize(
    "mode", ["verified", "no_native_evidence", "different_quantization", "different_model_dtype", "wrong_cell_gemm"]
)
def test_mxfp4_execution_requires_matching_initialized_native_evidence(snapshot, mode):
    from collector.fpm_forward.runtime_observations import _launch_cell
    from collector.fpm_forward.runtime_probe import build_runtime_probe_plan

    plan, cell, raw, collection, _ = snapshot
    workers, _ = execution_evidence._runtime_worker_evidence(plan, cell, raw, collection)
    worker = workers[(0, 0, 0)]
    config = copy.deepcopy(worker["resolved_config"])
    payload = {"resolved_config": config, "graph_config": config["compilation_config"]}
    probe = build_runtime_probe_plan(plan.runtime_configuration, plan.runtime_launch, plan.runtime_instrumentation)
    active = SimpleNamespace(options=probe.options, fpm_profile=SimpleNamespace(model_revision="immutable"))
    native_cell = _launch_cell(plan.runtime_launch, "decode")
    if mode == "different_quantization":
        config["quantization_config"]["type"] = "other.Quantization"
    elif mode == "different_model_dtype":
        config["model_config"]["dtype"] = "torch.float16"
    elif mode == "wrong_cell_gemm":
        from dataclasses import replace

        native_cell = replace(native_cell, gemm_quant_mode="fp8")
    missing, failures = [], []
    execution_evidence._inspect_config(
        payload, native_cell, active, None, missing, failures, None if mode == "no_native_evidence" else worker
    )
    assert ("runtime quantization mapping is unaudited: mxfp4" in missing) == (mode == "no_native_evidence")
    assert ("runtime MXFP4 execution differs from the validated initialized worker" in failures) == mode.startswith(
        "different_"
    )
    assert any("planned GEMM/MoE identity" in failure for failure in failures) == (mode == "wrong_cell_gemm")
