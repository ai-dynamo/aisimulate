# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-bound memory verification keeps independently collected phases distinct."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from collector.fpm_forward.cli import _parser
from collector.fpm_forward.config import with_kv_warmup_defaults
from collector.fpm_forward.entry import _load_generator_overrides
from collector.fpm_forward.planner import BackendPolicy, _canonical_hash
from collector.fpm_forward.repeatability import _subset_plan, load_repeatability_source
from collector.fpm_forward.runtime_instrumentation import load_instrumentation

from aisimulate.support import runtime

from .test_onboard_runtime import _completed_runtime_collection

pytestmark = pytest.mark.unit


def _write(path, value):
    path.write_text(json.dumps(value))


def _ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def source_revisions(request, monkeypatch):
    from . import test_onboard_runtime

    revisions = getattr(request, "param", (None, None))
    observation_fixture = test_onboard_runtime.observation_fixture

    def with_revision(*args, **kwargs):
        index, launches = observation_fixture(*args, **kwargs)
        if revisions[0] is not None:
            launches["tp2"]["identity"]["aisimulate_revision"] = revisions[0]
        return index, launches

    monkeypatch.setattr(test_onboard_runtime, "observation_fixture", with_revision)
    return revisions


@pytest.fixture
def split_collection(tmp_path, capsys, request, monkeypatch, source_revisions):
    if getattr(request, "param", None) is not None:
        from aisimulate_core.sdk import fpm_profile

        # The synthetic repo's immutable checkpoint is local to this fixture.
        monkeypatch.setattr(
            fpm_profile,
            "_load_pinned_checkpoint_config",
            lambda model, revision: json.loads((tmp_path / "model-checkpoint/config.json").read_text()),
        )
    _, request, _, root = _completed_runtime_collection(tmp_path, capsys, model=getattr(request, "param", None))
    original_checkpoint = json.loads((root / "fpm-checkpoint/fpm_forward.json").read_text())
    original_campaign = root / "fpm-artifacts" / original_checkpoint["plan_sha256"][:16]
    original_plan = load_repeatability_source(original_campaign)
    args = _parser().parse_args(json.loads((root / "commands.json").read_text())["fpm_plan_local"][3:])
    bindings = []
    for phase in ("prefill", "decode"):
        campaign = tmp_path / (phase + "-independent")
        shutil.copytree(original_campaign, campaign)
        observations = campaign / "observations/observations.json"
        document = json.loads(observations.read_text())
        entry = document["configurations"].pop("tp2")
        name = "tp2-long-context-readiness-full-recollection"
        document["configurations"][name] = entry
        attempt = entry["attempts"][0]
        attempt_id = entry["active_attempt_id"] = attempt["attempt_id"] = phase + "-parent"
        attempt["phases"] = {phase: attempt["phases"][phase]}
        launch = entry["launch"]
        launch["deployment"]["cpus_per_task"] = 32
        launch["identity"].pop("aisimulate_revision", None)
        if source_revisions[1] is not None:
            launch["identity"]["aisimulate_revision"] = source_revisions[1]
        bundle_path = observations.parent / attempt["bundle"]["manifest"]
        manifest = json.loads(bundle_path.read_text())
        manifest["runtime"]["source_files"]["vllm/v1/core/block_pool.py"] = ("d" if phase == "prefill" else "e") * 64
        _write(bundle_path, manifest)
        bundle = load_instrumentation(bundle_path)
        attempt["bundle"]["sha256"] = bundle.sha256
        for result in attempt["phases"].values():
            result["status"] = "passed"
            for ref in [result["launch_manifest"], *result["artifacts"]]:
                path = observations.parent / ref["path"]
                record = json.loads(path.read_text())
                record.update(configuration=name, attempt_id=attempt_id, launch=launch, bundle_sha256=bundle.sha256)
                if ref.get("kind") == "observation":
                    record["identity"] = launch["identity"]
                    record["runtime"] = {**manifest["runtime"], "image": launch["deployment"]["image"]}
                _write(path, record)
                ref["sha256"] = _ref(path)["sha256"]
        _write(observations, document)
        cell = next(cell for cell in original_plan.cells if cell.workload_kind == phase)
        plan = replace(
            original_plan,
            aic_revision="reviewed-collector-revision",
            options=replace(original_plan.options, slurm_cpus_per_task=32),
            _runtime_observation_json=json.dumps(
                {"manifest": str(bundle_path), "bundle_sha256": bundle.sha256, "launch": launch, "configuration": name}
            ),
        )
        plan = _subset_plan(plan, {"cell_id": cell.cell_id, "benchmark_points": {}}, comparison_mode="full_grid")
        _write(campaign / "collection-plan.json", plan.to_dict())
        _write(campaign / "generator-overrides.json", _load_generator_overrides(args))
        checkpoint = {
            **original_checkpoint,
            "plan_sha256": plan.sha256,
            "runtime_observation_attempt_id": attempt_id,
            "runtime_observations": str(observations),
            "cells": {cell.cell_id: original_checkpoint["cells"][cell.cell_id]},
        }
        checkpoint_path = campaign / "checkpoint.json"
        _write(checkpoint_path, checkpoint)
        bindings.append(
            {
                "phase": phase,
                "plan": {"path": str(campaign / "collection-plan.json"), "sha256": plan.sha256},
                "checkpoint": _ref(checkpoint_path),
                "observations": _ref(observations),
                "attempt_id": attempt_id,
                "collector_attempt_id": checkpoint["cells"][cell.cell_id]["attempt_id"],
            }
        )
    return request, bindings


def _review(request, bindings):
    for binding in bindings:
        binding["review"] = {
            **runtime.collection_runtime_review(request, binding),
            "rationale": "Synthetic regression: reviewed CPU allocation and phase-specific source changes.",
        }


@pytest.fixture
def archived_controls(split_collection, request):
    support_request, bindings = split_collection
    controls = getattr(
        request,
        "param",
        {
            "extra_cli_args": ["--benchmark-timeout", "12600"],
            "extra_env": [
                {"name": "DYN_BENCH_CONTENT_SEED", "value": "0"},
                {"name": "DYN_BENCH_PREFILL_CONTENT", "value": "random"},
                {"name": "DYN_BENCH_POOL_TAG", "value": ""},
                {"name": "DYN_BENCH_GIANT_KV_THRESHOLD", "value": "1000000"},
                {"name": "DYN_BENCH_GIANT_KV_REPEATS", "value": "3"},
                {"name": "DYN_FORWARDPASS_METRIC_PORT", "value": "7020"},
                {"name": "FPM_RESULT_WAIT_TIMEOUT_SECONDS", "value": "14100"},
                {"name": "FPM_FINALIZATION_TIMEOUT_SECONDS", "value": "600"},
            ],
        },
    )
    binding = bindings[0]
    path = Path(binding["plan"]["path"])
    plan = load_repeatability_source(path.parent)
    policy = plan.cells[0].backend_policy.to_dict()
    policy["generator_overrides"].setdefault("params", {}).setdefault("agg", {})["extra_cli_args"] = controls[
        "extra_cli_args"
    ]
    policy = BackendPolicy(**policy)
    deployment_path = path.parent / "generator-overrides.json"
    deployment = json.loads(deployment_path.read_text())
    deployment.setdefault("K8sConfig", {}).setdefault("extra_env", []).extend(controls.get("extra_env", []))
    _write(deployment_path, deployment)
    plan = replace(
        plan,
        cells=(replace(plan.cells[0], backend_policy=policy),),
        backend_policies=(policy,),
        generator_config_sha256=_canonical_hash(with_kv_warmup_defaults(deployment)),
    )
    plan = _subset_plan(plan, {"cell_id": plan.cells[0].cell_id, "benchmark_points": {}}, comparison_mode="full_grid")
    _write(path, plan.to_dict())
    binding["plan"]["sha256"] = plan.sha256
    checkpoint_path = Path(binding["checkpoint"]["path"])
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["plan_sha256"] = plan.sha256
    _write(checkpoint_path, checkpoint)
    binding["checkpoint"] = _ref(checkpoint_path)
    # Reload the archived representation through the real hash validation path.
    assert load_repeatability_source(path.parent).sha256 == plan.sha256
    return support_request, bindings


def test_archived_collection_controls_require_exact_review_and_preserve_plan(archived_controls):
    request, bindings = archived_controls
    original_request = request.model_dump(mode="json")
    original_source = runtime.verify_runtime_profile(request)
    archived = {
        path: path.read_bytes()
        for binding in bindings
        for path in Path(binding["plan"]["path"]).parent.rglob("*")
        if path.is_file()
    }
    changes = [
        {
            "field": "/collection_controls/prefill/benchmark_timeout",
            "accepted": [],
            "collection": ["--benchmark-timeout", "12600"],
        },
        *[
            {
                "field": "/collection_controls/environment/" + name,
                "accepted": {"missing": True},
                "collection": value,
            }
            for name, value in {
                "DYN_BENCH_CONTENT_SEED": "0",
                "DYN_BENCH_PREFILL_CONTENT": "random",
                "DYN_BENCH_POOL_TAG": "",
                "DYN_BENCH_GIANT_KV_THRESHOLD": "1000000",
                "DYN_BENCH_GIANT_KV_REPEATS": "3",
                "DYN_FORWARDPASS_METRIC_PORT": "7020",
                "FPM_RESULT_WAIT_TIMEOUT_SECONDS": "14100",
                "FPM_FINALIZATION_TIMEOUT_SECONDS": "600",
            }.items()
        ],
    ]
    review = runtime.collection_runtime_review(request, bindings[0])
    assert all(change in review["changes"] for change in changes)
    with pytest.raises(ValueError, match="exact runtime change review required"):
        runtime.verify_collection_runtime_plans(request, bindings)
    _review(request, bindings)
    for field in (change["field"] for change in changes):
        for mutation in ("missing", "accepted", "collection"):
            stale = copy.deepcopy(bindings)
            entries = stale[0]["review"]["changes"]
            entry = next(item for item in entries if item["field"] == field)
            if mutation == "missing":
                entries.remove(entry)
            else:
                entry[mutation] = ["unreviewed-timeout"]
            with pytest.raises(ValueError, match="exact runtime change review required"):
                runtime.verify_collection_runtime_bindings(request, stale)
    assert runtime.verify_collection_runtime_plans(request, bindings)["status"] == "verified"
    result = runtime.verify_collection_runtime_bindings(request, bindings)
    assert result["compatibility"]["status"] == "compatible"
    assert "timing or measurement equivalence" in result["compatibility"]["scope"]
    assert result["phases"]["prefill"]["binding"] == bindings[0]
    assert all(path.read_bytes() == contents for path, contents in archived.items())
    assert request.model_dump(mode="json") == original_request
    assert runtime.verify_runtime_profile(request) == original_source


@pytest.mark.parametrize(
    "archived_controls",
    [
        {
            "extra_cli_args": [],
            "extra_env": [{"name": "FPM_FINALIZATION_TIMEOUT_SECONDS", "value": value}],
        }
        for value in ("1", "600", "3600")
    ],
    indirect=True,
)
def test_archived_finalization_budget_requires_review_before_memory_binding(archived_controls):
    request, bindings = archived_controls
    originals = {
        path: path.read_bytes()
        for binding in bindings
        for path in Path(binding["plan"]["path"]).parent.rglob("*")
        if path.is_file()
    }
    deployment = json.loads((Path(bindings[0]["plan"]["path"]).parent / "generator-overrides.json").read_text())
    budget = next(
        item["value"]
        for item in deployment["K8sConfig"]["extra_env"]
        if item["name"] == "FPM_FINALIZATION_TIMEOUT_SECONDS"
    )
    review = runtime.collection_runtime_review(request, bindings[0])
    assert {
        "field": "/collection_controls/environment/FPM_FINALIZATION_TIMEOUT_SECONDS",
        "accepted": {"missing": True},
        "collection": budget,
    } in review["changes"]
    with pytest.raises(ValueError, match="exact runtime change review required"):
        runtime.verify_collection_runtime_bindings(request, bindings)
    _review(request, bindings)
    assert runtime.verify_collection_runtime_plans(request, bindings)["status"] == "verified"
    assert runtime.verify_collection_runtime_bindings(request, bindings)["compatibility"]["status"] == "compatible"
    assert all(path.read_bytes() == contents for path, contents in originals.items())


@pytest.mark.parametrize(
    "archived_controls",
    [{"extra_cli_args": []}, {"extra_cli_args": ["--benchmark-timeout=12600"]}],
    indirect=True,
)
def test_archived_empty_policy_and_equals_deadline_are_explicitly_verified(archived_controls):
    request, bindings = archived_controls
    _review(request, bindings)
    assert runtime.verify_collection_runtime_bindings(request, bindings)["status"] == "complete"


@pytest.mark.parametrize(
    "archived_controls",
    [
        {"extra_cli_args": args}
        for args in (
            ["--benchmark-timeout"],
            ["--benchmark-timeout", ""],
            ["--benchmark-timeout", "0"],
            ["--benchmark-timeout=-1"],
            ["--benchmark-timeout", "1.5"],
            ["--benchmark-timeout", 12600],
            ["--benchmark-timeout 12600"],
            ["--benchmark-timeout", "12600", "--benchmark-timeout=12600"],
            ["--benchmark-timeout", "12600", "--benchmark-timeout=10800"],
            ["--benchmark-timeout", "12600", "--unknown-option", "1"],
            ["--benchmark-timeout", "12600", "--max-num-seqs", "32"],
            ["--benchmark-timeout", "12600", "--enforce-eager"],
            ["--benchmark-timeout", "12600", "--kv-cache-dtype", "fp8"],
        )
    ]
    + [
        {
            "extra_cli_args": [],
            "extra_env": [{"name": "FPM_FINALIZATION_TIMEOUT_SECONDS", "value": value}],
        }
        for value in (
            None,
            600,
            True,
            "",
            "0",
            "3601",
            "1.5",
            "-1",
            "+1",
            "01",
            " 600",
            "600 ",
            "600\n",
            "600\x00",
            "٦٠٠",
            "\uff16\uff10\uff10",
            "${TIMEOUT}",
            "9" * 5000,
        )
    ]
    + [
        {
            "extra_cli_args": [],
            "extra_env": [
                {"name": "FPM_FINALIZATION_TIMEOUT_SECONDS", "value": "600"},
                {"name": "FPM_FINALIZATION_TIMEOUT_SECONDS", "value": value},
            ],
        }
        for value in ("600", "1200")
    ]
    + [
        {
            "extra_cli_args": [],
            "extra_env": [{"name": "FPM_FINALIZATION_TIMEOUT_SECONDS", **value}],
        }
        for value in ({"valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}}, {"value": "600", "valueFrom": {}})
    ]
    + [
        {
            "extra_cli_args": ["--benchmark-timeout", "12600"],
            "extra_env": [{"name": "FPM_RESULT_WAIT_TIMEOUT_SECONDS", "value": value}],
        }
        for value in (None, 14100, True, "", "0", "43201", "1.5", "-1", " 14100", "${TIMEOUT}")
    ]
    + [
        {
            "extra_cli_args": ["--benchmark-timeout", "12600"],
            "extra_env": [
                {"name": "FPM_RESULT_WAIT_TIMEOUT_SECONDS", "value": "14100"},
                {"name": "FPM_RESULT_WAIT_TIMEOUT_SECONDS", "value": value},
            ],
        }
        for value in ("14100", "14000")
    ]
    + [
        {"extra_cli_args": [], "extra_env": [{"name": name, "value": "off"}]}
        for name in (
            "UNKNOWN_CONTROL",
            "DYN_BENCH_PREFILL_CONTENT",
            "DYN_BENCH_KV_WARMUP",
            "DYN_BENCH_PREFILL_REAL_SEED",
            "VLLM_ATTENTION_BACKEND",
        )
    ]
    + [
        {"extra_cli_args": [], "extra_env": [{"name": name, "value": value}]}
        for name, value in (
            ("DYN_BENCH_CONTENT_SEED", "seed\nchanged"),
            ("DYN_BENCH_POOL_TAG", None),
            ("DYN_BENCH_GIANT_KV_THRESHOLD", "-1"),
            ("DYN_BENCH_GIANT_KV_REPEATS", "1.5"),
            ("DYN_FORWARDPASS_METRIC_PORT", "0"),
            ("DYN_FORWARDPASS_METRIC_PORT", "65536"),
        )
    ]
    + [
        {
            "extra_cli_args": [],
            "extra_env": [{"name": "DYN_BENCH_CONTENT_SEED", "value": "0"}] * 2,
        },
        {
            "extra_cli_args": [],
            "extra_env": [{"name": "DYN_BENCH_POOL_TAG", "value": "", "valueFrom": "unreviewed"}],
        },
    ],
    indirect=True,
)
def test_archived_controls_reject_malformed_deadlines_and_serving_changes(archived_controls):
    request, bindings = archived_controls
    with pytest.raises(ValueError):
        runtime.collection_runtime_review(request, bindings[0])


def test_legacy_launch_validation_remains_strict_on_collection_controls(archived_controls):
    from collector.fpm_forward.repeatability import load_repeatability_deployment
    from collector.fpm_forward.runtime_probe import validate_collection_probe_launch

    _, bindings = archived_controls
    directory = Path(bindings[0]["plan"]["path"]).parent
    plan = load_repeatability_source(directory)
    with pytest.raises(ValueError, match="topology or precision differs"):
        validate_collection_probe_launch(
            plan.runtime_launch,
            plan.runtime_instrumentation,
            model_path=plan.model_path,
            system=plan.system,
            backend=plan.backend,
            backend_version=plan.capability.aic_database_version,
            cells=plan.cells,
            options=plan.options,
            profile=plan.fpm_profile,
            generator_overrides=load_repeatability_deployment(directory),
            phases=("prefill",),
        )


def test_renamed_split_collection_requires_exact_review_and_keeps_attempts(split_collection):
    request, bindings = split_collection
    original_request = request.model_dump(mode="json")
    with pytest.raises(ValueError, match="review"):
        runtime.verify_collection_runtime_bindings(request, bindings)
    _review(request, bindings)
    before = {
        Path(ref["path"]): Path(ref["path"]).read_bytes()
        for item in bindings
        for ref in (item["plan"], item["checkpoint"], item["observations"])
    }
    planned = [
        {key: copy.deepcopy(value) for key, value in item.items() if key in {"phase", "plan", "review"}}
        for item in bindings
    ]
    assert runtime.verify_collection_runtime_plans(request, planned)["status"] == "verified"
    result = runtime.verify_collection_runtime_bindings(request, bindings)
    assert result["compatibility"]["status"] == "compatible"
    assert {item["provenance"]["attempt_id"] for item in result["phases"].values()} == {
        "prefill-parent",
        "decode-parent",
    }
    assert (
        result["resources"]["runtime_memory"]["kv_cache_bytes"]
        == request.profile_deployment().resources.runtime_memory.kv_cache_bytes
    )
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert request.model_dump(mode="json") == original_request
    inputs = {item["path"]: item["sha256"] for item in result["inputs"]}
    assert all(str(path) in inputs for path in before)
    assert any("worker-0-0.json" in path for path in inputs)
    assert any("manifest.yaml" in path for path in inputs)
    assert all(_ref(Path(path))["sha256"] == digest for path, digest in inputs.items())
    with pytest.raises(ValueError, match="incomplete"):
        runtime.verify_collection_runtime(request, Path(bindings[0]["observations"]["path"]))


@pytest.mark.parametrize(
    "source_revisions",
    [("1" * 40, "2" * 40), (None, "2" * 40), ("1" * 40, None)],
    indirect=True,
)
def test_aisimulate_source_revision_requires_exact_review_and_preserves_provenance(split_collection, source_revisions):
    request, bindings = split_collection
    original_request = request.model_dump(mode="json")
    original_source = runtime.verify_runtime_profile(request)
    expected = {
        "field": "/identity/aisimulate_revision",
        "accepted": source_revisions[0] if source_revisions[0] is not None else {"missing": True},
        "collection": source_revisions[1] if source_revisions[1] is not None else {"missing": True},
    }
    for binding in bindings:
        review = runtime.collection_runtime_review(request, binding)
        assert expected in review["changes"]
        assert review["collector_revision"] == "reviewed-collector-revision"
    with pytest.raises(ValueError, match="exact runtime change review required"):
        runtime.verify_collection_runtime_plans(request, bindings)
    _review(request, bindings)
    for change in ("missing", "accepted", "collection"):
        stale = copy.deepcopy(bindings)
        changes = stale[0]["review"]["changes"]
        entry = next(item for item in changes if item["field"] == expected["field"])
        if change == "missing":
            changes.remove(entry)
        else:
            entry[change] = "unreviewed-revision"
        with pytest.raises(ValueError, match="exact runtime change review required"):
            runtime.verify_collection_runtime_bindings(request, stale)
    assert runtime.verify_collection_runtime_plans(request, bindings)["status"] == "verified"
    result = runtime.verify_collection_runtime_bindings(request, bindings)
    assert result["compatibility"]["status"] == "compatible"
    for phase in result["phases"].values():
        identity = phase["provenance"]["launch"]["identity"]
        assert identity.get("aisimulate_revision", {"missing": True}) == expected["collection"]
        assert expected in phase["binding"]["review"]["changes"]
    assert request.model_dump(mode="json") == original_request
    assert runtime.verify_runtime_profile(request) == original_source


def _change_index(binding, change):
    path = Path(binding["observations"]["path"])
    document = json.loads(path.read_text())
    entry = next(iter(document["configurations"].values()))
    change(path.parent, entry, entry["attempts"][0]["phases"][binding["phase"]])
    _write(path, document)
    binding["observations"] = _ref(path)


def _change_records(binding, change):
    def apply(root, entry, phase):
        for ref in phase["artifacts"]:
            if ref.get("kind") != "observation":
                continue
            path = root / ref["path"]
            record = json.loads(path.read_text())
            change(record)
            _write(path, record)
            ref["sha256"] = _ref(path)["sha256"]

    _change_index(binding, apply)


@pytest.mark.parametrize(
    "change",
    [
        "missing-phase",
        "duplicate-phase",
        "runtime-attempt",
        "native-attempt",
        "checkpoint-plan",
        "plan-hash",
        "review-bundle",
        "review-diff",
        "review-revision",
        "review-rationale",
    ],
)
def test_explicit_phase_plan_attempt_and_review_bindings_are_required(split_collection, change):
    request, bindings = split_collection
    _review(request, bindings)
    binding = bindings[0]
    if change == "missing-phase":
        bindings.pop()
    elif change == "duplicate-phase":
        bindings.append(copy.deepcopy(binding))
    elif change == "runtime-attempt":
        binding["attempt_id"] = "another-parent"
    elif change == "native-attempt":
        binding["collector_attempt_id"] = "another-native-attempt"
    elif change == "checkpoint-plan":
        path = Path(binding["checkpoint"]["path"])
        checkpoint = json.loads(path.read_text())
        checkpoint["plan_sha256"] = "0" * 64
        _write(path, checkpoint)
        binding["checkpoint"] = _ref(path)
    elif change == "plan-hash":
        binding["plan"]["sha256"] = "0" * 64
    elif change == "review-bundle":
        binding["review"]["collection_bundle_sha256"] = "0" * 64
    elif change == "review-diff":
        binding["review"]["changes"].pop()
    elif change == "review-revision":
        binding["review"]["collector_revision"] = "unreviewed-collector"
    else:
        binding["review"]["rationale"] = ""
    with pytest.raises(ValueError):
        runtime.verify_collection_runtime_bindings(request, bindings)


@pytest.mark.parametrize("collection", ["split_collection", "archived_controls"])
@pytest.mark.parametrize(
    "change",
    [
        "missing-rank",
        "duplicate-rank",
        "artifact-bytes",
        "artifact-escape",
        "launch-attempt",
        "model",
        "revision",
        "model-config-hash",
        "model-sidecar-hash",
        "precision",
        "scheduler",
        "graph",
        "settings",
        "geometry",
        "capacity",
    ],
)
def test_raw_evidence_is_independently_validated_under_each_frozen_launch(request, collection, change):
    request, bindings = request.getfixturevalue(collection)
    _review(request, bindings)
    binding = bindings[0]
    if change in {"missing-rank", "duplicate-rank", "artifact-bytes", "artifact-escape", "launch-attempt"}:

        def mutate(root, entry, phase):
            ref = next(ref for ref in phase["artifacts"] if "worker" in ref["path"])
            path = root / ref["path"]
            if change == "missing-rank":
                phase["artifacts"].remove(ref)
            elif change == "duplicate-rank":
                duplicate = root / "duplicate-worker.json"
                duplicate.write_bytes(path.read_bytes())
                phase["artifacts"].append({**ref, "path": duplicate.name})
            elif change == "artifact-bytes":
                path.write_text(path.read_text() + "\n")
            elif change == "artifact-escape":
                outside = root.parent / "outside.json"
                outside.write_bytes(path.read_bytes())
                ref["path"] = "../outside.json"
            else:
                context_path = root / phase["launch_manifest"]["path"]
                context = json.loads(context_path.read_text())
                context["collector_attempt_id"] = "unrelated-native"
                _write(context_path, context)
                phase["launch_manifest"]["sha256"] = _ref(context_path)["sha256"]

        _change_index(binding, mutate)
    else:

        def mutate(record):
            config = record["resolved_config"]
            if change == "model":
                config["model_config"]["model"] = "other/model"
            elif change == "revision":
                config["model_config"].update(revision="wrong", loaded_config_commit_hash="wrong")
            elif change == "model-config-hash":
                record["model_config_sha256"] = "0" * 64
            elif change == "model-sidecar-hash":
                record["model_config_source_files"] = {"hf_quant_config.json": "0" * 64}
            elif change == "precision":
                config["cache_config"]["cache_dtype"] = "fp8"
            elif change == "scheduler":
                config["scheduler_config"]["max_num_seqs"] += 1
            elif change == "graph":
                config["compilation_config"]["cudagraph_capture_sizes"] = [1, 4]
            elif change == "settings":
                config["model_config"]["trust_remote_code"] = True
            elif change == "capacity" and record["kind"] == "scheduler":
                record["cache"]["initial_free_blocks"] -= 100
                record["cache"]["reserved_blocks"] += 100
                record["cache"]["permanent_reserved_block_ids"].extend(range(2, 102))
            elif change == "geometry":
                cache = record["cache"]
                for group in cache["groups"]:
                    group["layer_names"] = ["changed" if name == "layer0" else name for name in group["layer_names"]]
                    if "layer_classes" in group and "layer0" in group["layer_classes"]:
                        group["layer_classes"]["changed"] = group["layer_classes"].pop("layer0")
                if record["kind"] == "worker":
                    cache["layer_tensors"]["changed"] = cache["layer_tensors"].pop("layer0")
                    for allocation in cache["tensor_allocations"]:
                        allocation["shared_by"] = [
                            "changed" if name == "layer0" else name for name in allocation["shared_by"]
                        ]

        _change_records(binding, mutate)
    with pytest.raises(ValueError, match="prefill:"):
        runtime.verify_collection_runtime_bindings(request, bindings)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("identity", "model", "other/model"),
        ("identity", "model_revision", "another-revision"),
        ("identity", "framework_version", "0.29.0"),
        ("topology", "tp", 4),
        ("precision", "gemm_quant_mode", "fp8"),
        ("collection", "max_num_seqs", 32),
        ("collection", "cudagraph_mode", "NONE"),
        ("model_config", "sha256", "0" * 64),
    ],
)
def test_preflight_rejects_incompatible_frozen_plans_before_attempts(split_collection, section, field, value):
    request, bindings = split_collection
    _review(request, bindings)
    binding = bindings[0]
    path = Path(binding["plan"]["path"])
    plan = load_repeatability_source(path.parent)
    observation = json.loads(plan._runtime_observation_json)
    observation["launch"][section][field] = value
    plan = replace(plan, _runtime_observation_json=json.dumps(observation))
    plan = _subset_plan(plan, {"cell_id": plan.cells[0].cell_id, "benchmark_points": {}}, comparison_mode="full_grid")
    _write(path, plan.to_dict())
    binding["plan"]["sha256"] = plan.sha256
    with pytest.raises(ValueError, match="immutable model/precision/settings differ"):
        runtime.verify_collection_runtime_plans(request, bindings)


def test_readiness_uses_phase_bound_memory_and_keeps_native_timing_failures(split_collection, monkeypatch):
    from aisimulate.support import collection_readiness

    request, bindings = split_collection
    _review(request, bindings)
    # Timing assessment is independently covered by native readiness fixtures;
    # exercise its composition with real plan/raw-memory verification here.
    monkeypatch.setattr(
        collection_readiness,
        "_cell_report",
        lambda plan, cell, campaign, entry: {"status": "ready", "blockers": [], "cell": cell.to_dict()},
    )
    report = collection_readiness.assess_readiness(request, Path("unused"), Path("unused"), runtime_bindings=bindings)
    assert report["ready_for_full_collection"]
    assert set(report["runtime_compatibility"]["phases"]) == {"prefill", "decode"}
    monkeypatch.setattr(
        collection_readiness,
        "_cell_report",
        lambda plan, cell, campaign, entry: {
            "status": "blocked",
            "blockers": ["native timing rank missing"],
            "cell": cell.to_dict(),
        },
    )
    report = collection_readiness.assess_readiness(request, Path("unused"), Path("unused"), runtime_bindings=bindings)
    assert not report["ready_for_full_collection"]
    assert "native timing rank missing" in " ".join(report["blockers"])
    bindings[0]["attempt_id"] = "unrelated"
    report = collection_readiness.assess_readiness(request, Path("unused"), Path("unused"), runtime_bindings=bindings)
    assert not report["ready_for_full_collection"]
    assert "runtime checkpoint" in " ".join(report["blockers"])


@pytest.mark.parametrize("conflicting_plan", [False, True])
def test_readiness_rejects_expected_plan_with_explicit_bindings(split_collection, monkeypatch, conflicting_plan):
    from aisimulate.support import collection_readiness

    request, bindings = split_collection
    _review(request, bindings)
    expected_plan = load_repeatability_source(Path(bindings[0]["plan"]["path"]).parent)
    if conflicting_plan:
        expected_plan = replace(expected_plan, sha256="0" * 64)
    monkeypatch.setattr(
        collection_readiness,
        "_cell_report",
        lambda plan, cell, campaign, entry: {"status": "ready", "blockers": [], "cell": cell.to_dict()},
    )
    with pytest.raises(ValueError, match="expected_plan and runtime_bindings are mutually exclusive"):
        collection_readiness.assess_readiness(
            request,
            Path("unused"),
            Path("unused"),
            expected_plan=expected_plan,
            runtime_bindings=bindings,
        )


@pytest.mark.parametrize("split_collection", ["example/model"], indirect=True)
def test_exact_checkpoint_snapshot_spelling_preserves_full_settings_evidence(split_collection):
    request, bindings = split_collection
    _review(request, bindings)
    snapshot = "/cache/models--example--model/snapshots/" + request.identity.model_revision
    for binding in bindings:
        _change_records(binding, lambda record: record["resolved_config"]["model_config"].update(model=snapshot))
    result = runtime.verify_collection_runtime_bindings(request, bindings)
    assert result["compatibility"]["status"] == "compatible"
    for phase in result["phases"].values():
        assert phase["compatibility"]["runtime_settings_differences"] == [
            {"field": "/runtime_settings/model_config/model", "accepted": "example/model", "collection": snapshot}
        ]
        assert phase["provenance"]["runtime_settings"]["model_config"]["model"] == snapshot
    _change_records(
        bindings[0],
        lambda record: record["resolved_config"]["model_config"].update(model=snapshot.replace("4" * 40, "5" * 40)),
    )
    with pytest.raises(ValueError, match="observed model path"):
        runtime.verify_collection_runtime_bindings(request, bindings)


def test_default_readiness_requires_compatible_raw_memory_without_explicit_bindings(tmp_path, capsys, monkeypatch):
    from aisimulate.support import collection_readiness

    _, request, _, root = _completed_runtime_collection(tmp_path, capsys)
    checkpoint = json.loads((root / "fpm-checkpoint/fpm_forward.json").read_text())
    campaign = root / "fpm-artifacts" / checkpoint["plan_sha256"][:16]
    args = _parser().parse_args(json.loads((root / "commands.json").read_text())["fpm_plan_local"][3:])
    _write(campaign / "generator-overrides.json", _load_generator_overrides(args))
    monkeypatch.setattr(
        collection_readiness,
        "_cell_report",
        lambda plan, cell, campaign, entry: {"status": "ready", "blockers": [], "cell": cell.to_dict()},
    )
    report = collection_readiness.assess_readiness(request, root, root / "fpm-checkpoint")
    assert report["ready_for_full_collection"], report["blockers"]
    assert report["runtime_compatibility"]["compatibility"]["status"] == "compatible"
    plan = load_repeatability_source(campaign)
    report = collection_readiness.assess_readiness(request, root, root / "fpm-checkpoint", expected_plan=plan)
    assert report["ready_for_full_collection"], report["blockers"]
    report = collection_readiness.assess_readiness(
        request,
        root,
        root / "fpm-checkpoint",
        expected_plan=replace(plan, sha256="0" * 64),
    )
    assert not report["ready_for_full_collection"]
    assert "current frozen runtime/launch plan" in " ".join(report["blockers"])
    binding = {"phase": "prefill", "observations": _ref(Path(checkpoint["runtime_observations"]))}
    _change_records(
        binding, lambda record: record["runtime"]["source_files"].update({"vllm/v1/core/block_pool.py": "0" * 64})
    )
    report = collection_readiness.assess_readiness(request, root, root / "fpm-checkpoint")
    assert not report["ready_for_full_collection"]
    assert "runtime build/source" in " ".join(report["blockers"])


@pytest.fixture
def default_collection(tmp_path, capsys):
    _, request, _, root = _completed_runtime_collection(tmp_path, capsys)
    checkpoint_path = root / "fpm-checkpoint/fpm_forward.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    campaign = root / "fpm-artifacts" / checkpoint["plan_sha256"][:16]
    args = _parser().parse_args(json.loads((root / "commands.json").read_text())["fpm_plan_local"][3:])
    _write(campaign / "generator-overrides.json", _load_generator_overrides(args))
    return request, root, checkpoint_path


def test_missing_runtime_index_preserves_all_cell_reports(default_collection):
    from aisimulate.support.collection_readiness import assess_readiness

    request, root, checkpoint_path = default_collection
    checkpoint = json.loads(checkpoint_path.read_text())
    original = assess_readiness(request, root, checkpoint_path.parent)
    assert original["runtime_compatibility"]["status"] == "complete"
    assert len(original["selected_cells"]) == 2
    checkpoint.pop("runtime_observations")
    _write(checkpoint_path, checkpoint)

    report = assess_readiness(request, root, checkpoint_path.parent)

    assert report["selected_cells"] == original["selected_cells"]
    assert report["campaigns"][0]["cells"] == original["campaigns"][0]["cells"]
    assert not report["ready_for_full_collection"]
    assert report["runtime_compatibility"] is None
    assert "collector checkpoint records no runtime observation index" in " ".join(report["blockers"])


@pytest.mark.parametrize("path_kind", ["relative", "symlinked_directory"])
def test_default_readiness_resolves_index_from_checkpoint(default_collection, tmp_path, monkeypatch, path_kind):
    from aisimulate.support.collection_readiness import assess_readiness

    request, root, checkpoint_path = default_collection
    checkpoint = json.loads(checkpoint_path.read_text())
    index = Path(checkpoint["runtime_observations"])
    if path_kind == "relative":
        checkpoint["runtime_observations"] = str(Path("..") / index.relative_to(root))
    else:
        alias = tmp_path / "campaign-alias"
        alias.symlink_to(index.parent, target_is_directory=True)
        checkpoint["runtime_observations"] = str(alias / index.name)
    _write(checkpoint_path, checkpoint)
    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    report = assess_readiness(request, root, checkpoint_path.parent)

    assert report["runtime_compatibility"] is not None, report["blockers"]
    assert report["runtime_compatibility"]["status"] == "complete"
    assert len(report["selected_cells"]) == 2
    for phase in report["runtime_compatibility"]["phases"].values():
        assert phase["binding"]["observations"] == _ref(index.resolve())


@pytest.mark.parametrize("index_state", ["empty", "null", "non_string", "directory", "missing_file", "invalid_json"])
def test_unusable_runtime_index_preserves_all_cell_reports(default_collection, index_state):
    from aisimulate.support.collection_readiness import assess_readiness

    request, root, checkpoint_path = default_collection
    original = assess_readiness(request, root, checkpoint_path.parent)
    checkpoint = json.loads(checkpoint_path.read_text())
    index = Path(checkpoint["runtime_observations"])
    if index_state == "invalid_json":
        index.write_text("{")
    else:
        checkpoint["runtime_observations"] = {
            "empty": "",
            "null": None,
            "non_string": 17,
            "directory": str(index.parent),
            "missing_file": str(index.parent / "missing.json"),
        }[index_state]
        _write(checkpoint_path, checkpoint)

    report = assess_readiness(request, root, checkpoint_path.parent)

    assert report["selected_cells"] == original["selected_cells"]
    assert report["campaigns"][0]["cells"] == original["campaigns"][0]["cells"]
    assert not report["ready_for_full_collection"]
    assert report["runtime_compatibility"] is None
    assert any("runtime memory:" in reason for reason in report["blockers"])


def test_unreadable_runtime_index_preserves_all_cell_reports(default_collection, monkeypatch):
    from aisimulate.support.collection_readiness import assess_readiness

    request, root, checkpoint_path = default_collection
    original = assess_readiness(request, root, checkpoint_path.parent)
    index = Path(json.loads(checkpoint_path.read_text())["runtime_observations"]).resolve()
    read_bytes = Path.read_bytes

    def denied_index(path):
        if path.resolve() == index:
            raise PermissionError("runtime observation index is not readable")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", denied_index)
    report = assess_readiness(request, root, checkpoint_path.parent)

    assert report["selected_cells"] == original["selected_cells"]
    assert report["campaigns"][0]["cells"] == original["campaigns"][0]["cells"]
    assert not report["ready_for_full_collection"]
    assert report["runtime_compatibility"] is None
    assert "runtime observation index is not readable" in " ".join(report["blockers"])


@pytest.mark.parametrize("formal_phases", [(), ("prefill",), ("prefill", "decode")])
@pytest.mark.parametrize("index_state", ["missing", "missing_file", "directory", "unreadable"])
def test_only_selected_smoke_index_errors_block_readiness(default_collection, monkeypatch, formal_phases, index_state):
    from aisimulate.support import collection_readiness

    request, root, checkpoint_path = default_collection
    monkeypatch.setattr(
        collection_readiness,
        "_cell_report",
        lambda plan, cell, campaign, entry: {"status": "ready", "blockers": [], "cell": cell.to_dict()},
    )
    baseline = collection_readiness.assess_readiness(request, root, checkpoint_path.parent)
    assert baseline["ready_for_full_collection"], baseline["blockers"]
    formal = json.loads(checkpoint_path.read_text())
    campaign = root / "fpm-artifacts" / formal["plan_sha256"][:16]
    smoke = campaign / "smoke"
    shutil.copytree(campaign, smoke, ignore=shutil.ignore_patterns("smoke"))
    smoke_checkpoint = copy.deepcopy(formal)
    index = smoke / Path(formal["runtime_observations"]).relative_to(campaign)
    smoke_checkpoint["runtime_observations"] = str(index)
    if index_state == "missing":
        smoke_checkpoint.pop("runtime_observations")
    elif index_state == "missing_file":
        index.unlink()
    elif index_state == "directory":
        smoke_checkpoint["runtime_observations"] = str(index.parent)
    else:
        read_bytes = Path.read_bytes

        def denied_smoke_index(path):
            if path.resolve() == index.resolve():
                raise PermissionError("smoke runtime observation index is not readable")
            return read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", denied_smoke_index)
    _write(checkpoint_path.with_name("fpm_forward_smoke.json"), smoke_checkpoint)
    cells_by_phase = {item["cell"]["workload_kind"]: item["cell"]["cell_id"] for item in baseline["selected_cells"]}
    formal["cells"] = {cells_by_phase[phase]: formal["cells"][cells_by_phase[phase]] for phase in formal_phases}
    _write(checkpoint_path, formal)

    report = collection_readiness.assess_readiness(request, root, checkpoint_path.parent)

    assert report["campaigns"][0]["kind"] == "smoke"
    assert report["campaigns"][0]["errors"]
    assert all("runtime memory:" in error for error in report["campaigns"][0]["errors"])
    assert report["campaigns"][1]["errors"] == []
    assert all(len(item["cells"]) == 2 for item in report["campaigns"])
    assert {item["cell"]["workload_kind"]: item["campaign"] for item in report["selected_cells"]} == {
        phase: "formal" if phase in formal_phases else "smoke" for phase in ("prefill", "decode")
    }
    if len(formal_phases) == 2:
        assert report["ready_for_full_collection"], report["blockers"]
        assert report["blockers"] == []
        assert report["runtime_compatibility"]["status"] == "complete"
        assert set(report["runtime_compatibility"]["phases"]) == {"prefill", "decode"}
    else:
        assert not report["ready_for_full_collection"]
        assert report["runtime_compatibility"] is None
        assert any("smoke: runtime memory:" in reason for reason in report["blockers"])


@pytest.mark.parametrize("formal_has_cells", [False, True])
def test_formal_runtime_evidence_supersedes_only_attempted_smoke_cells(default_collection, formal_has_cells):
    from aisimulate.support.collection_readiness import assess_readiness

    request, root, checkpoint_path = default_collection
    formal = json.loads(checkpoint_path.read_text())
    campaign = root / "fpm-artifacts" / formal["plan_sha256"][:16]
    smoke = campaign / "smoke"
    shutil.copytree(campaign, smoke, ignore=shutil.ignore_patterns("smoke"))
    smoke_checkpoint = copy.deepcopy(formal)
    smoke_checkpoint["runtime_observations"] = str(smoke / Path(formal["runtime_observations"]).relative_to(campaign))
    _write(checkpoint_path.with_name("fpm_forward_smoke.json"), smoke_checkpoint)
    checkpoint_path.unlink()
    smoke_report = assess_readiness(request, root, checkpoint_path.parent)
    assert smoke_report["runtime_compatibility"]["status"] == "complete"
    assert {item["campaign"] for item in smoke_report["selected_cells"]} == {"smoke"}

    formal.pop("runtime_observations")
    if not formal_has_cells:
        formal["cells"] = {}
    _write(checkpoint_path, formal)
    report = assess_readiness(request, root, checkpoint_path.parent)

    assert not report["ready_for_full_collection"]
    if formal_has_cells:
        assert report["runtime_compatibility"] is None
        assert "collector checkpoint records no runtime observation index" in " ".join(report["blockers"])
    else:
        assert report["runtime_compatibility"]["status"] == "complete"
        assert not any("runtime memory:" in reason for reason in report["blockers"])
    assert {item["campaign"] for item in report["selected_cells"]} == {"formal" if formal_has_cells else "smoke"}
    assert all(len(campaign["cells"]) == 2 for campaign in report["campaigns"])
    assert {item["cell"]["workload_kind"] for item in report["selected_cells"]} == {"prefill", "decode"}


@pytest.mark.parametrize("path_kind", ["relative", "symlinked_directory"])
def test_explicit_bindings_resolve_checkpoint_index_paths(split_collection, tmp_path, monkeypatch, path_kind):
    request, bindings = split_collection
    _review(request, bindings)
    for binding in bindings:
        checkpoint_path = Path(binding["checkpoint"]["path"])
        checkpoint = json.loads(checkpoint_path.read_text())
        if path_kind == "relative":
            checkpoint["runtime_observations"] = str(
                Path(checkpoint["runtime_observations"]).relative_to(checkpoint_path.parent)
            )
            _write(checkpoint_path, checkpoint)
            binding["checkpoint"] = _ref(checkpoint_path)
        else:
            alias = tmp_path / (binding["phase"] + "-alias")
            alias.symlink_to(checkpoint_path.parent, target_is_directory=True)
            for field in ("plan", "checkpoint", "observations"):
                original = Path(binding[field]["path"])
                binding[field]["path"] = str(alias / original.relative_to(checkpoint_path.parent))
    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = runtime.verify_collection_runtime_bindings(request, bindings)

    assert result["status"] == "complete"
    assert set(result["phases"]) == {"prefill", "decode"}


@pytest.mark.parametrize("change", ["mismatch", "escape", "symlink_escape", "symlink_file", "hash"])
def test_runtime_index_path_resolution_preserves_binding_checks(split_collection, tmp_path, change):
    request, bindings = split_collection
    _review(request, bindings)
    binding = bindings[0]
    checkpoint_path = Path(binding["checkpoint"]["path"])
    checkpoint = json.loads(checkpoint_path.read_text())
    index = Path(binding["observations"]["path"])
    if change == "mismatch":
        another = index.with_name("another-index.json")
        another.write_bytes(index.read_bytes())
        checkpoint["runtime_observations"] = str(another)
        expected = "runtime checkpoint does not match"
    elif change in {"escape", "symlink_escape"}:
        outside = tmp_path / "outside-collection"
        outside.mkdir()
        (outside / index.name).write_bytes(index.read_bytes())
        if change == "symlink_escape":
            alias = checkpoint_path.parent / "escaped-observations"
            alias.symlink_to(outside, target_is_directory=True)
            outside = alias
        binding["observations"] = _ref(outside / index.name)
        checkpoint["runtime_observations"] = str(outside / index.name)
        expected = "index escapes the frozen collection directory"
    elif change == "symlink_file":
        alias = index.with_name("symlink-index.json")
        alias.symlink_to(index)
        checkpoint["runtime_observations"] = str(alias)
        expected = "runtime observation index must be a regular file"
    else:
        binding["observations"]["sha256"] = "0" * 64
        expected = "bound artifact hash mismatch"
    _write(checkpoint_path, checkpoint)
    binding["checkpoint"] = _ref(checkpoint_path)

    with pytest.raises(ValueError, match=expected):
        runtime.verify_collection_runtime_bindings(request, bindings)
