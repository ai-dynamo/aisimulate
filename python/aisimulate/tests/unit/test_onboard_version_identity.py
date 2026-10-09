# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic public-workflow checks for declared labels and deferred detection.

The model configuration, resource bounds and 2/3 ms timing rows are fixtures,
not measurements or evidence of GPU/runtime compatibility.
"""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from aisimulate import main as cli
from aisimulate.support.plan import check_plan, create_plan, request_id
from aisimulate.support.schema import FPMDeployment, SupportRequest

from .test_onboard_model_config import _args, _files

pytestmark = pytest.mark.unit


@pytest.fixture
def forbid_host_version_detection(monkeypatch):
    """The CLI host's installed packages are not the collection runtime."""
    original = importlib.metadata.version

    def version(distribution):
        if distribution in {"vllm", "ai-dynamo", "dynamo"}:
            pytest.fail(f"host package {distribution} must not supply collection version identity")
        return original(distribution)

    monkeypatch.setattr(importlib.metadata, "version", version)


@pytest.mark.parametrize("label", ["my-vllm-patch-3", "nightly+cuda13.commit-abc123"])
def test_custom_backend_label_survives_public_init_plan_and_preview(tmp_path, forbid_host_version_detection, label):
    source, resources = _files(tmp_path)
    draft_path = tmp_path / "draft.yaml"
    assert cli.main(_args(draft_path, source, resources, framework_version=label)) == 0
    request = SupportRequest.from_yaml(draft_path)
    assert request.identity.framework_version == label
    assert request.profile_deployment().backend_version == label

    root = tmp_path / "campaign"
    assert cli.main(["onboard", "plan", "-c", str(draft_path), "--output-dir", str(root)]) == 0
    saved = SupportRequest.from_yaml(root / "request.yaml")
    assert request_id(saved) == request_id(request)
    check_plan(saved, root)
    for relative in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        config = yaml.safe_load((root / relative).read_text())
        assert config["engine"]["backend_version"] == label
        assert {item["backend_version"] for item in config["engine"]["fpm_profile"]["deployments"]} == {label}

    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert (
        cli.main(
            [
                "onboard",
                "collect-fpm",
                "-c",
                str(root / "request.yaml"),
                "--output-dir",
                str(root),
                "--dynamo-version",
                "custom-dynamo-build-2026",
                "--image",
                "registry.example.invalid/benchmark:fixture",
            ]
        )
        == 0
    )
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_omitted_backend_version_stays_pending_without_executable_simulation_configs(
    tmp_path, forbid_host_version_detection
):
    source, resources = _files(tmp_path)
    draft_path = tmp_path / "draft.yaml"
    assert cli.main(_args(draft_path, source, resources, framework_version=None)) == 0
    draft = SupportRequest.from_yaml(draft_path)
    assert draft.identity.framework_version is None

    root = tmp_path / "campaign"
    assert cli.main(["onboard", "plan", "-c", str(draft_path), "--output-dir", str(root)]) == 0
    pending = SupportRequest.from_yaml(root / "request.yaml")
    assert pending.identity.framework_version is None
    check_plan(pending, root)
    assert not (root / "predict/pilot.yaml").exists()
    assert not (root / "recommend/pilot.yaml").exists()
    assert not (root / "fpm-checkpoint").exists()
    assert not (root / "fpm-artifacts").exists()
    commands = json.loads((root / "commands.json").read_text())
    assert not commands.get("predict")
    assert not commands.get("recommend")

    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert (
        cli.main(
            [
                "onboard",
                "collect-fpm",
                "-c",
                str(root / "request.yaml"),
                "--output-dir",
                str(root),
                "--image",
                "registry.example.invalid/benchmark:fixture",
            ]
        )
        == 0
    )
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_custom_label_external_fpm_data_queries_through_native_public_api(tmp_path, monkeypatch):
    """An exact custom-label query needs neither a packaged slot nor semver."""
    import aisimulate_core
    from aisimulate_core.sdk import RustForwardPassPerfModel

    from .sdk.test_fpm_profile import _write_external_profile_pair

    source, resources = _files(tmp_path)
    label = "my-vllm-patch-3"
    draft_path = tmp_path / "draft.yaml"
    assert cli.main(_args(draft_path, source, resources, framework_version=label)) == 0
    profile = SupportRequest.from_yaml(draft_path).fpm_profile.model_dump(mode="json")
    profile["model"] = str(tmp_path)
    deployment = profile["deployments"][0]

    systems = tmp_path / "systems"
    systems.mkdir()
    system = deployment["system"]
    packaged = Path(aisimulate_core.__file__).parent / "systems" / f"{system}.yaml"
    (systems / packaged.name).write_bytes(packaged.read_bytes())
    destination = systems / "data" / system / "vllm" / label
    destination.mkdir(parents=True)
    pair = _write_external_profile_pair(profile, destination)
    parquet = destination / "fpm_forward_perf.parquet"
    pair.rename(parquet)
    pair.with_suffix(".metadata.json").rename(parquet.with_suffix(".metadata.json"))
    monkeypatch.delenv("AIC_ALLOW_UNLISTED_VERSIONS", raising=False)

    config = {
        "model": profile["model"],
        "system": system,
        "backend": "vllm",
        "backend_version": label,
        "worker_type": "aggregated",
        "tp": deployment["tp"],
        "attention_dp": deployment["dp"],
        "moe_tp_size": deployment["moe_tp"],
        "moe_ep_size": deployment["moe_ep"],
        "estimation_mode": "fpm_interpolation",
        "fallback_policy": "deny",
        "estimator_config": {"fpm_interpolation": {"method": "direct"}},
        "fpm_profile": profile,
        "systems_paths": [str(systems)],
    }
    model = RustForwardPassPerfModel.best_available(config)
    queries = [
        ({"scheduled_requests": {"num_prefill_requests": 1, "sum_prefill_tokens": 1}}, 2.0),
        ({"scheduled_requests": {"num_decode_requests": 1, "sum_decode_kv_tokens": 1}}, 3.0),
    ]
    for query, expected in queries:
        assert model.estimate_forward_pass_time_ms(query) == pytest.approx(expected)
    saved = json.loads(json.dumps(model.diagnostics()["provenance"]["config"]))
    assert saved["backend_version"] == label
    assert saved["fpm_profile"]["deployments"][0]["backend_version"] == label
    reloaded = RustForwardPassPerfModel.best_available(saved)
    for query, expected in queries:
        assert reloaded.estimate_forward_pass_time_ms(query) == pytest.approx(expected)


def test_unresolved_registration_profile_cannot_construct_a_simulation_model(tmp_path):
    from aisimulate_core.sdk import RustForwardPassPerfModel

    source, resources = _files(tmp_path)
    draft_path = tmp_path / "draft.yaml"
    assert cli.main(_args(draft_path, source, resources, framework_version=None)) == 0
    request = SupportRequest.from_yaml(draft_path)
    profile = request.fpm_profile.model_dump(mode="json")
    assert profile["deployments"][0]["backend_version"] is None
    config = {
        "model": profile["model"],
        "system": request.identity.gpu,
        "backend": "vllm",
        "backend_version": None,
        "worker_type": "aggregated",
        "tp": 1,
        "estimation_mode": "fpm_interpolation",
        "fallback_policy": "deny",
        "fpm_profile": profile,
    }
    with pytest.raises(ValueError, match="literal.*backend_version"):
        RustForwardPassPerfModel.best_available(config)


@pytest.mark.parametrize("label", [None, "my-vllm-patch-3"])
def test_runtime_binding_preserves_source_resolves_identity_and_reuses_verified_child(tmp_path, monkeypatch, label):
    from aisimulate.support import versioning

    source, resources = _files(tmp_path)
    draft_path = tmp_path / "draft.yaml"
    assert cli.main(_args(draft_path, source, resources, framework_version=label)) == 0
    request = SupportRequest.from_yaml(draft_path)
    root = tmp_path / "campaign"
    create_plan(request, root)
    original = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    original_draft = draft_path.read_bytes()
    deployment = FPMDeployment(
        image="registry.example.invalid/benchmark:fixture",
        dynamo_version="user-dynamo-metadata",
    )
    probes = []

    def detect(*args, **kwargs):
        probes.append((args, kwargs))
        return {"backend_version": "0.28.0", "dynamo_version": "1.5.0+fixture"}

    monkeypatch.setattr(versioning, "detect_runtime_versions", detect)
    resolved, child = versioning.resolve_runtime_request(request, deployment, root)
    assert child == root / "runtime-resolved"
    assert resolved.identity.framework_version == (label or "0.28.0")
    assert resolved.identity.runtime_framework_version == "0.28.0"
    assert resolved.profile_deployment().backend_version == (label or "0.28.0")
    assert draft_path.read_bytes() == original_draft
    assert all(path.read_bytes() == content for path, content in original.items())
    check_plan(resolved, child)
    check_plan(request, root)
    for relative in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        config = yaml.safe_load((child / relative).read_text())
        assert config["engine"]["backend_version"] == (label or "0.28.0")
    assert len(probes) == 1

    reused, reused_root = versioning.resolve_runtime_request(request, deployment, root)
    assert request_id(reused) == request_id(resolved)
    assert reused_root == child
    assert len(probes) == 1
    changed = deployment.model_copy(update={"image": "registry.example.invalid/benchmark:other"})
    with pytest.raises(ValueError, match="(?i)(deployment|image|binding|different)"):
        versioning.resolve_runtime_request(request, changed, root)
    assert len(probes) == 1
    for addressed_root in (root, child):
        hydrated = versioning.runtime_deployment(None, addressed_root)
        assert hydrated.image == deployment.image
        metadata_only = versioning.runtime_deployment(
            FPMDeployment(dynamo_version="another-dynamo-description"), addressed_root
        )
        assert metadata_only.image == deployment.image
        assert metadata_only.dynamo_version == "another-dynamo-description"


def test_failed_target_version_probe_leaves_unresolved_plan_retryable(tmp_path, monkeypatch):
    from aisimulate.support import versioning

    source, resources = _files(tmp_path)
    draft_path = tmp_path / "draft.yaml"
    assert cli.main(_args(draft_path, source, resources, framework_version=None)) == 0
    request = SupportRequest.from_yaml(draft_path)
    root = tmp_path / "campaign"
    create_plan(request, root)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    def detect(*args, **kwargs):
        raise ValueError("target runtime has no vllm package metadata")

    monkeypatch.setattr(versioning, "detect_runtime_versions", detect)
    with pytest.raises(ValueError, match="target runtime has no vllm package metadata"):
        versioning.resolve_runtime_request(
            request, FPMDeployment(image="registry.example.invalid/benchmark:fixture"), root
        )
    assert not (root / "runtime-resolved").exists()
    assert all(path.read_bytes() == content for path, content in before.items())
    check_plan(request, root)


@pytest.mark.parametrize("label", ["../outside", "a/b", "a\\b", ".", "..", "two words"])
def test_version_label_cannot_change_publication_path(tmp_path, label):
    source, resources = _files(tmp_path)
    draft_path = tmp_path / "new" / "draft.yaml"
    with pytest.raises(SystemExit) as error:
        cli.main(_args(draft_path, source, resources, framework_version=label))
    assert error.value.code == 2
    assert not draft_path.exists()


@pytest.mark.parametrize("ambiguous", [False, True])
def test_supplied_profile_version_is_preserved_or_requires_explicit_selection(
    tmp_path, forbid_host_version_detection, ambiguous
):
    source, resources = _files(tmp_path)
    first_path = tmp_path / "first.yaml"
    label = "my-vllm-patch-3"
    assert cli.main(_args(first_path, source, resources, framework_version=label)) == 0
    first = SupportRequest.from_yaml(first_path)
    profile = first.fpm_profile.model_dump(mode="json")
    if ambiguous:
        second = deepcopy(profile["deployments"][0])
        second["backend_version"] = "my-vllm-patch-4"
        profile["deployments"].append(second)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    output = tmp_path / "from-profile.yaml"
    args = _args(output, source, framework_version=None)
    index = args.index("--model-config")
    del args[index : index + 2]
    args.extend(
        [
            "--fpm-profile",
            str(profile_path),
            "--model",
            first.identity.model,
            "--model-kind",
            first.identity.model_kind,
        ]
    )
    if ambiguous:
        with pytest.raises(SystemExit) as error:
            cli.main(args)
        assert error.value.code == 2
        assert not output.exists()
        # Selecting an existing identity resolves ambiguity without relabeling.
        assert cli.main([*args, "--framework-version", label]) == 0
    else:
        assert cli.main(args) == 0
    saved = SupportRequest.from_yaml(output)
    assert saved.identity.framework_version == label
    assert saved.fpm_profile.model_dump(mode="json") == profile


@pytest.mark.parametrize("executor", ["kubernetes", "slurm"])
def test_metadata_probe_targets_selected_container_without_gpu_requests(tmp_path, monkeypatch, executor):
    from aisimulate.support import versioning

    deployment = FPMDeployment(
        executor=executor,
        image="registry.example.invalid/benchmark:fixture",
        **(
            {"namespace": "synthetic-probe", "image_pull_secret": "synthetic-secret"}
            if executor == "kubernetes"
            else {}
        ),
    )
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-allocation")
    calls = []
    result = {"backend_version": "0.28.0+runtime", "dynamo_version": "custom-dynamo"}

    def run(command, *, payload=None):
        calls.append((command, payload))
        if command[0] == "srun" or "logs" in command:
            return "AISIMULATE_RUNTIME_VERSION=" + json.dumps(result) + "\n"
        if "get" in command:
            return json.dumps({"status": {"containerStatuses": [{"state": {"terminated": {"exitCode": 0}}}]}})
        return ""

    monkeypatch.setattr(versioning, "_run", run)
    assert versioning.detect_runtime_versions(deployment) == result
    if executor == "slurm":
        assert len(calls) == 1
        assert f"--container-image={deployment.image}" in calls[0][0]
        assert "--gpus=0" in calls[0][0]
    else:
        manifest = json.loads(next(payload for _, payload in calls if payload is not None))
        assert manifest["metadata"]["namespace"] == deployment.namespace
        assert manifest["spec"]["imagePullSecrets"] == [{"name": deployment.image_pull_secret}]
        container = manifest["spec"]["containers"][0]
        assert container["image"] == deployment.image
        assert "nvidia.com/gpu" not in container["resources"].get("requests", {})
        assert any("delete" in command for command, _ in calls)


@pytest.mark.parametrize(
    "output",
    [
        "vllm package metadata unavailable\n",
        "AISIMULATE_RUNTIME_VERSION={}\n",
        'AISIMULATE_RUNTIME_VERSION={"backend_version": ""}\n',
        'AISIMULATE_RUNTIME_VERSION={"backend_version": "0.28.0"}\n' * 2,
    ],
)
def test_probe_rejects_missing_or_ambiguous_metadata_instead_of_using_host_version(
    monkeypatch, forbid_host_version_detection, output
):
    from aisimulate.support import versioning

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-allocation")
    monkeypatch.setattr(versioning, "_run", lambda *_args, **_kwargs: output)
    with pytest.raises(ValueError, match="(?i)(runtime version|backend version)"):
        versioning.detect_runtime_versions(
            FPMDeployment(executor="slurm", image="registry.example.invalid/benchmark:fixture")
        )


@pytest.mark.parametrize("conflicting", ["--execute", "--smoke", "--resume"])
def test_readiness_rejects_execution_flags_before_any_runtime_probe(tmp_path, monkeypatch, conflicting):
    from aisimulate.support import versioning

    source, resources = _files(tmp_path)
    draft = tmp_path / "draft.yaml"
    assert cli.main(_args(draft, source, resources, framework_version=None)) == 0
    request = SupportRequest.from_yaml(draft)
    root = tmp_path / "campaign"
    create_plan(request, root)
    monkeypatch.setattr(
        versioning,
        "detect_runtime_versions",
        lambda *_args, **_kwargs: pytest.fail("read-only readiness flags must be validated before probing"),
    )
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "onboard",
                "collect-fpm",
                "-c",
                str(root / "request.yaml"),
                "--output-dir",
                str(root),
                "--image",
                "registry.example.invalid/benchmark:fixture",
                "--check-readiness",
                conflicting,
            ]
        )
    assert error.value.code == 2
    assert not (root / "runtime-resolved").exists()


def test_public_collect_smoke_readiness_and_resume_route_to_same_resolved_campaign(tmp_path, monkeypatch):
    """Run CLI orchestration with a synthetic worker and checkpoint, no cluster."""
    from collector.fpm_forward import cli as collector_cli
    from collector.fpm_forward import entry

    from aisimulate.support import collection_readiness, fpm, versioning

    source, resources = _files(tmp_path)
    draft = tmp_path / "draft.yaml"
    assert cli.main(_args(draft, source, resources, framework_version=None)) == 0
    request = SupportRequest.from_yaml(draft)
    root = tmp_path / "campaign"
    create_plan(request, root)
    original_request = (root / "request.yaml").read_bytes()
    resolved_root = root / "runtime-resolved"
    image = "registry.example.invalid/benchmark:fixture"
    probes, launches, inspections = [], [], []

    def detect(deployment):
        probes.append(deployment)
        return {"backend_version": "0.28.0", "dynamo_version": "1.5.0+fixture"}

    frozen = SimpleNamespace(sha256="a" * 64)

    def resolve(command):
        args = collector_cli._parser().parse_args(command[3:])
        return args, (frozen, {})

    def run(args, resolved):
        launches.append(args)
        assert resolved[0] is frozen
        assert Path(args.checkpoint_dir) == resolved_root / "fpm-checkpoint"
        assert Path(args.fpm_artifact_root) == resolved_root / "fpm-artifacts"
        assert Path(args.fpm_database_root) == resolved_root / "systems/data"
        assert args.fpm_runtime_backend_version == "0.28.0"
        assert any(image in value for value in args.generator_set)
        checkpoint = Path(args.checkpoint_dir)
        checkpoint.mkdir(exist_ok=True)
        (checkpoint / "fpm_forward_smoke.json").write_text(
            json.dumps({"schema": "synthetic-smoke", "plan_sha256": frozen.sha256, "cells": {}})
        )
        return []

    def assess(selected, output, checkpoint, **kwargs):
        inspections.append((selected, output, checkpoint))
        assert selected.identity.framework_version == selected.identity.runtime_framework_version == "0.28.0"
        assert output == resolved_root
        assert checkpoint == resolved_root / "fpm-checkpoint"
        return {"status": "ready", "ready_for_full_collection": True}

    monkeypatch.setattr(versioning, "detect_runtime_versions", detect)
    monkeypatch.setattr(fpm, "_resolve_execution", resolve)
    monkeypatch.setattr(entry, "run_resolved", run)
    monkeypatch.setattr(collection_readiness, "assess_readiness", assess)
    base = ["onboard", "collect-fpm", "-c", str(root / "request.yaml"), "--output-dir", str(root)]
    assert cli.main([*base, "--execute", "--smoke", "--image", image]) == 0
    assert cli.main([*base, "--check-readiness"]) == 0
    assert cli.main([*base, "--execute", "--smoke", "--resume"]) == 0
    child_command = [
        "onboard",
        "collect-fpm",
        "-c",
        str(resolved_root / "request.yaml"),
        "--output-dir",
        str(resolved_root),
    ]
    assert cli.main([*child_command, "--execute", "--smoke", "--resume"]) == 0
    assert len(probes) == 1
    assert len(launches) == 3
    assert len(inspections) == 4
    assert launches[0].resume is False
    assert launches[1].resume is launches[2].resume is True
    assert (root / "request.yaml").read_bytes() == original_request
    assert not (root / "fpm-checkpoint").exists()


def test_probe_subprocess_missing_package_exits_with_runtime_error(monkeypatch, forbid_host_version_detection):
    """Execute the real metadata script in a child with missing-package metadata."""
    from aisimulate.support import versioning

    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-allocation")
    original_run = subprocess.run
    observed_commands = []
    synthetic_runtime = (
        "import importlib.metadata as m\n"
        "def unavailable(name):\n"
        "    raise m.PackageNotFoundError(name)\n"
        "m.version = unavailable\n"
    )

    def run(command, **kwargs):
        observed_commands.append(command)
        assert command[0] == "srun"
        return original_run([sys.executable, "-c", synthetic_runtime + command[-1]], **kwargs)

    monkeypatch.setattr(versioning.subprocess, "run", run)
    with pytest.raises(ValueError, match="runtime version detection failed.*", check=lambda exc: "vllm" in str(exc)):
        versioning.detect_runtime_versions(
            FPMDeployment(executor="slurm", image="registry.example.invalid/benchmark:missing-vllm")
        )
    assert len(observed_commands) == 1


def test_custom_label_formal_publication_and_memory_finalization_use_actual_runtime(tmp_path):
    """Independent native timing/cache fixtures pass real producer and finalizer."""
    import pyarrow.parquet as pq

    from .test_onboard_finalization import build_completed_collection

    label = "my-vllm-patch-3"
    request, root = build_completed_collection(tmp_path, extra_init_args=("--framework-version", label))
    assert request.identity.framework_version == label
    assert request.identity.runtime_framework_version == "0.27.0"
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    parquet = root / "systems/data/h200_sxm/vllm" / label / "fpm_forward_perf.parquet"
    rows = pq.read_table(parquet).to_pylist()
    assert {row["backend_version"] for row in rows} == {label}
    assert {row["runtime_backend_version"] for row in rows} == {"0.27.0"}

    target = tmp_path / "finalized"
    assert (
        cli.main(
            [
                "onboard",
                "finalize",
                "--config",
                str(root / "request.yaml"),
                "--output-dir",
                str(root),
                "--resolved-output-dir",
                str(target),
            ]
        )
        == 0
    )
    resolved = SupportRequest.from_yaml(target / "request.yaml")
    check_plan(resolved, target)
    assert resolved.identity.framework_version == label
    assert resolved.identity.runtime_framework_version == "0.27.0"
    resource = resolved.profile_deployment().resources
    assert resource.memory_source == "runtime"
    assert resource.runtime_memory.kv_cache_bytes == 899 * 1024
    for relative in ("predict/pilot.yaml", "recommend/pilot.yaml"):
        config = yaml.safe_load((target / relative).read_text())
        assert config["engine"]["backend_version"] == label
    assert all(path.read_bytes() == content for path, content in before.items())


def test_runtime_probe_does_not_launch_a_configuration_whose_metadata_probe_failed(tmp_path, monkeypatch):
    from collector.fpm_forward import runtime_probe

    from aisimulate.support import versioning

    from .test_onboard_runtime import _campaign

    checkpoint, index, _ = _campaign(tmp_path)

    def missing_package(_deployment):
        raise ValueError("target image has no vllm package metadata")

    monkeypatch.setattr(versioning, "detect_runtime_versions", missing_package)
    monkeypatch.setattr(
        runtime_probe,
        "probe_runtime",
        lambda *_args, **_kwargs: pytest.fail("failed version preflight must not launch runtime workers"),
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
                str(tmp_path / "failed-probe"),
                "--execute",
            ]
        )
        == 1
    )


@pytest.mark.parametrize("runtime_pin", [None, "0.28.0", "0.27.0"])
def test_custom_label_runtime_observation_import_keeps_actual_adapter_version(tmp_path, monkeypatch, runtime_pin):
    from . import test_onboard_runtime as workflow

    original = workflow.normalize_probe_launch
    label = "my-vllm-patch-3"

    def normalize(value):
        launch = original(value)
        launch["identity"]["framework_version"] = label
        if runtime_pin is not None:
            launch["identity"]["runtime_framework_version"] = runtime_pin
        return launch

    # Adapt the independent observation fixture before it writes every launch,
    # identity record and digest, keeping the real observer bundle at 0.28.0.
    monkeypatch.setattr(workflow, "normalize_probe_launch", normalize)
    checkpoint, index, launch = workflow._campaign(tmp_path)
    assert launch["identity"]["framework_version"] == label
    before = {path: path.read_bytes() for path in index.parent.rglob("*") if path.is_file()}
    output = tmp_path / "imported"
    status = workflow._import(checkpoint, index, output, configurations=["tp2"])
    assert all(path.read_bytes() == content for path, content in before.items())
    if runtime_pin == "0.27.0":
        assert status == 1
        report = json.loads((output / "import.json").read_text())
        assert "instrumentation runtime version differs" in str(report)
        return
    assert status == 0
    state = workflow._load(checkpoint)
    imported = SupportRequest.model_validate(state.configurations["tp2"].draft_request)
    assert imported.identity.framework_version == imported.profile_deployment().backend_version == label
    assert imported.identity.runtime_framework_version == runtime_pin
    assert imported.profile_deployment().resources.memory_source == "runtime"
    assert imported.profile_deployment().resources.runtime_memory.kv_cache_bytes == 93 * 128
    assert all(path.read_bytes() == content for path, content in before.items())


def test_runtime_probe_resolves_omitted_backend_label_before_launch_validation(tmp_path, monkeypatch):
    from collector.fpm_forward import runtime_probe

    from aisimulate.support import versioning

    from . import test_onboard_runtime as workflow

    checkpoint, index, _ = workflow._campaign(tmp_path)
    state = workflow._load(checkpoint)
    draft = deepcopy(state.configurations["tp2"].draft_request)
    draft["identity"]["framework_version"] = None
    workflow.save_checkpoint(
        checkpoint,
        patch={"configurations": {"tp2": {"draft_request": draft}}},
        expected_revision=state.revision,
        accept=[],
    )
    detections = []

    def detect(deployment):
        detections.append(deployment)
        return {"backend_version": "0.28.0"}

    def probe(configurations, **kwargs):
        assert configurations["tp2"]["identity"]["framework_version"] == "0.28.0"
        return {"status": "completed", "configurations": {"tp2": {"status": "completed"}}}

    monkeypatch.setattr(versioning, "detect_runtime_versions", detect)
    monkeypatch.setattr(runtime_probe, "probe_runtime", probe)
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
                str(tmp_path / "probe-with-detected-version"),
                "--execute",
            ]
        )
        == 0
    )
    assert len(detections) == 1


@pytest.mark.parametrize("preview", [False, True])
def test_grouped_topology_with_pending_version_defers_native_memory_estimate(tmp_path, monkeypatch, preview):
    from aisimulate_core.sdk.rust_engine_step import RustForwardPassPerfModel

    from . import test_onboard_topology as workflow

    source, resources = workflow._inputs(
        tmp_path,
        {**workflow._SMALL, "sliding_window": 128},
        {
            **workflow._PRECISION,
            "cache_block_sizes": {"sliding_attention": 16},
            "max_num_tokens": 256,
            "max_batch_size": 8,
        },
    )
    monkeypatch.setattr(
        RustForwardPassPerfModel,
        "estimate_cache_budget",
        lambda *_args, **_kwargs: pytest.fail("pending version cannot construct native memory model"),
    )
    output = tmp_path / "new" / "request.yaml"
    command = workflow._args(source, resources, output, model=str(tmp_path), framework_version=None)
    if not preview:
        # A deferred estimate cannot rank candidates; choose one explicitly.
        command += ["--tensor-parallel", "1"]
    assert cli.main(command + (["--suggest-parallel"] if preview else [])) == 0
    if not preview:
        request = SupportRequest.from_yaml(output)
        assert request.identity.framework_version is None
        assert request.profile_deployment().resources.cache_layout == "grouped"
        root = tmp_path / "pending-plan"
        create_plan(request, root)
        check_plan(request, root)
        assert not (root / "predict/pilot.yaml").exists()


def test_detected_version_reopens_checkpoint_with_a_version_pending_profile(tmp_path, monkeypatch):
    from collector.fpm_forward import runtime_probe

    from aisimulate.support import runtime, versioning
    from aisimulate.support.checkpoint import _load, save_checkpoint

    source, resources = _files(tmp_path)
    draft_path = tmp_path / "draft.yaml"
    assert cli.main(_args(draft_path, source, resources, framework_version=None)) == 0
    request = SupportRequest.from_yaml(draft_path)
    checkpoint = tmp_path / "checkpoint.json"
    save_checkpoint(
        checkpoint,
        patch={
            "inputs": {"model_config": str(source)},
            "configurations": {
                "tp1": {
                    "draft_request": request.model_dump(mode="json"),
                    "inputs": {"collection_deployment": {"image": "registry.example.invalid/runtime:fixture"}},
                }
            },
        },
        expected_revision=None,
        accept=[],
    )
    monkeypatch.setattr(versioning, "detect_runtime_versions", lambda *_: {"backend_version": "0.28.0"})
    monkeypatch.setattr(
        runtime_probe,
        "probe_runtime",
        lambda *_args, **_kwargs: {"status": "completed", "configurations": {"tp1": {"status": "completed"}}},
    )
    assert (
        cli.main(
            [
                "onboard",
                "probe-runtime",
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                str(tmp_path / "runtime-probe"),
                "--execute",
            ]
        )
        == 0
    )
    # Import and resume reconstruct this same launch from the saved checkpoint.
    _, launch, _ = runtime.checkpoint_launch(_load(checkpoint), "tp1", checkpoint)
    assert launch["identity"]["framework_version"] == "0.28.0"


@pytest.mark.parametrize("existing", ["runtime_pin", "probe_manifest", "checkpoint"])
@pytest.mark.parametrize("execute", [False, True])
def test_unresolved_version_execute_does_not_report_success(tmp_path, monkeypatch, existing, execute, capsys):
    from aisimulate.support import runtime, versioning

    source, resources = _files(tmp_path)
    draft = tmp_path / "draft.yaml"
    assert cli.main(_args(draft, source, resources, framework_version=None)) == 0
    payload = yaml.safe_load(draft.read_text())
    if existing == "runtime_pin":
        payload["identity"]["runtime_framework_version"] = "0.28.0"
    request = SupportRequest.model_validate(payload)
    root = tmp_path / "campaign"
    create_plan(request, root)
    if existing == "probe_manifest":
        monkeypatch.setattr(runtime, "runtime_probe_manifest", lambda _request: {"synthetic": True})
    elif existing == "checkpoint":
        checkpoint = root / "fpm-checkpoint/fpm_forward.json"
        checkpoint.parent.mkdir()
        checkpoint.write_text("{}")
    monkeypatch.setattr(
        versioning, "detect_runtime_versions", lambda *_: pytest.fail("an existing runtime binding skips detection")
    )
    capsys.readouterr()
    command = ["onboard", "collect-fpm", "-c", str(root / "request.yaml"), "--output-dir", str(root)]
    assert cli.main([*command, *(["--execute"] if execute else [])]) == int(execute)
    assert json.loads(capsys.readouterr().out)["status"] == "pending_runtime_version"


@pytest.mark.parametrize(
    "status,diagnostic",
    [
        (
            {
                "containerStatuses": [
                    {
                        "state": {
                            "waiting": {"reason": "ImagePullBackOff", "message": "manifest unknown for image:fixture"}
                        }
                    }
                ]
            },
            "ImagePullBackOff: manifest unknown for image:fixture",
        ),
        (
            {
                "phase": "Pending",
                "conditions": [
                    {
                        "type": "PodScheduled",
                        "status": "False",
                        "reason": "Unschedulable",
                        "message": "Insufficient cpu",
                    }
                ],
            },
            "Unschedulable: Insufficient cpu",
        ),
        ({}, "synthetic wait timeout"),
        (None, "synthetic wait timeout"),
        ("malformed", "synthetic wait timeout"),
    ],
)
def test_metadata_probe_reports_pod_startup_failure_before_cleanup(monkeypatch, status, diagnostic):
    from aisimulate.support import versioning

    calls = []

    def run(command, *, payload=None):
        calls.append(command)
        if "wait" in command:
            raise ValueError("synthetic wait timeout")
        if "get" in command:
            if status is None:
                raise ValueError("synthetic status retrieval failed")
            return json.dumps({"status": status})
        if "logs" in command:
            pytest.fail("a container that never started has no probe output")
        return ""

    monkeypatch.setattr(versioning, "_run", run)
    with pytest.raises(ValueError) as failure:
        versioning.detect_runtime_versions(FPMDeployment(image="registry.example.invalid/image:fixture"))
    assert diagnostic in str(failure.value)
    assert any("get" in command for command in calls)
    assert "delete" in calls[-1]
