# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch-only probes use real rendering and synthetic executor evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from .test_fpm_profile_collection import no_models_or_timing_data  # noqa: F401

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_models_or_timing_data")]


def _inputs(tmp_path, *, graph_policy="runtime"):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "architectures": ["ExternalMoeForCausalLM"],
                "hidden_size": 128,
                "intermediate_size": 256,
                "moe_intermediate_size": 256,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "num_hidden_layers": 5,
                "max_position_embeddings": 8192,
                "vocab_size": 1024,
                "num_experts": 8,
                "torch_dtype": "bfloat16",
            }
        )
    )
    source = tmp_path / "observer.py"
    marker = tmp_path / "observer-was-imported"
    source.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('never import')\n")
    (tmp_path / "source-notes.txt").write_text("Synthetic source mapping fixture; no live verification.")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "aisimulate-runtime-instrumentation/v1",
                "runtime": {
                    "framework": "vllm",
                    "version": "0.28.0",
                    "source_revision": "a" * 40,
                    "source_files": {"vllm/v1/core/block_pool.py": "b" * 64},
                },
                "files": ["observer.py", "source-notes.txt"],
                "worker_class": "observer.ObservedWorker",
                "scheduler_class": "observer.ObservedScheduler",
                "observation_schema": "aisimulate-runtime-observation/v1",
                "source_notes": "source-notes.txt",
            }
        )
    )
    launch = {
        "identity": {
            "model": "example/external-moe",
            "model_revision": "pinned-model-revision",
            "model_kind": "moe",
            "framework": "vllm",
            "framework_version": "0.28.0",
            "gpu": "gb200",
            "interconnect": "nvlink",
        },
        "topology": {"tp": 4, "pp": 1, "dp": 1, "moe_tp": 1, "moe_ep": 4, "cp": 1},
        "precision": {
            "gemm_quant_mode": "bfloat16",
            "moe_quant_mode": "bfloat16",
            "fmha_quant_mode": "bfloat16",
            "kvcache_quant_mode": "bfloat16",
            "comm_quant_mode": "half",
            "moe_backend": "auto",
            "attention_backend": "auto",
            "enable_wideep": False,
            "enable_eplb": False,
        },
        "collection": {
            "max_model_len": 4096,
            "max_num_batched_tokens": 1024,
            "max_num_seqs": 64,
            "gpu_memory_utilization": 0.85,
            "prefill_cudagraph_policy": graph_policy,
            "max_prefill_cudagraph_size": 512 if graph_policy == "explicit" else None,
        },
        "model_config": {"path": str(config), "sha256": hashlib.sha256(config.read_bytes()).hexdigest()},
        "deployment": {"executor": "slurm", "image": "image@sha256:synthetic", "container_mount": ["/cache:/cache"]},
    }
    return launch, manifest, marker


class _SyntheticExecutor:
    def __init__(self, plan, cell, cell_dir, executions, *, fail_configuration=None):
        self.plan, self.cell, self.cell_dir = plan, cell, cell_dir
        self.executions = executions
        self.fail_configuration = fail_configuration

    def cleanup(self):
        pass

    def apply(self):
        pass

    def wait_ready(self, expected_nodes):
        return ["synthetic-node", *(f"synthetic-node-{index}" for index in range(1, expected_nodes))]

    def stage(self, pods, files):
        self.files = {path.name: path for path in files}
        assert "runtime-instrumentation.zip" in self.files
        assert "runtime-probe-context.json" in self.files

    def prepare_attempt(self, pods, **kwargs):
        self.identity = kwargs

    def execute(self, pods):
        context = json.loads(self.files["runtime-probe-context.json"].read_text())
        self.executions.append((context["configuration"], context["phase"]))
        for pod in pods:
            raw = self.cell_dir / "raw" / pod
            raw.mkdir(parents=True, exist_ok=True)
            (raw / "synthetic-observation.json").write_text(
                json.dumps({**context, "schema_version": "aisimulate-runtime-observation/v1", "synthetic": True})
            )
        if context["configuration"] == self.fail_configuration:
            raise RuntimeError("synthetic runtime failure")

    def collect(self, pods, *, require_benchmark=True):
        pass


@pytest.mark.parametrize("graph_policy", ["runtime", "explicit"])
def test_preview_requires_no_cache_geometry_and_never_imports_bundle(tmp_path, monkeypatch, graph_policy):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, marker = _inputs(tmp_path, graph_policy=graph_policy)
    monkeypatch.setattr(runner, "_cell_runner", lambda *_args: pytest.fail("preview launched an executor"))
    result = runtime_probe.probe_runtime(
        {"arbitrary-label": launch}, instrumentation=manifest, output_dir=tmp_path / "probe"
    )

    assert result["status"] == "preview"
    assert not marker.exists()
    preview = result["configurations"]["arbitrary-label"]
    assert preview["minimum_gpus"] == 4
    assert set(preview["phases"]) == {"prefill", "decode"}
    for phase, info in preview["phases"].items():
        params = json.loads(Path(info["generator_request"]).read_text())
        args = params["params"]["agg"]["extra_cli_args"]
        assert args[args.index("--max-model-len") + 1] == "4096"
        assert args[args.index("--max-num-batched-tokens") + 1] == "1024"
        assert args[args.index("--max-num-seqs") + 1] == "64"
        assert params["params"]["agg"]["kv_cache_free_gpu_memory_fraction"] == 0.85
        assert args[args.index("--revision") + 1] == launch["identity"]["model_revision"]
        assert args.count("--revision") == 1
        assert args[args.index("--worker-cls") + 1] == "observer.ObservedWorker"
        assert "--no-async-scheduling" in args
        batch_flag = "--prefix-max-batch-size-samples" if phase == "prefill" else "--decode-max-batch-size-samples"
        context = json.loads((tmp_path / "probe" / info["launch_manifest"]["path"]).read_text())
        assert int(args[args.index(batch_flag) + 1]) == context["sampling"]["max_batch_size_samples"] == 2
        if phase == "prefill":
            assert ("--compilation-config" in args) == (graph_policy == "explicit")
            assert args[args.index("--prefill-max-new-token-samples") + 1] == "2"
            if graph_policy == "explicit":
                graph = json.loads(args[args.index("--compilation-config") + 1])
                assert graph["max_cudagraph_capture_size"] == 512


@pytest.mark.parametrize("executor", ["kubernetes", "slurm"])
def test_multinode_preview_labels_every_resource(tmp_path, monkeypatch, executor):
    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, marker = _inputs(tmp_path)
    launch["identity"]["gpu"] = "gb300"
    if executor == "kubernetes":
        launch["deployment"] = {"executor": executor, "image": launch["deployment"]["image"]}
    launch["topology"].update(tp=8, dp=1, moe_tp=8, moe_ep=1)
    dep = copy.deepcopy(launch)
    dep["topology"].update(tp=1, dp=8, moe_tp=1, moe_ep=8)
    configurations = {"tp8": launch, "dep8": dep}
    monkeypatch.setattr(runner, "_cell_runner", lambda *_args: pytest.fail("preview launched an executor"))
    output = tmp_path / "probe"

    result = runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output)

    assert result["status"] == "preview", result
    assert not marker.exists()
    bundle = load_instrumentation(manifest)
    for name, facts in configurations.items():
        plan = runtime_probe.build_runtime_probe_plan(name, facts, bundle)
        preview = result["configurations"][name]
        assert preview["minimum_gpus"] == 8
        assert set(preview["phases"]) == {"prefill", "decode"}
        for cell in plan.cells:
            phase = preview["phases"][cell.workload_kind]
            resource = output / phase["resource_manifest"]["path"]
            documents = runner._manifest_documents(resource)
            assert [document["kind"] for document in documents] == ["ComputeDomain", "LeaderWorkerSet"]
            assert runner._expected_nodes(resource) == 2
            expected = {
                "aiconfigurator.nvidia.com/owned-by": "fpm-forward-collector",
                "aiconfigurator.nvidia.com/plan": plan.sha256[:16],
                runner.FPM_CELL_LABEL: cell.cell_id,
            }
            for document in documents:
                assert expected.items() <= document["metadata"]["labels"].items()
            assert documents[0]["spec"]["numNodes"] == 0


@pytest.mark.parametrize("flag", [["--revision", "another-model-revision"], ["--revision=another-model-revision"]])
def test_runtime_revision_cannot_be_replaced_by_backend_policy(tmp_path, flag):
    from dataclasses import replace

    from collector.fpm_forward import runner, runtime_probe
    from collector.fpm_forward.planner import BackendPolicy
    from collector.fpm_forward.runtime_instrumentation import load_instrumentation

    launch, manifest, _marker = _inputs(tmp_path)
    plan = runtime_probe.build_runtime_probe_plan("worker", launch, load_instrumentation(manifest))
    cell = replace(
        plan.cells[0], backend_policy=BackendPolicy("test", {"params": {"agg": {"extra_cli_args": flag}}}, {})
    )
    with pytest.raises(ValueError, match="selected model revision"):
        runner._cell_generator_overrides(plan, cell, runtime_probe.probe_generator_overrides(plan.launch))


def test_execute_covers_each_configuration_and_both_phases_then_resumes(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, marker = _inputs(tmp_path)
    second = copy.deepcopy(launch)
    second["topology"].update(tp=1, dp=4)
    configurations = {"tp4": launch, "dep4": second}
    executions = []
    monkeypatch.setattr(
        runner,
        "_cell_runner",
        lambda plan, cell, _manifest, directory: _SyntheticExecutor(plan, cell, directory, executions),
    )
    output = tmp_path / "probe"
    runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output)
    result = runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output, execute=True)
    assert result["status"] == "completed"
    assert executions == [(label, phase) for label in configurations for phase in ("prefill", "decode")]
    assert not marker.exists()
    index = json.loads((output / "observations.json").read_text())
    original = {
        str(output / artifact["path"]): (output / artifact["path"]).read_bytes()
        for entry in index["configurations"].values()
        for phase in entry["attempts"][0]["phases"].values()
        for artifact in phase["artifacts"]
        if artifact["kind"] == "observation"
    }
    assert len(original) == 4
    for entry in index["configurations"].values():
        assert len(entry["attempts"]) == 1
        assert set(entry["attempts"][0]["phases"]) == {"prefill", "decode"}
    runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output, execute=True, resume=True)
    assert len(executions) == 4
    assert all(Path(path).read_bytes() == contents for path, contents in original.items())
    # Resume rechecks collector-owned launch bytes as well as observations.
    altered = output / index["configurations"]["tp4"]["attempts"][0]["phases"]["prefill"]["artifacts"][0]["path"]
    altered.write_bytes(altered.read_bytes() + b"\n")
    runtime_probe.probe_runtime(configurations, instrumentation=manifest, output_dir=output, execute=True, resume=True)
    assert len(executions) == 6
    updated = json.loads((output / "observations.json").read_text())
    assert len(updated["configurations"]["tp4"]["attempts"]) == 2
    assert len(updated["configurations"]["dep4"]["attempts"]) == 1


def test_configuration_failure_preserves_successes_and_new_bundle_gets_new_attempt(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _marker = _inputs(tmp_path)
    second = copy.deepcopy(launch)
    second["topology"].update(tp=1, dp=4)
    executions = []
    monkeypatch.setattr(
        runner, "_salvage_artifacts", lambda resource, _cell: resource.collect([], require_benchmark=False)
    )
    monkeypatch.setattr(
        runner,
        "_cell_runner",
        lambda plan, cell, _manifest, directory: _SyntheticExecutor(
            plan, cell, directory, executions, fail_configuration="broken"
        ),
    )
    output = tmp_path / "probe"
    result = runtime_probe.probe_runtime(
        {"broken": launch, "good": second}, instrumentation=manifest, output_dir=output, execute=True
    )
    assert result["status"] == "partial"
    assert len(executions) == 4
    index = json.loads((output / "observations.json").read_text())
    first = index["configurations"]["good"]["attempts"][0]
    preserved = (output / first["bundle"]["manifest"]).read_bytes()
    (tmp_path / "observer.py").write_text("# revised synthetic observer\n")
    runtime_probe.probe_runtime(
        {"good": second}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    updated = json.loads((output / "observations.json").read_text())
    attempts = updated["configurations"]["good"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0] == first
    assert attempts[0]["bundle"]["sha256"] != attempts[1]["bundle"]["sha256"]
    assert (output / first["bundle"]["manifest"]).read_bytes() == preserved


def test_dense_tensor_parallel_probe_keeps_dense_moe_axes(tmp_path):
    from collector.fpm_forward import runtime_probe

    launch, manifest, _marker = _inputs(tmp_path)
    launch["identity"]["model_kind"] = "dense"
    launch["topology"].update(moe_ep=1)
    config = Path(launch["model_config"]["path"])
    payload = json.loads(config.read_text())
    payload.pop("num_experts")
    config.write_text(json.dumps(payload))
    launch["model_config"]["sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
    result = runtime_probe.probe_runtime({"dense": launch}, instrumentation=manifest, output_dir=tmp_path / "probe")
    assert result["status"] == "preview"
    assert result["configurations"]["dense"]["launch"]["topology"] == launch["topology"]


def test_probe_rejects_ignored_checkpoint_precision_and_resolves_explicit_graph_default(tmp_path):
    from collector.fpm_forward import runtime_probe

    launch, manifest, _marker = _inputs(tmp_path, graph_policy="explicit")
    launch["collection"]["max_prefill_cudagraph_size"] = None
    wrong = copy.deepcopy(launch)
    wrong["precision"]["gemm_quant_mode"] = "nvfp4"
    result = runtime_probe.probe_runtime(
        {"valid": launch, "wrong": wrong}, instrumentation=manifest, output_dir=tmp_path / "probe"
    )
    assert result["status"] == "partial"
    assert "checkpoint-native" in result["configurations"]["wrong"]["diagnostics"][0]
    assert result["configurations"]["valid"]["launch"]["collection"]["max_prefill_cudagraph_size"] == 2048


def _formal_cli_inputs(tmp_path):
    from .test_fpm_profile_collection import _profile

    launch, manifest, _marker = _inputs(tmp_path)
    profile = _profile()
    profile.update(
        model=launch["identity"]["model"],
        model_revision=launch["identity"]["model_revision"],
        architecture="ExternalMoeForCausalLM",
        context_length=4096,
        num_experts=8,
    )
    profile["deployments"] = [profile["deployments"][0]]
    deployment = profile["deployments"][0]
    deployment.update(system="gb200", backend_version="0.28.0", kv_cache_dtype="bfloat16", **launch["topology"])
    deployment.update({key: value for key, value in launch["precision"].items() if key != "kvcache_quant_mode"})
    deployment["resources"].update(max_num_tokens=1024, max_batch_size=64)
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(json.dumps(launch))
    argv = [
        "--model-path",
        launch["identity"]["model"],
        "--gpu",
        "gb200",
        "--fpm-model-profile",
        str(profile_path),
        "--fpm-model-config",
        launch["model_config"]["path"],
        "--fpm-max-gpus",
        "4",
        "--fpm-gpu-counts",
        "4",
        "--fpm-parallel-presets",
        "tep",
        "--fpm-kv-cache-dtypes",
        "bfloat16",
        "--fpm-executor",
        "slurm",
        "--fpm-slurm-container-image",
        launch["deployment"]["image"],
        "--fpm-slurm-container-mount",
        "/cache:/cache",
        "--fpm-max-model-len",
        "4096",
        "--fpm-max-num-batched-tokens",
        "1024",
        "--fpm-max-num-seqs",
        "64",
        "--fpm-gpu-memory-utilization",
        "0.85",
        "--fpm-prefill-cudagraph-policy",
        "runtime",
    ]
    extra = [
        "--fpm-runtime-instrumentation",
        str(manifest),
        "--fpm-runtime-launch",
        str(launch_path),
        "--fpm-runtime-configuration",
        "tp4",
    ]
    return argv, extra, launch


def test_formal_collector_cli_requires_all_instrumentation_inputs(tmp_path, capsys, monkeypatch):
    from collector.fpm_forward import cli

    monkeypatch.setenv("COLLECTOR_MODEL_PATH", "")
    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    with pytest.raises(SystemExit) as error:
        cli.main([*argv, *extra[:2], "--plan-only"])
    assert error.value.code == 2
    assert "required together" in capsys.readouterr().err
    assert cli.main([*argv, *extra, "--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["runtime_memory_policy"]["observation"] == "enabled"
    assert plan["runtime_observation"]["configuration"] == "tp4"
    assert plan["runtime_observation"]["launch"]["identity"]["framework_version"] == "0.28.0"


def test_repeatability_roundtrips_and_hashes_instrumented_formal_plan(tmp_path):
    from collector.fpm_forward import cli, entry, repeatability
    from collector.fpm_forward.runtime_memory import validate_saved_plan
    from collector.model_cases import build_collection_case_plan

    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    args = cli._parser().parse_args([*argv, *extra])
    case_plan = build_collection_case_plan(backend="vllm", model_path=args.model_path, gpu_type=args.gpu)
    plan, _overrides = entry.resolve_inputs(args, case_plan)
    saved = plan.to_dict()
    validate_saved_plan(saved)
    (tmp_path / "collection-plan.json").write_text(json.dumps(saved))
    loaded = repeatability.load_repeatability_source(tmp_path)
    assert loaded.to_dict() == saved
    assert loaded.runtime_instrumentation.sha256 == plan.runtime_instrumentation.sha256
    selected = {
        "cell_id": plan.cells[0].cell_id,
        "benchmark_points": {
            "schema_version": 3,
            "prefill": [{"batch_size": 1, "total_prefill_tokens": 16, "total_kv_read_tokens": 0}],
            "decode": [],
        },
    }
    subset = repeatability._subset_plan(loaded, selected)
    validate_saved_plan(subset.to_dict())
    assert subset.to_dict()["runtime_observation"] == saved["runtime_observation"]
    assert subset.sha256 != plan.sha256


def test_formal_collection_instruments_resolved_memory_and_archives_phase_attempts(tmp_path, monkeypatch):
    from collector.fpm_forward import cli, entry, runner
    from collector.model_cases import build_collection_case_plan

    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    args = cli._parser().parse_args([*argv, *extra])
    case_plan = build_collection_case_plan(backend="vllm", model_path=args.model_path, gpu_type=args.gpu)
    plan, overrides = entry.resolve_inputs(args, case_plan)
    assert plan.deployment_profile(plan.cells[0]).resources.memory_source != "pending"
    executions = []
    monkeypatch.setattr(
        runner,
        "_cell_runner",
        lambda plan, cell, _manifest, directory: _SyntheticExecutor(plan, cell, directory, executions),
    )
    monkeypatch.setattr(runner, "_runtime_collection_summary", lambda *_args, **_kwargs: {})
    errors = runner.run_collection(
        plan,
        generator_overrides=overrides,
        checkpoint_dir=str(tmp_path / "checkpoint"),
        artifact_root=str(tmp_path / "collection"),
        resume=False,
        retry_failed=False,
        publish_database=False,
    )
    assert errors == []
    root = tmp_path / "collection" / plan.sha256[:16]
    index = json.loads((root / "runtime-observations.json").read_text())
    attempt = index["configurations"]["tp4"]["attempts"][0]
    assert attempt["status"] == "captured"
    assert set(attempt["phases"]) == {"prefill", "decode"}
    contexts = [
        json.loads((root / phase["launch_manifest"]["path"]).read_text()) for phase in attempt["phases"].values()
    ]
    assert {context["attempt_id"] for context in contexts} == {attempt["attempt_id"]}
    assert len({context["collector_attempt_id"] for context in contexts}) == 2
    assert all(len(items) == 1 for items in attempt["phase_attempts"].values())
    for path in (root / "cells").glob("*/generator-request.json"):
        args = json.loads(path.read_text())["params"]["agg"]["extra_cli_args"]
        assert args[args.index("--revision") + 1] == plan.runtime_launch["identity"]["model_revision"]
        assert args.count("--revision") == 1


def _sidecar_inputs(tmp_path):
    launch, manifest, marker = _inputs(tmp_path)
    sidecar = tmp_path / "hf_quant_config.json"
    sidecar.write_text(json.dumps({"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None}}))
    launch["model_config"]["source_files"] = {sidecar.name: hashlib.sha256(sidecar.read_bytes()).hexdigest()}
    launch["precision"].update(gemm_quant_mode="nvfp4", moe_quant_mode="nvfp4")
    return launch, manifest, marker, sidecar


def test_probe_binds_raw_primary_and_adjacent_quantization_metadata(tmp_path):
    from collector.fpm_forward import runtime_probe

    launch, manifest, marker, sidecar = _sidecar_inputs(tmp_path)
    primary = Path(launch["model_config"]["path"])
    original = {"model-config.json": primary.read_bytes(), sidecar.name: sidecar.read_bytes()}
    output = tmp_path / "probe"
    result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=output)
    assert result["status"] == "preview", result
    assert primary.read_bytes() == original["model-config.json"]
    assert not marker.exists()
    for phase in result["configurations"]["tp4"]["phases"].values():
        context = json.loads((output / phase["launch_manifest"]["path"]).read_text())
        assert context["launch"]["model_config"] == launch["model_config"]
        artifacts = {Path(a["path"]).name: a for a in phase["artifacts"]}
        for name, content in original.items():
            assert (output / artifacts[name]["path"]).read_bytes() == content
            assert artifacts[name]["sha256"] == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize("mutation", ["missing", "changed", "malformed", "unrecorded", "escape", "symlink", "bad-hash"])
def test_probe_rejects_unbound_quantization_metadata(tmp_path, mutation):
    from collector.fpm_forward import runtime_probe

    launch, manifest, _marker, sidecar = _sidecar_inputs(tmp_path)
    # Keep native precision unchanged so rejection cannot be a precision mismatch.
    launch["precision"].update(gemm_quant_mode="bfloat16", moe_quant_mode="bfloat16")
    sidecar.write_text("{}")
    launch["model_config"]["source_files"][sidecar.name] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    if mutation == "missing":
        sidecar.unlink()
    elif mutation == "changed":
        sidecar.write_text("{}\n")
    elif mutation == "malformed":
        sidecar.write_text("[]")
        launch["model_config"]["source_files"][sidecar.name] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    elif mutation == "unrecorded":
        launch["model_config"].pop("source_files")
    elif mutation == "escape":
        launch["model_config"]["source_files"] = {"../hf_quant_config.json": "a" * 64}
    elif mutation == "symlink":
        sidecar.rename(tmp_path / "other.json")
        (tmp_path / "other.json").write_text("{}\n")
        sidecar.symlink_to(tmp_path / "other.json")
    elif mutation == "bad-hash":
        launch["model_config"]["source_files"][sidecar.name] = "not-a-digest"
    result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=tmp_path / "probe")
    assert result["status"] == "failed", result


def test_probe_preserves_logical_adjacency_for_hf_cache_symlinks(tmp_path):
    from collector.fpm_forward import runtime_probe

    launch, manifest, _marker, sidecar = _sidecar_inputs(tmp_path)
    primary = Path(launch["model_config"]["path"])
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for path, blob_name in ((primary, "primary-blob"), (sidecar, "quantization-blob")):
        target = blobs / blob_name
        path.rename(target)
        path.symlink_to(target)
    result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=tmp_path / "probe")
    assert result["status"] == "preview", result


def test_probe_rechecks_sidecar_between_plan_and_stage(tmp_path, monkeypatch):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _marker, sidecar = _sidecar_inputs(tmp_path)
    original_render = runner._render_cell

    def render_then_mutate(*args, **kwargs):
        result = original_render(*args, **kwargs)
        sidecar.write_bytes(sidecar.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(runner, "_render_cell", render_then_mutate)
    result = runtime_probe.probe_runtime({"tp4": launch}, instrumentation=manifest, output_dir=tmp_path / "probe")
    assert result["status"] == "failed"
    assert any("SHA-256" in item for item in result["configurations"]["tp4"]["diagnostics"])


@pytest.mark.parametrize("version", ["0.27.0", "0.28.0"])
@pytest.mark.parametrize("pending", [False, True])
def test_instrumented_saved_plan_hash_round_trip(tmp_path, version, pending):
    from collector.fpm_forward import cli, entry, runtime_memory
    from collector.model_cases import build_collection_case_plan

    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    for filename in ("profile.json", "manifest.json", "launch.json"):
        path = tmp_path / filename
        payload = json.loads(path.read_text().replace("0.28.0", version))
        if pending and filename == "profile.json":
            resources = payload["deployments"][0]["resources"]
            for field in ("weights_bytes", "activations_bytes", "runtime_overhead_bytes", "comm_overhead_bytes"):
                resources.pop(field, None)
        path.write_text(json.dumps(payload))
    args = cli._parser().parse_args([*argv, *extra])
    case_plan = build_collection_case_plan(backend="vllm", model_path=args.model_path, gpu_type=args.gpu)
    plan, _overrides = entry.resolve_inputs(args, case_plan)
    assert (plan.deployment_profile(plan.cells[0]).resources.memory_source == "pending") is pending
    payload = plan.to_dict()
    runtime_memory.validate_saved_plan(payload)
    for field in ("runtime_observation", "runtime_memory_policy"):
        changed = copy.deepcopy(payload)
        changed[field]["bundle_sha256"] = "f" * 64
        with pytest.raises(ValueError, match="SHA-256"):
            runtime_memory.validate_saved_plan(changed)


@pytest.mark.parametrize(
    "mutation",
    [
        "context",
        "binding",
        "manifest",
        "redirect",
        "ownership",
        "compute_domain_ownership",
        "compute_domain_missing_ownership",
    ],
)
def test_probe_resume_rejects_changed_ownership_before_external_commands(tmp_path, monkeypatch, mutation):
    import shutil

    import yaml
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _marker = _inputs(tmp_path)
    launch["deployment"] = {"executor": "kubernetes", "image": "image@sha256:synthetic"}
    auxiliary_ownership = mutation in {"compute_domain_ownership", "compute_domain_missing_ownership"}
    if auxiliary_ownership:
        launch["identity"]["gpu"] = "gb300"
        launch["topology"].update(tp=8, dp=1, moe_tp=8, moe_ep=1)
    configuration = "tp8" if auxiliary_ownership else "tp4"
    real_runner = runner._cell_runner

    class Interrupted(_SyntheticExecutor):
        def execute(self, pods):
            super().execute(pods)
            raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_salvage_artifacts", lambda *_args: None)
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: Interrupted(p, c, d, []))
    output = tmp_path / "probe"
    with pytest.raises(KeyboardInterrupt):
        runtime_probe.probe_runtime({configuration: launch}, instrumentation=manifest, output_dir=output, execute=True)
    index_path = output / "observations.json"
    index = json.loads(index_path.read_text())
    phase = index["configurations"][configuration]["attempts"][0]["phases"]["prefill"]
    context = output / phase["launch_manifest"]["path"]
    resource = context.parent / runner.FPM_MANIFEST_FILENAME
    if mutation == "context":
        context.write_bytes(context.read_bytes() + b"\n")
    elif mutation == "binding":
        payload = json.loads(context.read_text())
        payload["attempt_id"] = "unrelated-attempt"
        context.write_text(json.dumps(payload))
        phase["launch_manifest"]["sha256"] = hashlib.sha256(context.read_bytes()).hexdigest()
    elif mutation in {"manifest", "ownership"} or auxiliary_ownership:
        documents = list(yaml.safe_load_all(resource.read_text()))
        if auxiliary_ownership:
            metadata = next(document for document in documents if document["kind"] == "ComputeDomain")["metadata"]
        else:
            metadata = runner._workload_document(documents)["metadata"]
            metadata["name"] = "unrelated-production-worker"
        if mutation == "compute_domain_missing_ownership":
            for label in (
                "aiconfigurator.nvidia.com/owned-by",
                "aiconfigurator.nvidia.com/plan",
                runner.FPM_CELL_LABEL,
            ):
                metadata["labels"].pop(label)
        else:
            metadata["labels"][runner.FPM_CELL_LABEL] = "unrelated-cell"
        resource.write_text(yaml.safe_dump_all(documents))
        if (mutation == "ownership" or auxiliary_ownership) and "resource_manifest" in phase:
            phase["resource_manifest"]["sha256"] = hashlib.sha256(resource.read_bytes()).hexdigest()
    else:
        redirected = output / "redirected"
        shutil.copytree(context.parent, redirected)
        phase["launch_manifest"]["path"] = str((redirected / context.name).relative_to(output))
    index_path.write_text(json.dumps(index))
    monkeypatch.setattr(runner, "_cell_runner", real_runner)
    monkeypatch.setattr(runner, "_kubectl_command", lambda: ["kubectl"])
    commands = []

    def no_external(args, **kwargs):
        commands.append(args)
        raise RuntimeError("unexpected external command")

    monkeypatch.setattr(runner, "_run_command", no_external)
    result = runtime_probe.probe_runtime(
        {configuration: launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert result["status"] == "failed"
    assert commands == []
    if auxiliary_ownership:
        assert any("collector ownership" in item for item in result["configurations"][configuration]["diagnostics"])


@pytest.mark.parametrize(
    ("executor", "changed_bundle", "configuration"),
    [
        ("kubernetes", False, "tp4"),
        ("kubernetes", True, "tp4"),
        ("slurm", False, "tp4"),
        ("slurm", True, "tp4"),
        ("kubernetes", False, "tp8"),
        ("slurm", False, "dep8"),
    ],
)
def test_probe_resume_preserves_valid_interrupted_recovery(
    tmp_path, monkeypatch, executor, changed_bundle, configuration
):
    from collector.fpm_forward import runner, runtime_probe

    launch, manifest, _marker = _inputs(tmp_path)
    if executor == "kubernetes":
        launch["deployment"] = {"executor": executor, "image": launch["deployment"]["image"]}
    if configuration == "tp8":
        launch["identity"]["gpu"] = "gb300"
        launch["topology"].update(tp=8, dp=1, moe_tp=8, moe_ep=1)
    elif configuration == "dep8":
        launch["identity"]["gpu"] = "gb300"
        launch["topology"].update(tp=1, dp=8, moe_tp=1, moe_ep=8)
    executions, cleanups, salvages = [], [], []

    class InterruptedOnce(_SyntheticExecutor):
        def cleanup(self):
            cleanups.append(str(self.cell_dir))

        def execute(self, pods):
            super().execute(pods)
            if len(executions) == 1:
                raise KeyboardInterrupt()

    monkeypatch.setattr(runner, "_salvage_artifacts", lambda resource, _cell: salvages.append(str(resource.cell_dir)))
    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: InterruptedOnce(p, c, d, executions))
    output = tmp_path / "probe"
    with pytest.raises(KeyboardInterrupt):
        runtime_probe.probe_runtime({configuration: launch}, instrumentation=manifest, output_dir=output, execute=True)
    previous = json.loads((output / "observations.json").read_text())["configurations"][configuration]["attempts"][0]
    interrupted_dir = str((output / previous["phases"]["prefill"]["launch_manifest"]["path"]).parent)
    if changed_bundle:
        (tmp_path / "observer.py").write_text("# a revised observer retained alongside the interrupted bundle\n")
    result = runtime_probe.probe_runtime(
        {configuration: launch}, instrumentation=manifest, output_dir=output, execute=True, resume=True
    )
    assert result["status"] == "completed", result
    assert len(executions) == 3
    assert cleanups.count(interrupted_dir) == 3
    assert salvages.count(interrupted_dir) == 2
    attempts = json.loads((output / "observations.json").read_text())["configurations"][configuration]["attempts"]
    previous = attempts[0]
    assert any(item["kind"] == "observation" for item in previous["phases"]["prefill"]["recovery_artifacts"])
    for phase in attempts[-1]["phases"].values():
        observations = [item for item in phase["artifacts"] if item["kind"] == "observation"]
        assert len(observations) == (1 if configuration == "tp4" else 2)


@pytest.mark.parametrize("saved_status", ["running", "passed"])
@pytest.mark.parametrize("archive_failure", [None, "copy", "index", "context", "stale"])
def test_formal_resume_archives_recovered_native_evidence_before_skip(
    tmp_path, monkeypatch, saved_status, archive_failure
):
    from collector.fpm_forward import cli, entry, runner, runtime_probe
    from collector.model_cases import build_collection_case_plan

    from .test_fpm_runner import _native_payload

    argv, extra, _launch = _formal_cli_inputs(tmp_path)
    args = cli._parser().parse_args([*argv, *extra])
    case_plan = build_collection_case_plan(backend="vllm", model_path=args.model_path, gpu_type=args.gpu)
    plan, overrides = entry.resolve_inputs(args, case_plan)
    artifact_root = tmp_path / "collection"
    root = artifact_root / plan.sha256[:16]
    parent = "interrupted-parent"
    frozen = runtime_probe.prepare_collection_observations(plan, root, parent)
    prefill = next(cell for cell in plan.cells if cell.workload_kind == "prefill")
    cell_dir = root / "cells" / prefill.cell_id
    cell_dir.mkdir(parents=True)
    runner._render_cell(plan, prefill, cell_dir, overrides)
    context = runtime_probe.launch_context(plan, prefill, configuration=plan.runtime_configuration, attempt_id=parent)
    context.update(collector_attempt_id="prefill-native-attempt", cell_id=prefill.cell_id)
    runtime_probe.stage_runtime_instrumentation(frozen, cell_dir, context)
    raw = cell_dir / "raw" / "synthetic-node"
    raw.mkdir(parents=True)

    def native_files(cell, directory, attempt_id):
        (directory / "collector-provenance.json").write_text(
            json.dumps(
                {
                    "schema_name": "aic_fpm_collector_provenance",
                    "schema_version": 1,
                    "cell_id": cell.cell_id,
                    "plan_sha256": plan.sha256,
                    "attempt_id": attempt_id,
                    "runtime": {"backend": "vllm", "backend_version": "0.28.0"},
                }
            )
        )
        (directory / "benchmark.json").write_text(json.dumps(_native_payload(phase=cell.workload_kind, rank=0, dp=1)))

    native_files(prefill, raw, "prefill-native-attempt")
    observation = raw / "synthetic-observation.json"
    observation.write_text(
        json.dumps({**context, "schema_version": "aisimulate-runtime-observation/v1", "synthetic": True})
    )
    original_observation = observation.read_bytes()
    if archive_failure == "stale":
        # A prior failed snapshot can predate the final observation write.
        observation.unlink()
        runtime_probe.record_collection_observations(
            plan, root, parent, prefill, "prefill-native-attempt", cell_dir, "failed"
        )
        observation.write_bytes(original_observation)
    runner._runtime_collection_summary(
        prefill, cell_dir / "raw", expected_plan_sha256=plan.sha256, expected_attempt_id="prefill-native-attempt"
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "fpm_forward.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "schema": runner.CHECKPOINT_SCHEMA,
                "plan_sha256": plan.sha256,
                "runtime_observation_attempt_id": parent,
                "cells": {prefill.cell_id: {"status": saved_status, "attempt_id": "prefill-native-attempt"}},
            }
        )
    )
    executions = []

    class NativeExecutor(_SyntheticExecutor):
        def execute(self, pods):
            super().execute(pods)
            native_files(self.cell, self.cell_dir / "raw" / "synthetic-node", self.identity["attempt_id"])

    monkeypatch.setattr(runner, "_cell_runner", lambda p, c, m, d: NativeExecutor(p, c, d, executions))
    original_copy = runtime_probe.shutil.copy2
    original_atomic = runner._atomic_json
    context_path = cell_dir / runtime_probe.CONTEXT_FILENAME
    original_context = context_path.read_bytes()
    if archive_failure == "context":
        corrupted = {**context, "collector_attempt_id": "unrelated-attempt"}
        context_path.write_text(json.dumps(corrupted))
    if archive_failure == "copy":

        def interrupted_copy(source, target, *args, **kwargs):
            if Path(source) == context_path:
                raise OSError("synthetic observation snapshot interruption")
            return original_copy(source, target, *args, **kwargs)

        monkeypatch.setattr(runtime_probe.shutil, "copy2", interrupted_copy)
    if archive_failure == "index":

        def interrupted_index(path, payload):
            if Path(path).name == "runtime-observations.json":
                raise OSError("synthetic observation index interruption")
            return original_atomic(path, payload)

        monkeypatch.setattr(runner, "_atomic_json", interrupted_index)
    kwargs = dict(
        generator_overrides=overrides,
        checkpoint_dir=str(checkpoint_dir),
        artifact_root=str(artifact_root),
        resume=True,
        retry_failed=True,
        publish_database=False,
    )
    errors = runner.run_collection(plan, **kwargs)
    if archive_failure not in {None, "stale"}:
        assert any(item["classification"] == "runtime_observation_archive_failed" for item in errors)
        assert json.loads(checkpoint_path.read_text())["cells"][prefill.cell_id]["status"] == "failed"
        assert observation.read_bytes() == original_observation
        monkeypatch.setattr(runtime_probe.shutil, "copy2", original_copy)
        monkeypatch.setattr(runner, "_atomic_json", original_atomic)
        context_path.write_bytes(original_context)
        assert runner.run_collection(plan, **kwargs) == []
    else:
        assert errors == []
    index_path = root / "runtime-observations.json"
    index = json.loads(index_path.read_text())
    attempt = index["configurations"]["tp4"]["attempts"][0]
    assert set(attempt["phases"]) == {"prefill", "decode"}
    assert attempt["status"] == "captured"
    assert attempt["phases"]["prefill"]["collector_attempt_id"] == "prefill-native-attempt"
    archived = [item for item in attempt["phases"]["prefill"]["artifacts"] if item["kind"] == "observation"]
    assert len(archived) == 1
    assert (root / archived[0]["path"]).read_bytes() == original_observation
    if archive_failure == "stale":
        assert [phase["status"] for phase in attempt["phase_attempts"]["prefill"]] == ["failed", "passed"]
        assert not any(item["kind"] == "observation" for item in attempt["phase_attempts"]["prefill"][0]["artifacts"])
    assert observation.read_bytes() == original_observation
    checkpoint = json.loads(checkpoint_path.read_text())
    assert checkpoint["runtime_observations"] == str(index_path)
    assert all(entry["status"] == "passed" for entry in checkpoint["cells"].values())
    assert executions == [("tp4", "decode")]
    before = index_path.read_bytes()
    assert runner.run_collection(plan, **kwargs) == []
    assert index_path.read_bytes() == before
    assert executions == [("tp4", "decode")]
