# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public onboarding with independent synthetic initialization/native evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import aisimulate.main as cli
from aisimulate.config import CorePredictionConfig, CoreRecommendationConfig
from aisimulate.support.checkpoint import save_checkpoint
from aisimulate.support.finalization import finalize
from aisimulate.support.plan import check_plan
from aisimulate.support.schema import SupportRequest

pytestmark = pytest.mark.unit


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _native(phase):
    """Authored schema-v2 timings, deliberately no production observer helpers."""
    points, iterations = [], []
    for index, kv in enumerate([0] if phase == "prefill" else [8, 32], 1):
        prefill = 16 if phase == "prefill" else 0
        point = {
            "point_type": phase,
            "benchmark_id": index,
            "total_prefill_tokens": prefill,
            "total_kv_read_tokens": kv,
            "batch_size": 1,
            "expected_cudagraph_mode": "PIECEWISE" if prefill else "FULL",
            "expected_capture_size": prefill or 1,
            "padding_tokens": 0,
            "sample_reasons": ["capture"] if prefill else ["capture", "kvwarm_real_kv"],
        }
        fpm = {
            "counter_id": index,
            "dp_rank": 0,
            "wall_time": 0.001 * index,
            "scheduled_requests": {
                "num_prefill_requests": 1 if prefill else 0,
                "sum_prefill_tokens": prefill,
                "sum_prefill_kv_tokens": kv if prefill else 0,
                "num_decode_requests": 0 if prefill else 1,
                "sum_decode_kv_tokens": 0 if prefill else kv,
            },
        }
        points.append({"point": point, "fpms": [fpm]})
        iterations.append(
            {
                "benchmark_id": index,
                "point": point,
                "expected_dp_ranks": [0],
                "complete": True,
                "wall_time": fpm["wall_time"],
                "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
            }
        )
    result = {
        "schema_version": 2,
        "artifact_type": "rank",
        "status": "complete",
        "valid": True,
        "usable": True,
        "timing_valid": True,
        "run_id": f"synthetic-{phase}",
        "grid_digest": f"synthetic-{phase}-grid",
        "config": {"mode": phase},
        "coverage": {"expected_points": len(points), "completed_points": len(points), "skipped_points": 0},
        "dp": {"rank": 0, "size": 1},
        "results": points,
        "iteration_groups": iterations,
        "skipped_points": [],
        "missing_phases": [],
        "timing": {
            "benchmark_elapsed_seconds": 1.0,
            "measured_iteration_seconds": sum(item["wall_time"] for item in iterations),
        },
    }
    if phase == "decode":
        result["kvwarm"] = {"enabled": True, "warm_eligible": True, "skip_reason": None}
    return result


def _memory(provenance, phase, model, revision):
    count = 1000 if phase == "prefill" else 900
    group = {
        "layer_names": ["model.layers.0.attn", "model.layers.1.attn"],
        "spec_type": "vllm.v1.kv_cache_interface.FullAttentionSpec",
        "block_size_tokens": 16,
        "spec_page_size_bytes": 512,
        "sliding_window": None,
        "dtype": "torch.bfloat16",
    }
    settings = {
        "model_config": {
            "model": model,
            "revision": revision,
            "dtype": "torch.bfloat16",
            "quantization": None,
            "max_model_len": 128,
            "enforce_eager": False,
        },
        "cache_config": {
            "cache_dtype": "bfloat16",
            "gpu_memory_utilization": 0.9,
            "kv_cache_memory_bytes": None,
            "enable_prefix_caching": phase == "decode",
        },
        "scheduler_config": {"max_num_batched_tokens": 64, "max_num_seqs": 4, "async_scheduling": False},
        "parallel_config": {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "enable_expert_parallel": False,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1,
        },
        "compilation_config": {
            "mode": "VLLM_COMPILE",
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
            "max_cudagraph_capture_size": 64,
        },
        "kernel_config": {"moe_backend": "auto"},
        "quantization_config": None,
        "offload_config": {
            "offload_backend": "auto",
            "uva": {"cpu_offload_gb": 0},
            "prefetch": {"offload_group_size": 0, "offload_num_in_group": 1, "offload_prefetch_step": 1},
        },
    }
    common = {
        "schema_name": "aisimulate_fpm_runtime_memory",
        "schema_version": 1,
        "collector_provenance": provenance,
        "backend_version": "0.27.0",
        "status": "resolved",
        "dp_rank": 0,
        "resolved_config": settings,
    }
    worker = {
        **common,
        "kind": "worker",
        "tp_rank": 0,
        "pp_rank": 0,
        "effective_offloader": "vllm.model_executor.offloader.base.NoopOffloader",
        "cache": {
            "num_blocks": count,
            "allocated_cache_bytes": count * 1024,
            "available_cache_bytes": count * 1024 + 512,
            "storages": [
                {"device": "cuda:0", "pointer": 100, "size_bytes": count * 512},
                {"device": "cuda:0", "pointer": 200, "size_bytes": count * 512},
            ],
            "groups": [
                {
                    **group,
                    "kind": "attention",
                    "layer_classes": dict.fromkeys(
                        group["layer_names"], "vllm.model_executor.layers.attention.Attention"
                    ),
                }
            ],
        },
    }
    scheduler = {
        **common,
        "kind": "scheduler",
        "tp_rank": None,
        "pp_rank": None,
        "cache": {
            "num_blocks": count,
            "initial_free_blocks": count - 1,
            "reserved_blocks": 1,
            "null_block_id": 0,
            "watermark_blocks": 0,
            "pool_count": 1,
            "groups": [group],
        },
    }
    return worker, scheduler


def _prepare_collection(tmp_path: Path, extra_init_args: tuple[str, ...] = ()) -> tuple[SupportRequest, Path]:
    source = tmp_path / "config.json"
    _write(
        source,
        {
            "_name_or_path": "example/runtime-memory-test",
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "torch_dtype": "bfloat16",
        },
    )
    overrides = tmp_path / "overrides.json"
    _write(
        overrides,
        {
            "fmha_quant_mode": "bfloat16",
            "comm_quant_mode": "half",
            "kv_cache_dtype": "bfloat16",
            "max_num_tokens": 64,
            "max_batch_size": 4,
        },
    )
    original = tmp_path / "draft.yaml"
    assert (
        cli.main(
            [
                "onboard",
                "init",
                "--model-config",
                str(source),
                "--resource-overrides",
                str(overrides),
                "--model-revision",
                "synthetic-immutable-revision",
                "--framework-version",
                "0.27.0",
                "--gpu",
                "h200_sxm",
                "--interconnect",
                "nvswitch",
                "--tensor-parallel",
                "1",
                "--input-tokens",
                "16",
                "--output-tokens",
                "4",
                "--concurrency",
                "1",
                "--request-count",
                "1",
                "--context-length",
                "128",
                "--output",
                str(original),
                *extra_init_args,
            ]
        )
        == 0
    )
    root = (tmp_path / "collection").resolve()
    assert cli.main(["onboard", "plan", "--config", str(original), "--output-dir", str(root)]) == 0
    request = SupportRequest.from_yaml(root / "request.yaml")
    assert request.profile_deployment().resources.memory_source == "pending"
    return request, root


def build_completed_collection(
    tmp_path: Path, *, extra_init_args: tuple[str, ...] = (), plan_changes: dict | None = None
) -> tuple[SupportRequest, Path]:
    """Reusable CPU-only fixture for source and installed-wheel CLI checks.

    Timings and cache allocations are synthetic test declarations, not measured
    silicon. The production validator/resolver and formal publisher run unchanged.
    """
    from collector.fpm_forward import cli as collector_cli
    from collector.fpm_forward.config import FPMCollectionOptions
    from collector.fpm_forward.database import aggregate_cell, write_formal_database
    from collector.fpm_forward.planner import build_collection_plan
    from collector.fpm_forward.runner import CHECKPOINT_SCHEMA

    request, root = _prepare_collection(tmp_path, extra_init_args)
    source = tmp_path / "config.json"
    manifest = json.loads((root / "support-plan.json").read_text())
    args = collector_cli._parser().parse_args(manifest["fpm"]["plan_command"][3:])
    plan = build_collection_plan(
        backend="vllm",
        model_path=request.identity.model,
        model_architecture=args.model_architecture,
        model_config_path=str(source),
        system="h200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        options=replace(FPMCollectionOptions.from_args(args), **(plan_changes or {})),
        fpm_profile=request.fpm_profile,
    )
    artifact_root = root / "fpm-artifacts" / plan.sha256[:16]
    _write(artifact_root / "collection-plan.json", plan.to_dict())
    entries, rows = {}, []
    for cell in plan.cells:
        directory = artifact_root / "cells" / cell.cell_id
        _write(directory / "cell.json", cell.to_dict())
        raw = directory / "raw/pod-0"
        provenance = {
            "schema_name": "aic_fpm_collector_provenance",
            "schema_version": 1,
            "cell_id": cell.cell_id,
            "plan_sha256": plan.sha256,
            "attempt_id": f"attempt-{cell.workload_kind}",
            "runtime": {"backend": "vllm", "backend_version": "0.27.0"},
        }
        _write(raw / "collector-provenance.json", provenance)
        _write(raw / "benchmark.json", _native(cell.workload_kind))
        worker, scheduler = _memory(
            provenance, cell.workload_kind, request.identity.model, request.identity.model_revision
        )
        sampling = plan.options.prefill_sampling
        if sampling.cudagraph_policy == "explicit":
            worker["resolved_config"]["compilation_config"].update(
                cudagraph_capture_sizes=list(sampling.cudagraph_capture_sizes),
                max_cudagraph_capture_size=sampling.max_cudagraph_capture_size,
            )
        _write(raw / "fpm-memory-worker-dp0-tp0-pp0.json", worker)
        _write(raw / "fpm-memory-scheduler-dp0.json", scheduler)
        rows.extend(aggregate_cell(plan, cell, directory, expected_attempt_id=provenance["attempt_id"]))
        entries[cell.cell_id] = {"status": "passed", "attempt_id": provenance["attempt_id"]}
    parquet, metadata, skipped = write_formal_database(plan, rows, systems_root=root / "systems/data")
    assert not skipped
    _write(
        root / "fpm-checkpoint/fpm_forward.json",
        {
            "schema": CHECKPOINT_SCHEMA,
            "plan_sha256": plan.sha256,
            "cells": entries,
            "database": {
                "status": "passed",
                "parquet": str(parquet),
                "metadata": str(metadata),
                "row_count": len(rows),
                "published_cells": len(plan.cells),
                "plan_cells": len(plan.cells),
                "missing_cells": [],
                "skipped_first_publisher_wins": [],
            },
        },
    )
    # Finalization must use immutable saved inputs, never refetch/rederive config.
    source.unlink()
    (tmp_path / "overrides.json").unlink()
    return request, root


def _snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    "plan_changes",
    [{}, {"max_prefill_cudagraph_size": 32}, {"max_prefill_isl": 32}, {"warmup_iterations": 2}],
)
@pytest.mark.parametrize("executor", ["kubernetes", "slurm"])
def test_public_finalize_binds_reviewed_collection_options(tmp_path, capsys, plan_changes, executor):
    deployment = (
        {
            "executor": "slurm",
            "slurm_container_image": "image@sha256:synthetic",
            "slurm_container_mounts": ("/cache:/cache",),
        }
        if executor == "slurm"
        else {}
    )
    _request, root = build_completed_collection(
        tmp_path,
        extra_init_args=("--prefill-cudagraph-policy", "explicit", "--max-prefill-cudagraph-size", "64"),
        plan_changes={**deployment, **plan_changes},
    )
    target = tmp_path / "resolved"
    before = _snapshot(root)
    try:
        status = cli.main(
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
    except SystemExit as error:
        status = error.code
    assert status == (2 if plan_changes else 0)
    assert _snapshot(root) == before
    if plan_changes:
        assert "saved collection options differ" in capsys.readouterr().err
        assert not target.exists()
    else:
        assert SupportRequest.from_yaml(target / "request.yaml").collection.max_prefill_cudagraph_size == 64


@pytest.mark.parametrize("tamper", [None, "image", "mounts", "executor", "worker"])
def test_public_slurm_collect_to_finalize_preserves_frozen_deployment(tmp_path, monkeypatch, capsys, tamper):
    """Real CLI, Slurm campaign and finalizer; only cluster/runtime output is synthetic."""
    from collector.fpm_forward import runner

    request, root = _prepare_collection(tmp_path, ("--model", str(tmp_path)))
    image = "registry.example/fpm@sha256:" + "a" * 64
    commands = []

    def cluster_command(args, **_kwargs):
        commands.append(args)
        if args[0] == "scontrol":
            output = "JobId=1234 JobState=RUNNING NodeList=node-a" if "job" in args else "node-a"
            return subprocess.CompletedProcess(args, 0, stdout=output, stderr="")
        if args[0] == "squeue":
            return subprocess.CompletedProcess(args, 0, stdout="1234.99|unrelated-job", stderr="")
        assert args[0] == "srun", args
        assert "--jobid=1234" in args and "--gpus-per-node=1" in args
        assert f"--container-image={image}" in args
        mounts = next(value.split("=", 1)[1] for value in args if value.startswith("--container-mounts="))
        assert mounts.startswith("/cache:/cache,/models:/models,")
        raw = Path(next(value.rsplit(":", 1)[0] for value in mounts.split(",") if value.endswith(":/results")))
        command = args[args.index("/usr/bin/env") + 3 :]
        if command[:2] == ["python3", "-c"]:
            with monkeypatch.context() as patch:
                patch.setattr(sys, "argv", ["-c", *command[3:]])
                patch.setattr(importlib.metadata, "version", lambda _name: "0.27.0")
                exec(command[2].replace("/results", str(raw)), {})
        else:
            assert command == ["bash", "/tmp/fpm-bench/fpm_exec.sh"]
            cell = json.loads((raw.parent.parent / "cell.json").read_text())
            provenance = json.loads((raw / "collector-provenance.json").read_text())
            _write(raw / "benchmark.json", _native(cell["workload_kind"]))
            worker, scheduler = _memory(
                provenance, cell["workload_kind"], request.identity.model, request.identity.model_revision
            )
            _write(raw / "fpm-memory-worker-dp0-tp0-pp0.json", worker)
            _write(raw / "fpm-memory-scheduler-dp0.json", scheduler)
        return subprocess.CompletedProcess(args, 0, stdout="synthetic Slurm runtime", stderr="")

    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setattr("collector.fpm_forward.slurm.shutil.which", lambda name: f"/fake/{name}")
    monkeypatch.setattr(runner, "_run_command", cluster_command)
    assert (
        cli.main(
            [
                "onboard",
                "collect-fpm",
                "--config",
                str(root / "request.yaml"),
                "--output-dir",
                str(root),
                "--executor",
                "slurm",
                "--image",
                image,
                "--container-mount",
                "/cache:/cache",
                "--container-mount",
                "/models:/models",
                "--execute",
            ]
        )
        == 0
    )
    collection_path = next(root.glob("fpm-artifacts/*/collection-plan.json"))
    payload = json.loads(collection_path.read_text())
    assert payload["options"]["executor"] == "slurm"
    assert payload["options"]["slurm_container_image"] == image
    assert payload["options"]["slurm_container_mounts"] == ["/cache:/cache", "/models:/models"]
    if tamper == "worker":
        next(root.glob("fpm-artifacts/*/cells/*/raw/node0000/fpm-memory-worker*.json")).unlink()
    elif tamper is not None:
        key, value = {
            "image": ("slurm_container_image", "changed-image"),
            "mounts": ("slurm_container_mounts", ["/different:/cache"]),
            "executor": ("executor", "kubernetes"),
        }[tamper]
        payload["options"][key] = value
        _write(collection_path, payload)
    (tmp_path / "config.json").unlink()
    before, calls = _snapshot(root), list(commands)
    target = tmp_path / "resolved"
    try:
        status = cli.main(
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
    except SystemExit as error:
        status = error.code
    assert status == (2 if tamper else 0)
    assert _snapshot(root) == before
    assert commands == calls  # Finalization never starts or reconnects to the executor.
    if tamper:
        diagnostic = "worker rank evidence is incomplete" if tamper == "worker" else "SHA-256"
        assert diagnostic in capsys.readouterr().err
        assert not target.exists()
    else:
        resolved = SupportRequest.from_yaml(target / "request.yaml")
        assert resolved.profile_deployment().resources.runtime_memory.kv_cache_bytes == 899 * 1024
        check_plan(resolved, target)


@pytest.mark.parametrize(
    "revision,loaded,accepted",
    [
        ("2" * 40, None, False),
        (None, "2" * 40, False),
        ("main", "1" * 40, True),
        (None, None, True),
        ("1" * 40, "1" * 40, True),
    ],
)
def test_public_finalize_checks_positive_immutable_revision_evidence(tmp_path, capsys, revision, loaded, accepted):
    _request, root = build_completed_collection(tmp_path, extra_init_args=("--model-revision", "1" * 40))
    for path in root.glob("fpm-artifacts/*/cells/*/raw/*/fpm-memory-*.json"):
        evidence = json.loads(path.read_text())
        model = evidence["resolved_config"]["model_config"]
        model.update(model="/mnt/pvc/local-checkpoint", revision=revision)
        if loaded is not None:
            model["loaded_config_commit_hash"] = loaded
        _write(path, evidence)
    target = tmp_path / "resolved"
    before = _snapshot(root)
    try:
        status = cli.main(
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
    except SystemExit as error:
        status = error.code
    assert status == (0 if accepted else 2)
    assert _snapshot(root) == before
    if accepted:
        assert SupportRequest.from_yaml(target / "request.yaml").identity.model_revision == "1" * 40
    else:
        assert "contradicts the reviewed immutable revision" in capsys.readouterr().err
        assert not target.exists()


def test_public_finalize_preserves_collection_and_resolves_ordinary_configs(tmp_path, capsys):
    from aisimulate_core.sdk.memory import estimate_kv_cache

    request, root = build_completed_collection(tmp_path)
    checkpoint = tmp_path / "session.json"
    state, _ = save_checkpoint(
        checkpoint,
        patch={"configurations": {"tp1": {"draft_request": request.model_dump(mode="json")}}},
        expected_revision=None,
        accept=["tp1"],
    )
    assert state.configurations["tp1"].acceptance is not None
    before, session = _snapshot(root), checkpoint.read_bytes()
    target = tmp_path / "resolved"
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
    assert _snapshot(root) == before
    assert checkpoint.read_bytes() == session
    resolved = SupportRequest.from_yaml(target / "request.yaml")
    assert resolved.profile_deployment().resources.memory_source == "runtime"
    resources = resolved.profile_deployment().resources
    assert resources.runtime_memory.kv_cache_bytes == 899 * 1024
    assert resources.weights_bytes is resources.activations_bytes is None
    assert len(resources.cache_groups) == 1
    assert resources.cache_groups[0].page_size_bytes == 1024
    assert resources.cache_groups[0].num_layers == 2
    manifest = json.loads((target / "finalization.json").read_text())
    assert manifest["review_status"] == "requires_review"
    assert len(manifest["cell_observations"]) == 2
    for item in manifest["formal_data"]:
        copied = target / item["relative_path"]
        assert copied.read_bytes() == (root / item["relative_path"]).read_bytes()
        assert hashlib.sha256(copied.read_bytes()).hexdigest() == item["sha256"]
    assert sorted(path.name for path in (target / "systems/data").rglob("*") if path.is_file()) == [
        "fpm_forward_perf.metadata.json",
        "fpm_forward_perf.parquet",
    ]
    check_plan(resolved, target)
    for config in (
        CorePredictionConfig.from_yaml(target / "predict/pilot.yaml"),
        CoreRecommendationConfig.from_yaml(target / "recommend/pilot.yaml"),
    ):
        assert config.engine.fpm_profile == resolved.fpm_profile
        assert not config.engine.workers.aggregated.kv_cache.prefix_caching
        assert [str(path) for path in config.engine.systems_paths] == [str(target / "systems")]
    estimate = estimate_kv_cache(
        resolved.identity.model,
        "h200_sxm",
        "vllm",
        backend_version="0.27.0",
        tp_size=1,
        pp_size=1,
        attention_dp_size=1,
        moe_tp_size=1,
        moe_ep_size=1,
        max_num_tokens=64,
        max_batch_size=4,
        memory_fraction_kind="of_total",
        memory_fraction_value=0.9,
        context_length=128,
        fpm_profile=resolved.fpm_profile,
    )
    assert estimate["total_kv_size_bytes"] == 899 * 1024
    assert json.loads(resolved.fpm_profile.provenance)["memory_source"] == "runtime"
    for command in ("predict", "recommend"):
        assert (
            cli.main(
                [
                    command,
                    "--config",
                    str(target / command / "pilot.yaml"),
                    "--output-dir",
                    str(tmp_path / f"{command}-result"),
                    "--format",
                    "json",
                ]
            )
            == 0
        )
    trace = tmp_path / "trace.jsonl"
    _write(
        trace,
        {
            "id": "synthetic-play",
            "models": ["trace/source-model"],
            "block_size": 16,
            "hash_id_scope": "local",
            "requests": [{"t": 0.0, "type": "s", "model": "trace/source-model", "in": 16, "out": 4, "hash_ids": [1]}],
        },
    )
    assert (
        cli.main(
            [
                "onboard",
                "validate-fpm",
                "--config",
                str(target / "request.yaml"),
                "--output-dir",
                str(target),
                "--trace",
                str(trace),
                "--validation-output-dir",
                str(tmp_path / "validation"),
            ]
        )
        == 0
    )
    assert json.loads((tmp_path / "validation/validation.json").read_text())["status"] == "covered"
    assert "Review the new profile" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change,diagnostic",
    [
        ("missing_worker", "worker rank evidence is incomplete"),
        ("attempt", "different plan, attempt"),
        ("phase_graph", "incompatible resolved launch settings"),
        ("layout", "incompatible runtime cache layouts"),
        ("plan", "SHA-256"),
        ("partial", "complete formal publication"),
        ("formal", "commit record"),
        ("formal_row", "differ from the verified native collection"),
    ],
)
def test_finalize_refuses_unverified_or_incompatible_collection(tmp_path, change, diagnostic):
    request, root = build_completed_collection(tmp_path)
    if change in {"missing_worker", "attempt", "phase_graph", "layout"}:
        worker_path = next(root.glob("fpm-artifacts/*/cells/*/raw/pod-0/fpm-memory-worker*.json"))
        worker = json.loads(worker_path.read_text())
        if change == "missing_worker":
            worker_path.unlink()
        elif change == "attempt":
            worker["collector_provenance"]["attempt_id"] = "stale-attempt"
            _write(worker_path, worker)
        elif change == "phase_graph":
            worker["resolved_config"]["compilation_config"]["cudagraph_capture_sizes"] = [1, 2]
            worker["resolved_config"]["compilation_config"]["max_cudagraph_capture_size"] = 2
            _write(worker_path, worker)
            scheduler_path = worker_path.with_name("fpm-memory-scheduler-dp0.json")
            scheduler = json.loads(scheduler_path.read_text())
            scheduler["resolved_config"]["compilation_config"] = worker["resolved_config"]["compilation_config"]
            _write(scheduler_path, scheduler)
        else:
            worker["cache"]["groups"][0]["block_size_tokens"] = 8
            _write(worker_path, worker)
            scheduler_path = worker_path.with_name("fpm-memory-scheduler-dp0.json")
            scheduler = json.loads(scheduler_path.read_text())
            scheduler["cache"]["groups"][0]["block_size_tokens"] = 8
            _write(scheduler_path, scheduler)
    elif change == "plan":
        path = next(root.glob("fpm-artifacts/*/collection-plan.json"))
        payload = json.loads(path.read_text())
        payload["options"]["max_num_seqs"] = 9
        _write(path, payload)
    elif change == "partial":
        path = root / "fpm-checkpoint/fpm_forward.json"
        payload = json.loads(path.read_text())
        payload["database"]["missing_cells"] = ["unfinished"]
        _write(path, payload)
    elif change == "formal":
        path = next(root.glob("systems/data/**/fpm_forward_perf.parquet"))
        path.write_bytes(path.read_bytes() + b"changed")
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = next(root.glob("systems/data/**/fpm_forward_perf.parquet"))
        rows = pq.read_table(path).to_pylist()
        rows[0]["latency_ms"] *= 2
        pq.write_table(pa.Table.from_pylist(rows), path)
        metadata = path.with_name("fpm_forward_perf.metadata.json")
        payload = json.loads(metadata.read_text())
        payload["parquet_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        _write(metadata, payload)
    before = _snapshot(root)
    with pytest.raises(ValueError, match=diagnostic):
        finalize(request, root, tmp_path / "resolved")
    assert _snapshot(root) == before
    assert not (tmp_path / "resolved").exists()


def test_finalized_data_changes_fail_normal_saved_plan_validation(tmp_path):
    request, root = build_completed_collection(tmp_path)
    target = tmp_path / "resolved"
    finalize(request, root, target)
    resolved = SupportRequest.from_yaml(target / "request.yaml")
    data = next(target.glob("systems/data/**/fpm_forward_perf.parquet"))
    data.write_bytes(data.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="finalized FPM data changed"):
        check_plan(resolved, target)


@pytest.mark.parametrize("target_kind", ["original", "inside", "ancestor", "existing"])
def test_finalize_requires_new_separate_output(tmp_path, target_kind):
    request, root = build_completed_collection(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep").write_text("untouched")
    target = {"original": root, "inside": root / "new", "ancestor": tmp_path, "existing": existing}[target_kind]
    before = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="separate|must be new"):
        finalize(request, root, target)
    assert _snapshot(tmp_path) == before


def test_pending_profile_collection_plan_checkpoint_and_replay_gate(tmp_path, capsys):
    request, root = build_completed_collection(tmp_path)
    check_plan(request, root)
    plan = json.loads((root / "support-plan.json").read_text())
    assert plan["resources"] == {"memory_source": "pending", "simulation_ready": False}
    assert "Synchronous" in plan["fpm"]["scheduling_policy"]
    assert "finalize" in json.loads((root / "commands.json").read_text())
    _write(tmp_path / "trace.json", [])
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "onboard",
                "validate-fpm",
                "--config",
                str(root / "request.yaml"),
                "--output-dir",
                str(root),
                "--trace",
                str(tmp_path / "trace.json"),
                "--validation-output-dir",
                str(tmp_path / "validation"),
            ]
        )
    assert error.value.code == 2
    assert "pending" in capsys.readouterr().err
    assert not (tmp_path / "validation").exists()
