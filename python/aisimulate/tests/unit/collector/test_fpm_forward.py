# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from collector.fpm_forward.capabilities import resolve_model_capability
from collector.fpm_forward.config import FPMCollectionOptions, PrefillSamplingProfile, add_fpm_arguments
from collector.fpm_forward.database import (
    aggregate_cell,
    validate_formal_database_commit,
    write_formal_database,
)
from collector.fpm_forward.memory_admission import filter_memory_infeasible_topologies
from collector.fpm_forward.model_capability import load_model_config
from collector.fpm_forward.planner import (
    BackendPolicy,
    FPMCell,
    backend_identity_columns,
    build_collection_plan,
)
from collector.fpm_forward.topology import enumerate_fpm_topologies
from collector.fpm_forward.types import ParallelTopology

from aisimulate.sdk.utils import HuggingFaceDownloadError

pytestmark = pytest.mark.unit

_REQUIRED_INSTALLED_FPM_PAYLOAD = {
    "collector/fpm_forward/planner.py": b"planner-content",
    "collector/fpm_forward/runner.py": b"runner-content",
    "collector/fpm_forward/runtime/fpm_exec.sh": b"runtime-content",
    "collector/fpm_forward/runtime/preflight.py": b"preflight-content",
}


def _record_sha256(payload: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    return f"sha256={digest}"


def _installed_distribution(tmp_path, *, version="0.12.0", rows=None):
    root = tmp_path / "site-packages"
    for relative_path, payload in _REQUIRED_INSTALLED_FPM_PAYLOAD.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if rows is None:
        rows = [
            (relative_path, _record_sha256(payload), str(len(payload)))
            for relative_path, payload in _REQUIRED_INSTALLED_FPM_PAYLOAD.items()
        ]
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(rows)

    class InstalledDistribution:
        @property
        def version(self):
            return version

        @staticmethod
        def locate_file(path):
            return root / str(path)

        @staticmethod
        def read_text(filename):
            assert filename == "RECORD"
            return stream.getvalue()

    return InstalledDistribution(), root


@pytest.fixture(autouse=True)
def _pinned_git_revision(monkeypatch):
    """Keep plan-identity tests hermetic: no git binary or checkout required
    (CI test containers ship without git), and plan hashes must not depend on
    the ambient repository HEAD. Yields the real function for the tests that
    exercise revision derivation itself."""

    from collector.fpm_forward import planner as planner_module

    real = planner_module._git_revision
    monkeypatch.setattr(planner_module, "_git_revision", lambda: "test-revision")
    yield real


def _write_provenance(path, *, cell_id: str, plan_sha256: str = "plan-sha", attempt_id: str = "attempt"):
    path.write_text(
        json.dumps(
            {
                "schema_name": "aic_fpm_collector_provenance",
                "schema_version": 1,
                "cell_id": cell_id,
                "plan_sha256": plan_sha256,
                "attempt_id": attempt_id,
                "runtime": {"backend": "vllm", "backend_version": "0.24.0"},
            }
        )
    )


def _concurrent_database_writer(root: str, row: dict, start_event) -> None:
    start_event.wait(timeout=10)
    plan = SimpleNamespace(system="b200_sxm", backend="vllm", aic_revision="revision")
    write_formal_database(plan, [row], systems_root=Path(root))


def _args(**overrides):
    values = {
        "fpm_max_gpus": 4,
        "fpm_gpu_counts": [4],
        "fpm_parallel_presets": None,
        "fpm_parallel_axes": None,
        "fpm_moe_backend": None,
        "fpm_attention_backend": None,
        "fpm_enable_wideep": None,
        "fpm_enable_eplb": None,
        "fpm_weight_quantizations": None,
        "fpm_kv_cache_dtypes": None,
        "fpm_model_config": None,
        "fpm_tp_sizes": None,
        "fpm_pp_sizes": None,
        "fpm_dp_sizes": None,
        "fpm_moe_tp_sizes": None,
        "fpm_moe_ep_sizes": None,
        "fpm_cp_sizes": None,
        "fpm_warmup_iterations": None,
        "fpm_max_prefill_isl": None,
        "fpm_max_prefill_batch_size": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_options_leave_point_generation_to_dynamo():
    options = FPMCollectionOptions.from_args(_args())

    assert options.warmup_iterations == 5
    assert options.gpu_counts == (4,)
    assert options.parallel_presets == ("auto",)
    assert options.to_dict()["point_source"] == "dynamo_native_self_benchmark"
    assert options.to_dict()["measurement_repeats"] == 1
    assert options.max_prefill_isl == 8192
    assert options.max_prefill_batch_size is None
    assert options.vllm_max_model_len == -1
    assert options.prefill_sampling.max_total_prefill_tokens == 8192
    assert len(options.prefill_sampling.cudagraph_capture_sizes) == 99
    assert options.prefill_sampling.max_cudagraph_capture_size == 2048
    assert len(options.prefill_sampling.new_token_axis_points) == 199
    assert options.prefill_sampling.max_new_token_samples == 199
    assert "sampling_budget" not in options.to_dict()
    assert "kv_block_size" not in options.to_dict()

    warmed = FPMCollectionOptions.from_args(_args(fpm_warmup_iterations=3))
    assert warmed.to_dict()["global_warmup_iterations"] == 3
    disabled = FPMCollectionOptions.from_args(_args(fpm_warmup_iterations=0))
    assert disabled.to_dict()["global_warmup_iterations"] == 0

    with pytest.raises(ValueError, match="exceed"):
        FPMCollectionOptions.from_args(_args(fpm_gpu_counts=[4, 8]))


def test_prefill_limits_expose_no_cli_aliases():
    parser = argparse.ArgumentParser()
    add_fpm_arguments(parser)

    help_text = parser.format_help()
    assert "--fpm-max-prefill-isl" in help_text
    assert "--fpm-max-prefill-batch-size" in help_text
    assert "--fpm-model-config" in help_text
    assert "--fpm-max-isl" not in help_text
    assert "--fpm-max-prefill-bs" not in help_text


def test_prefill_sampling_profile_keeps_vllm_strides_and_exact_endpoint():
    short = PrefillSamplingProfile.build(max_isl=1000, max_batch_size=16)

    assert short.max_cudagraph_capture_size == 1000
    assert short.cudagraph_capture_sizes[:7] == (1, 2, 4, 8, 16, 24, 32)
    assert short.cudagraph_capture_sizes[-4:] == (928, 960, 992, 1000)
    assert len(short.cudagraph_capture_sizes) == 67
    assert len(short.new_token_axis_points) == 132
    assert short.max_new_token_samples == 132

    long = PrefillSamplingProfile.build(max_isl=8192, max_batch_size=None)
    assert long.cudagraph_capture_sizes[-4:] == (1952, 1984, 2016, 2048)
    assert long.new_token_axis_points[-4:] == (2048, 2049, 4096, 8192)
    assert long.to_dict()["new_token_axis_point_count"] == 199


def test_parallel_topologies_are_delegated_to_aic_enumerator():
    options = FPMCollectionOptions.from_args(_args())
    topologies = enumerate_fpm_topologies(backend="vllm", is_moe=True, options=options)
    assert topologies == (
        ParallelTopology(tp=1, pp=1, dp=4, moe_tp=1, moe_ep=4, cp=1),
        ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4, cp=1),
    )

    with_pure_tp = enumerate_fpm_topologies(
        backend="vllm",
        is_moe=True,
        options=options,
        allow_pure_tp=True,
    )
    assert with_pure_tp == (
        ParallelTopology(tp=1, pp=1, dp=4, moe_tp=1, moe_ep=4, cp=1),
        ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1),
        ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4, cp=1),
    )


def test_pure_tp_requires_explicit_model_runtime_capability():
    options = FPMCollectionOptions.from_args(
        _args(
            fpm_parallel_presets=["pure_tp"],
            fpm_moe_tp_sizes=[4],
        )
    )

    with pytest.raises(ValueError, match="does not explicitly admit"):
        enumerate_fpm_topologies(backend="vllm", is_moe=True, options=options)


@pytest.mark.parametrize("explicit", [False, True])
def test_plan_contains_only_cell_matrix_and_native_point_contract(tmp_path, explicit):
    points_file = tmp_path / "points.json"
    points_file.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "prefill": [{"batch_size": 1, "total_prefill_tokens": 128, "total_kv_read_tokens": 0}],
                "decode": [],
            }
        )
    )
    options = FPMCollectionOptions.from_args(
        _args(
            fpm_parallel_axes=["dp", "moe_ep"],
            fpm_dp_sizes=[4],
            fpm_moe_ep_sizes=[4],
            fpm_benchmark_points_file=str(points_file) if explicit else None,
        )
    )
    kwargs = {
        "backend": "vllm",
        "model_path": "nvidia/GLM-5.2-NVFP4",
        "system": "b200_sxm",
        "selected_ops": {"dsa_context_module", "dsa_generation_module"},
        "options": options,
    }
    first = build_collection_plan(
        **kwargs,
        generator_overrides={"K8sConfig": {"k8s_image": "example/vllm-runtime:first"}},
    )
    second = build_collection_plan(
        **kwargs,
        generator_overrides={"K8sConfig": {"k8s_image": "example/vllm-runtime:second"}},
    )

    assert first.sha256 != second.sha256
    assert first.dtype_profile.gemm_quant_mode == "nvfp4"
    assert first.dtype_profile.kv_cache_dtypes == ("fp8",)
    assert len(first.cells) == 2
    assert {cell.workload_kind for cell in first.cells} == {"prefill", "decode"}
    assert {cell.parallel_strategy for cell in first.cells} == {"dep"}
    payload = first.to_dict()
    assert payload["schema_version"] == 11
    assert payload["capability"]["model_config"]["source_kind"] == "aic_cache"
    assert len(payload["capability"]["model_config"]["sha256"]) == 64
    assert payload["capability"]["model_config"]["payload"]["architectures"] == ["GlmMoeDsaForCausalLM"]
    point_generation = dict(payload["point_generation"])
    prefill_sampling = point_generation.pop("prefill_sampling")
    assert point_generation == {
        "owner": "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler",
        "method": "native_self_benchmark",
        "source": "frozen_explicit_manifest" if explicit else "native_auto_grid",
        "manifest_sha256": options.benchmark_points_sha256,
        "coordinates": ["batch_size", "total_prefill_tokens", "total_kv_read_tokens"],
        "partition_policy": "balanced_v1",
        "point_admission": "dynamo_live_scheduler",
        "precondition": "vllm_engine_initialized",
        "planned_point_count": 1 if explicit else None,
    }
    assert prefill_sampling["cudagraph_capture_size_count"] == 99
    assert prefill_sampling["new_token_axis_point_count"] == 199
    assert prefill_sampling["prefill_max_new_token_samples"] == 199
    assert "prefix_max_batch_size_samples" not in prefill_sampling
    assert payload["counts"]["prefill_cudagraph_capture_sizes"] == 99
    assert payload["counts"]["prefill_new_token_axis_points"] == 199
    assert payload["counts"]["points"] == "runtime-determined"
    assert "population" not in payload
    assert "sampling" not in payload
    assert "capacity_admission" not in payload
    assert payload["counts"]["candidate_topologies"] == 1
    assert payload["counts"]["memory_rejected_topologies"] == 0
    assert payload["topology_memory_admission"][0]["disposition"] == "admitted"
    assert "runtime_overlay" not in payload


def test_backend_policy_is_deeply_immutable():
    source = {"nested": {"values": [1, 2]}}
    policy = BackendPolicy("baseline", source, {"runtime.mode": "FULL"})
    original = policy.to_dict()

    source["nested"]["values"].append(3)
    detached = policy.generator_overrides
    detached["nested"]["values"].append(4)

    assert policy.to_dict() == original


def test_glm_auto_matrix_keeps_aic_parallel_and_dtype_resolution():
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.allow_pure_tp is True
    assert {cell.parallel_strategy for cell in plan.cells} == {"pure_tp", "tep", "dep"}
    assert ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1) in plan.topologies
    assert len(plan.cells) == len(plan.topologies) * 2
    assert all(cell.to_dict()["point_source"] == "dynamo_native_self_benchmark" for cell in plan.cells)


def test_glm_memory_admission_uses_configured_max_new_tokens_and_warns_on_drops(caplog):
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_gpu_counts=[1, 2, 4],
                fpm_max_prefill_isl=16384,
            )
        ),
    )

    assert {topology.total_gpus for topology in plan.topologies} == {4}
    assert len(plan.topologies) == 3
    payload = plan.to_dict()
    assert payload["counts"]["candidate_topologies"] == 7
    assert payload["counts"]["memory_rejected_topologies"] == 4
    assert {
        decision["topology"]["tp"] * decision["topology"]["dp"]
        for decision in payload["topology_memory_admission"]
        if decision["disposition"] == "rejected"
    } == {1, 2}
    assert "fpm_forward: dropped 4/7 topologies" in caplog.text
    assert "max_new_tokens=16384" in caplog.text
    assert {decision["activation_envelope"]["max_new_tokens"] for decision in payload["topology_memory_admission"]} == {
        16384
    }


def test_glm_memory_admission_fails_after_warning_when_every_topology_is_impossible(caplog):
    with pytest.raises(ValueError, match="rejected every FPM topology"):
        build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            model_architecture="GlmMoeDsaForCausalLM",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=FPMCollectionOptions.from_args(
                _args(
                    fpm_gpu_counts=[1, 2],
                )
            ),
        )

    assert "fpm_forward: dropped 4/4 topologies" in caplog.text


def test_memory_admission_keeps_unknown_estimates_for_runtime_verification(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("model is not supported by the AIC memory estimator")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        unavailable,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert len(plan.topologies) == 3
    assert {decision.disposition for decision in plan.topology_memory_admission} == {"unknown"}


def test_memory_admission_drops_only_the_rejected_dtype_cells(monkeypatch, caplog):
    class Estimate:
        def __init__(self, *, admitted: bool):
            self.breakdown = {
                "non_kv_bytes": 50 if admitted else 150,
                "gpu_memory_capacity_bytes": 100,
            }

    def estimate(*_args, **kwargs):
        return Estimate(admitted=kwargs["kvcache_quant_mode"] == "fp8")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        estimate,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args(fpm_kv_cache_dtypes=["bfloat16", "fp8"])),
    )

    assert {cell.kv_cache_dtype for cell in plan.cells} == {"fp8"}
    assert {
        estimate.kv_cache_dtype
        for decision in plan.topology_memory_admission
        for estimate in decision.estimates
        if estimate.disposition == "rejected"
    } == {"bfloat16"}
    # Per-dtype drops must be counted, never silent (collector rules: the
    # memory filter's drops are logged; whole-topology logging alone would
    # hide these).
    assert "fpm_forward: dropped 3/6 (topology, kv_dtype) cell groups (memory budget" in caplog.text


def test_plan_identity_ignores_memory_estimator_error_text(monkeypatch):
    """Estimator failure diagnostics vary across runs and hosts (transient
    network errors, host paths); hashing them would spuriously invalidate
    resume for identical plans. Only dispositions belong to the identity."""

    kwargs = {
        "backend": "vllm",
        "model_path": "nvidia/GLM-5.2-NVFP4",
        "model_architecture": "GlmMoeDsaForCausalLM",
        "system": "b200_sxm",
        "selected_ops": {"dsa_context_module", "dsa_generation_module"},
        "options": FPMCollectionOptions.from_args(_args()),
    }

    def failing(message):
        def from_request(*_args, **_kwargs):
            raise RuntimeError(message)

        return from_request

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        failing("connection timed out to huggingface.co"),
    )
    first = build_collection_plan(**kwargs)
    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        failing("HTTP Error 503: Service Unavailable under /Users/someone"),
    )
    second = build_collection_plan(**kwargs)

    assert first.sha256 == second.sha256
    # The human-readable plan still carries the full diagnostic.
    assert "connection timed out to huggingface.co" in json.dumps(first.to_dict())


def test_git_revision_folds_dirty_tracked_state_into_identity(_pinned_git_revision, monkeypatch):
    """Uncommitted tracked edits change collector behavior without moving
    HEAD; the revision must distinguish them, resume identically for an
    unchanged tree, and ignore untracked noise (artifact dirs)."""

    from collector.fpm_forward import planner as fpm_planner

    real_git_revision = _pinned_git_revision
    outputs = {
        ("rev-parse", "HEAD"): "abc123\n",
        ("status", "--porcelain", "--untracked-files=no"): "",
        ("diff-index", "--no-ext-diff", "--full-index", "-p", "HEAD"): "",
    }

    def fake_run(args, **_kwargs):
        return SimpleNamespace(stdout=outputs[tuple(args[1:])], returncode=0)

    monkeypatch.setattr(fpm_planner.subprocess, "run", fake_run)

    assert real_git_revision() == "abc123"

    outputs[("status", "--porcelain", "--untracked-files=no")] = " M collector/fpm_forward/runner.py\n"
    outputs[("diff-index", "--no-ext-diff", "--full-index", "-p", "HEAD")] = "-a\n+b\n"
    dirty = real_git_revision()
    assert dirty.startswith("abc123-dirty-")
    assert real_git_revision() == dirty

    outputs[("diff-index", "--no-ext-diff", "--full-index", "-p", "HEAD")] = "-a\n+c\n"
    changed = real_git_revision()
    assert changed.startswith("abc123-dirty-")
    assert changed != dirty


def test_source_revision_override_precedes_git_and_installed_metadata(_pinned_git_revision, monkeypatch):
    monkeypatch.setenv("FPM_COLLECTOR_SOURCE_REVISION", "  release-candidate-17  ")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("explicit provenance must not inspect Git or installed metadata")

    monkeypatch.setattr(subprocess, "run", unexpected)
    monkeypatch.setattr(importlib.metadata, "distribution", unexpected)

    assert _pinned_git_revision() == "release-candidate-17"


def test_source_revision_prefers_content_addressed_installed_distribution_over_ambient_git(
    _pinned_git_revision, monkeypatch, tmp_path
):
    monkeypatch.delenv("FPM_COLLECTOR_SOURCE_REVISION", raising=False)
    distribution, root = _installed_distribution(tmp_path)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    from collector.fpm_forward import planner as fpm_planner

    monkeypatch.setattr(fpm_planner, "__file__", str(root / "collector/fpm_forward/planner.py"))

    def ambient_git(*_args, **_kwargs):
        return SimpleNamespace(stdout="ambient-host-head\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", ambient_git)

    first = _pinned_git_revision()
    second = _pinned_git_revision()

    assert first == second
    assert first.startswith("installed:aisimulate==0.12.0:record-sha256:")
    assert "/" not in first
    assert "ambient-host-head" not in first


def test_installed_source_revision_is_stable_across_record_order_quoting_and_installer_noise(
    _pinned_git_revision, monkeypatch, tmp_path
):
    from collector.fpm_forward import planner as fpm_planner

    quoted_path = "collector/fpm_forward/quoted,name.py"
    quoted_payload = b"quoted-content"
    base_rows = [
        (relative_path, _record_sha256(payload), str(len(payload)))
        for relative_path, payload in _REQUIRED_INSTALLED_FPM_PAYLOAD.items()
    ] + [(quoted_path, _record_sha256(quoted_payload), str(len(quoted_payload)))]
    distribution, root = _installed_distribution(tmp_path, rows=base_rows)
    quoted_file = root / quoted_path
    quoted_file.write_bytes(quoted_payload)
    monkeypatch.setattr(fpm_planner, "__file__", str(root / "collector/fpm_forward/planner.py"))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    first = _pinned_git_revision()

    noisy_rows = [
        ("../../../bin/aiconfigurator", "sha256=environment-specific", "123"),
        ("../../../bin/aisimulate", "sha256=environment-specific", "123"),
        ("../../../Scripts/aiconfigurator.exe", "sha256=environment-specific", "123"),
        ("../../../Scripts/aiconfigurator-script.py", "sha256=environment-specific", "123"),
        ("../../../Scripts/aisimulate.exe", "sha256=environment-specific", "123"),
        ("../../../Scripts/aisimulate-script.py", "sha256=environment-specific", "123"),
        ("aisimulate-0.12.0.dist-info/direct_url.json", "sha256=checkout-specific", "99"),
        *reversed(base_rows),
    ]
    noisy_distribution, other_root = _installed_distribution(tmp_path / "other-install", rows=noisy_rows)
    (other_root / quoted_path).write_bytes(quoted_payload)
    monkeypatch.setattr(fpm_planner, "__file__", str(other_root / "collector/fpm_forward/planner.py"))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: noisy_distribution)

    assert _pinned_git_revision() == first


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("tampered", "does not match RECORD"),
        ("missing_file", "missing or unreadable"),
        ("missing_version", "has no version"),
        ("missing_record", "has no RECORD"),
        ("metadata_only", "no content-addressed payload"),
        ("missing_required", "missing required payload rows"),
        ("malformed_row", "malformed row"),
        ("unhashed", "unhashed payload row"),
        ("malformed_hash", "malformed SHA-256"),
        ("malformed_size", "malformed size"),
        ("duplicate", "duplicates payload row"),
        ("traversal", "unsafe path"),
        ("unknown_console_script", "unsafe path"),
    ),
)
def test_installed_source_revision_rejects_untrusted_record_or_payload(
    _pinned_git_revision, monkeypatch, tmp_path, case, match
):
    rows = [
        (relative_path, _record_sha256(payload), str(len(payload)))
        for relative_path, payload in _REQUIRED_INSTALLED_FPM_PAYLOAD.items()
    ]
    version = "0.12.0"
    if case == "metadata_only":
        rows = [("aisimulate-0.12.0.dist-info/METADATA", _record_sha256(b"metadata"), "8")]
    elif case == "missing_required":
        rows.pop()
    elif case == "malformed_row":
        rows = [rows[0][0:2], *rows[1:]]
    elif case == "unhashed":
        rows[0] = (rows[0][0], "", rows[0][2])
    elif case == "malformed_hash":
        rows[0] = (rows[0][0], "md5=not-sha256", rows[0][2])
    elif case == "malformed_size":
        rows[0] = (rows[0][0], rows[0][1], "12 bytes")
    elif case == "duplicate":
        rows.append(rows[0])
    elif case == "traversal":
        rows.insert(0, ("collector/../escape.py", _record_sha256(b"escape"), "6"))
    elif case == "unknown_console_script":
        rows.insert(0, ("../../../bin/untrusted", _record_sha256(b"escape"), "6"))
    elif case == "missing_version":
        version = None

    distribution, root = _installed_distribution(tmp_path, version=version, rows=rows)
    if case == "tampered":
        (root / "collector/fpm_forward/planner.py").write_bytes(b"tampered")
    elif case == "missing_file":
        (root / "collector/fpm_forward/runner.py").unlink()
    elif case == "missing_record":
        monkeypatch.setattr(distribution, "read_text", lambda filename: None)

    from collector.fpm_forward import planner as fpm_planner

    monkeypatch.setattr(fpm_planner, "__file__", str(root / "collector/fpm_forward/planner.py"))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="ambient-host-head\n", returncode=0),
    )

    with pytest.raises(ValueError, match=match):
        _pinned_git_revision()


def test_source_checkout_uses_git_even_when_editable_distribution_metadata_exists(
    _pinned_git_revision, monkeypatch, tmp_path
):
    monkeypatch.delenv("FPM_COLLECTOR_SOURCE_REVISION", raising=False)
    distribution, _ = _installed_distribution(tmp_path)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    outputs = {
        ("rev-parse", "HEAD"): "source-head\n",
        ("status", "--porcelain", "--untracked-files=no"): "",
    }

    def source_git(args, **_kwargs):
        return SimpleNamespace(stdout=outputs[tuple(args[1:])], returncode=0)

    monkeypatch.setattr(subprocess, "run", source_git)

    assert _pinned_git_revision() == "source-head"


def test_source_revision_fails_when_no_explicit_git_or_installed_identity(_pinned_git_revision, monkeypatch):
    monkeypatch.delenv("FPM_COLLECTOR_SOURCE_REVISION", raising=False)
    metadata_attempted = False

    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    def missing_distribution(name):
        nonlocal metadata_attempted
        metadata_attempted = True
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(subprocess, "run", missing_git)
    monkeypatch.setattr(importlib.metadata, "distribution", missing_distribution)

    with pytest.raises(ValueError, match="requires the collector source revision"):
        _pinned_git_revision()

    assert metadata_attempted is True


def test_minimax_m3_keeps_family_dtype_and_parallel_capabilities():
    plan = build_collection_plan(
        backend="vllm",
        model_path="MiniMaxAI/MiniMax-M3",
        model_architecture="MiniMaxM3ForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_max_gpus=16,
                fpm_gpu_counts=[8, 16],
            )
        ),
    )

    assert plan.capability.support_level == "family_template"
    assert plan.capability.template_id == "aic_family:minimaxm3:moe_msa"
    assert plan.capability.attention_kind == "moe_msa"
    assert plan.capability.attention_source == "dsa_module"
    assert plan.capability.allow_pure_tp is True
    assert {cell.parallel_strategy for cell in plan.cells} == {"pure_tp", "tep", "dep"}


_DSV4_ATTENTION_OPS = {
    "dsv4_csa_context_module",
    "dsv4_hca_context_module",
    "dsv4_csa_generation_module",
    "dsv4_hca_generation_module",
}


@pytest.mark.parametrize(
    ("model_path", "max_prefill_tokens", "expected_strategies", "expected_memory_rejections"),
    [
        # Residual-width MoE workspace fits Pro DEP at 8192 tokens; a larger
        # activation envelope must still reject it on physical capacity.
        ("sgl-project/DeepSeek-V4-Pro-FP8", 8192, {"pure_tp", "tep", "dep"}, 0),
        ("sgl-project/DeepSeek-V4-Pro-FP8", 32768, {"pure_tp", "tep"}, 1),
        ("sgl-project/DeepSeek-V4-Flash-FP8", 8192, {"pure_tp", "tep", "dep"}, 0),
    ],
)
def test_dsv4_fp8_keeps_exact_capabilities_and_applies_max_new_token_memory_admission(
    model_path,
    max_prefill_tokens,
    expected_strategies,
    expected_memory_rejections,
):
    plan = build_collection_plan(
        backend="vllm",
        model_path=model_path,
        model_architecture="DeepseekV4ForCausalLM",
        system="b200_sxm",
        selected_ops=_DSV4_ATTENTION_OPS,
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_max_gpus=16,
                fpm_gpu_counts=[16],
                fpm_max_prefill_isl=max_prefill_tokens,
            )
        ),
    )

    assert plan.capability.support_level == "exact"
    assert plan.capability.template_id == "aic_exact:dsv4_module"
    assert plan.capability.attention_kind == "moe_dsv4"
    assert plan.capability.attention_source == "dsv4_module"
    assert plan.capability.allow_pure_tp is True
    assert plan.dtype_profile.fmha_quant_mode == "bfloat16"
    assert plan.dtype_profile.kv_cache_dtypes == ("fp8",)
    assert {cell.parallel_strategy for cell in plan.cells} == expected_strategies
    assert plan.to_dict()["counts"]["memory_rejected_topologies"] == expected_memory_rejections
    assert plan.to_dict()["counts"]["memory_unknown_topologies"] == 0
    for decision in plan.topology_memory_admission:
        assert decision.max_new_tokens == max_prefill_tokens
        for estimate in decision.estimates:
            assert (estimate.estimated_non_kv_bytes < estimate.gpu_capacity_bytes) == (
                decision.disposition == "admitted"
            )
    assert all(cell.to_dict()["point_source"] == "dynamo_native_self_benchmark" for cell in plan.cells)


@pytest.mark.parametrize(
    "model_path",
    [
        "deepseek-ai/DeepSeek-V4-Pro",
        "deepseek-ai/DeepSeek-V4-Flash",
    ],
)
def test_dsv4_native_fp4_uses_vllm_sm100_dtype_capability(model_path):
    plan = build_collection_plan(
        backend="vllm",
        model_path=model_path,
        model_architecture="DeepseekV4ForCausalLM",
        system="b200_sxm",
        selected_ops=_DSV4_ATTENTION_OPS,
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_max_gpus=16,
                fpm_gpu_counts=[16],
            )
        ),
    )

    assert plan.capability.support_level == "exact"
    assert plan.capability.template_id == "aic_exact:dsv4_module"
    assert plan.capability.attention_kind == "moe_dsv4"
    assert plan.capability.allow_pure_tp is True
    assert plan.dtype_profile.gemm_quant_mode == "fp8_block"
    assert plan.dtype_profile.moe_quant_mode == "w4a8_mxfp4_mxfp8"
    assert plan.dtype_profile.fmha_quant_mode == "bfloat16"
    assert plan.dtype_profile.kv_cache_dtypes == ("fp8",)
    assert {cell.parallel_strategy for cell in plan.cells} == {"pure_tp", "tep", "dep"}
    assert all(cell.to_dict()["point_source"] == "dynamo_native_self_benchmark" for cell in plan.cells)


def test_unknown_model_keeps_auditable_capability_template(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["NewMoeDsaForCausalLM"],
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "hidden_size": 4096,
                "intermediate_size": 8192,
                "num_hidden_layers": 4,
                "vocab_size": 32000,
                "n_routed_experts": 8,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "max_position_embeddings": 4096,
            }
        )
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path=str(tmp_path),
        model_architecture="NewMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.support_level == "bootstrap_template"
    assert plan.capability.template_id == "generic:moe_dsa"
    assert {cell.parallel_strategy for cell in plan.cells} == {"dep", "pure_tp", "tep"}


def _unknown_fp8_moe_config(*, hidden_size: int = 4096) -> dict[str, object]:
    return {
        "architectures": ["NewMoeDsaForCausalLM"],
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "hidden_size": hidden_size,
        "intermediate_size": 8192,
        "num_hidden_layers": 4,
        "vocab_size": 32000,
        "n_routed_experts": 8,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "max_position_embeddings": 4096,
    }


def test_explicit_real_config_keeps_unregistered_fp8_moe_bootstrap(monkeypatch, tmp_path):
    config_dir = tmp_path / "private-model-config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(json.dumps(_unknown_fp8_moe_config()))
    (config_dir / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}})
    )

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("the checkpoint is accessible only inside the runtime Pod")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        unavailable,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="private-org/runtime-only-model",
        model_config_path=str(config_dir),
        model_architecture="NewMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(_args()),
    )

    evidence = plan.capability.model_config
    assert evidence.source_kind == "explicit"
    assert len(evidence.sha256) == 64
    assert evidence.payload["hf_quant_config"]["quantization"]["quant_algo"] == "FP8"
    detached = evidence.payload
    detached["architectures"] = ["MutatedForCausalLM"]
    assert evidence.payload["architectures"] == ["NewMoeDsaForCausalLM"]
    assert plan.capability.support_level == "bootstrap_template"
    assert plan.capability.is_moe is True
    assert plan.dtype_profile.gemm_quant_mode == "fp8_static"
    assert plan.dtype_profile.moe_quant_mode == "fp8"
    assert plan.dtype_profile.kv_cache_dtypes == ("fp8",)


def test_huggingface_config_is_used_when_model_is_not_in_aic_cache(monkeypatch):
    calls = []

    def download(model_path, filename, *, raise_on_404):
        calls.append((model_path, filename, raise_on_404))
        if filename == "config.json":
            return _unknown_fp8_moe_config()
        return None

    monkeypatch.setattr("collector.fpm_forward.model_capability._download_hf_json", download)

    resolved = load_model_config("new-org/New-MoE-Model")

    assert resolved.source_kind == "huggingface"
    assert resolved.payload["architectures"] == ["NewMoeDsaForCausalLM"]
    assert calls == [
        ("new-org/New-MoE-Model", "config.json", True),
        ("new-org/New-MoE-Model", "hf_quant_config.json", False),
    ]


def test_unresolvable_model_config_fails_loudly(monkeypatch):
    def missing(*_args, **_kwargs):
        raise HuggingFaceDownloadError("Hugging Face returned HTTP error 404")

    monkeypatch.setattr("collector.fpm_forward.model_capability._download_hf_json", missing)

    with pytest.raises(ValueError, match="cannot resolve a real model config"):
        build_collection_plan(
            backend="vllm",
            model_path="missing-org/does-not-exist",
            model_architecture="MissingForCausalLM",
            system="b200_sxm",
            selected_ops={"attention_context", "attention_generation"},
            has_model_cases=False,
            options=FPMCollectionOptions.from_args(_args()),
        )


def test_empty_explicit_model_config_is_rejected(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}\n")

    with pytest.raises(ValueError, match="must not be empty"):
        load_model_config("private-org/model", explicit_config_path=str(config))


def test_empty_base_config_with_sibling_quant_config_is_rejected(tmp_path):
    """The emptiness gate must fire BEFORE the quantization merge: a sibling
    hf_quant_config.json would otherwise make an empty config.json non-empty
    and admit a contentless model identity into the frozen plan."""

    (tmp_path / "config.json").write_text("{}\n")
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"}})
    )

    with pytest.raises(ValueError, match="must not be empty"):
        load_model_config(str(tmp_path))


def test_empty_huggingface_config_with_quant_config_is_rejected(monkeypatch):
    def download(_model_path, filename, *, raise_on_404):
        if filename == "config.json":
            return {}
        return {"quantization": {"quant_algo": "FP8"}}

    monkeypatch.setattr("collector.fpm_forward.model_capability._download_hf_json", download)

    with pytest.raises(ValueError, match="must not be empty"):
        load_model_config("new-org/empty-config-model")


def test_empty_kv_dtype_request_fails_before_any_resolution():
    """The CLI maps an empty --fpm-kv-cache-dtypes to ("auto",); a direct
    caller with an empty tuple must get an actionable error instead of a
    bare IndexError after config resolution."""

    with pytest.raises(ValueError, match="at least one KV-cache dtype"):
        resolve_model_capability(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            model_architecture="GlmMoeDsaForCausalLM",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            has_model_cases=True,
            system="b200_sxm",
            requested_weight_quantizations=(),
            requested_kv_cache_dtypes=(),
        )


def test_memory_admission_fails_loudly_on_capability_invariant_violation():
    """A kv dtype without an fmha mapping violates resolve_model_capability's
    invariant; it must raise instead of being recorded as a fail-open
    'unknown' memory-estimate outcome."""

    capability = SimpleNamespace(
        aic_database_version="0.24.0",
        dtype=SimpleNamespace(
            gemm_quant_mode="nvfp4",
            moe_quant_mode="nvfp4",
            comm_quant_mode="half",
            kv_cache_dtypes=("fp8",),
            fmha_by_kv_dtype={},
        ),
    )

    with pytest.raises(ValueError, match="no fmha mapping"):
        filter_memory_infeasible_topologies(
            backend="vllm",
            model_path="org/model",
            system="b200_sxm",
            capability=capability,
            topologies=(ParallelTopology(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1),),
            max_new_tokens=8192,
        )


def test_model_config_content_changes_the_frozen_plan_hash(monkeypatch, tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    same_as_first = tmp_path / "same-as-first.json"
    first.write_text(json.dumps(_unknown_fp8_moe_config(hidden_size=4096)))
    second.write_text(json.dumps(_unknown_fp8_moe_config(hidden_size=5120)))
    same_as_first.write_text(json.dumps(_unknown_fp8_moe_config(hidden_size=4096)))

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("memory estimate intentionally unavailable")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        unavailable,
    )
    common = {
        "backend": "vllm",
        "model_path": "private-org/runtime-only-model",
        "model_architecture": "NewMoeDsaForCausalLM",
        "system": "b200_sxm",
        "selected_ops": {"attention_context", "attention_generation"},
        "has_model_cases": False,
        "options": FPMCollectionOptions.from_args(_args()),
    }

    first_plan = build_collection_plan(**common, model_config_path=str(first))
    second_plan = build_collection_plan(**common, model_config_path=str(second))
    same_content_plan = build_collection_plan(**common, model_config_path=str(same_as_first))

    assert first_plan.capability.model_config.sha256 != second_plan.capability.model_config.sha256
    assert first_plan.sha256 != second_plan.sha256
    assert first_plan.sha256 == same_content_plan.sha256


def test_unregistered_dense_model_keeps_gqa_bootstrap_template(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Phi3ForCausalLM"],
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "num_hidden_layers": 32,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
                "vocab_size": 200064,
                "max_position_embeddings": 131072,
                "torch_dtype": "bfloat16",
            }
        )
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path=str(tmp_path),
        model_architecture="Phi3ForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.model_family is None
    assert plan.capability.support_level == "bootstrap_template"
    assert plan.capability.template_id == "generic:dense_gqa"
    assert plan.capability.attention_source == "dense_attention"
    assert plan.capability.allow_pure_tp is False
    assert {cell.parallel_strategy for cell in plan.cells} == {"tp"}


def test_registered_dense_model_without_case_file_keeps_family_template():
    plan = build_collection_plan(
        backend="vllm",
        model_path="Qwen/Qwen3-32B",
        model_architecture="Qwen3ForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=False,
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.support_level == "family_template"
    assert plan.capability.template_id == "aic_family:llama:dense_gqa"
    assert plan.capability.attention_source == "dense_attention"
    assert {cell.parallel_strategy for cell in plan.cells} == {"tp"}


def test_exact_dense_mla_model_does_not_become_moe(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DeepSeekForCausalLM"],
                "hidden_size": 2048,
                "intermediate_size": 8192,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "kv_lora_rank": 512,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "torch_dtype": "bfloat16",
            }
        )
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path=str(tmp_path),
        model_architecture="DeepSeekForCausalLM",
        system="b200_sxm",
        selected_ops={"mla_context_module", "mla_generation_module"},
        has_model_cases=True,
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert plan.capability.support_level == "exact"
    assert plan.capability.attention_kind == "dense_mla"
    assert plan.capability.is_moe is False
    assert {cell.parallel_strategy for cell in plan.cells} == {"tp"}


def test_explicit_kv_dtype_still_requires_aic_runtime_capability():
    options = FPMCollectionOptions.from_args(_args(fpm_kv_cache_dtypes=["int8"]))
    with pytest.raises(ValueError, match="does not support KV-cache dtype"):
        build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=options,
        )


def test_arbitrary_backend_variant_is_not_a_capability_declaration():
    with pytest.raises(ValueError, match="no longer an admission mechanism"):
        build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=FPMCollectionOptions.from_args(_args()),
            collector_config={"backend_variants": {"moe": [{"id": "invented"}]}},
        )


def _synthetic_plan_and_cell(tmp_path):
    topology = ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1)
    cell = FPMCell(
        cell_id="fpm-test",
        workload_kind="prefill",
        topology=topology,
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline_auto", {}, {}),
        parallel_strategy="dep",
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )
    plan = SimpleNamespace(
        sha256="plan-sha",
        aic_revision="revision",
        model_path="org/model",
        system="b200_sxm",
        backend="vllm",
        cells=(cell,),
        options=SimpleNamespace(warmup_iterations=0),
        capability=SimpleNamespace(
            support_level="exact",
            template_id="aic_exact:dsa_module",
            template_version=1,
            aic_database_version="0.24.0",
        ),
    )
    point = {
        "point_type": "prefill",
        "benchmark_id": 1,
        "total_prefill_tokens": 257,
        "total_kv_read_tokens": 128,
        "batch_size": 4,
        "expected_cudagraph_mode": "PIECEWISE",
        "expected_capture_size": 272,
        "padding_tokens": 15,
        "sample_reasons": ["post_capture"],
    }
    cell_dir = tmp_path / "cell"
    rank_fpms = []
    for rank, latency in ((0, 0.004), (1, 0.006)):
        rank_fpms.append(
            {
                "counter_id": 1,
                "dp_rank": rank,
                "wall_time": latency,
                "scheduled_requests": {
                    "num_prefill_requests": 4,
                    "sum_prefill_tokens": 257,
                    "sum_prefill_kv_tokens": 128,
                    "num_decode_requests": 0,
                    "sum_decode_kv_tokens": 0,
                },
            }
        )
    iteration_group = {
        "benchmark_id": 1,
        "point": point,
        "expected_dp_ranks": [0, 1],
        "complete": True,
        "wall_time": 0.006,
        "rank_results": [{"dp_rank": rank, "fpms": [fpm]} for rank, fpm in enumerate(rank_fpms)],
    }
    for rank, fpm in enumerate(rank_fpms):
        output = cell_dir / "raw" / f"pod-{rank}" / ("benchmark.json" if rank == 0 else f"benchmark_dp{rank}.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_provenance(output.parent / "collector-provenance.json", cell_id=cell.cell_id)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "artifact_type": "rank",
                    "status": "complete",
                    "valid": True,
                    "usable": True,
                    "timing_valid": True,
                    "stop_reason": None,
                    "error": None,
                    "run_id": "run",
                    "grid_digest": "grid",
                    "config": {"mode": "prefill"},
                    "coverage": {"expected_points": 1, "completed_points": 1, "skipped_points": 0},
                    "dp": {"rank": rank, "size": 2},
                    "results": [{"point": point, "fpms": [fpm]}],
                    "iteration_groups": [iteration_group],
                    "skipped_points": [],
                    "missing_phases": [],
                    "timing": {
                        "benchmark_elapsed_seconds": 1.0 + rank,
                        "measured_iteration_seconds": 0.006,
                    },
                }
            )
        )
    return plan, cell, cell_dir


def test_native_aggregation_preserves_iteration_totals(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert len(rows) == 1
    assert rows[0]["latency_ms"] == pytest.approx(6.0)
    assert rows[0]["batch_size"] == 4
    assert rows[0]["total_prefill_tokens"] == 257
    assert rows[0]["total_kv_read_tokens"] == 128
    assert rows[0]["partition_policy"] == "balanced_v1"
    assert rows[0]["measurement_policy"] == "dynamo_native_single_sample_v1"
    assert rows[0]["backend_version"] == "0.24.0"
    assert rows[0]["collector_attempt_id"] == "attempt"
    assert rows[0]["runtime_run_id"] == "run"
    assert rows[0]["runtime_grid_digest"] == "grid"
    assert "suffix_length" not in rows[0]
    assert "prefix_length" not in rows[0]


def test_native_aggregation_rejects_rank_grid_drift(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    second_rank = next((cell_dir / "raw" / "pod-1").glob("benchmark*.json"))
    payload = json.loads(second_rank.read_text())
    payload["grid_digest"] = "different"
    second_rank.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="different run identities"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_native_aggregation_rejects_stale_collector_attempt(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    for provenance in (cell_dir / "raw").glob("*/collector-provenance.json"):
        payload = json.loads(provenance.read_text())
        payload["attempt_id"] = "stale-attempt"
        provenance.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="attempt mismatch"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_native_aggregation_requires_attempt_identity(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)

    with pytest.raises(ValueError, match="without an expected Collector attempt"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="")


@pytest.mark.parametrize("bad_value", [4.0, True, "4"], ids=["float", "bool", "str"])
def test_native_validation_rejects_non_integer_point_dimensions(tmp_path, bad_value):
    """Native grid coordinates are integers by contract; a float/bool/str
    dimension means a noncompliant engine or tampered artifact and must fail
    validation instead of being silently truncated into a database row."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    artifact = cell_dir / "raw" / "pod-0" / "benchmark.json"
    payload = json.loads(artifact.read_text())
    payload["results"][0]["point"]["batch_size"] = bad_value
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="'batch_size' must be an integer"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_native_validation_rejects_sub_batch_token_totals(tmp_path):
    """Every scheduled request contributes at least one prefill token (prefill)
    or reads at least one KV token (decode), so totals below the request count
    are impossible coordinates and must fail instead of becoming database rows."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    artifact = cell_dir / "raw" / "pod-0" / "benchmark.json"
    payload = json.loads(artifact.read_text())
    payload["results"][0]["point"]["total_prefill_tokens"] = 3
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="fewer tokens than requests"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    # The decode floor cannot be reached through the synthetic prefill cell
    # (workload-kind contract fires first), so exercise the guard directly.
    from collector.fpm_forward.native_artifact import _expected_scheduled

    decode_point = {
        "point_type": "decode",
        "batch_size": 4,
        "total_prefill_tokens": 0,
        "total_kv_read_tokens": 2,
    }
    with pytest.raises(ValueError, match="fewer KV tokens than requests"):
        _expected_scheduled(decode_point)


def test_formal_database_uses_schema_v7_and_rejects_conflicts(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    parquet, metadata, skipped = write_formal_database(plan, rows, systems_root=tmp_path / "systems")
    write_formal_database(plan, rows, systems_root=tmp_path / "systems")

    assert parquet.exists()
    metadata_payload = json.loads(metadata.read_text())
    assert metadata_payload["schema_version"] == 7
    assert metadata_payload["coordinate_system"] == "iteration_totals_balanced_v1"
    assert metadata_payload["backend_version"] == "0.24.0"
    assert metadata_payload["collector_attempt_ids"] == ["attempt"]
    assert metadata_payload["runtime_run_ids"] == ["run"]
    assert metadata_payload["runtime_grid_digests"] == ["grid"]
    assert parquet.parent.name == "0.24.0"

    conflicting = [{**rows[0], "latency_ms": 7.0}]
    with pytest.raises(ValueError, match="conflicting"):
        write_formal_database(plan, conflicting, systems_root=tmp_path / "systems")


def test_formal_database_first_publisher_wins_on_rerun_overlap(tmp_path):
    """A cell republished under a different run identity is skipped whole
    (first publisher wins, sealed rows never mixed or overwritten) while the
    database file stays byte-stable and the skip is reported to the caller."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    systems_root = tmp_path / "systems"
    parquet, _metadata, first_skipped = write_formal_database(plan, rows, systems_root=systems_root)
    assert first_skipped == ()
    sealed = parquet.read_bytes()

    rerun_rows = [
        {
            **rows[0],
            "total_prefill_tokens": rows[0]["total_prefill_tokens"] + 1,
            "latency_ms": 999.0,
            "collector_attempt_id": "different-attempt",
            "runtime_run_id": "different-run",
        }
    ]
    parquet2, _metadata2, skipped = write_formal_database(plan, rerun_rows, systems_root=systems_root)

    assert skipped == (rows[0]["cell_id"],)
    assert parquet2.read_bytes() == sealed


def test_formal_database_commit_validation_accepts_sealed_schema_v7_pair(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    parquet, metadata, _skipped = write_formal_database(plan, rows, systems_root=tmp_path / "systems")

    commit = validate_formal_database_commit(parquet, metadata, plan)

    assert commit["schema_version"] == 7
    assert commit["row_count"] == len(rows)


def test_formal_database_commit_validation_rejects_digest_drift(tmp_path):
    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    parquet, metadata, _skipped = write_formal_database(plan, rows, systems_root=tmp_path / "systems")
    parquet.write_bytes(parquet.read_bytes() + b"drift")

    with pytest.raises(ValueError, match="does not match its commit record"):
        validate_formal_database_commit(parquet, metadata, plan)


def _synthetic_decode_cell(
    tmp_path,
    *,
    kvwarm,
    markers=("kvwarm_real_kv", "kvwarm_fake_fallback"),
    parallel_strategy="tep",
):
    """Single-rank decode cell with one point per marker entry (None = no
    kvwarm marker in sample_reasons); ``kvwarm`` is the artifact's top-level
    engine envelope (None = legacy artifact without the block)."""

    topology = ParallelTopology(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1)
    cell = FPMCell(
        cell_id="fpm-test-decode",
        workload_kind="decode",
        topology=topology,
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline_auto", {}, {}),
        parallel_strategy=parallel_strategy,
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )
    plan = SimpleNamespace(
        sha256="plan-sha",
        aic_revision="revision",
        model_path="org/model",
        system="b200_sxm",
        backend="vllm",
        options=SimpleNamespace(warmup_iterations=0),
        capability=SimpleNamespace(
            support_level="exact",
            template_id="aic_exact:dsa_module",
            template_version=1,
            aic_database_version="0.24.0",
        ),
    )
    rows = []
    groups = []
    for index, marker in enumerate(markers, start=1):
        point = {
            "point_type": "decode",
            "benchmark_id": index,
            "total_prefill_tokens": 0,
            "total_kv_read_tokens": 128 * index,
            "batch_size": 4,
            "expected_cudagraph_mode": "FULL",
            "expected_capture_size": 4,
            "padding_tokens": 0,
            "sample_reasons": ["capture"] + ([marker] if marker else []),
        }
        fpm = {
            "counter_id": index,
            "dp_rank": 0,
            "wall_time": 0.01 * index,
            "scheduled_requests": {
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "sum_prefill_kv_tokens": 0,
                "num_decode_requests": 4,
                "sum_decode_kv_tokens": 128 * index,
            },
        }
        rows.append({"point": point, "fpms": [fpm]})
        groups.append(
            {
                "benchmark_id": index,
                "point": point,
                "expected_dp_ranks": [0],
                "complete": True,
                "wall_time": fpm["wall_time"],
                "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
            }
        )
    measured = sum(0.01 * index for index in range(1, len(markers) + 1))
    payload = {
        "schema_version": 2,
        "artifact_type": "rank",
        "status": "complete",
        "valid": True,
        "usable": True,
        "timing_valid": True,
        "stop_reason": None,
        "error": None,
        "run_id": "run",
        "grid_digest": "grid",
        "config": {"mode": "decode"},
        "coverage": {"expected_points": len(markers), "completed_points": len(markers), "skipped_points": 0},
        "dp": {"rank": 0, "size": 1},
        "results": rows,
        "iteration_groups": groups,
        "skipped_points": [],
        "missing_phases": [],
        "timing": {"benchmark_elapsed_seconds": measured + 1.0, "measured_iteration_seconds": measured},
    }
    if kvwarm is not None:
        payload["kvwarm"] = kvwarm
    cell_dir = tmp_path / "cell"
    output = cell_dir / "raw" / "pod-0" / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_provenance(output.parent / "collector-provenance.json", cell_id=cell.cell_id)
    output.write_text(json.dumps(payload))
    return plan, cell, cell_dir


def test_kv_seed_regime_derives_from_point_markers_in_warm_cells(tmp_path):
    """A warm-eligible cell records the per-point protocol: real_kv for
    warm-chain points, fake_fallback for points the chain could not reach."""

    plan, cell, cell_dir = _synthetic_decode_cell(
        tmp_path,
        kvwarm={"enabled": True, "warm_eligible": True, "skip_reason": None},
    )
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    regimes = [row["kv_seed_regime"] for row in sorted(rows, key=lambda row: row["total_kv_read_tokens"])]
    assert regimes == ["real_kv", "fake_fallback"]


def test_kv_seed_regime_cell_skip_reason_overrides_point_markers(tmp_path):
    """Warm-ineligible topologies mark EVERY point kvwarm_fake_fallback
    (measured tp4: 1659/1659), so the cell-level skip_reason must win --
    otherwise a downstream fake_fallback filter would wipe whole legitimate
    cells."""

    plan, cell, cell_dir = _synthetic_decode_cell(
        tmp_path,
        kvwarm={
            "enabled": True,
            "warm_eligible": False,
            "skip_reason": "moe_tp_balanced_by_construction",
        },
        markers=("kvwarm_fake_fallback", "kvwarm_fake_fallback"),
        parallel_strategy="tp",
    )
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert {row["kv_seed_regime"] for row in rows} == {"skip:moe_tp_balanced_by_construction"}


def test_kv_seed_regime_is_legacy_without_kvwarm_block_and_na_for_prefill(tmp_path):
    """Artifacts predating the kvwarm runtime carry no envelope -> legacy;
    prefill rows are out of the decode seeding protocol entirely -> n/a."""

    plan, cell, cell_dir = _synthetic_decode_cell(
        tmp_path,
        kvwarm=None,
        markers=(None, None),
        parallel_strategy="tp",
    )
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    assert {row["kv_seed_regime"] for row in rows} == {"legacy"}

    prefill_plan, prefill_cell, prefill_dir = _synthetic_plan_and_cell(tmp_path / "prefill")
    prefill_rows = aggregate_cell(prefill_plan, prefill_cell, prefill_dir, expected_attempt_id="attempt")
    assert {row["kv_seed_regime"] for row in prefill_rows} == {"n/a"}


@pytest.mark.parametrize(
    "kvwarm",
    [None, {"enabled": True, "warm_eligible": False, "skip_reason": "prefix_caching_disabled"}],
)
def test_native_validation_rejects_pure_tp_without_real_kv_warmup(tmp_path, kvwarm):
    """pure_tp is a warm-required strategy; neither a legacy artifact nor a
    warm-ineligible result may be published for that rendered protocol."""

    plan, cell, cell_dir = _synthetic_decode_cell(
        tmp_path,
        kvwarm=kvwarm,
        markers=("kvwarm_fake_fallback", "kvwarm_fake_fallback"),
        parallel_strategy="pure_tp",
    )

    with pytest.raises(ValueError, match=r"required KV warm-up metadata|protocol mismatch"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


@pytest.mark.parametrize("skip_reason", ["prefix_caching_disabled", "unknown_architecture"])
def test_non_protocol_kvwarm_skips_remain_fake_fallback(tmp_path, skip_reason):
    plan, cell, cell_dir = _synthetic_decode_cell(
        tmp_path,
        kvwarm={"enabled": True, "warm_eligible": False, "skip_reason": skip_reason},
        markers=("kvwarm_fake_fallback", "kvwarm_fake_fallback"),
        parallel_strategy="tp",
    )

    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert {row["kv_seed_regime"] for row in rows} == {"fake_fallback"}


def test_native_validation_rejects_cross_rank_kvwarm_disagreement(tmp_path):
    """The kvwarm regime describes one shared measurement protocol; DP ranks
    reporting different warm_eligible/skip_reason facts are a contract
    violation, not a mergeable difference."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    for rank, block in enumerate(
        (
            {"enabled": True, "warm_eligible": True, "skip_reason": None},
            {"enabled": True, "warm_eligible": False, "skip_reason": "moe_tp_balanced_by_construction"},
        )
    ):
        artifact = cell_dir / "raw" / f"pod-{rank}" / ("benchmark.json" if rank == 0 else f"benchmark_dp{rank}.json")
        payload = json.loads(artifact.read_text())
        payload["kvwarm"] = block
        artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="disagree on the KV warm-up regime"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_formal_database_merges_rows_without_kv_seed_regime_as_null(tmp_path):
    """kv_seed_regime is additive (not in the row key): merging onto a
    parquet written before the column existed keeps the old rows (null) and
    lands the new rows' values, with the row count intact."""

    import pyarrow.parquet as pq

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    old_rows = [
        {key: value for key, value in row.items() if key != "kv_seed_regime"}
        for row in aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    ]
    systems_root = tmp_path / "systems"
    write_formal_database(plan, old_rows, systems_root=systems_root)

    decode_plan, decode_cell, decode_dir = _synthetic_decode_cell(
        tmp_path / "decode",
        kvwarm={"enabled": True, "warm_eligible": True, "skip_reason": None},
    )
    decode_plan.sha256 = plan.sha256
    new_rows = aggregate_cell(decode_plan, decode_cell, decode_dir, expected_attempt_id="attempt")
    parquet, _metadata, skipped = write_formal_database(decode_plan, new_rows, systems_root=systems_root)

    assert skipped == ()
    published = pq.read_table(parquet).to_pylist()
    assert len(published) == len(old_rows) + len(new_rows)
    by_cell = {row["cell_id"]: row["kv_seed_regime"] for row in published}
    assert by_cell["fpm-test"] is None
    assert by_cell["fpm-test-decode"] in {"real_kv", "fake_fallback"}


def test_formal_database_mixed_overlap_skips_sealed_cell_and_lands_the_rest(tmp_path):
    """A plan overlapping one sealed cell publishes its fresh cells while the
    sealed cell's rows stay first-publisher-owned: partial skip, rest lands."""

    import pyarrow.parquet as pq

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    systems_root = tmp_path / "systems"
    write_formal_database(plan, rows, systems_root=systems_root)

    overlapping = {
        **rows[0],
        "latency_ms": 123.0,
        "collector_attempt_id": "attempt-2",
        "runtime_run_id": "run-2",
    }
    fresh = {
        **rows[0],
        "cell_id": "fpm-test-other",
        "batch_size": rows[0]["batch_size"] + 4,
        "collector_attempt_id": "attempt-2",
        "runtime_run_id": "run-2",
    }
    parquet, _metadata, skipped = write_formal_database(plan, [overlapping, fresh], systems_root=systems_root)

    assert skipped == (rows[0]["cell_id"],)
    published = pq.read_table(parquet).to_pylist()
    assert {row["cell_id"] for row in published} == {rows[0]["cell_id"], "fpm-test-other"}
    sealed_rows = [row for row in published if row["cell_id"] == rows[0]["cell_id"]]
    assert {row["collector_attempt_id"] for row in sealed_rows} == {"attempt"}
    fresh_rows = [row for row in published if row["cell_id"] == "fpm-test-other"]
    assert {row["collector_attempt_id"] for row in fresh_rows} == {"attempt-2"}


def test_formal_database_refuses_parquet_without_commit_record(tmp_path):
    """Published rows are sha-sealed evidence: a parquet without its metadata
    commit record could be a partial write or a foreign file, so the merge
    must refuse instead of building on unvouched bytes."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    systems_root = tmp_path / "systems"
    _parquet, metadata, _skipped = write_formal_database(plan, rows, systems_root=systems_root)

    metadata.unlink()
    with pytest.raises(ValueError, match="no commit record"):
        write_formal_database(plan, rows, systems_root=systems_root)


def test_formal_database_refuses_parquet_that_mismatches_commit_record(tmp_path):
    """A parquet whose bytes disagree with the recorded sha256 (manual edit,
    torn write) must abort the merge instead of silently absorbing it."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    systems_root = tmp_path / "systems"
    _parquet, metadata, _skipped = write_formal_database(plan, rows, systems_root=systems_root)

    committed = json.loads(metadata.read_text())
    committed["parquet_sha256"] = "0" * 64
    metadata.write_text(json.dumps(committed))
    with pytest.raises(ValueError, match="does not match its commit record"):
        write_formal_database(plan, rows, systems_root=systems_root)


def test_backend_marker_validation_tolerates_native_json_types(tmp_path):
    """Markers are declared as strings ("True") while the resolved config
    stores native JSON types (true); validation compares canonical string
    forms, so the type gap is not a mismatch but a real value gap still is."""

    from collector.fpm_forward.database import _validate_backend_markers

    cell = FPMCell(
        cell_id="fpm-test",
        workload_kind="prefill",
        topology=ParallelTopology(tp=1, pp=1, dp=2, moe_tp=1, moe_ep=2, cp=1),
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("eplb_pinned", {}, {"config.engine_args.enable_eplb": "True"}),
        parallel_strategy="dep",
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )
    cell_dir = tmp_path / "cell"
    config_path = cell_dir / "raw" / "pod-0" / "resolved-config-node0.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"config": {"engine_args": {"enable_eplb": True}}}))

    _validate_backend_markers(cell, cell_dir)

    config_path.write_text(json.dumps({"config": {"engine_args": {"enable_eplb": False}}}))
    with pytest.raises(ValueError, match="backend marker mismatch"):
        _validate_backend_markers(cell, cell_dir)


def test_formal_database_merge_gate_names_missing_row_key_columns(tmp_path):
    """An existing parquet that satisfies the run-identity columns but lacks a
    _ROW_KEY column must be rejected with the actionable schema ValueError,
    never a bare KeyError from the merge index."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    destination = tmp_path / "systems" / "b200_sxm" / "vllm" / "0.24.0"
    destination.mkdir(parents=True)
    stale_rows = [{key: value for key, value in rows[0].items() if key != "weight_quantization"}]
    parquet_path = destination / "fpm_forward_perf.parquet"
    pq.write_table(pa.Table.from_pylist(stale_rows), parquet_path)
    (destination / "fpm_forward_perf.metadata.json").write_text(
        json.dumps({"schema_version": 7, "parquet_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest()})
    )

    with pytest.raises(ValueError, match=r"missing columns: \['weight_quantization'\]"):
        write_formal_database(plan, rows, systems_root=tmp_path / "systems")


def test_formal_database_serializes_concurrent_publishers(tmp_path):
    plan, _cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    base_row = aggregate_cell(plan, _cell, cell_dir, expected_attempt_id="attempt")[0]
    rows = [
        {
            **base_row,
            "cell_id": f"fpm-concurrent-{index}",
            "source_plan_sha256": f"plan-{index}",
        }
        for index in range(4)
    ]
    systems_root = tmp_path / "systems"
    # The publisher uses ``fcntl`` and is therefore POSIX-only. Use the
    # matching process model here so the child inherits the already-loaded
    # extension from the wheel under test instead of re-importing a source
    # checkout that intentionally contains no in-tree native artifact.
    context = multiprocessing.get_context("fork")
    start_event = context.Event()
    processes = [
        context.Process(
            target=_concurrent_database_writer,
            args=(str(systems_root), row, start_event),
        )
        for row in rows
    ]
    try:
        for process in processes:
            process.start()
        start_event.set()
        for process in processes:
            process.join(timeout=30)
        assert [process.exitcode for process in processes] == [0] * len(processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    import pyarrow.parquet as pq

    destination = systems_root / "b200_sxm" / "vllm" / "0.24.0"
    table = pq.read_table(destination / "fpm_forward_perf.parquet")
    metadata = json.loads((destination / "fpm_forward_perf.metadata.json").read_text())
    assert table.num_rows == len(rows)
    assert metadata["row_count"] == len(rows)


def test_minimax_m3_family_routing_survives_inherited_base_attention_ops():
    """include_base-inherited dense ops must not count as exact evidence."""

    plan = build_collection_plan(
        backend="vllm",
        model_path="MiniMaxAI/MiniMax-M3",
        model_architecture="MiniMaxM3ForCausalLM",
        system="b200_sxm",
        selected_ops={"attention_context", "attention_generation"},
        has_model_cases=True,
        options=FPMCollectionOptions.from_args(
            _args(
                fpm_max_gpus=16,
                fpm_gpu_counts=[8, 16],
            )
        ),
    )

    assert plan.capability.support_level == "family_template"
    assert plan.capability.template_id == "aic_family:minimaxm3:moe_msa"
    assert plan.capability.attention_source == "dsa_module"


def test_fmha_resolves_per_kv_dtype_against_joint_evidence(monkeypatch):
    """fp8 fmha exists only under the fp8 kv slice: bf16-kv cells must carry
    the bfloat16 transfer slice (recorded via fmha_resolution), never a flat
    fp8 label the database cannot serve jointly."""

    class Estimate:
        def __init__(self):
            self.breakdown = {"non_kv_bytes": 50, "gpu_memory_capacity_bytes": 100}

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        lambda *_args, **_kwargs: Estimate(),
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args(fpm_kv_cache_dtypes=["bfloat16", "fp8"])),
    )

    # The shipped b200/vllm dsa tables carry only a bfloat16 fmha slice (the
    # pre-0.24 fp8-fmha donor lanes were retired in the 2026-08 prune and the
    # stock 0.24 collector does not produce them), so BOTH kv slices resolve
    # to the bfloat16 transfer slice — the fp8-kv cells record the
    # data-availability fallback, the bf16-kv cells the kv-coupled dispatch.
    assert plan.dtype_profile.fmha_by_kv_dtype == {"bfloat16": "bfloat16", "fp8": "bfloat16"}
    assert plan.dtype_profile.fmha_resolution_by_kv_dtype == {
        "bfloat16": "kv_dtype_dispatch_from_fp8",
        "fp8": "aic_data_fallback_from_fp8",
    }
    labels = {(cell.kv_cache_dtype, cell.fmha_quant_mode, cell.fmha_resolution) for cell in plan.cells}
    assert labels == {
        ("bfloat16", "bfloat16", "kv_dtype_dispatch_from_fp8"),
        ("fp8", "bfloat16", "aic_data_fallback_from_fp8"),
    }


def test_formal_database_rejects_path_bearing_backend_versions(tmp_path):
    """backend_version comes from pod provenance; a path-bearing value must
    never become a directory component under the database root."""

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    for row in rows:
        row["backend_version"] = "0.24.0/../../evil"

    with pytest.raises(ValueError, match="not a safe database directory name"):
        write_formal_database(plan, rows, systems_root=tmp_path / "systems")


def test_generator_overrides_reject_malformed_model_cache():
    from collector.fpm_forward.entry import _load_generator_overrides

    args = argparse.Namespace(
        generator_config=None,
        generator_set=None,
        generator_dynamo_version=None,
        generated_config_version=None,
        namespace=None,
        transport=None,
        image_pull_secret=None,
        model_cache="pvc:mount:sub:extra",
    )
    with pytest.raises(ValueError, match="NAME\\[:MOUNT\\[:SUBPATH\\]\\]"):
        _load_generator_overrides(args)


def test_formal_database_requires_family_measured_version_in_curated_tree(tmp_path, monkeypatch):
    """Default publication targets <system>/<backend>/<version> (the fpm
    forward-model consumer's path), but only for versions the curated tree
    already measures under a family dir: the SDK treats any populated version
    dir as a declared database, so an undeclared (or marker-only) version must
    not be materialized."""

    from collector.fpm_forward import database as fpm_database

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
    curated = tmp_path / "curated"
    monkeypatch.setattr(fpm_database, "_curated_systems_root", lambda: curated)

    # Nothing declares the version anywhere in the curated tree.
    with pytest.raises(ValueError, match="not a curated AIC database version"):
        write_formal_database(plan, rows, systems_root=None)

    # Evidence under a dot-prefixed dir is invisible to SDK version discovery
    # and must not count.
    hidden = curated / plan.system / ".backup" / plan.backend / "0.24.0"
    hidden.mkdir(parents=True)
    (hidden / "attention_perf.parquet").write_bytes(b"stub")
    with pytest.raises(ValueError, match="not a curated AIC database version"):
        write_formal_database(plan, rows, systems_root=None)

    # A marker-only family dir declares the version but holds no measured
    # data; publishing would flip a marker-only version into a "has data"
    # version and change op-level default-version resolution.
    family_dir = curated / plan.system / "attention" / plan.backend / "0.24.0"
    family_dir.mkdir(parents=True)
    (family_dir / "reuse.yaml").write_text("reuse: []\n")
    with pytest.raises(ValueError, match="not a curated AIC database version"):
        write_formal_database(plan, rows, systems_root=None)

    # Measured files in a mid-collection (partial) dir are vetoed too: the SDK
    # treats a partial version dir as undeclared regardless of its perf files,
    # so publishing would newly declare the version to version discovery.
    (family_dir / "attention_perf.parquet").write_bytes(b"stub")
    (family_dir / "collection_meta.yaml").write_text("tables:\n  attention:\n    status: partial\n")
    with pytest.raises(ValueError, match="not a curated AIC database version"):
        write_formal_database(plan, rows, systems_root=None)

    # A completed collection admits the version; publication creates the
    # two-level consumer path next to the family layout.
    (family_dir / "collection_meta.yaml").write_text("tables:\n  attention:\n    status: complete\n")
    parquet, _metadata, _skipped = write_formal_database(plan, rows, systems_root=None)
    assert parquet.is_file()
    assert parquet == curated / plan.system / plan.backend / "0.24.0" / "fpm_forward_perf.parquet"

    explicit = tmp_path / "explicit"
    parquet2, _metadata2, _skipped2 = write_formal_database(plan, rows, systems_root=explicit)
    assert parquet2.is_file()


def test_curated_systems_root_resolves_to_the_sdk_default_tree():
    """The default publication root must be the tree the SDK's
    --systems-paths default actually reads (the aisimulate_core package
    data), not a repo-relative guess."""

    from collector.fpm_forward.database import _curated_systems_root

    root = _curated_systems_root()
    assert root.parts[-2:] == ("systems", "data")
    assert "aisimulate_core" in root.parts
    assert root.is_dir()


def _args_cell(workload_kind: str, parallel_strategy: str = "tep") -> FPMCell:
    if parallel_strategy == "pure_tp":
        topology = ParallelTopology(tp=4, pp=1, dp=1, moe_tp=4, moe_ep=1, cp=1)
    elif parallel_strategy == "tp":
        topology = ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1)
    else:
        topology = ParallelTopology(tp=4, pp=1, dp=1, moe_tp=1, moe_ep=4, cp=1)
    return FPMCell(
        cell_id=f"fpm-args-{workload_kind}-{parallel_strategy}",
        workload_kind=workload_kind,
        topology=topology,
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline_auto", {}, {}),
        parallel_strategy=parallel_strategy,
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )


def _args_plan():
    return SimpleNamespace(
        sha256="plan-sha",
        model_path="org/model",
        system="b200_sxm",
        backend="vllm",
        options=SimpleNamespace(
            warmup_iterations=5,
            vllm_max_model_len=-1,
            prefill_sampling=PrefillSamplingProfile.build(max_isl=8192, max_batch_size=None),
        ),
    )


def _cell_cli_args(workload_kind: str, parallel_strategy: str = "tep") -> list[str]:
    from collector.fpm_forward.runner import _cell_generator_overrides

    merged = _cell_generator_overrides(_args_plan(), _args_cell(workload_kind, parallel_strategy), {})
    return merged["params"]["agg"]["extra_cli_args"]


@pytest.mark.parametrize("strategy", ["tep", "dep", "pure_tp"])
def test_kvwarm_decode_cells_keep_prefix_caching_and_async_overlap(strategy):
    """Warm-required MoE decode keeps engine defaults.

    Prefix caching stays enabled: KV warm-up reuses warmed prefixes across
    points and refuses to warm without it
    (skip_reason="prefix_caching_disabled"), which would collapse every
    decode point into the fake-KV fallback regime convicted of
    underestimating capture-mode decode. Under warm-up, full-context block
    hashing runs at seed time, outside the measured step (r15 parity: MAPE
    1.61/1.70% with prefix ON + kvwarm). Async scheduling also stays on:
    production overlaps scheduler CPU work with the GPU (async 26.5 ms =
    1.03x of the 25.8 ms measured on real traffic, sync 31.3 ms = 1.21x).
    """
    args = _cell_cli_args("decode", strategy)

    assert "--no-enable-prefix-caching" not in args
    assert "--no-async-scheduling" not in args


def test_fake_kv_decode_cells_still_disable_prefix_caching():
    """Dense decode (kvwarm-exempt: no experts, physically immune to
    activation dispersion) keeps the fake-KV pin.

    These cells re-admit batch_size synthetic full-context requests per
    point; with prefix caching on, ``Request.__init__`` hashes the whole
    prompt through the block hasher INSIDE the measured step (26.5 ms ->
    121 ms at (batch 256, 2.1M KV) on M2.7 tp4+EP), while production
    steady-state decode has no such per-step hash. Disabling remains the
    serving-faithful choice for this regime.
    """
    args = _cell_cli_args("decode", "tp")

    assert "--no-enable-prefix-caching" in args
    assert "--no-async-scheduling" not in args


def test_prefill_cells_keep_prefix_caching_for_seeded_kv_reads():
    """Prefill must NOT disable prefix caching.

    Points with total_kv_read_tokens > 0 stage their context through the fake
    prefix cache, and ``_bench_cached_kv_read_tokens`` reads
    ``Request.block_hashes`` -- only populated while prefix caching installs a
    block hasher. Disabling it fails every cached-prefill point's seed
    validation. Prefill is insensitive to both flags anyway (100-108 ms across
    all four combinations at 8192 new tokens).
    """
    args = _cell_cli_args("prefill")

    assert "--no-enable-prefix-caching" not in args
    assert "--no-async-scheduling" in args


def _decode_cell_with_coordinate_collision(tmp_path, *, clamp_first):
    """Two decode plan points that measure the same physical coordinate.

    Under the steady-state decode policy a context-clamped point (requested
    ctx=1, measured at ctx=2) lands on the coordinate the native ctx=2 point
    already samples.
    """
    topology = ParallelTopology(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1, cp=1)
    cell = FPMCell(
        cell_id="fpm-test-decode",
        workload_kind="decode",
        topology=topology,
        weight_quantization="nvfp4",
        kv_cache_dtype="fp8",
        backend_policy=BackendPolicy("baseline_auto", {}, {}),
        parallel_strategy="single",
        gemm_quant_mode="nvfp4",
        moe_quant_mode="nvfp4",
        fmha_quant_mode="fp8",
        comm_quant_mode="half",
    )
    plan = SimpleNamespace(
        sha256="plan-sha",
        aic_revision="revision",
        model_path="org/model",
        system="b200_sxm",
        backend="vllm",
        options=SimpleNamespace(warmup_iterations=0),
        capability=SimpleNamespace(
            support_level="exact",
            template_id="aic_exact:dsa_module",
            template_version=1,
            aic_database_version="0.24.0",
        ),
    )
    if clamp_first == "both":
        reason_pairs = (["capture", "context_clamped"], ["capture", "context_clamped"])
    elif clamp_first:
        reason_pairs = (["capture", "context_clamped"], ["capture"])
    else:
        reason_pairs = (["capture"], ["capture"])
    points = []
    for benchmark_id, (reasons, wall) in enumerate(zip(reason_pairs, (0.0070, 0.0068), strict=True), start=1):
        points.append(
            (
                {
                    "point_type": "decode",
                    "benchmark_id": benchmark_id,
                    "total_prefill_tokens": 0,
                    "total_kv_read_tokens": 2,
                    "batch_size": 1,
                    "expected_cudagraph_mode": "FULL",
                    "expected_capture_size": 1,
                    "padding_tokens": 0,
                    "sample_reasons": reasons,
                },
                {
                    "counter_id": benchmark_id,
                    "dp_rank": 0,
                    "wall_time": wall,
                    "scheduled_requests": {
                        "num_prefill_requests": 0,
                        "sum_prefill_tokens": 0,
                        "sum_prefill_kv_tokens": 0,
                        "num_decode_requests": 1,
                        "sum_decode_kv_tokens": 2,
                    },
                },
            )
        )
    cell_dir = tmp_path / "cell"
    output = cell_dir / "raw" / "pod-0" / "benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_provenance(output.parent / "collector-provenance.json", cell_id=cell.cell_id)
    output.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "artifact_type": "rank",
                "status": "complete",
                "valid": True,
                "usable": True,
                "timing_valid": True,
                "stop_reason": None,
                "error": None,
                "run_id": "run",
                "grid_digest": "grid",
                "config": {"mode": "decode"},
                "coverage": {"expected_points": 2, "completed_points": 2, "skipped_points": 0},
                "dp": {"rank": 0, "size": 1},
                "results": [{"point": point, "fpms": [fpm]} for point, fpm in points],
                "iteration_groups": [
                    {
                        "benchmark_id": point["benchmark_id"],
                        "point": point,
                        "expected_dp_ranks": [0],
                        "complete": True,
                        "wall_time": fpm["wall_time"],
                        "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
                    }
                    for point, fpm in points
                ],
                "skipped_points": [],
                "missing_phases": [],
                "timing": {
                    "benchmark_elapsed_seconds": 1.0,
                    "measured_iteration_seconds": 0.0138,
                },
            }
        )
    )
    return plan, cell, cell_dir


def test_native_aggregation_drops_clamped_duplicate_of_native_coordinate(tmp_path):
    plan, cell, cell_dir = _decode_cell_with_coordinate_collision(tmp_path, clamp_first=True)
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert len(rows) == 1
    assert rows[0]["latency_ms"] == pytest.approx(6.8)
    assert rows[0]["batch_size"] == 1
    assert rows[0]["total_kv_read_tokens"] == 2

    write_formal_database(plan, rows, systems_root=tmp_path / "systems")


def test_native_aggregation_keeps_first_when_all_duplicates_are_clamped(tmp_path):
    plan, cell, cell_dir = _decode_cell_with_coordinate_collision(tmp_path, clamp_first="both")
    rows = aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")

    assert len(rows) == 1
    assert rows[0]["latency_ms"] == pytest.approx(7.0)


def test_native_aggregation_rejects_native_coordinate_collision(tmp_path):
    plan, cell, cell_dir = _decode_cell_with_coordinate_collision(tmp_path, clamp_first=False)

    with pytest.raises(ValueError, match="unclamped samples share one key"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


def test_backend_identity_defaults_record_auto_everywhere():
    """v6: unspecified knobs record "auto" (engine decided), plumb nothing."""

    options = FPMCollectionOptions.from_args(_args())
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        system="b200_sxm",
        selected_ops={"dsa_context_module"},
        options=options,
        generator_overrides={},
    )
    policy = plan.cells[0].backend_policy
    assert backend_identity_columns(policy) == {
        "moe_backend": "auto",
        "attention_backend": "auto",
        "enable_wideep": False,
        "enable_eplb": False,
    }
    assert policy.policy_id == "baseline_auto"
    assert policy.generator_overrides == {}
    assert policy.expected_markers == {}


def test_pinned_moe_backend_plumbs_kernel_config_and_moves_cell_identity():
    """v6: a pinned value must reach the engine (kernel-config), demand
    resolved-config evidence, land in the row columns, and change cell ids."""

    kwargs = {
        "backend": "vllm",
        "model_path": "nvidia/GLM-5.2-NVFP4",
        "system": "b200_sxm",
        "selected_ops": {"dsa_context_module"},
        "generator_overrides": {},
    }
    auto_plan = build_collection_plan(**kwargs, options=FPMCollectionOptions.from_args(_args()))
    pinned_plan = build_collection_plan(
        **kwargs, options=FPMCollectionOptions.from_args(_args(fpm_moe_backend="flashinfer_cutlass"))
    )

    policy = pinned_plan.cells[0].backend_policy
    assert policy.expected_markers == {"config.engine_args.kernel_config.moe_backend": "flashinfer_cutlass"}
    cli_args = policy.generator_overrides["params"]["agg"]["extra_cli_args"]
    assert cli_args[0] == "--kernel-config"
    assert json.loads(cli_args[1]) == {"moe_backend": "flashinfer_cutlass"}
    assert backend_identity_columns(policy)["moe_backend"] == "flashinfer_cutlass"
    assert policy.policy_id == "explicit-moe_backend=flashinfer_cutlass"
    assert {cell.cell_id for cell in pinned_plan.cells}.isdisjoint({cell.cell_id for cell in auto_plan.cells})


def test_pinned_eplb_sets_engine_flag_and_marker():
    options = FPMCollectionOptions.from_args(_args(fpm_enable_eplb="true"))
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        system="b200_sxm",
        selected_ops={"dsa_context_module"},
        options=options,
        generator_overrides={},
    )
    policy = plan.cells[0].backend_policy
    assert policy.expected_markers == {"config.engine_args.enable_eplb": "True"}
    assert "--enable-eplb" in policy.generator_overrides["params"]["agg"]["extra_cli_args"]
    assert backend_identity_columns(policy)["enable_eplb"] is True


def test_unplumbed_backend_identity_fails_closed():
    """A pinned value the collector cannot deliver to the engine must be
    rejected up front - a row claiming an unapplied backend would be a lie."""

    for overrides, match in (
        ({"fpm_enable_wideep": "true"}, "SGLang-only"),  # true only; false is the default
        ({"fpm_attention_backend": "fa3"}, "no verified vllm plumbing"),
    ):
        options = FPMCollectionOptions.from_args(_args(**overrides))
        with pytest.raises(ValueError, match=match):
            build_collection_plan(
                backend="vllm",
                model_path="nvidia/GLM-5.2-NVFP4",
                system="b200_sxm",
                selected_ops={"dsa_context_module"},
                options=options,
                generator_overrides={},
            )


def test_aic_structural_validation_failure_stays_runnable(monkeypatch):
    """Model-layer validation is observable evidence, not permission for the
    Collector to predict a skip; the live runtime must still receive every
    topology unless a concrete memory estimate exceeds capacity."""

    def structurally_invalid(*_args, **_kwargs):
        raise ValueError(
            "Invalid quantized MoE configuration: (moe_intermediate_size=1536 / moe_tp_size=8) "
            "% weight_block_size=128 != 0"
        )

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        structurally_invalid,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert len(plan.topologies) == 3
    assert {decision.disposition for decision in plan.topology_memory_admission} == {"unknown"}
    assert all("ValueError" in decision.estimates[0].reason for decision in plan.topology_memory_admission)


def test_structural_estimator_failure_does_not_drop_selected_topologies(monkeypatch):
    from types import SimpleNamespace

    def selective(*_args, **kwargs):
        if kwargs.get("moe_tp_size", 1) > 1:
            raise ValueError("Invalid quantized MoE configuration: 1536 % 128 != 0")
        return SimpleNamespace(breakdown={"non_kv_bytes": 10 * 2**30, "gpu_memory_capacity_bytes": 100 * 2**30})

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        selective,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert len(plan.topologies) == 3
    by_moe_tp = {decision.topology.moe_tp: decision.disposition for decision in plan.topology_memory_admission}
    assert by_moe_tp[1] == "admitted"
    assert by_moe_tp[4] == "unknown"


def test_missing_perf_data_stays_runnable_under_memory_admission(monkeypatch):
    """Coverage gaps are not structural invalidity: collection may be exactly
    what fills them, so strict admission must not reject them."""

    from aisimulate_core.sdk.errors import PerfDataNotAvailableError

    def unavailable(*_args, **_kwargs):
        raise PerfDataNotAvailableError("no perf rows for this shape")

    monkeypatch.setattr(
        "collector.fpm_forward.memory_admission.KVCacheEstimator.from_request",
        unavailable,
    )
    plan = build_collection_plan(
        backend="vllm",
        model_path="nvidia/GLM-5.2-NVFP4",
        model_architecture="GlmMoeDsaForCausalLM",
        system="b200_sxm",
        selected_ops={"dsa_context_module", "dsa_generation_module"},
        options=FPMCollectionOptions.from_args(_args()),
    )

    assert len(plan.topologies) == 3
    assert {decision.disposition for decision in plan.topology_memory_admission} == {"unknown"}


def _write_v41_token_streams(payload, path):
    import hashlib

    lines = []
    for row in payload["results"]:
        point = row["point"]
        batch = point["batch_size"]
        decode = point["point_type"] == "decode"
        seed = point["total_kv_read_tokens"] - (batch if decode else 0)
        prompt_total = seed + (0 if decode else point["total_prefill_tokens"])
        lengths = [prompt_total // batch + (index < prompt_total % batch) for index in range(batch)]
        stream = {
            "benchmark_id": point["benchmark_id"],
            "requests": [
                {
                    "request_index": index,
                    "prompt_token_ids": [11 + index % 2] * length,
                    "output_token_ids": [37],
                    "computed_tokens": length + (2 if decode else 0),
                }
                for index, length in enumerate(lengths)
            ],
        }
        encoded = json.dumps(stream, sort_keys=True, separators=(",", ":")).encode()
        lines.append(encoded)
        row["real_kv_witness"] = {
            "same_request": True,
            "allocated_fake_tokens": 0,
            "completed_seed_tokens": seed,
            "token_stream_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    raw = b"\n".join(lines) + b"\n"
    sidecar = path.with_suffix(".token-streams.jsonl")
    sidecar.write_bytes(raw)
    payload["input_provenance"]["token_stream_manifest"] = {
        "file": sidecar.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "records": len(lines),
    }


@pytest.mark.parametrize("marker", ["kvwarm_real_kv", "kvwarm_fake_fallback", None])
def test_v41_cached_prefill_requires_real_computed_state(tmp_path, marker):
    from dataclasses import replace

    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    identity = ("c" * 64, "full", "hbm_tp_sharded", "text")
    cell = replace(cell, execution_identity=identity, input_text_sha256="a" * 64)
    for path in (cell_dir / "raw").glob("*/benchmark*.json"):
        payload = json.loads(path.read_text())
        payload["execution_identity"] = dict(zip(EXECUTION_COLUMNS, identity, strict=True))
        payload["execution_mode"] = "eager"
        payload["input_provenance"] = {
            "source": "tokenizer_text",
            "text_sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
            "tokenizer_revision": "pinned",
            "token_count": 100,
            "unique_token_count": 20,
        }
        payload["kvwarm"] = {"enabled": True, "warm_eligible": True, "skip_reason": None}

        def mark(value):
            if isinstance(value, dict):
                if "point_type" in value:
                    value["sample_reasons"] = [marker] if marker else []
                for child in value.values():
                    mark(child)
            elif isinstance(value, list):
                for child in value:
                    mark(child)

        mark(payload)
        _write_v41_token_streams(payload, path)
        path.write_text(json.dumps(payload))
    if marker == "kvwarm_real_kv":
        assert aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")[0]["kv_seed_regime"] == "real_kv"
    else:
        with pytest.raises(ValueError, match="requires real_kv"):
            aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")


@pytest.mark.parametrize("mode", [None, "PIECEWISE", "FULL", False])
def test_v41_reader_rejects_unqualified_graph_or_missing_execution_mode(tmp_path, mode):
    from collector.fpm_forward.native_artifact import _validate_execution_provenance

    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS

    identity = ("c" * 64, "full", "hbm_tp_sharded", "text")
    cell = SimpleNamespace(execution_identity=identity)
    payload = {"execution_identity": dict(zip(EXECUTION_COLUMNS, identity, strict=True)), "execution_mode": mode}
    with pytest.raises(ValueError, match="verified eager"):
        _validate_execution_provenance(cell, payload, tmp_path / "rank.json")


@pytest.mark.parametrize("v41,eager", [(True, False), (False, True)])
def test_explicit_eager_collection_admission(v41, eager, monkeypatch):
    from collector.fpm_forward import planner

    resolve = planner.resolve_model_capability

    def resolve_architecture(**kwargs):
        from dataclasses import replace

        capability = resolve(**kwargs)
        return replace(capability, architecture="DeepseekV41ForCausalLM") if v41 else capability

    monkeypatch.setattr(planner, "resolve_model_capability", resolve_architecture)
    monkeypatch.setattr(
        planner,
        "execution_identity",
        lambda *args, **kwargs: ("c" * 64, "full", "hbm_tp_sharded", "text") if v41 else ("", "", "", ""),
    )
    with pytest.raises(ValueError, match="eager"):
        build_collection_plan(
            backend="vllm",
            model_path="nvidia/GLM-5.2-NVFP4",
            system="b200_sxm",
            selected_ops={"dsa_context_module", "dsa_generation_module"},
            options=FPMCollectionOptions.from_args(_args(fpm_enforce_eager=eager)),
        )


@pytest.mark.parametrize("corruption", ["missing", "tampered", "fake", "seed", "path", "coverage"])
def test_v41_real_token_stream_validation_rejects_broken_witness(tmp_path, corruption):
    from collector.fpm_forward.native_artifact import _validate_token_streams

    _plan, _cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    path = next((cell_dir / "raw").glob("*/benchmark*.json"))
    payload = json.loads(path.read_text())
    payload["input_provenance"] = {}
    _write_v41_token_streams(payload, path)
    _validate_token_streams(payload, path)
    manifest = payload["input_provenance"]["token_stream_manifest"]
    if corruption == "missing":
        del payload["input_provenance"]["token_stream_manifest"]
    elif corruption == "tampered":
        path.with_name(manifest["file"]).write_text("changed")
    elif corruption == "fake":
        payload["results"][0]["real_kv_witness"]["allocated_fake_tokens"] = 1
    elif corruption == "seed":
        payload["results"][0]["real_kv_witness"]["completed_seed_tokens"] -= 1
    elif corruption == "path":
        manifest["file"] = "../outside.token-streams.jsonl"
    else:
        manifest["records"] += 1
    with pytest.raises(ValueError, match="V4.1"):
        _validate_token_streams(payload, path)


@pytest.mark.parametrize(
    "corruption", [None, "role", "missing_result", "missing_expected", "measured_warmup", "legacy"]
)
def test_v41_eager_warmup_histories_are_preserved_but_not_measured(tmp_path, corruption):
    from copy import deepcopy

    from collector.fpm_forward.native_artifact import _validate_token_streams

    _plan, _cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    path = next((cell_dir / "raw").glob("*/benchmark*.json"))
    payload = json.loads(path.read_text())
    payload["input_provenance"] = {}
    warmup = deepcopy(payload["results"][0])
    warmup_id = len(payload["results"]) + 1
    warmup["point"].update(benchmark_id=warmup_id, sample_reasons=["eager_warmup"])
    payload["results"].append(warmup)
    _write_v41_token_streams(payload, path)
    payload["warmup_results"] = [payload["results"].pop()]
    manifest = payload["input_provenance"]["token_stream_manifest"]
    manifest.update(schema_version=2, warmup_benchmark_ids=[warmup_id])
    sidecar = path.with_name(manifest["file"])
    all_rows = payload["results"] + payload["warmup_results"]
    encoded = []
    for line, row in zip(sidecar.read_bytes().splitlines(), all_rows, strict=True):
        stream = json.loads(line)
        stream["sampling_role"] = "warmup" if stream["benchmark_id"] == warmup_id else "measurement"
        if corruption == "role" and stream["benchmark_id"] == warmup_id:
            stream["sampling_role"] = "measurement"
        changed = json.dumps(stream, sort_keys=True, separators=(",", ":")).encode()
        encoded.append(changed)
        row["real_kv_witness"]["token_stream_sha256"] = hashlib.sha256(changed).hexdigest()
    raw = b"\n".join(encoded) + b"\n"
    sidecar.write_bytes(raw)
    manifest["sha256"] = hashlib.sha256(raw).hexdigest()
    if corruption == "missing_result":
        payload["warmup_results"] = []
    elif corruption == "missing_expected":
        manifest["warmup_benchmark_ids"] = []
    elif corruption == "measured_warmup":
        payload["results"].append(payload["warmup_results"].pop())
    elif corruption == "legacy":
        manifest["schema_version"] = 1
    if corruption is None:
        _validate_token_streams(payload, path)
        assert len(payload["results"]) + 1 == manifest["records"]
    else:
        with pytest.raises(ValueError, match="V4.1"):
            _validate_token_streams(payload, path)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("smoke", [False, True])
def test_v41_native_grid_bounds_reach_both_runtime_phases(phase, smoke):
    from collector.fpm_forward.runner import _cell_generator_overrides

    plan = _args_plan()
    plan.capability = SimpleNamespace(architecture="DeepseekV41ForCausalLM")
    plan.options = FPMCollectionOptions.from_args(
        _args(
            fpm_max_prefill_isl=64,
            fpm_max_prefill_batch_size=1,
            fpm_max_decode_batch_size=1,
            fpm_max_model_len=258,
            fpm_warmup_iterations=0,
        )
    )
    generated = _cell_generator_overrides(plan, _args_cell(phase, "pure_tp"), {}, smoke=smoke)
    args = generated["params"]["agg"]["extra_cli_args"]
    assert args[args.index("--max-num-batched-tokens") + 1] == "64"
    assert args[args.index("--max-num-seqs") + 1] == "1"
    assert '--engram-config={"cpu_offload":false}' in args


@pytest.mark.parametrize(
    "corruption",
    [
        "results_missing",
        "results_null",
        "row_null",
        "point_missing",
        "point_null",
        "groups_missing",
        "provenance_missing",
    ],
)
def test_v41_truncated_native_artifact_raises_actionable_value_error(tmp_path, corruption):
    from dataclasses import replace

    from aisimulate_core.sdk.fpm_identity import EXECUTION_COLUMNS

    plan, cell, cell_dir = _synthetic_plan_and_cell(tmp_path)
    identity = ("c" * 64, "full", "hbm_tp_sharded", "text")
    cell = replace(cell, execution_identity=identity, input_text_sha256="a" * 64)
    for path in (cell_dir / "raw").glob("*/benchmark*.json"):
        payload = json.loads(path.read_text())
        payload["execution_identity"] = dict(zip(EXECUTION_COLUMNS, identity, strict=True))
        payload["execution_mode"] = "eager"
        payload["input_provenance"] = {
            "source": "tokenizer_text",
            "text_sha256": "a" * 64,
            "token_ids_sha256": "b" * 64,
            "tokenizer_revision": "pinned",
            "token_count": 100,
            "unique_token_count": 20,
        }
        _write_v41_token_streams(payload, path)
        if corruption == "results_missing":
            del payload["results"]
        elif corruption == "results_null":
            payload["results"] = None
        elif corruption == "row_null":
            payload["results"][0] = None
        elif corruption == "point_missing":
            del payload["results"][0]["point"]
        elif corruption == "point_null":
            payload["results"][0]["point"] = None
        elif corruption == "groups_missing":
            del payload["iteration_groups"]
        else:
            del payload["input_provenance"]
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="native"):
        aggregate_cell(plan, cell, cell_dir, expected_attempt_id="attempt")
