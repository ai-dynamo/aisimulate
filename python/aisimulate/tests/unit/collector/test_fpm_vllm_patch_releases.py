# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
from collector.fpm_forward import repeatability, runner, runtime_memory, vllm_source_manifest
from collector.fpm_forward.runtime import fpm_memory_observer as observer

from .test_fpm_profile_collection import _plan, _profile, no_models_or_timing_data  # noqa: F401
from .test_fpm_repeatability import _write_campaign
from .test_fpm_runner import _write_provenance
from .test_fpm_runtime_memory import (  # noqa: F401
    _initialized_layer_strided,
    _pending_plan,
    _vllm_config,
    fake_runtime_offloader,
)

pytestmark = pytest.mark.unit

_AUDITED_FILES = (
    "vllm/compilation/decorators.py",
    "vllm/config/cache.py",
    "vllm/config/compilation.py",
    "vllm/config/kernel.py",
    "vllm/config/model.py",
    "vllm/config/offload.py",
    "vllm/config/parallel.py",
    "vllm/config/scheduler.py",
    "vllm/config/vllm.py",
    "vllm/distributed/__init__.py",
    "vllm/distributed/parallel_state.py",
    "vllm/engine/arg_utils.py",
    "vllm/model_executor/layers/attention/attention.py",
    "vllm/model_executor/layers/attention_layer_base.py",
    "vllm/model_executor/layers/quantization/__init__.py",
    "vllm/model_executor/layers/quantization/fp8.py",
    "vllm/model_executor/layers/quantization/modelopt.py",
    "vllm/model_executor/offloader/__init__.py",
    "vllm/model_executor/offloader/base.py",
    "vllm/models/inkling/nvidia/sconv_swa_attn.py",
    "vllm/platforms/cuda.py",
    "vllm/platforms/interface.py",
    "vllm/v1/core/block_pool.py",
    "vllm/v1/core/kv_cache_coordinator.py",
    "vllm/v1/core/kv_cache_manager.py",
    "vllm/v1/core/kv_cache_planning.py",
    "vllm/v1/core/kv_cache_utils.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/single_type_kv_cache_manager.py",
    "vllm/v1/engine/core.py",
    "vllm/v1/kv_cache_interface.py",
    "vllm/v1/worker/gpu/attn_utils.py",
    "vllm/v1/worker/gpu/cudagraph_utils.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu/model_states/encoder_only.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "vllm/v1/worker/gpu_worker.py",
    "vllm/v1/worker/utils.py",
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


_CACHE = "vllm/config/cache.py"
_UTILS = "vllm/v1/core/kv_cache_utils.py"


@pytest.fixture
def fake_vllm(tmp_path_factory, monkeypatch):
    site = tmp_path_factory.mktemp("site")
    root = site / "vllm"
    (root / "config").mkdir(parents=True)
    (root / "v1" / "core").mkdir(parents=True)
    (root / "config" / "cache.py").write_text("cache = 1\n")
    (root / "v1" / "core" / "kv_cache_utils.py").write_text("utils = 1\n")
    module = ModuleType("vllm")
    module.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "vllm", module)
    manifest = {_CACHE: _sha("cache = 1\n"), _UTILS: _sha("utils = 1\n")}
    monkeypatch.setattr(observer, "SOURCE_MANIFESTS", {"0.32.0": dict(manifest), "0.33.0": dict(manifest)})
    monkeypatch.setattr(observer, "_SOURCE_PROBLEMS", {})
    return root


@pytest.mark.parametrize(
    ("version", "release"),
    [
        ("0.27.0", "0.27.0"),
        ("0.32.0", "0.32.0"),
        ("0.33.0", "0.33.0"),
        ("0.33.1", "0.33.0"),
        ("0.33.10", "0.33.0"),
        ("0.32.2", "0.32.0"),
        ("0.33.0.1", "0.33.0"),
        ("0.27.1", None),
        ("0.33.1rc0", None),
        ("0.33.1.dev3+gabc1234", None),
        ("0.33.1+cu130", None),
        ("0.33.0.post1", None),
        ("0.34.0", None),
        ("0.31.2", None),
        ("v0.33.1", None),
        (" 0.33.1", None),
        ("0.33", None),
        (None, None),
    ],
)
def test_memory_version_rule(version, release):
    assert observer.audited_release(version, observer.MEMORY_SUPPORTED_VERSIONS) == release


@pytest.mark.parametrize(("version", "release"), [("0.28.0", "0.28.0"), ("0.28.1", None), ("0.33.1", "0.33.0")])
def test_execution_version_rule(version, release):
    assert observer.audited_release(version, observer.EXECUTION_SUPPORTED_VERSIONS) == release


def test_newest_audited_release_of_the_line_is_the_reference(monkeypatch):
    extra = {"0.33.1": {}, "0.33.9": {}, "0.33.10": {}}
    monkeypatch.setattr(observer, "SOURCE_MANIFESTS", {**observer.SOURCE_MANIFESTS, **extra})
    audited = (*observer.MEMORY_SUPPORTED_VERSIONS, "0.33.1")
    assert observer.audited_release("0.33.2", audited) == "0.33.1"
    assert observer.audited_release("0.33.1", audited) == "0.33.1"
    assert observer.audited_release("0.33.11", (*audited, "0.33.9", "0.33.10")) == "0.33.10"


def test_source_manifests_cover_the_audited_files():
    assert sorted(observer.SOURCE_MANIFESTS["0.33.0"]) == list(_AUDITED_FILES)
    assert sorted(observer.SOURCE_MANIFESTS["0.32.0"]) == [
        path
        for path in _AUDITED_FILES
        if path not in {"vllm/platforms/interface.py", "vllm/v1/core/kv_cache_planning.py"}
    ]
    assert set(observer.SOURCE_MANIFESTS) == {"0.32.0", "0.33.0"}
    assert all(
        re.fullmatch(r"[0-9a-f]{64}", digest) for item in observer.SOURCE_MANIFESTS.values() for digest in item.values()
    )


def test_identical_files_pass_the_source_check(fake_vllm):
    assert observer._source_problem("0.33.0") is None


def test_changed_files_name_the_first_file(fake_vllm):
    (fake_vllm / "config" / "cache.py").write_text("cache = 2\n")
    (fake_vllm / "v1" / "core" / "kv_cache_utils.py").write_text("utils = 2\n")
    assert observer._source_problem("0.33.0") == "vllm/config/cache.py differs"


def test_missing_file_is_refused(fake_vllm):
    (fake_vllm / "v1" / "core" / "kv_cache_utils.py").unlink()
    assert observer._source_problem("0.33.0") == "vllm/v1/core/kv_cache_utils.py is not installed exactly once"


def test_file_under_two_package_roots_is_refused(fake_vllm, tmp_path):
    second = tmp_path / "other" / "vllm"
    (second / "config").mkdir(parents=True)
    (second / "config" / "cache.py").write_text("cache = 1\n")
    sys.modules["vllm"].__path__.append(str(second))
    assert observer._source_problem("0.33.0") == "vllm/config/cache.py is not installed exactly once"


def test_source_check_runs_once_per_process(fake_vllm):
    assert observer._source_problem("0.33.0") is None
    (fake_vllm / "config" / "cache.py").write_text("cache = 2\n")
    assert observer._source_problem("0.33.0") is None
    assert observer._SOURCE_PROBLEMS == {"0.33.0": None}


def _observe_both(tmp_path, monkeypatch, version):
    tmp_path.mkdir(exist_ok=True)
    worker, _scheduler, _config = _initialized_layer_strided()
    monkeypatch.setattr(observer, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(observer.importlib.metadata, "version", lambda _name: version)
    _write_provenance(tmp_path / "collector-provenance.json", cell_id="test")
    observer.observe("worker", worker, dp_rank=0, tp_rank=0, pp_rank=0)
    memory = json.loads((tmp_path / "fpm-memory-worker-dp0-tp0-pp0.json").read_text())
    group = SimpleNamespace(
        backend=type("ResolvedBackend", (), {"__module__": "actual.runtime"}),
        layer_names=["layer.0"],
        kv_cache_group_id=0,
    )
    execution_worker = SimpleNamespace(
        vllm_config=_vllm_config(tp=1, dp=1), model_runner=SimpleNamespace(attn_groups=[[group]])
    )
    observer.observe_execution(execution_worker, dp_rank=0, tp_rank=0, pp_rank=0)
    execution = json.loads((tmp_path / "fpm-execution-worker-dp0-tp0-pp0.json").read_text())
    return memory, execution


def test_patch_release_records_name_the_audited_release(fake_vllm, tmp_path, monkeypatch):
    memory, execution = _observe_both(tmp_path, monkeypatch, "0.33.1")
    assert memory["status"] == "resolved", memory.get("error")
    assert memory["backend_version"] == "0.33.1"
    assert memory["audited_backend_version"] == "0.33.0"
    assert execution["status"] == "observed", execution.get("error")
    assert execution["audited_backend_version"] == "0.33.0"


def test_changed_file_leaves_patch_records_unresolved(fake_vllm, tmp_path, monkeypatch):
    (fake_vllm / "config" / "cache.py").write_text("cache = 2\n")
    memory, execution = _observe_both(tmp_path, monkeypatch, "0.33.1")
    error = "ValueError: installed vLLM 0.33.1 does not match audited vLLM 0.33.0: vllm/config/cache.py differs"
    for record in (memory, execution):
        assert record["status"] == "unresolved"
        assert record["error"] == error
        assert record["audited_backend_version"] == "0.33.0"


def test_audited_versions_skip_the_source_check(tmp_path, monkeypatch):
    def fail(_release):
        raise AssertionError("source check must not run")

    monkeypatch.setattr(observer, "_source_problem", fail)
    memory, execution = _observe_both(tmp_path, monkeypatch, "0.33.0")
    assert memory["status"] == "resolved", memory.get("error")
    assert "audited_backend_version" not in memory
    assert "audited_backend_version" not in execution


def test_refused_versions_keep_their_messages(tmp_path, monkeypatch):
    memory, _ = _observe_both(tmp_path / "memory", monkeypatch, "0.33.1rc0")
    assert memory["error"] == (
        "ValueError: memory observation currently supports vLLM 0.27.0, 0.32.0, 0.33.0; found 0.33.1rc0"
    )
    _, execution = _observe_both(tmp_path / "execution", monkeypatch, "0.28.1")
    assert execution["error"] == (
        "ValueError: execution observation requires audited vLLM ('0.27.0', '0.28.0', '0.32.0', '0.33.0')"
    )
    assert "audited_backend_version" not in memory
    assert "audited_backend_version" not in execution


def test_manifest_script_hashes_the_pinned_commit(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    git = [
        "git",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "tag.gpgsign=false",
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.com",
    ]
    repo = tmp_path / "repo"
    (repo / "vllm").mkdir(parents=True)
    (repo / "vllm" / "a.py").write_text("a = 1\n")
    subprocess.run([*git, "init", "-q"], cwd=repo, check=True)
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "one"], cwd=repo, check=True)
    subprocess.run([*git, "tag", "v9.9.0"], cwd=repo, check=True)
    commit = subprocess.run(
        [*git, "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    digest = _sha("a = 1\n")
    monkeypatch.setattr(vllm_source_manifest, "VLLM_TAGS", {"9.9.0": ("v9.9.0", commit)})
    monkeypatch.setattr(observer, "SOURCE_MANIFESTS", {"9.9.0": {"vllm/a.py": ""}})
    assert vllm_source_manifest.build_manifests(repo) == {"9.9.0": {"vllm/a.py": digest}}
    assert vllm_source_manifest.main(["--vllm-source", str(repo), "--check"]) == 1
    monkeypatch.setattr(observer, "SOURCE_MANIFESTS", {"9.9.0": {"vllm/a.py": digest}})
    assert vllm_source_manifest.main(["--vllm-source", str(repo), "--check"]) == 0
    monkeypatch.setattr(vllm_source_manifest, "VLLM_TAGS", {"9.9.0": ("v9.9.0", "0" * 40)})
    with pytest.raises(ValueError, match="does not resolve to"):
        vllm_source_manifest.build_manifests(repo)


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_patch_release_plans_observation_against_the_audited_release(tmp_path):
    plan = _pending_plan("0.33.1")
    assert plan.to_dict()["runtime_memory_policy"] == {
        "source": "vllm_initialization",
        "supported_vllm_version": "0.33.1",
        "selected_vllm_version": "0.33.1",
        "audited_vllm_version": "0.33.0",
        "observation": "enabled",
        "evidence_schema_version": 1,
        "async_scheduling": False,
    }
    for cell in plan.cells:
        output = tmp_path / cell.cell_id
        output.mkdir()
        runner._render_cell(plan, cell, output, {})
        script = (output / "run.sh").read_text()
        assert "--worker-cls fpm_memory_worker.FpmResourceWorker" in script
        assert "--scheduler-cls fpm_memory_scheduler.FpmResourceInstrumentedScheduler" in script
        assert script.count("--no-async-scheduling") == 1
    assert all(runner._observe_runtime_execution(plan, cell) for cell in plan.cells)


@pytest.mark.parametrize("version", ["0.27.1", "0.28.1", "0.31.2", "0.33.1rc0", "0.33.1+cu130"])
@pytest.mark.usefixtures("no_models_or_timing_data")
def test_refused_patch_versions_keep_the_legacy_policy(version):
    plan = _pending_plan(version)
    assert plan.to_dict()["runtime_memory_policy"] == {
        "source": "vllm_initialization",
        "supported_vllm_version": "0.27.0",
        "selected_vllm_version": version,
        "observation": "unavailable_for_runtime",
        "evidence_schema_version": 1,
        "async_scheduling": None,
    }
    assert not any(runner._observe_runtime_memory(plan, cell) for cell in plan.cells)


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_execution_patch_of_an_unmanifested_line_is_not_observed():
    profile = _profile()
    for deployment in profile["deployments"]:
        deployment["backend_version"] = "0.28.1"
    plan = _plan(profile, collector_config={"runtime_backend_version": "0.28.1"})
    for cell in plan.cells:
        args = runner._cell_generator_overrides(plan, cell, {})["params"]["agg"]["extra_cli_args"]
        assert runner._observe_runtime_execution(plan, cell) is False
        assert "--worker-cls" not in args


@pytest.mark.usefixtures("no_models_or_timing_data")
def test_patch_release_plan_reloads_exactly(tmp_path):
    plan = _pending_plan("0.33.1")
    runtime_memory.validate_saved_plan(plan.to_dict())
    source = tmp_path / "source"
    _write_campaign(source, tmp_path / "source-checkpoint.json", plan)
    assert repeatability.load_repeatability_source(source).to_dict() == plan.to_dict()
