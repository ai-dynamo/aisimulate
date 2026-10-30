# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device-lane identity in collect.py: vLLM registry dispatch and checkpoints.

Two ambiguities appear the moment a second non-CUDA vLLM lane exists:

1. ``collect_vllm`` picked a registry from "CUDA or else XPU".
2. A checkpoint's owning registry was re-derived from
   ``backend == "vllm" and sm_version is None``, which is true for both the
   Intel XPU and the Ascend NPU lane.

Both are now driven by the explicit ``collector_framework`` lane key, with the
old inference kept only as a migration for checkpoints written before the
field existed.
"""

import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_COLLECTOR_DIR = str(Path(__file__).resolve().parents[3] / "collector")
if _COLLECTOR_DIR not in sys.path:
    sys.path.insert(0, _COLLECTOR_DIR)

# collect.py imports torch at module scope; mock it like the rest of the suite.
if "torch" not in sys.modules:
    _torch = MagicMock()
    _torch.AcceleratorError = type("AcceleratorError", (Exception,), {})
    sys.modules["torch"] = _torch

import collect as collect_mod
from collector.registry_types import PerfFile
from collector.vllm import registry as vllm_registry

pytestmark = pytest.mark.unit

collect_mod.logger = logging.getLogger("test_collect_device_lane")


# ---------------------------------------------------------------------------
# collect_vllm registry dispatch
# ---------------------------------------------------------------------------


def _stub_vllm_version(monkeypatch, version="0.28.0"):
    module = type(sys)("vllm.version")
    module.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm", type(sys)("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.version", module)


def _stub_vllm_ascend_version(monkeypatch, version="0.30.0rc2"):
    module = type(sys)("vllm_ascend")
    module.__version__ = version
    monkeypatch.setitem(sys.modules, "vllm_ascend", module)


def _run_collect_vllm(monkeypatch, *, cuda, npu, xpu, with_vllm_ascend=True):
    """Run collect_vllm on a stubbed host, returning what it dispatched to."""
    from collector import framework_manifest

    _stub_vllm_version(monkeypatch)
    if with_vllm_ascend:
        _stub_vllm_ascend_version(monkeypatch)
    else:
        # A None entry makes `import vllm_ascend` raise, as on a host without it.
        monkeypatch.setitem(sys.modules, "vllm_ascend", None)
    monkeypatch.setattr(collect_mod, "_cuda_available", lambda: cuda)
    monkeypatch.setattr(collect_mod, "_npu_available", lambda: npu)
    monkeypatch.setattr(collect_mod, "_xpu_available", lambda: xpu)

    observed = {}

    def fake_require_collector_runtime(framework, installed_version, **kwargs):
        observed["framework"] = framework
        observed["installed_version"] = installed_version
        observed["wideep_ops"] = kwargs.get("wideep_ops")
        return framework_manifest.CollectorRuntime(framework=framework, version=installed_version, images={})

    def fake_collect_ops(num_processes, collections, version, **kwargs):
        observed["collections"] = collections
        observed["backend"] = kwargs.get("backend")
        observed["collector_framework"] = kwargs.get("collector_framework")
        return []

    monkeypatch.setattr(framework_manifest, "require_collector_runtime", fake_require_collector_runtime)
    monkeypatch.setattr(collect_mod, "collect_ops", fake_collect_ops)
    monkeypatch.setattr(collect_mod, "generate_collection_summary", lambda *a, **k: None)

    _errors, provenance_ctx = collect_mod.collect_vllm(1, ops=[])
    observed["provenance_ctx"] = provenance_ctx
    return observed


def test_npu_host_selects_the_npu_lane(monkeypatch):
    observed = _run_collect_vllm(monkeypatch, cuda=False, npu=True, xpu=False)
    assert observed["framework"] == "vllm_npu"
    assert observed["collector_framework"] == "vllm_npu"
    # The data backend directory stays "vllm" — only the lane key differs.
    assert observed["backend"] == "vllm"
    assert observed["provenance_ctx"]["collector_framework"] == "vllm_npu"
    # The lane's version axis is vllm-ascend, not base vLLM (0.28.0 here).
    assert observed["installed_version"] == "0.30.0rc2"
    assert [collection["module"] for collection in observed["collections"]] == ["collector.vllm.collect_gemm_npu"]


def test_npu_lane_refuses_to_run_without_vllm_ascend(monkeypatch):
    """Fail closed: the lane never falls back to the base vLLM version."""
    observed = _run_collect_vllm(monkeypatch, cuda=False, npu=True, xpu=False, with_vllm_ascend=False)
    assert observed["provenance_ctx"] is None
    assert "collector_framework" not in observed


def test_cuda_wins_over_npu_and_npu_wins_over_xpu(monkeypatch):
    cuda_host = _run_collect_vllm(monkeypatch, cuda=True, npu=True, xpu=True)
    assert cuda_host["framework"] == "vllm"
    assert cuda_host["collector_framework"] == "vllm"

    npu_host = _run_collect_vllm(monkeypatch, cuda=False, npu=True, xpu=True)
    assert npu_host["framework"] == "vllm_npu"


def test_xpu_lane_is_unchanged(monkeypatch):
    observed = _run_collect_vllm(monkeypatch, cuda=False, npu=False, xpu=True)
    assert observed["framework"] == "vllm_xpu"
    assert observed["collector_framework"] == "vllm_xpu"
    assert observed["wideep_ops"] == set()


def test_no_accelerator_names_all_three_lanes(monkeypatch):
    with pytest.raises(RuntimeError, match="Neither CUDA, NPU nor XPU"):
        _run_collect_vllm(monkeypatch, cuda=False, npu=False, xpu=False)


def test_npu_registry_symbol_exists_and_is_separate():
    assert vllm_registry.REGISTRY_NPU is not vllm_registry.REGISTRY_XPU
    assert all(entry.perf_filename in set(PerfFile) for entry in vllm_registry.REGISTRY_NPU)


# ---------------------------------------------------------------------------
# Checkpoint identity
# ---------------------------------------------------------------------------


def _write_checkpoint(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _legacy_document(**overrides) -> dict:
    document = {
        "schema": collect_mod.RESUME_SCHEMA_VERSION,
        "backend": "vllm",
        "module": "vllm.gemm",
        "run_func": "run_gemm",
        "framework_version": "0.28.0",
        "sm_version": None,
        "updated_at": "2026-07-20T00:00:00",
        "done": [],
        "failed": [],
        "attempted": [],
    }
    document.update(overrides)
    return document


def test_identity_fields_carry_the_lane_key():
    assert "collector_framework" in collect_mod._CHECKPOINT_IDENTITY_FIELDS
    assert "collector_framework" not in collect_mod._LEGACY_CHECKPOINT_IDENTITY_FIELDS


def test_new_checkpoints_record_their_lane(tmp_path):
    checkpoint = collect_mod.ResumeCheckpoint(
        backend="vllm",
        module_name="vllm.gemm",
        run_func_name="run_gemm",
        checkpoint_dir=str(tmp_path),
        framework_version="0.28.0",
        sm_version=None,
        collector_framework="vllm_npu",
    )
    assert checkpoint._metadata["collector_framework"] == "vllm_npu"


def test_lane_defaults_to_the_backend_when_unset(tmp_path):
    checkpoint = collect_mod.ResumeCheckpoint(
        backend="sglang",
        module_name="sglang.gemm",
        run_func_name="run_gemm",
        checkpoint_dir=str(tmp_path),
        framework_version="0.5.14",
        sm_version=100,
    )
    assert checkpoint._metadata["collector_framework"] == "sglang"


def test_legacy_document_is_migrated_to_the_xpu_lane():
    identity = collect_mod._checkpoint_identity_from_document(_legacy_document())
    assert identity["collector_framework"] == "vllm_xpu"

    cuda_identity = collect_mod._checkpoint_identity_from_document(_legacy_document(sm_version=90))
    assert cuda_identity["collector_framework"] == "vllm"


def test_explicit_lane_in_a_document_is_never_overridden():
    document = _legacy_document(collector_framework="vllm_npu")
    identity = collect_mod._checkpoint_identity_from_document(document)
    assert identity["collector_framework"] == "vllm_npu"


def test_legacy_checkpoint_file_still_resumes(tmp_path):
    checkpoint_dir = tmp_path / ".collector_checkpoint"
    _write_checkpoint(
        checkpoint_dir / "vllm" / "vllm.gemm.json",
        _legacy_document(done=["case-1"], attempted=["case-1"]),
    )
    tracker = collect_mod.ResumeCheckpoint(
        backend="vllm",
        module_name="vllm.gemm",
        run_func_name="run_gemm",
        checkpoint_dir=str(checkpoint_dir),
        framework_version="0.28.0",
        sm_version=None,
        collector_framework="vllm_xpu",
    )
    tracker.load_existing()
    assert tracker._done == {"case-1"}


def test_legacy_checkpoint_is_refused_by_a_different_lane(tmp_path):
    """The exact ambiguity that motivated the field: an XPU-era checkpoint must
    not be silently adopted by an NPU run that shares backend+sm_version."""
    checkpoint_dir = tmp_path / ".collector_checkpoint"
    _write_checkpoint(
        checkpoint_dir / "vllm" / "vllm.gemm.json",
        _legacy_document(done=["case-1"], attempted=["case-1"]),
    )
    tracker = collect_mod.ResumeCheckpoint(
        backend="vllm",
        module_name="vllm.gemm",
        run_func_name="run_gemm",
        checkpoint_dir=str(checkpoint_dir),
        framework_version="0.28.0",
        sm_version=None,
        collector_framework="vllm_npu",
    )
    with pytest.raises(RuntimeError, match="collector_framework"):
        tracker.load_existing()


def test_registered_table_uses_the_lane_not_the_missing_sm_version():
    xpu_identity = collect_mod._checkpoint_identity_from_document(_legacy_document(collector_framework="vllm_xpu"))
    assert collect_mod._registered_checkpoint_table(xpu_identity, backend="vllm") == "gemm_perf"

    # The NPU lane resolves against REGISTRY_NPU — its own gemm entry, never
    # the XPU one.
    npu_identity = collect_mod._checkpoint_identity_from_document(_legacy_document(collector_framework="vllm_npu"))
    assert collect_mod._registered_checkpoint_table(npu_identity, backend="vllm") == "gemm_perf"


def test_registered_table_rejects_a_lane_from_another_backend():
    identity = collect_mod._checkpoint_identity_from_document(_legacy_document(collector_framework="sglang"))
    with pytest.raises(RuntimeError, match="invalid checkpoint producer identity"):
        collect_mod._registered_checkpoint_table(identity, backend="vllm")


def test_legacy_identity_record_is_accepted_and_migrated():
    legacy_record = {field: _legacy_document()[field] for field in collect_mod._LEGACY_CHECKPOINT_IDENTITY_FIELDS}
    migrated = collect_mod._normalized_identity_record(legacy_record)
    assert migrated["collector_framework"] == "vllm_xpu"
    assert collect_mod._normalized_identity_record({"backend": "vllm"}) is None
    assert collect_mod._normalized_identity_record("not-a-mapping") is None
