# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden forward-pass estimator resolution for Sweeper candidates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import aisimulate.sweeper.forward_pass_estimator as forward_pass_estimator_mod
from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.forward_pass_estimator import (
    ForwardPassEstimatorResolutionError,
    resolve_forward_pass_estimator_specs,
)


def _space(systems_root, **overrides):
    values = {
        "model_name": "example/model",
        "hardware_sku": "example_system",
        "backend": ["vllm"],
        "systems_paths": [str(systems_root)],
    }
    values.update(overrides)
    return SearchSpace(**values)


def _stub_core(monkeypatch, systems_root, *, versions=("0.10.0", "0.11.0")):
    calls = []
    system_spec = {"data_dir": "data/example_system"}
    monkeypatch.setattr(
        forward_pass_estimator_mod,
        "get_model_config_from_model_path",
        lambda model: {"architecture": "ExampleForCausalLM"},
    )
    monkeypatch.setattr(
        forward_pass_estimator_mod.perf_database,
        "load_system_spec",
        lambda system, systems_paths=None: system_spec,
    )
    monkeypatch.setattr(
        forward_pass_estimator_mod.perf_database,
        "get_supported_databases",
        lambda systems_paths=None: {"example_system": {"vllm": list(versions)}},
    )
    monkeypatch.setattr(
        forward_pass_estimator_mod.perf_database,
        "get_latest_database_version",
        lambda system, backend, systems_paths=None: versions[-1] if versions else None,
    )

    def get_database_view(system, backend, version, **kwargs):
        calls.append((system, backend, version, kwargs))
        return SimpleNamespace(
            systems_root=str(systems_root),
            system_spec=system_spec,
        )

    monkeypatch.setattr(forward_pass_estimator_mod.perf_database, "get_database_view", get_database_view)
    return calls


def test_default_resolution_is_concrete_and_reproducible(monkeypatch, tmp_path):
    calls = _stub_core(monkeypatch, tmp_path)

    spec = resolve_forward_pass_estimator_specs(_space(tmp_path))["vllm"]

    assert spec.model_path == "example/model"
    assert spec.model_architecture == "ExampleForCausalLM"
    assert spec.system == "example_system"
    assert spec.backend == "vllm"
    assert spec.backend_version == "0.11.0"
    assert spec.database_mode == "SILICON"
    assert spec.transfer_policy == ("xshape", "xquant", "xprofile", "xop")
    assert spec.forward_model == "op_level"
    assert spec.systems_paths == (str(tmp_path),)
    assert spec.performance_data_root == str(tmp_path)
    assert calls[0][3]["database_mode"] == "SILICON"
    assert calls[0][3]["allow_missing_data"] is False


def test_pinned_version_mode_and_transfer_policy_reach_database_view(monkeypatch, tmp_path):
    calls = _stub_core(monkeypatch, tmp_path)
    monkeypatch.setattr(
        forward_pass_estimator_mod.perf_database,
        "get_latest_database_version",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a pinned backend version must not resolve latest")
        ),
    )
    search = _space(
        tmp_path,
        backend_version="0.10.0",
        database_mode="HYBRID",
        transfer_policy="balanced,xop",
    )

    spec = resolve_forward_pass_estimator_specs(search)["vllm"]

    assert spec.backend_version == "0.10.0"
    assert spec.database_mode == "HYBRID"
    assert spec.transfer_policy == ("xshape", "xquant", "xop")
    assert calls[0][2] == "0.10.0"
    assert calls[0][3]["database_mode"] == "HYBRID"
    assert calls[0][3]["transfer_policy"] == ["xshape", "xquant", "xop"]
    assert calls[0][3]["allow_missing_data"] is True


def test_invalid_transfer_policy_fails_before_database_load(monkeypatch, tmp_path):
    calls = _stub_core(monkeypatch, tmp_path)

    with pytest.raises(ForwardPassEstimatorResolutionError, match="transfer_policy"):
        resolve_forward_pass_estimator_specs(_space(tmp_path, transfer_policy="mystery"))

    assert calls == []


def test_unknown_pinned_version_fails_before_database_load(monkeypatch, tmp_path):
    calls = _stub_core(monkeypatch, tmp_path)

    with pytest.raises(ForwardPassEstimatorResolutionError, match="unsupported backend_version"):
        resolve_forward_pass_estimator_specs(_space(tmp_path, backend_version="9.9.9"))

    assert calls == []


def test_fpm_requires_exact_data_pair(monkeypatch, tmp_path):
    _stub_core(monkeypatch, tmp_path)
    search = _space(
        tmp_path,
        backend_version="0.11.0",
        forward_model="fpm",
    )

    with pytest.raises(ForwardPassEstimatorResolutionError, match="requires fpm_forward_perf"):
        resolve_forward_pass_estimator_specs(search)

    version_dir = tmp_path / "data/example_system/dense/vllm/0.11.0"
    version_dir.mkdir(parents=True)
    (version_dir / "fpm_forward_perf.parquet").touch()
    (version_dir / "fpm_forward_perf.metadata.json").write_text("{}")

    spec = resolve_forward_pass_estimator_specs(search)["vllm"]
    assert spec.forward_model == "fpm"


def test_fpm_rejects_mtp_before_search(monkeypatch, tmp_path):
    _stub_core(monkeypatch, tmp_path)
    version_dir = tmp_path / "data/example_system/dense/vllm/0.11.0"
    version_dir.mkdir(parents=True)
    (version_dir / "fpm_forward_perf.parquet").touch()
    (version_dir / "fpm_forward_perf.metadata.json").write_text("{}")

    with pytest.raises(ForwardPassEstimatorResolutionError, match="does not support aic_nextn"):
        resolve_forward_pass_estimator_specs(
            _space(
                tmp_path,
                backend_version="0.11.0",
                forward_model="fpm",
                aic_nextn=2,
            )
        )


def test_invalid_system_path_fails_concisely(tmp_path):
    with pytest.raises(ForwardPassEstimatorResolutionError, match="not an existing directory"):
        resolve_forward_pass_estimator_specs(_space(tmp_path / "missing"))
