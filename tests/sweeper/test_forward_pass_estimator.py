# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden Core-owned forward-pass estimator resolution for Sweeper candidates."""

from __future__ import annotations

from dataclasses import asdict

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

    class _Model:
        def __init__(self, diagnostics):
            self._diagnostics = diagnostics

        def diagnostics(self):
            return self._diagnostics

        def close(self):
            return None

    class _Core:
        @staticmethod
        def best_available(model_config, tuning_config):
            calls.append((model_config, tuning_config))
            if (
                model_config.backend_version is not None
                and model_config.backend_version not in versions
            ):
                raise ValueError(
                    f"unsupported backend_version {model_config.backend_version!r}"
                )
            if model_config.transfer_policy == "mystery":
                raise ValueError("invalid transfer_policy 'mystery'")
            if model_config.forward_model == "fpm" and model_config.nextn:
                raise ValueError("forward_model='fpm' does not support aic_nextn/MTP")
            if model_config.forward_model == "fpm":
                complete = any(
                    path.name == "fpm_forward_perf.parquet"
                    and (path.parent / "fpm_forward_perf.metadata.json").is_file()
                    for path in systems_root.rglob("fpm_forward_perf.parquet")
                )
                if not complete:
                    raise ValueError(
                        "forward_model='fpm' requires fpm_forward_perf data"
                    )

            resolved = asdict(model_config)
            resolved["backend_version"] = model_config.backend_version or (
                versions[-1] if versions else None
            )
            if model_config.transfer_policy is None:
                resolved["transfer_policy"] = ["xshape", "xquant", "xprofile", "xop"]
            elif model_config.transfer_policy == "balanced,xop":
                resolved["transfer_policy"] = ["xshape", "xquant", "xop"]
            else:
                resolved["transfer_policy"] = list(model_config.transfer_policy)
            resolved["systems_paths"] = [str(systems_root)]
            return _Model(
                {
                    "source": "aic",
                    "readiness": "ready",
                    "provenance": {
                        "config": resolved,
                        "selected_systems_root": str(systems_root),
                    },
                }
            )

    monkeypatch.setattr(forward_pass_estimator_mod, "RustForwardPassPerfModel", _Core)
    return calls


def test_default_resolution_is_core_owned_concrete_and_reproducible(
    monkeypatch, tmp_path
):
    calls = _stub_core(monkeypatch, tmp_path)

    spec = resolve_forward_pass_estimator_specs(_space(tmp_path))["vllm"]

    assert spec.model_path == "example/model"
    assert spec.system == "example_system"
    assert spec.backend == "vllm"
    assert spec.backend_version == "0.11.0"
    assert spec.database_mode == "SILICON"
    assert spec.transfer_policy == ("xshape", "xquant", "xprofile", "xop")
    assert spec.forward_model == "op_level"
    assert spec.systems_paths == (str(tmp_path),)
    assert spec.performance_data_root == str(tmp_path)
    assert calls[0][0].backend_version is None
    assert spec.config == spec.diagnostics["provenance"]["config"]


def test_pinned_policy_and_tuning_config_reach_the_canonical_constructor(
    monkeypatch, tmp_path
):
    calls = _stub_core(monkeypatch, tmp_path)
    search = _space(
        tmp_path,
        backend_version="0.10.0",
        database_mode="HYBRID",
        transfer_policy="balanced,xop",
        forward_pass_tuning_config={"min_observations": 3},
    )

    spec = resolve_forward_pass_estimator_specs(search)["vllm"]

    request, tuning_config = calls[0]
    assert request.backend_version == "0.10.0"
    assert request.database_mode == "HYBRID"
    assert request.transfer_policy == "balanced,xop"
    assert tuning_config is not None and tuning_config.min_observations == 3
    assert spec.transfer_policy == ("xshape", "xquant", "xop")
    assert spec.tuning_config is not None
    assert spec.tuning_config["min_observations"] == 3


def test_invalid_transfer_policy_fails_through_core(monkeypatch, tmp_path):
    calls = _stub_core(monkeypatch, tmp_path)

    with pytest.raises(ForwardPassEstimatorResolutionError, match="transfer_policy"):
        resolve_forward_pass_estimator_specs(
            _space(tmp_path, transfer_policy="mystery")
        )

    assert len(calls) == 1


def test_unknown_pinned_version_fails_through_core(monkeypatch, tmp_path):
    calls = _stub_core(monkeypatch, tmp_path)

    with pytest.raises(
        ForwardPassEstimatorResolutionError, match="unsupported backend_version"
    ):
        resolve_forward_pass_estimator_specs(_space(tmp_path, backend_version="9.9.9"))

    assert len(calls) == 1


def test_fpm_support_is_validated_by_core_before_search(monkeypatch, tmp_path):
    _stub_core(monkeypatch, tmp_path)
    search = _space(tmp_path, backend_version="0.11.0", forward_model="fpm")

    with pytest.raises(
        ForwardPassEstimatorResolutionError, match="requires fpm_forward_perf"
    ):
        resolve_forward_pass_estimator_specs(search)

    version_dir = tmp_path / "data/example_system/dense/vllm/0.11.0"
    version_dir.mkdir(parents=True)
    (version_dir / "fpm_forward_perf.parquet").touch()
    (version_dir / "fpm_forward_perf.metadata.json").write_text("{}")

    spec = resolve_forward_pass_estimator_specs(search)["vllm"]
    assert spec.forward_model == "fpm"


def test_fpm_rejects_mtp_through_core_before_search(monkeypatch, tmp_path):
    _stub_core(monkeypatch, tmp_path)

    with pytest.raises(
        ForwardPassEstimatorResolutionError, match="does not support aic_nextn"
    ):
        resolve_forward_pass_estimator_specs(
            _space(
                tmp_path,
                backend_version="0.11.0",
                forward_model="fpm",
                aic_nextn=2,
            )
        )


def test_invalid_system_path_fails_concisely(tmp_path):
    with pytest.raises(
        ForwardPassEstimatorResolutionError, match="not an existing directory"
    ):
        resolve_forward_pass_estimator_specs(_space(tmp_path / "missing"))
