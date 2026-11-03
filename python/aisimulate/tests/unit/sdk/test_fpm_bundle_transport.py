# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from aisimulate_core.sdk import rust_engine_step

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("dataclass_input", [False, True])
def test_unresolved_online_learned_source_receives_existing_ordered_roots(monkeypatch, dataclass_input):
    roots = ["/first-root", "/second-root"]
    monkeypatch.setattr(rust_engine_step, "_resolve_forward_pass_systems_paths", lambda *args, **kwargs: roots)
    payload = {
        "model": "test",
        "system": "test",
        "backend": "vllm",
        "worker_type": "decode",
        "estimation_mode": "fpm_learned",
    }
    value = rust_engine_step.ForwardPassPerfModelConfig(**payload) if dataclass_input else payload
    resolved = json.loads(rust_engine_step._forward_pass_config_json(value))
    assert resolved["systems_paths"] == roots
    assert resolved["estimation_mode"] == "fpm_learned"
    assert "bundle_path" not in resolved.get("estimator_config", {}).get("fpm_learned", {})


@pytest.mark.parametrize(
    "learned",
    [
        {"source_policy": "explicit_only"},
        {"bundle_path": "/bundle/bundle.json"},
        {"checkpoint_path": "/checkpoint"},
        {"artifact_path": "/artifact.json", "adaptation": "frozen"},
        {"resolved_source": {"kind": "none"}},
    ],
)
@pytest.mark.parametrize("dataclass_input", [False, True])
def test_explicit_or_pinned_learned_sources_do_not_expand_unneeded_roots(monkeypatch, learned, dataclass_input):
    def fail(*args, **kwargs):
        raise AssertionError("source-pinned construction must not inspect default roots")

    monkeypatch.setattr(rust_engine_step, "_resolve_forward_pass_systems_paths", fail)
    payload = {
        "model": "test",
        "system": "test",
        "backend": "vllm",
        "worker_type": "decode",
        "estimation_mode": "fpm_learned",
        "estimator_config": {"fpm_learned": learned},
    }
    value = rust_engine_step.ForwardPassPerfModelConfig(**payload) if dataclass_input else payload
    resolved = json.loads(rust_engine_step._forward_pass_config_json(value))
    assert not resolved.get("systems_paths")
    assert resolved["estimator_config"]["fpm_learned"] == learned


def test_explicit_learned_roots_are_retained_for_prior_dependencies(monkeypatch):
    calls = []

    def resolve(entries, **kwargs):
        calls.append(entries)
        return list(entries)

    monkeypatch.setattr(rust_engine_step, "_resolve_forward_pass_systems_paths", resolve)
    payload = {
        "estimation_mode": "fpm_learned",
        "systems_paths": ["/prior-root"],
        "estimator_config": {"fpm_learned": {"bundle_path": "/seed/bundle.json"}},
    }
    assert json.loads(rust_engine_step._forward_pass_config_json(payload))["systems_paths"] == ["/prior-root"]
    assert calls == [("/prior-root",)]


def test_dependency_info_is_a_native_passthrough():
    expected = {
        "systems_paths": ["/native-prior"],
        "files": [{"path": "/seed", "sha256": "1" * 64, "byte_length": 12, "purpose": "seed"}],
    }
    model = rust_engine_step.RustForwardPassPerfModel(
        SimpleNamespace(dependency_info_json=lambda: json.dumps(expected))
    )
    assert model.dependency_info() == expected
