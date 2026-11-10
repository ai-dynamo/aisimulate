# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact platform builds share one audited native source, never a version range."""

import copy

import pytest
from collector import framework_manifest as manifest
from collector.sglang import collect_kimi_k31_dsa_v1 as native
from collector.sglang.registry import REGISTRY
from collector.version_resolver import resolve_module

pytestmark = pytest.mark.unit

BUILDS = {
    "x86_64": "0.0.0.dev1+g41208373c.d20261104",
    "aarch64": "0.0.0.dev1+ge79de62d2.d20261104",
}
OPS = {"kimi_k31_dsa_context_module", "kimi_k31_dsa_generation_module"}


@pytest.mark.parametrize("machine", BUILDS)
def test_platform_selects_only_its_exact_build(monkeypatch, machine):
    declared = manifest.load_manifest()["frameworks"]["sglang"]["models"][native.MODEL]["platform_versions"]
    assert declared == native.RUNTIME_VERSIONS == BUILDS
    monkeypatch.setattr(manifest.platform, "machine", lambda: machine)
    runtime = manifest.require_collector_runtime("sglang", BUILDS[machine], requested_ops=OPS, model_path=native.MODEL)
    assert runtime.version == BUILDS[machine]
    assert runtime.source_commit == native.SOURCE_COMMIT
    assert runtime.image().endswith("67fc77d1cbe67de27ca54d874c82757a4844743fea899ea7d0df547e30c2c391")
    for entry in REGISTRY:
        if entry.op in OPS:
            assert resolve_module(entry, runtime.version) == native.__name__
            assert {90, 100} <= set(entry.unverified_sms)
    wrong = next(version for arch, version in BUILDS.items() if arch != machine)
    for version in (wrong, "0.0.0.dev1", "0.0.0.dev1+unknown"):
        with pytest.raises(RuntimeError, match="requires exactly"):
            manifest.require_collector_runtime("sglang", version, requested_ops=OPS, model_path=native.MODEL)


def test_unknown_platform_has_no_default_build_fallback(monkeypatch):
    monkeypatch.setattr(manifest.platform, "machine", lambda: "unverified_machine")
    with pytest.raises(RuntimeError, match="no package build"):
        manifest.require_collector_runtime("sglang", BUILDS["x86_64"], requested_ops=OPS, model_path=native.MODEL)


@pytest.mark.parametrize("builds", [[], {}, {"x86_64": ""}, {"x86_64": "sglang>=1"}, {"aarch64": BUILDS["aarch64"]}])
def test_platform_map_rejects_inexact_or_missing_default_identity(builds):
    contents = copy.deepcopy(manifest.load_manifest())
    contents["frameworks"]["sglang"]["models"][native.MODEL]["platform_versions"] = builds
    with pytest.raises(ValueError, match="platform_versions"):
        manifest.validate_manifest(contents)


def test_platform_map_requires_immutable_source():
    contents = copy.deepcopy(manifest.load_manifest())
    del contents["frameworks"]["sglang"]["models"][native.MODEL]["source_commit"]
    with pytest.raises(ValueError, match="immutable source_commit"):
        manifest.validate_manifest(contents)


def test_standalone_rejects_other_package_before_framework_import(monkeypatch):
    monkeypatch.setattr(native.importlib.metadata, "version", lambda _: "0.0.0.dev1+unknown")
    with pytest.raises(RuntimeError, match="requires exactly"):
        native._assert_runtime()
