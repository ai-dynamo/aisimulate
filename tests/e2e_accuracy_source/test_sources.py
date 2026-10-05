# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from types import SimpleNamespace

import pytest
import requests
from e2e_accuracy_source import sources
from e2e_accuracy_source.recipes.inferencex_recipe import InferenceXRecipeError


def test_source_verification_preserves_records_and_checks_every_file(monkeypatch):
    content = b"reviewed source"
    records = [
        {"url": url, "sha256": hashlib.sha256(content).hexdigest(), "path": "upstream.py"}
        for url in ("https://example.com/first", "https://example.com/second")
    ]
    calls = []

    def get(url, *, timeout):
        calls.append((url, timeout))
        return SimpleNamespace(content=content, raise_for_status=lambda: None)

    monkeypatch.setattr(sources.requests, "get", get)
    assert sources.verify_sources(records) is records
    assert calls == [(record["url"], 30) for record in records]
    records[1]["sha256"] = "0" * 64
    with pytest.raises(InferenceXRecipeError, match="reviewed benchmark source changed: https://example.com/second"):
        sources.verify_sources(records, kind="benchmark")


def test_http_failure_is_not_accepted_as_reviewed_source(monkeypatch):
    def get(*args, **kwargs):
        raise requests.HTTPError("source unavailable")

    monkeypatch.setattr(sources.requests, "get", get)
    with pytest.raises(requests.HTTPError, match="source unavailable"):
        sources.verify_sources([{"url": "https://example.com/source", "sha256": "0" * 64}])


def test_manifests_load_outside_repository_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    sources.load_manifest.cache_clear()
    sources.manifest_text.cache_clear()
    for name in (
        "framework_default_sources.json",
        "launcher_recipe_sources.json",
        "launcher_workload_sources.json",
        "sglang_additional_default_sources.json",
        "trt_additional_default_sources.json",
    ):
        manifest = sources.load_manifest(name)
        assert manifest
        assert sources.load_manifest(name) is manifest
