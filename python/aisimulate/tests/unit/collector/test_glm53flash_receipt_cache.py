# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared callback parsing preserves complete graph evidence and rejects drift."""

import hashlib
import json
import os

import pytest
from collector.glm53flash_receipt_cache import ReceiptCache

pytestmark = pytest.mark.unit


def receipt(tmp_path):
    path = tmp_path / "callbacks.json"
    raw = b'{"callbacks":[{"kind":"TEST_ONLY"}]}\n'
    path.write_bytes(raw)
    return path, {"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def test_shared_receipt_is_parsed_once_per_scope_and_reverified(tmp_path, monkeypatch):
    path, reference = receipt(tmp_path)
    loads = json.loads
    parsed = []

    def counted(raw):
        parsed.append(raw)
        return loads(raw)

    monkeypatch.setattr(json, "loads", counted)
    files = set()
    cache = ReceiptCache(tmp_path, files)
    first = cache.read(reference)
    for _ in range(200):
        assert cache.read(reference) is first
    cache.verify()
    assert parsed == [path.read_bytes()]
    assert files == {path.name}
    other = ReceiptCache(tmp_path, set())
    assert other.read(reference) == first
    other.verify()
    assert len(parsed) == 2


@pytest.mark.parametrize("when", ["reuse", "final"])
@pytest.mark.parametrize("defect", ["same_size", "replaced", "symlink", "removed"])
def test_shared_receipt_cannot_change_after_first_read(tmp_path, when, defect):
    path, reference = receipt(tmp_path)
    cache = ReceiptCache(tmp_path, set())
    cache.read(reference)
    original = path.stat()
    if defect == "same_size":
        path.write_bytes(path.read_bytes().replace(b"TEST_ONLY", b"MUTATED!!"))
        os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    elif defect == "replaced":
        replacement = tmp_path / "replacement.json"
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    elif defect == "symlink":
        target = tmp_path / "target.json"
        path.rename(target)
        path.symlink_to(target)
    else:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        cache.read(reference) if when == "reuse" else cache.verify()


@pytest.mark.parametrize("cached", [False, True])
def test_each_shared_reference_requires_same_original_hash(tmp_path, cached):
    _, reference = receipt(tmp_path)
    cache = ReceiptCache(tmp_path, set())
    if cached:
        cache.read(reference)
    with pytest.raises(ValueError):
        cache.read({**reference, "sha256": "0" * 64})


def test_final_verification_checks_content_even_if_stat_identity_matches(tmp_path, monkeypatch):
    from collector import glm53flash_receipt_cache as module

    path, reference = receipt(tmp_path)
    cache = ReceiptCache(tmp_path, set())
    cache.read(reference)
    identity = module._identity(path.stat())
    path.write_bytes(path.read_bytes().replace(b"TEST_ONLY", b"MUTATED!!"))
    monkeypatch.setattr(module, "_identity", lambda _: identity)
    with pytest.raises(ValueError, match="changed during validation"):
        cache.verify()


def test_complete_piecewise_result_equals_uncached_original_receipt_path(tmp_path, monkeypatch):
    from collector import glm53flash_receipt_cache as cache_module
    from collector import glm53flash_vllm_serving_export as serving
    from collector.glm53flash_graph_export import _receipt

    from .test_glm53flash_vllm_serving_export import piecewise_files

    policy, manifest, provenance = piecewise_files(tmp_path)
    uncached_calls = []

    class Uncached:
        def __init__(self, root, files):
            self.root, self.files = root, files

        def read(self, reference):
            uncached_calls.append(reference)
            return _receipt(self.root, reference, self.files)

        def verify(self):
            pass

    with monkeypatch.context() as patch:
        patch.setattr(cache_module, "ReceiptCache", Uncached)
        old_files = set()
        old = serving._piecewise_captures(tmp_path, 0, policy, manifest, provenance, old_files, {})
    new_files = set()
    new = serving._piecewise_captures(tmp_path, 0, policy, manifest, provenance, new_files, {})
    assert len(uncached_calls) == 6
    assert new == old
    assert new_files == old_files


def test_piecewise_final_check_rejects_last_segment_mutation(tmp_path, monkeypatch):
    from collector import glm53flash_graph_callbacks as callbacks
    from collector import glm53flash_vllm_serving_export as serving

    from .test_glm53flash_vllm_serving_export import piecewise_files

    policy, manifest, provenance = piecewise_files(tmp_path)
    resolve, count = callbacks.resolve_registry, 0

    def changing(*args, **kwargs):
        nonlocal count
        result = resolve(*args, **kwargs)
        count += 1
        if count == 6:
            path = next(tmp_path.glob("*-piecewise-callbacks.json"))
            path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(callbacks, "resolve_registry", changing)
    with pytest.raises(ValueError, match="shared native receipt changed"):
        serving._piecewise_captures(tmp_path, 0, policy, manifest, provenance, set(), {})
