# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

CHECK_COPYRIGHT = Path(__file__).resolve().parents[1] / "scripts" / "check_copyright.py"


@pytest.fixture
def checker():
    spec = importlib.util.spec_from_file_location("check_copyright", CHECK_COPYRIGHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["app.js", "runtime.pyi"])
def test_source_suffixes_cover_javascript_and_python_stubs(checker, tmp_path, name):
    path = tmp_path / name
    path.write_text("source\n")

    assert checker.is_source(path)


def test_source_detection_covers_extensionless_shebang_script(checker, tmp_path):
    path = tmp_path / "ibstat"
    path.write_text("#!/usr/bin/env bash\n")

    assert checker.is_source(path)


def test_hash_stamped_patch_is_not_rewritten_as_owned_source(checker, tmp_path):
    path = tmp_path / "upstream.patch"
    path.write_text("diff --git a/a b/a\n")

    assert not checker.is_source(path)
