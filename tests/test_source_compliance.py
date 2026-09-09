# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "python" / "aisimulate"
CHECK_COPYRIGHT = ROOT / "scripts" / "check_copyright.py"

ROOT_ONLY_GOVERNANCE_FILES = (
    "AGENTS.md",
    "CODEOWNERS",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "CONTRIBUTORS.md",
    "DEVELOPMENT.md",
    "SECURITY.md",
)
PACKAGED_LEGAL_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md")
PACKAGED_README_ROOT_LINKS = (
    "DEVELOPMENT.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
)


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


@pytest.mark.parametrize(
    "line",
    [
        "# SPDX-License-Identifier: Apache-2.0",
        "// SPDX-License-Identifier: Apache-2.0",
        "/* SPDX-License-Identifier: Apache-2.0 */",
        "/** SPDX-License-Identifier: Apache-2.0 */",
        "* SPDX-License-Identifier: Apache-2.0",
        "* SPDX-License-Identifier: Apache-2.0 */",
        "SPDX-License-Identifier: Apache-2.0",
        "<!-- SPDX-License-Identifier: Apache-2.0 -->",
    ],
)
def test_apache_spdx_identifier_accepts_complete_tag_lines(checker, line):
    assert checker.has_required_license_identifier(f"{line}\n")


@pytest.mark.parametrize(
    "line",
    [
        "{# SPDX-License-Identifier: Apache-2.0 -#}",
        "{# SPDX-License-Identifier: Apache-2.0 #}",
        "SPDX-License-Identifier: Apache-2.0 -#}",
        "SPDX-License-Identifier: MIT",
        "/* SPDX-License-Identifier: Apache-2.0",
        "/** SPDX-License-Identifier: Apache-2.0",
        "<!-- SPDX-License-Identifier: Apache-2.0",
        "SPDX-License-Identifier: Apache-2.0 */",
        "SPDX-License-Identifier: Apache-2.0 -->",
        "/* SPDX-License-Identifier: Apache-2.0 -->",
        "<!-- SPDX-License-Identifier: Apache-2.0 */",
    ],
)
def test_apache_spdx_identifier_rejects_malformed_or_wrong_expressions(checker, line):
    assert not checker.has_required_license_identifier(f"{line}\n")


@pytest.mark.parametrize("name", ROOT_ONLY_GOVERNANCE_FILES)
def test_repository_governance_file_lives_only_at_root(name):
    assert (ROOT / name).is_file()
    assert not (PACKAGE_ROOT / name).exists()


@pytest.mark.parametrize("name", PACKAGED_LEGAL_FILES)
def test_wheel_local_legal_file_matches_root(name):
    canonical = ROOT / name
    packaged = PACKAGE_ROOT / name

    assert canonical.is_file()
    assert packaged.is_file()
    assert packaged.read_bytes() == canonical.read_bytes()


@pytest.mark.parametrize("name", PACKAGED_README_ROOT_LINKS)
def test_packaged_readme_links_root_governance_to_github(name):
    readme = (PACKAGE_ROOT / "README.md").read_text()

    assert f"https://github.com/ai-dynamo/aisimulate/blob/main/{name}" in readme
