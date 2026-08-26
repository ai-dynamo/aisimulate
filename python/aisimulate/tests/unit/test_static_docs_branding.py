# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

AISIMULATE_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = AISIMULATE_ROOT / "docs"
PACKAGE_README = AISIMULATE_ROOT / "README.md"


def test_landing_page_is_aisimulate_branded_and_hides_outdated_universe():
    page = (DOCS_ROOT / "index.html").read_text()

    assert "<title>AISimulate</title>" in page
    assert ">AISimulate</h1>" in page
    assert "AISimulate Support Matrix" in page
    assert "https://github.com/ai-dynamo/aisimulate" in page
    assert "ai-dynamo/aiconfigurator" not in page
    assert "Architecture Universe" not in page
    assert 'href="./universe/"' not in page


def test_support_matrix_uses_aisimulate_navigation_and_data():
    page = (DOCS_ROOT / "support-matrix" / "index.html").read_text()

    assert "<title>AISimulate — Support Matrix</title>" in page
    assert "const REPO = 'ai-dynamo/aisimulate';" in page
    assert "const DATA_REPO = 'ai-dynamo/aiconfigurator';" in page
    assert "const PUBLIC_DATA_MAIN_REF = '2ed278a91bc599c5149ddfcd527a20c3421b471d';" in page
    assert "raw.githubusercontent.com/${DATA_REPO}" in page
    assert "api.github.com/repos/${DATA_REPO}" in page
    assert "aic-core/src/aiconfigurator_core/systems/support_matrix" in page
    assert "python/aisimulate/src/aiconfigurator_core/systems/support_matrix" in page
    assert 'href="../">AISimulate</a>' in page
    assert "https://github.com/ai-dynamo/aisimulate" in page
    assert 'href="https://github.com/ai-dynamo/aiconfigurator"' not in page
    assert 'href="/aiconfigurator/"' not in page
    assert "AI Configurator Support Matrix" not in page


def test_package_readme_only_exposes_current_static_page_entrypoints():
    readme = PACKAGE_README.read_text()

    assert "https://deepwiki.com/ai-dynamo/aisimulate" in readme
    assert "AIC Developer Universe" not in readme
    assert "ai-dynamo.github.io/aiconfigurator" not in readme
    assert "[AISimulate Support Matrix](docs/support-matrix/)" in readme
    assert "[per-system CSV files](src/aiconfigurator_core/systems/support_matrix)" in readme
