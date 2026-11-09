# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

AISIMULATE_ROOT = Path(__file__).resolve().parents[2]
PAGES_ROOT = next(root / "pages" for root in (AISIMULATE_ROOT, *AISIMULATE_ROOT.parents) if (root / "pages").is_dir())
PACKAGE_README = AISIMULATE_ROOT / "README.md"


def test_landing_page_is_aisim_branded_and_hides_outdated_universe():
    page = (PAGES_ROOT / "index.html").read_text()

    assert "<title>AISim</title>" in page
    assert ">AISim</h1>" in page
    assert 'href="./e2e-accuracy/"' in page
    assert 'href="./fpe-support-matrix/"' in page
    assert 'href="./feature-support-matrix/"' in page
    assert "Legacy AIC Support Matrix" in page
    assert "Prefer the FPE Support Matrix" in page
    assert "https://github.com/ai-dynamo/aisim" in page
    assert "ai-dynamo/aiconfigurator" not in page
    assert "Architecture Universe" not in page
    assert 'href="./universe/"' not in page


def test_support_matrix_uses_aisim_navigation_and_data():
    page = (PAGES_ROOT / "support-matrix" / "index.html").read_text()

    assert "<title>AISim — Legacy AIC Support Matrix</title>" in page
    assert "const DEPLOYED_SUPPORT_MATRIX_PATH = '../data/support-matrix';" in page
    assert "DATA_REPO" not in page
    assert "PUBLIC_DATA" not in page
    assert "raw.githubusercontent.com" not in page
    assert "python/aisim/src/aisim_core/systems/support_matrix" in page
    assert "Release branches..." not in page
    assert "matching-refs/heads/release" not in page
    assert "HARDCODED_RELEASE_BRANCHES" not in page
    assert 'href="../">AISim</a>' in page
    assert 'href="../fpe-support-matrix/"' in page
    assert "FPE coverage is estimator evidence, not deployment certification." in page
    assert "https://github.com/ai-dynamo/aisim" in page
    assert "ai-dynamo/aiconfigurator" not in page
    assert 'href="/aiconfigurator/"' not in page
    assert "AISim Support Matrix" not in page
    assert "AI Configurator Support Matrix" not in page


def test_fpe_support_matrix_uses_packaged_pages_data():
    page = (PAGES_ROOT / "fpe-support-matrix" / "index.html").read_text()

    assert "<title>AISim — FPE Support Matrix</title>" in page
    assert "const DEPLOYED_SUPPORT_MATRIX_PATH = '../data/fpe-support-matrix';" in page
    assert "raw.githubusercontent.com" not in page
    assert "api.github.com/repos/ai-dynamo/aisim" not in page
    assert "matching-refs/heads/release" not in page
    assert 'href="../">AISim</a>' in page
    assert 'htmlFor="fpe-branch"' in page
    assert "Branch:</label>" in page
    assert '<select id="fpe-branch"' in page
    assert "${DEPLOYED_SUPPORT_MATRIX_PATH}/branches.json" in page


def test_package_readme_only_exposes_current_static_page_entrypoints():
    readme = PACKAGE_README.read_text()

    assert "AIC Developer Universe" not in readme
    assert "ai-dynamo.github.io/aiconfigurator" not in readme
    assert "[Website](https://ai-dynamo.org/aisim/)" in readme
    assert "[FPE Support Matrix](https://ai-dynamo.org/aisim/fpe-support-matrix/)" in readme
    assert "[Feature Support Matrix](https://ai-dynamo.org/aisim/feature-support-matrix/)" in readme
    assert "[E2E Accuracy Overview](https://ai-dynamo.org/aisim/e2e-accuracy/)" in readme
    assert "release branches" not in readme
    assert "[FPM Accuracy Overview](https://ai-dynamo.org/aisim/fpm-accuracy/)" in readme


def test_support_matrix_pages_state_which_versions_the_cli_accepts():
    # Both matrices list backend versions outside the queryable slots; a PASS
    # there is not runnable from predict/recommend without saying so.
    for page_dir in ("support-matrix", "fpe-support-matrix"):
        page = (PAGES_ROOT / page_dir / "index.html").read_text()
        assert "function CliVersionNotice()" in page
        assert "<CliVersionNotice />" in page
        assert "Versions the CLI accepts." in page
        assert "<code>aisim predict</code> and <code>aisim recommend</code> accept only" in page
        assert "python/aisim/src/aisim_core/systems/query_versions.yaml" in page
        assert "<code>backend_version: current</code>" in page
