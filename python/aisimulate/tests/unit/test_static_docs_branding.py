from pathlib import Path


AISIMULATE_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = AISIMULATE_ROOT / "docs"


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
    assert "python/aisimulate/src/aiconfigurator_core/systems/support_matrix" in page
    assert 'href="../">AISimulate</a>' in page
    assert "https://github.com/ai-dynamo/aisimulate" in page
    assert "ai-dynamo/aiconfigurator" not in page
    assert 'href="/aiconfigurator/"' not in page
    assert "AI Configurator Support Matrix" not in page
