# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static checks of the mirror CI template: same validator, no secrets, never runs here."""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

TEMPLATE = Path(__file__).resolve().parents[5] / ".github/mirror/aisim-data-internal-validate.yml"


@pytest.fixture(scope="module")
def text():
    if not TEMPLATE.is_file():
        pytest.skip("template not present in this checkout")
    return TEMPLATE.read_text()


def test_template_is_not_an_active_workflow_here(text):
    assert TEMPLATE.parent.name == "mirror"
    assert "github.repository == 'ai-dynamo/aisim-data-internal'" in text


def test_runs_the_shared_validator_from_a_pinned_checkout(text):
    doc = yaml.safe_load(text)
    run = "\n".join(s.get("run", "") for s in doc["jobs"]["validate"]["steps"])
    assert "aisim_core.data.mirror validate" in run and "--profile mirror" in run
    assert re.fullmatch(r"[0-9a-f]{40}", doc["env"]["AISIM_VALIDATOR_REF"])
    assert 'AISIM_VALIDATOR_REF}" != 0000*' in text  # the placeholder SHA fails closed


def test_least_privilege_and_pinned_actions(text):
    doc = yaml.safe_load(text)
    assert doc["permissions"] == {"contents": "read"}
    assert "secrets." not in text and "pull_request_target" not in text
    assert "persist-credentials: false" in text
    assert all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u) for u in re.findall(r"uses:\s*(\S+)", text))
