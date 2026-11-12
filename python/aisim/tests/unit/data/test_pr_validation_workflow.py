# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static trust-model checks for the public PR-validation workflow."""

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = Path(__file__).resolve().parents[5] / ".github/workflows/pr-validation.yml"


@pytest.fixture(scope="module")
def text():
    if not WORKFLOW.is_file():
        pytest.skip("workflow not present in this checkout")
    return WORKFLOW.read_text()


def test_dispatch_only_and_read_only(text):
    doc = yaml.safe_load(text)
    triggers = doc.get("on", doc.get(True))
    assert set(triggers) == {"workflow_dispatch"}
    assert doc["permissions"] == {"contents": "read"}
    assert doc["jobs"]["validate"]["permissions"] == {"contents": "read", "pull-requests": "read"}


def test_no_secrets_or_token_leaks(text):
    assert "secrets." not in text and "HF_TOKEN" not in text
    assert text.count("github.token") == 3  # API steps only, never the evaluation step


def test_actions_pinned_to_commit(text):
    uses = re.findall(r"uses:\s*(\S+)", text)
    assert uses and all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u) for u in uses)


def test_pr_code_is_never_checked_out_or_interpolated(text):
    assert "pull_request_target" not in text and "head.ref" not in text
    assert "ref:" not in text.replace("?ref=", "")  # checkout stays on the default branch
    for line in text.splitlines():
        if line.strip().startswith(("run:", "python", "gh ")):
            assert "${{ inputs" not in line
    assert "persist-credentials: false" in text
