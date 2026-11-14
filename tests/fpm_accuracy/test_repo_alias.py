# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FPM accuracy consumers accept both public repository names (AISIM-2339)."""

from __future__ import annotations

import pytest

from aisim_core.data import repos
from scripts.fpm_accuracy import contract
from scripts.fpm_accuracy.hf import HfDataset
from tests.fpm_accuracy.test_hf_dataset import REVISION
from tests.fpm_accuracy.test_hf_unified import _legacy_and_unified


def test_script_constants_agree_with_the_library_alias_table():
    assert (contract.HF_REPO, contract.LEGACY_HF_REPO) == (repos.PUBLIC_REPO, repos.RETIRED_FPM_REPO)
    assert contract.HF_REPOS == (repos.PUBLIC_REPO, repos.RETIRED_FPM_REPO)
    assert repos.SUCCESSORS == {contract.LEGACY_HF_REPO: contract.HF_REPO}


@pytest.mark.parametrize("requested", [repos.PUBLIC_REPO, repos.RETIRED_FPM_REPO])
def test_old_and_new_names_load_the_same_catalog_for_either_layout(tmp_path, requested):
    legacy_root, unified_root = _legacy_and_unified(tmp_path)  # legacy catalog names the retired repo
    old = HfDataset.from_local(legacy_root, revision=REVISION, repo_id=requested)
    new = HfDataset.from_local(unified_root, revision=REVISION, repo_id=requested)
    assert (old.layout, new.layout) == ("legacy", "unified")
    assert [c.configuration_id for c in old.configurations()] == [c.configuration_id for c in new.configurations()]
