# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host profile identity: matching covers the sampled workload, the digest covers the content."""

import pytest

from aisimulate.config.engine import CostFnConfig, FrontendPredictionConfig, FrontendStageConfig, HostPredictionConfig
from aisimulate.vl.profile import (
    SGLANG_REVISION,
    HostProfile,
    ProfileIdentity,
    ProfileImages,
    match_host_profile,
    profile_digest,
)

pytestmark = pytest.mark.unit


def _profile(**host) -> HostProfile:
    return HostProfile(
        identity=ProfileIdentity(
            sglang_revision=SGLANG_REVISION,
            model="Qwen/Qwen3-VL-8B-Instruct",
            frontend="python",
            images=ProfileImages(height=480, width=480, count=1, encoding="png"),
            text_tokens=128,
            processor="transformers.Qwen3VLProcessor",
            cpu="test-cpu",
            threads=8,
        ),
        host=HostPredictionConfig(**host),
        frontend=FrontendPredictionConfig(
            stages=[FrontendStageConfig(resource="tm_loop", unit="request", cost=CostFnConfig(const_ms=1.0))]
        ),
        provenance={"sampled_at": "now"},
    )


def test_matching_checks_the_whole_sampled_workload():
    profile = _profile()
    match_host_profile(
        profile, model="Qwen/Qwen3-VL-8B-Instruct", frontend="python", images={"height": 480, "width": 480}
    )
    with pytest.raises(ValueError) as error:
        match_host_profile(
            profile,
            model="Qwen/Qwen3-VL-8B-Instruct",
            frontend="rust",
            images={"height": 4096, "width": 4096, "count": 16, "encoding": "jpeg"},
            text_tokens=512,
        )
    message = str(error.value)
    for field in ("frontend", "text_tokens", "images.height", "images.width", "images.count", "images.encoding"):
        assert field in message


def test_digest_follows_the_cost_tables_but_not_provenance():
    base = _profile(launch_extend=CostFnConfig(const_ms=5.0))
    same_costs = base.model_copy(update={"provenance": {"sampled_at": "later"}})
    changed_costs = _profile(launch_extend=CostFnConfig(const_ms=105.0))
    assert profile_digest(base) == profile_digest(same_costs)
    assert profile_digest(base) != profile_digest(changed_costs)
