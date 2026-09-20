# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured SGLang host-cost profiles: identity, matching, and lowering to engine tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from ..config.common import StrictModel, load_yaml
from ..config.engine import FrontendPredictionConfig, HostPredictionConfig, NonNegativeFloat

SGLANG_REVISION = "0bcd822377da7b5718e674eaf9c870d349424dd1"
"""The SGLang revision whose scheduler and frontend behavior the cost tables describe."""

FrontendKind = Literal["python", "rust"]


class ProfileIdentity(StrictModel):
    """What a profile's costs were measured for; a prediction must match it exactly."""

    sglang_revision: str
    model: str
    frontend: FrontendKind
    image_encoding: Literal["png", "jpeg"]


class HostProfile(StrictModel):
    """Scheduler-thread and frontend costs sampled on a serving host.

    The tables are the same shapes the public configuration accepts explicitly;
    a profile only adds the identity they were measured under and the evidence
    behind them. ``host.tp_sync_ms`` holds the single-rank value; ``tp_sync_ms``
    supplies larger tensor-parallel groups and a missing entry makes those
    deployments unsupported rather than free.
    """

    schema_version: Literal[1] = 1
    identity: ProfileIdentity
    host: HostPredictionConfig
    frontend: FrontendPredictionConfig
    tp_sync_ms: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


def profile_id(profile: HostProfile) -> str:
    """Stable identity digest; equal digests mean equal measured conditions."""
    canonical = json.dumps(profile.identity.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_host_profile(path: str | Path) -> HostProfile:
    return HostProfile.model_validate(load_yaml(path))


def match_host_profile(profile: HostProfile, *, model: str, frontend: FrontendKind, image_encoding: str) -> None:
    """Reject a profile whose measured conditions differ from the prediction's."""
    expected = {
        "sglang_revision": SGLANG_REVISION,
        "model": model,
        "frontend": frontend,
        "image_encoding": image_encoding,
    }
    actual = profile.identity.model_dump(mode="json")
    mismatches = [
        f"{key}: profile={actual[key]!r}, prediction={value!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if mismatches:
        raise ValueError("host profile does not match this prediction: " + "; ".join(mismatches))


def lower_host_profile(
    profile: HostProfile, *, tensor_parallel: int
) -> tuple[HostPredictionConfig, FrontendPredictionConfig]:
    """Resolve the engine tables for one deployment shape; unmeasured costs are errors."""
    if profile.missing:
        raise ValueError("host profile lacks measured costs for: " + ", ".join(sorted(profile.missing)))
    host = profile.host
    if tensor_parallel > 1:
        tp_sync_ms = profile.tp_sync_ms.get(str(tensor_parallel))
        if tp_sync_ms is None:
            raise ValueError(f"host profile has no tp_sync_ms entry for tensor parallel {tensor_parallel}")
        host = host.model_copy(update={"tp_sync_ms": tp_sync_ms})
    return host, profile.frontend
