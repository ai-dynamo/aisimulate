# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured SGLang host-cost profiles: identity, matching, and lowering to engine tables."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, PositiveInt

from ..config.common import StrictModel, load_yaml
from ..config.engine import FrontendPredictionConfig, HostPredictionConfig, HostProfileConfig, NonNegativeFloat

SGLANG_REVISION = "0bcd822377da7b5718e674eaf9c870d349424dd1"
"""The SGLang revision whose scheduler and frontend behavior the cost tables describe."""

SGLANG_VERSION = "0.5.19"
"""The installed `sglang.version.__version__` the sampler accepts."""

FrontendKind = Literal["python", "rust"]


class ProfileImages(StrictModel):
    """The fixed image workload every cost in a profile was sampled with."""

    height: PositiveInt
    width: PositiveInt
    count: PositiveInt
    encoding: Literal["png", "jpeg"]
    # Processor pixel budget the workload overrides; None follows the checkpoint.
    min_pixels: PositiveInt | None = None
    max_pixels: PositiveInt | None = None


class ProfileIdentity(StrictModel):
    """What a profile's costs were measured for; a prediction must match it exactly.

    Stage costs are constants of one workload shape, so the shape (images and
    text length) is part of the identity, as are the serving code, the
    processor that produced the features, and the CPU and thread budget the
    sampler ran under.
    """

    sglang_revision: str
    model: str
    frontend: FrontendKind
    images: ProfileImages
    text_tokens: PositiveInt
    processor: str
    cpu: str
    threads: PositiveInt


class HostProfile(StrictModel):
    """Scheduler-thread and frontend costs sampled on a serving host.

    The tables are the same shapes the public configuration accepts explicitly;
    a profile only adds the identity they were measured under and the evidence
    behind them. ``host.tp_sync_ms`` holds the single-rank value; ``tp_sync_ms``
    supplies larger tensor-parallel groups and a missing entry makes those
    deployments unsupported rather than free.
    """

    schema_version: Literal[2] = 2
    identity: ProfileIdentity
    host: HostPredictionConfig
    frontend: FrontendPredictionConfig
    tp_sync_ms: dict[str, NonNegativeFloat] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


def profile_digest(profile: HostProfile) -> str:
    """Content digest of the resolved profile: identity and every cost table, provenance excluded."""
    content = profile.model_dump(mode="json", exclude={"provenance"})
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def load_host_profile(path: str | Path) -> HostProfile:
    return HostProfile.model_validate(load_yaml(path))


def _images_identity(images: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "height": int(images["height"]),
        "width": int(images["width"]),
        "count": int(images.get("count", 1)),
        "encoding": str(images.get("encoding", "png")),
        "min_pixels": images.get("min_pixels"),
        "max_pixels": images.get("max_pixels"),
    }


def match_host_profile(
    profile: HostProfile,
    *,
    model: str,
    frontend: FrontendKind,
    images: Mapping[str, Any],
    text_tokens: int | None = None,
) -> None:
    """Reject a profile whose measured conditions differ from the prediction's.

    The image shape, count and encoding must all match: a fixed-shape cost has
    no validity outside the shape it was sampled with. `text_tokens` is checked
    when the caller knows the prompt length.
    """
    expected: dict[str, Any] = {
        "sglang_revision": SGLANG_REVISION,
        "model": model,
        "frontend": frontend,
    }
    if text_tokens is not None:
        expected["text_tokens"] = int(text_tokens)
    actual = profile.identity.model_dump(mode="json")
    mismatches = [
        f"{key}: profile={actual[key]!r}, prediction={value!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    mismatches.extend(
        f"images.{field}: profile={actual['images'][field]!r}, prediction={value!r}"
        for field, value in _images_identity(images).items()
        if actual["images"][field] != value
    )
    if mismatches:
        raise ValueError("host profile does not match this prediction: " + "; ".join(mismatches))


def resolve_host_profile(
    config: HostProfileConfig,
    *,
    model: str,
    images: Mapping[str, Any],
    tensor_parallel: int,
    text_tokens: int | None = None,
) -> tuple[HostProfile, HostPredictionConfig, FrontendPredictionConfig]:
    """Load, or when configured sample, the profile a worker names and lower it for `images`."""
    if not Path(config.path).exists() and config.on_missing == "calibrate":
        from .calibrate import calibrate_in_subprocess

        profile = calibrate_in_subprocess(
            output=config.path, model=model, frontend=config.frontend, images=images, text_tokens=text_tokens
        )
    else:
        profile = load_host_profile(config.path)
    match_host_profile(profile, model=model, frontend=config.frontend, images=images, text_tokens=text_tokens)
    host, frontend = lower_host_profile(profile, tensor_parallel=tensor_parallel)
    return profile, host, frontend


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
