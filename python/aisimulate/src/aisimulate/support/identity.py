# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical identities for support requests, cells, and candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .schema import SupportRequest


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_mapping(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def support_cell_payload(request: SupportRequest) -> dict[str, Any]:
    """Return only fields that define the exact supported cell."""

    return {
        "schema_version": "aisimulate-support-cell/v1",
        "identity": request.identity.model_dump(mode="json", exclude_none=True),
        "workloads": [workload.model_dump(mode="json", exclude_none=True) for workload in request.workloads],
    }


def support_cell_id(request: SupportRequest) -> str:
    return f"asc1-{digest_mapping(support_cell_payload(request))[:20]}"


def candidate_id(value: Mapping[str, Any]) -> str:
    return f"candidate-{digest_mapping(value)[:16]}"
