# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load reviewed manifests and verify their upstream source bytes."""

import hashlib
import json
from functools import cache
from pathlib import Path

import requests

from scripts.e2e_accuracy.source.recipes.inferencex_recipe import InferenceXRecipeError


@cache
def manifest_text(name: str) -> str:
    return (Path(__file__).with_name("manifests") / name).read_text()


@cache
def load_manifest(name: str) -> dict:
    """Read once per campaign; callers treat the reviewed records as immutable."""
    return json.loads(manifest_text(name))


def verify_sources(records: list[dict], *, kind: str = "framework") -> list[dict]:
    """Keep HTTP errors for callers to contextualize; never accept a changed hash."""
    for record in records:
        response = requests.get(record["url"], timeout=30)
        response.raise_for_status()
        if hashlib.sha256(response.content).hexdigest() != record["sha256"]:
            raise InferenceXRecipeError(f"reviewed {kind} source changed: {record['url']}")
    return records
