# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream native evidence without retaining a campaign's token payloads."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def iter_records(source: bytes | Path):
    """Yield strict JSONL records, retaining at most one physical line."""
    with source.open("rb") if isinstance(source, Path) else io.BytesIO(source) as stream:
        for line in stream:
            yield json.loads(line)
