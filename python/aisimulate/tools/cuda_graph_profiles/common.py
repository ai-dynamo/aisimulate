# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

GIB = 1 << 30
SCHEMA_VERSION = 1

PROFILE_IDENTITY_FIELDS = (
    "model_id",
    "model_revision",
    "model_config_sha256",
    "system",
    "backend",
    "backend_version",
    "backend_build",
    "tp_size",
    "pp_size",
    "attention_dp_size",
    "dcp_size",
    "pcp_size",
    "moe_tp_size",
    "moe_ep_size",
    "quantization",
    "compute_dtype",
    "kv_cache_dtype",
    "cuda_graph_mode",
    "cuda_graph_capture_sizes",
    "max_num_seqs",
    "max_num_batched_tokens",
    "max_model_len",
    "attention_backend",
    "speculative_method",
    "speculative_tokens",
)

REQUIRED_PROFILE_FIELDS = (
    "schema_version",
    "measurement_id",
    "profile_id",
    *PROFILE_IDENTITY_FIELDS,
    "graph_disabled",
    "training_eligible",
    "identity_completeness",
    "source_repository",
    "run_id",
    "run_attempt",
    "head_sha",
    "artifact_id",
    "artifact_name",
    "artifact_sha256",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def profile_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in PROFILE_IDENTITY_FIELDS}


def profile_id(row: dict[str, Any]) -> str:
    return f"cgrp-{stable_hash(profile_identity(row))[:20]}"


def measurement_id(row: dict[str, Any]) -> str:
    payload = {
        "profile_id": row["profile_id"],
        "run_id": row["run_id"],
        "run_attempt": row["run_attempt"],
        "artifact_id": row["artifact_id"],
        "artifact_sha256": row["artifact_sha256"],
    }
    return f"cgrm-{stable_hash(payload)[:20]}"


def normalize_capture_sizes(values: Any) -> str:
    if values is None:
        return "[]"
    if isinstance(values, str):
        values = json.loads(values)
    return canonical_json(sorted({int(value) for value in values}))


def normalize_model_id(value: str | None, *, precision: str | None = None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    if "minimax-m3" in lowered:
        if precision and "4" in precision:
            return "nvidia/MiniMax-M3-NVFP4"
        return "MiniMaxAI/MiniMax-M3-MXFP8"
    if "deepseek-v4" in lowered or "deepseek_v4" in lowered or "dsv4" in lowered:
        return "deepseek-ai/DeepSeek-V4-Pro"
    if "kimi-k3" in lowered or "kimik3" in lowered:
        return "moonshotai/Kimi-K3"
    if "/snapshots/" in value:
        return value.split("/snapshots/", maxsplit=1)[0].rsplit("models--", maxsplit=1)[-1].replace("--", "/")
    if value.startswith("/"):
        return Path(value).name
    return value


def model_revision_from_path(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"/snapshots/([0-9a-f]{40})(?:/|$)", value)
    return match.group(1) if match else None


def normalize_system(value: str | None) -> str | None:
    if not value:
        return None
    lowered = value.lower()
    mappings = (
        ("h100", "h100_sxm"),
        ("h200", "h200_sxm"),
        ("b200", "b200_sxm"),
        ("b300", "b300_sxm"),
    )
    for token, normalized in mappings:
        if token in lowered:
            return normalized
    return value.removeprefix("cluster:").replace("-", "_")


def backend_family(version: str | None) -> str | None:
    if not version:
        return None
    match = re.match(r"(\d+)\.(\d+)", version)
    return ".".join(match.groups()) if match else version.split("+", maxsplit=1)[0]
