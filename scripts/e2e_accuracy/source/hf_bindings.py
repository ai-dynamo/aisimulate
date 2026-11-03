# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reviewed, complete-row bindings across HF and historical database transports."""

import hashlib
import json
from dataclasses import asdict

from .sources import load_manifest


def row_digest(row):
    # Preserve the original runtime archive's hash contract.
    values = asdict(row)
    values.pop("run_started_at")
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def bind_row(hf_row, database_row, server_log_uses):
    """Build a binding only after comparing all non-transport measurement fields."""

    def measurement(row):
        values = asdict(row)
        for key in ("config_id", "workflow_run_id", "server_log_id", "run_started_at", "workflow_url"):
            values.pop(key)
        values["infx_config"].pop("id", None)
        # The dump omits the API's explicit default; active offload must still match.
        values["infx_config"].setdefault("offload_mode", "off")
        values["infx_config"].setdefault("workers", None)
        return values

    if not str(hf_row.config_id).startswith("hf:") or measurement(hf_row) != measurement(database_row):
        raise ValueError("HF and database measurement identity mismatch")
    return {
        "database_measurement_sha256": row_digest(database_row),
        "database_config_id": database_row.config_id,
        "server_log_uses": server_log_uses,
    }


def reviewed_binding(row):
    if not str(row.config_id).startswith("hf:"):
        return None
    manifest = load_manifest("hf_evidence_bindings.json")
    if manifest["schema_version"] != "reviewed-hf-evidence-bindings/1":
        raise ValueError("unsupported HF evidence bindings")
    return manifest["records"].get(row_digest(row))
