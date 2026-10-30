# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin measured text-forward identity without inventing a resource profile."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

from scripts.fpm_accuracy.contract import strict_json
from scripts.fpm_accuracy.exceptions import ConfigurationError, DependencyError

_IDENTITY = {
    "model": "model_path",
    "system": "system",
    "backend": "backend",
    "backend_version": "backend_version",
    "tp": "tp",
    "pp": "pp",
    "attention_dp": "dp",
    "moe_tp_size": "moe_tp",
    "moe_ep_size": "moe_ep",
    "cp_size": "cp",
}
_PRECISION = {
    "gemm_quant_mode": "gemm_quant_mode",
    "moe_quant_mode": "moe_quant_mode",
    "fmha_quant_mode": "fmha_quant_mode",
    "comm_quant_mode": "comm_quant_mode",
    "kvcache_quant_mode": "kv_cache_dtype",
    "attention_backend": "attention_backend",
}


def measured_fpm_identity(artifact: Any) -> dict:
    """Resolve one immutable, explicit schema-6 identity from its recorded rows."""
    import pyarrow.parquet as pq

    parquet, sidecar = Path(artifact.local_path), Path(artifact.local_metadata_path)
    metadata_bytes = sidecar.read_bytes()
    metadata = strict_json(metadata_bytes.decode())
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    if (
        digest != artifact.sha256
        or digest != metadata.get("parquet_sha256")
        or hashlib.sha256(metadata_bytes).hexdigest() != artifact.metadata_sha256
    ):
        raise ConfigurationError("FPM identity requires the pinned Parquet and metadata hashes.")
    if metadata.get("schema_version") != 6:
        raise DependencyError("The Gym worker adapter currently supports schema-6 FPM identity only.")
    columns = list(
        dict.fromkeys((*_IDENTITY.values(), *_PRECISION.values(), "moe_backend", "enable_wideep", "enable_eplb"))
    )
    try:
        rows = pq.read_table(parquet, columns=columns).to_pylist()
    except Exception as exc:
        raise ConfigurationError("FPM identity requires every recorded schema-6 identity column.") from exc
    if not rows or type(metadata.get("row_count")) is not int or len(rows) != metadata["row_count"]:
        raise ConfigurationError("FPM identity requires the recorded nonempty row count.")
    identity = rows[0]
    if any(row != identity for row in rows[1:]):
        raise ConfigurationError("FPM artifact contains ambiguous deployment identities.")
    if any(
        type(identity[name]) is not int or identity[name] < 1 for name in ("tp", "pp", "dp", "moe_tp", "moe_ep", "cp")
    ):
        raise ConfigurationError("FPM identity requires explicitly recorded positive topology dimensions.")
    if (
        identity["enable_wideep"] is not False
        or identity["enable_eplb"] is not False
        or identity["moe_backend"] != "auto"
    ):
        raise DependencyError("Gym interpolation requires recorded automatic MoE without WideEP or EPLB.")
    return identity


def configure_scenario_interpolation(request: Any, artifact: Any, prepared: Any) -> Any:
    """Retain native method selection and bind text-only measurements exactly.

    Production direct interpolation requires an explicit verified FPM profile.
    Profile-less measurements use the canonical SOL path with text_only=True;
    their table alone makes no memory-capacity or encoder-timing claim.
    """
    identity = measured_fpm_identity(artifact)
    for field, column in _IDENTITY.items():
        requested = getattr(request, field)
        if field == "cp_size" and requested is None:
            requested = 1
        if requested != identity[column]:
            raise ConfigurationError(f"Recorded FPM {column} differs from the requested {field}.")
    precision = {field: identity[column] for field, column in _PRECISION.items()}
    # The product profile compiler uses None for the table's "auto" backend
    # (sdk/engine.py); "auto" is not an analytical ModelConfig backend enum.
    if precision["attention_backend"] == "auto":
        precision["attention_backend"] = None
    for field, value in precision.items():
        requested = getattr(request, field)
        if field == "attention_backend" and requested in {"auto", "default"}:
            requested = None
        if requested is not None and requested != value:
            raise ConfigurationError(f"Recorded FPM {field} conflicts with the requested precision.")
    if request.moe_backend is not None or request.fpm_fmha_quant_mode is not None:
        raise ConfigurationError("Gym FPM cannot discard explicit MoE or FMHA backend overrides.")
    paths = list(Path(prepared.systems_root).glob("**/fpm_forward_perf.parquet"))
    if len(paths) != 1:
        raise ConfigurationError("The staged FPM database must contain one pinned timing table.")
    estimator = deepcopy(request.estimator_config)
    interpolation = estimator.setdefault("fpm_interpolation", {})
    interpolation["fpm_parquet_path"] = str(paths[0])
    if request.fpm_profile is None:
        if interpolation.get("method") == "direct":
            raise DependencyError("Direct FPM requires a verified FPM profile in the production API.")
        interpolation["text_only"] = True
        interpolation["unrecorded_quant_modes"] = [
            mode
            for mode, field in (("fmha", "fmha_quant_mode"), ("comm", "comm_quant_mode"))
            if precision[field] is None
        ]
    return replace(request, estimator_config=estimator, **precision)


def is_interpolation_support_miss(error: Exception) -> bool:
    """Recognize only typed, explicit domain misses; retain native capability errors."""
    try:
        from aisimulate_core.sdk.errors import PerfDataNotAvailableError
    except ImportError:
        return False
    if not isinstance(error, PerfDataNotAvailableError):
        return False
    message = str(error)
    # A failed analytical operation after a table miss is a native capability
    # limitation, not an ordinary unsupported table coordinate.
    if "SOL roofline unavailable" in message or "not implemented" in message.lower():
        return False
    return (
        "perf_interp: no data to anchor query" in message
        or ("FPM " in message and "outside the collected domain" in message)
        or (
            "FPM direct " in message
            and any(
                reason in message
                for reason in (
                    "no same-batch, two-sided KV bracket whose token curves cover this query",
                    "no two-sided covered batch bracket within one capture regime",
                )
            )
        )
    )
