# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned HF identity at the canonical interpolation boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.fpm_accuracy.exceptions import ConfigurationError, DependencyError
from scripts.fpm_accuracy.models.scenario_interpolation import (
    configure_scenario_interpolation,
    measured_fpm_identity,
)


@dataclass
class Request:
    model: str = "Org/Model"
    system: str = "h200_sxm"
    backend: str = "vllm"
    backend_version: str = "0.25.1"
    tp: int = 4
    pp: int = 1
    attention_dp: int = 1
    moe_tp_size: int = 1
    moe_ep_size: int = 4
    cp_size: int | None = None
    gemm_quant_mode: str | None = None
    moe_quant_mode: str | None = None
    fmha_quant_mode: str | None = None
    comm_quant_mode: str | None = None
    kvcache_quant_mode: str | None = None
    attention_backend: str | None = None
    moe_backend: str | None = None
    fpm_fmha_quant_mode: str | None = None
    fpm_profile: dict | None = None
    estimator_config: dict = field(default_factory=lambda: {"fpm_interpolation": {"method": "sol"}})


@pytest.fixture
def source(tmp_path):
    identity = {
        "model_path": "Org/Model",
        "system": "h200_sxm",
        "backend": "vllm",
        "backend_version": "0.25.1",
        "tp": 4,
        "pp": 1,
        "dp": 1,
        "moe_tp": 1,
        "moe_ep": 4,
        "cp": 1,
        "gemm_quant_mode": "fp8_block",
        "moe_quant_mode": "fp8_block",
        "fmha_quant_mode": None,
        "comm_quant_mode": None,
        "kv_cache_dtype": "fp8",
        "attention_backend": "auto",
        "moe_backend": "auto",
        "enable_wideep": False,
        "enable_eplb": False,
    }
    parquet = tmp_path / "fpm_forward_perf.parquet"
    metadata = tmp_path / "fpm_forward_perf.metadata.json"

    def write(rows=None, schema=6):
        rows = rows or [identity, identity]
        pq.write_table(pa.Table.from_pylist(rows), parquet)
        digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
        metadata.write_text(json.dumps({"schema_version": schema, "row_count": len(rows), "parquet_sha256": digest}))
        return SimpleNamespace(
            local_path=parquet,
            local_metadata_path=metadata,
            sha256=digest,
            metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest(),
        )

    return identity, write, SimpleNamespace(systems_root=tmp_path)


def test_text_only_sol_preserves_recorded_identity_and_product_auto_backend_mapping(source):
    identity, write, prepared = source
    artifact = write()
    assert measured_fpm_identity(artifact) == identity
    original = Request()
    resolved = configure_scenario_interpolation(original, artifact, prepared)
    assert resolved.gemm_quant_mode == "fp8_block" and resolved.kvcache_quant_mode == "fp8"
    assert resolved.attention_backend is None  # Source "auto" uses the product's None/default mapping.
    controls = resolved.estimator_config["fpm_interpolation"]
    assert controls["method"] == "sol" and controls["text_only"] is True
    assert controls["unrecorded_quant_modes"] == ["fmha", "comm"]
    assert controls["fpm_parquet_path"] == str(artifact.local_path)
    assert "timing_only" not in controls and resolved.fpm_profile is None
    assert original.estimator_config == {"fpm_interpolation": {"method": "sol"}}


@pytest.mark.parametrize("changes", [{"tp": 8}, {"attention_backend": "fa3"}, {"gemm_quant_mode": "half"}])
def test_conflicting_requested_identity_cannot_be_replaced(source, changes):
    _, write, prepared = source
    with pytest.raises(ConfigurationError, match="differs|conflicts"):
        configure_scenario_interpolation(replace(Request(), **changes), write(), prepared)


def test_mutated_sources_and_ambiguous_identity_fail_closed(source):
    identity, write, _ = source
    artifact = write()
    artifact.local_metadata_path.write_text(artifact.local_metadata_path.read_text() + " ")
    with pytest.raises(ConfigurationError, match="hashes"):
        measured_fpm_identity(artifact)
    artifact = write([identity, {**identity, "tp": 8}])
    with pytest.raises(ConfigurationError, match="ambiguous"):
        measured_fpm_identity(artifact)


def test_schema7_and_unprofiled_direct_are_explicitly_unsupported(source):
    _, write, prepared = source
    with pytest.raises(DependencyError, match="schema-6"):
        measured_fpm_identity(write(schema=7))
    request = replace(Request(), estimator_config={"fpm_interpolation": {"method": "direct"}})
    with pytest.raises(DependencyError, match="verified FPM profile"):
        configure_scenario_interpolation(request, write(), prepared)


def test_only_typed_domain_misses_are_unavailable(monkeypatch):
    import sys

    from scripts.fpm_accuracy.models.scenario_interpolation import is_interpolation_support_miss

    class Missing(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "aisimulate_core.sdk.errors", SimpleNamespace(PerfDataNotAvailableError=Missing))
    message = "perf_interp: no data to anchor query {batch_size=1} (no usable neighbour site)"
    assert is_interpolation_support_miss(Missing(message))
    assert not is_interpolation_support_miss(ValueError(message))
    assert not is_interpolation_support_miss(Missing(message + "; SOL roofline unavailable: no Rust implementation"))
    assert not is_interpolation_support_miss(Missing("missing op gemm table"))
