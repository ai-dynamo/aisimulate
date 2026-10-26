# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN context-scan lane preference: vLLM's own physical rows (AIC-1782 V5b).

The vLLM 0.24.0 and 0.27.1 collectors persist the executed context-scan
label (``chunk_gated_delta_rule_flashinfer`` on SM100 systems) while the
modeling layer queries the logical ``chunk_gated_delta_rule``. The alias
walk in ``perf_database/state_space.rs`` must therefore serve the physical
row for both versions.

Same real-data engine-query pattern as ``test_gdn_flashinfer_lane.py``
(``Operation._engine_query`` — the sanctioned single-op plumbing — plus
raw-table pins in the ``test_gdn_donor_fill_pins.py`` convention). AISimulate
does not currently ship the old cross-backend logical donor lane assumed by
the AIC source test; precedence over a competing donor is covered with an
explicit synthetic table by the Rust unit tests in ``perf_database/state_space.rs``
(``vllm_0271_gdn_own_physical_lane_wins_over_logical_donor_lane``,
``vllm_024_gdn_own_physical_lane_wins_over_logical_lane``,
``gdn_physical_aliases_are_version_gated``) cover the same claims with
synthetic in-memory tables; this file is the real-data-grounded companion.
"""

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate.sdk.perf_database import PerfDatabase, get_database, get_systems_paths
from aisimulate_core.sdk.operations.mamba import GDNKernel

pytestmark = pytest.mark.unit


def _gdn_context_op(kernel_source, model_key):
    d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = model_key
    return GDNKernel(
        "gdn", 1.0, kernel_source, "context", d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv
    )


def _raw_context_latency(db, kernel_source, model_key, batch, seq_len):
    GDNKernel.load_data(db)
    return db._gdn_data[kernel_source]["context"][model_key][batch][seq_len]["latency"]


def test_query_gdn_vllm_0271_context_scan_prefers_own_flashinfer_rows():
    """The logical query resolves the shipped Qwen3.8 physical row."""
    db = get_database("gb300", "vllm", "0.27.1", allow_unlisted_version=True)
    model_key = (8192, 1, 128, 8, 128, 4)  # Qwen3.8-Max GDN tp16 shard

    own_raw = _raw_context_latency(db, "chunk_gated_delta_rule_flashinfer", model_key, 1, 1024)
    assert own_raw == pytest.approx(0.05215519905090332, rel=1e-9)

    # The precedence claim: querying the logical kernel_source must resolve
    # the own physical row, not the donor.
    result = _gdn_context_op("chunk_gated_delta_rule", model_key)._engine_query(db, batch_size=1, s=1024)
    assert result.source == "silicon"
    assert float(result) == pytest.approx(own_raw, rel=1e-9)


def test_query_gdn_vllm_0240_context_scan_still_resolves_own_flashinfer_rows():
    """Version-gate widening preserves the existing 0.24.0 alias walk."""
    db = get_database("gb300", "vllm", "0.24.0")
    model_key = (5120, 2, 128, 6, 128, 4)  # Qwen3.5-397B GDN tp8 shard

    own_raw = _raw_context_latency(db, "chunk_gated_delta_rule_flashinfer", model_key, 1, 1024)
    assert own_raw == pytest.approx(0.05393343925476074, rel=1e-9)

    result = _gdn_context_op("chunk_gated_delta_rule", model_key)._engine_query(db, batch_size=1, s=1024)
    assert result.source == "silicon"
    assert float(result) == pytest.approx(own_raw, rel=1e-9)


@pytest.mark.parametrize(
    "logical,physical,phase,latency",
    [
        ("chunk_gated_delta_rule", "chunk_gated_delta_rule_flashinfer", "context", 2.0),
        (
            "fused_sigmoid_gating_delta_rule_update",
            "fused_recurrent_gated_delta_rule_packed_decode",
            "generation",
            3.0,
        ),
    ],
)
def test_vllm_031_logical_query_uses_own_measured_lane(tmp_path, logical, physical, phase, latency):
    """Synthetic stored values expose SOL fallback without claiming GPU evidence."""
    shutil.copyfile(Path(get_systems_paths()[0]) / "b200_sxm.yaml", tmp_path / "b200_sxm.yaml")
    path = tmp_path / "data/b200_sxm/linear_attention/vllm/0.31.0/gdn_perf.parquet"
    path.parent.mkdir(parents=True)
    row = {
        "kernel_source": physical,
        "phase": phase,
        "batch_size": 1,
        "seq_len": 1024 if phase == "context" else 1,
        "d_model": 5120,
        "d_conv": 4,
        "num_k_heads": 16,
        "head_k_dim": 128,
        "num_v_heads": 48,
        "head_v_dim": 128,
        "latency": latency,
    }
    pq.write_table(pa.Table.from_pylist([row]), path)
    db = PerfDatabase("b200_sxm", "vllm", "0.31.0", str(tmp_path), shared_layer=False)
    # Packed decode is normalized to its logical name by the existing loader;
    # only context keeps a physical table key and needs the 0.31 alias repair.
    for source in (logical, physical) if phase == "context" else (logical,):
        op = GDNKernel("gdn", 1.0, source, phase, 5120, 16, 128, 48, 128, 4)
        result = op._engine_query(db, batch_size=1, s=row["seq_len"])
        assert result.source == "silicon"
        assert float(result) == pytest.approx(latency, rel=1e-9)
