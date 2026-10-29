# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN context-scan lane preference: vLLM's own physical rows (AIC-1782 V5b).

The vLLM collectors from 0.24.0 on persist the executed context-scan label
(``chunk_gated_delta_rule_flashinfer`` on SM100 systems) while the modeling
layer queries the logical ``chunk_gated_delta_rule``. The alias walk in
``perf_database/state_space.rs`` (``GdnAliasProfile``) must therefore serve
the physical row for every such version — 0.24.0, 0.27.1 and the 2026-10
campaign's 0.30.0 — and, since that campaign, the TRT-LLM 1.3.0rc29
context scan and decode recurrence (FlashInfer, or Triton on L40S / RTX PRO
6000) and the SGLang 0.5.21 fused decode convolution and Hopper FlashInfer
context scan as well (PR #517 review, simone-chen). When the shared layer
adds a donor's physical label next to the requested version's own one, the
own row wins instead of the pair failing as ambiguous.

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

import pytest

from aisimulate.sdk.perf_database import get_database
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


def _gdn_generation_op(kernel_source, model_key):
    d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv = model_key
    return GDNKernel(
        "gdn", 1.0, kernel_source, "generation", d_model, num_k_heads, head_k_dim, num_v_heads, head_v_dim, d_conv
    )


def _raw_generation_latency(db, kernel_source, model_key, batch):
    GDNKernel.load_data(db)
    return db._gdn_data[kernel_source]["generation"][model_key][batch]["latency"]


def test_query_gdn_vllm_0300_context_scan_prefers_own_flashinfer_rows():
    """The 2026-10 campaign tables carry the same physical label as 0.24.0; a
    version whitelist sent this query to SOL (about 9x too fast)."""
    db = get_database("b200_sxm", "vllm", "0.30.0", allow_unlisted_version=True)
    model_key = (2048, 8, 128, 8, 128, 4)  # Qwen3.5-2B

    own_raw = _raw_context_latency(db, "chunk_gated_delta_rule_flashinfer", model_key, 1, 1024)
    result = _gdn_context_op("chunk_gated_delta_rule", model_key)._engine_query(db, batch_size=1, s=1024)
    assert result.source == "silicon"
    assert float(result) == pytest.approx(own_raw, rel=1e-9)


def test_query_gdn_trtllm_rc29_resolves_its_flashinfer_labels():
    """TRT-LLM 1.3.0rc29 renamed both GDN kernels with a ``_flashinfer``
    suffix; the logical queries must land on those rows, not on the rc20
    logical rows reachable through the earlier-version fallback."""
    db = get_database("b200_sxm", "trtllm", "1.3.0rc29", allow_unlisted_version=True)
    model_key = (1024, 2, 128, 2, 128, 4)  # Qwen3.5-0.8B, tp2

    own_ctx = _raw_context_latency(db, "chunk_gated_delta_rule_flashinfer", model_key, 1, 1024)
    ctx = _gdn_context_op("chunk_gated_delta_rule", model_key)._engine_query(db, batch_size=1, s=1024)
    assert ctx.source == "silicon"
    assert float(ctx) == pytest.approx(own_ctx, rel=1e-9)

    own_gen = _raw_generation_latency(db, "fused_sigmoid_gating_delta_rule_update_flashinfer", model_key, 1)
    gen = _gdn_generation_op("fused_sigmoid_gating_delta_rule_update", model_key)._engine_query(db, batch_size=1, s=1)
    assert gen.source == "silicon"
    assert float(gen) == pytest.approx(own_gen, rel=1e-9)


def test_query_gdn_sglang_0521_conv_update_resolves_the_fused_kernel():
    """SGLang 0.5.21 replaced ``causal_conv1d_update`` with the fused
    ``fused_qkvzba_causal_conv1d_update_contiguous``; the logical query must
    serve that row rather than the 0.5.14 row the earlier-version fallback
    still carries."""
    db = get_database("b200_sxm", "sglang", "0.5.21", allow_unlisted_version=True)
    model_key = (1024, 4, 128, 4, 128, 4)  # Qwen3.5-0.8B, tp1

    own_raw = _raw_generation_latency(db, "fused_qkvzba_causal_conv1d_update_contiguous", model_key, 1)
    result = _gdn_generation_op("causal_conv1d_update", model_key)._engine_query(db, batch_size=1, s=1)
    assert result.source == "silicon"
    assert float(result) == pytest.approx(own_raw, rel=1e-9)


# Qwen3.5 GDN shape of the second review round: (d_model, nk, hk, nv, hv, conv).
_QWEN35_REVIEW_KEY = (2048, 16, 128, 32, 128, 4)


@pytest.mark.parametrize(
    ("version", "own_label"),
    [("0.24.0", "chunk_gated_delta_rule_triton"), ("0.30.0", "chunk_gated_delta_rule_flashinfer")],
)
def test_query_gdn_rtx_vllm_context_scan_prefers_own_label_over_donor_label(version, own_label):
    """RTX PRO 6000 vLLM 0.24.0 collected the Triton context scan, 0.30.0 the
    FlashInfer one; through the shared layer each also sees the other's label
    (and TRT-LLM rc29's FlashInfer label). Both are vLLM physical labels, so
    the lookup used to reject the pair as ambiguous and fall to SOL
    (0.035109 ms instead of 0.101526 / 0.116720 ms)."""
    db = get_database("rtx_pro_6000_server", "vllm", version, allow_unlisted_version=True)

    own_raw = _raw_context_latency(db, own_label, _QWEN35_REVIEW_KEY, 1, 1024)
    result = _gdn_context_op("chunk_gated_delta_rule", _QWEN35_REVIEW_KEY)._engine_query(db, batch_size=1, s=1024)
    assert result.source == "silicon"
    assert float(result) == pytest.approx(own_raw, rel=1e-9)


@pytest.mark.parametrize("system", ["h100_sxm", "h200_sxm"])
def test_query_gdn_sglang_0521_hopper_context_scan_resolves_flashinfer_label(system):
    """SGLang 0.5.21 on Hopper persists ``flashinfer_chunk_gated_delta_rule``;
    without the mapping H100 served a donor row (0.132997 ms instead of its
    own 0.093043 ms)."""
    db = get_database(system, "sglang", "0.5.21", allow_unlisted_version=True)

    own_raw = _raw_context_latency(db, "flashinfer_chunk_gated_delta_rule", _QWEN35_REVIEW_KEY, 1, 1024)
    result = _gdn_context_op("chunk_gated_delta_rule", _QWEN35_REVIEW_KEY)._engine_query(db, batch_size=1, s=1024)
    assert result.source == "silicon"
    assert float(result) == pytest.approx(own_raw, rel=1e-9)


@pytest.mark.parametrize("system", ["l40s", "rtx_pro_6000_server"])
def test_query_gdn_trtllm_rc29_resolves_the_triton_decode_recurrence(system):
    """L40S and RTX PRO 6000 TRT-LLM rc29 persist
    ``fused_sigmoid_gating_delta_rule_update_triton``; without the mapping
    L40S served a donor row (0.006557 ms instead of its own 0.005315 ms)."""
    db = get_database(system, "trtllm", "1.3.0rc29", allow_unlisted_version=True)

    own_raw = _raw_generation_latency(db, "fused_sigmoid_gating_delta_rule_update_triton", _QWEN35_REVIEW_KEY, 1)
    result = _gdn_generation_op("fused_sigmoid_gating_delta_rule_update", _QWEN35_REVIEW_KEY)._engine_query(
        db, batch_size=1, s=1
    )
    assert result.source == "silicon"
    assert float(result) == pytest.approx(own_raw, rel=1e-9)
