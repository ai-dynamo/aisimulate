# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``AFDInferenceSession`` per-phase simulation and summary layout.

These exercise the still-live SDK session directly (no ``cli_estimate``), so
they guard the modeling behavior that feeds ``sweeper.afd_measure``'s layer
measurements (``afd_layer_measurements`` -> ``AFDLayerTimes``) and the public
summary schema. They were migrated out of the deleted
``tests/unit/cli/test_afd_phase_completion.py`` when the legacy AFD estimate
surface was dropped; the ``cli_estimate``/``_run_afd_estimate``/
``_combine_afd_static_estimate_results`` cases were not carried over because
those entry points no longer exist.
"""

import logging
import math
from types import SimpleNamespace
from typing import ClassVar

import pytest

from aisimulate.sdk.config import AFDConfig, RuntimeConfig
from aisimulate.sdk.inference_session import AFDInferenceSession
from aisimulate.sdk.inference_summary import InferenceSummary

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fake_model_architecture(monkeypatch):
    """Keep the synthetic model local while exercising the AFD capability guard."""

    def resolve_model(model_path):
        assert model_path == "test-model"
        return {"architecture": "DeepseekV3ForCausalLM"}

    monkeypatch.setattr("aisimulate_core.sdk.utils.get_model_config_from_model_path", resolve_model)


def _fake_phase_metrics(
    *,
    t_a_layer: float,
    t_f_layer: float,
    balance_ratio: float,
    t_a2f_layer: float = 0.1,
    t_f2a_layer: float = 0.1,
    t_step: float = 50.0,
    comm_hidden: bool = True,
) -> dict:
    """Build a minimal ``_simulate_phase``-style metrics dict for AFD tests.

    Only the per-phase layer scalars vary across cases; memory/per-op shape
    stays trivial so ``_build_summary`` can run without further plumbing.
    """
    return {
        "t_a_layer": t_a_layer,
        "t_f_layer": t_f_layer,
        "t_a2f_layer": t_a2f_layer,
        "t_f2a_layer": t_f2a_layer,
        "t_c_layer": t_a2f_layer + t_f2a_layer,
        "t_step": t_step,
        "comm_hidden": comm_hidden,
        "balance_ratio": balance_ratio,
        "a_per_op": {},
        "f_per_op": {},
        "a_memory": {"total": 1.0, "weights": 1.0, "activations": 0.0, "kvcache": 0.0, "nccl": 0.0, "others": 0.0},
        "f_memory": {"total": 1.0, "weights": 1.0, "activations": 0.0, "kvcache": 0.0, "nccl": 0.0, "others": 0.0},
        "a_is_oom": False,
        "f_is_oom": False,
        "a_is_kv_cache_oom": False,
        "f_is_kv_cache_oom": False,
        "num_layers": 4,
    }


def _build_afd_session_with_phase_metrics(
    monkeypatch,
    *,
    prefill_metrics,
    decode_metrics,
    nextn: int = 0,
) -> AFDInferenceSession:
    """Wire ``AFDInferenceSession`` so ``_simulate_phase`` returns the
    caller-supplied prefill / decode metrics dicts.

    Lets tests inject *different* per-phase scalars (e.g. distinct
    ``t_a_layer`` between prefill and decode) and inspect how
    ``_build_summary`` lays them out in ``result_dict``.
    """

    def fake_simulate_phase(self, *, phase, **_kwargs):
        return dict(prefill_metrics if phase == "prefill" else decode_metrics)

    monkeypatch.setattr(
        AFDInferenceSession,
        "_build_models",
        lambda self, **_kwargs: (SimpleNamespace(_num_layers=4), SimpleNamespace(_num_layers=4)),
    )
    monkeypatch.setattr(AFDInferenceSession, "_simulate_phase", fake_simulate_phase)

    class FakeDatabase:
        version = "current"
        system = "test-system"
        system_spec: ClassVar[dict] = {"gpu": {"mem_capacity": 80 * (1 << 30)}}

    afd_config = AFDConfig(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=8,
        tp_a=2,
        a_batch_size=4,
        num_microbatches=3,
        f_moe_ep_size=1,
        combined_with_pd=False,
    )
    return AFDInferenceSession(
        model_path="test-model",
        a_model_config=SimpleNamespace(nextn=nextn),
        f_model_config=SimpleNamespace(nextn=nextn),
        database=FakeDatabase(),
        backend=SimpleNamespace(
            name=SimpleNamespace(value="test-backend"),
            get_default_free_gpu_memory_fraction=lambda *_a, **_k: 0.9,
        ),
        afd_config=afd_config,
    )


def test_afd_prefill_uses_uncached_prefix_suffix_for_token_math(monkeypatch):
    """Prefill comm/compute math must size token volume by the uncached
    suffix ``isl - prefix``, not the raw ``isl``.

    All five AFD comm ops (cross-pool A↔F transfers + F-node AG/RS +
    A-side combine) receive ``x = a_batch_size * (isl - prefix)`` —
    the per-A-rank token count for that phase. ``_sum_latency`` runs
    once per pool with the same suffix length. Regressing this silently
    over-counts prefill bandwidth by the prefix-cache hit rate.
    """
    from aisimulate.sdk.inference_session import _AFDCommOps

    captured = {"x_queries": [], "sum_latency_seq_lens": []}

    class FakeCommOp:
        def __init__(self, name):
            self._name = name

        def query(self, _database, *, x):
            captured["x_queries"].append((self._name, x))
            # Any float-like works: the session only calls ``float(result)``
            # before folding it into the breakdown dict.
            return 0.0

    def fake_build_comm_ops(self, _a_model, _f_model, *, rank_mapping="one_to_one"):
        return _AFDCommOps(
            a2f=FakeCommOp("afd_a2f_transfer"),
            f2a=FakeCommOp("afd_f2a_transfer"),
            f_ag=FakeCommOp("afd_f_node_allgather"),
            f_rs=FakeCommOp("afd_f_node_reducescatter"),
            a_combine=FakeCommOp("afd_a_side_combine"),
        )

    def fake_sum_latency(self, _ops, *, batch_size, seq_len, model, runtime_config, is_context):
        captured["sum_latency_seq_lens"].append(seq_len)
        return 2.0, {}

    def fake_memory_summary(self, _memory, runtime_config, _free_gpu_memory_fraction):
        summary = InferenceSummary(runtime_config)
        summary.set_oom(False)
        summary.set_kv_cache_oom(False)
        return summary

    monkeypatch.setattr(
        "aisimulate.sdk.afd_partition.build_afd_ops_partition",
        lambda *_args, **_kwargs: SimpleNamespace(attn_ops=[], ffn_ops=[]),
    )
    monkeypatch.setattr(AFDInferenceSession, "_build_afd_comm_ops", fake_build_comm_ops)
    monkeypatch.setattr(AFDInferenceSession, "_sum_latency", fake_sum_latency)
    monkeypatch.setattr(AFDInferenceSession, "_estimate_a_memory_dict", lambda *_args, **_kwargs: {"total": 1.0})
    monkeypatch.setattr(AFDInferenceSession, "_estimate_f_memory_dict", lambda *_args, **_kwargs: {"total": 1.0})
    monkeypatch.setattr(AFDInferenceSession, "_check_memory_dict", fake_memory_summary)

    afd_config = AFDConfig(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=8,
        tp_a=2,
        a_batch_size=3,
        num_microbatches=1,
        f_moe_ep_size=1,
    )
    session = AFDInferenceSession(
        model_path="test-model",
        a_model_config=SimpleNamespace(),
        f_model_config=SimpleNamespace(),
        database=object(),
        backend=object(),
        afd_config=afd_config,
    )

    session._simulate_phase(
        phase="prefill",
        runtime_config=RuntimeConfig(isl=128, osl=10, prefix=48),
        a_model=SimpleNamespace(_num_layers=2),
        f_model=SimpleNamespace(_num_layers=2),
        free_gpu_memory_fraction=None,
        max_seq_len=None,
    )

    expected_x = afd_config.a_batch_size * 80  # a_batch_size * (isl - prefix)
    assert [x for _, x in captured["x_queries"]] == [expected_x] * 5
    assert {name for name, _ in captured["x_queries"]} == {
        "afd_a2f_transfer",
        "afd_f2a_transfer",
        "afd_f_node_allgather",
        "afd_f_node_reducescatter",
        "afd_a_side_combine",
    }
    assert captured["sum_latency_seq_lens"] == [80, 80]


@pytest.mark.parametrize(("nextn", "verify_width"), [(0, 1), (1, 2)])
def test_afd_decode_mtp_widens_compute_and_communication_queries(monkeypatch, caplog, nextn, verify_width):
    from aisimulate.sdk.inference_session import _AFDCommOps

    captured = {"x_queries": [], "batch_sizes": []}

    class FakeCommOp:
        def __init__(self, name):
            self._name = name

        def query(self, _database, *, x):
            captured["x_queries"].append(x)
            return 0.0

    def fake_build_comm_ops(self, _a_model, _f_model, *, rank_mapping="one_to_one"):
        return _AFDCommOps(
            a2f=FakeCommOp("afd_a2f_transfer"),
            f2a=FakeCommOp("afd_f2a_transfer"),
            f_ag=FakeCommOp("afd_f_node_allgather"),
            f_rs=FakeCommOp("afd_f_node_reducescatter"),
            a_combine=FakeCommOp("afd_a_side_combine"),
        )

    def fake_sum_latency(self, _ops, *, batch_size, **_kwargs):
        captured["batch_sizes"].append(batch_size)
        return 2.0, {}

    def fake_memory_summary(self, _memory, runtime_config, _free_gpu_memory_fraction):
        summary = InferenceSummary(runtime_config)
        summary.set_oom(False)
        summary.set_kv_cache_oom(False)
        return summary

    monkeypatch.setattr(
        "aisimulate.sdk.afd_partition.build_afd_ops_partition",
        lambda *_args, **_kwargs: SimpleNamespace(attn_ops=[], ffn_ops=[]),
    )
    monkeypatch.setattr(AFDInferenceSession, "_build_afd_comm_ops", fake_build_comm_ops)
    monkeypatch.setattr(AFDInferenceSession, "_sum_latency", fake_sum_latency)
    monkeypatch.setattr(AFDInferenceSession, "_estimate_a_memory_dict", lambda *_args, **_kwargs: {"total": 1.0})
    monkeypatch.setattr(AFDInferenceSession, "_estimate_f_memory_dict", lambda *_args, **_kwargs: {"total": 1.0})
    monkeypatch.setattr(AFDInferenceSession, "_check_memory_dict", fake_memory_summary)

    afd_config = AFDConfig(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=8,
        tp_a=2,
        a_batch_size=3,
        num_microbatches=1,
        f_moe_ep_size=1,
    )
    session = AFDInferenceSession(
        model_path="test-model",
        a_model_config=SimpleNamespace(nextn=nextn),
        f_model_config=SimpleNamespace(nextn=nextn),
        database=object(),
        backend=object(),
        afd_config=afd_config,
    )

    with caplog.at_level(logging.WARNING):
        session._simulate_phase(
            phase="decode",
            runtime_config=RuntimeConfig(isl=128, osl=2),
            a_model=SimpleNamespace(_num_layers=2),
            f_model=SimpleNamespace(_num_layers=2),
            free_gpu_memory_fraction=None,
            max_seq_len=None,
        )

    assert captured["x_queries"] == [3 * verify_width] * 5
    assert captured["batch_sizes"] == [3 * verify_width, 12 * verify_width]
    assert ("verify positions share sequence KV history" in caplog.text) is (nextn > 0)


def test_afd_summary_concurrency_reflects_total_in_flight_batch(monkeypatch):
    """``concurrency`` in the AFD summary equals the configured total batch.

    ``a_batch_size`` is the total in-flight batch per A-Worker.  The
    pipeline executes derived microbatches internally, so summary
    concurrency must not multiply the total batch by ``num_microbatches``.
    """
    metrics = _fake_phase_metrics(t_a_layer=1.0, t_f_layer=1.0, balance_ratio=1.0)
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=metrics,
        decode_metrics=metrics,
    )
    expected_b_total = session._afd_config.n_a_workers * session._afd_config.a_batch_size

    summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="decode")
    result = summary.get_result_dict()

    assert result["b_total"] == expected_b_total
    assert result["concurrency"] == expected_b_total
    assert result["b_micro_total"] == session._afd_config.n_a_workers * 2


def test_afd_summary_uses_global_batch_tpot_for_pipeline(monkeypatch):
    decode_metrics = _fake_phase_metrics(
        t_a_layer=1.0,
        t_f_layer=2.0,
        balance_ratio=0.5,
        t_a2f_layer=0.5,
        t_f2a_layer=0.5,
        t_step=26.0,
        comm_hidden=True,
    )
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=decode_metrics,
        decode_metrics=decode_metrics,
    )

    global_step, cycle, comm_hidden = session._pipeline_global_step_latency(
        1.0,
        2.0,
        0.5,
        0.5,
        num_layers=4,
    )
    assert cycle == pytest.approx(2.0)
    assert global_step == pytest.approx(26.0)
    assert comm_hidden is True

    summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="decode")
    result = summary.get_result_dict()
    expected_b_total = session._afd_config.n_a_workers * session._afd_config.a_batch_size

    assert result["tpot"] == pytest.approx(26.0)
    assert result["request_latency"] == pytest.approx(26.0 * 9)
    assert result["tokens/s"] == pytest.approx(expected_b_total * 1000.0 / 26.0, rel=1e-3)
    assert result["tokens/s/user"] == pytest.approx(1000.0 / 26.0, rel=1e-3)


def test_afd_summary_marks_power_as_unknown(monkeypatch):
    metrics = _fake_phase_metrics(t_a_layer=1.0, t_f_layer=2.0, balance_ratio=0.5)
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=metrics,
        decode_metrics=metrics,
    )

    result = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="decode").get_result_dict()

    assert math.isnan(result["power_w"])


def test_afd_summary_surfaces_effective_nextn(monkeypatch):
    metrics = _fake_phase_metrics(t_a_layer=1.0, t_f_layer=2.0, balance_ratio=0.5)
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=metrics,
        decode_metrics=metrics,
        nextn=2,
    )

    result = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="decode").get_result_dict()

    assert result["nextn"] == 2


def test_afd_summary_phase_both_paired_scalars_and_nan_unprefixed(monkeypatch):
    """``phase='both'`` writes both ``prefill_*`` and ``decode_*`` paired
    scalars and leaves the un-prefixed "headline" form NaN/None.

    Picking decode-only (or prefill-only) values as the un-prefixed scalar
    would silently discard the other phase's estimate when the two diverge,
    so NaN/None on the un-prefixed form makes the paired columns the only
    readable source of truth and prevents accidental misuse downstream.
    """
    prefill_metrics = _fake_phase_metrics(
        t_a_layer=0.5,
        t_f_layer=0.7,
        balance_ratio=0.71,
        t_a2f_layer=0.05,
        t_f2a_layer=0.05,
        t_step=12.5,
        comm_hidden=True,
    )
    decode_metrics = _fake_phase_metrics(
        t_a_layer=1.2,
        t_f_layer=0.9,
        balance_ratio=1.33,
        t_a2f_layer=0.1,
        t_f2a_layer=0.1,
        t_step=50.0,
        comm_hidden=False,
    )
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=prefill_metrics,
        decode_metrics=decode_metrics,
    )

    summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="both")
    result = summary.get_result_dict()

    assert result["phase"] == "both"

    # Paired scalars carry the per-phase values directly.
    assert result["prefill_t_a_layer"] == pytest.approx(0.5)
    assert result["prefill_t_f_layer"] == pytest.approx(0.7)
    assert result["prefill_balance_ratio"] == pytest.approx(0.71)
    assert result["prefill_t_step"] == pytest.approx(12.5)
    assert result["prefill_comm_hidden"] is True
    assert result["decode_t_a_layer"] == pytest.approx(1.2)
    assert result["decode_t_f_layer"] == pytest.approx(0.9)
    assert result["decode_balance_ratio"] == pytest.approx(1.33)
    assert result["decode_t_step"] == pytest.approx(50.0)
    assert result["decode_comm_hidden"] is False
    assert result["afd_layer_measurements"] == {
        "prefill": {
            "attention_ms": 0.5,
            "ffn_ms": 0.7,
            "a_to_f_ms": 0.05,
            "f_to_a_ms": 0.05,
            "num_layers": 4,
        },
        "decode": {
            "attention_ms": 1.2,
            "ffn_ms": 0.9,
            "a_to_f_ms": 0.1,
            "f_to_a_ms": 0.1,
            "num_layers": 4,
        },
    }

    # Un-prefixed scalars are NaN (numeric) / None (bool) so consumers
    # cannot accidentally treat decode-only values as the both-phase answer.
    for key in (
        "t_a_layer",
        "t_f_layer",
        "t_a2f_layer",
        "t_f2a_layer",
        "t_c_layer",
        "t_step",
        "balance_ratio",
    ):
        assert math.isnan(result[key]), f"expected NaN un-prefixed {key} in phase=both, got {result[key]!r}"
    assert result["comm_hidden"] is None


def test_afd_summary_phase_prefill_mirrors_unprefixed_into_prefill_pair(monkeypatch):
    """Single-phase prefill: un-prefixed == ``prefill_*``; ``decode_*`` NaN/None.

    Guards the back-compat contract for existing single-phase AFD users
    (the un-prefixed columns still carry the headline values) while still
    populating the new paired columns so combined-with-PD merges can rely
    on a uniform schema.
    """
    prefill_metrics = _fake_phase_metrics(
        t_a_layer=0.5,
        t_f_layer=0.7,
        balance_ratio=0.71,
        t_step=12.5,
        comm_hidden=True,
    )
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=prefill_metrics,
        decode_metrics=prefill_metrics,
    )

    summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="prefill")
    result = summary.get_result_dict()

    assert result["phase"] == "prefill"

    # Un-prefixed mirrors the prefill values.
    assert result["t_a_layer"] == pytest.approx(0.5)
    assert result["t_f_layer"] == pytest.approx(0.7)
    assert result["balance_ratio"] == pytest.approx(0.71)
    assert result["t_step"] == pytest.approx(12.5)
    assert result["comm_hidden"] is True

    # Prefill-pair matches; decode-pair is NaN/None.
    assert result["prefill_t_a_layer"] == pytest.approx(0.5)
    assert result["prefill_comm_hidden"] is True
    for key in (
        "decode_t_a_layer",
        "decode_t_f_layer",
        "decode_t_a2f_layer",
        "decode_t_f2a_layer",
        "decode_t_c_layer",
        "decode_t_step",
        "decode_balance_ratio",
    ):
        assert math.isnan(result[key]), f"expected NaN {key} in phase=prefill, got {result[key]!r}"
    assert result["decode_comm_hidden"] is None


def test_afd_summary_phase_decode_mirrors_unprefixed_into_decode_pair(monkeypatch):
    """Mirror of the prefill case: single-phase decode populates ``decode_*``
    and the un-prefixed form; ``prefill_*`` are NaN/None.
    """
    decode_metrics = _fake_phase_metrics(
        t_a_layer=1.2,
        t_f_layer=0.9,
        balance_ratio=1.33,
        t_step=50.0,
        comm_hidden=False,
    )
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=decode_metrics,
        decode_metrics=decode_metrics,
    )

    summary = session.run_afd(RuntimeConfig(isl=128, osl=10), phase="decode")
    result = summary.get_result_dict()

    assert result["phase"] == "decode"

    assert result["t_a_layer"] == pytest.approx(1.2)
    assert result["balance_ratio"] == pytest.approx(1.33)
    assert result["comm_hidden"] is False

    assert result["decode_t_a_layer"] == pytest.approx(1.2)
    assert result["decode_comm_hidden"] is False
    for key in (
        "prefill_t_a_layer",
        "prefill_t_f_layer",
        "prefill_t_a2f_layer",
        "prefill_t_f2a_layer",
        "prefill_t_c_layer",
        "prefill_t_step",
        "prefill_balance_ratio",
    ):
        assert math.isnan(result[key]), f"expected NaN {key} in phase=decode, got {result[key]!r}"
    assert result["prefill_comm_hidden"] is None


def test_afd_serial_pipeline_cycle_has_no_overlap(monkeypatch):
    metrics = _fake_phase_metrics(t_a_layer=1.0, t_f_layer=2.0, balance_ratio=0.5)
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=metrics,
        decode_metrics=metrics,
    )
    session._afd_config.pipeline_model = "serial"

    cycle, comm_hidden = session._pipeline_tcycle(1.0, 2.0, 0.5, 0.25)

    assert cycle == pytest.approx(3.75)
    assert comm_hidden is False


def test_afd_serial_pipeline_global_step_is_strict_stage_sum(monkeypatch):
    metrics = _fake_phase_metrics(t_a_layer=1.0, t_f_layer=2.0, balance_ratio=0.5)
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=metrics,
        decode_metrics=metrics,
    )
    session._afd_config.pipeline_model = "serial"

    global_step, cycle, comm_hidden = session._pipeline_global_step_latency(
        1.0,
        2.0,
        0.5,
        0.25,
        num_layers=4,
    )

    assert cycle == pytest.approx(3.75)
    assert global_step == pytest.approx(3.75 * 3 * 4)
    assert comm_hidden is False


def test_afd_pipeline_cycle_rejects_unknown_model(monkeypatch):
    metrics = _fake_phase_metrics(t_a_layer=1.0, t_f_layer=2.0, balance_ratio=0.5)
    session = _build_afd_session_with_phase_metrics(
        monkeypatch,
        prefill_metrics=metrics,
        decode_metrics=metrics,
    )
    session._afd_config.pipeline_model = "unexpected"

    with pytest.raises(ValueError, match="Unsupported AFD pipeline_model: 'unexpected'"):
        session._pipeline_tcycle(1.0, 2.0, 0.5, 0.25)


def test_afd_config_phase_both_with_combined_with_pd_raises():
    """``phase='both'`` + ``combined_with_pd=True`` must fail at construction.

    AFD already covers prefill+decode internally in 'both' mode, so combining
    it with a separate static pool is conceptually inconsistent; the invariant
    is enforced in ``AFDConfig.__post_init__`` for defense-in-depth.
    """
    with pytest.raises(ValueError, match="combined_with_pd=True is incompatible with phase='both'"):
        AFDConfig(
            n_a_nodes=1,
            n_f_nodes=1,
            gpus_per_node=8,
            tp_a=1,
            phase="both",
            combined_with_pd=True,
        )


def test_afd_config_phase_decode_with_combined_with_pd_allowed():
    """``phase='decode'`` + ``combined_with_pd=True`` must construct cleanly.

    This is the default sizing scenario (AFD decode pool + regular prefill
    pool); regressing it would break every default CLI invocation.
    """
    cfg = AFDConfig(
        n_a_nodes=1,
        n_f_nodes=1,
        gpus_per_node=8,
        tp_a=1,
        phase="decode",
        combined_with_pd=True,
    )
    assert cfg.combined_with_pd is True
    assert cfg.phase == "decode"
