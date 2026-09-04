# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic staged AFD foreground-pass tests."""

import pytest

from aisimulate.sweeper.afd import AFDLayerTimes, AFDPipelineModel, AFDTopology
from aisimulate.sweeper.afd_engine import AFDForegroundEngine, AFDStage


def _topology(**overrides) -> AFDTopology:
    values = {
        "n_a_nodes": 1,
        "n_f_nodes": 1,
        "gpus_per_node": 4,
        "tp_a": 2,
        "a_batch_size": 8,
        "num_microbatches": 3,
        "phase": "decode",
        "combined_with_pd": False,
    }
    values.update(overrides)
    return AFDTopology(**values)


def _times(phase: str = "decode", **overrides) -> AFDLayerTimes:
    values = {
        "phase": phase,
        "attention_ms": 2.0,
        "ffn_ms": 3.0,
        "a_to_f_ms": 0.5,
        "f_to_a_ms": 0.5,
        "num_layers": 2,
    }
    values.update(overrides)
    return AFDLayerTimes(**values)


def test_foreground_pass_expands_every_stage_and_matches_formula_boundary():
    engine = AFDForegroundEngine(_topology(), (_times(),))

    planned = engine.execute_pass(
        phase="decode",
        now_ms=10.0,
        input_length=128,
        output_length=32,
    )

    assert len(planned.intervals) == 2 * 3 * 4
    assert [interval.stage for interval in planned.intervals[:4]] == [
        AFDStage.ATTENTION,
        AFDStage.A_TO_F,
        AFDStage.FFN,
        AFDStage.F_TO_A,
    ]
    assert planned.intervals[4].start_ms - planned.intervals[
        0
    ].start_ms == pytest.approx(planned.evaluation.cycle_ms)
    assert planned.intervals[-1].end_ms == pytest.approx(planned.end_ms)
    assert planned.end_ms - planned.started_at_ms == pytest.approx(
        planned.evaluation.step_latency_ms
    )

    completed = engine.complete_pass(planned.pass_id, now_ms=planned.end_ms + 5.0)
    assert completed.completed_at_ms == planned.end_ms
    assert completed.completed_sequences == 16
    assert completed.completed_tokens == 16
    assert engine.in_flight is None


def test_optimistic_schedule_records_conservative_fallback_cadence():
    topology = _topology(num_microbatches=2, pipeline_model="optimistic")
    times = _times(a_to_f_ms=5.0, f_to_a_ms=5.0)
    engine = AFDForegroundEngine(topology, (times,))

    planned = engine.execute_pass(
        phase="decode",
        now_ms=0.0,
        input_length=16,
        output_length=8,
    )

    assert planned.evaluation.effective_pipeline_model is AFDPipelineModel.CONSERVATIVE
    assert planned.intervals[4].start_ms == pytest.approx(planned.evaluation.cycle_ms)


def test_both_phase_engine_reuses_pool_but_requires_completion_barrier():
    topology = _topology(phase="both")
    engine = AFDForegroundEngine(topology, (_times("prefill"), _times("decode")))
    prefill = engine.execute_pass(
        phase="prefill",
        now_ms=0.0,
        input_length=128,
        output_length=32,
    )

    with pytest.raises(RuntimeError, match="already in flight"):
        engine.execute_pass(
            phase="decode",
            now_ms=0.0,
            input_length=128,
            output_length=32,
        )

    completion = engine.complete_pass(prefill.pass_id, now_ms=prefill.end_ms)
    assert completion.completed_tokens == 16 * 128
    decode = engine.execute_pass(
        phase="decode",
        now_ms=prefill.end_ms,
        input_length=128,
        output_length=32,
    )
    assert decode.pass_id == prefill.pass_id + 1


def test_combined_topology_executes_only_its_afd_phase():
    topology = _topology(phase="prefill", combined_with_pd=True)
    engine = AFDForegroundEngine(topology, (_times("prefill"),))

    with pytest.raises(ValueError, match="no 'decode' measurement"):
        engine.execute_pass(
            phase="decode",
            now_ms=0.0,
            input_length=128,
            output_length=32,
        )

    planned = engine.execute_pass(
        phase="prefill",
        now_ms=0.0,
        input_length=128,
        output_length=32,
    )
    assert planned.phase.value == "prefill"


def test_completion_rejects_wrong_id_and_early_boundary_without_releasing_pass():
    engine = AFDForegroundEngine(_topology(), (_times(),))
    planned = engine.execute_pass(
        phase="decode",
        now_ms=5.0,
        input_length=128,
        output_length=32,
    )

    with pytest.raises(ValueError, match="ID mismatch"):
        engine.complete_pass(planned.pass_id + 1, now_ms=planned.end_ms)
    with pytest.raises(ValueError, match="before its modeled boundary"):
        engine.complete_pass(planned.pass_id, now_ms=planned.end_ms - 0.1)
    assert engine.in_flight is planned
