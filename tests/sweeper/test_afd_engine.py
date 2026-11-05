# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic staged AFD foreground-pass tests."""

import pytest

from aisimulate.sweeper.afd_engine import (
    AFDForegroundEngine,
    AFDStage,
    evaluate_afd_phase,
)
from aisimulate.sweeper.afd_parallel import AFDPipelineModel, AFDTopology
from aisimulate.sweeper.afd_perfmodel import AFDLayerTimes


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


def test_pipeline_evaluator_matches_legacy_optimistic_formula():
    topology = _topology()
    result = evaluate_afd_phase(
        topology,
        _times(attention_ms=1.0, ffn_ms=2.0, a_to_f_ms=0.25, f_to_a_ms=0.25),
        input_length=128,
        output_length=16,
    )

    # fill=3.5; cycle=max(1,2,.5)=2; global step=3.5+2*(3*2-1)
    assert result.pipeline_fill_ms == pytest.approx(3.5)
    assert result.cycle_ms == pytest.approx(2.0)
    assert result.step_latency_ms == pytest.approx(13.5)
    assert result.communication_hidden is True
    assert result.effective_pipeline_model.value == "optimistic"
    assert result.balance_ratio == pytest.approx(0.5)
    assert result.tokens_per_second == pytest.approx(topology.total_batch_size / 0.0135)
    assert result.sequence_rate == pytest.approx((topology.total_batch_size / 0.0135) / 16)


def test_optimistic_pipeline_falls_back_when_microbatch_count_is_too_small():
    topology = _topology(num_microbatches=2)
    result = evaluate_afd_phase(
        topology,
        _times(attention_ms=1, ffn_ms=1, a_to_f_ms=1, f_to_a_ms=1),
        input_length=128,
        output_length=16,
    )

    assert result.requested_pipeline_model.value == "optimistic"
    assert result.effective_pipeline_model.value == "conservative"
    assert result.communication_hidden is False
    assert result.cycle_ms == pytest.approx(2.0)
    assert result.step_latency_ms == pytest.approx(10.0)


def test_communication_overhead_is_applied_and_provenanced():
    result = evaluate_afd_phase(
        _topology(comm_overhead_factor=2.0),
        _times(attention_ms=1.0, ffn_ms=2.0, a_to_f_ms=0.25, f_to_a_ms=0.25),
        input_length=128,
        output_length=16,
    )

    assert result.provenance["layer_times"]["a_to_f_ms"] == pytest.approx(0.5)
    assert result.provenance["layer_times"]["f_to_a_ms"] == pytest.approx(0.5)


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
    assert planned.intervals[4].start_ms - planned.intervals[0].start_ms == pytest.approx(planned.evaluation.cycle_ms)
    assert planned.intervals[-1].end_ms == pytest.approx(planned.end_ms)
    assert planned.end_ms - planned.started_at_ms == pytest.approx(planned.evaluation.step_latency_ms)

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


@pytest.mark.parametrize("pipeline_model", ["conservative", "optimistic", "serial"])
@pytest.mark.parametrize("num_microbatches", [1, 2, 4])
def test_next_layer_waits_for_its_microbatch_to_return(pipeline_model, num_microbatches):
    topology = _topology(num_microbatches=num_microbatches, pipeline_model=pipeline_model)
    times = _times(attention_ms=1.0, ffn_ms=1.0, a_to_f_ms=1.0, f_to_a_ms=1.0, num_layers=3)
    planned = AFDForegroundEngine(topology, (times,)).execute_pass(
        phase="decode",
        now_ms=10.0,
        input_length=16,
        output_length=8,
    )

    returns = {
        (interval.layer, interval.microbatch): interval.end_ms
        for interval in planned.intervals
        if interval.stage is AFDStage.F_TO_A
    }
    for interval in planned.intervals:
        if interval.stage is AFDStage.ATTENTION and interval.layer > 0:
            assert interval.start_ms >= returns[interval.layer - 1, interval.microbatch]


@pytest.mark.parametrize("pipeline_model", ["conservative", "optimistic", "serial"])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_single_microbatch_reports_serial_timing_and_requested_model(pipeline_model, phase):
    topology = _topology(
        gpus_per_node=1,
        tp_a=1,
        a_batch_size=1,
        num_microbatches=1,
        pipeline_model=pipeline_model,
        phase=phase,
        comm_overhead_factor=2.0,
    )
    times = _times(phase, attention_ms=1.0, ffn_ms=1.0, a_to_f_ms=1.0, f_to_a_ms=1.0)
    engine = AFDForegroundEngine(topology, (times,))
    planned = engine.execute_pass(
        phase=phase,
        now_ms=10.0,
        input_length=16,
        output_length=8,
        latency_correction=1.5,
    )

    # With no other microbatch to overlap, each layer takes 1.5+3+1.5+3 ms.
    assert [interval.end_ms for interval in planned.intervals] == pytest.approx(
        [11.5, 14.5, 16.0, 19.0, 20.5, 23.5, 25.0, 28.0]
    )
    assert planned.end_ms == pytest.approx(28.0)
    evaluation = planned.evaluation
    assert evaluation.step_latency_ms == pytest.approx(18.0)
    assert evaluation.requested_pipeline_model.value == pipeline_model
    assert evaluation.effective_pipeline_model is AFDPipelineModel.SERIAL
    assert evaluation.communication_hidden is False
    assert evaluation.provenance["topology"]["topology"]["pipeline_model"] == pipeline_model
    assert evaluation.provenance["layer_times"]["a_to_f_ms"] == pytest.approx(2.0)
    assert evaluation.provenance["layer_times"]["f_to_a_ms"] == pytest.approx(2.0)
    assert evaluation.provenance["latency_correction"] == pytest.approx(1.5)
    if phase == "decode":
        assert evaluation.tokens_per_second == pytest.approx(1 / 0.018)
        assert evaluation.sequence_rate == pytest.approx(1 / 0.018 / 8)
    else:
        assert evaluation.tokens_per_second == pytest.approx(16 / 0.018)
        assert evaluation.sequence_rate == pytest.approx(1 / 0.018)

    with pytest.raises(ValueError, match="before its modeled boundary"):
        engine.complete_pass(planned.pass_id, now_ms=27.0)
    completion = engine.complete_pass(planned.pass_id, now_ms=planned.end_ms)
    assert completion.completed_at_ms == pytest.approx(28.0)
    assert completion.pass_latency_ms == pytest.approx(18.0)


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
