# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.support_matrix.fpe_support_matrix import (
    STATUS_BUILD_FAILED,
    STATUS_FRAMEWORK_INCOMPATIBLE,
    STATUS_HW_INCOMPATIBLE,
    STATUS_MODEL_UNSUPPORTED,
    STATUS_PASS,
    STATUS_PERF_DATA_MISSING,
    STATUS_QUERY_FAILED,
    STATUS_SDK_UNREPRESENTABLE,
    EngineProbePlan,
    MatrixRunMetrics,
    ParallelTopology,
    ProbeWorkload,
    build_probe_plans,
    classify_failure,
    probe_plan,
    write_outputs,
)

pytestmark = pytest.mark.unit


def _plan(**overrides) -> EngineProbePlan:
    values = {
        "model": "test/model",
        "architecture": "TestForCausalLM",
        "system": "b200_sxm",
        "backend": "sglang",
        "backend_version": "0.5.14",
        "forward_model": "op_level",
        "topology": ParallelTopology(2, 1, 1, 1, 2, 1),
        "roles": ("agg", "prefill", "decode"),
        "gemm_quant_mode": "fp8",
        "moe_quant_mode": "fp8",
        "kvcache_quant_mode": "bfloat16",
        "fmha_quant_mode": "bfloat16",
        "comm_quant_mode": "half",
    }
    values.update(overrides)
    return EngineProbePlan(**values)


class _FakeTask:
    def __init__(self, mode: str):
        self.forward_model = "op_level"
        if mode == "agg":
            self.model_path = "test/model"
            self.system_name = "b200_sxm"
            self.backend_name = "sglang"
            self.backend_version = "0.5.14"
        else:
            for role in ("prefill", "decode"):
                setattr(self, f"{role}_model_path", "test/model")
                setattr(self, f"{role}_system_name", "b200_sxm")
                setattr(self, f"{role}_backend_name", "sglang")
                setattr(self, f"{role}_backend_version", "0.5.14")

    def iter_parallel(self, _role):
        # Production Task.iter_parallel() yields mutable lists.  The matrix
        # planner must normalize them before deduplicating topologies.
        return iter([[2, 1, 1, 1, 2, 1]])

    def build_model_config(self, *, role, parallel):
        assert role in {"agg", "prefill", "decode"}
        assert parallel == (2, 1, 1, 1, 2, 1)
        return SimpleNamespace(
            gemm_quant_mode="fp8",
            moe_quant_mode="fp8",
            kvcache_quant_mode="bfloat16",
            fmha_quant_mode="bfloat16",
            comm_quant_mode="half",
            nextn=0,
            moe_comm_backend=None,
            enable_eplb=False,
            moe_backend=None,
            attention_backend="flashinfer",
            language_only=False,
            enable_encoder_dp=True,
        )


def test_build_probe_plans_uses_live_inventory_and_merges_equivalent_roles():
    class FakeMatrix:
        def generate_combinations(self):
            return [("test/model", "b200_sxm", "sglang", "0.5.14")]

        def get_architecture(self, model):
            assert model == "test/model"
            return "TestForCausalLM"

    def create_task(**kwargs):
        assert kwargs["database_mode"] == "SILICON"
        return _FakeTask(kwargs["mode"])

    plans = build_probe_plans(
        matrix=FakeMatrix(),
        create_task=create_task,
        constraints_for_model=lambda _model: object(),
        forward_models=("op_level",),
    )

    assert len(plans) == 1
    assert plans[0].roles == ("agg", "prefill", "decode")
    assert plans[0].topology == ParallelTopology(2, 1, 1, 1, 2, 1)
    assert plans[0].compile_kwargs()["forward_model"] == "op_level"


def test_build_probe_plans_rejects_non_op_level_forward_models():
    with pytest.raises(ValueError, match="unsupported forward models:.*fpm"):
        build_probe_plans(forward_models=("fpm",))


def test_build_probe_plans_keeps_planning_failures_as_fail_closed_rows():
    class FakeMatrix:
        def generate_combinations(self):
            return [("test/model", "h100_sxm", "sglang", "0.5.14")]

        def get_architecture(self, _model):
            return "TestForCausalLM"

    def create_task(**_kwargs):
        raise ValueError("native FP4 weights are not supported on Hopper systems")

    plans = build_probe_plans(
        matrix=FakeMatrix(),
        create_task=create_task,
        constraints_for_model=lambda _model: object(),
        forward_models=("op_level",),
    )

    assert len(plans) == 1
    assert plans[0].roles == ("agg", "prefill", "decode")
    assert {plan.planning_status for plan in plans} == {STATUS_HW_INCOMPATIBLE}

    called = False

    def factory(_plan):
        nonlocal called
        called = True
        raise AssertionError("planner failures must not build a substitute engine")

    results = [
        result
        for plan in plans
        for result in probe_plan(
            plan,
            workload=ProbeWorkload(),
            source_version="0.12.0",
            source_sha="abc123",
            engine_factory=factory,
        )
    ]
    assert not called
    assert {result.status for result in results} == {STATUS_HW_INCOMPATIBLE}
    assert {result.failure_stage for result in results} == {"plan"}


def test_probe_plan_builds_once_and_runs_strict_native_shapes():
    calls = []

    class FakeEngine:
        def predict_prefill_latency(self, **kwargs):
            calls.append(("prefill", kwargs))
            return 1.25

        def predict_decode_latency(self, **kwargs):
            calls.append(("decode", kwargs))
            return 0.5

        def mixed_step_latency(self, **kwargs):
            calls.append(("mixed", kwargs))
            return 1.75

        def last_provenance(self):
            return None

    builds = []

    def factory(plan):
        builds.append(plan.compile_kwargs())
        return FakeEngine()

    results = probe_plan(
        _plan(),
        workload=ProbeWorkload(),
        source_version="0.12.0",
        source_sha="abc123",
        engine_factory=factory,
    )

    assert len(builds) == 1
    assert [result.phase for result in results] == ["prefill", "decode_start", "decode_end", "mixed"]
    assert {result.status for result in results} == {STATUS_PASS}
    assert {result.source for result in results} == {"silicon"}
    assert len(calls) == 4
    assert all("best_available" not in result.reproducer for result in results)


def test_probe_plan_classifies_native_build_data_miss_for_every_required_phase():
    class PerfDataNotAvailableError(RuntimeError):
        pass

    def factory(_plan):
        raise PerfDataNotAvailableError("performance data not available for this shape")

    results = probe_plan(
        _plan(roles=("decode",)),
        workload=ProbeWorkload(),
        source_version="0.12.0",
        source_sha="abc123",
        engine_factory=factory,
    )

    assert [result.phase for result in results] == ["decode_start", "decode_end"]
    assert {result.status for result in results} == {STATUS_PERF_DATA_MISSING}
    assert {result.failure_stage for result in results} == {"build"}


def test_probe_plan_keeps_query_failure_separate_and_continues_other_shapes():
    class FakeEngine:
        def predict_prefill_latency(self, **_kwargs):
            return 1.0

        def predict_decode_latency(self, **kwargs):
            if kwargs["osl"] == 256:
                raise RuntimeError("late decode query failed")
            return 0.5

        def mixed_step_latency(self, **_kwargs):
            return 2.0

        def last_provenance(self):
            return None

    results = probe_plan(
        _plan(),
        workload=ProbeWorkload(),
        source_version="0.12.0",
        source_sha="abc123",
        engine_factory=lambda _plan: FakeEngine(),
    )

    by_phase = {result.phase: result for result in results}
    assert by_phase["decode_end"].status == STATUS_QUERY_FAILED
    assert by_phase["decode_end"].failure_stage == "query"
    assert by_phase["prefill"].status == STATUS_PASS
    assert by_phase["mixed"].status == STATUS_PASS


def test_probe_plan_fails_closed_when_public_sdk_cannot_represent_topology():
    called = False

    def factory(_plan):
        nonlocal called
        called = True
        raise AssertionError("unrepresentable plans must not build a substitute engine")

    results = probe_plan(
        _plan(
            topology=ParallelTopology(2, 1, 1, 1, 2, 2),
            unrepresentable_reasons=("public EngineHandle.compile does not expose cp_size",),
        ),
        workload=ProbeWorkload(),
        source_version="0.12.0",
        source_sha="abc123",
        engine_factory=factory,
    )

    assert not called
    assert {result.status for result in results} == {STATUS_SDK_UNREPRESENTABLE}
    assert all("cp_size" in result.error_message for result in results)


@pytest.mark.parametrize(
    ("error", "stage", "expected"),
    [
        (RuntimeError("plain build failure"), "build", STATUS_BUILD_FAILED),
        (RuntimeError("plain query failure"), "query", STATUS_QUERY_FAILED),
        (RuntimeError("hardware incompatible"), "build", STATUS_HW_INCOMPATIBLE),
        (RuntimeError("framework unsupported"), "build", STATUS_FRAMEWORK_INCOMPATIBLE),
        (RuntimeError("unsupported model family"), "build", STATUS_MODEL_UNSUPPORTED),
    ],
)
def test_classify_failure(error, stage, expected):
    assert classify_failure(error, stage=stage) == expected


def test_result_outputs_are_deterministic_while_run_metrics_remain_separate(tmp_path):
    class FakeEngine:
        def predict_prefill_latency(self, **_kwargs):
            return 1.0

        def last_provenance(self):
            return "sol"

    results = probe_plan(
        _plan(roles=("prefill",)),
        workload=ProbeWorkload(),
        source_version="0.12.0",
        source_sha="abc123",
        engine_factory=lambda _plan: FakeEngine(),
    )
    metrics_a = MatrixRunMetrics(1.0, 0.5, 100, 2, 1, 1)
    metrics_b = MatrixRunMetrics(9.0, 4.5, 900, 2, 1, 1)
    first = write_outputs(
        results=results,
        metrics=metrics_a,
        output_dir=tmp_path / "first",
        source_version="0.12.0",
        source_sha="abc123",
        workload=ProbeWorkload(),
    )
    second = write_outputs(
        results=results,
        metrics=metrics_b,
        output_dir=tmp_path / "second",
        source_version="0.12.0",
        source_sha="abc123",
        workload=ProbeWorkload(),
    )

    for name in ("json", "csv", "markdown"):
        assert first[name].read_bytes() == second[name].read_bytes()
    assert first["metrics"].read_bytes() != second["metrics"].read_bytes()
    assert "does not certify the AISimulate" in first["markdown"].read_text()


def test_probe_workload_rejects_invalid_prefix():
    with pytest.raises(ValueError, match="prefix"):
        ProbeWorkload(isl=128, prefix=129).validate()
