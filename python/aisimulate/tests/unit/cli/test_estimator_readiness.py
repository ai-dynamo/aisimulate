# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message selection when prediction cannot use a trained estimator."""

import pytest

from aisimulate.estimator_readiness import (
    FPM_SELF_SERVICE_GUIDE,
    perf_data_missing_message,
    unready_estimator_message,
)
from aisimulate_core.sdk import perf_database

pytestmark = pytest.mark.unit

SLOTS = {"current": "0.24.0", "next": "0.27.1"}


@pytest.fixture
def vllm_slots(monkeypatch):
    """Pin queryable versions so the tests do not depend on shipped data."""

    def slots(system, backend, systems_paths=None):
        return dict(SLOTS) if (system, backend) == ("h200_sxm", "vllm") else None

    monkeypatch.setattr(perf_database, "get_version_slots", slots)
    monkeypatch.setattr(
        perf_database,
        "get_supported_databases",
        lambda systems_paths=None: {"h200_sxm": {"vllm": ["0.24.0", "0.27.1"], "trtllm": ["1.3.0rc20"]}},
    )
    monkeypatch.delenv("AIC_ALLOW_UNLISTED_VERSIONS", raising=False)


def _diagnostics(failures, **config):
    return {
        "readiness": "unsupported_config",
        "last_warning": "; ".join(failures) or None,
        "provenance": {
            "config": {
                "model": "Qwen/Qwen3-32B-FP8",
                "system": "h200_sxm",
                "backend": "vllm",
                "backend_version": "0.24.0",
                "systems_paths": [],
                **config,
            },
            "selection_failures": failures,
        },
    }


def _architecture_failure(mode, architecture="Phi3ForCausalLM"):
    return (
        f"{mode}: unsupported model for Rust core estimator: compile_engine: ValueError: "
        f"The model's architecture {architecture} is not supported. Supported architectures: LlamaForCausalLM"
    )


def _version_failure(mode, version):
    return (
        f"{mode}: unsupported model for Rust core estimator: compile_engine: ValueError: "
        f"vllm/{version!r} looks like an old-style raw version query"
    )


def test_unsupported_architecture_names_architecture_and_self_service_guide(vllm_slots):
    message = unready_estimator_message(
        _diagnostics(
            [_architecture_failure("OpLevel"), _architecture_failure("FpmInterpolation")],
            model="microsoft/phi-4",
        )
    )

    assert message.startswith(
        "architecture Phi3ForCausalLM (microsoft/phi-4) is not supported by the op-level model; "
        "supported architectures: "
    )
    assert "Qwen3ForCausalLM" in message
    assert FPM_SELF_SERVICE_GUIDE in message
    assert "not ready" not in message


@pytest.mark.parametrize("version", ["0.29.0", "0.25.1"])
def test_unlisted_backend_version_lists_available_versions(vllm_slots, version):
    message = unready_estimator_message(
        _diagnostics(
            [_version_failure("OpLevel", version), _version_failure("FpmInterpolation", version)],
            backend_version=version,
        )
    )

    assert message == f"no timing data for vllm {version} on h200_sxm; available: 0.24.0 (current), 0.27.1 (next)"


def test_missing_version_alias_lists_available_versions(vllm_slots):
    message = unready_estimator_message(
        _diagnostics(["OpLevel: vllm on h200_sxm has no 'previous' version"], backend_version="previous")
    )

    assert message == "no timing data for vllm previous on h200_sxm; available: 0.24.0 (current), 0.27.1 (next)"


def test_backend_without_data_lists_backends_with_data(vllm_slots):
    message = unready_estimator_message(
        _diagnostics(["OpLevel: perf database error: no database"], backend="sglang", backend_version="0.5.16")
    )

    assert message == "no timing data for sglang on h200_sxm; backends with data on h200_sxm: trtllm, vllm"


def test_missing_op_data_reports_each_mode_reason_once(vllm_slots):
    message = unready_estimator_message(
        _diagnostics(
            [
                "OpLevel: perf database error: no rows in dsv41_module_perf.parquet",
                "FpmInterpolation: perf database error: No fpm_forward data collected for this backend/version.",
            ],
            model="org/new-model",
            backend_version="current",
        )
    )

    assert message.startswith(
        "no timing data for org/new-model with vllm current (0.24.0) on h200_sxm: "
        "op_level: perf database error: no rows in dsv41_module_perf.parquet; "
        "fpm_interpolation: perf database error: No fpm_forward data collected for this backend/version. "
    )
    assert "will not fall back to an untrained regression estimator" in message
    assert FPM_SELF_SERVICE_GUIDE in message


def test_identical_mode_reasons_are_merged(vllm_slots):
    message = unready_estimator_message(
        _diagnostics(["OpLevel: shared reason", "FpmInterpolation: shared reason"], model="org/new-model")
    )

    assert "op_level/fpm_interpolation: shared reason." in message
    assert message.count("shared reason") == 1


def test_explicit_untrained_regression_keeps_training_message(vllm_slots):
    diagnostics = _diagnostics([])
    diagnostics["last_warning"] = None

    assert unready_estimator_message(diagnostics) == (
        "regression estimator is not ready; replay requires training observations"
    )


def test_missing_perf_table_row_is_explained():
    error = RuntimeError(
        "AISimulate replay failed: engine error: generalized engine worker 0 is now poisoned because a rank may "
        "have been partially mutated: executing attention-DP rank 0: external prefill prediction failed: AIC "
        "static_ctx evidence prediction failed: PerfDataNotAvailableError: perf database error: MoE data missing "
        'for MoeKey { quant: "bfloat16", topk: 22 } at /data/b200_sxm/vllm/0.24.0'
    )

    message = perf_data_missing_message(error)

    assert message is not None
    assert message.startswith(
        'missing performance data: MoE data missing for MoeKey { quant: "bfloat16", topk: 22 } '
        "at /data/b200_sxm/vllm/0.24.0. "
    )
    assert "poisoned" not in message
    assert FPM_SELF_SERVICE_GUIDE in message


def test_unrelated_replay_error_keeps_original_message():
    assert perf_data_missing_message(RuntimeError("replay deadline exceeded")) is None


def test_prediction_lowering_reports_cause_instead_of_untrained_regression(tmp_path):
    from aisimulate.capacity import materialize_aic_num_gpu_blocks

    raw = {
        "timing_model": {
            "type": "external",
            "provider": "aic",
            "config": {
                "model": "Qwen/Qwen3-30B-A3B",
                "system": "gb300",
                "backend": "sglang",
                "backend_version": "0.5.17",
                "worker_type": "aggregated",
                "estimation_mode": "auto",
                # The empty root makes every native estimator unavailable without a network dependency.
                "systems_paths": [str(tmp_path)],
            },
        }
    }

    with pytest.raises(ValueError, match=r"^no timing data for sglang on gb300; backends with data on gb300: none$"):
        materialize_aic_num_gpu_blocks(raw)
