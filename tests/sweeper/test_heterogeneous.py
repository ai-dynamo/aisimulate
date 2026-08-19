# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from aisimulate import RoleEngineRequestSpec
from aisimulate.sweeper.heterogeneous import (
    DisaggBackendPair,
    DisaggRateMatchControls,
    DisaggRole,
    RoleEstimate,
    RoleEstimatorSpecs,
    RoleFailureCategory,
    RoleIdentity,
    RoleSearchError,
    rate_match_disaggregated,
)
from aisimulate.sweeper.replay import EstimatorSpec, canonical_json


def _identity(
    role: DisaggRole,
    *,
    model: str,
    system: str,
    backend: str,
    version: str,
    inherited_fields: tuple[str, ...] = (),
) -> RoleIdentity:
    return RoleIdentity(
        role=role,
        model_name=model,
        hardware_sku=system,
        backend=backend,
        backend_version=version,
        inherited_fields=inherited_fields,
        provenance={"resolved_by": "test"},
    )


def _estimate(
    identity: RoleIdentity,
    *,
    rate: float,
    latency_ms: float,
    workers: int,
    gpus_per_worker: int,
) -> RoleEstimate:
    return RoleEstimate(
        identity=identity,
        sequence_rate_per_worker=rate,
        latency_ms=latency_ms,
        workers=workers,
        gpus_per_worker=gpus_per_worker,
        parallel_config={"tp": gpus_per_worker, "replicas": workers},
        provenance={"estimator": identity.backend_version},
    )


def _estimator(
    role: DisaggRole, *, architecture: str = "ExampleForCausalLM"
) -> EstimatorSpec:
    return EstimatorSpec(
        model_path=f"{role.value}/model",
        model_architecture=architecture,
        system=f"{role.value}_system",
        backend="vllm",
        backend_version="1",
        performance_data_version="1",
        database_mode="SILICON",
        transfer_policy=("xshape",),
        forward_model="op_level",
        engine_step_backend="rust",
        systems_paths=(f"/{role.value}/systems",),
        performance_data_root=f"/{role.value}/systems",
    )


def _role_estimators(
    *, decode_architecture: str = "ExampleForCausalLM"
) -> RoleEstimatorSpecs:
    prefill = _estimator(DisaggRole.PREFILL)
    decode = _estimator(DisaggRole.DECODE, architecture=decode_architecture)
    return RoleEstimatorSpecs(
        pair=DisaggBackendPair("vllm", "vllm"),
        prefill=prefill,
        decode=decode,
        identities={
            "prefill": _identity(
                DisaggRole.PREFILL,
                model=prefill.model_path,
                system=prefill.system,
                backend=prefill.backend,
                version=prefill.backend_version,
            ),
            "decode": _identity(
                DisaggRole.DECODE,
                model=decode.model_path,
                system=decode.system,
                backend=decode.backend,
                version=decode.backend_version,
            ),
        },
    )


def test_backend_pair_preserves_role_assignment() -> None:
    pair = DisaggBackendPair(prefill="sglang", decode="vllm")

    assert pair.homogeneous is False
    assert pair.label == "prefill=sglang,decode=vllm"
    assert pair.backend_for(DisaggRole.PREFILL) == "sglang"
    assert pair.backend_for("decode") == "vllm"
    assert DisaggBackendPair("vllm", "vllm").homogeneous is True


def test_role_engine_request_contract_is_public() -> None:
    request = RoleEngineRequestSpec(
        role="prefill",
        backend="sglang",
        backend_version="0.4.9",
    )

    assert request.role == "prefill"
    assert request.backend == "sglang"


def test_role_identity_is_losslessly_json_serializable() -> None:
    identity = _identity(
        DisaggRole.PREFILL,
        model="prefill-model",
        system="gb200_nv18",
        backend="sglang",
        version="0.4.9",
        inherited_fields=("systems_paths", "database_mode"),
    )

    payload = identity.as_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["role"] == "prefill"
    assert payload["inherited_fields"] == ["systems_paths", "database_mode"]
    assert payload["provenance"] == {"resolved_by": "test"}


def test_role_estimator_contract_rejects_identity_mismatch() -> None:
    estimators = _role_estimators()

    with pytest.raises(RoleSearchError, match="identity does not match") as exc_info:
        RoleEstimatorSpecs(
            pair=estimators.pair,
            prefill=estimators.prefill,
            decode=estimators.decode,
            identities={
                **estimators.identities,
                "decode": _identity(
                    DisaggRole.DECODE,
                    model="wrong/model",
                    system=estimators.decode.system,
                    backend=estimators.decode.backend,
                    version=estimators.decode.backend_version,
                ),
            },
        )

    assert exc_info.value.role is DisaggRole.DECODE
    assert exc_info.value.category is RoleFailureCategory.INVALID_IDENTITY


def test_role_estimator_contract_rejects_incompatible_kv_architectures() -> None:
    with pytest.raises(
        RoleSearchError, match="incompatible for KV handoff"
    ) as exc_info:
        _role_estimators(decode_architecture="DifferentForCausalLM")

    assert exc_info.value.role is DisaggRole.DECODE
    assert exc_info.value.category is RoleFailureCategory.INVALID_IDENTITY


def test_default_rate_matching_preserves_heterogeneous_role_provenance() -> None:
    prefill = _estimate(
        _identity(
            DisaggRole.PREFILL,
            model="prefill-model",
            system="gb200_nv18",
            backend="sglang",
            version="0.4.9",
        ),
        rate=10.0,
        latency_ms=20.0,
        workers=2,
        gpus_per_worker=4,
    )
    decode = _estimate(
        _identity(
            DisaggRole.DECODE,
            model="decode-model",
            system="h200_sxm",
            backend="vllm",
            version="0.10.1",
        ),
        rate=8.0,
        latency_ms=2.0,
        workers=3,
        gpus_per_worker=2,
    )

    result = rate_match_disaggregated(prefill, decode, output_length=5)
    payload = result.as_dict()

    assert result.sequence_rate == pytest.approx(18.0)
    assert result.tokens_per_second == pytest.approx(90.0)
    assert result.total_gpus == 14
    assert result.prefill_gpus == 8
    assert result.decode_gpus == 6
    assert result.tokens_per_second_per_gpu == pytest.approx(90.0 / 14.0)
    assert result.limiting_role is DisaggRole.PREFILL
    assert result.ttft_ms == pytest.approx(20.0 * 1.1 * 1.8)
    assert result.tpot_ms == pytest.approx(2.0 * 1.08)
    assert result.request_latency_ms == pytest.approx(
        result.ttft_ms + 4 * result.tpot_ms
    )
    assert payload["role_identities"]["prefill"]["backend"] == "sglang"
    assert payload["role_identities"]["decode"]["hardware_sku"] == "h200_sxm"
    assert payload["provenance"]["role_provenance"] == {
        "prefill": {"estimator": "0.4.9"},
        "decode": {"estimator": "0.10.1"},
    }
    assert json.loads(json.dumps(payload)) == payload


def test_custom_rate_controls_can_make_decode_limiting() -> None:
    prefill = _estimate(
        _identity(
            DisaggRole.PREFILL,
            model="m",
            system="h100_sxm",
            backend="vllm",
            version="1",
        ),
        rate=10.0,
        latency_ms=10.0,
        workers=2,
        gpus_per_worker=1,
    )
    decode = _estimate(
        _identity(
            DisaggRole.DECODE,
            model="m",
            system="h100_sxm",
            backend="vllm",
            version="2",
        ),
        rate=5.0,
        latency_ms=1.0,
        workers=2,
        gpus_per_worker=1,
    )
    controls = DisaggRateMatchControls(
        prefill_degradation=1.0,
        decode_degradation=0.5,
        prefill_latency_correction=2.0,
        decode_latency_correction=3.0,
        ttft_correction_factor=4.0,
    )

    result = rate_match_disaggregated(
        prefill,
        decode,
        output_length=2,
        controls=controls,
    )

    assert result.sequence_rate == pytest.approx(5.0)
    assert result.limiting_role is DisaggRole.DECODE
    assert result.ttft_ms == pytest.approx(80.0)
    assert result.tpot_ms == pytest.approx(3.0)


def test_rate_matching_rejects_finite_inputs_that_overflow_outputs() -> None:
    prefill = _estimate(
        _identity(
            DisaggRole.PREFILL,
            model="m",
            system="h100_sxm",
            backend="vllm",
            version="1",
        ),
        rate=1e308,
        latency_ms=1e308,
        workers=2,
        gpus_per_worker=1,
    )
    decode = _estimate(
        _identity(
            DisaggRole.DECODE,
            model="m",
            system="h100_sxm",
            backend="vllm",
            version="1",
        ),
        rate=1e308,
        latency_ms=1e308,
        workers=2,
        gpus_per_worker=1,
    )

    with pytest.raises(ValueError, match="finite"):
        rate_match_disaggregated(prefill, decode, output_length=2)


@pytest.mark.parametrize(
    ("workers", "controls", "field"),
    [
        (2, None, "standalone_sequence_rate"),
        (
            1,
            DisaggRateMatchControls(prefill_degradation=2.0),
            "effective_sequence_rate",
        ),
    ],
)
def test_rate_matching_rejects_role_rate_arithmetic_overflow(
    workers: int,
    controls: DisaggRateMatchControls | None,
    field: str,
) -> None:
    prefill = _estimate(
        _identity(
            DisaggRole.PREFILL,
            model="m",
            system="h100_sxm",
            backend="vllm",
            version="1",
        ),
        rate=1e308,
        latency_ms=1.0,
        workers=workers,
        gpus_per_worker=1,
    )
    decode = _estimate(
        _identity(
            DisaggRole.DECODE,
            model="m",
            system="h100_sxm",
            backend="vllm",
            version="1",
        ),
        rate=1.0,
        latency_ms=1.0,
        workers=1,
        gpus_per_worker=1,
    )

    with pytest.raises(RoleSearchError) as exc_info:
        rate_match_disaggregated(
            prefill, decode, output_length=2, controls=controls
        )

    assert exc_info.value.role is DisaggRole.PREFILL
    assert exc_info.value.category is RoleFailureCategory.INVALID_ESTIMATE
    assert exc_info.value.as_dict()["provenance"] == {
        "field": field,
        "value": float("inf"),
    }


def test_rate_match_result_validates_role_rates_before_canonical_json() -> None:
    prefill = _estimate(
        _identity(
            DisaggRole.PREFILL,
            model="m",
            system="h100_sxm",
            backend="vllm",
            version="1",
        ),
        rate=1.0,
        latency_ms=1.0,
        workers=1,
        gpus_per_worker=1,
    )
    decode = _estimate(
        _identity(
            DisaggRole.DECODE,
            model="m",
            system="h100_sxm",
            backend="vllm",
            version="1",
        ),
        rate=1.0,
        latency_ms=1.0,
        workers=1,
        gpus_per_worker=1,
    )

    result = rate_match_disaggregated(prefill, decode, output_length=2)

    assert canonical_json(result.as_dict())
    with pytest.raises(ValueError, match=r"role_rates\['prefill_standalone'\]"):
        type(result)(
            **{
                **result.__dict__,
                "role_rates": {
                    **result.role_rates,
                    "prefill_standalone": float("inf"),
                },
            }
        )


def test_gpu_budget_failure_is_role_attributed_and_actionable() -> None:
    prefill = _estimate(
        _identity(
            DisaggRole.PREFILL,
            model="m",
            system="gb200_nv18",
            backend="sglang",
            version="1",
        ),
        rate=1.0,
        latency_ms=1.0,
        workers=2,
        gpus_per_worker=4,
    )
    decode = _estimate(
        _identity(
            DisaggRole.DECODE,
            model="m",
            system="h200_sxm",
            backend="vllm",
            version="2",
        ),
        rate=100.0,
        latency_ms=1.0,
        workers=1,
        gpus_per_worker=2,
    )

    with pytest.raises(RoleSearchError) as exc_info:
        rate_match_disaggregated(
            prefill,
            decode,
            output_length=4,
            gpu_budget=9,
        )

    error = exc_info.value
    assert error.role is DisaggRole.PREFILL
    assert error.category is RoleFailureCategory.GPU_BUDGET
    assert error.as_dict()["provenance"] == {
        "prefill_gpus": 8,
        "decode_gpus": 2,
        "total_gpus": 10,
        "gpu_budget": 9,
    }
    assert "exceeding gpu_budget=9" in str(error)


@pytest.mark.parametrize(
    ("kwargs", "role", "field"),
    [
        ({"rate": 0.0, "workers": 1}, DisaggRole.PREFILL, "sequence_rate_per_worker"),
        ({"rate": 1.0, "workers": 0}, DisaggRole.DECODE, "workers"),
    ],
)
def test_invalid_estimates_report_the_responsible_role(
    kwargs: dict[str, float | int],
    role: DisaggRole,
    field: str,
) -> None:
    identity = _identity(
        role,
        model="m",
        system="h100_sxm",
        backend="vllm",
        version="1",
    )

    with pytest.raises(RoleSearchError) as exc_info:
        _estimate(
            identity,
            rate=float(kwargs["rate"]),
            latency_ms=1.0,
            workers=int(kwargs["workers"]),
            gpus_per_worker=1,
        )

    assert exc_info.value.role is role
    assert exc_info.value.category is RoleFailureCategory.INVALID_ESTIMATE
    assert exc_info.value.as_dict()["provenance"]["field"] == field
