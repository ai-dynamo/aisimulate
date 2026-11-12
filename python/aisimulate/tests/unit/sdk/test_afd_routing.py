# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AFD routing execution preflight and canonical identity contracts."""

import json
from pathlib import Path

import pytest
import yaml

from aisimulate.compiler import prediction_to_replay_spec
from aisimulate.config.cli import CorePredictionConfig
from aisimulate_core._native import (
    AFD_DEFAULT_ROUTING_LOCATION,
    RustForwardPassPerfModel,
    validate_afd_routing_location,
)
from aisimulate_core.sdk.rust_engine_step import ForwardPassPerfModelConfig

pytestmark = pytest.mark.unit


def test_native_routing_support_and_errors():
    assert AFD_DEFAULT_ROUTING_LOCATION == "a"
    assert validate_afd_routing_location("a") is None
    with pytest.raises(NotImplementedError, match="routing_location='f' is not implemented; use routing_location='a'"):
        validate_afd_routing_location("f")
    with pytest.raises(ValueError, match="routing_location"):
        validate_afd_routing_location("invalid")


@pytest.mark.parametrize("policy", ["deny", "allow", "regression"])
def test_canonical_f_request_fails_before_loading_or_fallback(policy, tmp_path):
    config = ForwardPassPerfModelConfig(
        model="missing/model",
        system="missing_system",
        backend="sglang",
        worker_type="aggregated",
        systems_paths=(str(tmp_path),),
        afd_routing_location="f",
        fallback_policy=policy,
    )
    payload = config.to_dict()
    assert payload["afd_routing_location"] == "f"
    payload["systems_paths"] = ["/does/not/exist"]
    with pytest.raises(NotImplementedError, match="routing_location='f'"):
        RustForwardPassPerfModel.best_available(json.dumps(payload))


def test_ordinary_canonical_identity_omits_routing_location(tmp_path):
    config = ForwardPassPerfModelConfig(
        model="missing/model",
        system="missing_system",
        backend="sglang",
        worker_type="aggregated",
        systems_paths=(str(tmp_path),),
    )
    assert "afd_routing_location" not in config.to_dict()


def test_prediction_f_request_fails_before_hardware_or_model_resolution(monkeypatch):
    root = Path(__file__).resolve().parents[5]
    payload = yaml.safe_load((root / "crates/core/parity_tests/perfmodel/configs/afd-predict-pure.yaml").read_text())
    payload["engine"]["workers"]["aggregated"]["afd"]["routing_location"] = "f"
    payload["engine"]["model"] = "missing/model"
    payload["engine"]["hardware"] = "missing_system"
    config = CorePredictionConfig.model_validate(payload)
    restored = CorePredictionConfig.model_validate_json(config.model_dump_json())
    assert restored.engine.workers.aggregated.afd.routing_location == "f"
    monkeypatch.setattr(
        "aisimulate.compiler.resolve_model_hardware", lambda *args, **kwargs: pytest.fail("unexpected hardware lookup")
    )
    with pytest.raises(NotImplementedError, match="routing_location='f'"):
        prediction_to_replay_spec(restored)


def test_normalized_canonical_identity_preserves_explicit_a():
    payload = {
        "model": "missing/model",
        "system": "missing_system",
        "backend": "sglang",
        "worker_type": "aggregated",
        "afd_routing_location": "a",
    }
    resolved = json.loads(RustForwardPassPerfModel.normalize_config(json.dumps(payload)))
    assert resolved["afd_routing_location"] == "a"
