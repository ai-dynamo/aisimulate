# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored hardware receipts validate admission, never establish GPU execution."""

from types import SimpleNamespace

import pytest
from collector.glm53flash_protocol import native_gpu_identity, validate_gb300_identity

pytestmark = pytest.mark.unit


def identity():
    return {
        "schema": "glm53flash_gpu_identity_v1",
        "name": "NVIDIA GB300",
        "compute_capability": [10, 3],
        "total_memory_bytes": 1 << 38,
        "cuda_device_index": 2,
        "uuid": "authored-device-uuid",
    }


def test_selected_worker_cuda_device_is_the_receipt_source(monkeypatch):
    monkeypatch.setenv("GPU_NAME", "NVIDIA H100")
    calls = []

    def selected(name, result):
        def read(index):
            calls.append((name, index))
            return result

        return read

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            current_device=lambda: 2,
            get_device_name=selected("name", "NVIDIA GB300"),
            get_device_capability=selected("capability", (10, 3)),
            get_device_properties=selected(
                "properties", SimpleNamespace(total_memory=1 << 38, uuid="authored-device-uuid")
            ),
        )
    )
    receipt = native_gpu_identity(torch)
    assert receipt == identity()
    assert calls == [("properties", 2), ("name", 2), ("capability", 2)]
    validate_gb300_identity(receipt)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema": "unknown"},
        {"name": "NVIDIA B200"},
        {"name": "NVIDIA GB3000"},
        {"compute_capability": [10, 0]},
        {"compute_capability": [True, 3]},
        {"compute_capability": [10.0, 3]},
        {"compute_capability": [10, 3, 0]},
        {"total_memory_bytes": True},
        {"total_memory_bytes": 0},
        {"cuda_device_index": False},
        {"cuda_device_index": -1},
        {"uuid": ""},
    ],
)
def test_hardware_admission_rejects_unknown_target_or_invalid_native_properties(changes):
    with pytest.raises(ValueError, match="native GPU"):
        validate_gb300_identity({**identity(), **changes})


def test_optional_uuid_does_not_replace_required_hardware_receipt():
    value = identity()
    del value["uuid"]
    validate_gb300_identity(value)
    for key in tuple(value):
        with pytest.raises(ValueError, match="native GPU"):
            validate_gb300_identity({name: item for name, item in value.items() if name != key})
