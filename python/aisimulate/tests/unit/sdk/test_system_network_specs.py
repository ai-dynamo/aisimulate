# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the sourced network defaults and node/rack boundaries.

Expected values come from docs/SYSTEM_NETWORK_SPECS.md, not measured latency.
Use this checkout's YAMLs explicitly so custom systems paths cannot mask drift.
"""

from pathlib import Path

import pytest

from aiconfigurator_core.sdk.perf_database import load_system_spec
from aiconfigurator_core.sdk.system_spec import SystemSpec

pytestmark = pytest.mark.unit

_SYSTEMS = Path(__file__).resolve().parents[3] / "src/aiconfigurator_core/systems"


@pytest.mark.parametrize(
    ("system", "nvlink_gbytes_per_s", "ports", "port_gbits_per_s"),
    [
        ("b200_sxm", 900, 1, 400),
        ("b300_sxm", 900, 2, 400),
        ("h100_sxm", 450, 2, 200),
        ("h200_sxm", 450, 1, 400),
    ],
)
def test_hgx_reference_bandwidth_is_one_direction_per_gpu(system, nvlink_gbytes_per_s, ports, port_gbits_per_s):
    spec = SystemSpec(load_system_spec(system, systems_paths=str(_SYSTEMS)))
    assert spec["node"]["num_gpus_per_node"] == 8
    assert "num_gpus_per_rack" not in spec["node"]
    assert spec.get_p2p_bandwidth(8) == nvlink_gbytes_per_s * 10**9
    # Aggregate independent ports, not TX + RX, and convert bits to bytes.
    scale_out = ports * port_gbits_per_s * 10**9 / 8
    assert spec["node"]["inter_node_bw"] == scale_out
    assert spec.get_p2p_bandwidth(9) == scale_out
    assert spec.get_p2p_bandwidth(16) == scale_out


@pytest.mark.parametrize("system", ["gb200", "gb300"])
@pytest.mark.parametrize("num_gpus", [4, 5, 8, 72, 73, 144])
def test_nvl72_keeps_nvlink_until_the_rack_boundary(system, num_gpus):
    spec = SystemSpec(load_system_spec(system, systems_paths=str(_SYSTEMS)))
    assert spec["node"]["num_gpus_per_node"] == 4
    assert spec["node"]["num_gpus_per_rack"] == 72
    # The cross-rack value is a retained assumption, NOT an InferenceX fact.
    assert spec["node"]["inter_rack_bw"] == 100 * 10**9
    expected = 900 * 10**9 if num_gpus <= 72 else 100 * 10**9
    assert spec.get_p2p_bandwidth(num_gpus) == expected
