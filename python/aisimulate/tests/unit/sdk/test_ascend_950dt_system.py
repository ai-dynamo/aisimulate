# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registration guards for the Ascend 950DT system (M2).

Covers the three hand-authored registration surfaces introduced with
``systems/ascend_950dt.yaml`` — the system YAML itself, ``sdk/common.py``
(``SupportedSystems`` / support-matrix order) and
``collector/evidence_policy.yaml`` — plus the no-regression contract for the
systems that already shipped.

The YAML's numeric content traces to the Ascend 950
architecture white paper and on-device observation: official per-chip
figures (4TB/s HBM bandwidth, 1 PFLOPS FP8, 2TB/s
interconnect) and the observed 96GiB capacity tier with
8 chips per node. The unofficial/placeholder entries are deliberately NOT
pinned here — M5 replaces them with measurements — but they must stay
positive so the strict ``*_tc_flops`` resolver can run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.perf_database.evidence_check import load_policy

from aisimulate.sdk import common
from aisimulate_core.sdk.perf_database import get_supported_databases, load_system_spec
from aisimulate_core.sdk.system_spec import SystemSpec

pytestmark = pytest.mark.unit

SYSTEM = "ascend_950dt"
REPO_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = REPO_ROOT / "collector" / "evidence_policy.yaml"


@pytest.fixture(scope="module")
def spec() -> SystemSpec:
    raw = load_system_spec(SYSTEM)
    assert raw, f"{SYSTEM}.yaml must be discoverable through the packaged systems dir"
    return SystemSpec(raw)


class TestAscend950dtSystemSpec:
    def test_spec_carries_the_verified_facts(self, spec: SystemSpec) -> None:
        gpu = spec["gpu"]
        node = spec["node"]

        assert spec["data_dir"] == "data/ascend_950dt"
        # Observed on g-master1: 98304 MB HBM per chip (the 96GB tier).
        assert gpu["mem_capacity"] == 96 * 1024**3
        assert gpu["mem_bw"] == 4e12  # official 4TB/s
        assert gpu["fp8_tc_flops"] == 1e15  # official 1 PFLOPS FP8/MXFP8/HiF8
        # Official BF16/FP16 peak of the top 36-Cube-Core tier (white paper
        # table 3-1); which of the 547/486/425 tiers this node is is an M5 check.
        assert gpu["bfloat16_tc_flops"] == 547e12
        assert node["num_gpus_per_node"] == 8
        # Per-NPU single direction, derived from the same white paper §4.6 figure
        # as the yaml comment: 2016 GB/s bidirectional = 72 lane x 112 Gbps x 2 / 8,
        # i.e. a ~1008 GB/s unidirectional bound that already assumes every one of
        # the 18 x4 ports is scale-up. M5 must calibrate the value downward.
        assert node["intra_node_bw"] == 1e12

    def test_vendor_is_ascend_and_no_sm_version(self, spec: SystemSpec) -> None:
        """Explicit vendor discrimination; non-CUDA devices carry no sm_version."""
        assert spec["gpu"]["vendor"] == "ascend"
        assert "sm_version" not in spec["gpu"]

    def test_unofficial_flops_entries_are_positive_placeholders(self, spec: SystemSpec) -> None:
        """Strict per-dtype resolution rejects missing or non-positive entries."""
        gpu = spec["gpu"]
        for key in ("int8_tc_flops", "fp32_flops"):
            assert gpu[key] > 0, f"{key} must stay positive until M5 measurement replaces it"

    def test_spec_satisfies_the_rust_system_spec_contract(self, spec: SystemSpec) -> None:
        """Fields the compiled FPE (crates/core .. common/system_spec.rs) requires.

        ``data_dir``, ``gpu.mem_bw`` and the three ``node`` entries have no
        serde default, so a spec missing them fails FPE construction at load.
        """
        assert isinstance(spec["data_dir"], str)
        assert isinstance(spec["gpu"]["mem_bw"], (int, float))
        node = spec["node"]
        assert isinstance(node["num_gpus_per_node"], int)
        for key in ("inter_node_bw", "intra_node_bw"):
            assert isinstance(node[key], (int, float))
        # Topology selection must resolve without KeyError at every tier.
        assert spec.get_p2p_bandwidth(8) == node["intra_node_bw"]
        assert spec.get_p2p_bandwidth(16) == node["inter_node_bw"]

    def test_misc_overheads_are_addressable(self, spec: SystemSpec) -> None:
        """Memory accounting indexes misc.nccl_mem by clamped TP and misc.other_mem."""
        misc = spec["misc"]
        assert set(misc["nccl_mem"]) == {1, 2, 4, 8}
        assert misc["other_mem"] > 0
        # comm (HCCL) data is post-MVP: no comm version is pinned yet.
        assert "nccl_version" not in misc
        assert "oneccl_version" not in misc


class TestAscend950dtRegistration:
    def test_registered_in_supported_systems(self) -> None:
        assert SYSTEM in common.SupportedSystems

    def test_sorts_after_the_nvidia_and_xpu_fleet(self) -> None:
        ordered = common.sort_support_matrix_systems(["a100_sxm", SYSTEM, "b60", "b200_sxm"])
        assert ordered == ["b200_sxm", "a100_sxm", "b60", SYSTEM]

    def test_evidence_policy_has_an_ascend_generation(self) -> None:
        policy = load_policy(POLICY_PATH)
        assert policy.system_generations["ascend"] == frozenset({SYSTEM})
        assert policy.evidence_systems["ascend"] == SYSTEM

    def test_registration_ships_no_perf_data_yet(self) -> None:
        """M2 is registration only: collection (M3/M5) adds the databases."""
        assert SYSTEM not in get_supported_databases()


class TestExistingSystemsUnaffected:
    def test_b60_spec_unchanged_by_the_new_registration(self) -> None:
        b60 = SystemSpec(load_system_spec("b60"))
        assert b60["gpu"]["mem_capacity"] == 24 * 1024**3
        assert b60["node"]["num_gpus_per_node"] == 8
        assert "sm_version" not in b60["gpu"]

    def test_nvidia_fleet_still_resolves(self) -> None:
        h200 = SystemSpec(load_system_spec("h200_sxm"))
        assert h200["gpu"]["sm_version"] == 90
        assert {"h200_sxm", "b200_sxm", "b60"} <= common.SupportedSystems

    def test_existing_support_matrix_order_preserved(self) -> None:
        legacy = [
            "b200_sxm",
            "gb200",
            "b300_sxm",
            "gb300",
            "rtx_pro_6000_server",
            "h200_sxm",
            "h100_sxm",
            "l40s",
            "a100_sxm",
            "b60",
        ]
        assert common.sort_support_matrix_systems(list(reversed(legacy))) == legacy

    def test_evidence_policy_generations_preserved(self) -> None:
        policy = load_policy(POLICY_PATH)
        assert policy.system_generations["xpu"] == frozenset({"b60"})
        assert policy.evidence_systems["hopper"] == "h200_sxm"
