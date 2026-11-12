# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the wheel op-data size report."""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import measure_wheel_data as m


def test_summarize_separates_op_data():
    out = m.summarize(
        [
            ("aisim_core/systems/data/b200_sxm/gemm/x.parquet", 100, 90),
            ("aisim_core/systems/b200_sxm.yaml", 10, 2),
            ("aisim/_runtime.abi3.so", 50, 20),
        ]
    )
    assert out["packaged_op_data"]["compressed_bytes"] == 90
    assert out["without_data_compressed_bytes"] == 22
    assert set(out["buckets"]) == {"data/b200_sxm", "systems-other", "native"}


def test_wheel_members_reads_sizes(tmp_path):
    wheel = tmp_path / "w.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("aisim_core/systems/data/h100_sxm/a.parquet", b"x" * 1000)
    out = m.summarize(m.wheel_members(wheel))
    assert out["packaged_op_data"]["bytes"] == 1000


def test_tree_estimate_includes_packaged_data():
    assert m.summarize(m.tree_members())["packaged_op_data"]["files"] > 0
