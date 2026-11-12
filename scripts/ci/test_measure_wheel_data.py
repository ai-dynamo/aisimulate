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


def test_tree_estimate_has_no_packaged_data():
    out = m.summarize(m.tree_members())
    assert out["packaged_op_data"]["files"] == 0
    assert "systems-other" in out["buckets"]


def _wheel(tmp_path, members):
    wheel = tmp_path / "w.whl"
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_STORED) as z:
        for name, payload in members.items():
            z.writestr(name, payload)
    return wheel


def test_limit_passes_small_wheel_without_data(tmp_path):
    wheel = _wheel(tmp_path, {"aisim_core/systems/b200_sxm.yaml": b"x" * 100})
    assert m.main(["--wheel", str(wheel), "--max-wheel-mib", "1"]) == 0


def test_limit_fails_oversized_wheel(tmp_path):
    wheel = _wheel(tmp_path, {"aisim/blob.bin": b"x" * (2 * 1024 * 1024)})
    assert m.main(["--wheel", str(wheel), "--max-wheel-mib", "1"]) == 1


def test_limit_fails_wheel_that_packages_op_data(tmp_path):
    wheel = _wheel(tmp_path, {"aisim_core/systems/data/h100_sxm/a.parquet": b"x"})
    assert m.main(["--wheel", str(wheel), "--max-wheel-mib", "1"]) == 1


def test_limit_requires_a_wheel():
    import pytest

    with pytest.raises(SystemExit):
        m.main(["--max-wheel-mib", "1"])
