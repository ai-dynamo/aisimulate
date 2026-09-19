# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from aisimulate.legacy_cli import report_and_save

pytestmark = pytest.mark.unit


def test_save_results_propagates_unrecoverable_failures(monkeypatch, tmp_path):
    task = SimpleNamespace(
        primary_backend_name="sglang",
        primary_model_path="Qwen/Qwen3-32B",
        primary_system_name="b200_sxm",
        isl=256,
        osl=256,
        ttft=2000.0,
        tpot=50.0,
    )

    def fail_mkdir(*_args, **_kwargs):
        raise OSError("cannot create result directory")

    monkeypatch.setattr(report_and_save, "safe_mkdir", fail_mkdir)

    with pytest.raises(OSError, match="cannot create result directory"):
        report_and_save.save_results(
            args=SimpleNamespace(inclusive_tpot=False),
            best_configs={},
            pareto_fronts={},
            tasks={"agg": task},
            save_dir=str(tmp_path),
            backend="sglang",
        )
