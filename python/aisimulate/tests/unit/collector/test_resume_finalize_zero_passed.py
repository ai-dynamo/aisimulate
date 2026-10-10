# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""--resume finalization of a producer whose every attempted case failed.

b200_sxm sglang 0.5.21 (GitLab job 469988017, 2026-10-05): moe/int4_wo had
3,078 attempted / 0 passed (lane guard) and therefore no staging table;
``_pending_resume_perf_outputs`` raised "open checkpoint event has no regular
staging table" and the run died in finalization although the failures were
already in the error report. An open event with completed cases and no table
is still a hard error (rows were produced and are missing).

Runs with ``PYTHONPATH=collector`` like test_registry_table_producers (collect.py
is a script-style module importing torch at module level; torch is stubbed).
"""
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _collect():
    if "torch" not in sys.modules:
        from unittest.mock import MagicMock

        _torch = MagicMock()
        _torch.AcceleratorError = type("AcceleratorError", (Exception,), {})
        sys.modules["torch"] = _torch
    import collect

    return collect


def _fake_tracker(attempted, done):
    return types.SimpleNamespace(_attempted=set(attempted), _done=set(done), _failed=set(attempted) - set(done))


@pytest.fixture
def provenance_ctx():
    return {
        "collections": [
            {"name": "sglang", "type": "moe", "module": "collector.sglang.collect_moe", "get_func": "g",
             "run_func": "run_moe_torch", "perf_filename": "moe_perf.txt", "extra_perf_filenames": ()},
        ]
    }


def test_all_failed_producer_without_staging_table_is_skipped_not_fatal(tmp_path, monkeypatch, provenance_ctx):
    collect = _collect()
    monkeypatch.setattr(collect, "_load_selected_producer_checkpoint",
                        lambda *a, **k: _fake_tracker(attempted={"a", "b"}, done=set()))
    pending = collect._pending_resume_perf_outputs(
        tmp_path, provenance_ctx, backend="sglang", checkpoint_dir="ckpt", sm_version=100
    )
    assert pending == []


def test_completed_cases_without_staging_table_still_fail_closed(tmp_path, monkeypatch, provenance_ctx):
    collect = _collect()
    monkeypatch.setattr(collect, "_load_selected_producer_checkpoint",
                        lambda *a, **k: _fake_tracker(attempted={"a", "b"}, done={"a"}))
    with pytest.raises(RuntimeError, match="open checkpoint event has no regular staging table"):
        collect._pending_resume_perf_outputs(
            tmp_path, provenance_ctx, backend="sglang", checkpoint_dir="ckpt", sm_version=100
        )


def test_present_staging_table_with_open_event_is_selected(tmp_path, monkeypatch, provenance_ctx):
    collect = _collect()
    staging = tmp_path / "moe_perf.txt"
    staging.write_text("row\n")
    monkeypatch.setattr(collect, "_load_selected_producer_checkpoint",
                        lambda *a, **k: _fake_tracker(attempted={"a"}, done={"a"}))
    pending = collect._pending_resume_perf_outputs(
        tmp_path, provenance_ctx, backend="sglang", checkpoint_dir="ckpt", sm_version=100
    )
    assert pending == [Path(staging)]
