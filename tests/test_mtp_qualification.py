# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification failures must remain failures under -O and retain receipts."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/qualify_agentx_mtp.py"


def _module():
    spec = importlib.util.spec_from_file_location("mtp_qualification", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qualification_checks_survive_optimized_python():
    program = "import runpy,sys; runpy.run_path(sys.argv[1])['require'](False, 'expected failure')"
    result = subprocess.run(
        [sys.executable, "-O", "-c", program, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "expected failure" in result.stderr


def test_native_report_envelope_preserves_acceptance_and_authored_metadata():
    module = _module()
    summary = {"speculative_acceptance": {"mean_accept_length": 2.5}, "completed_requests": 2}
    assert module.unpack_report(summary) == summary
    envelope = {
        "summary": summary,
        "speculation": {"aggregated": {"seed": 42}},
        "agentic_qualification": "functional_only",
    }
    assert module.unpack_report(envelope) == {**summary, **{k: v for k, v in envelope.items() if k != "summary"}}
    assert "agentic_qualification" not in module.unpack_report({"summary": summary})


def test_local_wheel_is_hashed_when_installer_omits_archive_hash(tmp_path, monkeypatch):
    wheel = tmp_path / "wheel artifact.whl"
    wheel.write_bytes(b"installed wheel fixture")
    metadata = {"direct_url.json": json.dumps({"url": wheel.as_uri(), "archive_info": {}}), "RECORD": "record"}
    module = _module()
    monkeypatch.setattr(module, "distribution", lambda _: SimpleNamespace(version="0.13.0", read_text=metadata.get))
    receipt = module.package_receipt("fixture")
    assert receipt["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert receipt["record_sha256"] == hashlib.sha256(b"record").hexdigest()


@pytest.mark.parametrize("exit_code,expected", [(3, "CLI exited"), (0, "FileNotFoundError")])
def test_failed_cli_or_missing_report_preserves_receipt(tmp_path, monkeypatch, exit_code, expected):
    module = _module()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], exit_code),
    )
    result = module.run_case(
        ("broken", {}, "engine", True),
        cli="fixture",
        output=tmp_path,
        duration=3600,
        trace_sha256="fixture",
    )
    saved = json.loads((tmp_path / "broken/receipt.json").read_text())
    assert saved == result
    assert saved["status"] == "failed"
    assert expected in saved["error"]


def test_empty_acceptance_has_useful_error():
    report = {
        "agentic_qualification": "functional_only",
        "agentic_model_projection": {"target_model": "fixture"},
        "completed_requests": 1,
        "speculative_acceptance": {"decode_forwards": 1, "mean_accept_length": None},
    }
    with pytest.raises(ValueError, match="missing or nonfinite mean"):
        _module().validate_report(report, {"engine": {"model": "fixture"}}, True, 3600)


def test_missing_sd_metadata_cannot_qualify():
    report = {
        "agentic_qualification": "functional_only",
        "agentic_model_projection": {"target_model": "fixture"},
        "completed_requests": 1,
        "speculation": {},
        "speculative_acceptance": {
            "decode_forwards": 1000,
            "mean_accept_length": 2.5,
            "sampling_population": "measurement_completed_decode_passes",
        },
    }
    raw = {"engine": {"model": "fixture", "workers": {"aggregated": {}}, "speculation": {"num_speculative_tokens": 3}}}
    with pytest.raises(ValueError, match="missing or unexpected SD role"):
        _module().validate_report(report, raw, True, 3600)


@pytest.mark.parametrize(
    "outcome",
    [
        {"status": "failed", "settled_at_ms": 12.0},
        {"status": "incomplete", "settled_at_ms": 12.0},
        {"status": "completed"},
    ],
)
def test_complete_trace_cannot_qualify_unsettled_or_failed_plays(outcome):
    report = {
        "agentic_qualification": "functional_only",
        "agentic_model_projection": {"target_model": "fixture"},
        "completed_requests": 1,
        "agentic_graph": {"play_count": 1},
        "agentic_play_outcomes": [outcome],
        "speculative_acceptance": {
            "decode_forwards": 10,
            "mean_accept_length": 1.0,
            "sampling_population": "measurement_completed_decode_passes",
        },
    }
    with pytest.raises(ValueError, match="unsuccessful or unsettled play"):
        _module().validate_report(report, {"engine": {"model": "fixture"}}, False, None)
    report["agentic_play_outcomes"] = [{"status": "completed", "settled_at_ms": 12.0}]
    _module().validate_report(report, {"engine": {"model": "fixture"}}, False, None)
