# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runtime probe preserves token-count transitions without retaining content."""

import io
import json
import urllib.error
from types import SimpleNamespace

import collect
import instrument
import pytest
import rsi_timing
from instrument import inject


def test_client_waits_for_ready_and_generation_before_reading_server_info(tmp_path, monkeypatch):
    events = []

    def response(url, **kwargs):
        events.append(url.rsplit("/", 1)[-1])
        if events == ["ready"]:
            raise urllib.error.URLError("server starting despite available server info")
        return io.BytesIO(b"{}")

    process = SimpleNamespace(pid=123456789, poll=lambda: None, wait=lambda **_: 0)
    monkeypatch.setattr(collect.subprocess, "Popen", lambda *_, **__: process)
    monkeypatch.setattr(collect.urllib.request, "urlopen", response)
    monkeypatch.setattr(collect.time, "sleep", lambda _: None)
    monkeypatch.setattr(collect.os, "killpg", lambda *_: None)
    monkeypatch.setattr(collect, "sample", lambda *_: None)
    monkeypatch.setattr(collect.subprocess, "run", lambda *_, **__: events.append("probe"))
    collect.control(tmp_path, tmp_path / "model", float("inf"), False, True)
    assert events == ["ready", "ready", "health_generate", "get_server_info", "probe"]


def test_launch_keeps_compilation_and_model_caches_inside_campaign(tmp_path, monkeypatch):
    monkeypatch.delenv("RSI_CACHE_ROOT", raising=False)
    captured = {}

    def failed_process(*args, **kwargs):
        captured.update(kwargs["env"])
        return SimpleNamespace(poll=lambda: 1)

    monkeypatch.setattr(collect.subprocess, "Popen", failed_process)
    with pytest.raises(RuntimeError, match="startup failed"):
        collect.control(tmp_path, tmp_path / "model", float("inf"), False, True)
    for variable in (
        "HF_HOME",
        "HF_MODULES_CACHE",
        "SGLANG_CACHE_DIR",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        assert captured[variable].startswith(str(tmp_path / "cache") + "/")


def test_editable_install_uses_package_origin_not_distribution_location(tmp_path, monkeypatch, capsys):
    path = tmp_path / "srt/managers/scheduler_components/output_streamer.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        "class SchedulerOutputStreamer:\n    def stream_output(self, reqs, return_logprob):\n        pass\n"
    )
    monkeypatch.setattr(instrument.importlib.metadata, "distribution", lambda _: SimpleNamespace(version="0.5.21"))
    monkeypatch.setattr(
        instrument.importlib.util, "find_spec", lambda _: SimpleNamespace(origin=str(tmp_path / "__init__.py"))
    )
    instrument.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["sglang_version"] == "0.5.21"
    assert receipt["original_sha256"] != receipt["instrumented_sha256"]
    assert '__import__("rsi_timing").record(reqs)' in path.read_text()


def test_both_output_layouts_require_one_known_interface():
    for owner in ("SchedulerOutputStreamer", "SchedulerOutputProcessorMixin"):
        source = f"class {owner}:\n    def stream_output(self, reqs, return_logprob):\n        pass\n"
        modified = inject(source)
        assert modified.count('__import__("rsi_timing").record(reqs)') == 1
        with pytest.raises(ValueError, match="already instrumented"):
            inject(modified)
    with pytest.raises(ValueError, match="Unexpected"):
        inject("class Unknown: pass\n")


def test_probe_records_numerical_transitions_and_ignores_other_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_TIMING_ROOT", str(tmp_path))
    monkeypatch.setattr(rsi_timing, "_last", {})
    monkeypatch.setattr(rsi_timing, "_stream", None)
    request = SimpleNamespace(rid="rsi-test", output_ids=[11, 12])
    unrelated = SimpleNamespace(rid="private-health-request", output_ids=[99])
    try:
        rsi_timing.record([request, unrelated])
        rsi_timing.record([request])
        request.output_ids += [13]
        rsi_timing.record([request])
        rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
        assert [row["output_tokens"] for row in rows] == [2, 3]
        assert all(set(row) == {"rid", "at_ns", "output_tokens", "batch_requests", "rank"} for row in rows)
        assert rows[0]["rank"] is None
        assert rows[0]["batch_requests"] == 2
    finally:
        if rsi_timing._stream:
            rsi_timing._stream.close()
