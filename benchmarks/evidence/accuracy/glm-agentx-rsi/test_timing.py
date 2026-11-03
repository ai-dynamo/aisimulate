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
import rsi_frontend
import rsi_timing
from instrument import inject


@pytest.mark.parametrize("speculation,policy", [(True, "agentx-golden"), (True, "natural"), (False, "agentx-golden")])
def test_original_acceptance_overrides_are_explicit_and_never_leak_to_plain(tmp_path, monkeypatch, speculation, policy):
    captured = {}
    for variable in collect.GOLDEN_ACCEPTANCE:
        monkeypatch.setenv(variable, "inherited-wrong-value")

    def failed_process(*args, **kwargs):
        captured.update(kwargs["env"])
        return SimpleNamespace(poll=lambda: 1)

    monkeypatch.setattr(collect.subprocess, "Popen", failed_process)
    with pytest.raises(RuntimeError, match="startup failed"):
        collect.control(tmp_path, tmp_path / "model", float("inf"), False, speculation, policy)
    actual = {key: captured[key] for key in collect.GOLDEN_ACCEPTANCE if key in captured}
    assert actual == (collect.GOLDEN_ACCEPTANCE if speculation and policy == "agentx-golden" else {})


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


def test_frontend_hook_requires_one_undecorated_known_sync_method():
    source = "class Chat:\n    def _render_and_encode_chat_template(self, messages):\n        return '', [1], None\n"
    updated = instrument.inject_frontend(source)
    assert updated.count('@__import__("rsi_frontend").measure_template') == 1
    with pytest.raises(ValueError, match="already instrumented"):
        instrument.inject_frontend(updated)
    with pytest.raises(ValueError, match="Unexpected"):
        instrument.inject_frontend("class Unknown: pass\n")


def test_frontend_timer_excludes_numerical_projection_and_preserves_result(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_TIMING_ROOT", str(tmp_path))
    times = iter([1000000, 4000000, 5000000])
    monkeypatch.setattr(rsi_frontend.time, "monotonic_ns", lambda: next(times))
    original = ("private rendered prompt", [1, 2, 70000], None)
    wrapped = rsi_frontend.measure_template(lambda *args, **kwargs: original)
    assert wrapped([], use_cache=False) is original
    text = next(tmp_path.glob("frontend-*.jsonl")).read_text()
    row = json.loads(text)
    assert row["service_ms"] == 3 and row["projection_ms"] == 1
    assert row["input_tokens"] == 3 and row["template_cache_requested"] is False
    assert row["input_sha256"] == rsi_frontend.token_digest(original[1])
    assert "private" not in text and "70000" not in text


def test_frontend_hook_is_inactive_without_measurement_root(monkeypatch):
    monkeypatch.delenv("RSI_TIMING_ROOT", raising=False)
    monkeypatch.setattr(rsi_frontend.time, "monotonic_ns", lambda: pytest.fail("timer should remain inactive"))
    assert rsi_frontend.measure_template(lambda: "unchanged")() == "unchanged"


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
