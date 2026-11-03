# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runtime probe preserves token-count transitions without retaining content."""

import asyncio
import hashlib
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import collect
import instrument
import pytest
import rsi_dispatch
import rsi_frontend
import rsi_timing
from instrument import inject


@pytest.mark.parametrize("invalid", ["model", "runtime", "cases"])
@pytest.mark.parametrize("suite", ["frontend", "blocking", "matrix", "coding-blocking"])
def test_frontend_preflight_mismatch_stops_before_server_start(tmp_path, monkeypatch, invalid, suite):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    receipt = {
        "model_config_sha256": hashlib.sha256(b"{}").hexdigest(),
        "transformers": "5.12.1",
        "cases": [{"context_label": n} for n in (8192, 131072, 393216)],
    }
    if invalid == "model":
        receipt["model_config_sha256"] = "wrong"
    elif invalid == "runtime":
        receipt["transformers"] = "wrong"
    else:
        receipt["cases"] = receipt["cases"][:1]
    (tmp_path / "frontend-template-preflight.json").write_text(json.dumps(receipt))
    output = tmp_path / "gpu-123/controls"
    monkeypatch.setattr(
        sys,
        "argv",
        ["collect.py", "--model", str(model), "--output", str(output), "--seconds", "30", "--suite", suite],
    )
    monkeypatch.setattr(collect.importlib.metadata, "version", lambda _: "5.12.1")
    monkeypatch.setattr(collect.subprocess, "Popen", lambda *_a, **_k: pytest.fail("server must remain unstarted"))
    with pytest.raises(ValueError, match="preflight"):
        collect.main()
    assert not output.exists()


def test_matrix_source_change_is_rejected_before_server_start(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (tmp_path / "frontend-template-preflight.json").write_text(json.dumps({
        "model_config_sha256": hashlib.sha256(b"{}").hexdigest(), "transformers": "5.12.1",
        "cases": [{"context_label": n} for n in (8192, 131072, 393216)],
    }))
    sources = {name: hashlib.sha256(Path(collect.__file__).with_name(name).read_bytes()).hexdigest()
               for name in rsi_dispatch.PREFLIGHT_SOURCE_FILES}
    sources["matrix_dispatch.py"] = "changed-after-preflight"
    (tmp_path / "matrix-dispatch-preflight.json").write_text(json.dumps({
        "status": "passed", "sglang": "0.5.20", "prepared_request": {"status": "passed"},
        "source_sha256": sources, "dispatch_names": {"status": "passed"},
    }))
    output = tmp_path / "gpu-123/controls"
    monkeypatch.setattr(sys, "argv", ["collect.py", "--model", str(model), "--output", str(output),
                                    "--seconds", "30", "--suite", "matrix"])
    monkeypatch.setattr(collect.importlib.metadata, "version", lambda name: "0.5.20" if name == "sglang" else "5.12.1")
    monkeypatch.setattr(collect.subprocess, "Popen", lambda *_a, **_k: pytest.fail("server must remain unstarted"))
    with pytest.raises(ValueError, match="source changed"):
        collect.main()
    assert not output.exists()


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
    request = SimpleNamespace(rid="rsi-test", output_ids=[11, 12], finished_len=None)
    unrelated = SimpleNamespace(rid="private-health-request", output_ids=[99])
    try:
        rsi_timing.record([request, unrelated])
        rsi_timing.record([request])
        request.output_ids += [13]
        rsi_timing.record([request])
        rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
        assert [row["output_tokens"] for row in rows] == [2, 3]
        assert all(set(row) == {
            "schema", "rid", "at_ns", "output_tokens", "batch_requests", "rank",
            "generated_output_tokens", "finished_len", "discarded_tail_tokens",
        } for row in rows)
        assert all(row["schema"] == "aisimulate.glm-rsi.output-ready.v2" for row in rows)
        assert rows[0]["rank"] is None
        assert rows[0]["batch_requests"] == 2
    finally:
        if rsi_timing._stream:
            rsi_timing._stream.close()


def test_output_probe_preserves_generated_emitted_and_discarded_tail_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_TIMING_ROOT", str(tmp_path))
    monkeypatch.setattr(rsi_timing, "_last", {})
    monkeypatch.setattr(rsi_timing, "_stream", None)
    request = SimpleNamespace(rid="rsi-tail", output_ids=list(range(18)), finished_len=16)
    try:
        rsi_timing.record([request])
        row = json.loads(next(tmp_path.glob("*.jsonl")).read_text())
        assert row["output_tokens"] == 16
        assert row["generated_output_tokens"] == 18
        assert row["finished_len"] == 16
        assert row["discarded_tail_tokens"] == 2
        assert row["generated_output_tokens"] == row["output_tokens"] + row["discarded_tail_tokens"]
    finally:
        if rsi_timing._stream:
            rsi_timing._stream.close()


def test_dispatch_receipt_follows_submission_and_rejects_unknown_source(monkeypatch):
    source = """class Manager:
    async def _send_one_request(self, tokenized_obj):
        self._dispatch_to_scheduler(tokenized_obj)
        self.finished()
"""
    events = []
    monkeypatch.setattr(rsi_dispatch, "record", lambda _: events.append("receipt"))
    namespace = {}
    exec(instrument.inject_dispatch(source), namespace)
    manager = namespace["Manager"]()
    manager._dispatch_to_scheduler = lambda _: events.append("submission")
    manager.finished = lambda: events.append("finished")
    asyncio.run(manager._send_one_request(None))
    assert events == ["submission", "receipt", "finished"]
    with pytest.raises(ValueError, match="already instrumented"):
        instrument.inject_dispatch(instrument.inject_dispatch(source))
    with pytest.raises(ValueError, match="boundary"):
        instrument.inject_dispatch(source.replace("self._dispatch_to_scheduler(tokenized_obj)", "pass"))
    with pytest.raises(ValueError, match="boundary"):
        instrument.inject_dispatch(source.replace("_dispatch_to_scheduler(tokenized_obj)", "_dispatch_to_scheduler()"))


@pytest.mark.parametrize("prefix", ["rsi-block-", "rsi-submit-"])
def test_dispatch_projection_cannot_publish_prompt_or_escape_root(tmp_path, monkeypatch, prefix):
    monkeypatch.setenv("RSI_DISPATCH_ROOT", str(tmp_path))
    rid = prefix + "../../outside"
    rsi_dispatch.record(SimpleNamespace(rid=rid, input_ids=[123456, 654321], text="private prompt"))
    path = rsi_dispatch.receipt_path(tmp_path, rid)
    assert path.parent == tmp_path
    value = json.loads(path.read_text())
    assert value["input_tokens"] == 2
    assert set(value) == {"rid", "at_ns", "input_tokens", "pid", "boundary"}
    assert "123456" not in path.read_text() and "private prompt" not in path.read_text()


@pytest.mark.parametrize("invalid", ["runtime", "source"])
def test_blocking_dispatch_preflight_rejects_identity_changes(tmp_path, monkeypatch, invalid):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (tmp_path / "frontend-template-preflight.json").write_text(
        json.dumps(
            {
                "model_config_sha256": hashlib.sha256(b"{}").hexdigest(),
                "transformers": "5.12.1",
                "cases": [{"context_label": n} for n in (8192, 131072, 393216)],
            }
        )
    )
    (tmp_path / "blocking-dispatch-preflight.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "sglang": "wrong" if invalid == "runtime" else "0.5.20",
                "source_sha256": {"instrument.py": "wrong"},
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect.py",
            "--model",
            str(model),
            "--output",
            str(tmp_path / "gpu-1/controls"),
            "--seconds",
            "30",
            "--suite",
            "blocking",
        ],
    )
    monkeypatch.setattr(
        collect.importlib.metadata, "version", lambda name: {"sglang": "0.5.20", "transformers": "5.12.1"}[name]
    )
    monkeypatch.setattr(collect.subprocess, "Popen", lambda *_a, **_k: pytest.fail("GPU server must not start"))
    with pytest.raises(ValueError, match="preflight"):
        collect.main()
