# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only protocol and accounting checks for the Rubin serving benchmark."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from collector.sglang_rubin import benchmark

pytestmark = pytest.mark.unit


def choice(text="", finish_reason=None):
    return {"choices": [{"index": 0, "text": text, "finish_reason": finish_reason}]}


def usage(**overrides):
    return {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, **overrides}}


def encode_events(events):
    return b"".join(
        b"data: " + (event if isinstance(event, str) else json.dumps(event, ensure_ascii=False)).encode() + b"\n\n"
        for event in events
    )


def good_events():
    return [choice("Hello"), choice(" world", "length"), usage(), "[DONE]"]


@pytest.fixture
def endpoint():
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            status, content_type, chunks = self.server.respond(body)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            try:
                for chunk in chunks:
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # Deadline tests intentionally close before the server finishes.

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.respond = lambda body: (200, "text/event-stream", [encode_events(good_events())])
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server, bodies, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def run_cli(endpoint, tmp_path, *extra):
    _, _, base = endpoint
    output = tmp_path / "result.json"
    status = benchmark.main(
        [
            "--api-base-url",
            base,
            "--model",
            "glm52",
            "--prompt",
            "measured prompt",
            "--concurrencies",
            "1,2",
            "--requests-per-worker",
            "1",
            "--output",
            str(output),
            *extra,
        ]
    )
    return status, json.loads(output.read_text())


def test_stream_timings_use_text_arrivals_and_reported_usage(monkeypatch):
    times = iter([10.1, 10.5, 10.6, 10.7])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    result = benchmark._consume(benchmark._events([encode_events(good_events())]), 10, 7, 3)
    assert result["generated_text"] == "Hello world"
    assert result["ttft_s"] == pytest.approx(0.1)
    assert result["latency_s"] == pytest.approx(0.7)
    assert result["tpot_s"] == pytest.approx(0.2)
    assert result["output_tokens_per_s"] == pytest.approx(3 / 0.7)
    assert result["text_chunks"] == 2


def test_sse_accepts_split_utf8_crlf_comments_and_multiline_data():
    raw = ': keepalive\r\ndata: {"choices": [],\r\ndata: "text": "π"}\r\n\r\n'.encode()
    events = list(benchmark._events([bytes([byte]) for byte in raw]))
    assert json.loads(events[0]) == {"choices": [], "text": "π"}


@pytest.mark.parametrize(
    "events,error",
    [
        ([choice("hello", "stop"), "[DONE]"], "without final usage"),
        ([choice("hello", "stop"), usage(completion_tokens=0), "[DONE]"], "positive integer completion_tokens"),
        ([choice("hello", "stop"), usage(completion_tokens=True), "[DONE]"], "positive integer completion_tokens"),
        ([choice("hello", "stop"), usage(completion_tokens="3"), "[DONE]"], "positive integer completion_tokens"),
        ([choice("hello", "stop"), usage(prompt_tokens=None), "[DONE]"], "positive integer prompt_tokens"),
        ([choice(" ", "stop"), usage(), "[DONE]"], "no nonempty generated text"),
        ([choice("hello"), "[DONE]"], "without a finish reason"),
        ([choice("hello", "stop"), usage(total_tokens=99), "[DONE]"], "total_tokens does not match"),
        ([choice("hello", "stop"), usage(), usage(), "[DONE]"], "after final usage"),
        ([choice("hello", "stop"), choice("more"), usage(), "[DONE]"], "after its finish reason"),
        ([{"error": {"message": "out of memory"}}], "Server error"),
        (good_events()[:-1], "without \\[DONE\\]"),
        ([choice("hello"), usage(), "[DONE]"], "after a completed choice"),
        ([choice("hello", "abort")], "Unsuccessful or unsupported finish reason"),
        ([{"choices": [choice()["choices"][0]] * 2}], "at most one completion choice"),
        (["{invalid"], "Expecting property name"),
        (["null"], "must be a JSON object"),
        (["NaN"], "Non-JSON numeric constant"),
    ],
)
def test_invalid_streams_fail(events, error):
    with pytest.raises(ValueError, match=error):
        benchmark._consume(benchmark._events([encode_events(events)]), time.perf_counter(), None, None)


def test_partial_event_and_oversized_event_fail(monkeypatch):
    with pytest.raises(benchmark.StreamError, match="inside an event"):
        list(benchmark._events([b"data: [DONE]\n"]))
    monkeypatch.setattr(benchmark, "_MAX_EVENT_BYTES", 10)
    with pytest.raises(benchmark.StreamError, match="size limit"):
        list(benchmark._events([b"data: " + b"a" * 10]))


@pytest.mark.parametrize(
    "prompt_tokens,output_tokens,error", [(8, None, "Expected 8 prompt"), (None, 500, "Expected 500 completion")]
)
def test_exact_lengths_are_optional_but_enforced_when_requested(prompt_tokens, output_tokens, error):
    with pytest.raises(benchmark.StreamError, match=error):
        benchmark._consume(
            benchmark._events([encode_events(good_events())]), time.perf_counter(), prompt_tokens, output_tokens
        )


def test_one_completion_token_has_no_tpot():
    events = [choice("Hello", "stop"), usage(completion_tokens=1), "[DONE]"]
    result = benchmark._consume(benchmark._events([encode_events(events)]), time.perf_counter(), None, None)
    assert result["tpot_s"] is None


def test_cli_measures_all_requests_and_keeps_warmup_separate(endpoint, tmp_path):
    status, artifact = run_cli(endpoint, tmp_path)
    assert status == 0
    assert artifact["valid"]
    assert len(artifact["warmup"]) == 1
    assert [wave["requests_total"] for wave in artifact["waves"]] == [1, 2]
    _, bodies, _ = endpoint
    assert len(bodies) == 4
    assert bodies[0]["prompt"] != bodies[1]["prompt"]
    assert bodies[0]["max_tokens"] == 32
    assert all(body["max_tokens"] == 500 for body in bodies[1:])
    assert all(body["stream_options"] == {"include_usage": True} for body in bodies)
    assert all(body["stream"] and body["n"] == 1 and body["temperature"] == 0 for body in bodies)
    for wave in artifact["waves"]:
        assert wave["requests_failed"] == 0
        assert wave["output_tokens_per_s"] == pytest.approx(3 * wave["requests_total"] / wave["wall_s"])
        assert all(request["usage"]["completion_tokens"] == 3 for request in wave["requests"])


def test_any_measured_failure_invalidates_wave_and_exit_status(endpoint, tmp_path):
    server, bodies, _ = endpoint
    server.respond = lambda body: (
        200,
        "text/event-stream",
        [encode_events([{"error": "bad request"}] if len(bodies) == 2 else good_events())],
    )
    status, artifact = run_cli(endpoint, tmp_path, "--concurrencies", "1", "--requests-per-worker", "2")
    wave = artifact["waves"][0]
    assert status == 1 and not artifact["valid"]
    assert wave["requests_total"] == 2
    assert wave["requests_failed"] == wave["requests_succeeded"] == 1
    assert wave["output_tokens_per_s"] is None
    assert wave["mean_ttft_s"] is None
    assert "Server error" in wave["requests"][0]["error"]


def test_failed_warmup_prevents_measured_requests_and_is_recorded(endpoint, tmp_path):
    server, bodies, _ = endpoint
    server.respond = lambda body: (503, "text/plain", [b"unavailable"])
    status, artifact = run_cli(endpoint, tmp_path)
    assert status == 1
    assert len(bodies) == 1
    assert artifact["waves"] == []
    assert "HTTP 503" in artifact["warmup"][0]["error"]


def test_fixed_length_workload_requests_ignore_eos_and_checks_actual_count(endpoint, tmp_path):
    status, artifact = run_cli(
        endpoint, tmp_path, "--concurrencies", "1", "--output-tokens", "3", "--ignore-eos", "--require-output-length"
    )
    assert status == 0
    assert all(body["ignore_eos"] for body in endpoint[1])
    assert artifact["configuration"]["ignore_eos"]
    status, artifact = run_cli(endpoint, tmp_path, "--concurrencies", "1", "--require-output-length")
    assert status == 1
    assert "Expected 500 completion tokens, observed 3" in artifact["waves"][0]["requests"][0]["error"]


def test_wrong_response_type_is_a_failure(endpoint):
    server, _, base = endpoint
    server.respond = lambda body: (200, "application/json", [b"{}"])
    result = benchmark.request_completion(api_base_url=base, model="glm52", prompt="hello", max_tokens=3, timeout=1)
    assert not result["ok"]
    assert "Content-Type" in result["error"]


def test_total_request_deadline_includes_continuing_heartbeats(endpoint):
    def heartbeats():
        for _ in range(30):
            time.sleep(0.01)
            yield b": keepalive\n\n"

    server, _, base = endpoint
    server.respond = lambda body: (200, "text/event-stream", heartbeats())
    started = time.perf_counter()
    result = benchmark.request_completion(api_base_url=base, model="glm52", prompt="hello", max_tokens=3, timeout=0.06)
    assert not result["ok"]
    assert result["error_type"] == "TimeoutError"
    assert time.perf_counter() - started < 1


def test_prompt_list_consumed_once_and_server_configuration_preserved(endpoint, tmp_path):
    _, bodies, base = endpoint
    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["prefix one", "prefix two", "other three"]))
    config = tmp_path / "server.json"
    config.write_text(json.dumps({"launch_args": ["--tp-size", "4"], "cache_policy": "caller-controlled"}))
    output = tmp_path / "result.json"
    status = benchmark.main(
        [
            "--api-base-url",
            base,
            "--model",
            "glm52",
            "--prompts-file",
            str(prompts),
            "--concurrencies",
            "1,2",
            "--requests-per-worker",
            "1",
            "--serving-config",
            str(config),
            "--output",
            str(output),
        ]
    )
    assert status == 0
    assert sorted(body["prompt"] for body in bodies[1:]) == sorted(json.loads(prompts.read_text()))
    assert json.loads(output.read_text())["configuration"]["server_configuration"]["launch_args"] == ["--tp-size", "4"]


@pytest.mark.parametrize(
    "extra", [["--concurrencies", "1,0"], ["--concurrencies", "1,1"], ["--request-timeout", "nan"], ["--prompt", " "]]
)
def test_invalid_configuration_is_rejected_before_requests(endpoint, tmp_path, extra):
    with pytest.raises(SystemExit) as exc:
        run_cli(endpoint, tmp_path, *extra)
    assert exc.value.code == 2
    assert not endpoint[1]


def test_prompt_list_must_cover_every_measured_request(endpoint, tmp_path):
    prompts = tmp_path / "prompts.json"
    prompts.write_text('["only one"]')
    with pytest.raises(SystemExit) as exc:
        benchmark.main(
            [
                "--api-base-url",
                endpoint[2],
                "--model",
                "glm52",
                "--prompts-file",
                str(prompts),
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
    assert exc.value.code == 2
    assert not endpoint[1]


@pytest.mark.parametrize("contents", ["null", '{"bad": NaN}'])
def test_invalid_server_configuration_is_rejected_before_requests(endpoint, tmp_path, contents):
    config = tmp_path / "server.json"
    config.write_text(contents)
    with pytest.raises(SystemExit) as exc:
        run_cli(endpoint, tmp_path, "--serving-config", str(config))
    assert exc.value.code == 2
    assert not endpoint[1]
