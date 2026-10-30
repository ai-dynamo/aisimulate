# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU checks for the evidence contract; the real cache gate runs on silicon."""

import asyncio
import importlib.util
import json
from itertools import pairwise
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

SPEC = importlib.util.spec_from_file_location("qwen_collect", Path(__file__).with_name("collect.py"))
collect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collect)


def event(prompt=None, tokens=None, finish=None, text=""):
    return {
        "choices": [
            {"index": 0, "prompt_token_ids": prompt, "token_ids": tokens, "finish_reason": finish, "text": text}
        ]
    }


def usage(prompt_length, output_length, cached=0):
    return {
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_length,
            "completion_tokens": output_length,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
        "metrics": {"time_to_first_token_ms": 1.5},
    }


def test_token_bursts_and_metadata_do_not_invent_token_timing():
    record = collect.StreamRecord([10, 11], 10)
    record.add(event(prompt=[10, 11]), 12)
    record.add(event(tokens=[20, 21], text="ab"), 20)
    record.add(event(tokens=[22], text="c", finish="length"), 50)
    record.add(usage(2, 3), 70)
    result = record.result(80, 3)
    assert result["output_token_ids"] == [20, 21, 22]
    assert result["stream_chunk_count"] == 4
    assert result["token_chunk_count"] == 2
    assert result["timing"] == {
        "dispatch_ms": 10,
        "first_token_ms": 20,
        "first_visible_ms": 20,
        "last_token_ms": 50,
        "completed_ms": 80,
        "ttft_ms": 10,
        "tpot_ms": 15,
    }
    assert collect.next_prompt([10, 11], result["output_token_ids"], [30]) == [10, 11, 20, 21, 22, 30]


def test_missing_or_modified_token_evidence_fails():
    record = collect.StreamRecord([10, 11], 0)
    with pytest.raises(RuntimeError, match="changed"):
        record.add(event(prompt=[10]), 1)
    record.add(event(prompt=[10, 11], tokens=[20], finish="length"), 2)
    record.add(usage(2, 1), 3)
    record.usage["prompt_tokens_details"] = {}
    with pytest.raises(RuntimeError, match="cached_tokens"):
        record.result(4, 1)
    record.usage["prompt_tokens_details"] = {"cached_tokens": 0}
    record.output.clear()
    with pytest.raises(RuntimeError, match="exact token IDs"):
        record.result(4, 1)


def test_idle_evidence_and_exact_token_filling():
    assert collect.idle_gauges('vllm:num_requests_running{model="q"} 0\nvllm:num_requests_waiting 0\n') == {
        "running": 0,
        "waiting": 0,
    }
    with pytest.raises(RuntimeError, match="both"):
        collect.idle_gauges("vllm:num_requests_running 0\n")
    assert collect.fit_tokens([1, 2], [3, 4], 7) == [1, 2, 3, 4, 3, 4, 3]


def test_http_stream_and_matching_warm_frontier(tmp_path, monkeypatch):
    async def scenario():
        cached_prompts = []
        resets = []

        async def completion(request):
            payload = await request.json()
            assert payload["ignore_eos"] and payload["return_token_ids"]
            assert not payload["add_special_tokens"]
            assert payload["min_tokens"] == payload["max_tokens"]
            prompt, count = payload["prompt"], payload["max_tokens"]
            common = 0
            for old in cached_prompts:
                prefix = 0
                for left, right in zip(prompt, old, strict=False):
                    if left != right:
                        break
                    prefix += 1
                common = max(common, min(prefix, len(prompt) - 1) // 4 * 4)
            output = list(range(1000, 1000 + count))
            cached_prompts.append(prompt + output[:-1])
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            messages = [
                event(prompt=prompt),
                event(tokens=output, text="x", finish="length"),
                usage(len(prompt), count, common),
            ]
            for message in messages:
                data = ("data: " + json.dumps(message) + "\n\n").encode()
                # Split inside both JSON and SSE separators, as TCP is free to do.
                midpoint = len(data) // 2
                await response.write(data[:midpoint])
                await response.write(data[midpoint:])
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        async def reset(request):
            resets.append(len(cached_prompts))
            if len(resets) == 6:
                return web.json_response({"success": False})
            cached_prompts.clear()
            return web.json_response({"success": True})

        async def metrics(request):
            return web.Response(text="vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n")

        app = web.Application()
        app.router.add_post("/v1/completions", completion)
        app.router.add_post("/reset_prefix_cache", reset)
        app.router.add_get("/metrics", metrics)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as session:
                client = collect.Collector(session, f"http://127.0.0.1:{port}", "test", tmp_path)
                prompt = list(range(600))
                tool = [700] * 512
                await client.case("cold", [prompt], tool, "cold")
                await client.case("warm", [prompt], tool, "warm")
                await client.case("smoke", [prompt[:9]], tool, "warm", smoke=True)
                await client.case("two-lanes", [prompt, [900] * 600], tool, "warm")
                # Deterministic clock: record writing costs 60 ms and prompt building
                # costs 20 ms. The next deadline still belongs to completion + 100.
                virtual_ms = [0.0]
                original_open, original_next_prompt = Path.open, collect.next_prompt

                class SlowRecord:
                    def __init__(self, handle):
                        self.handle = handle

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return self.handle.__exit__(*args)

                    def write(self, data):
                        virtual_ms[0] += 60
                        return self.handle.write(data)

                    def flush(self):
                        return self.handle.flush()

                def slow_open(path, *args, **kwargs):
                    handle = original_open(path, *args, **kwargs)
                    return SlowRecord(handle) if path.name == "requests.jsonl" and args == ("w",) else handle

                def slow_build(*args):
                    virtual_ms[0] += 20
                    return original_next_prompt(*args)

                async def virtual_sleep(seconds):
                    virtual_ms[0] += seconds * 1000

                with monkeypatch.context() as patch:
                    patch.setattr(client, "now", lambda: virtual_ms[0])
                    patch.setattr(Path, "open", slow_open)
                    patch.setattr(collect, "next_prompt", slow_build)
                    patch.setattr(asyncio, "sleep", virtual_sleep)
                    await client.case("slow-logging", [prompt], tool, "cold")
                with pytest.raises(RuntimeError, match="declined"):
                    await client.case("declined-reset", [prompt], tool, "cold")
        finally:
            await runner.cleanup()
        assert len(resets) == 6

    asyncio.run(scenario())
    cold = [json.loads(line) for line in (tmp_path / "cold/requests.jsonl").read_text().splitlines()]
    warm = [json.loads(line) for line in (tmp_path / "warm/requests.jsonl").read_text().splitlines()]
    cold_measured = [row for row in cold if row["phase"] == "measured"]
    warm_measured = [row for row in warm if row["phase"] == "measured"]
    primer = next(row for row in warm if row["phase"] == "warmup")
    assert len(cold_measured) == len(warm_measured) == 8
    assert cold_measured[0]["cached_tokens"] == 0
    assert warm_measured[0]["cached_tokens"] == 88
    assert primer["input_token_ids"] == warm_measured[0]["input_token_ids"][:-512]
    assert cold_measured[0]["input_token_ids"] == warm_measured[0]["input_token_ids"]
    assert warm_measured[0]["dependency_ids"] == [primer["request_id"]]
    for previous, current in pairwise(warm_measured):
        assert current["input_token_ids"] == previous["input_token_ids"] + previous["output_token_ids"] + [700] * 512
        assert current["dependency_ids"] == [previous["request_id"]]
        assert current["think_time_ms"] == 100
        assert current["timing"]["dispatch_ms"] - previous["timing"]["completed_ms"] >= 100
    smoke = [json.loads(line) for line in (tmp_path / "smoke/requests.jsonl").read_text().splitlines()]
    assert [row["cached_tokens"] for row in smoke if row["phase"] != "kernel_warmup"] == [0, 8, 8]
    two_lanes = [json.loads(line) for line in (tmp_path / "two-lanes/requests.jsonl").read_text().splitlines()]
    primers = [row for row in two_lanes if row["phase"] == "warmup"]
    for row in two_lanes:
        if row["phase"] == "measured" and row["turn_index"] == 0:
            assert set(row["dependency_ids"]) == {primer["request_id"] for primer in primers}
            assert row["timing"]["dispatch_ms"] >= max(primer["timing"]["completed_ms"] for primer in primers)
    declined = json.loads((tmp_path / "declined-reset/case.json").read_text())
    assert declined["status"] == "failed"
    assert declined["cache_reset"]["body"] == {"success": False}
    slow = [json.loads(line) for line in (tmp_path / "slow-logging/requests.jsonl").read_text().splitlines()]
    delayed_turns = [row for row in slow if row["phase"] == "measured" and row["turn_index"] > 0]
    assert len(delayed_turns) == 7
    assert all(row["timing"]["realized_dependency_delay_ms"] == 100 for row in delayed_turns)
    assert all(row["timing"]["think_time_overshoot_ms"] == 0 for row in delayed_turns)
    cold_info = json.loads((tmp_path / "cold/case.json").read_text())
    warm_info = json.loads((tmp_path / "warm/case.json").read_text())
    assert collect.compare_trajectories(cold_info, warm_info)["all_realized_tokens_identical"]
    warm_info["trajectory"][2]["output_sha256"] = "drift"
    assert collect.compare_trajectories(cold_info, warm_info)["differing_turns"] == [{"lane_id": 0, "turn_index": 2}]
