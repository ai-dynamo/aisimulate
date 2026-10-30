#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Collect controlled Qwen agent chains through the vLLM 0.31 completion API.

This is an independent experiment client, not an InferenceX benchmark importer.
Protocol reference: vllm-project/vllm@v0.31.0, completion/protocol.py,
completion/serving.py, and serve/dev/cache/api_router.py under vllm/entrypoints/.
Requires aiohttp and a dedicated, otherwise idle server with token details and
per-request metrics enabled. All times are client monotonic milliseconds.
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import aiohttp

CORPUS = (
    "Review the following engineering notes. Keep the supplied context and continue the analysis. "
    "The service receives a request, checks its dependencies, processes a batch, and records the result. "
    "Explain the next useful check with a concrete reason. Preserve identifiers and observations.\n"
)
SCHEMA = "qwen35-agentic-request/v1"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fit_tokens(header, body, length):
    if not body or length < 1:
        raise ValueError("Token body and requested length must be nonempty")
    return (header + body * ((length + len(body) - 1) // len(body)))[:length]


def next_prompt(previous_input, actual_output, tool_tokens):
    return previous_input + actual_output + tool_tokens


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def compare_trajectories(cold, warm):
    differences = []
    for before, after in zip(cold["trajectory"], warm["trajectory"], strict=True):
        if before["input_sha256"] != after["input_sha256"] or before["output_sha256"] != after["output_sha256"]:
            differences.append({"lane_id": before["lane_id"], "turn_index": before["turn_index"]})
    return {
        "cold_cell_id": cold["cell_id"],
        "warm_cell_id": warm["cell_id"],
        "all_realized_tokens_identical": not differences,
        "differing_turns": differences,
    }


def idle_gauges(metrics):
    values = {"running": [], "waiting": []}
    for line in metrics.splitlines():
        match = re.match(r"vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([\d.eE+-]+)", line)
        if match:
            values[match[1]].append(float(match[2]))
    if not all(values.values()):
        raise RuntimeError("Server /metrics does not expose both running and waiting request gauges")
    return {name: sum(samples) for name, samples in values.items()}


class StreamRecord:
    """Keep token bursts intact; a transport chunk is not an individual token."""

    def __init__(self, prompt, dispatch_ms):
        self.prompt = prompt
        self.dispatch_ms = dispatch_ms
        self.output = []
        self.chunks = []
        self.usage = None
        self.metrics = None
        self.prompt_verified = False
        self.finish_reason = None
        self.first_token_ms = None
        self.last_token_ms = None
        self.first_visible_ms = None

    def add(self, event, received_ms):
        if "error" in event:
            raise RuntimeError(f"Completion stream error: {event['error']}")
        self.chunks.append({"received_ms": received_ms, "event": event})
        if event.get("usage") is not None:
            self.usage = event["usage"]
        if event.get("metrics") is not None:
            self.metrics = event["metrics"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                raise RuntimeError("Expected exactly one completion choice")
            if choice.get("prompt_token_ids") is not None:
                if choice["prompt_token_ids"] != self.prompt:
                    raise RuntimeError("Server changed the supplied prompt token IDs")
                self.prompt_verified = True
            tokens = choice.get("token_ids") or []
            if any(type(token) is not int or token < 0 for token in tokens):
                raise RuntimeError("Malformed generated token IDs")
            if tokens:
                if self.first_token_ms is None:
                    self.first_token_ms = received_ms
                self.last_token_ms = received_ms
                self.output.extend(tokens)
            if choice.get("text") and self.first_visible_ms is None:
                self.first_visible_ms = received_ms
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]

    def result(self, completed_ms, output_length):
        if not self.prompt_verified or len(self.output) != output_length:
            raise RuntimeError("Missing exact token IDs or output length differs from the fixed requested length")
        if self.finish_reason != "length":
            raise RuntimeError(f"Unexpected finish reason: {self.finish_reason}")
        if not self.usage or self.usage.get("completion_tokens") != len(self.output):
            raise RuntimeError("Missing or inconsistent completion usage")
        if self.usage.get("prompt_tokens") != len(self.prompt):
            raise RuntimeError("Prompt usage differs from the supplied token IDs")
        cached = (self.usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if type(cached) is not int or not 0 <= cached < len(self.prompt):
            raise RuntimeError("Missing/invalid cached_tokens; enable --enable-prompt-tokens-details")
        if not self.metrics:
            raise RuntimeError("Missing server request metrics; enable --enable-per-request-metrics")
        return {
            "output_token_ids": self.output,
            "cached_tokens": cached,
            "usage": self.usage,
            "server_metrics": self.metrics,
            "stream_chunks": self.chunks,
            "stream_chunk_count": len(self.chunks),
            "token_chunk_count": sum(
                any(c.get("token_ids") for c in chunk["event"].get("choices", [])) for chunk in self.chunks
            ),
            "timing": {
                "dispatch_ms": self.dispatch_ms,
                "first_token_ms": self.first_token_ms,
                "first_visible_ms": self.first_visible_ms,
                "last_token_ms": self.last_token_ms,
                "completed_ms": completed_ms,
                "ttft_ms": self.first_token_ms - self.dispatch_ms,
                "tpot_ms": (self.last_token_ms - self.first_token_ms) / (output_length - 1)
                if output_length > 1
                else None,
            },
        }


class Collector:
    def __init__(self, session, base_url, model, output):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.output = output
        self.origin = time.perf_counter()
        self.outstanding = 0

    def now(self):
        return (time.perf_counter() - self.origin) * 1000

    async def json_request(self, method, endpoint, payload=None):
        async with self.session.request(method, self.base_url + endpoint, json=payload) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"{endpoint}: HTTP {response.status}: {body[:1000]}")
            return {"http_status": response.status, "body": json.loads(body), "received_ms": self.now()}

    async def tokenize(self, text):
        reply = await self.json_request(
            "POST", "/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False}
        )
        tokens = reply["body"]["tokens"]
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise RuntimeError("Tokenizer returned invalid token IDs")
        return tokens

    async def metrics(self):
        async with self.session.get(self.base_url + "/metrics") as response:
            response.raise_for_status()
            return await response.text()

    async def drain(self):
        if self.outstanding:
            raise RuntimeError("Attempted barrier with outstanding client requests")
        deadline = time.perf_counter() + 60
        while True:
            metrics = await self.metrics()
            gauges = idle_gauges(metrics)
            if not any(gauges.values()):
                return metrics, {"received_ms": self.now(), "client_outstanding": 0, "server_gauges": gauges}
            if time.perf_counter() >= deadline:
                raise RuntimeError(f"Server did not become idle: {gauges}")
            await asyncio.sleep(0.2)

    async def request(
        self,
        handle,
        cell_id,
        lane,
        turn,
        phase,
        prompt,
        output_length,
        parents=(),
        think_ms=0,
        dependency_completed_ms=None,
    ):
        request_id = f"{cell_id}:{phase}:l{lane}:t{turn}"
        row = {
            "schema": SCHEMA,
            "request_id": request_id,
            "cell_id": cell_id,
            "lane_id": lane,
            "turn_index": turn,
            "phase": phase,
            "parent_request_id": parents[0] if parents else None,
            "dependency_ids": list(parents),
            "think_time_ms": think_ms,
            "input_token_ids": prompt,
            "requested_output_tokens": output_length,
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": output_length,
            "min_tokens": output_length,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "add_special_tokens": False,
            "return_token_ids": True,
            "request_id": request_id,
            "stream": True,
            "stream_interval": 1,
            "stream_options": {"include_usage": True, "continuous_usage_stats": False},
        }
        record = StreamRecord(prompt, self.now())
        self.outstanding += 1
        try:
            async with self.session.post(self.base_url + "/v1/completions", json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"Completion HTTP {response.status}: {(await response.text())[:1000]}")
                done = False
                data = []
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip())
                    elif not line and data:
                        received_ms = self.now()
                        message = "\n".join(data)
                        data = []
                        if message == "[DONE]":
                            done = True
                            break
                        record.add(json.loads(message), received_ms)
                if not done:
                    raise RuntimeError("Completion stream ended without [DONE]")
            row.update(record.result(self.now(), output_length))
            if dependency_completed_ms is not None:
                actual_delay = record.dispatch_ms - dependency_completed_ms
                row["timing"].update(
                    dependency_completed_ms=dependency_completed_ms,
                    realized_dependency_delay_ms=actual_delay,
                    think_time_overshoot_ms=max(0, actual_delay - think_ms),
                )
            row["status"] = "completed"
            return row
        except Exception as exc:
            row.update(status="error", error=str(exc), stream_chunks=record.chunks, output_token_ids=record.output)
            raise
        finally:
            self.outstanding -= 1
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()

    async def batch(self, requests):
        results = await asyncio.gather(*requests, return_exceptions=True)
        errors = [value for value in results if isinstance(value, BaseException)]
        if errors:
            raise RuntimeError(f"{len(errors)} request(s) failed; first failure: {errors[0]}") from errors[0]
        return results

    async def prompts(self, identity, length, lanes):
        body = await self.tokenize(CORPUS)
        prompts = []
        for lane in range(lanes):
            identifier = hashlib.sha256(f"{identity}:lane{lane}".encode()).hexdigest()
            header = await self.tokenize(identifier + "\n" + CORPUS)
            prompts.append(fit_tokens(header, body, length))
        if len({tuple(prompt) for prompt in prompts}) != lanes:
            raise RuntimeError("Lane prompts unexpectedly collide")
        return prompts, fit_tokens([], await self.tokenize("\nTool result: checks completed. " + CORPUS), 512)

    async def case(self, case_id, prompts, tool, cache, smoke=False):
        directory = self.output / case_id
        directory.mkdir()
        info = {
            "schema": "qwen35-agentic-case/v1",
            "cell_id": case_id,
            "initial_input_tokens": len(prompts[0]),
            "lanes": len(prompts),
            "initial_cache": cache,
            "measured_turns_per_lane": 2 if smoke else 8,
            "output_tokens": 1 if smoke else 256,
            "tool_tokens": 0 if smoke else 512,
            "think_time_ms": 0 if smoke else 100,
            "warmup_kind": "identical_prompt" if smoke else "historical_prefix_without_latest_tool",
            "status": "running",
        }
        save_json(directory / "case.json", info)
        try:
            with (directory / "requests.jsonl").open("w") as handle:
                # Kernel work is deliberately outside the cache experiment and reset below.
                await self.batch(
                    [
                        self.request(handle, case_id, lane, -2, "kernel_warmup", prompt, 2)
                        for lane, prompt in enumerate(prompts)
                    ]
                )
                metrics, info["before_reset_barrier"] = await self.drain()
                (directory / "metrics-before-reset.prom").write_text(metrics)
                info["cache_reset"] = await self.json_request("POST", "/reset_prefix_cache")
                save_json(directory / "case.json", info)
                if info["cache_reset"]["body"].get("success") is not True:
                    raise RuntimeError("Server declined the prefix-cache reset")
                metrics, info["after_reset_barrier"] = await self.drain()
                (directory / "metrics-after-reset.prom").write_text(metrics)
                primers = []
                if cache == "warm" or smoke:
                    primers = await self.batch(
                        [
                            self.request(handle, case_id, lane, -1, "warmup", prompt if smoke else prompt[:-512], 1)
                            for lane, prompt in enumerate(prompts)
                        ]
                    )
                    if any(row["cached_tokens"] != 0 for row in primers):
                        raise RuntimeError("Cold preparation reused tokens after successful reset")
                metrics, info["measured_barrier"] = await self.drain()
                (directory / "metrics-before-measured.prom").write_text(metrics)
                primer_ids = [row["request_id"] for row in primers]

                async def run_lane(lane, prompt):
                    parents = primer_ids[lane:] + primer_ids[:lane] if primers else []
                    dependency_completed_ms = max(row["timing"]["completed_ms"] for row in primers) if primers else None
                    measured = []
                    for turn in range(info["measured_turns_per_lane"]):
                        if turn and not smoke:
                            await asyncio.sleep(max(0, (dependency_completed_ms + 100 - self.now()) / 1000))
                        row = await self.request(
                            handle,
                            case_id,
                            lane,
                            turn,
                            "measured",
                            prompt,
                            info["output_tokens"],
                            parents,
                            think_ms=100 if turn and not smoke else 0,
                            dependency_completed_ms=dependency_completed_ms,
                        )
                        measured.append(row)
                        if turn == 0 and cache == "cold" and not smoke and row["cached_tokens"] != 0:
                            raise RuntimeError("Cold measured frontier reused tokens after successful reset")
                        if not smoke:
                            prompt = next_prompt(prompt, row["output_token_ids"], tool)
                        parents = [row["request_id"]]
                        dependency_completed_ms = row["timing"]["completed_ms"]
                    return measured

                lanes = await self.batch([run_lane(lane, prompt) for lane, prompt in enumerate(prompts)])
                rows = [row for lane in lanes for row in lane]
                metrics, info["completed_barrier"] = await self.drain()
                (directory / "metrics-after-measured.prom").write_text(metrics)
                info["measured_requests"] = len(rows)
                info["measured_duration_ms"] = max(row["timing"]["completed_ms"] for row in rows) - min(
                    row["timing"]["dispatch_ms"] for row in rows
                )
                info["cached_tokens_by_lane"] = [[row["cached_tokens"] for row in lane] for lane in lanes]
                info["trajectory"] = [
                    {
                        "lane_id": row["lane_id"],
                        "turn_index": row["turn_index"],
                        "input_sha256": token_hash(row["input_token_ids"]),
                        "output_sha256": token_hash(row["output_token_ids"]),
                    }
                    for row in rows
                ]
                info["status"] = "complete"
        except Exception as exc:
            info.update(status="failed", error=str(exc))
            raise
        finally:
            save_json(directory / "case.json", info)
        print(f"Completed {case_id}: {info['measured_requests']} measured requests", flush=True)
        return info


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "qwen35-agentic-collection/v1",
        "started_utc": datetime.now(UTC).isoformat(),
        "command": sys.argv,
        "mode": "smoke" if args.smoke else "matrix",
        "base_url": args.base_url,
        "model": args.model,
        "repeats": args.repeats,
        "clock": "milliseconds since this collector process started its HTTP session; monotonic",
        "cases": [],
        "cold_warm_comparisons": [],
        "status": "running",
    }
    save_json(args.output / "collector.json", manifest)
    try:
        if args.cache_receipt:
            receipt = json.loads(args.cache_receipt.read_text())
            save_json(args.output / "cache-geometry.json", receipt)
            manifest["cache_geometry_id"] = receipt["geometry_id"]
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=3600), read_bufsize=4 * 1024 * 1024
        ) as session:
            collector = Collector(session, args.base_url, args.model, args.output)
            if args.smoke:
                block_size = receipt["manual_kv_cache"]["block_size"]
                for length in [block_size - 1, block_size, block_size + 1, 2 * block_size]:
                    case_id = f"smoke-p{length}"
                    prompts, tool = await collector.prompts(case_id, length, 1)
                    manifest["cases"].append(await collector.case(case_id, prompts, tool, "warm", smoke=True))
                    save_json(args.output / "collector.json", manifest)
            else:
                for length in (4096, 32768):
                    for lanes in (1, 4, 16):
                        for repeat in range(args.repeats):
                            identity = f"p{length}-l{lanes}-r{repeat}"
                            prompts, tool = await collector.prompts(identity, length, lanes)
                            for cache in ("cold", "warm"):
                                case_id = f"p{length}-l{lanes}-{cache}-r{repeat}"
                                manifest["cases"].append(await collector.case(case_id, prompts, tool, cache))
                                if cache == "warm":
                                    manifest["cold_warm_comparisons"].append(
                                        compare_trajectories(manifest["cases"][-2], manifest["cases"][-1])
                                    )
                                save_json(args.output / "collector.json", manifest)
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        save_json(args.output / "collector.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke", action="store_true", help="Qualify reuse at B-1, B, B+1 and 2B; requires cache receipt"
    )
    mode.add_argument("--matrix", action="store_true", help="Collect the 12 operating points")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing paths are never overwritten"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8800")
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache-receipt", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or (args.smoke and args.cache_receipt is None):
        parser.error("--repeats must be positive; --smoke requires --cache-receipt")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
