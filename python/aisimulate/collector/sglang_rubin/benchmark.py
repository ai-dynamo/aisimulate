# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and measure streaming completions for the Rubin serving pilot.

Run from ``python/aisimulate`` with ``python -m collector.sglang_rubin.benchmark``.
The API base includes ``/v1``; this client uses its ``/completions`` endpoint.
Token counts come exclusively from final streamed usage. TTFT measures the first
nonempty text chunk; TPOT measures first-to-last text arrival divided by the
reported completion count minus one. Chunk coalescing limits that measurement.

The client does not change server cache settings. Configure caching in the
serving launcher and record its configuration with ``--serving-config``.
Repeated prompts can reuse the prefix cache, including across concurrency waves.
For controlled prefixes, ``--prompts-file`` accepts a JSON list with one prompt
per measured request, consumed in order without cycling. Warmup uses its own
prompt and is excluded from measured waves.
"""

import argparse
import hashlib
import http.client
import json
import math
import statistics
import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

_MAX_EVENT_BYTES = 1024 * 1024


class StreamError(ValueError):
    """The endpoint did not produce a complete, valid completion stream."""


def _events(chunks: Iterable[bytes]) -> Iterator[str]:
    """Decode UTF-8 SSE data events, including split lines and multiline data."""
    pending = b""
    data = []
    event_bytes = 0
    for chunk in chunks:
        pending += chunk
        while b"\n" in pending:
            raw_line, pending = pending.split(b"\n", 1)
            line = raw_line.removesuffix(b"\r").decode("utf-8")
            event_bytes += len(raw_line)
            if event_bytes > _MAX_EVENT_BYTES:
                raise StreamError("SSE event exceeds size limit")
            if not line:
                if data:
                    yield "\n".join(data)
                data = []
                event_bytes = 0
            elif line.startswith("data:"):
                data.append(line[5:].removeprefix(" "))
            elif line == "data":
                data.append("")
        if len(pending) + event_bytes > _MAX_EVENT_BYTES:
            raise StreamError("SSE event exceeds size limit")
    if pending or data:
        raise StreamError("SSE stream ended inside an event")


def _token_count(usage: dict, name: str) -> int:
    value = usage.get(name)
    if type(value) is not int or value <= 0:
        raise StreamError(f"Final usage must report a positive integer {name}")
    return value


def _reject_constant(value: str):
    raise StreamError(f"Non-JSON numeric constant: {value}")


def _consume(
    events: Iterable[str], started: float, expected_prompt_tokens: int | None, expected_output_tokens: int | None
):
    text_parts = []
    first_text_at = last_text_at = None
    finish_reason = None
    usage = None
    for event in events:
        received = time.perf_counter()
        if event == "[DONE]":
            if not text_parts or not "".join(text_parts).strip():
                raise StreamError("Completion contains no nonempty generated text")
            if finish_reason is None:
                raise StreamError("Stream terminated without a finish reason")
            if usage is None:
                raise StreamError("Stream terminated without final usage")
            prompt_tokens = _token_count(usage, "prompt_tokens")
            completion_tokens = _token_count(usage, "completion_tokens")
            if "total_tokens" in usage and (
                type(usage["total_tokens"]) is not int or usage["total_tokens"] != prompt_tokens + completion_tokens
            ):
                raise StreamError("Final usage total_tokens does not match prompt_tokens + completion_tokens")
            if expected_prompt_tokens is not None and prompt_tokens != expected_prompt_tokens:
                raise StreamError(f"Expected {expected_prompt_tokens} prompt tokens, observed {prompt_tokens}")
            if expected_output_tokens is not None and completion_tokens != expected_output_tokens:
                raise StreamError(f"Expected {expected_output_tokens} completion tokens, observed {completion_tokens}")
            latency = received - started
            return {
                "generated_text": "".join(text_parts),
                "text_chunks": len(text_parts),
                "finish_reason": finish_reason,
                "usage": usage,
                "ttft_s": first_text_at - started,
                "latency_s": latency,
                "tpot_s": (last_text_at - first_text_at) / (completion_tokens - 1) if completion_tokens > 1 else None,
                "output_tokens_per_s": completion_tokens / latency,
            }
        value = json.loads(event, parse_constant=_reject_constant)
        if not isinstance(value, dict):
            raise StreamError("SSE completion must be a JSON object")
        if value.get("error") is not None:
            raise StreamError(f"Server error: {value['error']}")
        if usage is not None:
            raise StreamError("Received another completion event after final usage")
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamError("Expected at most one completion choice")
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0:
                raise StreamError("Expected completion choice index 0")
            text = choice.get("text")
            if not isinstance(text, str):
                raise StreamError("Completion choice text must be a string")
            if finish_reason is not None:
                raise StreamError("Received another choice after its finish reason")
            if text:
                text_parts.append(text)
                first_text_at = received if first_text_at is None else first_text_at
                last_text_at = received
            reason = choice.get("finish_reason")
            if reason is not None:
                if reason not in ("stop", "length"):
                    raise StreamError(f"Unsuccessful or unsupported finish reason: {reason}")
                finish_reason = reason
        if value.get("usage") is not None:
            if finish_reason is None or not isinstance(value["usage"], dict):
                raise StreamError("Final usage must be an object after a completed choice")
            usage = value["usage"]
    raise StreamError("SSE stream ended without [DONE]")


def _remaining(deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("Request deadline exceeded")
    return remaining


def request_completion(
    *,
    api_base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    expected_prompt_tokens: int | None = None,
    require_output_length: bool = False,
    ignore_eos: bool = False,
) -> dict:
    """Return either validated timings and usage or an explicit failure record."""
    started = time.perf_counter()
    deadline = started + timeout
    connection = None
    result = {"ok": False, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
    try:
        url = urlsplit(api_base_url.rstrip("/") + "/completions")
        connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(url.hostname, port=url.port, timeout=_remaining(deadline))
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "n": 1,
                # SGLang 02c5a855aceb968c310e6fbc6632270e26edc84b maps this
                # request option in srt/entrypoints/openai/serving_completions.py:163.
                "ignore_eos": ignore_eos,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode("utf-8")
        connection.connect()
        transport = connection.sock
        transport.settimeout(_remaining(deadline))
        connection.request("POST", url.path, body, {"Content-Type": "application/json", "Accept": "text/event-stream"})
        transport.settimeout(_remaining(deadline))
        with connection.getresponse() as response:
            if response.status != 200:
                raise StreamError(f"HTTP {response.status}: {response.reason}")
            if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                raise StreamError("Response Content-Type must be text/event-stream")

            def chunks():
                while True:
                    transport.settimeout(_remaining(deadline))
                    chunk = response.read1(65536)
                    if not chunk:
                        return
                    _remaining(deadline)
                    yield chunk

            result.update(
                _consume(
                    _events(chunks()), started, expected_prompt_tokens, max_tokens if require_output_length else None
                )
            )
        result["ok"] = True
    except (OSError, ValueError, http.client.HTTPException) as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), latency_s=time.perf_counter() - started)
    finally:
        if connection is not None:
            connection.close()
    return result


def benchmark_wave(concurrency: int, prompts: list[str], request_options: dict) -> dict:
    """Measure a closed-loop wave; failed requests invalidate its throughput."""
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(request_completion, prompt=prompt, **request_options) for prompt in prompts]
        results = [dict(request_index=index, **future.result()) for index, future in enumerate(futures)]
    wall = time.perf_counter() - started
    succeeded = [result for result in results if result["ok"]]
    valid = len(succeeded) == len(results)
    tpots = [result["tpot_s"] for result in succeeded if result["tpot_s"] is not None]
    return {
        "concurrency": concurrency,
        "valid": valid,
        "requests_total": len(results),
        "requests_succeeded": len(succeeded),
        "requests_failed": len(results) - len(succeeded),
        "wall_s": wall,
        "output_tokens_per_s": sum(result["usage"]["completion_tokens"] for result in succeeded) / wall
        if valid
        else None,
        "mean_ttft_s": statistics.mean(result["ttft_s"] for result in succeeded) if valid else None,
        "mean_latency_s": statistics.mean(result["latency_s"] for result in succeeded) if valid else None,
        "mean_tpot_s": statistics.mean(tpots) if valid and tpots else None,
        "requests": results,
    }


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parse_args(argv: list[str] | None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-base-url", required=True, help="OpenAI API base, for example http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True, help="Served model name")
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="One measured prompt, reused across requests")
    prompts.add_argument("--prompt-file", type=Path, help="UTF-8 text containing one measured prompt")
    prompts.add_argument("--prompts-file", type=Path, help="JSON list with exactly one prompt per measured request")
    parser.add_argument("--output-tokens", type=_positive_int, default=500, help="Requested max_tokens (default: 500)")
    parser.add_argument(
        "--require-output-length", action="store_true", help="Fail if actual completion_tokens differs from max_tokens"
    )
    parser.add_argument("--ignore-eos", action="store_true", help="Ask SGLang to ignore EOS for fixed-length workloads")
    parser.add_argument(
        "--expected-prompt-tokens", type=_positive_int, help="Require this exact server-reported input length"
    )
    parser.add_argument(
        "--concurrencies", default="1,8,32", help="Comma-separated concurrency levels (default: 1,8,32)"
    )
    parser.add_argument(
        "--requests-per-worker", type=_positive_int, default=2, help="Requests per concurrency slot (default: 2)"
    )
    parser.add_argument("--request-timeout", type=float, default=900, help="Request deadline in seconds (default: 900)")
    parser.add_argument(
        "--warmup-requests", type=_positive_int, default=1, help="Separate sequential warmup requests (default: 1)"
    )
    parser.add_argument("--warmup-prompt", default="Explain why reproducible measurements matter.")
    parser.add_argument(
        "--serving-config", type=Path, help="JSON object recording server launch options, including cache settings"
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON result artifact")
    args = parser.parse_args(argv)
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        parser.error("--request-timeout must be finite and positive")
    try:
        url = urlsplit(args.api_base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.query
            or url.fragment
            or url.username
            or url.password
        ):
            raise ValueError("--api-base-url must be an HTTP(S) URL without credentials, query, or fragment")
        if url.port == 0:
            raise ValueError("--api-base-url port must be positive")
        args.concurrencies = [_positive_int(value.strip()) for value in args.concurrencies.split(",")]
        if len(set(args.concurrencies)) != len(args.concurrencies):
            raise ValueError("duplicate concurrency levels")
        if not args.model.strip() or not args.warmup_prompt.strip():
            raise ValueError("model and warmup prompt must not be empty")
        count = sum(args.concurrencies) * args.requests_per_worker
        if args.prompts_file:
            args.prompts = json.loads(args.prompts_file.read_text(encoding="utf-8"))
            if not isinstance(args.prompts, list) or len(args.prompts) != count:
                raise ValueError(f"--prompts-file must contain exactly {count} prompts")
        else:
            prompt = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
            args.prompts = [prompt] * count
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in args.prompts):
            raise ValueError("each measured prompt must be a nonempty string")
        args.server_configuration = (
            json.loads(args.serving_config.read_text(encoding="utf-8"), parse_constant=_reject_constant)
            if args.serving_config
            else None
        )
        if args.serving_config and not isinstance(args.server_configuration, dict):
            raise ValueError("--serving-config must contain a JSON object")
    except (OSError, ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    options = {
        "api_base_url": args.api_base_url,
        "model": args.model,
        "max_tokens": args.output_tokens,
        "timeout": args.request_timeout,
        "expected_prompt_tokens": args.expected_prompt_tokens,
        "require_output_length": args.require_output_length,
        "ignore_eos": args.ignore_eos,
    }
    warmup_options = {
        **options,
        "max_tokens": min(32, args.output_tokens),
        "expected_prompt_tokens": None,
        "require_output_length": False,
    }
    warmup = [request_completion(prompt=args.warmup_prompt, **warmup_options) for _ in range(args.warmup_requests)]
    waves = []
    offset = 0
    if all(result["ok"] for result in warmup):
        for concurrency in args.concurrencies:
            count = concurrency * args.requests_per_worker
            wave = benchmark_wave(concurrency, args.prompts[offset : offset + count], options)
            waves.append(wave)
            offset += count
            print(json.dumps({key: value for key, value in wave.items() if key != "requests"}), flush=True)
    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "valid": bool(waves) and all(wave["valid"] for wave in waves),
        "configuration": {
            **options,
            "concurrencies": args.concurrencies,
            "requests_per_worker": args.requests_per_worker,
            "measured_requests_expected": len(args.prompts),
            "prompt_source": "prompts-file" if args.prompts_file else "prompt-file" if args.prompt_file else "prompt",
            "warmup_requests": args.warmup_requests,
            "warmup_max_tokens": warmup_options["max_tokens"],
            "server_configuration": args.server_configuration,
            "cache_note": "Caller must configure server caching; this client does not disable or clear caches.",
            "tpot_definition": "(last text arrival - first text arrival) / (completion_tokens - 1); null for one token",
        },
        "warmup": warmup,
        "waves": waves,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0 if artifact["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
