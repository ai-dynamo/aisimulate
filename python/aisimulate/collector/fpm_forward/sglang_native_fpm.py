# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect SGLang native ForwardPassMetrics (FPM) as a Dynamo-format rank artifact.

Integration: sgl-project/sglang@94602c9c2b7cbdb8efd5c52802dac6a1c180089e
(v0.5.20), Apache-2.0, through public interfaces only:

* ``python -m sglang.launch_server --enable-forward-pass-metrics`` (stock CLI;
  observability field ``enable_forward_pass_metrics``);
* the per-iteration ``ForwardPassMetrics`` ZMQ PUB stream bound by
  ``scheduler_components/metrics_reporter.py`` ``_init_fpm`` on the
  attention-TP rank 0 at ``{forward_pass_metrics_ipc_name}.{dp_rank}``. Its
  ``wall_time`` is the native ``DeviceTimer`` forward interval of that rank
  (``_emit_forward_pass_metrics``), the same schema Dynamo's SGLang publisher
  consumes (``dynamo.common.forward_pass_metrics``);
* ``/generate`` with token-id batches (one batched call = one native
  ``BatchTokenizedGenerateReqInput``).

The server runs its default serving configuration (radix cache on, overlap
scheduler on). This client never touches scheduler state: each geometry is
built from real requests (a warm request per prefix, then B requests that hit
the cached prefix), exactly as the stock-server ground truth. The measured
latency is the native FPM ``wall_time`` of the measured forward. The client
does no work while a round is in flight; FPM messages are read and validated
after the round's response returns. Code here is original; no SGLang source is
copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WARMUP_ROUNDS = 5
MEASURED_ROUNDS = 10
MAX_REJECTED_ROUNDS = 10
DECODE_LEAD = 3  # prompt = context - 3; measured = 4th decode step (full overlap pipeline)
MEASURED_DECODE_STEP = DECODE_LEAD + 1
HIT_UNIT = 64  # native radix page for this hybrid DSA/KDA model (mamba extra_buffer)
STATE_PROTOCOL = "glm53flash_sglang_radix_real_seed_v1"
TIMING_BOUNDARY = "sglang_native_fpm_rank0_device_timer_wall_time"
MESSAGE_WAIT_SECONDS = 10.0
QUIET_SECONDS = 0.5


def log(message: str) -> None:
    print(f"[sglang-native-fpm {time.strftime('%H:%M:%S')}] {message}", flush=True)


def rotation(pool: list[int], offset: int, length: int) -> list[int]:
    size = len(pool)
    return [pool[(offset + index) % size] for index in range(length)]


def http(url: str, payload=None, timeout: float = 3600.0):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body.decode(errors="replace")


def server_command(args) -> list[str]:
    command = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--tp-size", str(args.tp),
        "--pipeline-parallel-size", "1",
        "--data-parallel-size", "1",
        "--expert-parallel-size", "1",
        "--kv-cache-dtype", "fp8_e4m3",
        "--moe-runner-backend", "auto",
        "--context-length", "131079",
        "--chunked-prefill-size", "8192",
        "--mem-fraction-static", str(args.mem_fraction),
        "--max-running-requests", "32",
        "--cuda-graph-bs-decode", *[str(size) for size in range(1, 33)],
        "--cuda-graph-backend-prefill", "breakable",
        "--cuda-graph-max-bs-prefill", "8192",
        "--enable-forward-pass-metrics",
        "--forward-pass-metrics-ipc-name", args.ipc,
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]  # fmt: skip
    if args.quant == "nvfp4":
        command += ["--quantization", "modelopt_fp4"]
    return command


class FpmStream:
    """Rank-0 native FPM subscriber; read only between rounds."""

    def __init__(self, endpoint: str):
        import msgspec
        import zmq
        from sglang.srt.observability.forward_pass_metrics import decode

        self._zmq = zmq
        self._msgspec = msgspec
        self._decode = decode
        self._context = zmq.Context()
        self.socket = self._context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect(endpoint)
        self.last_sequence = None
        self.gaps = []

    def drain(self) -> list[dict]:
        out = []
        while True:
            try:
                parts = self.socket.recv_multipart(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                return out
            _topic, sequence, payload = parts
            number = int.from_bytes(sequence, "big")
            if self.last_sequence is not None and number != self.last_sequence + 1:
                self.gaps.append([self.last_sequence, number])
            self.last_sequence = number
            message = self._msgspec.to_builtins(self._decode(payload))
            message["sequence"] = number
            if message["wall_time"] == 0.0 and not any(message["scheduled_requests"].values()):
                continue  # native idle heartbeat
            out.append(message)

    def collect(self, done, timeout: float = MESSAGE_WAIT_SECONDS, quiet: float = QUIET_SECONDS) -> list[dict]:
        """Read until ``done`` and then until the stream is quiet.

        The native publisher sends asynchronously, so trailing steps of a round
        (e.g. the overlap scheduler's last step) can arrive after the response;
        waiting for quiet keeps every message with the round that produced it.
        """
        messages, deadline = [], time.monotonic() + timeout
        last = time.monotonic()
        while True:
            fresh = self.drain()
            now = time.monotonic()
            if fresh:
                messages += fresh
                last = now
            if (done(messages) and now - last >= quiet) or now >= deadline:
                return messages
            time.sleep(0.02)


def plan_rounds(point: dict, pool: list[int]) -> dict:
    """Same request construction as the stock-server ground truth."""
    bid, size = point["benchmark_id"], len(pool)
    count = WARMUP_ROUNDS + MEASURED_ROUNDS + MAX_REJECTED_ROUNDS
    if point["phase"] == "prefill":
        prefixes, news = point["prefix_tokens"], point["new_tokens"]
        offsets = [(131 * index + 17 * bid) % size for index in range(len(prefixes))]
        warm = [rotation(pool, o, k) if k else None for o, k in zip(offsets, prefixes, strict=True)]
        rounds, specs = [], []
        used = [set() for _ in prefixes]
        for r in range(count):
            prompts, spec = [], []
            for index, (o, k, t) in enumerate(zip(offsets, prefixes, news, strict=True)):
                if k:
                    shift = 1 + 97 * r
                    while pool[(o + k + shift) % size] in used[index]:
                        shift += 1
                    used[index].add(pool[(o + k + shift) % size])
                    prompts.append(warm[index] + rotation(pool, o + k + shift, t))
                    spec.append([[o, k], [o + k + shift, t]])
                else:
                    prompts.append(rotation(pool, o + 7 * r + 1, t))
                    spec.append([[o + 7 * r + 1, t]])
            rounds.append(prompts)
            specs.append(spec)
        return {
            "offsets": offsets,
            "warm": warm,
            "warm_specs": [[[o, k]] if k else None for o, k in zip(offsets, prefixes, strict=True)],
            "rounds": rounds,
            "round_specs": specs,
            "expected_cached": list(prefixes),
            "max_tokens": 1,
        }
    contexts = point["contexts"]
    offsets = [(131 * index + 17 * bid) % size for index in range(len(contexts))]
    prompts = [rotation(pool, o, c - DECODE_LEAD) for o, c in zip(offsets, contexts, strict=True)]
    hits = [((c - DECODE_LEAD - 1) // HIT_UNIT) * HIT_UNIT for c in contexts]
    return {
        "offsets": offsets,
        "warm": [p[:h] if h else None for p, h in zip(prompts, hits, strict=True)],
        "warm_specs": [[[o, h]] if h else None for o, h in zip(offsets, hits, strict=True)],
        "rounds": [prompts] * count,
        "round_specs": [[[[o, c - DECODE_LEAD]] for o, c in zip(offsets, contexts, strict=True)]] * count,
        "expected_cached": hits,
        "max_tokens": DECODE_LEAD + 3,
    }


def flush_cache(base: str) -> None:
    for _ in range(60):
        try:
            http(base + "/flush_cache", {}, timeout=120)
            return
        except Exception:  # native refuses while requests are still being released
            time.sleep(0.5)
    raise RuntimeError("native /flush_cache did not succeed")


def generate(base: str, prompts: list[list[int]], max_tokens: int) -> list[dict]:
    out = http(
        base + "/generate",
        {
            "input_ids": prompts,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "min_new_tokens": max_tokens,
                "temperature": 0.0,
                "ignore_eos": True,
            },
        },
    )
    out = out if isinstance(out, list) else [out]
    if len(out) != len(prompts):
        raise RuntimeError("batched /generate returned a different request count")
    return [
        {
            "prompt_tokens": item["meta_info"]["prompt_tokens"],
            "completion_tokens": item["meta_info"]["completion_tokens"],
            "cached_tokens": item["meta_info"].get("cached_tokens"),
            "output_ids": item.get("output_ids"),
        }
        for item in out
    ]


def select_measured(point: dict, messages: list[dict]) -> tuple[dict | None, str | None]:
    batch = point["batch_size"]
    if point["phase"] == "prefill":
        prefills = [m for m in messages if m["scheduled_requests"]["num_prefill_requests"] > 0]
        if len(prefills) != 1:
            return None, f"{len(prefills)} prefill forwards, expected one batched forward"
        measured = prefills[0]
        scheduled = measured["scheduled_requests"]
        if (
            scheduled["num_prefill_requests"] != batch
            or scheduled["sum_prefill_tokens"] != point["total_prefill_tokens"]
            or scheduled["sum_prefill_kv_tokens"] != point["total_kv_read_tokens"]
            or scheduled["num_decode_requests"] != 0
        ):
            return None, f"measured prefill geometry differs: {scheduled}"
        return measured, None
    decodes = [
        m
        for m in messages
        if m["scheduled_requests"]["num_decode_requests"] > 0 and m["scheduled_requests"]["num_prefill_requests"] == 0
    ]
    kv, step = point["total_kv_read_tokens"], MEASURED_DECODE_STEP
    # Native decode metrics sum seq_lens (prior tokens + the decoded token):
    # decode step j of this round has sum K - (step - j) * B + B.
    expected = [kv + batch - (step - j) * batch for j in range(1, step + 1)]
    sums = [m["scheduled_requests"]["sum_decode_kv_tokens"] for m in decodes]
    starts = [i for i in range(len(decodes) - step + 1) if sums[i : i + step] == expected]
    run = decodes[starts[0] : starts[0] + step] if len(starts) == 1 else []
    if not run or any(m["scheduled_requests"]["num_decode_requests"] != batch for m in run):
        return None, f"no unique all-B lockstep decode run {expected} in {sums}"
    measured = decodes[starts[0] + step - 1]
    return measured, None


def run_point(args, base: str, stream: FpmStream, point: dict, pool: list[int]) -> dict:
    plan = plan_rounds(point, pool)
    record = {
        "benchmark_id": point["benchmark_id"],
        "phase": point["phase"],
        "per_request": {key: point[key] for key in ("prefix_tokens", "new_tokens", "contexts") if key in point},
        "warm_specs": plan["warm_specs"],
        "expected_cached": plan["expected_cached"],
        "rounds": [],
        "rejected_rounds": [],
    }
    started = time.monotonic()
    # As the ground truth: an empty radix cache per point (and per round when no
    # prefix is planned), so only the planned prefix can be hit. Untimed.
    flush_each_round = point["phase"] == "prefill" and not any(point["prefix_tokens"])
    flush_cache(base)
    warm = [prompt for prompt in plan["warm"] if prompt]
    if warm:
        stream.drain()
        result = generate(base, warm, 1)
        record["warm"] = {"requests": len(warm), "cached_tokens": [item["cached_tokens"] for item in result]}
        time.sleep(0.2)
        record["warm"]["fpm_messages"] = len(stream.drain())
    accepted = 0
    for attempt, prompts in enumerate(plan["rounds"]):
        if accepted == WARMUP_ROUNDS + MEASURED_ROUNDS:
            break
        if flush_each_round and attempt:
            flush_cache(base)
        stray = stream.drain()
        responses = generate(base, prompts, plan["max_tokens"])
        expected = 1 if point["phase"] == "prefill" else MEASURED_DECODE_STEP + 1

        def done(messages, expected=expected):
            return sum(1 for m in messages if any(m["scheduled_requests"].values())) >= expected

        messages = stream.collect(done)
        round_record = {
            "attempt": attempt,
            "prompt_specs": plan["round_specs"][attempt],
            "responses": [
                {key: item[key] for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "output_ids")}
                for item in responses
            ],
            "fpm_messages": messages,
            "stray_messages_before": len(stray),
        }
        measured, reason = select_measured(point, messages)
        cached = [item["cached_tokens"] for item in responses]
        if reason is None and cached != plan["expected_cached"]:
            reason = f"API cached_tokens {cached} differ from plan {plan['expected_cached']}"
        if reason is None and any(item["completion_tokens"] != plan["max_tokens"] for item in responses):
            reason = "completion length differs from the frozen request"
        if reason is not None:
            round_record["rejection_reason"] = reason
            record["rejected_rounds"].append(round_record)
            log(f"  {point['phase']}-{point['benchmark_id']} attempt {attempt} rejected: {reason}")
            if len(record["rejected_rounds"]) > MAX_REJECTED_ROUNDS:
                raise RuntimeError(f"too many rejected rounds at benchmark_id={point['benchmark_id']}: {reason}")
            continue
        round_record["repetition"] = accepted
        round_record["role"] = "warmup" if accepted < WARMUP_ROUNDS else "measurement"
        round_record["measured_sequence"] = measured["sequence"]
        round_record["wall_time"] = measured["wall_time"]
        round_record["scheduled_requests"] = measured["scheduled_requests"]
        record["rounds"].append(round_record)
        accepted += 1
    if accepted != WARMUP_ROUNDS + MEASURED_ROUNDS:
        raise RuntimeError(f"benchmark_id={point['benchmark_id']} did not reach 5+10 accepted rounds")
    record["wall_seconds"] = time.monotonic() - started
    return record


def expected_scheduled(point: dict) -> dict:
    if point["phase"] == "prefill":
        return {
            "num_prefill_requests": point["batch_size"],
            "sum_prefill_tokens": point["total_prefill_tokens"],
            "sum_prefill_kv_tokens": point["total_kv_read_tokens"],
            "num_decode_requests": 0,
            "sum_decode_kv_tokens": 0,
        }
    return {
        "num_prefill_requests": 0,
        "sum_prefill_tokens": 0,
        "sum_prefill_kv_tokens": 0,
        "num_decode_requests": point["batch_size"],
        "sum_decode_kv_tokens": point["total_kv_read_tokens"],
    }


def artifact(args, points, records, provenance, started_at, elapsed) -> dict:
    results, groups = [], []
    for point, record, digest in records:
        measured = [r["wall_time"] for r in record["rounds"] if r["role"] == "measurement"]
        native = record["rounds"][-1]["scheduled_requests"]
        fpm = {
            "version": 1,
            "worker_id": provenance["worker_id"],
            "dp_rank": 0,
            "counter_id": point["benchmark_id"],
            "wall_time": statistics.median(measured),
            # Dynamo coordinate convention (decode KV = prior tokens); the
            # native message (seq_lens incl. the decoded token) is retained
            # verbatim in the repetition evidence.
            "scheduled_requests": expected_scheduled(point),
            "native_scheduled_requests": native,
        }
        dynamo_point = {
            "point_type": point["phase"],
            "benchmark_id": point["benchmark_id"],
            "batch_size": point["batch_size"],
            "total_prefill_tokens": point.get("total_prefill_tokens", 0),
            "total_kv_read_tokens": point["total_kv_read_tokens"],
            "rows": (
                [[t, k] for t, k in zip(point["new_tokens"], point["prefix_tokens"], strict=True)]
                if point["phase"] == "prefill"
                else None
            ),
            "sample_reasons": [
                "explicit",
                "kvwarm_real_kv"
                if point["phase"] == "decode"
                else "prefill_real_seed"
                if point["total_kv_read_tokens"]
                else "explicit_kv0",
            ],
        }
        regime = (
            "real_kv"
            if point["phase"] == "decode"
            else "real_prefix"
            if point["total_kv_read_tokens"]
            else "not_applicable"
        )
        results.append(
            {"point": dynamo_point, "fpms": [fpm], "kv_seed_regime": regime, "repetition_evidence_sha256": digest}
        )
        groups.append(
            {
                "benchmark_id": point["benchmark_id"],
                "point": dynamo_point,
                "expected_dp_ranks": [0],
                "complete": True,
                "wall_time": fpm["wall_time"],
                "rank_results": [{"dp_rank": 0, "fpms": [fpm]}],
                "kv_seed_regime": regime,
            }
        )
    measured_seconds = sum(group["wall_time"] for group in groups)
    return {
        "schema_version": 2,
        "artifact_type": "rank",
        "status": "complete",
        "valid": len(results) == len(points),
        "usable": len(results) == len(points),
        "stop_reason": None,
        "timing_valid": measured_seconds <= elapsed,
        "run_id": provenance["run_id"],
        "grid_digest": hashlib.sha256(json.dumps(points, sort_keys=True).encode()).hexdigest(),
        "timing": {
            "started_at": started_at,
            "benchmark_elapsed_seconds": elapsed,
            "measured_iteration_seconds": measured_seconds,
        },
        "dp": {"rank": 0, "size": 1},
        "coverage": {"expected_points": len(points), "completed_points": len(results), "skipped_points": 0},
        "config": {"mode": args.phase},
        "measurement_policy": {
            "decode": f"native_fpm_decode_step_{MEASURED_DECODE_STEP}_median_of_{MEASURED_ROUNDS}",
            "prefill": f"native_fpm_single_forward_median_of_{MEASURED_ROUNDS}",
        },
        "results": results,
        "iteration_groups": groups,
        "skipped_points": [],
        "missing_phases": [],
        "error": None,
        "timing_boundary": TIMING_BOUNDARY,
        "kvwarm": {
            "enabled": True,
            "warm_eligible": True,
            "skip_reason": None,
            "method": "native_radix_cache_real_seed",
            "state_protocol": STATE_PROTOCOL,
        },
        "execution_mode": "native_graph_policy",
        "observation_purpose": "fpm",
        "ops_instrumented": False,
        **provenance["artifact"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tp", type=int, choices=(2, 4), required=True)
    parser.add_argument("--quant", choices=("fp8", "nvfp4"), required=True)
    parser.add_argument("--mem-fraction", type=float, required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--select", default="", help="comma-separated benchmark ids (smoke)")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--ipc", default="")
    parser.add_argument("--run-id", default=os.environ.get("SLURM_JOB_ID", "local"))
    args = parser.parse_args(argv)
    args.ipc = args.ipc or f"ipc:///tmp/glm53-sglang-fpm-{args.run_id}"
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    manifest = json.loads(args.points.read_text())
    points = [point for point in manifest["points"] if point["phase"] == args.phase and point.get("collect", True)]
    if args.select:
        wanted = {int(value) for value in args.select.split(",")}
        points = [point for point in points if point["benchmark_id"] in wanted]
    if not points:
        raise SystemExit("no points selected")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = args.corpus.read_bytes()
    pool = list(tokenizer.encode(text.decode("utf-8"), add_special_tokens=False))
    command = server_command(args)
    (args.out / "server-command.json").write_text(json.dumps(command, indent=1))
    log(" ".join(command))
    server = subprocess.Popen(
        command,
        stdout=(args.out / "server.stdout.log").open("ab"),
        stderr=(args.out / "server.stderr.log").open("ab"),
        start_new_session=True,
    )
    base = f"http://127.0.0.1:{args.port}"
    evidence = args.out / "benchmark.repetitions.jsonl"
    records = []
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    try:
        deadline = time.monotonic() + 3600
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"SGLang server exited early rc={server.returncode}")
            try:
                http(base + "/health", timeout=5)
                http(base + "/health_generate", timeout=120)
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(5)
        info = http(base + "/server_info", timeout=60)
        (args.out / "server-info.json").write_text(json.dumps(info, indent=1, default=str))
        server_args = info.get("server_args", info) if isinstance(info, dict) else {}
        checks = {
            "enable_forward_pass_metrics": server_args.get("enable_forward_pass_metrics") is True,
            "radix_cache_enabled": server_args.get("disable_radix_cache") is False,
            "overlap_scheduler_enabled": server_args.get("disable_overlap_schedule") is False,
            "mem_fraction_static": server_args.get("mem_fraction_static") == args.mem_fraction,
            "tp_size": server_args.get("tp_size") == args.tp,
        }
        (args.out / "startup-config-check.json").write_text(json.dumps(checks, indent=1))
        if not all(checks.values()):
            raise RuntimeError(f"stock server configuration differs from the frozen identity: {checks}")
        stream = FpmStream(f"{args.ipc}.0")
        time.sleep(1.0)
        http(base + "/health_generate", timeout=120)
        probe = stream.collect(lambda messages: bool(messages), timeout=30)
        if not probe:
            raise RuntimeError("no native FPM message received from attention-TP rank 0")
        import importlib.metadata

        provenance = {
            "run_id": args.run_id,
            "worker_id": str(probe[-1].get("worker_id", "")),
            "artifact": {
                "producer": {
                    "backend": "sglang",
                    "backend_version": importlib.metadata.version("sglang"),
                    "timing_source": "sglang_native_forward_pass_metrics",
                    "timing_rank": "attention_tp_rank_0",
                    "warmup_repeats": WARMUP_ROUNDS,
                    "measurement_repeats": MEASURED_ROUNDS,
                    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "format": "dynamo_fpm_rank_artifact_v2",
                },
                "serving_config": {"command": command, "startup_check": checks, "hit_unit": HIT_UNIT},
                "input_provenance": {
                    "source": "tokenizer_text",
                    "text_sha256": hashlib.sha256(text).hexdigest(),
                    "token_ids_sha256": hashlib.sha256(json.dumps(pool, separators=(",", ":")).encode()).hexdigest(),
                    "token_ids": pool,
                    "tokenizer_revision": args.model_revision,
                    "token_count": len(pool),
                    "unique_token_count": len(set(pool)),
                    "points_manifest_sha256": hashlib.sha256(args.points.read_bytes()).hexdigest(),
                },
            },
        }
        for point in points:
            record = run_point(args, base, stream, point, pool)
            raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            with evidence.open("ab") as handle:
                handle.write(raw + b"\n")
            records.append((point, record, hashlib.sha256(raw).hexdigest()))
            measured = [r["wall_time"] for r in record["rounds"] if r["role"] == "measurement"]
            log(
                f"{point['phase']}-{point['benchmark_id']:03d} B={point['batch_size']} "
                f"T={point.get('total_prefill_tokens', 0)} KV={point['total_kv_read_tokens']} "
                f"median={1000 * statistics.median(measured):.3f} ms rejected={len(record['rejected_rounds'])} "
                f"wall={record['wall_seconds']:.1f}s"
            )
        payload = artifact(args, points, records, provenance, started_at, time.monotonic() - started)
        payload["fpm_sequence_gaps"] = stream.gaps
        with evidence.open("rb") as handle:
            payload["input_provenance"]["repetition_evidence_manifest"] = {
                "schema_version": 1,
                "file": evidence.name,
                "sha256": hashlib.file_digest(handle, "sha256").hexdigest(),
                "records": len(records),
            }
        temporary = args.out / "benchmark.json.tmp"
        temporary.write_text(json.dumps(payload, indent=1))
        os.replace(temporary, args.out / "benchmark.json")
        return 0
    except BaseException as error:
        (args.out / "failed.json").write_text(
            json.dumps({"error": repr(error), "completed_points": len(records)}, indent=1)
        )
        raise
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        except Exception:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
