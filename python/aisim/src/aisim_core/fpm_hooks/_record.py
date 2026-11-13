# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record an engine's FPM ZMQ stream to JSON Lines, keeping every key.

``python -m aisim_core.fpm_hooks record --endpoint ipc:///path.0 --output fpm.jsonl``

SGLang (``--enable-forward-pass-metrics``) publishes one message per scheduler
iteration on ``{--forward-pass-metrics-ipc-name}.{dp_rank}`` as the multipart
frame ``(topic, sequence as 8-byte big-endian, msgpack payload)``. Without
Dynamo nothing stores that stream; this recorder does.

The payload is decoded without a type, so fields added by the hooks are kept.
Decoding with the producer's typed ``ForwardPassMetrics`` would drop them
(see the learned-model design notes). Each line is

    {"sequence": n, "received_unix_s": t, "received_monotonic_s": m, "fpm": {...}}

Idle heartbeats are kept as published. Run one recorder per capture run and
split tiers afterwards by receive time. A summary with the message count and
sequence gaps (dropped messages) is written to ``<output>.summary.json`` on exit.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any

Frame = tuple[bytes, bytes]  # (sequence, payload)


def record_frames(
    frames: Iterable[Frame], out: IO[bytes], clock: Callable[[], tuple[float, float]] | None = None
) -> dict:
    """Write each ``(sequence, payload)`` frame as one JSON line; return the run summary."""
    import msgspec

    encode = msgspec.json.Encoder().encode
    decode = msgspec.msgpack.Decoder().decode  # untyped: every key survives
    clock = clock or (lambda: (time.time(), time.monotonic()))
    summary: dict[str, Any] = {
        "messages": 0,
        "undecodable": 0,
        "first_sequence": None,
        "last_sequence": None,
        "gaps": [],
    }
    for sequence_bytes, payload in frames:
        sequence = int.from_bytes(sequence_bytes, "big")
        unix_s, monotonic_s = clock()
        last = summary["last_sequence"]
        if last is not None and sequence != last + 1:
            summary["gaps"].append([last, sequence])  # a restart shows as a backward gap
        if summary["first_sequence"] is None:
            summary["first_sequence"] = sequence
        summary["last_sequence"] = sequence
        try:
            fpm = decode(payload)
        except msgspec.DecodeError:
            summary["undecodable"] += 1
            continue
        line = {"sequence": sequence, "received_unix_s": unix_s, "received_monotonic_s": monotonic_s, "fpm": fpm}
        out.write(encode(line) + b"\n")
        summary["messages"] += 1
    return summary


def _zmq_frames(endpoint: str, stop: Callable[[], bool], deadline: float | None) -> Iterable[Frame]:
    import zmq

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 0)  # never drop on our side
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    try:
        while not stop() and (deadline is None or time.monotonic() < deadline):
            if not poller.poll(200):
                continue
            while True:
                try:
                    parts = socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                if len(parts) == 3:
                    yield parts[1], parts[2]
    finally:
        socket.close(linger=0)
        context.term()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m aisim_core.fpm_hooks record", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--endpoint", required=True, help="ZMQ endpoint including the rank suffix, e.g. ipc:///tmp/fpm.0"
    )
    parser.add_argument("--output", required=True, type=Path, help="JSON Lines file to append to")
    parser.add_argument(
        "--duration", type=float, default=None, help="stop after this many seconds (default: until SIGINT/SIGTERM)"
    )
    args = parser.parse_args(argv)

    stopping = False

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    deadline = time.monotonic() + args.duration if args.duration is not None else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with args.output.open("ab", buffering=1 << 20) as out:
        summary = record_frames(_zmq_frames(args.endpoint, lambda: stopping, deadline), out)
    summary.update(
        {
            "endpoint": args.endpoint,
            "output": str(args.output),
            "started_unix_s": started,
            "stopped_unix_s": time.time(),
        }
    )
    summary_path = args.output.with_name(args.output.name + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"fpm record: {summary['messages']} messages, {len(summary['gaps'])} sequence gaps -> {args.output}",
        file=sys.stderr,
    )
    return 0
