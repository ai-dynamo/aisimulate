# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record raw ForwardPassMetrics from engine ZMQ publishers to JSONL for step captures.

    python -m aisim_core.fpm_record --endpoint tcp://127.0.0.1:20380 --output steps.jsonl.gz

Subscribes to one or more FPM publishers (for example Dynamo's
``InstrumentedScheduler``, one port per attention-DP rank) and writes one line
per message::

    {"recv_ms": <receiver Unix ms>, "endpoint": "...", "seq": <publisher seq>, "fpm": {...}}

The learned-model trainer reads this shape directly (the ``fpm`` envelope with
``recv_ms``). Payloads are decoded *untyped*, so fields added by
``aisim_core.fpm_hooks`` (per-request lists, step diagnostics) are kept; a
typed decode with a stock ``ForwardPassMetrics`` would silently drop them.

The recorder runs until SIGINT or SIGTERM, drains what is already queued, and
writes ``<output>.summary.json`` with per-endpoint counts, idle heartbeats,
sequence gaps and undecodable messages. Gaps are diagnostics only: a
publisher that drops a message before numbering it leaves no gap. Start the
recorder before the engine so the first steps are not lost to ZMQ's
subscription handshake. Existing output files are refused.
"""

from __future__ import annotations

import argparse
import gzip
import json
import signal
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

SUMMARY_SCHEMA = "aisim.fpm_record.summary.v1"


@dataclass
class EndpointStats:
    messages: int = 0
    heartbeats: int = 0
    undecodable: int = 0
    first_seq: int | None = None
    last_seq: int | None = None
    sequence_gaps: int = 0
    missing_messages: int = 0
    out_of_order: int = 0


@dataclass
class Recording:
    """Writes decoded messages and accumulates per-endpoint statistics."""

    stream: IO[str]
    clock_ms: Callable[[], float] = field(default=lambda: time.time() * 1000.0)
    endpoints: dict[str, EndpointStats] = field(default_factory=dict)

    def add(self, endpoint: str, frames: Sequence[bytes]) -> None:
        """Record one multipart message ``(topic, seq, payload)`` or ``(payload,)``."""
        import msgspec

        stats = self.endpoints.setdefault(endpoint, EndpointStats())
        stats.messages += 1
        seq = int.from_bytes(frames[-2], "big") if len(frames) >= 2 and len(frames[-2]) == 8 else None
        try:
            fpm = msgspec.msgpack.decode(frames[-1])
        except msgspec.DecodeError:
            stats.undecodable += 1
            return
        if not isinstance(fpm, dict) or "scheduled_requests" not in fpm:
            stats.undecodable += 1
            return
        if seq is not None:
            self._track_sequence(stats, seq)
        if not fpm.get("wall_time"):
            stats.heartbeats += 1
        line = {"recv_ms": self.clock_ms(), "endpoint": endpoint, "seq": seq, "fpm": fpm}
        self.stream.write(json.dumps(line, separators=(",", ":"), allow_nan=False) + "\n")

    @staticmethod
    def _track_sequence(stats: EndpointStats, seq: int) -> None:
        if stats.first_seq is None:
            stats.first_seq = seq
        elif stats.last_seq is not None:
            if seq <= stats.last_seq:
                stats.out_of_order += 1
                return
            if seq > stats.last_seq + 1:
                stats.sequence_gaps += 1
                stats.missing_messages += seq - stats.last_seq - 1
        stats.last_seq = seq

    def summary(self, output: Path, started_ms: float) -> dict[str, Any]:
        return {
            "schema": SUMMARY_SCHEMA,
            "output": output.name,
            "started_ms": started_ms,
            "ended_ms": self.clock_ms(),
            "endpoints": {name: asdict(stats) for name, stats in sorted(self.endpoints.items())},
        }


def open_output(path: Path) -> IO[str]:
    """Open a new JSONL output (gzip when the name ends in ``.gz``); never overwrite."""
    if path.name.endswith(".gz"):
        return gzip.open(path, "xt", encoding="utf-8")
    return path.open("x", encoding="utf-8")


def record(
    endpoints: Iterable[str],
    output: Path,
    stop: threading.Event,
    *,
    ready: threading.Event | None = None,
    flush_interval_s: float = 1.0,
) -> dict[str, Any]:
    """Subscribe to ``endpoints`` and record until ``stop`` is set; returns the summary."""
    import zmq

    endpoints = list(endpoints)
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    context = zmq.Context()
    poller = zmq.Poller()
    sockets: dict[Any, str] = {}
    started_ms = time.time() * 1000.0
    try:
        for endpoint in endpoints:
            socket = context.socket(zmq.SUB)
            socket.setsockopt(zmq.RCVHWM, 0)  # never drop on the receiving side
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            socket.connect(endpoint)
            poller.register(socket, zmq.POLLIN)
            sockets[socket] = endpoint
        with open_output(output) as stream:
            recording = Recording(stream)
            if ready is not None:
                ready.set()
            last_flush = time.monotonic()
            while True:
                stopping = stop.is_set()
                events = dict(poller.poll(0 if stopping else 200))
                for socket, endpoint in sockets.items():
                    if events.get(socket) == zmq.POLLIN:
                        while True:
                            try:
                                frames = socket.recv_multipart(zmq.NOBLOCK)
                            except zmq.Again:
                                break
                            recording.add(endpoint, frames)
                if stopping and not events:
                    break  # stop requested and everything queued is drained
                if time.monotonic() - last_flush >= flush_interval_s:
                    stream.flush()
                    last_flush = time.monotonic()
        summary = recording.summary(output, started_ms)
    finally:
        for socket in sockets:
            socket.close(0)
        context.term()
    summary_path = output.with_name(output.name + ".summary.json")
    with summary_path.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aisim_core.fpm_record", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--endpoint", action="append", required=True, help="ZMQ PUB endpoint; repeat per DP rank")
    parser.add_argument("--output", type=Path, required=True, help="new .jsonl or .jsonl.gz file")
    parser.add_argument("--ready-file", type=Path, help="created once all subscriptions are connected")
    args = parser.parse_args(argv)
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    ready = threading.Event()
    if args.ready_file is not None:
        ready_file = args.ready_file

        def announce() -> None:
            ready.wait()
            ready_file.write_text("ready\n")

        threading.Thread(target=announce, daemon=True).start()
    summary = record(args.endpoint, args.output, stop, ready=ready)
    totals = {name: (stats["messages"], stats["missing_messages"]) for name, stats in summary["endpoints"].items()}
    print(f"fpm_record: {args.output} messages/missing per endpoint: {totals}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
