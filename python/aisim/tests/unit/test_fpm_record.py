# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FPM step recorder keeps hook fields and accounts for publisher sequences."""

import gzip
import io
import json
import threading
import time
import types

import pytest

pytestmark = pytest.mark.unit

msgspec = pytest.importorskip("msgspec")

from aisim_core import fpm_record
from aisim_core.fpm_hooks import _struct


def _producer():
    class ScheduledRequestMetrics(msgspec.Struct, frozen=True, gc=False):
        num_prefill_requests: int = 0
        sum_prefill_tokens: int = 0
        num_decode_requests: int = 0
        sum_decode_kv_tokens: int = 0

    class ForwardPassMetrics(msgspec.Struct, frozen=True, gc=False):
        version: int = 1
        counter_id: int = 0
        wall_time: float = 0.0
        scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()

    return types.SimpleNamespace(scheduled=ScheduledRequestMetrics, fpm=ForwardPassMetrics)


def _step(producer, seq, wall_time=0.012, past=(700, 90)):
    extended = _struct.extend_struct(producer.scheduled)
    base = producer.scheduled(num_decode_requests=len(past), sum_decode_kv_tokens=sum(past))
    scheduled = _struct.with_pairs(extended, base, [(1, p) for p in past])
    payload = msgspec.msgpack.encode(producer.fpm(counter_id=seq, wall_time=wall_time, scheduled_requests=scheduled))
    return [b"", seq.to_bytes(8, "big"), payload]


def test_recording_keeps_hook_fields_and_counts_sequence_gaps():
    producer = _producer()
    stream = io.StringIO()
    recording = fpm_record.Recording(stream, clock_ms=lambda: 1000.0)
    endpoint = "tcp://127.0.0.1:20380"
    for frames in (
        _step(producer, 1),
        _step(producer, 2, wall_time=0.0),  # idle heartbeat
        _step(producer, 5),  # 3 and 4 were lost
        _step(producer, 4),  # late
        [b"", (6).to_bytes(8, "big"), b"\xc1"],  # not msgpack
        [msgspec.msgpack.encode({"unrelated": 1})],  # no scheduled_requests
    ):
        recording.add(endpoint, frames)
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [line["seq"] for line in lines] == [1, 2, 5, 4]
    first = lines[0]
    assert first["recv_ms"] == 1000.0 and first["endpoint"] == endpoint
    assert first["fpm"]["scheduled_requests"]["past_kv_lengths"] == [700, 90]
    assert first["fpm"]["scheduled_requests"]["extend_lengths"] == [1, 1]
    stats = recording.endpoints[endpoint]
    assert (stats.messages, stats.heartbeats, stats.undecodable) == (6, 1, 2)
    assert (stats.first_seq, stats.last_seq, stats.sequence_gaps, stats.missing_messages) == (1, 5, 1, 2)
    assert stats.out_of_order == 1


def test_recorded_lines_are_read_by_the_learned_trainer():
    from aisim_core.sdk import fpm_learned

    producer = _producer()
    stream = io.StringIO()
    fpm_record.Recording(stream, clock_ms=lambda: 5.0).add("e", _step(producer, 1))
    fpm = fpm_learned._unwrap_record(json.loads(stream.getvalue()))
    assert fpm["scheduled_requests"]["past_kv_lengths"] == [700, 90] and fpm["recv_ms"] == 5.0


def test_open_output_refuses_existing_files_and_writes_gzip(tmp_path):
    path = tmp_path / "steps.jsonl.gz"
    with fpm_record.open_output(path) as stream:
        stream.write("{}\n")
    with gzip.open(path, "rt") as stream:
        assert stream.read() == "{}\n"
    with pytest.raises(FileExistsError), fpm_record.open_output(path):
        pass


def test_record_over_zmq_drains_and_writes_summary(tmp_path):
    zmq = pytest.importorskip("zmq")
    producer = _producer()
    context = zmq.Context.instance()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    output = tmp_path / "steps.jsonl"
    stop, ready = threading.Event(), threading.Event()
    result = {}
    worker = threading.Thread(
        target=lambda: result.update(fpm_record.record([f"tcp://127.0.0.1:{port}"], output, stop, ready=ready))
    )
    worker.start()
    assert ready.wait(5)
    # The subscription handshake is asynchronous: publish until the first message lands.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not output.read_text():
        publisher.send_multipart(_step(producer, 1))
        time.sleep(0.05)
    for seq in range(2, 6):
        publisher.send_multipart(_step(producer, seq))
    time.sleep(0.2)
    stop.set()
    worker.join(5)
    publisher.close(0)
    seqs = [json.loads(line)["seq"] for line in output.read_text().splitlines()]
    assert seqs[-4:] == [2, 3, 4, 5]
    summary = json.loads((tmp_path / "steps.jsonl.summary.json").read_text())
    assert summary["schema"] == fpm_record.SUMMARY_SCHEMA
    assert summary["endpoints"][f"tcp://127.0.0.1:{port}"]["last_seq"] == 5
