"""Buffered receiver for PR52061 native multipart FPM; no Dynamo scheduler."""
import argparse
import json
import math
import signal
import time

import msgspec
import zmq

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, default=20380)
p.add_argument('--ranks', type=int, default=1)
p.add_argument('--output', required=True)
a = p.parse_args()
running = True


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b'')
socket.setsockopt(zmq.RCVHWM, 100000)
for rank in range(a.ranks):
    socket.connect(f'tcp://127.0.0.1:{a.port + rank}')
encoder = msgspec.json.Encoder()
last = {}
counts = {}
gaps = resets = rejected = 0
next_flush = time.monotonic() + 1
with open(a.output, 'xb', buffering=1024 * 1024) as f:
    while running:
        if socket.poll(200):
            frames = socket.recv_multipart()
            try:
                if len(frames) != 3 or len(frames[1]) != 8:
                    raise ValueError('Bad FPM envelope')
                m = msgspec.msgpack.decode(frames[2])
                seq = int.from_bytes(frames[1], 'big')
                if m['version'] != 1 or m['timing_scope'] != 'model_step_cuda' or m['counter_id'] != seq:
                    raise ValueError('Unexpected FPM wire contract')
                if not math.isfinite(m['wall_time']) or m['wall_time'] < 0:
                    raise ValueError('Invalid GPU timing')
                rank = m['dp_rank']
                if not 0 <= rank < a.ranks:
                    raise ValueError('Unexpected DP rank')
                key = (m['worker_id'], rank)
                if key in last:
                    resets += seq <= last[key]
                    gaps += max(0, seq - last[key] - 1)
                last[key] = max(seq, last.get(key, seq))
                counts[str(rank)] = counts.get(str(rank), 0) + 1
                f.write(encoder.encode(dict(received_at_ns=time.time_ns(), metrics=m)) + b'\n')
            except (ValueError, KeyError, TypeError, msgspec.DecodeError):
                rejected += 1
        if time.monotonic() >= next_flush:
            f.flush()
            next_flush = time.monotonic() + 1
socket.close(linger=0)
context.term()
print(json.dumps(dict(records_by_rank=counts, counter_gaps=gaps, counter_resets=resets, rejected=rejected)), flush=True)
if rejected:
    raise SystemExit(1)
