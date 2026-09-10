# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record native SGLang FPM from each DP rank, outside the engine process."""
import argparse
import json
import math
import signal
import time
from pathlib import Path


def validate(path, ranks):
    stats = {}
    errors = 0
    for line in path.open():
        item = json.loads(line)
        m = item['metrics']
        rank = m['dp_rank']
        s = stats.setdefault(rank, dict(records=0, active=0, decode=0, missing=0, resets=0, first=None, last=None))
        counter = m['counter_id']
        if s['last'] is not None:
            if counter <= s['last']:
                s['resets'] += 1
            else:
                s['missing'] += counter - s['last'] - 1
        if s['first'] is None:
            s['first'] = counter
        s['last'] = counter
        s['records'] += 1
        req = m['scheduled_requests']
        active = req['num_prefill_requests'] + req['num_decode_requests'] > 0
        s['active'] += active
        s['decode'] += req['num_decode_requests'] > 0
        good = m['version'] == 1 and item['wire_counter'] == counter and math.isfinite(m['wall_time']) and m['wall_time'] >= 0
        if req['num_decode_requests']:
            good &= req['sum_decode_kv_tokens'] > 0 and math.isfinite(req['var_decode_kv_tokens']) and req['var_decode_kv_tokens'] >= 0
        errors += not good
    result = dict(ranks=stats, invalid_records=errors, bytes=path.stat().st_size,
                  complete_rank_coverage=set(stats) == set(range(ranks)),
                  all_ranks_active=all(s['active'] > 0 for s in stats.values()),
                  all_ranks_decode=all(s['decode'] > 0 for s in stats.values()))
    result['valid'] = bool(stats) and not errors and result['complete_rank_coverage'] and result['all_ranks_active'] and result['all_ranks_decode']
    print(json.dumps(result, indent=2))
    return 0 if result['valid'] else 1


def record(endpoint, path, ranks):
    import msgspec
    import zmq
    from sglang.srt.observability.forward_pass_metrics import decode

    stopping = False
    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    context = zmq.Context()
    sockets = []
    poller = zmq.Poller()
    for rank in range(ranks):
        sock = context.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b'')
        sock.setsockopt(zmq.RCVHWM, 100000)
        sock.connect(f'{endpoint}.{rank}')
        poller.register(sock, zmq.POLLIN)
        sockets.append(sock)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', buffering=1024 * 1024) as stream:
            last_flush = time.monotonic()
            drain_deadline = None
            while True:
                if stopping and drain_deadline is None:
                    drain_deadline = time.monotonic() + 1
                if drain_deadline is not None and time.monotonic() >= drain_deadline:
                    break
                for sock, _ in poller.poll(100):
                    frames = sock.recv_multipart()
                    if len(frames) != 3:
                        raise ValueError(f'Unexpected FPM frame count {len(frames)}')
                    m = msgspec.to_builtins(decode(frames[2]))
                    stream.write(json.dumps(dict(received_at_ns=time.time_ns(), wire_counter=int.from_bytes(frames[1], 'big'), metrics=m), separators=(',', ':')) + '\n')
                if time.monotonic() - last_flush >= 1:
                    stream.flush()
                    last_flush = time.monotonic()
            stream.flush()
    finally:
        for sock in sockets:
            sock.close(linger=0)
        context.term()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['record', 'validate'])
    parser.add_argument('path', type=Path)
    parser.add_argument('--endpoint')
    parser.add_argument('--ranks', type=int, default=8)
    args = parser.parse_args()
    if args.mode == 'record' and not args.endpoint:
        parser.error('--endpoint is required for recording')
    raise SystemExit(validate(args.path, args.ranks) if args.mode == 'validate' else record(args.endpoint, args.path, args.ranks))
