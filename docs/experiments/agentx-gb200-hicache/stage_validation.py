# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded 1P/1D acceptance suite; does not automatically authorize expansion."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.request
from staged_control import DGD, cleanup, collect, now, pods, save

ROOT = Path(os.environ['AGENTX_RUN_DIR'])
ROOT.mkdir(parents=True, exist_ok=True)
MODEL = 'nvidia/GLM-5.2-NVFP4'
BASE = 'http://127.0.0.1:8000'
results = []


def get(url):
    with urllib.request.urlopen(url, timeout=15) as response:
        return response.read()


def metrics(label):
    items = pods()
    for pod in items:
        name = pod['metadata']['name']
        if '-prefill-wkr-' in name or 'frontend' in name:
            continue
        try:
            (ROOT / (label + '.' + name + '.prom')).write_bytes(get('http://' + pod['status']['podIP'] + ':9090/metrics'))
        except Exception as exc:
            save(ROOT, label + '.' + name + '.metrics-error.json', {'error': repr(exc)})


def request(label, length, rank, marker, output=1, prefill_worker_id=None):
    tokens = [14978, marker] + [14978] * (length - 2)
    body = {'model': MODEL, 'prompt': tokens, 'max_tokens': output,
            'stream': False, 'ignore_eos': True,
            'nvext': {'extra_fields': ['worker_id', 'timing', 'engine_data']}}
    if rank is not None:
        if prefill_worker_id is None:
            raise ValueError('A fixed DP rank requires a current prefill worker ID')
        body['nvext'].update(prefill_worker_id=prefill_worker_id, prefill_dp_rank=rank)
    started = time.monotonic()
    req = urllib.request.Request(BASE + '/v1/completions',
                                 data=json.dumps(body, separators=(',', ':')).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as response:
        payload = json.load(response)
    result = {'label': label, 'utc': now(), 'requested_input_tokens': length,
              'requested_prefill_dp_rank': rank,
              'elapsed_seconds': time.monotonic() - started, 'response': payload}
    save(ROOT, 'requests/' + label + '.json', result)
    assert 'error' not in payload, payload
    assert payload.get('usage', {}).get('prompt_tokens') == length, payload.get('usage')
    assert 1 <= payload.get('usage', {}).get('completion_tokens', 0) <= output, payload.get('usage')
    actual_rank = payload.get('nvext', {}).get('worker_id', {}).get('prefill_dp_rank')
    if rank is not None:
        assert actual_rank == rank, ('routing rank mismatch', rank, actual_rank, payload.get('nvext'))
        assert payload['nvext']['worker_id']['prefill_worker_id'] == prefill_worker_id
    print(json.dumps({k: v for k, v in result.items() if k != 'response'}), flush=True)
    return result


if (ROOT / 'STARTED.json').exists():
    save(ROOT, 'INTERRUPTED.json', {'utc': now(), 'reason': 'Refusing duplicate validation run'})
    cleanup()
    raise SystemExit(1)

try:
    ready_deadline = time.monotonic() + 1800
    while True:
        try:
            items = pods()
            leader = next(p for p in items if '-prefill-ldr-' in p['metadata']['name'])
            get('http://' + leader['status']['podIP'] + ':9090/live')
            get('http://' + DGD + '-decode:9090/live')
            assert any(m['id'] == MODEL for m in json.loads(get(BASE + '/v1/models'))['data'])
            break
        except Exception:
            if time.monotonic() > ready_deadline:
                raise TimeoutError('Models did not become ready within 1800s')
            time.sleep(10)
    collect(ROOT)
    save(ROOT, 'AWAITING_INTERCONNECT.json', {'utc': now()})
    gate_deadline = time.monotonic() + 1200
    while not (ROOT / 'INTERCONNECT_OK.json').exists():
        if time.monotonic() > gate_deadline:
            raise TimeoutError('Interconnect review was not completed within 1200s')
        time.sleep(5)
    with (ROOT / 'STARTED.json').open('x') as file:
        json.dump({'utc': now(), 'phase': '1P1D validation'}, file)
    metrics('before')
    decode_metrics = get('http://' + DGD + '-decode:9090/metrics').decode()
    capacities = [float(value) for value in re.findall(r'^sglang:max_total_num_tokens\{[^\n]*\}\s+([0-9.e+]+)', decode_metrics, re.M)]
    assert capacities, 'Missing real decode KV capacity metric'
    capacity = min(capacities)
    # Attempt5's 51 actual snapshot primers maxed at 688430 input tokens.
    # 996579 was the dataset-wide input+output context peak, NOT a primer ISL.
    # Exercise a separate, explicitly synthetic near-capacity input as well.
    primer_max_input = 688430
    lengths = [32768, 131072, 262144, 524288, primer_max_input, 786432, 983040]
    required = math.ceil(max(lengths) / 64) * 64 + 512
    save(ROOT, 'capacity.json', {'decode_kv_tokens': capacity,
                               'observed_primer_max_input': primer_max_input,
                               'largest_synthetic_test_input': max(lengths),
                               'required_with_reserve': required})
    if capacity < required:
        raise ValueError(('Synthetic acceptance bound cannot fit', capacity, required))
    # A rank-only hint is not a pin in this Dynamo revision: discover the
    # currently registered worker before setting the (worker, rank) pair.
    discovered = request('discover-worker', 64, None, 999)
    results.append(discovered)
    prefill_worker_id = discovered['response']['nvext']['worker_id']['prefill_worker_id']
    for rank in range(8):
        results.append(request('rank-' + str(rank), 4096, rank, 1000 + rank, output=8,
                               prefill_worker_id=prefill_worker_id))
    for index, length in enumerate(lengths):
        results.append(request('long-' + str(length), length, 0, 2000 + index,
                               prefill_worker_id=prefill_worker_id))
    metrics('before-reload')
    results.append(request('repeat-524288', 524288, 0, 2003,
                           prefill_worker_id=prefill_worker_id))
    metrics('after-reload')
    for concurrency in [2, 4, 8]:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending = [executor.submit(request, 'c' + str(concurrency) + '-' + str(i),
                                       131072, i % 8, 3000 + concurrency * 10 + i, 16,
                                       prefill_worker_id)
                       for i in range(concurrency)]
            results.extend(f.result() for f in pending)
        metrics('after-c' + str(concurrency))
    save(ROOT, 'HTTP_SUITE_PASS.json', {'utc': now(), 'requests': len(results),
                                      'errors': 0, 'requires_manual_review_before_stage2': True})
    print('HTTP_SUITE_PASS: inspect metrics/transport/G2 evidence before 3D expansion', flush=True)
    collect(ROOT)
    # Keep workers available briefly for final transport/counter inspection.
    for _ in range(120):
        if (ROOT / 'ACCEPTED.json').exists():
            break
        time.sleep(5)
except BaseException as exc:
    save(ROOT, 'FAILED.json', {'utc': now(), 'error': repr(exc), 'returned_requests': len(results)})
    raise
finally:
    try:
        metrics('final')
        collect(ROOT)
        os.sync()
    finally:
        cleanup()
