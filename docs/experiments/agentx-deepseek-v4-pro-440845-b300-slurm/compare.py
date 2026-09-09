# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare measured exports without substituting missing metrics with zero."""
import json
import sys
from pathlib import Path


def summarize(path):
    p = json.loads(path.read_text())
    def value(key, stat='avg', unit=None):
        m = p[key]
        if unit is not None and m['unit'] != unit:
            raise ValueError((key, m['unit'], unit))
        return m[stat]
    result = {
        'total_tokens_per_s_per_gpu': value('effective_total_throughput', unit='tokens/sec') / 8,
        'output_tokens_per_s_per_gpu': value('output_token_throughput', unit='tokens/sec') / 8,
        'ttft_p50_ms': value('time_to_first_token', 'p50', 'ms'),
        'ttft_p90_ms': value('time_to_first_token', 'p90', 'ms'),
        'itl_p90_ms': value('inter_token_latency', 'p90', 'ms'),
        'request_latency_p90_ms': value('request_latency', 'p90', 'ms'),
        'requests': value('request_count'),
        'mean_input_tokens': value('input_sequence_length'),
        'mean_output_tokens': value('output_sequence_length'),
        'osl_mismatch_count': value('osl_mismatch_count'),
        'was_cancelled': p['was_cancelled'], 'error_summary': p['error_summary'],
        'start_time': p['start_time'], 'end_time': p['end_time'],
        'request_activity_duration_s': value('benchmark_duration'),
        'scenario': p['input_config']['scenario'],
        'phases': p['input_config']['phases'],
    }
    cache = p.get('usage_prompt_cache_read_tokens', {}).get('sum')
    prompt = p.get('input_sequence_length', {}).get('sum')
    result['response_prompt_cache_hit_fraction'] = cache / prompt if cache is not None and prompt else None
    result['submission_valid'] = p.get('submission_valid')
    result['validity_evidence'] = {k:v for k,v in p.items() if 'valid' in k or k in ['error_request_count', 'cancelled_request_count']}
    return result


def main(root):
    rows = {case: summarize(root / case / 'aiperf/profile_export_aiperf.json') for case in ['off', 'on']}
    keys = ['total_tokens_per_s_per_gpu', 'output_tokens_per_s_per_gpu', 'ttft_p50_ms',
            'ttft_p90_ms', 'itl_p90_ms', 'request_latency_p90_ms', 'requests', 'response_prompt_cache_hit_fraction']
    changes = {key: 100 * (rows['on'][key] / rows['off'][key] - 1) if rows['off'][key] not in [None, 0] and rows['on'][key] is not None else None for key in keys}
    fpm = json.loads((root / 'on/fpm-validation.json').read_text())
    status = json.loads((root / 'campaign-result.json').read_text())
    result = dict(cases=rows, on_vs_off_percent=changes, fpm=fpm, campaign=status,
                  caveat='One ordered pair, off then on; warmed compilation cache and fresh engine/KV plus warmup per case. Closed-loop requests may differ; no statistical overhead claim.')
    (root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# DSv4 AgentX 440845: FPM off/on comparison', '',
             'Both cases use one B300 node, the same FPM-fixed image, c32 and HiCache disabled.', '',
             '| Metric | FPM off | FPM on | On/off change |', '| --- | ---: | ---: | ---: |']
    for key in keys:
        a, b, d = rows['off'][key], rows['on'][key], changes[key]
        lines.append(f'| {key} | {a} | {b} | {d:.2f}% |' if d is not None else f'| {key} | {a} | {b} | unavailable |')
    lines += ['', result['caveat'], '', 'FPM rank coverage, counter gaps, request validity and cancellations are retained in comparison.json.']
    (root / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
