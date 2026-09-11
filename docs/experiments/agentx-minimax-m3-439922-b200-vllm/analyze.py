# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit native FPM and compare final AIPerf exports; run away from measured GPUs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text())


def quantile(values, q):
    if not values:
        return None
    pos = (len(values) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def audit_fpm(path):
    counts, per_rank, last = Counter(), defaultdict(Counter), {}
    durations, active_ranks = defaultdict(list), set()
    gaps = resets = invalid = 0
    digest = hashlib.sha256()
    first_ns = last_ns = None
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            record = json.loads(line)
            metric = record['metrics']
            rank, seq = metric['dp_rank'], metric['counter_id']
            key = (metric['worker_id'], rank)
            if key in last:
                gaps += max(0, seq - last[key] - 1)
                resets += seq <= last[key]
            last[key] = seq
            scheduled = metric['scheduled_requests']
            p, d = scheduled['num_prefill_requests'], scheduled['num_decode_requests']
            kind = 'mixed' if p and d else 'prefill_only' if p else 'decode_only' if d else 'no_requests'
            counts[kind] += 1
            per_rank[str(rank)][kind] += 1
            seconds = metric['wall_time']
            valid = (metric['version'] == 1 and metric['timing_scope'] == 'model_step_cuda'
                     and rank == 0 and math.isfinite(seconds) and seconds >= 0)
            if p or d:
                valid = valid and seconds > 0
                active_ranks.add(rank)
            if d:
                valid = valid and scheduled['sum_decode_kv_tokens'] > 0
            for field, value in scheduled.items():
                valid = valid and math.isfinite(value) and value >= 0
            invalid += not valid
            if valid:
                durations[kind].append(seconds * 1000)
            ts = record['received_at_ns']
            first_ns = ts if first_ns is None else min(first_ns, ts)
            last_ns = ts if last_ns is None else max(last_ns, ts)
    stats = {}
    for kind, values in durations.items():
        values.sort()
        stats[kind] = dict(mean_ms=statistics.fmean(values), min_ms=values[0],
                          p50_ms=quantile(values,.5), p90_ms=quantile(values,.9),
                          p95_ms=quantile(values,.95), p99_ms=quantile(values,.99),
                          max_ms=values[-1], sum_rank_seconds=sum(values)/1000)
    return dict(records=sum(counts.values()), counts=counts, per_rank=dict(per_rank),
                active_ranks=sorted(active_ranks), counter_gaps=gaps, counter_resets=resets,
                invalid_records=invalid, bytes=path.stat().st_size, sha256=digest.hexdigest(),
                first_received_ns=first_ns, last_received_ns=last_ns, timing=stats,
                valid=(active_ranks=={0} and not (gaps or resets or invalid)),
                scope='Entire capture including smoke, warmup, profiling, drain and idle; one record per DP rank, not deduplicated global iterations.')


def summarize(root):
    export = root/'aiperf/profile_export_aiperf.json'
    if not export.exists():
        export = root/'profile_export_aiperf.json'
    p = read(export)
    def val(key, stat='avg', unit=None):
        m = p[key]
        if unit is not None:
            assert m['unit']==unit,(key,m['unit'])
        value=m[stat]
        assert math.isfinite(value),(key,stat,value)
        return value
    metrics = {
        'total_tput_tps':val('effective_total_throughput',unit='tokens/sec'),
        'output_tput_tps':val('output_token_throughput',unit='tokens/sec'),
        'total_requests_completed':val('request_count'),
        'mean_input_tokens':val('input_sequence_length'),
        'mean_output_tokens_actual':val('output_sequence_length'),
        'duration_seconds':val('benchmark_duration'),
    }
    metrics['tput_per_gpu']=metrics['total_tput_tps']/4
    metrics['output_tput_per_gpu']=metrics['output_tput_tps']/4
    for name,key in [('ttft','time_to_first_token'),('itl','inter_token_latency'),('e2el','request_latency')]:
        for label,stat in [('mean','avg'),('median','p50'),('p90','p90'),('p95','p95')]:
            metrics[f'{label}_{name}']=val(key,stat,'ms')/1000
    log=(root/'client.log').read_text() if (root/'client.log').exists() else ''
    cancelled=re.findall(r'PhaseRecordsStats\(phase=CreditPhase.PROFILING[^\n]*?final_requests_cancelled=(\d+)',log)
    warmup=p.get('warmup_metrics',{})
    return dict(metrics=metrics,submission_valid=p['metadata']['submission_valid'],
                was_cancelled=p['was_cancelled'],error_summary=p['error_summary'],
                osl_mismatch_count=val('osl_mismatch_count'),
                profiling_cancelled_requests=int(cancelled[-1]) if cancelled else None,
                warmup_requests=warmup.get('request_count',{}).get('avg'),
                warmup_errors=warmup.get('error_request_count',{}).get('avg'),
                metric_duration_coverage=p['metadata']['metric_duration_coverage'],
                start=p['start_time'],end=p['end_time'],cli=p['run_info']['cli_command'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--reference-api',type=Path,required=True)
    parser.add_argument('--baseline-root',type=Path)
    args=parser.parse_args()
    reference=read(args.reference_api)
    if isinstance(reference,dict) and 'reference' in reference:
        reference=reference['reference']
    if isinstance(reference,list):
        reference=next(row for row in reference if str(row['id'])=='439922')
    assert str(reference['id'])=='439922'
    off_root=args.baseline_root or args.root
    cases={'off':summarize(off_root/'off'), 'on':summarize(args.root/'on')}
    audit=audit_fpm(args.root/'on/fpm.jsonl')
    checksum=(args.root/'on/fpm.sha256').read_text().split()[0]
    assert audit['sha256']==checksum
    rows=[]
    for metric,off in cases['off']['metrics'].items():
        on=cases['on']['metrics'][metric]
        sa=reference['metrics'].get(metric)
        rows.append(dict(metric=metric,reference=sa,off=off,on=on,
                         off_vs_reference_pct=100*(off/sa-1) if sa else None,
                         on_vs_off_pct=100*(on/off-1) if off else None))
    result=dict(reference=reference,cases=cases,comparison=rows,fpm=audit,
                campaign=read(args.root/'campaign-result.json'),
                pairing=dict(separate_allocations=off_root.resolve()!=args.root.resolve(),
                    off_root=str(off_root), on_root=str(args.root),
                    off_campaign=read(off_root/'campaign-result.json'),
                    baseline_link=read(args.root/'baseline-link.json') if (args.root/'baseline-link.json').exists() else None),
                analyzed_at=datetime.now(timezone.utc).isoformat(),
                caveat='One off/on comparison; G2 disabled, fresh engines/KV and warmup per case. Inspect pairing for separate allocations/GPU UUID differences; non-exclusive host and newer runtime confound small differences. FPM timing is recorded-stream elapsed time, not summed kernel-busy time or client TTFT/ITL.')
    (args.root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    assert audit['valid'],audit
    assert result['campaign']['status']=='complete'
    for case in cases.values():
        assert case['submission_valid'] and not case['was_cancelled']


if __name__=='__main__':
    main()
