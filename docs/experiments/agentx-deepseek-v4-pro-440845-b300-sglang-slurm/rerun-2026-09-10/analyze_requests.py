# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit AgentX source-request overlap and context/cache strata from client records."""
import collections
import json
import math
from pathlib import Path
import statistics
import sys


def quantile(values, q):
    a=sorted(v for v in values if v is not None and math.isfinite(v))
    if not a:return None
    i=(len(a)-1)*q;lo=int(i);hi=min(lo+1,len(a)-1)
    return a[lo]+(a[hi]-a[lo])*(i-lo)


def read(path):
    rows=[];errors=collections.Counter()
    for line in path.open():
        x=json.loads(line);meta=x['metadata']
        if meta.get('benchmark_phase')!='profiling':continue
        if x.get('error'):
            errors[x['error'].get('type','unknown')]+=1;continue
        m=x['metrics']
        def value(k):return m[k]['value'] if k in m else None
        prompt=value('input_sequence_length');output=value('output_sequence_length')
        # The pinned SGLang UsageProcessor emits prompt_tokens_details only for
        # positive cached counts. Cache reporting is enabled in these runs.
        cached=value('usage_prompt_cache_read_tokens') or 0
        assert prompt is not None and 0<=cached<=prompt,(prompt,cached)
        key=tuple(meta.get(k) for k in ['source_trace_id','source_kind','source_outer_idx','source_inner_idx','conversation_id','turn_index'])
        rows.append(dict(key=key,prompt=prompt,output=output,cached=cached,uncached=prompt-cached,reuse=cached/prompt if prompt else 0,
                         ttft=value('time_to_first_token'),itl=value('inter_token_latency'),latency=value('request_latency')))
    return rows,dict(errors)


def bucket(row,dimension):
    if dimension=='reuse':
        v=row['reuse']
        return '0%' if v==0 else '(0,90%)' if v<.9 else '[90,99%)' if v<.99 else '[99,100%)' if v<1 else '100%'
    v=row[dimension]
    if dimension=='uncached' and v==0:return '0'
    return '<=8K' if v<=8192 else '(8K,32K]' if v<=32768 else '(32K,128K]' if v<=131072 else '(128K,256K]' if v<=262144 else '>256K'


def summary(rows):
    prompt=sum(r['prompt'] for r in rows)
    return dict(requests=len(rows),prompt_tokens=prompt,uncached_tokens=sum(r['uncached'] for r in rows),
                token_weighted_reuse=sum(r['cached'] for r in rows)/prompt if prompt else None,
                mean_prompt=statistics.mean(r['prompt'] for r in rows) if rows else None,
                mean_uncached=statistics.mean(r['uncached'] for r in rows) if rows else None,
                ttft_p50_ms=quantile([r['ttft'] for r in rows],.5),ttft_p90_ms=quantile([r['ttft'] for r in rows],.9),
                itl_p50_ms=quantile([r['itl'] for r in rows],.5),itl_p90_ms=quantile([r['itl'] for r in rows],.9),
                latency_p50_ms=quantile([r['latency'] for r in rows],.5),latency_p90_ms=quantile([r['latency'] for r in rows],.9))


def compare_pairs(pairs):
    deltas=[b['ttft']-a['ttft'] for a,b in pairs if a['ttft'] is not None and b['ttft'] is not None]
    ratios=[100*(b['ttft']/a['ttft']-1) for a,b in pairs if a['ttft'] and b['ttft'] is not None]
    itl_ratios=[100*(b['itl']/a['itl']-1) for a,b in pairs if a['itl'] and b['itl'] is not None]
    return dict(requests=len(pairs),median_paired_itl_delta_percent=quantile(itl_ratios,.5),off=summary([a for a,b in pairs]),on=summary([b for a,b in pairs]),
                median_paired_ttft_delta_ms=quantile(deltas,.5),median_paired_ttft_delta_percent=quantile(ratios,.5))


def main(root):
    cases={};error={}
    for c in ['off','on']:cases[c],error[c]=read(root/c/'aiperf/profile_export.jsonl')
    groups={c:collections.defaultdict(list) for c in cases}
    for c,rows in cases.items():
        for r in rows:groups[c][r['key']].append(r)
    common=set(groups['off']) & set(groups['on'])
    pairs=[(groups['off'][k][0],groups['on'][k][0]) for k in sorted(common,key=repr) if len(groups['off'][k])==len(groups['on'][k])==1]
    same_prompt=[(a,b) for a,b in pairs if a['prompt']==b['prompt']]
    same_prompt_cache=[(a,b) for a,b in same_prompt if a['cached']==b['cached']]
    same_lengths=[(a,b) for a,b in same_prompt if a['output']==b['output']]
    same_cache=[(a,b) for a,b in same_lengths if a['cached']==b['cached']]
    near_same_work=[(a,b) for a,b in pairs if abs(a['prompt']-b['prompt'])<=4 and abs(a['uncached']-b['uncached'])<=4 and a['output']==b['output']]
    strata={}
    for dim in ['prompt','uncached','reuse']:
        strata[dim]={}
        labels=sorted({bucket(r,dim) for rows in cases.values() for r in rows})
        for label in labels:
            strata[dim][label]={c:summary([r for r in rows if bucket(r,dim)==label]) for c,rows in cases.items()}
    result=dict(cases={c:summary(rows) for c,rows in cases.items()},request_errors=error,
                distinct_source_keys={c:len(v) for c,v in groups.items()},shared_keys=len(common),
                duplicate_keys={c:sum(len(v)>1 for v in g.values()) for c,g in groups.items()},
                matched_unique=compare_pairs(pairs),
                matched_within4_prompt_uncached_tokens_same_output=compare_pairs(near_same_work),
                matched_near_work_by_prompt={label:compare_pairs([(a,b) for a,b in near_same_work if bucket(a,'prompt')==label]) for label in sorted({bucket(a,'prompt') for a,b in near_same_work})},matched_same_prompt_length=compare_pairs(same_prompt),
                matched_same_prompt_and_cached_tokens=compare_pairs(same_prompt_cache),
                matched_same_input_output_lengths=compare_pairs(same_lengths),
                matched_same_lengths_and_cached_tokens=compare_pairs(same_cache),strata=strata,
                cache_change_matched={label:compare_pairs([(a,b) for a,b in pairs if ('more_uncached' if b['uncached']>a['uncached'] else 'less_uncached' if b['uncached']<a['uncached'] else 'same_uncached')==label]) for label in ['more_uncached','less_uncached','same_uncached']},
                matched_same_prompt_cache_by_prompt={label:compare_pairs([(a,b) for a,b in same_prompt_cache if bucket(a,'prompt')==label]) for label in sorted({bucket(a,'prompt') for a,b in same_prompt_cache})},
                caveats=['Only unique source keys are matched; duplicate/recycled keys are excluded.',
                         'Missing cached metric maps to zero according to pinned SGLang positive-only cache-detail serialization.',
                         'Conditioning on request lengths/cache counts does not control concurrent scheduler state or establish causality.'])
    (root/'request-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='strata'},indent=2))


if __name__=='__main__':main(Path(sys.argv[1]))
