# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize three independent off/on pairs per concurrency without hiding run variance."""
import json
import statistics
import sys
from pathlib import Path


def load(path):
    lines=path.read_text().splitlines()
    assert len(lines)==1,path
    return json.loads(lines[0])


def main(root):
    modes=['off','on','on','off','off','on']
    raw={}
    keys=['output_throughput','total_throughput','mean_ttft_ms','median_ttft_ms','p90_ttft_ms','mean_tpot_ms','median_tpot_ms','p90_tpot_ms','mean_itl_ms','p90_itl_ms','mean_e2e_latency_ms','p90_e2e_latency_ms','concurrency','duration','completed']
    for i,mode in enumerate(modes):
        block=root/f'block-{i}-{mode}'
        for c in [1,32,128]:
            p=block/f'c{c}-measured.jsonl'
            assert Path(str(p)+'.validated').exists(),p
            data=load(p)
            manifest=json.loads(Path(str(p)+'.inputs.json').read_text())
            raw[f'{i}:{c}']=dict(block=i,mode=mode,concurrency=c,input_manifest=manifest,metrics={k:data[k] for k in keys},cache_report=data['cache_report'])
        if mode=='on':
            assert json.loads((block/'fpm-validation.json').read_text())['valid']
    grouped={}
    for c in [1,32,128]:
        pairs=[]
        for a,b in [(0,1),(3,2),(4,5)]:
            off=raw[f'{a}:{c}'];on=raw[f'{b}:{c}']
            assert off['input_manifest']==on['input_manifest']
            gap={k:(100*(on['metrics'][k]/off['metrics'][k]-1) if off['metrics'][k] else None) for k in keys}
            pairs.append(dict(off_block=a,on_block=b,delta_percent=gap))
        metrics={}
        for k in keys:
            off=[raw[f'{i}:{c}']['metrics'][k] for i in [0,3,4]]
            on=[raw[f'{i}:{c}']['metrics'][k] for i in [1,2,5]]
            gap=[p['delta_percent'][k] for p in pairs]
            metrics[k]=dict(off_mean=statistics.mean(off),off_stdev=statistics.stdev(off),on_mean=statistics.mean(on),on_stdev=statistics.stdev(on),paired_delta_percent_mean=statistics.mean(gap) if all(g is not None for g in gap) else None,paired_delta_percent_stdev=statistics.stdev(gap) if all(g is not None for g in gap) else None)
        grouped[c]=dict(metrics=metrics,pairs=pairs)
    result=dict(protocol=json.loads((root/'protocol.json').read_text()),raw_runs=raw,by_concurrency=grouped,
        limitations=['Three finite-request repetitions per mode; report observed variance, not a universal overhead bound.', 'FPM-on includes buffered external recording; no-consumer emission cost is not isolated.', 'Synthetic 8192-token prompts and fixed1024-token output; acceptance length is simulated2.49.'])
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Fixed 8K/1K SGLang FPM performance comparison','','Same node and image; ISL8192, OSL1024; three off/on pairs per concurrency.','', '| Concurrency | Output tok/s off (mean ± SD) | Output tok/s on (mean ± SD) | Paired throughput gap (mean ± SD) | Mean TPOT gap | TTFT p90 gap |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for c,g in grouped.items():
        m=g['metrics'];v=m['output_throughput']
        lines.append(f"| {c} | {v['off_mean']:.2f} ± {v['off_stdev']:.2f} | {v['on_mean']:.2f} ± {v['on_stdev']:.2f} | {v['paired_delta_percent_mean']:+.2f}% ± {v['paired_delta_percent_stdev']:.2f}% | {m['mean_tpot_ms']['paired_delta_percent_mean']:+.2f}% | {m['p90_ttft_ms']['paired_delta_percent_mean']:+.2f}% |")
    lines+=['','Gap is100*(on/off-1) for each pair, then averaged. A negative throughput gap is slower; a positive latency gap is slower.','',*['- '+x for x in result['limitations']]]
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main(Path(sys.argv[1]))
