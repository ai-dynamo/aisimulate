# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check bounded review measurements and pair process-level repetitions."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)
args = parser.parse_args()
root = args.run_dir
ledger = json.loads((root/'run-ledger.json').read_text())
records = []
for path in sorted((root/'cases').glob('*/result.json')):
    result = json.loads(path.read_text())
    records.append({'name':path.parent.name, **result})
assert len(records) == len(ledger), (len(records), len(ledger))
failures = [r['name'] for r in records if r['status'] != 'passed']
measured=[]
for result in records:
    if result['status'] != 'passed':
        continue
    assert len(result['rows']) == len(result['shapes']), result['name']
    observed=set()
    for row in result['rows']:
        latency=float(row['latency'])
        assert math.isfinite(latency) and latency > 0, row
        key=(int(row['batch_size']),int(row['step']))
        assert key not in observed, (result['name'],key)
        observed.add(key)
        measured.append({'run':result['name'],'variant':result['variant'], 'mode':result['mode'],
                         'batch_size':key[0],'past_kv':key[1],'latency_ms':latency,
                         'diagnostic_dummy':result['diagnostic_dummy'],
                         'bucket_probe':result['bucket_probe'], 'kernel_source':row['kernel_source']})
    assert observed == {tuple(shape) for shape in result['shapes']}, result['name']

paired=[]
for mode in ['indexer','skip_indexer']:
    native=[r for r in measured if r['mode']==mode and not r['diagnostic_dummy'] and not r['bucket_probe']]
    keys=sorted({(r['batch_size'],r['past_kv']) for r in native})
    for batch,past in keys:
        item={'mode':mode,'batch_size':batch,'past_kv':past}
        for variant in ['baseline','collected','current']:
            values=[r['latency_ms'] for r in native if r['variant']==variant and r['batch_size']==batch and r['past_kv']==past]
            assert len(values) == (1 if variant=='current' else 2), (mode,batch,past,variant,values)
            item[variant+'_values_ms']=values
            item[variant+'_median_ms']=statistics.median(values)
        item['collected_over_baseline']=item['collected_median_ms']/item['baseline_median_ms']
        item['current_over_baseline']=item['current_median_ms']/item['baseline_median_ms']
        diagnostic=[r['latency_ms'] for r in measured if r['mode']==mode and r['diagnostic_dummy'] and r['batch_size']==batch and r['past_kv']==past]
        assert len(diagnostic)==1, (mode,batch,past,diagnostic)
        item['cached_latent_diagnostic_ms']=diagnostic[0]
        item['diagnostic_over_baseline']=diagnostic[0]/item['baseline_median_ms']
        paired.append(item)

telemetry=[]
with (root/'gpu-telemetry.csv').open() as stream:
    reader=csv.DictReader(stream, skipinitialspace=True)
    for row in reader:
        clean={key.strip():value.strip() for key,value in row.items()}
        try:
            stamp=datetime.strptime(clean['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone(timedelta(hours=-8))).timestamp()
            clean['epoch']=stamp
            telemetry.append(clean)
        except (KeyError, ValueError):
            continue
clock_summary=[]
for entry in ledger:
    samples=[r for r in telemetry if entry['start'] <= r['epoch'] <= entry['end']]
    item={'run':entry['name'],'samples':len(samples)}
    for column in ['clocks.current.sm [MHz]', 'power.draw [W]', 'temperature.gpu']:
        values=[]
        for sample in samples:
            try:
                values.append(float(sample[column].split()[0]))
            except (KeyError, ValueError):
                pass
        if values:
            item[column]={'min':min(values),'median':statistics.median(values),'max':max(values)}
    clock_summary.append(item)

bucket_comparison=[]
for mode in ['indexer','skip_indexer']:
    for past in [4096,16384]:
        by_batch={row['batch_size']:row['latency_ms'] for row in measured if row['bucket_probe'] and row['mode']==mode and row['past_kv']==past}
        assert set(by_batch)=={52,56}, (mode,past,by_batch)
        bucket_comparison.append({'mode':mode,'past_kv':past,'exact52_ms':by_batch[52],
                                  'exact56_ms':by_batch[56],'ratio56_over52':by_batch[56]/by_batch[52],
                                  'scope':'exact-shape module proxy; not actual serving padded graph'})
summary={'publish_eligible':False,'run_count':len(records),'passed':len(records)-len(failures),
         'failures':failures,'measurement_rows':len(measured),'paired':paired,'telemetry':clock_summary,
         'bucket_comparison':bucket_comparison,
         'ratios_by_mode':{mode:{'n':len(values),'median':statistics.median(values),'min':min(values),'max':max(values)}
            for mode in ['indexer','skip_indexer']
            if (values:=[row['collected_over_baseline'] for row in paired if row['mode']==mode])}}
(root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
with (root/'measurements.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(measured[0]))
    writer.writeheader()
    writer.writerows(measured)
print(json.dumps({k:v for k,v in summary.items() if k not in ['paired','telemetry']},indent=2))
