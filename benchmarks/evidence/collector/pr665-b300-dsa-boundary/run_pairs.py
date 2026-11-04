# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Alternating native collector runs; no performance database writes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path.cwd()
schedule = []
for mode in ['indexer', 'skip_indexer']:
    for index, variant in enumerate(['baseline', 'collected', 'collected', 'baseline']):
        schedule.append((f'{mode}-{index}-{variant}', variant, mode, []))
    schedule.append((f'{mode}-current', 'current', mode, []))
    schedule.append((f'{mode}-dummy-diagnostic', 'collected', mode, ['--diagnostic-dummy']))
    schedule.append((f'{mode}-bucket-probe', 'current', mode, ['--bucket-probe']))
(root/'schedule.json').write_text(json.dumps(schedule, indent=2)+'\n')
results=[]
for name, variant, mode, flags in schedule:
    env = dict(os.environ)
    source = root / 'sources' / variant / 'python' / 'aisimulate'
    env['PYTHONPATH'] = f'{source}:{source / "collector"}'
    command = [sys.executable, str(root/'paired_worker.py'), '--variant', variant,
               '--mode', mode, '--output', str(root/'cases'/name)] + flags
    started = time.time()
    with (root/f'{name}.log').open('w') as log:
        proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    record = dict(name=name, returncode=proc.returncode, start=started, end=time.time(), command=command)
    results.append(record)
    (root/'run-ledger.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(record), flush=True)
    if proc.returncode:
        print((root/f'{name}.log').read_text()[-12000:], flush=True)
        if len(results) <= 2:
            raise SystemExit(proc.returncode)
raise SystemExit(any(row['returncode'] for row in results))
