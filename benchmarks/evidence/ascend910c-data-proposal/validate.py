"""Validate the contributed artifact sample, without executing any NPU code."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
for filename, item in manifest['artifacts'].items():
    data = (root / filename).read_bytes()
    assert len(data) == item['bytes'], filename
    assert hashlib.sha256(data).hexdigest() == item['sha256'], filename
rows = pq.read_table(root / 'ascend_vector_perf.parquet').to_pylist()
assert len(rows) == 160
assert Counter(r['execution_mode'] for r in rows) == {'eager': 112, 'graph': 48}
keys = [(r['role'], r['dimensions'], r['tokens'], r['execution_mode'], r['kernel_source'], r['boundary_contract']) for r in rows]
assert len(set(keys)) == len(keys)
for row in rows:
    assert math.isfinite(row['latency_ms']) and row['latency_ms'] > 0
    assert math.isfinite(row['cv']) and row['cv'] >= 0
    assert row['sample_count'] > 0 and row['numerical_checks'] > 0
meta = yaml.safe_load((root / 'collection_meta.yaml').read_text())
table = meta['tables']['ascend_vector_perf']
assert table['rows'] == len(rows) == sum(c['rows'] for c in table['collections'])
print(json.dumps({'status': 'passed', 'rows': len(rows), 'modes': dict(Counter(r['execution_mode'] for r in rows)), 'roles': sorted(set(r['role'] for r in rows))}))
