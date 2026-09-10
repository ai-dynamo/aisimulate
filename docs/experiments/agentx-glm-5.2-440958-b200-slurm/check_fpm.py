# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
count = decode_count = 0
for line in path.open():
    item = json.loads(line)
    metrics = item['metrics']
    assert item['received_at_ns'] > 0
    assert math.isfinite(metrics['wall_time']) and metrics['wall_time'] >= 0
    scheduled = metrics['scheduled_requests']
    if scheduled['num_decode_requests']:
        decode_count += 1
        assert scheduled['sum_decode_kv_tokens'] > 0
        assert scheduled['var_decode_kv_tokens'] >= 0
    count += 1
assert count and decode_count, 'No valid decode FPM records received'
print(json.dumps({'records': count, 'decode_records': decode_count, 'bytes': path.stat().st_size}))
