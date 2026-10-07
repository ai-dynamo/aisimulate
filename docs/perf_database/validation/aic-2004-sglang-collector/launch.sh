#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 COLLECTOR_MEASURE_POWER=0
export FLASHINFER_WORKSPACE_BASE=/cache/flashinfer TRITON_CACHE_DIR=/cache/triton XDG_CACHE_HOME=/cache/xdg
export DG_JIT_CACHE_DIR=/cache/deepgemm CUDA_CACHE_PATH=/cache/cuda TORCHINDUCTOR_CACHE_DIR=/cache/inductor
export DIAG_IMAGE_DIGEST=sha256:9611bd4c5624b0e9e17829506188a12f17205f2083de0dd44d6c521733553a50
mkdir -p /cache/{flashinfer,triton,xdg,deepgemm,cuda,inductor}
python3 - <<'VERIFY'
import json,hashlib,pathlib
p=pathlib.Path('/task/target-008')
for f,s in json.loads((p/'manifest.json').read_text()).items(): assert hashlib.sha256((p/f).read_bytes()).hexdigest()==s,f
print('Full payload manifest passed',flush=True)
VERIFY
export PYTHONPATH=/task/target-008/payload/python/aisimulate COLLECTOR_REF=6a3f8fc6
cd "$PYTHONPATH"
python3 - <<'GROUPS' > /output/groups.txt
import json
for g in json.load(open('/task/target-008/plan.json'))['groups']: print(g['id'])
GROUPS
while read -r group; do
 timeout 900 python3 /task/target-008/collect_target.py --plan /task/target-008/plan.json --group "$group" --out "/output/$group" > "/output/$group.log" 2>&1
 python3 -c 'import json,sys;assert json.load(open(sys.argv[1]))["status"]=="passed"' "/output/$group/receipt.json"
done < /output/groups.txt
date -u > /output/FINISHED
