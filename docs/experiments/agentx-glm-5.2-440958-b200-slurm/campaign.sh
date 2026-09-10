#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
BUNDLE=/scratch/agentx-fpm-ab-20260909
OUT=/scratch/agentx-results/job-${SLURM_JOB_ID:?}
test ! -e "$OUT"
mkdir -p "$OUT"
cp "$BUNDLE"/{run-fpm.sh,campaign.sh,submit.sh,provenance.txt,check_fpm.py} "$OUT/"
cp /opt/fpm/{fpm-compat.patch,base.sha256} "$OUT/"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,power.limit --format=csv >"$OUT/hardware.csv"
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 8
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -c 'NVIDIA B200')" -eq 8
python3 /opt/fpm/validate.py
python3 -c 'import torch; assert torch.cuda.device_count()==8; assert all("B200" in torch.cuda.get_device_name(i) for i in range(8))'
CKPT=/scratch/models/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/53e0691e21895a3863a606dfd12910c69eba94ab
python3 - "$CKPT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
shards=set(json.loads((p/'model.safetensors.index.json').read_text())['weight_map'].values())
assert len(shards)==47, len(shards)
assert all((p/s).is_file() and (p/s).stat().st_size>0 for s in shards)
print('Verified existing checkpoint shards:',len(shards))
PY
for experiment_case in on; do
  printf '%s START %s\n' "$(date -Is)" "$experiment_case"
  bash "$BUNDLE/run-fpm.sh" "$experiment_case"
  printf '%s DONE %s\n' "$(date -Is)" "$experiment_case"
done
printf '%s CAMPAIGN_COMPLETE\n' "$(date -Is)"
