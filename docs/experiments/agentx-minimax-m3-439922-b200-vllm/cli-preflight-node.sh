#!/bin/bash
set -euo pipefail
ROOT=/home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911
LOCAL=/tmp/hongkuanz-minimax-fpm-${SLURM_JOB_ID:?}
OUT=$ROOT/build-$SLURM_JOB_ID
GPU_UUID=$(head -1 "$OUT/allocated-gpu-uuid.txt")
cp "$ROOT/benchmark.py" "$LOCAL/benchmark.py"
chmod a+r "$LOCAL/benchmark.py"
IMAGE=nvcr.io/nvidian/dynamo-dev/vllm-agentx:hzhou-fpm-b3563fc65a-amd64-20260911
docker run --rm --name minimax-cli-$SLURM_JOB_ID --gpus "device=$GPU_UUID" \
  -e SLURM_JOB_ID -e VLLM_PLUGINS= -e VLLM_USE_RUST_FRONTEND=0 -e VLLM_USE_V2_MODEL_RUNNER=0 \
  --mount type=bind,src="$LOCAL",dst=/work --entrypoint /opt/fpm/.venv/bin/python \
  "$IMAGE" /work/benchmark.py --preflight 2>&1 | tee "$OUT/cli-preflight.log"
docker run --rm --name minimax-event-$SLURM_JOB_ID --gpus "device=$GPU_UUID" \
  -e VLLM_PLUGINS= --entrypoint /opt/fpm/.venv/bin/python "$IMAGE" -c '
import time, torch
from types import SimpleNamespace
from vllm.v1.metrics.forward_pass_metrics import ForwardPassMetricsTimer
assert torch.cuda.device_count()==1
timer=ForwardPassMetricsTimer(1)
timer.start(SimpleNamespace(forward_pass_metrics_iteration_id=7,total_num_scheduled_tokens=1))
x=torch.ones(1024,device="cuda")
x.mul_(2)
timer.finish()
for _ in range(100):
    samples=timer.drain_samples()
    if samples:
        assert samples[0][0]==7 and samples[0][1]>0
        print("REAL_CUDA_EVENT_SMOKE_PASS",samples)
        break
    time.sleep(0.01)
else:
    raise RuntimeError("CUDA timing never became ready")
' 2>&1 | tee "$OUT/cuda-event-smoke.log"
