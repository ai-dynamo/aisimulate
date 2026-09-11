#!/bin/bash
set -euo pipefail
ROOT=/home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911
LOCAL=/tmp/hongkuanz-minimax-fpm-${SLURM_JOB_ID:?}
OUT=$ROOT/build-$SLURM_JOB_ID
mkdir -p "$LOCAL" "$OUT"
hostname > "$OUT/hostname.txt"
nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader > "$OUT/visible-gpus.csv"
GPU_UUID=$(awk -F ', ' -v wanted="${SLURM_JOB_GPUS:?}" '$1 == wanted {print $2}' "$OUT/visible-gpus.csv")
if [ -z "$GPU_UUID" ]; then
  GPU_UUID=$(awk -F ', ' '$2 ~ /^GPU-/ {print $2}' "$OUT/visible-gpus.csv")
fi
test -n "$GPU_UUID"
test "$(printf '%s\n' "$GPU_UUID" | wc -l)" -eq 1
printf '%s\n' "$GPU_UUID" > "$OUT/allocated-gpu-uuid.txt"
cp "$ROOT"/{Dockerfile,update-fpm.patch,parent.sha256,test_forward_pass_metrics.py,test_shm_broadcast.py,forward_pass_metrics_subscriber.py} "$LOCAL/"
PARENT=nvcr.io/nvidian/dynamo-dev/vllm-agentx@sha256:b58174af2daaf4d02c4845fc90dd3f18615fc7c3ef930fb852ed0d29e76f1d12
IMAGE=nvcr.io/nvidian/dynamo-dev/vllm-agentx:hzhou-fpm-b3563fc65a-amd64-20260911
docker pull "$PARENT" 2>&1 | tee "$OUT/pull.log"
docker buildx build --platform linux/amd64 --load --progress plain \
  --metadata-file "$OUT/build.json" -t "$IMAGE" "$LOCAL" 2>&1 | tee "$OUT/build.log"
docker run --rm --name minimax-fpm-tests-$SLURM_JOB_ID --gpus "device=$GPU_UUID" \
  -e VLLM_PLUGINS= --entrypoint /opt/fpm/.venv/bin/python "$IMAGE" \
  -m pytest /opt/fpm/tests/test_forward_pass_metrics.py \
  /opt/fpm/tests/test_shm_broadcast.py::test_message_queue_readiness_includes_overflow_payload \
  --confcutdir=/opt/fpm/tests -p no:cacheprovider -q 2>&1 | tee "$OUT/tests.log"
docker push "$IMAGE" 2>&1 | tee "$OUT/push.log"
docker image inspect "$IMAGE" > "$OUT/image-inspect.json"
SQSH=/home/scratch.hongkuanz_gpu/images/vllm-agentx-fpm-b3563fc65a-amd64.sqsh
test ! -e "$SQSH"
export ENROOT_CACHE_PATH=/home/scratch.hongkuanz_gpu/images/enroot-cache-vllm-fpm
enroot import -o "$SQSH" "dockerd://$IMAGE" 2>&1 | tee "$OUT/enroot-import.log"
sha256sum "$SQSH" > "$OUT/squashfs.sha256"
echo IMAGE_READY
