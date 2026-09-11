#!/bin/bash
# Read-only, local-workstation monitor. Terminal Slurm state is authoritative.
set -euo pipefail
JOB=${1:?job ID}
[[ "$JOB" =~ ^[0-9]+$ ]]
DEADLINE=$(date -u -d "${2:?UTC deadline}" +%s)
FRONTEND=hongkuanz@computelab-sc-01
ROOT=/home/scratch.hongkuanz_gpu/agentx-minimax-m3-results/job-$JOB
BATCH=/home/scratch.hongkuanz_gpu/minimax-m3-439922-vllm-20260911/benchmark-$JOB.log
while [ "$(date -u +%s)" -lt "$DEADLINE" ]; do
  STATUS=$(ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 "$FRONTEND" \
    "sacct -X -j $JOB -n -P --format=JobIDRaw,State,ExitCode,NodeList" | awk -F'|' -v job="$JOB" '$1 == job {print}')
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$STATUS"
  case "$STATUS" in
    *FAILED*|*CANCELLED*|*TIMEOUT*|*OUT_OF_MEMORY*|*NODE_FAIL*|*PREEMPTED*)
      ssh -o BatchMode=yes "$FRONTEND" "tail -70 $BATCH; test ! -f $ROOT/campaign-result.json || cat $ROOT/campaign-result.json"
      exit 1 ;;
    *COMPLETED*)
      ssh -o BatchMode=yes "$FRONTEND" "cat $ROOT/campaign-result.json; test ! -f $ROOT/on/fpm-validation.json || cat $ROOT/on/fpm-validation.json; test ! -f $ROOT/on/recorder.log || cat $ROOT/on/recorder.log"
      exit 0 ;;
  esac
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$FRONTEND" \
    "test ! -f $ROOT/state.json || cat $ROOT/state.json; test ! -f $ROOT/on/client.log || tail -3 $ROOT/on/client.log; test ! -f $ROOT/on/server.log || tail -2 $ROOT/on/server.log"
  sleep 60
done
printf 'MONITOR_WINDOW_END %s\n' "$(date -u +%FT%TZ)"
