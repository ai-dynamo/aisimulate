# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU-only GPU-allocation deadline guard, without a results mount."""
import json
import os
import time
import urllib.error
from staged_control import DGD, api, cleanup, now, pods

status_name = os.environ['GUARD_STATUS']
limit = int(os.environ['GPU_DEADLINE_SECONDS'])
started = time.monotonic()
allocated_at = None
seen = False
dgd_uid = cd_uid = None


def status(state, **extra):
    value = {'utc': now(), 'state': state, **extra}
    print(json.dumps(value), flush=True)
    api('/api/v1/namespaces/hzhou/configmaps/' + status_name, 'PATCH',
        {'data': {'status.json': json.dumps(value)}})


status('ARMED', gpu_deadline_seconds=limit)
while True:
    try:
        obj = json.loads(api('/apis/nvidia.com/v1beta1/namespaces/hzhou/dynamographdeployments/' + DGD))
        if dgd_uid is not None and obj['metadata']['uid'] != dgd_uid:
            status('WORKLOAD_REPLACED')
            break
        dgd_uid = obj['metadata']['uid']
        if cd_uid is None:
            cd_uid = json.loads(api('/apis/resource.nvidia.com/v1beta1/namespaces/hzhou/computedomains/agentx-glm52-cd'))['metadata']['uid']
        seen = True
        gpu_pods = [p for p in pods() if p['spec'].get('nodeName') and any(
            int(c.get('resources', {}).get('requests', {}).get('nvidia.com/gpu', 0)) > 0
            for c in p['spec']['containers'])]
        if gpu_pods and allocated_at is None:
            allocated_at = time.monotonic()
            status('GPU_ALLOCATED', pods=[p['metadata']['name'] for p in gpu_pods])
        if allocated_at is not None and time.monotonic() - allocated_at > limit:
            status('DEADLINE', elapsed_gpu_seconds=time.monotonic() - allocated_at)
            cleanup(dgd_uid, cd_uid)
            break
        # A finite queue lifetime also prevents allocation after this guard exits.
        if time.monotonic() - started > 43200:
            status('QUEUE_DEADLINE')
            cleanup(dgd_uid, cd_uid)
            break
    except urllib.error.HTTPError as exc:
        if exc.code == 404 and seen:
            status('WORKLOAD_REMOVED')
            break
        if exc.code != 404:
            print('API unavailable, retaining deadline', repr(exc), flush=True)
    except Exception as exc:
        print('API unavailable, retaining deadline', repr(exc), flush=True)
    time.sleep(15)
