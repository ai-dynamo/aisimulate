# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render staged GB200 validation/formal configurations; no cluster mutations."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['validate', 'formal'], required=True)
parser.add_argument('--part', choices=['support', 'workload', 'all'], default='all')
args = parser.parse_args()
here = Path(__file__).resolve().parent
docs = list(yaml.safe_load_all(subprocess.check_output([sys.executable, str(here / 'render-formal.py')], text=True)))
dgd = next(d for d in docs if d['kind'] == 'DynamoGraphDeployment')
cd = next(d for d in docs if d['kind'] == 'ComputeDomain')
stage_number = 1 if args.stage == 'validate' else 2
run_id = f'agentx-gb200-20260909-stage{stage_number}-rdma-v1'
status_name = f'agentx-stage{stage_number}-guard-status'
guard_name = f'agentx-stage{stage_number}-guard-0909-v2'
env = {item['name']: item for item in dgd['spec']['env']}
for name, value in {
    'UCX_TLS': 'cuda_ipc,cuda_copy,rc',
    'UCX_IB_GID_INDEX': '3', 'UCX_RC_TIMEOUT': '600s',
    'UCX_KEEPALIVE_INTERVAL': '300s', 'NCCL_IB_DISABLE': '0',
    'NCCL_NVLS_ENABLE': '1', 'NVIDIA_GDRCOPY': '1',
    'NCCL_STORE_TIMEOUT': '7200', 'NCCL_DEBUG': 'INFO',
    'GLOO_SOCKET_IFNAME': 'eth0', 'NCCL_SOCKET_IFNAME': 'eth0',
    'NIXL_LOG_LEVEL': 'DEBUG' if args.stage == 'validate' else 'INFO',
    'SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT': '100000',
    'SGLANG_DISAGGREGATION_WAITING_TIMEOUT': '100000',
    'SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE': '100000',
}.items():
    env[name] = {'name': name, 'value': value}
dgd['spec']['env'] = list(env.values())
cd['spec']['numNodes'] = 3 if args.stage == 'validate' else 5
for component in dgd['spec']['components']:
    if component['type'] not in ['prefill', 'decode']:
        continue
    if component['type'] == 'decode':
        component['replicas'] = 1 if args.stage == 'validate' else 3
    template = component['podTemplate']
    annotations = template.setdefault('metadata', {}).setdefault('annotations', {})
    annotations['networking.gke.io/default-interface'] = 'eth0'
    annotations['networking.gke.io/interfaces'] = json.dumps(
        [{'interfaceName': 'eth0', 'network': 'default'}] +
        [{'interfaceName': f'rdma{i}', 'network': f'rdma-{i}'} for i in range(4)])
    main = template['spec']['containers'][0]
    for kind in ['requests', 'limits']:
        main['resources'].setdefault(kind, {}).update({
            f'networking.gke.io.networks/rdma-{i}': '1' for i in range(4)})
    original = main['command'] + main['args']
    main['command'] = ['bash', '-lc']
    main['args'] = ['set -euo pipefail; ulimit -l unlimited; ulimit -n 1048576; exec ' + shlex.join(original)]

frontend = dgd['spec']['components'][0]['podTemplate']['spec']
client = next(c for c in frontend['containers'] if c['name'] == 'aiperf')
client['command'] = ['python3', '-u', '/runner/stage_validation.py' if args.stage == 'validate' else '/runner/formal-runner.py']
client['env'] = [{'name': 'AGENTX_RUN_DIR', 'value': '/results/' + run_id}]
config = next(d for d in docs if d['kind'] == 'ConfigMap')
for file in ['staged_control.py', 'stage_watchdog.py', 'stage_validation.py']:
    config['data'][file] = (here / file).read_text()
role = next(d for d in docs if d['kind'] == 'Role')
role['rules'].append({'apiGroups': [''], 'resources': ['configmaps'],
                      'resourceNames': [status_name], 'verbs': ['get', 'patch']})
guard = {
    'apiVersion': 'batch/v1', 'kind': 'Job',
    'metadata': {'name': guard_name, 'namespace': 'hzhou'},
    'spec': {'backoffLimit': 0, 'activeDeadlineSeconds': 45000,
             'template': {'spec': {
                 'restartPolicy': 'Never', 'serviceAccountName': 'agentx-formal-runner',
                 'nodeSelector': {'kubernetes.io/arch': 'amd64'},
                 'containers': [{
                     'name': 'guard', 'image': 'python:3.12-alpine',
                     'command': ['python3', '-u', '/runner/stage_watchdog.py'],
                     'env': [{'name': 'GUARD_STATUS', 'value': status_name},
                             {'name': 'GPU_DEADLINE_SECONDS', 'value': '5400' if args.stage == 'validate' else '14400'}],
                     'resources': {'requests': {'cpu': '100m', 'memory': '64Mi'},
                                   'limits': {'cpu': '500m', 'memory': '128Mi'}},
                     'volumeMounts': [{'name': 'runner', 'mountPath': '/runner', 'readOnly': True}],
                 }],
                 'volumes': [{'name': 'runner', 'configMap': {'name': 'agentx-formal-runner'}}],
             }}}}
status = {'apiVersion': 'v1', 'kind': 'ConfigMap',
          'metadata': {'name': status_name, 'namespace': 'hzhou'}, 'data': {}}
support = [d for d in docs if d['kind'] not in ['ComputeDomain', 'DynamoGraphDeployment']] + [status, guard]
workload = [cd, dgd]
selected = support if args.part == 'support' else workload if args.part == 'workload' else support + workload

def represent_string(dumper, value):
    style = '|' if '\n' in value else ("'" if value.lower() in {'y', 'n', 'yes', 'no', 'on', 'off'} else None)
    return dumper.represent_scalar('tag:yaml.org,2002:str', value, style=style)

yaml.SafeDumper.add_representer(str, represent_string)
print('# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.')
print('# SPDX-License-Identifier: Apache-2.0')
print('# Generated by render-staged.py. Arm and verify the CPU guard BEFORE applying workload resources.')
yaml.safe_dump_all(selected, sys.stdout, sort_keys=False)
