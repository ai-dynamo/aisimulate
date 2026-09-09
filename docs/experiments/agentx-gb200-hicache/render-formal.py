# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the validated smoke DGD as a one-hour, persistent c48 AgentX run."""
from pathlib import Path
import sys
import yaml

HERE = Path(__file__).resolve().parent
def represent_string(dumper, value):
    style = '|' if '\n' in value else ("'" if value.lower() in {'y', 'n', 'yes', 'no', 'on', 'off'} else None)
    return dumper.represent_scalar('tag:yaml.org,2002:str', value, style=style)

yaml.SafeDumper.add_representer(str, represent_string)
documents = list(yaml.safe_load_all((HERE / 'deploy.yaml').read_text()))
dgd = documents[1]
dgd['spec']['env'].extend([
    {'name': 'SGLANG_MOONCAKE_CUSTOM_MEM_POOL', 'value': 'True'},
    {'name': 'MC_FORCE_MNNVL', 'value': '1'},
    {'name': 'MC_TE_METRIC', 'value': 'true'},
    {'name': 'NVSHMEM_REMOTE_TRANSPORT', 'value': 'none'},
    {'name': 'SGLANG_ENABLE_THINKING', 'value': '1'},
    {'name': 'SGLANG_REASONING_EFFORT', 'value': 'max'},
    {'name': 'UCX_LOG_LEVEL', 'value': 'info'},
])
for component in dgd['spec']['components']:
    pod = component['podTemplate']['spec']
    pod['preemptionPolicy'] = 'Never'
    pod['priorityClassName'] = 'aiperf-paper-rig-nonpreempting'
    pod['nodeSelector'] = {
        'nvidia.com/gpu.product': 'NVIDIA-GB200',
        'cloud.google.com/gke-nodepool': 'customer-gpu-w0e',
        'nvidia.com/gpu.clique': '9b7e5103-edf7-455e-975a-70622c68dd26.2',
    }
    pod.pop('affinity', None)
    if component['type'] in {'prefill', 'decode'}:
        # Existing SGLang loader: avoids asynchronous CPU-tensor weight copies
        # that stalled the default NVFP4 loader on two fresh GB200 nodes.
        pod['containers'][0]['args'].extend(['--load-format', 'runai_streamer'])
        # The HTTP startup probe still verifies model readiness. Once started,
        # use process-liveness probes: long AgentX prefill can delay canaries.
        for probe in ['livenessProbe', 'readinessProbe']:
            pod['containers'][0][probe] = {
                'tcpSocket': {'port': 9090}, 'periodSeconds': 10,
                'timeoutSeconds': 5, 'failureThreshold': 3,
            }
frontend = dgd['spec']['components'][0]['podTemplate']['spec']
frontend['serviceAccountName'] = 'agentx-formal-runner'
frontend['containers'][0]['resources']['limits']['cpu'] = '4'
frontend['containers'].append({
    'name': 'aiperf',
    'image': frontend['containers'][0]['image'],
    'command': ['python3', '-u', '/runner/formal-runner.py'],
    'resources': {'requests': {'cpu': '4', 'memory': '24Gi'}, 'limits': {'cpu': '8', 'memory': '48Gi'}},
    'securityContext': {'runAsUser': 0},
    'volumeMounts': [
        {'name': 'model-cache', 'mountPath': '/model-cache', 'readOnly': True},
        {'name': 'results', 'mountPath': '/results'},
        {'name': 'runner', 'mountPath': '/runner', 'readOnly': True},
    ],
})
frontend['volumes'] = list(frontend['volumes']) + [
    {'name': 'results', 'persistentVolumeClaim': {'claimName': 'agentx-gb200-results-hyperdisk'}},
    {'name': 'runner', 'configMap': {'name': 'agentx-formal-runner'}},
]
meta = lambda name: {'name': name, 'namespace': 'hzhou'}
resources = [
    {'apiVersion': 'v1', 'kind': 'PersistentVolumeClaim', 'metadata': meta('agentx-gb200-results-hyperdisk'),
     'spec': {'storageClassName': 'jegu-hyperdisk-balanced-rwo', 'accessModes': ['ReadWriteOnce'], 'resources': {'requests': {'storage': '50Gi'}}}},
    {'apiVersion': 'v1', 'kind': 'ServiceAccount', 'metadata': meta('agentx-formal-runner')},
    {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'Role', 'metadata': meta('agentx-formal-runner'),
     'rules': [
         {'apiGroups': [''], 'resources': ['pods', 'pods/log', 'events'], 'verbs': ['get', 'list', 'watch']},
         {'apiGroups': ['discovery.k8s.io'], 'resources': ['endpointslices'], 'verbs': ['get', 'list', 'watch']},
         {'apiGroups': ['nvidia.com'], 'resources': ['dynamoworkermetadatas'], 'verbs': ['get', 'list', 'watch', 'create', 'patch', 'update']},
         {'apiGroups': ['nvidia.com'], 'resources': ['dynamographdeployments'], 'resourceNames': ['agentx-glm52-hicache'], 'verbs': ['get', 'delete']},
         {'apiGroups': ['resource.nvidia.com'], 'resources': ['computedomains'], 'resourceNames': ['agentx-glm52-cd'], 'verbs': ['get', 'delete']},
     ]},
    {'apiVersion': 'rbac.authorization.k8s.io/v1', 'kind': 'RoleBinding', 'metadata': meta('agentx-formal-runner'),
     'subjects': [{'kind': 'ServiceAccount', 'name': 'agentx-formal-runner', 'namespace': 'hzhou'}],
     'roleRef': {'apiGroup': 'rbac.authorization.k8s.io', 'kind': 'Role', 'name': 'agentx-formal-runner'}},
    {'apiVersion': 'v1', 'kind': 'ConfigMap', 'metadata': meta('agentx-formal-runner'), 'data': {
        'formal-runner.py': (HERE / 'formal-runner.py').read_text(),
        'prepare-client.py': (HERE / 'prepare-client.py').read_text(),
        'stage-client-data.py': (HERE / 'stage-client-data.py').read_text(),
    }},
] + documents
sys.stdout.write('# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n')
sys.stdout.write('# SPDX-License-Identifier: Apache-2.0\n# Generated by render-formal.py; edit the source and regenerate.\n')
yaml.safe_dump_all(resources, sys.stdout, sort_keys=False)
