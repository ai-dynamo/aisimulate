# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot, persistent AgentX runner, executed in the frontend sidecar."""
import datetime
import json
import os
from pathlib import Path
import ssl
import subprocess
import time
import urllib.request

ROOT = Path('/results/agentx-gb200-20260908-c48-attempt4')
ROOT.mkdir(parents=True, exist_ok=True)
MODEL = 'nvidia/GLM-5.2-NVFP4'
REVISION = '53e0691e21895a3863a606dfd12910c69eba94ab'
DGD = 'agentx-glm52-hicache'
API = 'https://kubernetes.default.svc'
SA = Path('/var/run/secrets/kubernetes.io/serviceaccount')
CONTEXT = ssl.create_default_context(cafile=str(SA / 'ca.crt'))


def api(path, method='GET'):
    request = urllib.request.Request(
        API + path, method=method,
        headers={'Authorization': 'Bearer ' + (SA / 'token').read_text().strip()},
    )
    with urllib.request.urlopen(request, context=CONTEXT, timeout=30) as response:
        return response.read()


def marker(name, **values):
    (ROOT / name).write_text(json.dumps({
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **values
    }, indent=2) + '\n')


def collect():
    try:
        pods = json.loads(api('/api/v1/namespaces/hzhou/pods?labelSelector=nvidia.com%2Fdynamo-graph-deployment-name%3D' + DGD))
        (ROOT / 'pods.json').write_text(json.dumps(pods, indent=2))
        (ROOT / 'events.json').write_bytes(api('/api/v1/namespaces/hzhou/events'))
        for pod in pods['items']:
            name = pod['metadata']['name']
            try:
                (ROOT / (name + '.log')).write_bytes(api('/api/v1/namespaces/hzhou/pods/' + name + '/log?container=main'))
                if any(status.get('restartCount', 0) > 0 for status in pod['status'].get('containerStatuses', []) if status['name'] == 'main'):
                    (ROOT / (name + '.previous.log')).write_bytes(api('/api/v1/namespaces/hzhou/pods/' + name + '/log?container=main&previous=true'))
            except Exception as exc:
                print('log collection failed', name, repr(exc), flush=True)
    except Exception as exc:
        print('resource collection failed', repr(exc), flush=True)


def cleanup():
    # Exact named resources only; PVC and namespace are deliberately retained.
    for path in [
        '/apis/resource.nvidia.com/v1beta1/namespaces/hzhou/computedomains/agentx-glm52-cd',
        '/apis/nvidia.com/v1beta1/namespaces/hzhou/dynamographdeployments/' + DGD,
    ]:
        try:
            api(path, 'DELETE')
        except Exception as exc:
            print('cleanup request failed', repr(exc), flush=True)


# An exclusive persistent marker prevents duplicate load after a pod restart.
# An interrupted run requires explicit operator review and a NEW run ID/PVC.
if (ROOT / 'STARTED.json').exists():
    if (ROOT / 'COMPLETE.json').exists() or (ROOT / 'FAILED.json').exists():
        cleanup()
    else:
        marker('INTERRUPTED.json', reason='Previous sidecar started; refusing duplicate replay')
        collect()
        cleanup()
    while True:
        time.sleep(60)

print('Waiting for P and D readiness and frontend model availability', flush=True)
prefill_url = None
while True:
    try:
        pods = json.loads(api('/api/v1/namespaces/hzhou/pods?labelSelector=nvidia.com%2Fdynamo-graph-deployment-name%3D' + DGD))
        leader = next(pod for pod in pods['items'] if '-prefill-ldr-' in pod['metadata']['name'] and pod['status'].get('podIP'))
        prefill_url = 'http://' + leader['status']['podIP'] + ':9090'
        for base_url in [prefill_url, 'http://' + DGD + '-decode:9090']:
            with urllib.request.urlopen(base_url + '/live', timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError('worker not ready')
        with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=10) as response:
            models = json.load(response)['data']
        if any(model['id'] == MODEL for model in models):
            break
    except Exception:
        pass
    time.sleep(15)

try:
    fd = os.open(ROOT / 'STARTED.json', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
except FileExistsError:
    raise SystemExit('Concurrent runner owns this persistent run ID')
with os.fdopen(fd, 'w') as file:
    json.dump({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'concurrency': 48, 'duration': 3600}, file)

client_config = ROOT / 'client-config.yaml'
command = [
    '/opt/agentx-aiperf/bin/aiperf', 'profile',
    '--scenario', 'inferencex-agentx-mvp', '--url', 'http://127.0.0.1:8000',
    '--endpoint', '/v1/chat/completions', '--endpoint-type', 'chat',
    '--model', MODEL, '--tokenizer', MODEL, '--tokenizer-revision', REVISION,
    '--tokenizer-trust-remote-code',
    '--public-dataset', 'semianalysis_cc_traces_weka_062126', '--num-dataset-entries', '393',
    '--concurrency', '48', '--benchmark-duration', '3600', '--random-seed', '42',
    '--trajectory-start-min-ratio', '0.25', '--trajectory-start-max-ratio', '0.75',
    '--warmup-requests-per-lane', '10', '--warmup-grace-period', '1800',
    '--trace-idle-gap-cap-seconds', '300', '--system-idle-gap-cap-seconds', '10',
    '--cache-bust', 'first_turn_prefix', '--streaming', '--extra-inputs', 'ignore_eos:true',
    '--use-server-token-count', '--no-gpu-telemetry', '--slice-duration', '1', '--stats-interval', '30',
    '--server-metrics', prefill_url + '/metrics',
    'http://' + DGD + '-decode:9090/metrics',
    '--artifact-dir', str(ROOT / 'aiperf'),
]
(ROOT / 'command.json').write_text(json.dumps(command, indent=2))
environment = os.environ.copy()
environment.update({
    'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
    'HF_HOME': '/results/hf', 'HF_HUB_CACHE': '/results/hf/hub',
    'HF_DATASETS_CACHE': '/results/hf/datasets',
    'XDG_CACHE_HOME': '/results/cache', 'TMPDIR': '/results/tmp',
    'AIPERF_DATASET_CONFIGURATION_TIMEOUT': '1800',
    'AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT': '1800',
    'AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES': '0',
    'AIPERF_HTTP_TCP_USER_TIMEOUT': '900000',
})
Path('/results/tmp').mkdir(exist_ok=True)
collect()
try:
    with (ROOT / 'aiperf-console.log').open('w') as output:
        stage_env = environment.copy()
        stage_env.pop('HF_HUB_OFFLINE', None)
        stage_env.pop('TRANSFORMERS_OFFLINE', None)
        subprocess.run(['/opt/agentx-aiperf/bin/python', '/runner/stage-client-data.py'],
                       env=stage_env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
        subprocess.run(['/opt/agentx-aiperf/bin/python', '/runner/prepare-client.py',
                        str(ROOT / 'command.json'), str(client_config)],
                       env=environment, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=120)
        result = subprocess.run(['/opt/agentx-aiperf/bin/aiperf', 'profile', '--config', str(client_config)],
                                env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=10800)
    marker('COMPLETE.json' if result.returncode == 0 else 'FAILED.json', returncode=result.returncode)
except BaseException as exc:
    marker('FAILED.json', error=repr(exc))
finally:
    collect()
    os.sync()
    cleanup()
while True:
    time.sleep(60)
