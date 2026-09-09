# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Namespaced API helpers for the bounded staged experiment."""
import datetime
import json
from pathlib import Path
import ssl
import urllib.error
import urllib.request

DGD = 'agentx-glm52-hicache'
CD = 'agentx-glm52-cd'
SA = Path('/var/run/secrets/kubernetes.io/serviceaccount')
TLS = ssl.create_default_context(cafile=str(SA / 'ca.crt'))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def api(path, method='GET', body=None):
    request = urllib.request.Request(
        'https://kubernetes.default.svc' + path,
        data=None if body is None else json.dumps(body).encode(), method=method,
        headers={'Authorization': 'Bearer ' + (SA / 'token').read_text().strip(),
                 'Content-Type': 'application/merge-patch+json' if method == 'PATCH' else 'application/json'},
    )
    with urllib.request.urlopen(request, context=TLS, timeout=20) as response:
        return response.read()


def pods():
    return json.loads(api('/api/v1/namespaces/hzhou/pods?labelSelector=nvidia.com%2Fdynamo-graph-deployment-name%3D' + DGD))['items']


def save(root, name, data):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def collect(root):
    try:
        items = pods()
        save(root, 'pods.json', items)
        (Path(root) / 'events.json').write_bytes(api('/api/v1/namespaces/hzhou/events'))
        for pod in items:
            name = pod['metadata']['name']
            for container in pod['spec']['containers']:
                cname = container['name']
                try:
                    data = api('/api/v1/namespaces/hzhou/pods/' + name + '/log?container=' + cname)
                    (Path(root) / (name + '.' + cname + '.log')).write_bytes(data)
                except Exception as exc:
                    print('log unavailable', name, cname, repr(exc), flush=True)
    except Exception as exc:
        print('snapshot unavailable', repr(exc), flush=True)


def cleanup(dgd_uid=None, cd_uid=None):
    # Exact resource names only. Never delete a PVC, node, or namespace.
    for path, uid in [('/apis/resource.nvidia.com/v1beta1/namespaces/hzhou/computedomains/' + CD, cd_uid),
                      ('/apis/nvidia.com/v1beta1/namespaces/hzhou/dynamographdeployments/' + DGD, dgd_uid)]:
        try:
            api(path, 'DELETE', None if uid is None else {'preconditions': {'uid': uid}})
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                print('cleanup failed', path, repr(exc), flush=True)
