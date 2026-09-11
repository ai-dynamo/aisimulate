# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage immutable public checkpoints inside a CPU allocation, preserving HF blobs."""
import json
import os
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi, snapshot_download

ROOT = Path('/scratch/minimax-m3-439922-vllm-20260911')
CACHE = Path('/scratch/models/hub')
result = {'job_id': os.environ['SLURM_JOB_ID'], 'status': 'running', 'models': {}}


def save():
    (ROOT/'checkpoint-result.json').write_text(json.dumps(result, indent=2)+'\n')


try:
    infos = []
    for repo, revision in (
        ('nvidia/MiniMax-M3-NVFP4', '901464083161bf8612a29ff7ad29914cd4ab4a85'),
        ('Inferact/MiniMax-M3-EAGLE3-GQA', '96692486b5fd38ebf8fd2a5f6bb53427d30819a8'),
    ):
        info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=False)
        infos.append(dict(id=repo, sha=info.sha, siblings=[dict(rfilename=f.rfilename, size=f.size) for f in info.siblings]))
    required = sum(f.get('size', 0) for info in infos for f in info['siblings'])
    assert shutil.disk_usage(CACHE).free > required + 50*1024**3
    save()
    for kind, info in zip(('target', 'draft'), infos):
        print('DOWNLOAD_START', kind, info['id'], info['sha'], flush=True)
        path = Path(snapshot_download(info['id'], revision=info['sha'], cache_dir=CACHE,
                                      max_workers=8, token=False))
        files = []
        for item in info['siblings']:
            file = path/item['rfilename']
            size = file.stat().st_size
            if 'size' in item:
                assert size == item['size'], (file, size, item['size'])
            files.append({'name': item['rfilename'], 'bytes': size})
        index = path/'model.safetensors.index.json'
        weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
        assert weights and all((path/name).stat().st_size > 0 for name in weights)
        result['models'][kind] = dict(repo=info['id'], revision=info['sha'], snapshot=str(path),
                                     weight_shards=len(weights), files=files, bytes=sum(f['bytes'] for f in files))
        save()
        print('DOWNLOAD_COMPLETE', kind, str(path), flush=True)
    result.update(status='complete', completed_at_ns=time.time_ns())
    save()
except BaseException as error:
    result.update(status='failed', error=str(error), failed_at_ns=time.time_ns())
    save()
    raise
