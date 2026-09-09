# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage only tokenizer assets and the public dataset on the private result PVC."""
import os
from pathlib import Path
import shutil

for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE']:
    os.environ.pop(key, None)

revision = '53e0691e21895a3863a606dfd12910c69eba94ab'
source = Path('/model-cache/models--nvidia--GLM-5.2-NVFP4/snapshots') / revision
target = Path('/results/hf/hub/models--nvidia--GLM-5.2-NVFP4/snapshots') / revision
target.mkdir(parents=True, exist_ok=True)
for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
             'merges.txt', 'tokenizer.model', 'chat_template.jinja',
             'generation_config.json']:
    if (source / name).is_file():
        shutil.copy2(source / name, target / name)
print('Tokenizer assets staged in private canonical HF cache; no weights copied', flush=True)

from datasets import load_dataset
dataset = load_dataset('semianalysisai/cc-traces-weka-062126', split='train')
assert len(dataset) == 393, len(dataset)
print('Public Weka dataset ready: 393 trajectories', flush=True)
