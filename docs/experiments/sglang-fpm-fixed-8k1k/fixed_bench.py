# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Use the image's native SGLang benchmark with exact-length, unique token IDs."""
import hashlib
import json
import os
import random
from pathlib import Path

from sglang.benchmark import serving
from sglang.benchmark.datasets.common import DatasetRow

SEED = int(os.environ.get('FIXED_INPUT_SEED', '42'))
DP = int(os.environ.get('FIXED_DP_SIZE', '8'))
CORPUS = '''The inference service processes a sequence of tokens using attention and
expert layers. A scheduler admits requests and records latency, throughput, and
memory usage. Engineers compare repeated experiments on the same hardware with
identical input lengths and output limits. Each request contains a distinct input
sequence. This synthetic text is used only to construct a valid vocabulary pool.
Numbers include 17 29 43 71 113 257 1024. Code example: result = sum(values) / count.
The quick brown fox jumps over the lazy dog. North south east west red green blue.
'''


def dataset(args, tokenizer, model_id=None):
    pool = sorted(set(tokenizer.encode(CORPUS, add_special_tokens=False)) - set(tokenizer.all_special_ids))
    assert len(pool) >= 32
    rng = random.Random(SEED)
    rows = []
    digest = hashlib.sha256()
    prefixes = set()
    for i in range(args.num_prompts):
        ids = rng.choices(pool, k=8192)
        prefix = tuple(ids[:256])
        assert prefix not in prefixes
        prefixes.add(prefix)
        digest.update(json.dumps(ids, separators=(',', ':')).encode())
        rows.append(DatasetRow(prompt=ids, prompt_len=8192, output_len=1024,
                               extra_request_body={'routed_dp_rank': i % DP}))
    Path(args.output_file + '.inputs.json').write_text(json.dumps(dict(
        seed=SEED, requests=len(rows), input_tokens=8192, output_tokens=1024,
        dp_size=DP, unique_first_pages=len(prefixes), input_sha256=digest.hexdigest(),
        vocabulary_pool=pool), indent=2) + '\n')
    return rows


original_run = serving.run_benchmark
original_json_loads = serving.orjson.loads
observed_usage = {}


def observe_usage(data, *args, **kwargs):
    decoded = original_json_loads(data, *args, **kwargs)
    if isinstance(decoded, dict):
        meta = decoded.get('meta_info')
        if isinstance(meta, dict) and 'id' in meta and 'completion_tokens' in meta:
            observed_usage[meta['id']] = (meta.get('prompt_tokens'), meta['completion_tokens'])
    return decoded


serving.orjson.loads = observe_usage


def checked_run(args):
    result = original_run(args)
    errors = [e for e in result['errors'] if e]
    assert result['completed'] == args.num_prompts, (result['completed'], args.num_prompts)
    assert not errors, errors[:3]
    assert set(result['input_lens']) == {8192}, set(result['input_lens'])
    assert set(result['output_lens']) == {1024}, set(result['output_lens'])
    assert len(observed_usage) == args.num_prompts, (len(observed_usage), args.num_prompts)
    assert set(observed_usage.values()) == {(8192, 1024)}, set(observed_usage.values())
    Path(args.output_file + '.server-usage.json').write_text(json.dumps(observed_usage, indent=2)+'\n')
    assert result['total_input_tokens'] == args.num_prompts * 8192
    assert result['total_output_tokens'] == args.num_prompts * 1024
    assert result['cache_report']['total_cached_tokens'] == 0, result['cache_report']
    Path(args.output_file + '.validated').write_text('completed; exact ISL8192/OSL1024; zero request errors; zero cached prompt tokens\n')
    return result


serving.get_dataset = dataset
serving.run_benchmark = checked_run
if __name__ == '__main__':
    serving.cli_main()
