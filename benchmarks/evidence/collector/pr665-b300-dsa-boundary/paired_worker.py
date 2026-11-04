# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local review evidence: native collector with bounded shapes and observed settings."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

import torch
from collector.sglang import collect_mla_module as collector
from collector.sglang.runtime_compat import resolved_arg

parser = argparse.ArgumentParser()
parser.add_argument('--variant', required=True)
parser.add_argument('--mode', choices=['indexer', 'skip_indexer'], required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--bucket-probe', action='store_true')
parser.add_argument('--diagnostic-dummy', action='store_true')
args = parser.parse_args()
output = Path(args.output)
output.mkdir(parents=True, exist_ok=False)
shapes = [(1, 4096), (1, 16384), (4, 4096), (4, 16384), (16, 4096),
          (16, 16384), (64, 1024), (64, 4096), (128, 1024), (128, 2048)]
if args.bucket_probe:
    shapes = [(52, 4096), (56, 4096), (52, 16384), (56, 16384)]
result = {'variant': args.variant, 'mode': args.mode, 'shapes': shapes,
          'diagnostic_dummy': args.diagnostic_dummy, 'bucket_probe': args.bucket_probe,
          'publish_eligible': False, 'env': {k: os.environ.get(k) for k in
          ['SGLANG_TEST_NUM_LAYERS', 'SGLANG_LOAD_FORMAT', 'CUDA_VISIBLE_DEVICES']},
          'packages': {k: importlib.metadata.version(k) for k in ['sglang', 'torch']},
          'collector_source_sha256': hashlib.sha256(Path(collector.__file__).read_bytes()).hexdigest()}
assert result['packages']['sglang'] == '0.5.21', result['packages']
assert torch.cuda.get_device_capability() == (10, 3), torch.cuda.get_device_name()
torch.manual_seed(20261102)
torch.cuda.manual_seed_all(20261102)
original_filter = collector._filter_cases_from_env
def bounded_filter(cases, **kwargs):
    kept = [case for case in cases if (case[0], case[1]) in shapes]
    assert {(case[0], case[1]) for case in kept} == set(shapes), kept
    return kept
collector._filter_cases_from_env = bounded_filter
original_load = collector.load_model_runner
def observed_load(*pos, **kwargs):
    result['runner_requested'] = kwargs
    runner = original_load(*pos, **kwargs)
    fields = ['context_length', 'context_len', 'max_total_num_tokens', 'max_running_requests',
              'max_req_len', 'page_size', 'mem_fraction_static']
    result['runner_observed'] = {k: getattr(runner, k, None) for k in fields}
    result['model_config_context_len'] = runner.model_config.context_len
    decode = resolved_arg(runner.server_args, 'cuda_graph_config').decode
    result['resolved_decode_graph'] = {k: getattr(decode, k, None) for k in ['backend', 'bs', 'max_bs']}
    fields += ['disable_cuda_graph', 'disable_cuda_graph_padding', 'cuda_graph_bs', 'cuda_graph_max_bs',
               'max_total_tokens', 'chunked_prefill_size', 'attention_backend',
               'dsa_decode_backend', 'dsa_prefill_backend', 'json_model_override_args']
    result['server_args'] = {k: resolved_arg(runner.server_args, k, None) for k in fields}
    result['kv_capacity_tokens'] = collector._kv_pool_capacity_tokens(runner)
    result['kv_page_size'] = collector.kv_pool_page_size(runner)
    (output / 'runner-settings.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    return runner
collector.load_model_runner = observed_load
original_decode = collector._run_decode
def observed_decode(*pos, **kwargs):
    if args.diagnostic_dummy:
        attention = kwargs['attention_module']
        latent = (getattr(attention, 'q_lora_rank', 1536) or 1536) + getattr(attention, 'kv_lora_rank', 512) + getattr(attention, 'qk_rope_head_dim', 64)
        cached = []
        def dummy(h, fb):
            if not cached:
                cached.append(torch.randn(h.shape[0], latent, dtype=h.dtype, device=h.device))
            return cached[0]
        kwargs['dummy_qkv_latent_func'] = dummy
    return original_decode(*pos, **kwargs)
collector._run_decode = observed_decode
started = time.perf_counter()
status = 1
try:
    kwargs = dict(attn_type='dsa', head_num=8, model_path='nvidia/GLM-5.2-NVFP4',
                  kv_cache_dtype='fp8', compute_dtype='bfloat16', gemm_type='bfloat16',
                  is_prefill=False, gpu_id=0, output_path=str(output),
                  target_tp_size=8, dsa_prefill_backend='trtllm',
                  skip_indexer=args.mode == 'skip_indexer')
    if args.variant != 'baseline':
        kwargs['inner_manifest'] = tuple((b, s, 0) for b, s in shapes)
    collector.run_mla_module(**kwargs)
    rows = []
    for path in output.glob('*perf.txt'):
        with path.open() as stream:
            rows.extend(csv.DictReader(stream))
    result['rows'] = rows
    assert len(rows) == len(shapes), (len(rows), len(shapes))
    result['status'] = 'passed'
    status = 0
except Exception as error:
    result.update(status='failed', error=str(error), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    result['elapsed_seconds'] = time.perf_counter() - started
    (output / 'result.json').write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps({'variant': args.variant, 'mode': args.mode, 'status': result['status'],
                      'seconds': result['elapsed_seconds']}), flush=True)
raise SystemExit(status)
