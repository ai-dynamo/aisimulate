"""Bounded real collector entrypoint; synthetic inputs, no timing/forward replacement."""
import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--plan', required=True)
p.add_argument('--group', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
group = next(g for g in json.loads(Path(a.plan).read_text())['groups'] if g['id'] == a.group)
receipt = {'group': group, 'status': 'started', 'cases': [],
           'measurement': 'production run_attention_torch, no observer/timing substitution',
           'accuracy_acceptance': 'NOT_EVALUATED', 'source_commit': os.getenv('COLLECTOR_REF'),
           'image_digest': os.getenv('DIAG_IMAGE_DIGEST'), 'job_id': os.getenv('SLURM_JOB_ID')}
def save():
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
save()
def clocks(label):
    fields='index,uuid,clocks.current.sm,clocks.current.memory,pstate,temperature.gpu,power.draw,power.limit,clocks_event_reasons.active,ecc.errors.uncorrected.volatile.total'
    result=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv'],capture_output=True,text=True)
    with (out/'gpu-clocks.jsonl').open('a') as f: f.write(json.dumps({'time':time.time(),'label':label,'rc':result.returncode,'stdout':result.stdout,'stderr':result.stderr})+'\n')
clocks('before-import')
try:
    import torch
    import sglang
    from collector.sglang import collect_mla_module as c
    from collector.helper import finalize_perf_files
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    assert sglang.__version__ == '0.5.14'
    assert torch.cuda.device_count() == 1
    assert torch.cuda.get_device_capability() == (10, 0)
    assert 'B200' in torch.cuda.get_device_name()
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    receipt['runtime'] = {'sglang': sglang.__version__, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'collector_sha256': hashlib.sha256(Path(c.__file__).read_bytes()).hexdigest()}
    max_tokens = max(x['b'] * (x['p'] + x['q'] if x['mode'] == 'context' else x['q'] + 1) for x in group['cases'])
    runner = c.load_model_runner(model_path=group['model'], head_num=group['heads'],
        kv_cache_dtype=group['kv'], attention_backend='dsa', dsa_prefill_backend='trtllm',
        device='cuda:0', gemm_type=group['gemm'], target_tp_size=group['tp'],
        max_total_tokens=max(4096, max_tokens + max(4096, max_tokens // 20)),
        chunked_prefill_size=8192)
    attn = runner.model.model.layers[0].self_attn
    receipt['module'] = {'class': type(attn).__name__, 'heads': attn.num_local_heads,
        'indexer': type(attn.indexer).__name__, 'pool': type(runner.token_to_kv_pool).__name__,
        'page_size': runner.token_to_kv_pool.page_size,
        'prefill_graph': str(runner.server_args.cuda_graph_config.prefill),
        'chunked_prefill_size': runner.server_args.chunked_prefill_size,
        'max_total_tokens': runner.max_total_num_tokens}
    original_skip, original_next = attn.skip_topk, attn.next_skip_topk
    original_init = ForwardBatch.init_new
    original_benchmark = c.benchmark_with_power

    @classmethod
    def observed_init(cls, *init_args, **init_kwargs):
        fb = original_init(*init_args, **init_kwargs)
        expected = case['b'] * (case['q'] if case['mode'] == 'context' else 1)
        positions = fb.positions.detach().cpu().reshape(case['b'], -1)
        current['native_batch'] = {
            'forward_mode': str(fb.forward_mode), 'input_ids_shape': list(fb.input_ids.shape),
            'input_ids_dtype': str(fb.input_ids.dtype),
            'num_token_non_padded_cpu': fb.num_token_non_padded_cpu,
            'positions_shape': list(fb.positions.shape),
            'position_ranges': [[int(row[0]), int(row[-1])] for row in positions],
            'out_cache_loc_shape': list(fb.out_cache_loc.shape),
            'unique_new_cache_locations': int(fb.out_cache_loc.unique().numel()),
            'seq_lens': fb.seq_lens.detach().cpu().tolist(), 'expected_new_tokens': expected,
        }
        save()
        assert fb.input_ids.numel() == fb.positions.numel() == fb.out_cache_loc.numel() == expected
        assert fb.num_token_non_padded_cpu == expected
        expected_start = case['p'] if case['mode'] == 'context' else case['q']
        expected_end = expected_start + (case['q'] - 1 if case['mode'] == 'context' else 0)
        assert current['native_batch']['position_ranges'] == [[expected_start, expected_end]] * case['b']
        return fb

    @contextlib.contextmanager
    def retain_benchmark(**kwargs):
        with original_benchmark(**kwargs) as result:
            current['benchmark_result'] = dict(result)
            yield result

    ForwardBatch.init_new = observed_init
    c.benchmark_with_power = retain_benchmark
    for case in group['cases']:
        current = {'case': case, 'status': 'started', 'started': time.time(),
                   'memory_before': {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()},
                   'dynamo_hook_count_before': len(torch._dynamo.convert_frame._bytecode_hooks)}
        receipt['cases'].append(current)
        save()
        attn.skip_topk, attn.next_skip_topk = original_skip, original_next
        c._SKIP_INDEXER_PASS = case['skip']
        print('START', case['id'], flush=True)
        count = c.run_attention_torch(runner,
            [(case['b'], case['q'], case['mode'] == 'context', case['p'])],
            head_num=group['heads'], test_layer=0, num_warmup=10, num_iterations=50,
            device='cuda:0', output_path=str(out), attn_type='dsa', model_path=group['model'],
            kv_cache_dtype=group['kv'], compute_dtype='bfloat16', gemm_type=group['gemm'],
            target_tp_size=group['tp'], dsa_prefill_backend='trtllm')
        current.update(status='passed' if count == 1 else 'failed', emitted_rows=count,
                       elapsed_seconds=time.time() - current['started'])
        gc.collect()
        torch.cuda.empty_cache()
        current['memory_after'] = {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}
        current['dynamo_hook_count_after'] = len(torch._dynamo.convert_frame._bytecode_hooks)
        save()
        print('COMPLETE', case['id'], current['status'], flush=True)
        clocks(case['id'])
    receipt['parquet_files'] = [str(x) for x in finalize_perf_files(out.glob('*_perf.txt'), delete_source=False)]
    receipt['status'] = 'passed' if all(x['status'] == 'passed' for x in receipt['cases']) else 'partial'
    save()
except BaseException as e:
    receipt['status'] = 'error'
    receipt['error'] = {'type': type(e).__name__, 'message': str(e), 'traceback': traceback.format_exc()}
    save()
    raise
