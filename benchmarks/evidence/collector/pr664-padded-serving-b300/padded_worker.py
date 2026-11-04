"""Bounded native load_batch parity and DSA module timing; never production data."""
import argparse
import csv
import hashlib
import inspect
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from collector.sglang import collect_mla_module as collector
from collector.sglang.runtime_compat import resolved_arg
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
from sglang.srt.model_executor.runner_utils.buffers import DecodeInputBuffers

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['indexer', 'skip_indexer'], required=True)
parser.add_argument('--variant', choices=['padded', 'exact-diagnostic'], required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()
out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
report = {'publish_eligible': False, 'mode': args.mode, 'variant': args.variant, 'native_load_batch_checks': []}
report['native_source_sha256'] = hashlib.sha256(inspect.getsource(DecodeCudaGraphRunner).encode()).hexdigest()
original_pad = collector._padded_dsa_decode_batch

def observed_pad(runner, fb, padded_bs):
    padded = original_pad(runner, fb, padded_bs)
    fill = runner.attn_backend.get_cuda_graph_seq_len_fill_value()
    b = DecodeInputBuffers.create(device=fb.input_ids.device, max_bs=padded_bs, max_num_token=padded_bs,
        hidden_size=runner.model.config.hidden_size, next_token_logits_buffer=torch.empty(0, device='cuda'),
        dtype=torch.bfloat16, dp_size=1, pp_size=1, is_encoder_decoder=False, require_mlp_tp_gather=False,
        seq_len_fill_value=fill, encoder_len_fill_value=0, num_tokens_per_req=1,
        cache_loc_dtype=fb.out_cache_loc.dtype, enable_mamba_track=False)
    registry = build_decode_registry(device=fb.input_ids.device, max_bs=padded_bs, max_num_token=padded_bs,
        seq_len_fill_value=fill, cache_loc_dtype=fb.out_cache_loc.dtype, register_global_num_tokens=False, source=b)
    native = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    native.dllm_uses_input_embeds = False
    native.ragged_verify_mode = False
    native.deepep_adapter = SimpleNamespace(replay=lambda: None)
    native.buffers = b
    native.buffer_registry = registry
    native.captured_req_width = 1
    native.capture_bs = [padded_bs]
    native.require_mlp_tp_gather = False
    native.enable_two_batch_overlap = False
    native.enable_pdmux = False
    native.model_runner = runner
    native._metadata_glue = None
    native.seq_len_fill_value = fill
    native.capture_forward_mode = fb.forward_mode
    native.is_encoder_decoder = False
    native.capture_hidden_mode = CaptureHiddenMode.FULL
    native._capture_graph_size = lambda **kw: kw['bs']
    native._global_num_tokens_for_graph = lambda _: None
    native._resolve_lora_variant = lambda _: None
    native._resolve_attention_variant = lambda _: None
    native._make_graph_key = lambda *items: items
    received = []
    native._replay_attn_backend = lambda: SimpleNamespace(init_forward_metadata_out_graph=received.append)
    # Force fresh replay metadata admission; the collector just prepared its own metadata.
    original_needs = fb.needs_forward_metadata_init
    fb.needs_forward_metadata_init = lambda: True
    try:
        DecodeCudaGraphRunner.load_batch(native, fb)
    finally:
        fb.needs_forward_metadata_init = original_needs
    assert native.raw_bs == fb.batch_size and native.bs == padded_bs
    assert len(received) == 1
    view = received[0]
    checks = {}
    for name in ['input_ids', 'positions', 'req_pool_indices', 'seq_lens', 'seq_lens_cpu']:
        a, expected = getattr(padded, name), getattr(view, name)
        checks[name] = torch.equal(a, expected)
        assert checks[name], name
    assert torch.equal(padded.out_cache_loc, b.out_cache_loc)
    assert padded.seq_lens_sum == view.seq_lens_sum
    report['native_load_batch_checks'].append({'real_batch': fb.batch_size, 'padded_batch': padded_bs,
        'seq_len_sentinel': fill, 'num_padding': view.num_padding, 'fields_equal': checks,
        'tail_seq_lens': padded.seq_lens_cpu[fb.batch_size:].tolist(),
        'tail_req_indices': padded.req_pool_indices[fb.batch_size:].tolist(),
        'scope': 'actual pinned load_batch, native buffers; metadata dispatch observed, not full model replay'})
    return padded

collector._padded_dsa_decode_batch = observed_pad
if args.variant == 'exact-diagnostic':
    collector._generation_cuda_graph_batch_size = lambda _runner, batch: batch
started = time.perf_counter()
try:
    collector.run_mla_module(attn_type='dsa', head_num=8, model_path='nvidia/GLM-5.2-NVFP4',
        kv_cache_dtype='fp8', compute_dtype='bfloat16', gemm_type='bfloat16', is_prefill=False,
        gpu_id=0, output_path=str(out), target_tp_size=8, dsa_prefill_backend='trtllm',
        skip_indexer=args.mode == 'skip_indexer', points=[(52,4096,0),(56,4096,0),(52,16384,0),(56,16384,0)])
    rows=[]
    for p in out.glob('*perf.txt'):
        rows.extend(csv.DictReader(p.open()))
    assert len(rows)==4, len(rows)
    if args.variant=='padded':
        assert len(report['native_load_batch_checks'])==2
    report.update(status='passed',rows=rows)
except Exception as exc:
    report.update(status='failed',error=str(exc))
    raise
finally:
    report['collection_seconds']=time.perf_counter()-started
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
