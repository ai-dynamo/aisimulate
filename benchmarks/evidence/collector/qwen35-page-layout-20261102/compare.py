"""Compare native page16 mappings with virtual pages grouped in manager528 blocks."""
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import subprocess
import time
import traceback
from pathlib import Path

import torch
from collector.vllm import collect_attn as collector
from vllm.v1.attention.backends.flashinfer import FlashInferBackend
from vllm.v1.worker.utils import select_common_block_size

root = Path('.')
identity = {'vllm': importlib.metadata.version('vllm'), 'torch': torch.__version__,
            'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
            'collector_sha256': hashlib.sha256(Path(collector.__file__).read_bytes()).hexdigest(),
            'manager_block': 528, 'kernel_block': select_common_block_size(528, [FlashInferBackend]),
            'selected_source': inspect.getsource(select_common_block_size), 'publish_eligible': False}
(root/'identity.json').write_text(json.dumps(identity, indent=2))
assert identity['vllm'] == '0.31.0' and 'B300' in identity['gpu'] and identity['kernel_block'] == 16
original = collector.create_kv_cache_and_block_mappings
mapping_records = []

def grouped_mapping(**kwargs):
    metadata = kwargs['common_attn_metadata']
    blocks_per_manager = 33
    lengths = metadata.seq_lens_cpu_upper_bound.tolist()
    manager_counts = [math.ceil(length / 528) for length in lengths]
    manager_total = sum(manager_counts)
    kwargs['num_blocks'] = max(kwargs['num_blocks'], (manager_total + 1) * blocks_per_manager)
    kwargs['randomize_blocks'] = False
    cache, history = original(**kwargs)
    assert history.numel() == 0, 'This harness only measures full causal prefill'
    managers = torch.randperm(manager_total, device=cache.device) + 1
    offset = 0
    for request, (length, count) in enumerate(zip(lengths, manager_counts)):
        page_count = math.ceil(length / 16)
        pages = (managers[offset:offset+count, None] * 33 + torch.arange(33, device=cache.device)).flatten()[:page_count]
        metadata.block_table_tensor[request, :page_count] = pages
        tokens = torch.arange(length, device=cache.device)
        begin = int(metadata.query_start_loc_cpu[request])
        metadata.slot_mapping[begin:begin+length] = pages[tokens//16] * 16 + tokens%16
        offset += count
    mapping_records.append({'lengths': lengths, 'manager_count': manager_total,
                            'cache_shape': list(cache.shape), 'kernel_block': 16,
                            'manager_block': 528, 'first_pages': metadata.block_table_tensor[0,:36].tolist()})
    return cache, history

rows = []
for batch, length in [(1,4096),(4,16384),(1,24576),(4,32768),(16,40960)]:
    for repeat in range(3):
        modes = ['native16','manager528_kernel16'] if repeat%2==0 else ['manager528_kernel16','native16']
        for mode in modes:
            record = {'batch':batch, 'length':length,'repeat':repeat,'mode':mode}
            collector.create_kv_cache_and_block_mappings = original if mode=='native16' else grouped_mapping
            path = root/f'{batch}-{length}-{repeat}-{mode}.csv'
            started = time.monotonic()
            try:
                collector.run_attention_torch(batch,length,4,1,256,False,True,perf_filename=str(path),device='cuda:0')
                torch.cuda.synchronize()
                record.update(status='passed',rows=list(csv.DictReader(path.open())))
            except Exception as error:
                record.update(status='failed',error=str(error),traceback=traceback.format_exc())
                traceback.print_exc()
            record['seconds']=time.monotonic()-started
            record['gpu_state']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,clocks.sm,temperature.gpu,power.draw,power.limit','--format=csv,noheader'],text=True)
            rows.append(record)
            (root/'results.json').write_text(json.dumps(rows,indent=2))
            (root/'mapping-records.json').write_text(json.dumps(mapping_records,indent=2))
            torch.cuda.empty_cache()
            print(json.dumps(record),flush=True)
assert all(row['status']=='passed' for row in rows)
