# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded H100 resident-weight refusal and smaller native FP8 control."""
import argparse
import csv
import hashlib
import json
import time
import sys
from pathlib import Path

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent / "source/python/aisimulate"))
from collector.trtllm import collect_moe

parser=argparse.ArgumentParser()
parser.add_argument('--kind',choices=['bf16-refusal','fp8-control','bf16-control'],required=True)
parser.add_argument('--output',required=True)
a=parser.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
case=['bfloat16' if a.kind=='bf16-refusal' else 'fp8',[65536] if a.kind=='bf16-refusal' else [16384],8192,5120,22,512,1,1,False,'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8','balanced',0.0]
if a.kind=='bf16-control':
 case=['bfloat16',[16384],6144,2048,8,256,4,1,False,'zai-org/GLM-5.2','balanced',0.0]
r={'publish_eligible':False,'case':case,'collector_ref':'c4b44d81f526b3c5299f7c15ccafe02d90b39229','collector_sha256':hashlib.sha256(Path(collect_moe.__file__).read_bytes()).hexdigest(),'gpu':torch.cuda.get_device_name(),'capacity_bytes':torch.cuda.get_device_properties(0).total_memory}
t=time.perf_counter()
try:
 collect_moe.run_moe_torch(*case,perf_filename=str(out/'native.csv'),device='cuda:0')
 if a.kind=='bf16-refusal':raise AssertionError('expected resident-weight capacity refusal')
 r['rows']=list(csv.DictReader((out/'native.csv').open()));assert len(r['rows'])==1
 r['status']='passed'
except RuntimeError as exc:
 r['error']=str(exc)
 if a.kind=='bf16-refusal' and str(exc).startswith('FIXME(kernel-limit): does not fit: resident expert weights'):
  r['status']='passed-expected-capacity-refusal'
 else:
  r['status']='failed';raise
finally:
 r['collection_seconds']=time.perf_counter()-t
 r['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
 (out/'result.json').write_text(json.dumps(r,indent=2)+'\n')
