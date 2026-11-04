# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recollect the exact canonical GLM B52 subset using the signed collector worker."""
import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

MODEL='nvidia/GLM-5.2-NVFP4'
REF='c4b44d81f526b3c5299f7c15ccafe02d90b39229'
os.environ['COLLECTOR_MODEL_PATH']=MODEL
os.environ['AISIM_COLLECTOR_REF']=REF
ROOT=Path(__file__).resolve().parent/'source/python/aisimulate'
sys.path.insert(0,str(ROOT))
from collector.helper import create_test_case_id
from collector.op_plans import plan
from collector.provenance import case_plan_hash,collector_hash
from collector.sglang import collect_mla_module as collector

parser=argparse.ArgumentParser()
parser.add_argument('--mode',choices=['indexer','skip_indexer'],required=True)
parser.add_argument('--output',required=True)
a=parser.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
past=[1,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,393216,524288,655360,720896]
op='dsa_generation_module'+('_skip_indexer' if a.mode=='skip_indexer' else '')
full=plan(op,'sglang',103,include_long_history=True)
selected=[];dispatches=[]
for case,runners in zip(full.cases,full.dispatches,strict=True):
 if (case.model_path==MODEL and case.num_heads==8 and case.kv_cache_dtype=='fp8' and case.compute_dtype=='bfloat16' and case.gemm_type=='bfloat16' and case.batch_size==52 and case.prefix_len==0 and case.seq_len in past):
  selected.append(case);dispatches.append(runners)
assert len(selected)==22 and sorted(c.seq_len for c in selected)==past,(len(selected),selected)
subset=dataclasses.replace(full,cases=selected,dispatches=dispatches)
rows=subset.rows
assert len(rows)==1,rows
row=rows[0]
assert len(row[-1])==22
row_id=create_test_case_id(row,'run_mla_module_worker',f'sglang.{op}')
sourcehash=collector_hash(ROOT)
report={'publish_eligible':True,'collection_scope':'canonical plan subset, 22 B52 points, one indexer mode',
 'collector_ref':REF,'collector_hash':sourcehash,'case_plan_hash':case_plan_hash([row_id]),
 'attempted_executor_row_ids':[row_id],'requested_physical_case_ids':[create_test_case_id(list(c),op,f'sglang.{op}') for c in selected],
 'requested_points':[[c.batch_size,c.seq_len,c.prefix_len] for c in selected],
 'worker_row':row,'model_path':MODEL,'op':op,'mode':a.mode,
 'runtime':{'framework':'sglang','version':importlib.metadata.version('sglang'),'image':'lmsysorg/sglang:v0.5.21','image_digest':'sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469'},
 'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'running'}
(out/'attempted-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
t=time.perf_counter()
try:
 collector.run_mla_module_worker(*row,perf_filename=str(out/'dsa_generation_module_perf.txt'),device='cuda:0',skip_indexer=a.mode=='skip_indexer')
 measured=list(csv.DictReader((out/'dsa_generation_module_perf.txt').open()))
 assert len(measured)==22
 assert sorted(int(r['step']) for r in measured)==past
 assert all(int(r['batch_size'])==52 and r['model']==MODEL for r in measured)
 assert collector_hash(ROOT)==sourcehash
 report.update(status='passed',rows=measured)
except BaseException as exc:
 report.update(status='failed',error=str(exc));raise
finally:
 report['collection_seconds']=time.perf_counter()-t
 (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
