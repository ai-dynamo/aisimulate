import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
import time
import torch
from collector.network.collect_nccl import nccl_benchmark, runtime_nccl_version
parser=argparse.ArgumentParser();parser.add_argument('--ranks',type=int,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
result={'source_commit':os.environ['AISIM_COLLECTOR_REF'],'participating_gpus':args.ranks,'allocation_seconds_included':False,'nccl_version':runtime_nccl_version(),'calls':[],'gpus':[{'name':torch.cuda.get_device_name(i),'uuid':str(getattr(torch.cuda.get_device_properties(i),'uuid','unavailable'))} for i in range(torch.cuda.device_count())]}
def save():
 p=args.output/'result.json.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(args.output/'result.json')
assert torch.cuda.device_count()==args.ranks, 'native ranks must match visible devices'
result['binary_linkage']={name:subprocess.run(['ldd',shutil.which(name)],capture_output=True,text=True).stdout for name in ('all_gather_perf','all_reduce_perf','alltoall_perf','reduce_scatter_perf') if shutil.which(name)}
result['audit_harness_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
save();os.chdir(args.output)
for op in ('all_gather','alltoall','reduce_scatter','all_reduce'):
 for dtype in ('half','int8'):
  for size in (536870912,1073741824):
   call={'op':op,'dtype':dtype,'size_bytes':size,'participating_gpus':args.ranks,'status':'running'};result['pending_call']=call;save();started=time.perf_counter()
   try:nccl_benchmark(dtype,op,f'{size},{size+1},2',args.ranks,False);call['status']='passed'
   except Exception as error:call.update(status='failed',error=str(error))
   call['collection_seconds']=time.perf_counter()-started;result['calls'].append(call);result.pop('pending_call');save()
result['status']='passed' if all(x['status']=='passed' for x in result['calls']) else 'failed';save()
raise SystemExit(0 if result['status']=='passed' else 1)
