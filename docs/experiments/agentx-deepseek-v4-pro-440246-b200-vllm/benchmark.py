# SPDX-License-Identifier: Apache-2.0
# Experiment configuration adapted from SemiAnalysisAI/InferenceX
# Copyright 2025 SemiAnalysis LLC, Advanced Micro Devices, NVIDIA CORPORATION
# 4552491d40b179c3323a3485c63090e5b8c964ad, benchmarks/single_node/agentic/dsv4_fp4_b200_vllm_mtp.sh.
# Changes: pinned Dynamo image with native FPM PR52061; 128GiB CPU KV/rank; paired capture.
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

JOB = os.environ['SLURM_JOB_ID']
BUNDLE = Path(__file__).resolve().parent
ROOT = Path('/scratch/agentx-vllm-results') / f'job-{JOB}'
MODEL = 'deepseek-ai/DeepSeek-V4-Pro'
CKPT = Path('/scratch/models/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1')
PY = '/opt/fpm/.venv/bin/python'
CLIENT = '/opt/agentx-aiperf/bin/aiperf'
G2_BYTES_PER_RANK = 128 * 1024**3
NUMA_NODES = []


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def stop(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=20)


def interrupted(*_):
    raise KeyboardInterrupt('Slurm termination')


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(BUNDLE, ROOT/'bundle')
    names = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    assert len(names)==8 and all('B200' in n for n in names),names
    buses=subprocess.check_output(['nvidia-smi','--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).splitlines()
    for bus in buses:
        domain,bus_id,device=bus.strip().lower().split(':')
        pci=f'{int(domain,16):04x}:{bus_id}:{device}'
        numa=int(Path(f'/sys/bus/pci/devices/{pci}/numa_node').read_text())
        assert numa>=0,(pci,numa)
        NUMA_NODES.append(numa)
    save(ROOT/'numa-nodes.json',NUMA_NODES)
    (ROOT/'hardware.csv').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version,power.limit','--format=csv'],text=True))
    (ROOT/'topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
    (ROOT/'host-memory.txt').write_text(Path('/proc/meminfo').read_text())
    shards=set(json.loads((CKPT/'model.safetensors.index.json').read_text())['weight_map'].values())
    assert len(shards)==64 and all((CKPT/s).stat().st_size>0 for s in shards)
    save(ROOT/'protocol.json',dict(reference=440246,model=MODEL,revision=CKPT.name,cases=['off','on'],
        gpus=8,tp=1,dp=8,ep=8,concurrency=64,g2_bytes_per_rank=G2_BYTES_PER_RANK,
        g2_total_bytes=8*G2_BYTES_PER_RANK,duration_seconds=3600,
        fpm_pr_revision='996fed467139edd7719a0063d57709b8a7fa6989',timing_scope='model_step_cuda',
        vllm_base='2cf0a6915ce544dc493a0990f2ea38d81601128a',scheduler='native AsyncScheduler',
        frontend='native vLLM Rust frontend plus vllm-router0.1.14',synthetic_acceptance=2.49))
    shutil.copy('/opt/agentx-aiperf/freeze.txt',ROOT/'client-freeze.txt')
    env=os.environ.copy()
    env.update(TZ='UTC',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',PYTHONHASHSEED='42',
        HF_HOME='/scratch/models',HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        VLLM_USE_V2_MODEL_RUNNER='1',VLLM_USE_RUST_FRONTEND='1',
        # The nightly bundles vLLM-Omni, whose auto-loaded registration plugin
        # replaces the native IPC structs with Omni variants. Rust expects the
        # native 16-field EngineCoreOutput. This text-only run needs no plugins.
        VLLM_PLUGINS='',
        VLLM_ENGINE_READY_TIMEOUT_S='3600',
        VLLM_PREFIX_CACHE_RETENTION_INTERVAL='32768',
        VLLM_FLOAT32_MATMUL_PRECISION='high',TORCH_CUDA_ARCH_LIST='10.0',
        PYTORCH_ALLOC_CONF='expandable_segments:True',
        XDG_CACHE_HOME=f'/tmp/vllm-agentx-cache-{JOB}',
        AIPERF_DATASET_CONFIGURATION_TIMEOUT='1800',AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT='1800',
        AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES='0',AIPERF_HTTP_TCP_USER_TIMEOUT='900000',
        AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID='1',
        AIPERF_DATASET_MMAP_CACHE_DIR=f'/tmp/vllm-agentx-mmap-cache-{JOB}')
    # Do not propagate a blocked SIGCHLD mask into the client's forkserver.
    inherited=signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGCHLD})
    save(ROOT/'inherited-signal-mask.json',list(map(int,inherited)))
    with (ROOT/'native-protocol-preflight.log').open('w') as log:
        subprocess.run([PY,str(BUNDLE/'protocol-preflight.py')],env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    for case in ('off','on'):
        run_case(case,env)
    save(ROOT/'campaign-result.json',dict(status='complete',completed_at_ns=time.time_ns()))


def server_command(case):
    kv=dict(kv_connector='SimpleCPUOffloadConnector',kv_role='kv_both',
        kv_connector_extra_config=dict(cpu_bytes_to_use_per_rank=G2_BYTES_PER_RANK,
            enable_cross_layers_blocks='true',lazy_offload=False))
    engine=[PY,'-m','vllm.entrypoints.cli.main','serve',str(CKPT),
        '--served-model-name',MODEL,'--host','0.0.0.0','--port','8889','--trust-remote-code',
        '--kv-cache-dtype','fp8','--block-size','256','--max-model-len','1048576',
        '--gpu-memory-utilization','0.9','--numa-bind','--enable-cumem-allocator',
        '--no-enable-flashinfer-autotune','--tokenizer-mode','deepseek_v4',
        '--tool-call-parser','deepseek_v4','--enable-auto-tool-choice','--reasoning-parser','deepseek_v4',
        '--attention-config',json.dumps(dict(backend='FLASHINFER_MLA_SPARSE_DSV4',use_prefill_query_quantization=True,use_fp4_indexer_cache=True)),
        '--speculative-config',json.dumps(dict(method='mtp',num_speculative_tokens=3,rejection_sample_method='synthetic',synthetic_acceptance_length=2.49)),
        '--no-disable-hybrid-kv-cache-manager','--disable-uvicorn-access-log',
        '--compilation-config',json.dumps(dict(cudagraph_mode='FULL_DECODE_ONLY',cudagraph_capture_sizes=list(range(4,65,4)),mode=0)),
        '--max-num-seqs','16','--tensor-parallel-size','1','--data-parallel-size','8',
        '--enable-expert-parallel','--enable-ep-weight-filter','--moe-backend','deep_gemm_mega_moe',
        '--prefill-schedule-interval','8','--long-prefill-token-threshold','512','--max-num-batched-tokens','8192',
        '--kv-transfer-config',json.dumps(kv)]
    if case=='on':
        engine+=['--forward-pass-metrics-port','20380','--forward-pass-metrics-worker-id',f'dsv4-vllm-{JOB}']
    if NUMA_NODES:
        engine+=['--numa-bind-nodes',*map(str,NUMA_NODES)]
    return engine


def run_case(case,env):
    out=ROOT/case;out.mkdir()
    local=Path(f'/tmp/vllm-agentx-fpm-{JOB}-{case}');local.mkdir()
    processes={};files=[]
    def spawn(name,command):
        save(out/f'{name}-command.json',command)
        log=(out/f'{name}.log').open('w');files.append(log)
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        processes[name]=p
        return p
    def health():
        for name,p in processes.items():
            if name!='client' and p.poll() is not None:
                raise RuntimeError(f'{case} {name} exited {p.returncode}')
    def ready(url,seconds=3600):
        end=time.monotonic()+seconds
        while time.monotonic()<end:
            health()
            try:
                with urllib.request.urlopen(url,timeout=5) as r:
                    if r.status==200:return
            except Exception:pass
            time.sleep(5)
        raise TimeoutError(url)
    success=False
    try:
        engine=server_command(case)
        if case=='on':
            spawn('recorder',[PY,'/opt/fpm/record_fpm.py','--ranks','8','--output',str(local/'fpm.jsonl')])
        spawn('server',engine)
        ready('http://localhost:8889/health')
        router=['/opt/agentx-aiperf/bin/vllm-router','--worker-urls','http://localhost:8889',
            '--policy','consistent_hash','--intra-node-data-parallel-size','8','--host','0.0.0.0','--port','8888',
            '--prometheus-host','127.0.0.1','--prometheus-port','18888','--request-timeout-secs','14400','--disable-retries']
        spawn('router',router);ready('http://localhost:8888/health',180)
        req=urllib.request.Request('http://localhost:8888/v1/chat/completions',
            data=json.dumps(dict(model=MODEL,messages=[dict(role='user',content='Say hello.')],max_tokens=16)).encode(),
            headers={'Content-Type':'application/json','X-Session-ID':'native-fpm-smoke'})
        try:
            with urllib.request.urlopen(req,timeout=300) as r:(out/'smoke.json').write_bytes(r.read())
        except urllib.error.HTTPError as e:
            (out/'smoke-error.txt').write_bytes(e.read())
            raise
        if case=='on':
            time.sleep(2)
            seen=set()
            for line in (local/'fpm.jsonl').open():
                seen.add(json.loads(line)['metrics']['dp_rank'])
            assert seen==set(range(8)),f'Native FPM publishers missing ranks: {seen}'
        client=[CLIENT,'profile','--scenario','inferencex-agentx-mvp','--url','http://localhost:8888',
            '--endpoint','/v1/chat/completions','--endpoint-type','chat','--streaming','--model',MODEL,
            '--tokenizer',str(CKPT),'--tokenizer-trust-remote-code',
            '--public-dataset','semianalysis_cc_traces_weka_062126','--num-dataset-entries','393',
            '--concurrency','64','--benchmark-duration','3600','--random-seed','42',
            '--trajectory-start-min-ratio','0.25','--trajectory-start-max-ratio','0.75',
            '--warmup-requests-per-lane','10','--warmup-grace-period','1800',
            '--trace-idle-gap-cap-seconds','300','--system-idle-gap-cap-seconds','10',
            '--cache-bust','first_turn_prefix','--extra-inputs','ignore_eos:true','--use-server-token-count',
            '--no-gpu-telemetry','--slice-duration','1','--stats-interval','30',
            '--server-metrics','http://localhost:8889/metrics','--artifact-dir',str(out/'aiperf')]
        p=spawn('client',client)
        while p.poll() is None:health();time.sleep(10)
        if p.returncode:raise RuntimeError(f'{case}: client exited {p.returncode}')
        summary=json.loads((out/'aiperf/profile_export_aiperf.json').read_text())
        assert summary['metadata']['submission_valid'],summary.get('metadata')
        assert not summary['was_cancelled'] and not summary['error_summary']
        success=True
    finally:
        stop(processes.get('recorder'))
        fpm=local/'fpm.jsonl'
        if fpm.exists():
            shutil.copyfile(fpm,out/'fpm.jsonl')
            with (out/'fpm.jsonl').open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
            (out/'fpm.sha256').write_text(digest+'  fpm.jsonl\n')
        for name in ('client','router','server'):stop(processes.get(name))
        for f in files:f.close()
        save(out/'result.json',dict(success=success,finished_at_ns=time.time_ns()))
    if case=='on':
        ranks=set();active=0
        for line in (out/'fpm.jsonl').open():
            m=json.loads(line)['metrics'];s=m['scheduled_requests']
            if s['num_prefill_requests'] or s['num_decode_requests']:
                assert m['wall_time']>0 and m['timing_scope']=='model_step_cuda'
                ranks.add(m['dp_rank']);active+=1
        assert ranks==set(range(8)),ranks
        save(out/'fpm-validation.json',dict(active_records=active,active_ranks=sorted(ranks)))
    print(case,'COMPLETE',flush=True)


if __name__=='__main__':
    if '--preflight' in sys.argv:
        NUMA_NODES.extend([0,0,0,0,1,1,1,1])
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        for case in ('off','on'):
            command=server_command(case)
            args=make_arg_parser(FlexibleArgumentParser()).parse_args(['--model',str(CKPT),*command[5:]])
            assert args.tensor_parallel_size==1 and args.data_parallel_size==8
            assert args.forward_pass_metrics_port==(20380 if case=='on' else 0)
            assert args.scheduler_cls is None or 'dynamo' not in str(args.scheduler_cls).lower()
            print(case, 'CLI_PREFLIGHT_PASS', flush=True)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:main()
    except BaseException as e:
        if ROOT.exists():save(ROOT/'campaign-result.json',dict(status='failed',error=str(e),at_ns=time.time_ns()))
        raise
