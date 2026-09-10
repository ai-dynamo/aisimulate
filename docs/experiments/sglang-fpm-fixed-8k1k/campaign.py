# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-length low/mid/high concurrency FPM experiment, paired across six blocks."""
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

BUNDLE = Path(__file__).resolve().parent
JOB = os.environ['SLURM_JOB_ID']
ROOT = Path('/scratch/sglang-fpm-fixed-8k1k-results') / f'job-{JOB}'
MODEL = 'deepseek-ai/DeepSeek-V4-Pro'
CKPT = Path('/scratch/models/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1')
CONCURRENCIES = [1, 32, 128]
COUNTS = {1: 24, 32: 256, 128: 1024}
MODES = ['off', 'on', 'on', 'off', 'off', 'on']
IMAGE = 'sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e'


def save(p, data):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(p)


def state(**kw):
    save(ROOT / 'state.json', dict(time_ns=time.time_ns(), **kw))
    print(json.dumps(kw), flush=True)


def stop(p):
    if p is None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait(timeout=10)


def gpu_snapshot(path):
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version,power.limit,power.draw,temperature.gpu,clocks.sm,clocks.mem', '--format=csv'], capture_output=True, text=True)
    path.write_text(r.stdout + r.stderr)


def environment():
    env = os.environ.copy()
    env.update(TZ='UTC', PYTHONUNBUFFERED='1', HF_HOME='/scratch/models', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        SGLANG_CACHE_DIR=f'/tmp/sglang-fixed-cache-{JOB}', XDG_CACHE_HOME=f'/tmp/sglang-fixed-xdg-{JOB}',
        TORCH_CUDA_ARCH_LIST='10.0', SGLANG_TIMEOUT_KEEP_ALIVE='900',
        SGLANG_ENABLE_UNIFIED_RADIX_TREE='1', SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS='1',
        SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT='1', SGLANG_OPT_USE_JIT_NORM='1',
        SGLANG_OPT_USE_JIT_INDEXER_METADATA='1', SGLANG_OPT_USE_TOPK_V2='1',
        SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2='1', SGLANG_JIT_DEEPGEMM_FAST_WARMUP='1',
        SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS='1', SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND='1',
        SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK='8320',
        SGLANG_SIMULATE_ACC_LEN='2.49', SGLANG_SIMULATE_ACC_METHOD='match-expected',
        SGLANG_SIMULATE_ACC_TOKEN_MODE='real-draft-token', FIXED_DP_SIZE='8')
    return env


def engine_args():
    return [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(CKPT),
        '--served-model-name', MODEL, '--trust-remote-code', '--host', '0.0.0.0', '--port', '8889',
        '--tp', '8', '--dp', '8', '--ep-size', '8', '--enable-dp-attention', '--tokenizer-worker-num', '8',
        '--enable-dp-attention-local-control-broadcast', '--enable-prefill-delayer',
        '--incremental-streaming-output', '--stream-interval', '20', '--dist-init-addr', '127.0.0.1:10888',
        '--moe-a2a-backend', 'megamoe', '--enable-deepseek-v4-fp4-indexer', '--disable-flashinfer-autotune',
        '--attention-backend', 'dsv4', '--page-size', '256', '--disable-shared-experts-fusion',
        '--mem-fraction-static', '0.93', '--swa-full-tokens-ratio', '0.075',
        '--max-running-requests', '256', '--cuda-graph-max-bs-decode', '544',
        '--chunked-prefill-size', '65536', '--watchdog-timeout', '1800',
        '--speculative-algorithm', 'EAGLE', '--speculative-num-steps', '3',
        '--speculative-eagle-topk', '1', '--speculative-num-draft-tokens', '4',
        '--enable-metrics', '--enable-cache-report', '--skip-server-warmup', '--random-seed', '784605205']


def run_block(index, mode):
    out = ROOT / f'block-{index}-{mode}'
    out.mkdir()
    local = Path(f'/tmp/sglang-fixed-fpm-{JOB}-{index}')
    local.mkdir()
    server = recorder = client = None
    handles = []
    env = environment()
    endpoint = f'ipc://{local}/metrics'
    def spawn(label, command, child_env=env):
        save(out / f'{label}-command.json', command)
        f = (out / f'{label}.log').open('w')
        handles.append(f)
        return subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=child_env, start_new_session=True)
    def check():
        for label, proc in [('server', server), ('recorder', recorder)]:
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f'block{index}: {label} exited {proc.returncode}')
    try:
        state(phase='loading', block=index, mode=mode)
        args = engine_args()
        if mode == 'on':
            args += ['--enable-forward-pass-metrics', '--forward-pass-metrics-worker-id', f'fixed-{JOB}-{index}', '--forward-pass-metrics-ipc-name', endpoint]
            recorder = spawn('recorder', [sys.executable, str(BUNDLE/'record_fpm.py'), 'record', str(local/'fpm.jsonl'), '--endpoint', endpoint])
        save(out/'environment.json', {k:v for k,v in env.items() if k.startswith(('SGLANG_', 'FIXED_')) or k=='TZ'})
        server = spawn('server', args)
        deadline = time.monotonic() + 1800
        while True:
            check()
            try:
                with urllib.request.urlopen('http://127.0.0.1:8889/health', timeout=5) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            if time.monotonic()>deadline:
                raise TimeoutError('Engine readiness')
            time.sleep(5)
        # Confirm that the native input-id interface does not add prompt tokens.
        payload = dict(input_ids=[1000+i%100 for i in range(8192)],
            sampling_params=dict(temperature=0, max_new_tokens=1024, ignore_eos=True), stream=False, routed_dp_rank=0)
        req = urllib.request.Request('http://127.0.0.1:8889/generate', data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=300) as response:
            smoke=json.load(response)
        save(out/'smoke-meta.json', smoke['meta_info'])
        assert smoke['meta_info']['prompt_tokens']==8192, smoke['meta_info']
        assert smoke['meta_info']['completion_tokens']==1024, smoke['meta_info']
        concs = CONCURRENCIES[(index // 2) % 3:] + CONCURRENCIES[:(index // 2) % 3]
        for conc in concs:
            for phase, count in [('warmup', max(8,2*conc)), ('measured', COUNTS[conc])]:
                name=f'c{conc}-{phase}'
                path=out/f'{name}.jsonl'
                child_env=env | {'FIXED_INPUT_SEED':str(10000+index//2 if phase=='warmup' else 42+index//2)}
                command=[sys.executable, str(BUNDLE/'fixed_bench.py'), '--backend','sglang', '--host','127.0.0.1', '--port','8889',
                    '--model',MODEL, '--tokenizer',str(CKPT), '--dataset-name','random', '--random-input-len','8192',
                    '--random-output-len','1024', '--random-range-ratio','1.0', '--num-prompts',str(count),
                    '--max-concurrency',str(conc), '--warmup-requests','0', '--tokenize-prompt', '--flush-cache',
                    '--cache-report', '--output-details', '--disable-tqdm', '--seed',child_env['FIXED_INPUT_SEED'], '--output-file',str(path)]
                state(phase=phase, block=index, mode=mode, concurrency=conc, requests=count)
                gpu_snapshot(out/f'{name}-gpu-before.csv')
                started=time.time_ns()
                client=spawn(name,command,child_env)
                deadline=time.monotonic()+1800
                while client.poll() is None:
                    check()
                    if time.monotonic()>deadline:
                        raise TimeoutError(name)
                    time.sleep(2)
                if client.returncode:
                    raise RuntimeError(f'{name} failed ({client.returncode})')
                client=None
                assert Path(str(path)+'.validated').exists()
                save(out/f'{name}-window.json',dict(started_ns=started,finished_ns=time.time_ns(),phase=phase,concurrency=conc))
                gpu_snapshot(out/f'{name}-gpu-after.csv')
    finally:
        if recorder is not None:
            stop(recorder)
        if (local/'fpm.jsonl').exists():
            shutil.copyfile(local/'fpm.jsonl',out/'fpm.jsonl')
            with (out/'fpm.jsonl').open('rb') as f:
                digest=hashlib.file_digest(f,'sha256').hexdigest()
            (out/'fpm.sha256').write_text(digest+'  fpm.jsonl\n')
        stop(client)
        stop(server)
        for f in handles:
            f.close()
        shutil.rmtree(local)
    if mode == 'on':
        with (out/'fpm-validation.json').open('w') as f:
            subprocess.run([sys.executable,str(BUNDLE/'record_fpm.py'),'validate',str(out/'fpm.jsonl')],stdout=f,check=True)
    state(phase='block_complete', block=index, mode=mode)


def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    shutil.copytree(BUNDLE,ROOT/'bundle',ignore=shutil.ignore_patterns('__pycache__'))
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).splitlines()
    assert len(names)==8 and all('B300' in x for x in names),names
    gpu_snapshot(ROOT/'hardware.csv')
    (ROOT/'topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
    subprocess.run([sys.executable,'/opt/fpm/validate.py'],check=True)
    for name in ['download-result.json','manifest.json']:
        shutil.copyfile(Path('/scratch/agentx-dsv4-pro-440845-checkpoint')/name,ROOT/name)
    verification=json.loads((ROOT/'download-result.json').read_text())
    assert verification['status']=='complete' and not verification['errors']
    shards=set(json.loads((CKPT/'model.safetensors.index.json').read_text())['weight_map'].values())
    assert len(shards)==64 and all((CKPT/s).is_file() for s in shards)
    save(ROOT/'protocol.json',dict(model=MODEL,revision=CKPT.name,image=IMAGE,concurrency=CONCURRENCIES,
        request_counts=COUNTS,isl=8192,osl=1024,modes=MODES,repetitions_per_mode=3,hicache=False,
        cache='unique token-id prompts and cache flush before every cell; assert zero cached tokens',
        native_endpoint='/generate',client='image-pinned sglang.benchmark.serving',speculative_acceptance=2.49))
    with (ROOT/'pip-freeze.txt').open('w') as f:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
    for i,mode in enumerate(MODES):
        run_block(i,mode)
    subprocess.run([sys.executable,str(BUNDLE/'compare.py'),str(ROOT)],check=True)
    state(phase='complete')


if __name__ == '__main__':
    def interrupted(*_):
        raise KeyboardInterrupt('Slurm signal')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        main()
    except BaseException as e:
        if ROOT.exists():
            state(phase='failed',error=str(e))
        raise
