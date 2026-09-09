# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Configuration adapted from InferenceX fb85931b1edec09f9498509835a8c814bebe3c65;
# modified for HiCache-off and paired FPM recording. See THIRD_PARTY_NOTICES.md.
"""Two fresh-server DSv4 AgentX runs: FPM disabled, then enabled with capture."""
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
SCRATCH = Path('/scratch')
JOB = os.environ['SLURM_JOB_ID']
ROOT = SCRATCH / 'agentx-dsv4-results' / f'job-{JOB}'
CKPT = SCRATCH / 'models/hub/models--deepseek-ai--DeepSeek-V4-Pro/snapshots/b5968e9190ef611bbf34a7229255be88a0e937c1'
CLIENT = '/opt/agentx-aiperf/bin/aiperf'
MODEL = 'deepseek-ai/DeepSeek-V4-Pro'
IMAGE = 'nvcr.io/nvidian/dynamo-dev/sglang-agentx@sha256:f856a45537f82e1900ea7607edcbaa7f77fbb2e70220eae522d1d50d0046727e'


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def stop(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)
    # Children can outlive their process-group leader.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def interrupted(*_):
    raise KeyboardInterrupt('Slurm termination signal')


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    shutil.copytree(BUNDLE, ROOT / 'bundle', ignore=shutil.ignore_patterns('__pycache__'))
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version,power.limit', '--format=csv'], text=True)
    (ROOT / 'hardware.csv').write_text(hardware)
    names = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).splitlines()
    assert len(names) == 8 and all('B300' in n for n in names), names
    (ROOT / 'topology.txt').write_text(subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True))
    (ROOT / 'host-memory.txt').write_text(Path('/proc/meminfo').read_text())
    subprocess.run([sys.executable, '/opt/fpm/validate.py'], check=True)
    report = json.loads((SCRATCH / 'agentx-dsv4-pro-440845-checkpoint/download-result.json').read_text())
    save(ROOT / 'checkpoint-validation.json', report)
    manifest = json.loads((SCRATCH / 'agentx-dsv4-pro-440845-checkpoint/manifest.json').read_text())
    save(ROOT / 'checkpoint-manifest.json', manifest)
    shards = set(json.loads((CKPT / 'model.safetensors.index.json').read_text())['weight_map'].values())
    assert len(shards) == 64 and all((CKPT / s).stat().st_size > 0 for s in shards)
    assert report.get('status') == 'complete' and report['errors'] == [], report
    assert report['revision'] == CKPT.name and report['files_verified'] == 91
    with (ROOT / 'engine-freeze.txt').open('w') as out:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=out, check=True)
    shutil.copy('/opt/agentx-aiperf/freeze.txt', ROOT / 'client-freeze.txt')
    save(ROOT / 'protocol.json', dict(image=IMAGE, cases=['off', 'on'], model=MODEL, checkpoint_revision=CKPT.name,
        concurrency=32, duration_seconds=3600, hicache=False, fpm_rank_count=8, seed=42,
        reference_id=440845, engine_entrypoint='sglang.launch_server', router='sglang-router consistent_hashing dp-aware'))
    # Warm the same compilation cache for both cases; each case has a new engine/KV cache.
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', HF_HOME='/scratch/models', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
        SGLANG_CACHE_DIR=f'/tmp/dsv4-cache-{JOB}', XDG_CACHE_HOME=f'/tmp/dsv4-xdg-{JOB}',
        TORCH_CUDA_ARCH_LIST='10.0', SGLANG_TIMEOUT_KEEP_ALIVE='900',
        SGLANG_ENABLE_UNIFIED_RADIX_TREE='1', SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS='1',
        SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT='1', SGLANG_OPT_USE_JIT_NORM='1',
        SGLANG_OPT_USE_JIT_INDEXER_METADATA='1', SGLANG_OPT_USE_TOPK_V2='1',
        SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2='1', SGLANG_JIT_DEEPGEMM_FAST_WARMUP='1',
        SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS='1', SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND='1',
        SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK='8320',
        SGLANG_SIMULATE_ACC_LEN='2.49', SGLANG_SIMULATE_ACC_METHOD='match-expected',
        SGLANG_SIMULATE_ACC_TOKEN_MODE='real-draft-token',
        AIPERF_DATASET_CONFIGURATION_TIMEOUT='1800', AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT='1800',
        AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES='0', AIPERF_HTTP_TCP_USER_TIMEOUT='900000',
        AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID='true')
    for case in ['off', 'on']:
        run_case(case, env)
    save(ROOT / 'campaign-result.json', dict(status='complete', completed_at_ns=time.time_ns(), cases=['off', 'on']))


def run_case(case, env):
    out = ROOT / case
    out.mkdir()
    local = Path(f'/tmp/dsv4-fpm-{JOB}-{case}')
    local.mkdir(exist_ok=False)
    endpoint = f'ipc:///tmp/dsv4-fpm-{JOB}-{case}/metrics'
    processes = {}
    files = []
    timeline = dict(case=case, started_at_ns=time.time_ns())
    def spawn(name, command):
        save(out / f'{name}-command.json', command)
        handle = (out / f'{name}.log').open('w')
        files.append(handle)
        p = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        processes[name] = p
        return p
    def health():
        for name, p in processes.items():
            if name != 'client' and p.poll() is not None:
                raise RuntimeError(f'{case}: {name} exited {p.returncode}; see {out}')
    def ready(url, limit=2400):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            health()
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        return
            except Exception:
                pass
            time.sleep(5)
        raise TimeoutError(url)
    save(out / 'environment.json', {k:v for k,v in env.items() if k.startswith(('SGLANG_', 'AIPERF_')) or k=='TORCH_CUDA_ARCH_LIST'})
    success = False
    try:
        engine = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(CKPT),
            '--served-model-name', MODEL, '--trust-remote-code', '--host', '0.0.0.0', '--port', '8889',
            '--tp', '8', '--dp', '8', '--ep-size', '8', '--enable-dp-attention',
            '--tokenizer-worker-num', '8', '--enable-dp-attention-local-control-broadcast',
            '--enable-prefill-delayer', '--prefill-decode-interval', '20',
            '--incremental-streaming-output', '--stream-interval', '20', '--dist-init-addr', '127.0.0.1:10888',
            '--moe-a2a-backend', 'megamoe', '--enable-deepseek-v4-fp4-indexer', '--disable-flashinfer-autotune',
            '--attention-backend', 'dsv4', '--page-size', '256', '--disable-shared-experts-fusion',
            '--mem-fraction-static', '0.93', '--swa-full-tokens-ratio', '0.075',
            '--max-running-requests', '64', '--cuda-graph-max-bs-decode', '544',
            '--allow-auto-truncate', '--chunked-prefill-size', '65536',
            '--tool-call-parser', 'deepseekv4', '--reasoning-parser', 'deepseek-v4',
            '--chat-template', str(BUNDLE / 'deepseek_v4_thinking.jinja'),
            '--watchdog-timeout', '1800', '--speculative-algorithm', 'EAGLE',
            '--speculative-num-steps', '3', '--speculative-eagle-topk', '1', '--speculative-num-draft-tokens', '4',
            '--enable-metrics', '--enable-cache-report', '--skip-server-warmup', '--random-seed', '784605205']
        if case == 'on':
            engine += ['--enable-forward-pass-metrics', '--forward-pass-metrics-worker-id', f'dsv4-{JOB}',
                       '--forward-pass-metrics-ipc-name', endpoint]
            spawn('recorder', [sys.executable, str(BUNDLE / 'record_fpm.py'), 'record', str(local / 'fpm.jsonl'), '--endpoint', endpoint])
        spawn('server', engine)
        ready('http://localhost:8889/health')
        timeline['server_ready_at_ns'] = time.time_ns()
        router = [sys.executable, '-m', 'sglang_router.launch_router', '--worker-urls', 'http://localhost:8889',
            '--policy', 'consistent_hashing', '--request-id-headers', 'x-correlation-id', '--dp-aware',
            '--host', '0.0.0.0', '--port', '8888', '--prometheus-host', '127.0.0.1', '--prometheus-port', '18888',
            '--connect-timeout-secs', '900', '--request-timeout-secs', '14400', '--disable-health-check',
            '--retry-max-retries', '8', '--retry-initial-backoff-ms', '500', '--retry-max-backoff-ms', '10000',
            '--retry-backoff-multiplier', '2']
        spawn('router', router)
        ready('http://localhost:8888/health', 180)
        req = urllib.request.Request('http://localhost:8888/v1/chat/completions',
            data=json.dumps(dict(model=MODEL, messages=[dict(role='user', content='Say hello.')], max_tokens=16)).encode(),
            headers={'Content-Type': 'application/json', 'x-correlation-id': 'dsv4-smoke'})
        with urllib.request.urlopen(req, timeout=300) as response:
            (out / 'smoke.json').write_bytes(response.read())
        client = [CLIENT, 'profile', '--scenario', 'inferencex-agentx-mvp', '--url', 'http://localhost:8888',
            '--endpoint', '/v1/chat/completions', '--endpoint-type', 'chat', '--streaming',
            '--model', MODEL, '--tokenizer', str(CKPT), '--tokenizer-trust-remote-code',
            '--public-dataset', 'semianalysis_cc_traces_weka_062126', '--num-dataset-entries', '393',
            '--concurrency', '32', '--benchmark-duration', '3600', '--random-seed', '42',
            '--trajectory-start-min-ratio', '0.25', '--trajectory-start-max-ratio', '0.75',
            '--warmup-requests-per-lane', '10', '--warmup-grace-period', '1800',
            '--trace-idle-gap-cap-seconds', '300', '--system-idle-gap-cap-seconds', '10',
            '--cache-bust', 'first_turn_prefix', '--extra-inputs', 'ignore_eos:true',
            '--use-server-token-count', '--no-gpu-telemetry', '--slice-duration', '1', '--stats-interval', '30',
            '--server-metrics', 'http://localhost:8889/metrics', '--artifact-dir', str(out / 'aiperf')]
        timeline['client_started_at_ns'] = time.time_ns()
        save(out / 'timeline.json', timeline)
        proc = spawn('client', client)
        deadline = time.monotonic() + 7200
        while proc.poll() is None:
            health()
            if time.monotonic() > deadline:
                raise TimeoutError('Client exceeded setup+warmup+measurement deadline')
            time.sleep(10)
        if proc.returncode:
            raise RuntimeError(f'Client exited {proc.returncode}')
        timeline['client_finished_at_ns'] = time.time_ns()
        success = True
    finally:
        stop(processes.get('client'))
        stop(processes.get('router'))
        # Stop publishers first, then drain and close recorder before copying raw evidence.
        stop(processes.get('server'))
        if 'recorder' in processes:
            stop(processes['recorder'])
        for f in files:
            f.close()
        fpm = local / 'fpm.jsonl'
        if fpm.exists():
            shutil.copyfile(fpm, out / 'fpm.jsonl')
            digest = hashlib.file_digest((out / 'fpm.jsonl').open('rb'), 'sha256').hexdigest()
            (out / 'fpm.sha256').write_text(digest + '  fpm.jsonl\n')
        timeline.update(finished_at_ns=time.time_ns(), client_succeeded=success)
        save(out / 'timeline.json', timeline)
        shutil.rmtree(local)
    if case == 'on':
        with (out / 'fpm-validation.json').open('w') as stream:
            subprocess.run([sys.executable, str(BUNDLE / 'record_fpm.py'), 'validate', str(out / 'fpm.jsonl')], stdout=stream, check=True)
    print(f'{case} completed: {out}', flush=True)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as exc:
        if ROOT.exists():
            save(ROOT / 'campaign-result.json', dict(status='failed', error=str(exc), at_ns=time.time_ns()))
        raise
