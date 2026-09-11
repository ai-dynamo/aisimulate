# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 SemiAnalysis LLC, Advanced Micro Devices, NVIDIA CORPORATION
# Serving/replay settings adapted from SemiAnalysisAI/InferenceX at
# 5c3e65cf4c59db9966a9b16eb0035702bc5cf692,
# benchmarks/single_node/agentic/minimaxm3_fp4_b200_mtp.sh and benchmarks/benchmark_lib.sh.
# Modified: pinned checkpoints/image, native FPM off/on, G2 disabled, Slurm artifacts.
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

JOB = os.environ['SLURM_JOB_ID']
BUNDLE = Path(__file__).resolve().parent
ROOT = Path('/scratch/agentx-minimax-m3-results') / f'job-{JOB}'
PY = '/opt/fpm/.venv/bin/python'
MODEL = 'nvidia/MiniMax-M3-NVFP4'
TARGET = Path('/scratch/models/hub/models--nvidia--MiniMax-M3-NVFP4/snapshots/901464083161bf8612a29ff7ad29914cd4ab4a85')
DRAFT = Path('/scratch/models/hub/models--Inferact--MiniMax-M3-EAGLE3-GQA/snapshots/96692486b5fd38ebf8fd2a5f6bb53427d30819a8')


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def stage(case, phase):
    save(ROOT/'state.json', dict(case=case, phase=phase, at_ns=time.time_ns()))
    print(case, phase, flush=True)


def stop(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=20)


def server_command(case):
    command = [PY, '-m', 'vllm.entrypoints.cli.main', 'serve', str(TARGET),
        '--served-model-name', MODEL, '--host', '0.0.0.0', '--port', '8888',
        '--tensor-parallel-size', '4', '--data-parallel-size', '1',
        '--gpu-memory-utilization', '0.9', '--block-size', '128',
        '--max-model-len', '1048576', '--language-model-only',
        '--enable-prefix-caching', '--no-enable-flashinfer-autotune',
        '--reasoning-parser', 'minimax_m3', '--tool-call-parser', 'minimax_m3',
        '--enable-auto-tool-choice', '--default-chat-template-kwargs', '{"thinking_mode":"enabled"}',
        '--attention-config', '{"backend":"FLASHINFER","use_trtllm_attention":true,"indexer_kv_dtype":"fp8","minimax_m3_msa_decode_backend":"triton"}',
        '--kv-cache-dtype', 'fp8', '--max-cudagraph-capture-size', '512',
        '--max-num-batched-tokens', '16384', '--stream-interval', '20',
        '--trust-remote-code', '--cpu-offload-gb', '0',
        '--speculative-config', json.dumps(dict(method='eagle3', model=str(DRAFT),
            num_speculative_tokens=3, attention_backend='FLASH_ATTN',
            rejection_sample_method='synthetic', synthetic_acceptance_length=2.78))]
    if case == 'on':
        command += ['--forward-pass-metrics-port', '20380',
                    '--forward-pass-metrics-worker-id', f'minimax-m3-{JOB}']
    return command


def preflight():
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    for case in ('off', 'on'):
        args = make_arg_parser(FlexibleArgumentParser()).parse_args(['--model', str(TARGET), *server_command(case)[5:]])
        assert args.tensor_parallel_size == 4 and args.data_parallel_size == 1
        assert args.kv_transfer_config is None and args.cpu_offload_gb == 0
        assert args.forward_pass_metrics_port == (20380 if case == 'on' else 0)
        assert args.scheduler_cls is None or 'dynamo' not in str(args.scheduler_cls).lower()
        print(case, 'NATIVE_CLI_G2_OFF_PREFLIGHT_PASS', flush=True)


def run_case(case, env):
    out = ROOT/case
    out.mkdir()
    local = Path(f'/tmp/minimax-m3-fpm-{JOB}-{case}')
    local.mkdir()
    processes, logs = {}, []
    def spawn(name, command):
        save(out/f'{name}-command.json', command)
        log = (out/f'{name}.log').open('w')
        logs.append(log)
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        processes[name] = proc
        return proc
    def health():
        for name, proc in processes.items():
            if name != 'client' and proc.poll() is not None:
                raise RuntimeError(f'{case} {name} exited {proc.returncode}')
    success = False
    try:
        stage(case, 'server_starting')
        if case == 'on':
            spawn('recorder', [PY, '/opt/fpm/record_fpm.py', '--ranks', '1', '--output', str(local/'fpm.jsonl')])
        spawn('server', server_command(case))
        deadline = time.monotonic()+3600
        while True:
            health()
            try:
                with urllib.request.urlopen('http://localhost:8888/health', timeout=5) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Model startup exceeded 3600 seconds')
            time.sleep(5)
        stage(case, 'smoke')
        request = urllib.request.Request('http://localhost:8888/v1/chat/completions',
            data=json.dumps(dict(model=MODEL, messages=[dict(role='user', content='Explain why the sky is blue.')], max_tokens=64)).encode(),
            headers={'Content-Type':'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            (out/'smoke-error.txt').write_bytes(error.read())
            raise
        (out/'smoke.json').write_bytes(body)
        assert json.loads(body)['usage']['completion_tokens'] > 0
        if case == 'on':
            time.sleep(2)
            metrics = [json.loads(line)['metrics'] for line in (local/'fpm.jsonl').open()]
            assert {m['dp_rank'] for m in metrics} == {0}
            assert any(m['scheduled_requests']['num_decode_requests'] for m in metrics)
        stage(case, 'aiperf')
        client = ['/opt/agentx-aiperf/bin/aiperf', 'profile', '--scenario', 'inferencex-agentx-mvp',
            '--url', 'http://localhost:8888', '--endpoint', '/v1/chat/completions', '--endpoint-type', 'chat',
            '--streaming', '--model', MODEL, '--tokenizer', str(TARGET), '--tokenizer-trust-remote-code',
            '--public-dataset', 'semianalysis_cc_traces_weka_062126', '--num-dataset-entries', '393',
            '--concurrency', '15', '--benchmark-duration', '3600', '--stats-interval', '30',
            '--random-seed', '42', '--failed-request-threshold', '0.1',
            '--trajectory-start-min-ratio', '0.25', '--trajectory-start-max-ratio', '0.75',
            '--warmup-requests-per-lane', '10', '--warmup-grace-period', '1800',
            '--trace-idle-gap-cap-seconds', '300', '--system-idle-gap-cap-seconds', '10',
            '--cache-bust', 'first_turn_prefix', '--extra-inputs', 'ignore_eos:true',
            '--use-server-token-count', '--no-gpu-telemetry', '--slice-duration', '1',
            '--artifact-dir', str(out/'aiperf')]
        proc = spawn('client', client)
        while proc.poll() is None:
            health()
            time.sleep(10)
        if proc.returncode:
            raise RuntimeError(f'{case} client exited {proc.returncode}')
        summary = json.loads((out/'aiperf/profile_export_aiperf.json').read_text())
        assert summary['metadata']['submission_valid'] and not summary['was_cancelled']
        assert not summary['error_summary'], summary['error_summary']
        success = True
    finally:
        stage(case, 'preserving_artifacts')
        stop(processes.get('recorder'))
        if (local/'fpm.jsonl').exists():
            shutil.copyfile(local/'fpm.jsonl', out/'fpm.jsonl')
            with (out/'fpm.jsonl').open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            (out/'fpm.sha256').write_text(digest+'  fpm.jsonl\n')
        for name in ('client', 'server'):
            stop(processes.get(name))
        for log in logs:
            log.close()
        save(out/'result.json', dict(success=success, finished_at_ns=time.time_ns()))
    if case == 'on':
        counts = dict(prefill_only=0, decode_only=0, mixed=0, no_requests=0)
        previous = None
        gaps = resets = 0
        for line in (out/'fpm.jsonl').open():
            metric = json.loads(line)['metrics']
            assert metric['dp_rank'] == 0 and metric['timing_scope'] == 'model_step_cuda'
            seq = metric['counter_id']
            if previous is not None:
                gaps += max(0, seq-previous-1)
                resets += seq <= previous
            previous = seq
            scheduled = metric['scheduled_requests']
            p, d = scheduled['num_prefill_requests'], scheduled['num_decode_requests']
            kind = 'mixed' if p and d else 'prefill_only' if p else 'decode_only' if d else 'no_requests'
            counts[kind] += 1
            assert math.isfinite(metric['wall_time']) and metric['wall_time'] >= 0
            if p or d:
                assert metric['wall_time'] > 0
            assert all(math.isfinite(value) and value >= 0 for value in scheduled.values())
        save(out/'fpm-validation.json', dict(counts=counts, records=sum(counts.values()),
            counter_gaps=gaps, counter_resets=resets, active_ranks=[0]))
        assert counts['decode_only'] and not (gaps or resets)
    stage(case, 'complete')


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    bundle = ROOT/'bundle'
    bundle.mkdir()
    for file in BUNDLE.iterdir():
        if file.is_file() and file.suffix in ('.py', '.sh', '.patch', '.sha256', '.json'):
            shutil.copy(file, bundle/file.name)
    checkpoints = json.loads((BUNDLE/'checkpoint-result.json').read_text())
    assert checkpoints['status'] == 'complete'
    assert checkpoints['models']['target']['revision'] == TARGET.name
    assert checkpoints['models']['draft']['revision'] == DRAFT.name
    for kind in ('target', 'draft'):
        model = checkpoints['models'][kind]
        for file in model['files']:
            assert (Path(model['snapshot'])/file['name']).stat().st_size == file['bytes']
    # Probe in a short-lived child so the controller holds no CUDA context.
    probe = '''import json,torch
assert torch.cuda.device_count()==4
devices=[]
for index in range(4):
    props=torch.cuda.get_device_properties(index)
    assert 'B200' in props.name
    devices.append(dict(index=index,name=props.name,uuid=str(props.uuid),bytes=props.total_memory))
print(json.dumps(devices))
'''
    allocated = json.loads(subprocess.check_output([PY, '-c', probe], text=True))
    save(ROOT/'allocated-gpus.json', allocated)
    (ROOT/'hardware.csv').write_text(subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,memory.total,driver_version,power.limit', '--format=csv'], text=True))
    (ROOT/'topology.txt').write_text(subprocess.check_output(['nvidia-smi','topo','-m'], text=True))
    save(ROOT/'allocation.json', {key:os.environ.get(key) for key in ['SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_JOB_NODELIST','CUDA_VISIBLE_DEVICES','SLURM_CPUS_PER_TASK']})
    shutil.copy('/opt/agentx-aiperf/freeze.txt', ROOT/'client-freeze.txt')
    save(ROOT/'protocol.json', dict(reference=439922, model=MODEL, target_revision=TARGET.name,
        draft_revision=DRAFT.name, gpus=4, tp=4, dp=1, expert_parallel_enabled=False, pp=1,
        concurrency=15, g2=False, cpu_offload_gb=0, kv_connector=None, duration_seconds=3600,
        spec_method='eagle3', speculative_tokens=3, synthetic_acceptance=2.78,
        stream_interval=20, frontend='native Python', scheduler='native vLLM', model_runner='V1',
        fpm_revision='b3563fc65ae0f5359802593d78e7ea097e1fed31',
        vllm_base='2cf0a6915ce544dc493a0990f2ea38d81601128a', fpm_publishers=1,
        cases=['off','on'], isolation='Four Slurm-assigned GPUs; not an exclusive whole-node allocation'))
    env = os.environ.copy()
    for name in ('VLLM_USE_SIMPLE_KV_OFFLOAD', 'VLLM_PREFIX_CACHE_RETENTION_INTERVAL', 'AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID'):
        env.pop(name, None)
    env.update(TZ='UTC', PYTHONUNBUFFERED='1', PYTHONNOUSERSITE='1', PYTHONHASHSEED='42',
        VLLM_PLUGINS='', VLLM_USE_RUST_FRONTEND='0', VLLM_USE_V2_MODEL_RUNNER='0',
        VLLM_ENGINE_READY_TIMEOUT_S='3600', VLLM_FLOAT32_MATMUL_PRECISION='high',
        VLLM_FLASHINFER_ALLREDUCE_BACKEND='trtllm', HF_HOME='/scratch/models',
        HF_HUB_DISABLE_IMPLICIT_TOKEN='1', XDG_CACHE_HOME=f'/tmp/minimax-m3-cache-{JOB}',
        AIPERF_DATASET_CONFIGURATION_TIMEOUT='1800', AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT='1800',
        AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES='0', AIPERF_HTTP_TCP_USER_TIMEOUT='900000',
        AIPERF_DATASET_MMAP_CACHE_DIR=f'/tmp/minimax-m3-mmap-{JOB}')
    mask = signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGCHLD})
    save(ROOT/'inherited-signal-mask.json', list(map(int,mask)))
    with (ROOT/'cli-preflight.log').open('w') as log:
        subprocess.run([PY, str(BUNDLE/'benchmark.py'), '--preflight'], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    for case in ('off', 'on'):
        run_case(case, env)
    save(ROOT/'campaign-result.json', dict(status='complete', completed_at_ns=time.time_ns()))


if __name__ == '__main__':
    if '--preflight' in sys.argv:
        preflight()
        raise SystemExit(0)
    def interrupted(*_):
        raise KeyboardInterrupt('Slurm termination')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as error:
        if ROOT.exists():
            save(ROOT/'campaign-result.json', dict(status='failed', error=str(error), at_ns=time.time_ns()))
        raise
