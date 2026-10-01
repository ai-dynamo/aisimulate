# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-local fixed-identity adapter around the real Dynamo replay runner.

AISimulate/Vizier owns candidate selection, ask/tell, scoring and timeout policy.
This wrapper preserves the previous fixed-arrival-day objective and engine
identity using canonical AIS performance configuration. It accepts the native
legal attention-DP mappings and records both original and scoring metrics.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback
import uuid

from aisimulate.sweeper.replay import ReplayOutputRequirements, ReplayReport, ReplaySpec, canonical_json


class ScenarioReplayError(RuntimeError):
    """A failed/incomplete candidate with a durable attempt-evidence path."""


class IncompleteReplayError(ValueError):
    """The virtual replay cap was reached before the fixed cohort completed."""


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _pin(payload, name, expected):
    previous = payload.get(name)
    if previous is not None and previous != expected:
        raise ValueError(f'fixed scenario field {name}: expected {expected!r}, received {previous!r}')
    payload[name] = expected


@dataclass(frozen=True)
class ScenarioRunnerFactory:
    """Pickleable settings; native imports and runner construction occur per worker."""

    attempts_dir: str
    scenario: int
    history_trace: str | None
    history_sha256: str | None
    trace_path: str
    dynamo_sha: str
    trace_sha256: str = 'd659a8ec3263b00d4690b5a323771dfa5c7b8dc0b91cfdcefe0d6ccc7fb5d143'
    expected_requests: int = 537600
    expected_output_tokens: int = 160730624
    expected_input_tokens: int = 1476413952
    observation_seconds: float = 86400.0
    model: str = 'deepseek-ai/DeepSeek-V3'
    hardware: str = 'h200_sxm'
    backend_versions: tuple[tuple[str, str], ...] = (('vllm', '0.24.0'), ('sglang', '0.5.14'))
    fmha_dtype: str = 'bfloat16'
    kv_cache_dtype: str = 'fp8'
    kv_bytes_per_token: int = 35136
    kv_transfer_bandwidth: float = 64.0
    kv_transfer_timing_mode: str = 'full_prompt'
    trace_block_size: int = 64
    context_length: int = 8192
    ttft_ms: float = 1000.0
    itl_ms: float = 50.0
    max_address_space_gib: float | None = None

    def _delegate_factory(self):
        from dynamo.replay.simulation import DynamoReplayRunnerFactory
        return DynamoReplayRunnerFactory(trace_block_size=self.trace_block_size)

    def capabilities(self):
        # This experiment adapter validates and lowers exactly these fixed
        # precision controls before calling Dynamo. Stock Dynamo's factory does
        # not yet advertise them on upstream canonical timing specifications.
        return replace(
            self._delegate_factory().capabilities(),
            supported_engine_model_controls=('fmha_quant_mode', 'kvcache_quant_mode'),
        )

    def create(self, worker_id: int):
        if self.max_address_space_gib is not None:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            requested = int(self.max_address_space_gib * 1024**3)
            if requested <= 0:
                raise ValueError('max_address_space_gib must be positive')
            if hard != resource.RLIM_INFINITY:
                requested = min(requested, hard)
            resource.setrlimit(resource.RLIMIT_AS, (requested, hard))
        base = self._delegate_factory().create(worker_id)
        from dataclasses import fields
        delegate = SummaryDynamoRunner(**{f.name: getattr(base, f.name) for f in fields(base)})
        return ScenarioRunner(self, worker_id, delegate)


class ScenarioRunner:
    def __init__(self, policy: ScenarioRunnerFactory, worker_id: int, delegate):
        self.policy = policy
        self.worker_id = worker_id
        self.delegate = delegate
        self._verified_trace_fingerprint = None
        self._verified_history_fingerprint = None

    def prepare_spec(self, spec: ReplaySpec) -> ReplaySpec:
        """Validate identity, then use Dynamo's shared canonical capacity lowering."""
        from dynamo.replay.config import lower_upstream_engine_args

        p = self.policy
        expected_hooks = set() if p.scenario == 1 else {'dynamo.router'}
        if p.scenario == 3:
            expected_hooks.add('dynamo.planner')
        if {h.provider for h in spec.runtime_hooks} != expected_hooks:
            raise ValueError(f'scenario {p.scenario} has incorrect runtime hooks')
        for hook in spec.runtime_hooks:
            if hook.provider == 'dynamo.router' and hook.config.get('router_mode') != 'kv_router':
                raise ValueError('Router experiment requires KV routing')
            if hook.provider == 'dynamo.planner':
                cfg = hook.config['planner_config']
                if cfg.get('load_predictor_warmup_trace') != p.history_trace:
                    raise ValueError('Planner must bootstrap from the frozen four-day history')
                if cfg.get('max_gpu_budget') != 256:
                    raise ValueError('Planner must have the full 256-GPU ceiling')
        if spec.goal.get('target') != 'goodput_per_gpu':
            raise ValueError('experiment requires the native goodput_per_gpu objective')
        sla = spec.goal.get('sla') or {}
        if sla.get('ttft_ms') != p.ttft_ms or sla.get('itl_ms') != p.itl_ms or sla.get('e2e_ms') is not None:
            raise ValueError('experiment requires TTFT 1000ms / request mean ITL 50ms')
        w = spec.workload
        paths = w.get('trace_paths') or ([w['trace_path']] if w.get('trace_path') else [])
        if len(paths) != 1 or Path(paths[0]).resolve() != Path(p.trace_path).resolve():
            raise ValueError('experiment requires the frozen single day-five trace')
        if w.get('trace_format', 'mooncake') != 'mooncake' or w.get('trace_block_size', p.trace_block_size) != p.trace_block_size:
            raise ValueError('experiment requires Mooncake-format 64-token source blocks')
        if w.get('arrival_speedup_ratio', 1.0) != 1.0 or w.get('replay_concurrency') is not None or spec.concurrency is not None:
            raise ValueError('experiment preserves open-loop trace timestamps without speedup')
        if w.get('max_sim_time_ms') != 90000000.0:
            raise ValueError('experiment preserves the 90000-second virtual replay cap')
        deployment = spec.backend_deployment
        version = dict(p.backend_versions).get(deployment.backend)
        if version is None or deployment.backend_version != version:
            raise ValueError(f'unexpected backend/version: {deployment.backend}/{deployment.backend_version}')
        if deployment.deployment_mode not in ('agg', 'disagg'):
            raise ValueError('experiment supports only agg/disagg')
        fields = {'agg': 'agg_engine_args'} if deployment.deployment_mode == 'agg' else {
            'prefill': 'prefill_engine_args', 'decode': 'decode_engine_args'}
        replacements = {}
        model_metadata = {}
        for role, field in fields.items():
            raw = getattr(deployment, field)
            if raw is None:
                raise ValueError(f'missing {role} engine arguments')
            args = deepcopy(raw)
            if any(key.startswith('aic_') for key in args):
                raise ValueError('expected canonical AIS compiler output, not legacy AIC fields')
            timing = args.get('timing_model') or {}
            if timing.get('type') != 'external' or timing.get('provider') != 'aic':
                raise ValueError('experiment requires native AIS external timing')
            identity = timing.get('config') or {}
            for key, value in {
                'backend': deployment.backend, 'backend_version': version,
                'model': p.model, 'system': p.hardware,
                'fmha_quant_mode': p.fmha_dtype, 'kvcache_quant_mode': p.kv_cache_dtype,
                'estimation_mode': 'op_level', 'fallback_policy': 'deny',
                'worker_type': 'aggregated' if role == 'agg' else role,
            }.items():
                _pin(identity, key, value)
            if identity.get('nextn') not in (None, 0) or identity.get('speculation') is not None:
                raise ValueError('experiment excludes speculation')
            if identity.get('dcp') not in (None, 1):
                raise ValueError('experiment preserves DCP=1')
            timing['config'] = identity
            args['timing_model'] = timing
            _pin(args, 'engine_type', deployment.backend)
            _pin(args, 'block_size', p.trace_block_size)
            _pin(args, 'enable_prefix_caching', True)
            _pin(args, 'enable_chunked_prefill', True)
            _pin(args, 'speedup_ratio', 1.0)
            _pin(args, 'decode_speedup_ratio', 1.0)
            _pin(args, 'worker_type', 'aggregated' if role == 'agg' else role)
            dp = int(identity['attention_dp'])
            _pin(args, 'dp_size', dp)
            _pin(args, 'tensor_parallel_size', int(identity['tp']))
            prefix = '' if role == 'agg' else role + '_'
            for canonical_key, parallel_key in (
                ('tp', 'tp'), ('attention_dp', 'attention_dp'),
                ('moe_tp_size', 'moe_tp'), ('moe_ep_size', 'moe_ep'),
            ):
                expected = int(deployment.parallel_config.get(prefix + parallel_key, 1))
                if int(identity.get(canonical_key) or 1) != expected:
                    raise ValueError(f'canonical {role} {canonical_key} differs from parallel mapping')
            if int(deployment.parallel_config.get(prefix + 'pp', 1)) != 1 or identity['pp'] != 1:
                raise ValueError('experiment keeps pipeline parallelism at 1')
            if args.get('num_gpu_blocks') is not None:
                raise ValueError('experiment requires auto-derived KV capacity')
            if args.get('native_host_offload') is not None or args.get('g3_offload') is not None:
                raise ValueError('experiment excludes host offload and speculation')
            if deployment.backend == 'vllm':
                _pin(args, 'max_model_len', p.context_length)
            else:
                # Current Dynamo SGLang binding rejects max_model_len. This
                # matches the old wrapper; max(ISL+OSL)=7599 makes 8192 nonbinding.
                if args.pop('max_model_len', p.context_length) != p.context_length:
                    raise ValueError('unexpected SGLang context length')
                sg = deepcopy(args.get('sglang') or {})
                _pin(sg, 'page_size', p.trace_block_size)
                _pin(sg, 'schedule_policy', 'fifo')
                _pin(sg, 'chunked_prefill_size', int(args['max_num_batched_tokens']))
                _pin(sg, 'max_prefill_tokens', max(16384, int(args['max_num_batched_tokens'])))
                _pin(sg, 'clip_max_new_tokens', 4096)
                _pin(sg, 'schedule_conservativeness', 1.0)
                args['sglang'] = sg
            if role in ('prefill', 'decode'):
                alias = args.pop('kv_transfer_bytes_per_token', None)
                if alias is not None and alias != p.kv_bytes_per_token:
                    raise ValueError('explicit transfer geometry must match the fixed FP8 MLA geometry')
                _pin(args, 'kv_bytes_per_token', p.kv_bytes_per_token)
                _pin(args, 'kv_transfer_bandwidth', p.kv_transfer_bandwidth)
                _pin(args, 'kv_transfer_timing_mode', p.kv_transfer_timing_mode)
            args = lower_upstream_engine_args(args)
            if 'timing_model' in args or 'ais_perf_config' not in args:
                raise ValueError('canonical AIS lowering failed')
            if any(key.startswith('aic_') for key in args):
                raise ValueError('legacy AIS identity survived lowering')
            replacements[field] = args
            model_metadata['aggregated' if role == 'agg' else role] = {
                'provider': 'aic', 'config': deepcopy(args['ais_perf_config'])
            }
        return replace(spec, backend_deployment=replace(deployment, **replacements,
                       performance_model_metadata=model_metadata),
                       workload=deepcopy(spec.workload), goal=deepcopy(spec.goal), adapters=deepcopy(spec.adapters))

    def _verify_trace(self):
        path = Path(self.policy.trace_path)
        stat = path.stat()
        fingerprint = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if self._verified_trace_fingerprint is not None:
            if fingerprint != self._verified_trace_fingerprint:
                raise ValueError('frozen trace changed during the sweep')
            return
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(8 * 1024**2), b''):
                digest.update(block)
        if digest.hexdigest() != self.policy.trace_sha256:
            raise ValueError('trace bytes do not match the fixed scenario SHA256')
        self._verified_trace_fingerprint = fingerprint

    def _verify_history(self):
        if self.policy.scenario != 3:
            return
        path = Path(self.policy.history_trace)
        st = path.stat()
        fingerprint = (str(path.resolve()), st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if self._verified_history_fingerprint is not None:
            if fingerprint != self._verified_history_fingerprint:
                raise ValueError('frozen history changed during search')
            return
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != self.policy.history_sha256:
            raise ValueError('history checksum does not match execution manifest')
        self._verified_history_fingerprint = fingerprint

    def _validate_summary(self, spec, summary):
        p = self.policy
        required = ('num_requests', 'completed_requests', 'total_input_tokens', 'total_output_tokens',
                    'duration_ms', 'wall_time_ms', 'gpu_hours', 'goodput_completed_requests',
                    'goodput_output_throughput_tok_s')
        for name in required:
            value = summary.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'missing/non-finite required native metric: {name}')
        if summary['num_requests'] != p.expected_requests or summary['completed_requests'] != p.expected_requests:
            raise IncompleteReplayError(
                f'incomplete replay at virtual stop: completed={summary["completed_requests"]}/'
                f'{p.expected_requests}, observed={summary["num_requests"]}; candidate is not rankable'
            )
        if summary['total_output_tokens'] != p.expected_output_tokens or summary['total_input_tokens'] != p.expected_input_tokens:
            raise ValueError('native token totals differ from the fixed trace')
        if summary['duration_ms'] <= 0 or not 0 <= summary['goodput_completed_requests'] <= p.expected_requests:
            raise ValueError('invalid native duration or SLA request count')
        d = spec.backend_deployment
        roles = [('agg', d.agg_engine_args, d.num_workers)] if d.deployment_mode == 'agg' else [
            ('prefill', d.prefill_engine_args, d.num_prefill_workers), ('decode', d.decode_engine_args, d.num_decode_workers)]
        gpus = 0
        for role, args, replicas in roles:
            width = int(args['tensor_parallel_size']) * int(args['dp_size'])
            if replicas < 1 or width < 1:
                raise ValueError('invalid worker count or width')
            native_role = 'decode' if role == 'agg' else role
            if summary.get(native_role + '_gpus_per_worker') != width:
                raise ValueError('native GPU width differs from effective engine identity')
            gpus += replicas * width
        duration_s = summary['duration_ms'] / 1000.0
        if p.scenario != 3 and not math.isclose(summary['gpu_hours'], gpus * duration_s / 3600, rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError('native GPU-hours differ from the fixed deployment geometry')
        average_gpus = summary['gpu_hours'] / (duration_s / 3600)
        if not 0 < average_gpus <= 256 + 1e-6:
            raise ValueError('invalid time-averaged GPU allocation')
        if p.scenario == 3 and summary.get('planner_total_ticks', 0) <= 0:
            raise ValueError('Planner scenario completed without Planner ticks')
        good_tokens = summary['goodput_output_throughput_tok_s'] * duration_s
        if not 0 <= good_tokens <= p.expected_output_tokens + 1e-5:
            raise ValueError('invalid native compliant-output count')
        return dict(total_gpus=gpus, initial_gpus=gpus, average_gpus=average_gpus, observation_seconds=p.observation_seconds,
                    compliant_output_tokens_reconstructed=good_tokens,
                    fixed_day_goodput_tok_s=good_tokens / p.observation_seconds,
                    fixed_day_goodput_per_gpu=good_tokens / p.observation_seconds / average_gpus,
                    request_sla_pass_fraction=summary['goodput_completed_requests'] / p.expected_requests,
                    normalized_day_gpu_hours=average_gpus * p.observation_seconds / 3600)

    def run(self, spec: ReplaySpec, *, output_requirements=None):
        details = bool(output_requirements and output_requirements.include_raw_report)
        if output_requirements is not None and output_requirements.capture_per_request:
            raise ValueError('per-request detail is disabled for this cohort runner')
        started = time.perf_counter()
        started_monotonic = time.monotonic()
        started_utc = _utc()
        import psutil
        attempt = Path(self.policy.attempts_dir) / f'worker-{self.worker_id}-{uuid.uuid4().hex}'
        attempt.mkdir(parents=True, exist_ok=False)
        record = dict(status='running', pid=os.getpid(), worker_id=self.worker_id,
                      process_create_time=psutil.Process().create_time(),
                      started_utc=started_utc, started_monotonic=started_monotonic,
                      started_perf_counter=started,
                      requested_spec_sha256=_digest(spec), dynamo_sha=self.policy.dynamo_sha,
                      experiment_adapter_engine_controls=list(self.policy.capabilities().supported_engine_model_controls),
                      trace_sha256=self.policy.trace_sha256,
                      objective_normalization='fixed_arrival_day_86400s')
        _write_json(attempt / 'requested-spec.json', json.loads(canonical_json(spec)))
        _write_json(attempt / 'attempt.json', record)
        try:
            effective = self.prepare_spec(spec)
            record['effective_spec_sha256'] = _digest(effective)
            _write_json(attempt / 'effective-spec.json', json.loads(canonical_json(effective)))
            self._verify_trace()
            self._verify_history()
            api_start = time.perf_counter()
            report = self.delegate.run(effective, output_requirements=ReplayOutputRequirements(include_raw_report=details, capture_telemetry=bool(output_requirements and output_requirements.capture_telemetry)))
            api_finished = time.perf_counter()
            native_report = report.metadata.get('native_report') or {}
            summary = dict(report.metadata.get('native_summary') or native_report.get('summary') or report.metrics)
            if 'planner_total_ticks' in report.metrics:
                summary['planner_total_ticks'] = report.metrics['planner_total_ticks']
            _write_json(attempt / 'native-report.json', native_report or {'summary': summary})
            _write_json(attempt / 'native-summary.json', summary)
            # Keep full simulation timing even if the virtual-cap completion
            # criterion rejects this candidate before it becomes rankable.
            record.update(replay_api_wall_seconds=api_finished-api_start,
                          preparation_wall_seconds=api_start-started,
                          native_replay_wall_seconds=summary.get('wall_time_ms', 0)/1000.0,
                          summary=summary)
            derived = self._validate_summary(effective, summary)
            record.update(status='completed', derived=derived)
            # Explicit task scoring contract: replace only goodput rates with
            # compliant arrival-cohort work / 86400s. Latencies, native duration
            # and native GPU-hours remain unchanged, including drain semantics.
            metrics = dict(report.metrics)
            metrics['goodput_output_throughput_tok_s'] = derived['fixed_day_goodput_tok_s']
            if 'goodput_request_throughput_rps' in metrics:
                metrics['goodput_request_throughput_rps'] = summary['goodput_completed_requests'] / self.policy.observation_seconds
            _write_json(attempt / 'scoring-metrics.json', metrics)
            metadata = dict(report.metadata)
            metadata.pop('native_report', None)
            metadata.pop('native_summary', None)
            record.update(finished_utc=_utc(), outer_wall_seconds=time.perf_counter()-started,
                          process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _write_json(attempt / 'attempt.json', record)
            metadata['scenario'] = dict(attempt_path=str(attempt), requested_spec_sha256=record['requested_spec_sha256'],
                effective_spec_sha256=record['effective_spec_sha256'], dynamo_sha=self.policy.dynamo_sha,
                trace_sha256=self.policy.trace_sha256, fixed_precision={'fmha':self.policy.fmha_dtype,'kv_cache':self.policy.kv_cache_dtype},
                replay_api_wall_seconds=record['replay_api_wall_seconds'], native_replay_wall_seconds=record['native_replay_wall_seconds'],
                outer_wall_seconds=record['outer_wall_seconds'], process_peak_rss_kib=record['process_peak_rss_kib'],
                rss_scope='worker-process lifetime high water; worker may serve multiple candidates', derived=derived,
                objective_normalization='fixed_arrival_day_86400s',
                native_metrics=dict(report.metrics), native_summary=summary)
            return ReplayReport(metrics=metrics, metadata=metadata)
        except Exception as exc:
            record.update(status='failed', finished_utc=_utc(), outer_wall_seconds=time.perf_counter()-started,
                          error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc(),
                          failure_category='virtual_cap_incomplete' if isinstance(exc,IncompleteReplayError) else 'runner_error',
                          process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _write_json(attempt / 'attempt.json', record)
            raise ScenarioReplayError(f'attempt_path={attempt}: {type(exc).__name__}: {exc}') from exc

    def close(self):
        self.delegate.close()


# Preserve summary evidence without collecting every Planner tick during search.
from dynamo.replay.simulation import DynamoReplayRunner


class SummaryDynamoRunner(DynamoReplayRunner):
    @staticmethod
    def _normalize_report(report, output_requirements):
        metrics, metadata = DynamoReplayRunner._normalize_report(report, output_requirements)
        if hasattr(report, 'summary'):
            metadata['native_summary'] = dict(report.summary)
        elif hasattr(report, 'trace_report'):
            metadata['native_summary'] = dict(report.trace_report)
        return metrics, metadata
