"""Self-authored consumer compatibility harness; no AISimulate or Dynamo mocks.

Execute with exact upstream Dynamo Python modules on PYTHONPATH and candidate AIS SDK.
All arithmetic observations are synthetic functional inputs, not performance evidence.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback
from contextlib import closing
from dataclasses import asdict, replace

import aisimulate_core
import msgspec
from aisimulate_core.sdk import RustForwardPassPerfModel, fpm_learned
from dynamo.common.forward_pass_metrics import ForwardPassMetrics, QueuedRequestMetrics, ScheduledRequestMetrics
from dynamo.planner.config.planner_config import AISPerfModelSpec, PlannerConfig
from dynamo.planner.core.perf_model.ais_adapter import PlannerEnginePerfModel
from dynamo.planner.core.types import EngineCapabilities

EVIDENCE = Path(os.environ['DYNAMO_COMPAT_EVIDENCE'])
RESULTS = []


def check(condition, detail):
    if not condition:
        raise AssertionError(detail)


def case(name, run):
    try:
        detail = run()
        result = {'name': name, 'status': 'passed', 'detail': detail}
    except Exception as error:
        result = {'name': name, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()}
    RESULTS.append(result)
    print(json.dumps(result, sort_keys=True), flush=True)


def caps(tokens=1024):
    return EngineCapabilities(max_num_batched_tokens=tokens, max_num_seqs=32, max_kv_tokens=100000)


def config(role, request=None):
    kwargs = dict(namespace='gbdt-consumer-compatibility', backend='vllm', max_num_fpm_samples=256,
                  load_min_observations=5, fpm_sample_bucket_size=16,
                  mode={'prefill':'prefill', 'decode':'decode', 'aggregated':'agg'}[role])
    if request is not None:
        kwargs['ais_perf_model'] = AISPerfModelSpec(roles={role: request})
    return PlannerConfig(**kwargs)


def metrics(role, index, rank=0, dp=1):
    p = (index % 8 + 1) * 32 if role != 'decode' else 0
    d = index % 8 + 1 if role != 'prefill' else 0
    kv = d * (index % 13 + 1) * 64
    return ForwardPassMetrics(worker_id='synthetic-worker', counter_id=index, dp_rank=rank,
        wall_time=(1 + 0.01*p + 0.2*d + 0.0005*kv)/1000,
        scheduled_requests=ScheduledRequestMetrics(num_prefill_requests=int(p>0), sum_prefill_tokens=p,
            num_decode_requests=d, sum_decode_kv_tokens=kv),
        queued_requests=QueuedRequestMetrics(num_prefill_requests=1, sum_prefill_tokens=128))


def native(adapter):
    check(adapter._engine_model is not None, 'Adapter did not construct an engine model')
    model = adapter._engine_model._model
    check(isinstance(model, RustForwardPassPerfModel), 'Engine did not retain the real SDK model')
    check(isinstance(model._inner, aisimulate_core.RustForwardPassPerfModel), 'SDK did not retain a native object')
    return model


def positive(value, name):
    check(value is not None and math.isfinite(value) and value > 0, f'{name} was unavailable/nonpositive: {value}')
    return value


def regression(role, delayed=False, dp=1, require_capacity=True):
    request = None if dp == 1 else {'model':'synthetic/dp-regression', 'system':'unprofiled', 'backend':'vllm',
        'worker_type':role, 'estimation_mode':'fpm_regression', 'attention_dp':dp}
    adapter = PlannerEnginePerfModel(worker_type=role, config=config(role, request), capabilities=None if delayed else caps())
    if delayed:
        check(adapter._engine_model is None, 'Missing capabilities should defer engine construction')
    else:
        check(native(adapter).diagnostics()['source'] == 'fallback_regression', 'Default path changed estimator')
        check(not adapter.has_sufficient_data(), 'Regression unexpectedly ready before feedback')
    for index in range(1,80):
        adapter.add_observations({('synthetic-worker',rank):metrics(role,index,rank,dp) for rank in range(dp)})
    if delayed:
        check(len(adapter._pending_iterations)==79, 'Pending feedback was lost before capability publication')
        adapter.update_capabilities(caps())
    model = native(adapter)
    check(adapter.has_sufficient_data(), 'Regression did not become ready')
    before_count = adapter.num_observations
    groups = adapter.query_groups({('synthetic-worker',rank):metrics(role,100,rank,dp) for rank in range(dp)})
    check(len(groups)==1 and len(groups[0][1])==dp, 'Adapter grouped complete rank metrics incorrectly')
    group = groups[0][1]
    query = [msgspec.to_builtins(item) for item in group]
    before = positive(model.estimate_forward_pass_time_ms(query), 'native estimate')
    output = {'native_prediction_ms':before, 'retained_observations':before_count, 'actual_native_type':str(type(model._inner))}
    if role != 'decode':
        output['queued_prefill_seconds'] = positive(adapter.estimate_queued_prefill_time(group,max_num_batched_tokens=1024), 'queued prefill')
    if role != 'prefill':
        output['scheduled_decode_seconds'] = positive(adapter.estimate_scheduled_decode_itl(group), 'decode ITL')
    capacity = adapter.find_engine_capacity_rps(isl=128,osl=16)
    output['capacity'] = asdict(capacity) if capacity is not None else None
    adapter.update_capabilities(caps(2048))
    rebuilt = native(adapter)
    check(rebuilt is not model, 'Capability change did not rebuild the model')
    check(adapter.has_sufficient_data(), 'History replay did not restore readiness')
    check(adapter.num_observations == before_count, 'Rebuild lost or duplicated retained observations')
    after = positive(rebuilt.estimate_forward_pass_time_ms(query), 'rebuilt native estimate')
    check(before.hex()==after.hex(), 'Rebuilt regression changed prediction bits for identical retained history')
    output['rebuild_prediction_bits_equal'] = True
    output['pending_iterations_after_rebuild'] = len(adapter._pending_iterations)
    model.close(); rebuilt.close()
    if require_capacity:
        check(capacity is not None, 'Capacity query was unavailable; live tune/query/rebuild succeeded')
    return output


BUNDLE = Path(aisimulate_core.__file__).parent/'systems/fpm_learned/examples/synthetic-functional-v1/bundle.json'
if os.environ.get('DYNAMO_COMPAT_BUNDLE'):
    BUNDLE = Path(os.environ['DYNAMO_COMPAT_BUNDLE'])
if os.environ.get('DYNAMO_COMPAT_CONTROL'):
    BUNDLE=Path(os.environ['DYNAMO_COMPAT_CANDIDATE_BUNDLE'])


def legacy_learned_request(adaptation='online'):
    return {**json.loads(BUNDLE.read_text())['identity'],
        'estimation_mode':'fpm_learned','fallback_policy':'deny',
        'estimator_config':{'correction':{'enabled':False},'fpm_learned':{
            'adaptation':adaptation,'checkpoint_path':str(BUNDLE.parent/'seed')}}}


def learned_request(adaptation='online'):
    if os.environ.get('DYNAMO_COMPAT_CONTROL'):
        return legacy_learned_request(adaptation)
    return fpm_learned.inspect_bundle(BUNDLE, adaptation=adaptation, config={
        **json.loads(BUNDLE.read_text())['identity'],
        'estimation_mode':'fpm_learned','fallback_policy':'deny',
        'estimator_config':{'correction':{'enabled':False},'fpm_learned':{'adaptation':adaptation,'source_policy':'explicit_only','fallback':'disabled'}}})


def learned_query():
    return [ForwardPassMetrics(worker_id='synthetic-worker', counter_id=1, wall_time=0.01,
        scheduled_requests=ScheduledRequestMetrics(num_decode_requests=3,sum_decode_kv_tokens=384))]


def seeded(adaptation, legacy=False):
    request = legacy_learned_request(adaptation) if legacy else learned_request(adaptation)
    adapter = PlannerEnginePerfModel(worker_type='aggregated',config=config('aggregated',request),capabilities=caps())
    model=native(adapter)
    with closing(RustForwardPassPerfModel.best_available(request)) as direct:
        expected=positive(direct.estimate_forward_pass_time_ms([msgspec.to_builtins(x) for x in learned_query()]), 'direct seeded query')
    actual=positive(adapter.estimate_scheduled_decode_itl(learned_query(),add_next_request=False), 'adapter seeded decode')
    check(actual.hex()==(expected/1000).hex(), 'Seeded adapter query differs from the same native direct query')
    details={'adaptation':adaptation,'native_ms':expected,'adapter_seconds':actual,
        'source_kind':model.diagnostics()['provenance']['config']['estimator_config']['fpm_learned'].get('resolved_source',{}).get('kind','legacy_checkpoint_path'),
        'adapter_has_sufficient_data':adapter.has_sufficient_data(), 'native_execution_readiness':model.readiness_report().get('execution',model.readiness_report())}
    if adaptation == 'online':
        for index in range(1,11):
            adapter.add_observations({('synthetic-worker',0):metrics('decode',index)})
        details['online_after_tune']=model.learned_online_diagnostics()
        adapter.update_capabilities(caps(2048))
        rebuilt=native(adapter)
        details['online_after_rebuild']=rebuilt.learned_online_diagnostics()
        details['rebuilt_adapter_seconds']=positive(adapter.estimate_scheduled_decode_itl(learned_query(),add_next_request=False),'rebuilt seeded query')
        rebuilt.close()
    model.close()
    return details


def partial_coverage_guard():
    adapter=PlannerEnginePerfModel(worker_type='aggregated',config=config('aggregated',learned_request()),capabilities=caps())
    model=native(adapter)
    report=model.readiness_report()
    mixed_query=[{'dp_rank':0,'scheduled_requests':{'num_prefill_requests':1,'sum_prefill_tokens':32,
        'num_decode_requests':1,'sum_decode_kv_tokens':128}}]
    mixed_prediction=model.estimate_forward_pass_time_ms(mixed_query)
    execution=report.get('execution',{'ready':mixed_prediction is not None,'legacy_report':report,
        'actual_mixed_prediction_ms':mixed_prediction})
    reported=adapter.has_sufficient_data()
    model.close()
    check(not reported or execution['ready'], 'Pinned adapter reports sufficient data while native aggregated execution lacks required stores: '+json.dumps(execution))
    return {'adapter_has_sufficient_data':reported,'native_execution':execution}


def learned_snapshot_transport():
    request=learned_request('online')
    with tempfile.TemporaryDirectory(prefix='dynamo-compat-snapshot-') as temp:
        with closing(RustForwardPassPerfModel.best_available(request)) as source:
            source.save_state_snapshot(Path(temp)/'snapshot')
        summary=RustForwardPassPerfModel.inspect_state_snapshot(Path(temp)/'snapshot')
        frozen=summary['resolved_config']
        adapter=PlannerEnginePerfModel(worker_type='aggregated',config=config('aggregated',frozen),capabilities=caps())
        model=native(adapter)
        result=positive(adapter.estimate_scheduled_decode_itl(learned_query(),add_next_request=False),'snapshot adapter query')
        model.close()
        return {'adapter_seconds':result,'manifest_schema':summary.get('schema_version')}


def main():
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    for role in ('prefill','decode','aggregated'):
        case('regression_live_'+role,lambda role=role:regression(role))
        case('regression_delayed_capabilities_'+role,lambda role=role:regression(role,delayed=True))
        case('regression_attention_dp2_lifecycle_'+role,lambda role=role:regression(role,dp=2,require_capacity=False))
    for role in ('decode','aggregated'):
        case('regression_attention_dp2_capacity_'+role,lambda role=role:regression(role,dp=2))
    case('learned_required_store_guard',partial_coverage_guard)
    case('legacy_checkpoint_frozen_passthrough',lambda:seeded('frozen',legacy=True))
    case('legacy_checkpoint_online_tune_and_rebuild',lambda:seeded('online',legacy=True))
    if not os.environ.get('DYNAMO_COMPAT_CONTROL'):
        case('seeded_frozen_passthrough',lambda:seeded('frozen'))
        case('seeded_online_tune_and_rebuild',lambda:seeded('online'))
        case('learned_snapshot_transport',learned_snapshot_transport)
    from aisimulate import _runtime
    native_files=[Path(_runtime.__file__)]
    env={'python':sys.version,'platform':platform.platform(),'aisimulate_core_file':aisimulate_core.__file__,
        'packages':{name:importlib.metadata.version(name) for name in ('aisimulate','numpy','scipy','pydantic','msgspec','scikit-learn')},
        'native_files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in native_files],
        'stubs':[], 'bundle_path':str(BUNDLE),'bundle_sha256':hashlib.sha256(BUNDLE.read_bytes()).hexdigest()}
    report={'environment':env,'results':RESULTS,'passed':sum(r['status']=='passed' for r in RESULTS),'failed':sum(r['status']=='failed' for r in RESULTS)}
    (EVIDENCE/'real-native-results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FINAL',json.dumps({'passed':report['passed'],'failed':report['failed']}))
    return int(report['failed']>0)


if __name__=='__main__':
    raise SystemExit(main())
