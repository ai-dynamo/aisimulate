"""New compatibility harness; all imported Gym adapter code stays at PR658 a8.

Self-authored deterministic targets exercise interfaces, not hardware accuracy.
No mock of the trainer, native model, or Gym adapter is used.
"""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel, fpm_learned
from scripts.fpm_accuracy.models.learned_scenario import LearnedScenarioPredictor
from scripts.fpm_accuracy.models.mixed_scenario import MixedOriginalA
from scripts.fpm_accuracy.types.forward_pass import ForwardPassIteration, ForwardPassMetric, RequestMetrics
from python.aisimulate.tests.unit.sdk.test_fpm_phase_readiness import phase_roots


def observed(index, kind='decode', worker='worker', target=12.0, small=False):
    return ForwardPassIteration.single_rank(ForwardPassMetric(
        fpm_id=index, configuration_id='compat-synthetic', version=1, worker_id=worker,
        dp_rank=0, counter_id=index, wall_time_s=target/1000,
        scheduled=RequestMetrics(num_prefill_requests=int(kind != 'decode'),
            sum_prefill_tokens=(1 if small else 32)*int(kind != 'decode'),
            num_decode_requests=int(kind != 'prefill'),
            sum_decode_kv_tokens=(1 if small else 64)*int(kind != 'prefill')),
        queued=RequestMetrics(), capture_id='synthetic-interface-check'))


def request():
    return ForwardPassPerfModelConfig(model='Qwen/Qwen3-8B', system='h200_sxm',
        worker_type='aggregated', backend='vllm', backend_version='0.25.1')


def record(name, data):
    root=Path(os.environ['GYM_COMPAT_RECEIPTS']); root.mkdir(parents=True,exist_ok=True)
    (root/(name+'.json')).write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')


@pytest.mark.parametrize('method', ['gbdt','hybrid','selector'])
def test_unchanged_cold_c_external_first_fit(method):
    model=LearnedScenarioPredictor(method,RustForwardPassPerfModel,request(),fpm_learned,
        online=True,cold_start=True)
    try:
        predictions=[]
        for index in range(20):
            row=observed(index)
            predictions.append(model.predict(row.prediction_input()).value_ms)
            assert predictions[-1] is None
            model.tune([row])
        first=model.predict(observed(20).prediction_input()).value_ms
        assert first == pytest.approx(12)
        assert model.predict(observed(20,worker='other').prediction_input()).value_ms is None
        assert model.predict(observed(20,kind='prefill').prediction_input()).value_ms is None
        before=model.diagnostics()['scopes'][0]
        assert before['bootstrap_observations']==20 and before['feedback_observations']==0
        assert len(model.seed_artifact_paths())==1
        assert len(model.seed_checkpoint_paths())==int(method!='hybrid')
        assert 'checkpoint_path' in next(iter(model.seed_configurations().values()))['estimator_config']['fpm_learned'] if method!='hybrid' else True
        model.tune([observed(20,target=15)])
        after=model.diagnostics()['scopes'][0]
        assert after['feedback_observations']==1
        record('cold-'+method,{'predictions_before_admission':predictions,'first_ready_prediction':first,
            'next_prediction':model.predict(observed(21).prediction_input()).value_ms,
            'bootstrap_observations':after['bootstrap_observations'],'feedback_observations':after['feedback_observations'],
            'stores':after['stores'],'adaptation':model.diagnostics()['adaptation']})
    finally:
        model.close()


@pytest.mark.parametrize('count',[18,20])
def test_unchanged_b_c_external_first_fit(count):
    model=LearnedScenarioPredictor('gbdt',RustForwardPassPerfModel,request(),fpm_learned,
        [observed(index) for index in range(count)],online=True)
    try:
        for index in range(count,20):
            assert model.predict(observed(index).prediction_input()).value_ms is None
            model.tune([observed(index)])
        assert model.predict(observed(20).prediction_input()).value_ms == pytest.approx(12)
        scope=model.diagnostics()['scopes'][0]
        assert scope['offline_observations']==count
        assert scope['bootstrap_observations']==20-count
        assert len(model.seed_checkpoint_paths())==1
        model.tune([observed(20,target=15)])
        record('bc-'+str(count),{'first_ready_prediction':12.,'next_prediction':model.predict(observed(21).prediction_input()).value_ms,
            'offline_observations':count,'bootstrap_observations':20-count,'feedback_observations':model.diagnostics()['scopes'][0]['feedback_observations']})
    finally:
        model.close()


@pytest.mark.parametrize('method,online',[('gbdt',False),('gbdt',True),('hybrid',True),('selector',True)])
def test_unchanged_seeded_b_store_and_future_feedback(method,online):
    seeds=[observed(index,kind=kind,target=target) for kind,target in [('prefill',2),('decode',3)] for index in range(20)]
    model=LearnedScenarioPredictor(method,RustForwardPassPerfModel,request(),fpm_learned,seeds,online=online)
    try:
        predictions={kind:model.predict(observed(21,kind=kind).prediction_input()).value_ms for kind in ['prefill','decode','mixed']}
        assert predictions['prefill']==pytest.approx(2)
        assert predictions['decode']==pytest.approx(3)
        assert predictions['mixed'] is None
        if online:
            model.tune([observed(21,target=5)])
            assert sum(scope['feedback_observations'] for scope in model.diagnostics()['scopes'])==1
        else:
            with pytest.raises(ValueError,match='frozen'):
                model.tune([observed(21,target=5)])
        record('seeded-'+method+'-'+str(online),{'initial_predictions':predictions,
            'later_decode':model.predict(observed(22).prediction_input()).value_ms,
            'scope_count':len(model.diagnostics()['scopes'])})
    finally:
        model.close()


@pytest.mark.parametrize('method,online',[('gbdt',False),('gbdt',True),('hybrid',True),('selector',True)])
def test_unchanged_original_a_mixed_prior_and_external_append(phase_roots,method,online):
    config,make_root=phase_roots
    root=Path(make_root('gym-mixed'))
    parquet=root/'data/h200_sxm/vllm/0.25.1/fpm_forward_perf.parquet'
    rows=pq.read_table(parquet).to_pylist()
    zero=dict(next(row for row in rows if row['workload_kind']=='decode'))
    zero.update(total_kv_read_tokens=0,latency_ms=1.,cell_id='synthetic-zero')
    rows.append(zero)
    for row in rows:
        row.update(pp=1,cp=1,attention_backend='auto',moe_backend='auto',enable_wideep=False,enable_eplb=False)
    pq.write_table(pa.Table.from_pylist(rows),parquet)
    sidecar=parquet.with_suffix('.metadata.json')
    metadata=json.loads(sidecar.read_text())
    metadata.update(row_count=len(rows),parquet_sha256=hashlib.sha256(parquet.read_bytes()).hexdigest())
    sidecar.write_text(json.dumps(metadata))
    artifact=SimpleNamespace(artifact_id='synthetic-functional-compat',local_path=parquet,
        local_metadata_path=sidecar,sha256=metadata['parquet_sha256'],
        metadata_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest())
    engine={'system_name':'h200_sxm','backend':'vllm','backend_version':'0.25.1'}
    mixed=MixedOriginalA(engine,artifact,'ac' if online else 'a')
    config['systems_paths']=[str(mixed.prepared.systems_root)]
    seeds=[observed(index,kind=kind,target=target,small=True) for kind,target in [('prefill',2),('decode',3)] for index in range(20)]
    model=None
    try:
        model=LearnedScenarioPredictor(method,RustForwardPassPerfModel,ForwardPassPerfModelConfig(**config),fpm_learned,seeds,online=online,mixed=mixed)
        first=model.predict(observed(100,kind='mixed',target=8,small=True).prediction_input())
        assert first.value_ms==pytest.approx(4.)
        pure_before=model.diagnostics()['scopes'][0]['initial_store_sha256']
        predictions=[]
        if online:
            for index in range(20):
                row=observed(100+index,kind='mixed',target=8,small=True)
                predictions.append(model.predict(row.prediction_input()).value_ms)
                model.tune([row])
        scope=model.diagnostics()['scopes'][0]
        expected_append=online and method!='hybrid'
        assert scope['store_additions']==int(expected_append)
        assert ('contains_locally_mixed' in scope['stores'])==expected_append
        assert all(scope['final_store_sha256'][kind]==digest for kind,digest in pure_before.items())
        assert len(model.seed_checkpoint_paths())== (2 if expected_append else 0)
        record('mixed-'+method+'-'+str(online),{'initial_prediction':first.value_ms,'initial_route':first.metadata,
            'predictions_before_admission':predictions,'later_prediction':model.predict(observed(120,kind='mixed',target=8,small=True).prediction_input()).value_ms,
            'bootstrap_observations':scope['bootstrap_observations'],'feedback_observations':scope['feedback_observations'],'store_additions':scope['store_additions'],'stores':scope['stores']})
    finally:
        if model is not None: model.close()
        else: mixed.close()
