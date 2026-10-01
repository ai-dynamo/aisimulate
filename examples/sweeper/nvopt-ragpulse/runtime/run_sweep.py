# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run native AISimulate/Vizier with a frozen scenario runner and resource supervisor."""
from __future__ import annotations
import argparse,dataclasses,datetime,functools,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(path)
def event(path,value):
    with Path(path).open('a') as f:f.write(json.dumps({'observed_utc':now(),'observed_monotonic':time.monotonic(),**value},allow_nan=False,default=str)+'\n')

def optimizer_accounting(path,expected_budget):
    branches={}
    for line in Path(path).read_text().splitlines():
        row=json.loads(line);name=row.get('branch');state=branches.setdefault(name,{'suggested':0,'observed':0,'infeasible':0,'issued_trial_ids':[],'terminal_trial_ids':[]})
        if row['event']=='suggest':
            ids=[item['trial_id'] for item in row['suggestions']]
            state['suggested']+=len(ids);state['issued_trial_ids'].extend(ids)
        elif row['event'] in ('observe','observe_infeasible'):
            state['observed' if row['event']=='observe' else 'infeasible']+=1
            state['terminal_trial_ids'].append(row['trial_id'])
    from collections import Counter
    for state in branches.values():
        issued=Counter(state['issued_trial_ids']);terminal=Counter(state['terminal_trial_ids'])
        state['unreceipted_trial_ids']=list((issued-terminal).elements())
        state['unexpected_terminal_trial_ids']=list((terminal-issued).elements())
        state['duplicate_issued_trial_ids']=[key for key,value in issued.items() if value>1]
        state['duplicate_terminal_trial_ids']=[key for key,value in terminal.items() if value>1]
        state['null_trial_ids']=None in issued or None in terminal
    count=sum(state['suggested'] for state in branches.values())
    all_receipted=count<=expected_budget and all(
        not state['unreceipted_trial_ids'] and not state['unexpected_terminal_trial_ids']
        and not state['duplicate_issued_trial_ids'] and not state['duplicate_terminal_trial_ids']
        and not state['null_trial_ids'] for state in branches.values())
    return {'suggested':count,'expected_budget':expected_budget,'branches':branches,
            'termination_reason':('suggestion_budget_exceeded' if count>expected_budget else
                                  'invalid_or_incomplete_trial_feedback' if not all_receipted else
                                  'native_suggestion_budget_exhausted' if count==expected_budget else
                                  'native_projection_stall'),
            'all_suggestions_receipted':all_receipted}

class AuditedSampler:
    """Instrumentation only: every suggestion and observation goes to native AIS Vizier."""
    def __init__(self,inner,path):self.inner=inner;self.branch=inner.branch;self.path=path
    def suggest(self,count):
        t=time.perf_counter();items=self.inner.suggest(count)
        event(self.path,{'event':'suggest','branch':self.branch.deployment_mode,'seconds':time.perf_counter()-t,'requested':count,'suggestions':[{'trial_id':getattr(x.handle,'id',None),'selection':x.selection,'parallel_config':dataclasses.asdict(x.parallel_config),'projection':dataclasses.asdict(x.projection) if x.projection is not None else None,'infeasible_reason':x.infeasible_reason} for x in items]})
        return items
    def observe(self,suggestion,metrics):
        t=time.perf_counter();self.inner.observe(suggestion,metrics)
        event(self.path,{'event':'observe','branch':self.branch.deployment_mode,'trial_id':getattr(suggestion.handle,'id',None),'metrics':metrics,'seconds':time.perf_counter()-t})
    def observe_infeasible(self,suggestion,reason):
        t=time.perf_counter();self.inner.observe_infeasible(suggestion,reason)
        event(self.path,{'event':'observe_infeasible','branch':self.branch.deployment_mode,'trial_id':getattr(suggestion.handle,'id',None),'reason':reason,'seconds':time.perf_counter()-t})

class AuditedSamplerFactory:
    def __init__(self,path):self.path=path
    def __call__(self,branch,**kwargs):
        from aisimulate.sweeper.sampler import make_branch_sampler
        t=time.perf_counter();inner=make_branch_sampler(branch,**kwargs)
        assert type(inner).__name__=='SeededBayesianBranchSampler',type(inner)
        event(self.path,{'event':'sampler_created','branch':branch.deployment_mode,'class':type(inner).__module__+'.'+type(inner).__name__,'settings':kwargs,'seconds':time.perf_counter()-t})
        return AuditedSampler(inner,self.path)

def child(args):
    import yaml
    from scenario_runner import ScenarioRunnerFactory
    from aisimulate.config.cli import CoreRecommendationConfig
    from aisimulate.config.common import split_config_sections
    from aisimulate.recommend import recommendation_to_sweeper, _candidate_prediction
    from aisimulate.sweeper.search import Sweeper
    from importlib.metadata import version
    from adapters import CampaignAdapter
    out=args.output
    contract=yaml.safe_load(args.protocol.read_text())
    if version('aisimulate') != contract['runtime']['aisimulate_python']:
        raise ValueError('AISimulate differs from frozen protocol')
    import dynamo._core
    with Path(dynamo._core.__file__).open('rb') as binary:
        binary_hash=hashlib.file_digest(binary,'sha256').hexdigest()
    if binary_hash != '2f47cbb1def5e6b71af0271f0e189dd00e65ee739e801232928ce6dd2254277b':
        raise ValueError('Native Dynamo binary differs from frozen release build')
    raw=yaml.safe_load(args.config.read_text())
    core,adapter_configs=split_config_sections(raw,command='recommend')
    if args.parallelism is not None:core['optimizer']['parallelism']=args.parallelism
    if args.candidate_timeout is not None:core['optimizer']['candidate_timeout_seconds']=args.candidate_timeout
    config=CoreRecommendationConfig.model_validate(core)
    if config.optimizer.max_trials != 256 or config.optimizer.seed != 20260929:
        raise ValueError('Formal search must preserve the256suggestion budget/seed')
    trace=config.traffic.source.paths[0]
    history=contract['traffic']['planner_history']['path'] if args.scenario==3 else None
    factory=ScenarioRunnerFactory(attempts_dir=str(out/'attempts'),scenario=args.scenario,
        trace_path=trace,dynamo_sha=contract['runtime']['dynamo_commit'],
        history_trace=history,history_sha256=args.history_sha256 if history else None,
        max_address_space_gib=96)
    t=time.perf_counter();start=now()
    write(out/'run-start.json',{'started_utc':start,'started_monotonic':time.monotonic(),
        'command':sys.argv,'config':{**core,**adapter_configs},'runner_contract':contract,
        'mode':'fresh_winner_validation' if args.replay_spec else 'native_vizier_sweep',
        'scenario':args.scenario,'pid':os.getpid(),'binding_sha256':binary_hash,
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        'history_sha256':args.history_sha256 if history else None})
    try:
        if args.replay_spec:
            from deserialize import replay_spec_from_dict
            spec=replay_spec_from_dict(json.loads(args.replay_spec.read_text()))
            runner=factory.create(0)
            try: report=runner.run(spec)
            finally:runner.close()
            write(out/'validation-report.json',dataclasses.asdict(report))
            write(out/'completion.json',{'status':'completed','mode':'fresh_winner_validation',
                'started_utc':start,'finished_utc':now(),'wall_seconds':time.perf_counter()-t})
            return 0
        smart=recommendation_to_sweeper(config,adapter_configs=adapter_configs,stack='dynamo')
        write(out/'compiled-search-config.json',smart.model_dump(mode='json'))
        providers={}
        for section,public in adapter_configs.items():
            if section=='router':
                from dynamo.router.simulation.provider import create_provider
            elif section=='planner':
                from dynamo.planner.simulation.provider import create_provider
            else:raise ValueError(f'unsupported adapter {section}')
            adapter=create_provider()
            providers[adapter.name]=CampaignAdapter(adapter,public,config,history,out)
        last_round_start=time.perf_counter();best_so_far=None
        def on_round(index,candidates):
            nonlocal last_round_start,best_so_far
            current=time.perf_counter();best=max((c.score for c in candidates),default=None)
            event(out/'rounds.jsonl',{'event':'round_completed','round':index,
                'seconds_since_previous_round':current-last_round_start,'elapsed_seconds':current-t,
                'feasible_seen':len(candidates),'best_score':best})
            write(out/'feasible-checkpoint.json',{'round':index,'candidates':[c.model_dump(mode='json') for c in candidates]})
            if best is not None and (best_so_far is None or best>best_so_far):
                best_so_far=best
                winner=max(candidates,key=lambda c:c.score)
                event(out/'best-events.jsonl',{'event':'best_improved','round':index,
                    'elapsed_seconds':current-t,'candidate':winner.model_dump(mode='json')})
            last_round_start=current
            print(json.dumps({'round':index,'feasible_seen':len(candidates),'best_score':best,'elapsed_s':current-t}),flush=True)
        sweeper=Sweeper(runner_factory=factory,sampler_factory=AuditedSamplerFactory(out/'optimizer-events.jsonl'),
            providers=providers,show_progress=False,
            prediction_config_factory=functools.partial(_candidate_prediction,config,
                adapter_sections={name:p.section for name,p in providers.items()}))
        result=sweeper.run(smart,top_n=None,candidate_retention='all',on_round=on_round)
        write(out/'sweep-result.json',result.model_dump(mode='json'))
        accounting=optimizer_accounting(out/'optimizer-events.jsonl',smart.sweep.max_trials)
        write(out/'optimizer-accounting.json',accounting)
        if not accounting['all_suggestions_receipted']:
            raise RuntimeError('native sweep returned with unreceipted or duplicated trial feedback')
        write(out/'completion.json',{'status':'completed','started_utc':start,'finished_utc':now(),
            'wall_seconds':time.perf_counter()-t,'counts':result.counts.model_dump(mode='json'),
            'optimizer_accounting':accounting})
        return 0
    except BaseException as exc:
        write(out/'completion.json',{'status':'failed','started_utc':start,'finished_utc':now(),
            'wall_seconds':time.perf_counter()-t,'error':str(exc),'traceback':traceback.format_exc()})
        raise

def supervise(args):
    import psutil
    if args.output.exists() and any(args.output.iterdir()):raise ValueError(f'Refusing nonempty output directory {args.output}')
    args.output.mkdir(parents=True,exist_ok=True)
    policy={'max_process_tree_rss_gib':args.memory_limit_gib,'min_host_available_gib':16,
        'log_limit_mib':512,'max_sweep_wall_seconds':None, 'candidate_wall_seconds':args.candidate_timeout or 7200}
    if args.max_wall_seconds is not None:policy={**policy,'max_sweep_wall_seconds':args.max_wall_seconds}
    logfile=args.output/'process.log';cmd=[sys.executable,str(Path(__file__).resolve()),'--child','--config',str(args.config),'--output',str(args.output),
        '--scenario',str(args.scenario),'--protocol',str(args.protocol),'--history-sha256',args.history_sha256]
    if args.parallelism is not None:cmd+=['--parallelism',str(args.parallelism)]
    if args.candidate_timeout is not None:cmd+=['--candidate-timeout',str(args.candidate_timeout)]
    if args.replay_spec:cmd+=['--replay-spec',str(args.replay_spec)]
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',RUST_LOG='error',DYN_LOG='error',MALLOC_ARENA_MAX='2',TMPDIR=os.environ.get('TMPDIR','/tmp'),JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true')
    t=time.monotonic();started=now();peak=0.0;reason=None;pending={};seen_exits=set();last=0;proc=None;rc=1;failure=None
    def terminate_owned_group():
        if proc is None:return
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            proc.poll()
            try:os.killpg(proc.pid,0)
            except ProcessLookupError:break
            time.sleep(.1)
        # A leader exiting does not establish that every worker exited.
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:pass
    def interrupted(signum,frame):
        del frame
        raise KeyboardInterrupt(f'supervisor received signal{signum}')
    previous={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    try:
        with logfile.open('w') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            while proc.poll() is None:
                try:processes=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)
                except psutil.NoSuchProcess:processes=[]
                rss=0.0;live={}
                for p in processes:
                    try:
                        if p.status()==psutil.STATUS_ZOMBIE:continue
                        rss+=p.memory_info().rss/1024**3;live[p.pid]=p.create_time()
                    except psutil.NoSuchProcess:pass
                peak=max(peak,rss);available=psutil.virtual_memory().available/1024**3
                for path in (args.output/'attempts').glob('*/attempt.json'):
                    try:d=json.loads(path.read_text())
                    except (FileNotFoundError,json.JSONDecodeError):continue
                    if d.get('status')=='running' and str(path) not in seen_exits:pending.setdefault(str(path),d)
                    else:pending.pop(str(path),None)
                for pth,d in list(pending.items()):
                    pid=d.get('pid')
                    expected_birth=d.get('process_create_time')
                    if pid and (pid not in live or (expected_birth is not None and live.get(pid)!=expected_birth)):
                        stop=time.monotonic();terminal={'status':'process_exited_without_attempt_completion','event':'attempt_process_exit_observed','attempt_path':pth,'pid':pid,'process_exit_observed_monotonic':stop,'wall_seconds_upper_bound':stop-d['started_monotonic'],'sampling_resolution_seconds':2.0,'method':'supervisor_observed_pid_exit'}
                        event(args.output/'worker-exits.jsonl',terminal);write(Path(pth).parent/'supervisor-terminal.json',terminal);seen_exits.add(pth);pending.pop(pth,None)
                if time.monotonic()-last>=15:
                    event(args.output/'resource-samples.jsonl',{'rss_gib':rss,'host_available_gib':available,'processes':len(processes),'log_bytes':logfile.stat().st_size,'elapsed_seconds':time.monotonic()-t});last=time.monotonic()
                if rss>policy['max_process_tree_rss_gib']:reason='process_tree_rss_limit'
                elif available<policy['min_host_available_gib']:reason='host_memory_headroom'
                elif logfile.stat().st_size>policy['log_limit_mib']*1024**2:reason='native_log_limit'
                elif policy['max_sweep_wall_seconds'] is not None and time.monotonic()-t>policy['max_sweep_wall_seconds']:reason='whole_sweep_wall_limit'
                if reason:
                    event(args.output/'resource-samples.jsonl',{'event':'terminating_owned_process_group','reason':reason})
                    break
                time.sleep(2)
            if proc.poll() is not None:rc=proc.returncode
    except BaseException as exc:
        reason=reason or 'supervisor_exception';failure={'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()};rc=130 if isinstance(exc,KeyboardInterrupt) else 1
    finally:
        # Always stop/reap owned workers, including interrupted/error paths.
        # Prevent a second termination signal from skipping this cleanup.
        for s in previous:signal.signal(s,signal.SIG_IGN)
        terminate_owned_group()
        if proc is not None and reason is not None and failure is None:rc=proc.returncode if proc.returncode else 1
        for s,h in previous.items():signal.signal(s,h)
    for path in (args.output/'attempts').glob('*/attempt.json'):
        d=json.loads(path.read_text())
        if d.get('status')=='running' and not (path.parent/'supervisor-terminal.json').exists():
            stop=time.monotonic();write(path.parent/'supervisor-terminal.json',{'status':'process_exited_without_attempt_completion','wall_seconds_upper_bound':stop-d['started_monotonic'],'method':'supervisor_final_process_group_exit_observation','termination_reason':reason,'observed_utc':now()})
    write(args.output/'supervisor.json',{'started_utc':started,'finished_utc':now(),'wall_seconds':time.monotonic()-t,'peak_process_tree_rss_gib':peak,'returncode':rc,'termination_reason':reason,'supervisor_error':failure,'policy':policy,'log_bytes':logfile.stat().st_size if logfile.exists() else 0})
    print(json.dumps({'output':str(args.output),'returncode':rc,'wall_s':time.monotonic()-t,'peak_tree_rss_gib':peak,'termination_reason':reason}))
    return rc

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scenario',type=int,choices=[1,2,3],required=True)
    p.add_argument('--protocol',type=Path,default=ROOT/'experiment-contract.yaml')
    p.add_argument('--history-sha256',default='21eebb38837ea85ebe06d85ef5f5a1fbb644c411c927142b337c1ce61eb44205')
    p.add_argument('--parallelism',type=int)
    p.add_argument('--candidate-timeout',type=float)
    p.add_argument('--memory-limit-gib',type=float,default=320)
    p.add_argument('--replay-spec',type=Path,help='Fresh validation of an archived requested ReplaySpec, not a new study')
    p.add_argument('--child',action='store_true')
    p.add_argument('--max-wall-seconds',type=float,default=None)
    args=p.parse_args()
    for value in (args.parallelism,args.candidate_timeout,args.memory_limit_gib,args.max_wall_seconds):
        if value is not None and value<=0:p.error('resource/deadline values must be positive')
    args.config=args.config.resolve();args.output=args.output.resolve();args.protocol=args.protocol.resolve()
    if args.replay_spec:args.replay_spec=args.replay_spec.resolve()
    return child(args) if args.child else supervise(args)
if __name__=='__main__':raise SystemExit(main())
