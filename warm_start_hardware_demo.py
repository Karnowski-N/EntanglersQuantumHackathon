"""Warm-start Reduced Qubit Hybrid Hourly: simulation-trained, QPU-sampled.

Preparation and report functions never submit jobs. Hardware submission requires
an explicit credit ceiling and resumes known jobs without duplicate submissions.
Uses the verified greedy warm start and aligned mixer; no renewables or unsampled repair.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import argparse,hashlib,json,time
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Statevector
from qbraid.runtime import QbraidProvider
from qbraid.runtime.native.job import QbraidJob
from qbraid_core.services.runtime.schemas import Program
import reduced_quantum_method as rq
import warm_start_scaling as warm
import mimiq_mps_pilot as m
from hourly_seed_shot_sweep import dump,clean_json

DEVICE='aws:rigetti:qpu:cepheus-1-108q'
FLEETS=(5,10);HOURS=tuple(range(13,18));SHOTS=200;SEED=9100
ROOT=rq.HERE


def quote(device,shots=SHOTS,jobs=10):
    if device.profile.simulator:raise RuntimeError('Hardware protocol requires a QPU device')
    price=device.profile.pricing
    if price is None:raise RuntimeError('A fixed price quote is required')
    rates={k:float(getattr(price,k)) for k in ('perTask','perShot','perMinute')}
    if rates['perMinute']!=0:raise RuntimeError('Time-priced device not authorized by this fixed-cost protocol')
    per_job=rates['perTask']+shots*rates['perShot']
    return dict(device=device.id,rates=rates,shots_per_job=shots,max_jobs=jobs,
        estimated_credits_per_job=per_job,max_estimated_credits=round(jobs*per_job,8),
        at_utc=datetime.now(timezone.utc).isoformat())


def circuit(trace):
    return warm.from_trace(trace,trace['angles'])


def score(counts,trace,ns):
    return warm.summarize(counts,trace,ns)


def draw(probabilities,shots,seed):
    p=np.asarray(probabilities,float);p=p/p.sum()
    ids=np.random.default_rng(seed).choice(len(p),size=shots,p=p)
    width=int(np.log2(len(p)))
    return Counter(format(int(i),f'0{width}b') for i in ids)


def prepare_hour(ns,hour,history,previous=None):
    """Same six-angle warm-start protocol, finite local training, fresh final draw."""
    start=time.perf_counter();trace=warm.layout(ns,hour,history)
    training=[];probs=[]
    for index,angles in enumerate(warm.ANGLES):
        p=Statevector.from_instruction(warm.from_trace(trace,angles,False)).probabilities()
        counts=draw(p,warm.TRAIN_SHOTS,SEED+100000*ns['N']+100*hour+index)
        training.append(dict(angles=angles,counts=counts,score=score(counts,trace,ns)));probs.append(p)
    winner=warm.ft.choose([x['score'] for x in training]);probabilities=probs[winner]
    trace.update(angles=warm.ANGLES[winner],training=training,chosen_index=winner,
        qubits=trace['reduced_qubits'],logical_rzz=int(np.count_nonzero(np.triu(trace['small_Q'],1))),
        local_training=dict(method='six fixed angles, 256 local simulated draws each',shots=6*warm.TRAIN_SHOTS,
            statevector_evaluations=6,seed_rule='9100 + 100000*fleet + 100*hour + angle index'),
        training_wall_seconds=time.perf_counter()-start)
    counts=draw(probabilities,SHOTS,SEED+1000*ns['N']+hour)
    trace['same_circuit_ideal_counts']=counts;trace['same_circuit_ideal']=score(counts,trace,ns)
    bits=rq.bit_table(len(trace['small_Q']));states=m.restore(bits,trace['free'],trace['values'])[:,:ns['N']]
    on,off=ns['locks'](ns['plants'],history)
    valid=np.array([ns['hour_ok'](ns['plants'],ns['demand'][hour],ns['reserve'][hour],x,on,off) for x in states])
    trace['exact_ideal_feasible_probability']=float(probabilities[valid].sum())
    trace['ideal_probabilities']=probabilities
    return trace


def prepare_payload(device,trace):
    """Read-only provider conversion; verify probabilities survive serialization."""
    qc=circuit(trace);payload=device.apply_runtime_profile(qc)
    if payload.format!='qasm3':raise RuntimeError('Expected explicit QASM3 payload')
    restored=qasm3.loads(payload.data);restored.remove_final_measurements(inplace=True)
    probs=Statevector.from_instruction(restored).probabilities()
    np.testing.assert_allclose(probs,trace['ideal_probabilities'],atol=1e-10,rtol=1e-9)
    trace['payload_roundtrip_passed']=True
    trace['payload_sha256']=hashlib.sha256(payload.data.encode()).hexdigest()
    return payload.model_dump(mode='json')


def check_day(ns,schedule,reference):
    check=rq.ck.check_schedule(ns['plants'],ns['demand'],ns['reserve'],schedule,milp_cost=reference,verbose=False)
    if not check['valid']:raise AssertionError('Returned whole-day schedule violates constraints')
    return check


def prepare(output=None):
    output=Path(output) if output else ROOT/'results/warm_start_hardware'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False);device=QbraidProvider().get_device(DEVICE)
    pricing=quote(device);record=dict(created_utc=datetime.now(timezone.utc).isoformat(),device=DEVICE,
        settings=dict(fleets=FLEETS,hours=HOURS,shots=SHOTS,seed=SEED,p=1,slack_qubits=3,warm_start_state=True,
            refinement=False,renewables=False,ramps=False,training='warm-start p=1; six fixed angles, 256 local simulated training shots each; fresh 200 final shots',
            physical_measurements='pending',initial_full_day_milp=True,milp_tail=True,strict=True,
            bit_order='qBraid AWS: classical bit 0 is rightmost',max_jobs=10,max_retries=0),
        quote=pricing,jobs=[],fleets={},approval=None,hardware_complete=False,
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['warm_start_hardware_demo.py','warm_start_scaling.py','mimiq_feasibility_training.py','reduced_quantum_method.py','rq_classical.py','check.py']})
    record['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for n in FLEETS for p in
        [ROOT/'data'/('plants'+('' if n==5 else str(n))+'.csv'),ROOT/'data'/('demand'+('' if n==5 else str(n))+'.csv')]}
    dump(output/'experiment.json',record)
    for fleet in FLEETS:
        ns,ref_stats,baseline,plan=m.initial_problem(fleet)
        history=plan[:13].tolist();sim_history=plan[:13].tolist();sim_traces=[];start=time.perf_counter()
        for hour in HOURS:
            t=prepare_hour(ns,hour,sim_history);sim_traces.append(t)
            pick=t['same_circuit_ideal']['best_sampled_plant_state']
            if pick is None:raise RuntimeError('Prepared ideal simulation failed; inspect before hardware submission')
            sim_history.append(list(pick))
        sim=warm.complete_tail(ns,sim_history,baseline)
        sim.update(hourly=sim_traces,wall_seconds=time.perf_counter()-start,milp_stats=ref_stats)
        trace=prepare_hour(ns,13,history);payload=prepare_payload(device,trace)
        dump(output/f'fleet{fleet}_hour13_payload.json',payload)
        (output/f'fleet{fleet}_hour13.qasm').write_text(payload['data'])
        record['fleets'][str(fleet)]=dict(fleet=fleet,reference_cost=baseline,reference_schedule=plan,
            reference_milp_stats=ref_stats,simulation=sim,hardware=dict(status='PREPARED',valid=None,
                completed_day=False,history=history,hourly=[trace],milp_tail_stats=None))
        dump(output/'experiment.json',record)
        print('PREPARED WARM START',fleet,'ideal full-day gap',sim['checked']['gap_pct'],'first circuit',trace['qubits'],'qubits',flush=True)
    dump(output/'submission_plan.json',dict(device=DEVICE,quote=pricing,settings=record['settings'],
        first_circuits=[dict(fleet=n,file=f'fleet{n}_hour13.qasm',qubits=record['fleets'][str(n)]['hardware']['hourly'][0]['qubits']) for n in FLEETS],
        later_circuits='adapt to measured schedule history using the same tested implementation; stop a fleet if no feasible sample',
        no_hardware_jobs_submitted=True))
    report(output);print('READY WARM START',output,'MAX CREDITS',pricing['max_estimated_credits'],flush=True);return output


def run_hardware(output,approved_credits):
    output=Path(output);record=json.loads((output/'experiment.json').read_text())
    if approved_credits<=0:raise ValueError('Explicit positive approved credit ceiling required')
    if record['hardware_complete']:return record
    for name,digest in record.get('source_sha256',{}).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Implementation changed after preparation; prepare and review a new plan')
    for name,digest in record.get('input_sha256',{}).items():
        if hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Input data changed after preparation')
    if record['approval'] is None:
        record['approval']=dict(credit_ceiling=float(approved_credits),at_utc=datetime.now(timezone.utc).isoformat(),basis='User explicitly requested real QPU runs with 500 credits; execution capped below that at 385 credits')
    elif approved_credits>record['approval']['credit_ceiling']:raise RuntimeError('Cannot silently increase an existing credit ceiling')
    budget=min(approved_credits,record['approval']['credit_ceiling'],385.0)
    if not record['settings'].get('warm_start_state'):raise RuntimeError('This runner only executes the selected warm-start method')
    for name,sha in record['source_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Source changed since preparation: '+name)
    for name,sha in record['input_sha256'].items():
        if hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Input changed since preparation: '+name)
    dump(output/'experiment.json',record)
    for hour in HOURS:
        for fleet in FLEETS:
            row=record['fleets'][str(fleet)];hw=row['hardware']
            if hw['status'] in ('FAILED','COMPLETED'):continue
            history=hw['history']
            if len(history)>hour:continue
            ns=rq.make_namespace(fleet,strict=True)
            trace=next((t for t in hw['hourly'] if t['hour']==hour),None)
            if trace is None:
                previous=hw['hourly'][-1]['angles'] if hw['hourly'] else None
                trace=prepare_hour(ns,hour,history,previous);hw['hourly'].append(trace);dump(output/'experiment.json',record)
            label=f'fleet{fleet}_hour{hour}';info=next((j for j in record['jobs'] if j['label']==label),None)
            device=QbraidProvider().get_device(DEVICE)
            if info is None:
                current=quote(device);reserved=sum(max(j['reserved_credits'],float(j.get('metadata',{}).get('cost') or 0)) for j in record['jobs'])
                if len(record['jobs'])>=10 or reserved+current['estimated_credits_per_job']>budget+1e-8:
                    raise RuntimeError('Approved hardware credit/job ceiling reached')
                if current['estimated_credits_per_job']>record['quote']['estimated_credits_per_job']+1e-8:
                    raise RuntimeError('Device price rose; no submission')
                if device.status().name!='ONLINE':raise RuntimeError('Device is not ONLINE; no submission')
                import subprocess,re
                balance_text=subprocess.run(['qbraid','account','credits'],capture_output=True,text=True,check=True).stdout
                match=re.search(r'credits remaining:\s*([0-9.,]+)',balance_text)
                if not match or float(match.group(1).replace(',',''))<current['estimated_credits_per_job']:
                    raise RuntimeError('Insufficient verified qBraid balance; no submission')
                payload=prepare_payload(device,trace);dump(output/f'{label}_payload.json',payload)
                (output/f'{label}.qasm').write_text(payload['data'])
                info=dict(label=label,fleet=fleet,hour=hour,reserved_credits=current['estimated_credits_per_job'],
                    status='SUBMISSION_INTENT',shots=SHOTS,device=DEVICE,quote=current,submitted_utc=datetime.now(timezone.utc).isoformat())
                record['jobs'].append(info);hw['status']='RUNNING';dump(output/'experiment.json',record)
                # Submission happens exactly once. An interrupted intent without a
                # job ID must be reconciled against the account before any retry.
                job=device.submit(Program(**payload),shots=SHOTS,name=f'Warm-start hourly QPU {output.name} {label}')
                info.update(job_id=job.id,status='SUBMITTED');dump(output/'experiment.json',record)
                print('SUBMITTED',label,job.id,'reserved credits',info['reserved_credits'],flush=True)
            elif 'job_id' not in info:raise RuntimeError('Unresolved submission intent; reconcile jobs before retrying')
            else:job=QbraidJob(info['job_id'],device=device)
            started=time.perf_counter();last=None
            while True:
                status=job.status().name;info['status']=status;info['last_polled_utc']=datetime.now(timezone.utc).isoformat()
                if status!=last:print('STATUS',label,status,flush=True);last=status
                dump(output/'experiment.json',record)
                if status in ('COMPLETED','FAILED','CANCELLED','CANCELED'):break
                if time.perf_counter()-started>900:
                    report(output);raise TimeoutError('Hardware job still pending; resume this directory without resubmission')
                time.sleep(10)
            result=job.result(timeout=30)
            if not result.success:
                hw.update(status='FAILED',valid=False,failure=f'hardware job {info["job_id"]} ended {status}; no retry')
                dump(output/'experiment.json',record);report(output);continue
            if result.device_id!=DEVICE or result.job_id!=info['job_id']:
                raise AssertionError('Result provenance does not match the requested hardware job')
            counts=result.data.get_counts()
            if sum(counts.values())!=SHOTS:raise AssertionError('Hardware shot count mismatch')
            details=result.details or {};safe={k:details[k] for k in ('cost','status','time_stamps','device_id','vendor','provider','backend','execution_time','executionTime') if k in details}
            info.update(raw_counts=counts,metadata=json.loads(json.dumps(clean_json(safe),default=str)))
            try:
                compiled=job.compiled_program()
                if compiled is not None:
                    dump(output/f'{label}_compiled_program.json',compiled.model_dump(mode='json'))
                    info['compiled_program_file']=f'{label}_compiled_program.json'
            except Exception as exc:
                info['compiled_program_unavailable']=type(exc).__name__
            trace['hardware']=score(counts,trace,ns);trace['job_id']=info['job_id']
            dump(output/f'{label}_hardware_counts.json',dict(job_id=info['job_id'],device=DEVICE,shots=SHOTS,counts=counts,metadata=info['metadata']))
            best=trace['hardware']['best_sampled_plant_state']
            if best is None:hw.update(status='FAILED',valid=False,failure=f'hour {hour}: no feasible fresh hardware sample; no repair')
            else:history.append(list(best))
            dump(output/'experiment.json',record);report(output)
            print('HARDWARE',fleet,hour,'feasible shots',trace['hardware']['valid_shots'],flush=True)
    for fleet in FLEETS:
        row=record['fleets'][str(fleet)];hw=row['hardware']
        if hw['status']=='FAILED' or hw['completed_day']:continue
        if len(hw['history'])==18:
            ns=rq.make_namespace(fleet,strict=True);stats=rq.instrument_milp(ns);h=hw['history'];on,off=ns['remaining_locks'](ns['plants'],h)
            try:
                _,tail,_=ns['solve_milp_window'](ns['plants'],ns['demand'][18:],ns['reserve'][18:],h[-1],on,off)
                schedule=np.vstack([h,tail]);checked=check_day(ns,schedule,row['reference_cost'])
                hw.update(status='COMPLETED',completed_day=True,valid=True,schedule=schedule,checked=checked,milp_tail_stats=stats)
            except RuntimeError as e:hw.update(status='FAILED',valid=False,failure=str(e),milp_tail_stats=stats)
    record['hardware_complete']=all(x['hardware']['status'] in ('COMPLETED','FAILED') for x in record['fleets'].values())
    record['settings']['physical_measurements']='recorded' if any('raw_counts' in j for j in record['jobs']) else 'none'
    dump(output/'experiment.json',record);report(output);return record


def report(output):
    try:
        import matplotlib
    except ImportError:
        import subprocess,os
        plotting_python='/home/jovyan/.qbraid/environments/qbraid_8lh8/pyenv/bin/python'
        subprocess.run([plotting_python,str(Path(__file__).resolve()),'report','--directory',str(Path(output).resolve())],
            check=True,cwd=ROOT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1'))
        return 'Report generated by the installed plotting environment'

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);r=json.loads((output/'experiment.json').read_text());x=np.arange(2);width=.32
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8));fig.subplots_adjust(bottom=.27,top=.78,wspace=.3)
    gaps=[r['fleets'][str(n)]['simulation']['checked']['gap_pct'] for n in FLEETS]
    hardware=[r['fleets'][str(n)]['hardware'] for n in FLEETS]
    axes[0].bar(x-width/2,gaps,width,label='Ideal simulation (own trajectory)',color='#175cd3')
    hgaps=[h.get('checked',{}).get('gap_pct',np.nan) for h in hardware]
    axes[0].bar(x+width/2,hgaps,width,label='Real hardware (own trajectory)',color='#ea580c')
    for i,(sg,hg,h) in enumerate(zip(gaps,hgaps,hardware)):
        axes[0].annotate(f'{sg:.4f}%',(i-width/2,sg),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        if np.isfinite(hg):axes[0].annotate(f'{hg:.4f}%',(i+width/2,hg),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        else:axes[0].annotate('FAILED' if h['status']=='FAILED' else 'PENDING',(i+width/2,0),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9,color='#b54708')
    axes[0].set(title='Full-day cost gap to MILP',ylabel='Cost gap (%)');axes[0].margins(y=.3)
    expected=[];actual=[]
    for h in hardware:
        done=[t for t in h['hourly'] if 'hardware' in t]
        expected.append(100*np.mean([t['exact_ideal_feasible_probability'] for t in done]) if done else np.nan)
        actual.append(100*np.mean([t['hardware']['valid_share'] for t in done]) if done else np.nan)
    axes[1].bar(x-width/2,expected,width,label='Ideal probability, same circuits',color='#175cd3')
    axes[1].bar(x+width/2,actual,width,label='Measured hardware samples',color='#ea580c')
    for i,h in enumerate(hardware):
        if not np.isfinite(actual[i]):axes[1].text(i,5,'HARDWARE PENDING',ha='center',fontsize=9,color='#b54708')
    axes[1].set(title='Feasibility on identical hardware circuits',ylabel='Feasible outcomes (%)',ylim=(0,108))
    for ax in axes:
        ax.set_xticks(x,['5 plants','10 plants']);ax.legend(fontsize=8,loc='upper left');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Warm-start reduced hourly: hardware vs simulation',fontsize=16,fontweight='bold',y=.97)
    fig.text(.08,.86,'Rigetti Cepheus via AWS • p=1 • 200 shots/hour • hours 13–17 • local angle training',fontsize=10,color='#475467')
    fig.text(.08,.08,'One trial per fleet; greedy warm start, matched mixer; no renewables. All methods retain the MILP prefix and tail.\nMissing hardware data are labeled PENDING, never plotted as a zero gap. Failed schedules have no valid gap.',fontsize=9,color='#475467')
    for suffix in ('png','pdf','svg'):fig.savefig(output/f'hardware_vs_simulation.{suffix}',dpi=180)
    plt.close(fig)
    lines=['| Plants | Ideal daily gap | Hardware status | Hardware daily gap | Hardware feasible shots |','|---:|---:|:---|---:|---:|']
    for n,h,g in zip(FLEETS,hardware,gaps):
        done=[t for t in h['hourly'] if 'hardware' in t];gap=h.get('checked',{}).get('gap_pct');text='—' if gap is None else f'{gap:.4f}%'
        lines.append(f"| {n} | {g:.4f}% | {h['status']} | {text} | {sum(t['hardware']['valid_shots'] for t in done)}/{len(done)*SHOTS} |")
    table='\n'.join(lines)
    body=f'''# Warm-start reduced hourly method: real hardware demonstration\n\n{table}\n\n![Hardware comparison](hardware_vs_simulation.png)\n\nDevice: `{DEVICE}`. This is the selected greedy-warm-start p=1 reduced hourly QAOA with biased initialization and its matching mixer. Angles are selected from six candidates using 256 local simulated training shots each; final candidates are measured on the real QPU under the user's 500-credit instruction and a tighter 385-credit execution cap. Demand, reserve, min-up/down locks, economic dispatch, datasets and the MILP reference match the ideal simulation. Both schedulers retain the initial full-day MILP and conditional tail. No solar, wind, block refinement, unsampled repair or post-hoc bit reversal. AWS-path counts use classical bit 0 on the right, as documented by qBraid.\n\nThe daily-gap bars compare independent complete trajectories with the same protocol. The feasibility bars compare hardware measurements against exact ideal probabilities for the very same hardware circuits/histories. One trial per fleet does not estimate repeated-run reliability. Invalid or unfinished days have no reported cost gap.\n\n## Submission plan\n\nUp to 10 jobs: five sequential quantum hours per fleet, 200 shots each. Current quote: {r['quote']['estimated_credits_per_job']:.2f} credits/job, maximum {r['quote']['max_estimated_credits']:.2f} qBraid credits. No extra paid calibration/training jobs; no automatic paid retries. Each new submission checks the price and the approved cumulative ceiling. Existing job IDs are resumed instead of resubmitted. A failed or infeasible hardware result remains a failure.\n\nRaw counts, payloads, source/data hashes, settings, job IDs and costs are saved in `experiment.json` and per-job files. `submission_plan.json` records the plan before hardware submission.\n\nSources: [qBraid AWS execution and bit ordering](https://docs.qbraid.com/v2/sdk/user-guide/providers/aws), [Rigetti submission paths](https://docs.qbraid.com/v2/sdk/user-guide/providers/rigetti).\n'''
    (output/'REPORT.md').write_text(body)
    svg=(output/'hardware_vs_simulation.svg').read_text();svg=svg[svg.index('<svg'):]
    (output/'report.html').write_text('<!doctype html><html><head><meta charset="utf-8"><style>:root{color-scheme:light dark}body{font:16px system-ui;max-width:1250px;margin:24px auto;padding:20px}svg{width:100%;height:auto}</style></head><body><h1>Warm-start hardware comparison</h1><p>Warm-start QAOA with local angle training and real Rigetti QPU sampling. Missing hardware results are labeled pending.</p>'+svg+'</body></html>')
    return table

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','report']);parser.add_argument('--directory');parser.add_argument('--approved-credits',type=float,default=0)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.directory)
    elif args.action=='run':run_hardware(args.directory,args.approved_credits)
    else:print(report(args.directory))
