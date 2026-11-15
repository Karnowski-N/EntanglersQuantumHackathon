"""Vectorized equivalent of check.check_schedule(..., ramps=False).
Only changes how independent candidate schedules are evaluated, not their rules.
"""
import numpy as np


def score_batch(plants,demand,reserve,schedules):
    s=np.asarray(schedules,dtype=int)
    assert s.ndim==3 and s.shape[1:]==(len(demand),len(plants)) and np.isin(s,[0,1]).all()
    b,T,n=s.shape
    lo=np.array([p['pmin_mw'] for p in plants]);hi=np.array([p['pmax_mw'] for p in plants])
    mc=np.array([p['marginal_cost_per_mwh'] for p in plants])
    nl=np.array([p['no_load_cost_per_h'] for p in plants]);su=np.array([p['startup_cost'] for p in plants])
    start=np.array([int(p['on_at_start']) for p in plants])
    out=s*lo
    remaining=np.asarray(demand)[None,:]-out.sum(axis=2)
    # Python sorted is stable for equal marginal costs, like the original.
    for i in sorted(range(n),key=lambda i:mc[i]):
        extra=np.minimum(np.maximum(remaining,0),s[:,:,i]*(hi[i]-lo[i]))
        out[:,:,i]+=extra;remaining-=extra
    cap=(s*hi).sum(axis=2)
    valid=(np.abs(remaining)<=1e-6).all(axis=1)&(cap>=np.asarray(demand)[None,:]+np.asarray(reserve)[None,:]-1e-6).all(axis=1)
    before=np.concatenate([np.broadcast_to(start,(b,1,n)),s[:,:-1,:]],axis=1)
    startups=((s==1)&(before==0))*su
    total=(out*mc).sum(axis=(1,2))+(s*nl).sum(axis=(1,2))+startups.sum(axis=(1,2))
    prev=np.broadcast_to(start,(b,n)).copy();since=np.full((b,n),-1,dtype=int)
    up=np.array([int(p['min_up_h']) for p in plants]);down=np.array([int(p['min_down_h']) for p in plants])
    for t in range(T):
        changed=s[:,t,:]!=prev
        need=np.where(prev==1,up,down)
        broken=changed&(since>=0)&((t-since)<need)
        valid &= ~broken.any(axis=1)
        since=np.where(changed,t,since);prev=s[:,t,:]
    return valid,total


def candidate_results(plants,demand,reserve,schedule,start,width,pair,values):
    candidates=np.repeat(np.asarray(schedule)[None,:,:],len(values),axis=0)
    bits=((np.asarray(values,dtype=np.uint64)[:,None]>>np.arange(2*width,dtype=np.uint64))&1).reshape(len(values),width,2)
    for a,i in enumerate(pair):candidates[:,start:start+width,i]=bits[:,:,a]
    valid,cost=score_batch(plants,demand,reserve,candidates)
    return {int(v):dict(valid=bool(ok),total_cost=float(c)) for v,ok,c in zip(values,valid,cost)}
