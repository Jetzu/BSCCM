#!/usr/bin/env -S python3 -I -B
"""POT-free inductive KL-unbalanced entropic OT, clinical fit -> source GS."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

class UOT:
    def __init__(self,epsilon=.05,rho=.5,iterations=200,cap=2048,seed=560):
        if epsilon<=0 or rho<=0: raise ValueError('epsilon and marginal KL rho must be positive')
        self.epsilon,self.rho,self.iterations,self.cap,self.seed=epsilon,rho,iterations,cap,seed
    @torch.no_grad()
    def fit(self,clinical,source):
        q=torch.as_tensor(clinical,dtype=torch.float32)
        x=torch.as_tensor(source,dtype=torch.float32,device=q.device)
        if q.ndim!=2 or x.ndim!=2 or q.shape[1]!=x.shape[1] or min(len(q),len(x))<2: raise ValueError('need >=2 finite vectors per distribution')
        if not torch.isfinite(q).all() or not torch.isfinite(x).all(): raise ValueError('nonfinite fit vectors')
        rng=np.random.default_rng(self.seed)
        self.fit_ids=rng.permutation(len(q))[:self.cap]; self.source_ids=rng.permutation(len(x))[:self.cap]
        q=q[self.fit_ids]; self.source=x[self.source_ids].clone()
        c=torch.cdist(q,self.source).square()
        self.scale=float(c.median().clamp_min(1e-6)); c=c/self.scale
        kernel=-c/self.epsilon
        loga=q.new_full((len(q),),-np.log(len(q))); logb=q.new_full((len(self.source),),-np.log(len(self.source)))
        self.v=torch.zeros_like(logb); u=torch.zeros_like(loga)
        tau=self.rho/(self.rho+self.epsilon)
        for _ in range(self.iterations):
            u=tau*(loga-torch.logsumexp(kernel+self.v[None,:],dim=1))
            self.v=tau*(logb-torch.logsumexp(kernel+u[:,None],dim=0))
        coupling=torch.exp(kernel+u[:,None]+self.v[None,:])
        self.diagnostics=dict(mass=float(coupling.sum()),row_mass_min=float(coupling.sum(1).min()),
                              row_mass_max=float(coupling.sum(1).max()),tau=tau,scale=self.scale,
                              fit_n=len(q),source_n=len(self.source),iterations=self.iterations)
        return self
    @torch.no_grad()
    def transform(self,z,batch=256):
        z=torch.as_tensor(z,dtype=self.source.dtype,device=self.source.device)
        if not torch.isfinite(z).all(): raise ValueError('nonfinite evaluation vectors')
        # The row potential cancels from the conditional distribution. Fixed v
        # and source support extend the fitted coupling without any eval refit.
        out=[]
        for s in range(0,len(z),batch):
            logits=-torch.cdist(z[s:s+batch],self.source).square()/(self.scale*self.epsilon)+self.v[None,:]
            out.append(F.normalize(torch.softmax(logits,1)@self.source,dim=1))
        return torch.cat(out) if out else z.clone()
    def state(self):
        return dict(params=dict(epsilon=self.epsilon,rho=self.rho,iterations=self.iterations,cap=self.cap,seed=self.seed),
                    source=self.source.cpu(),v=self.v.cpu(),scale=self.scale,fit_ids=self.fit_ids,
                    source_ids=self.source_ids,diagnostics=self.diagnostics)
    @classmethod
    def load(cls,state,device='cpu'):
        obj=cls(**state['params'])
        for key in ['source','v','scale','fit_ids','source_ids','diagnostics']: setattr(obj,key,state[key])
        obj.source=obj.source.to(device); obj.v=obj.v.to(device); return obj

class TrivialAlignment:
    """Same capped cached mean embeddings as OT; no evaluation fitting."""
    def __init__(self,kind,cap=2048,seed=560,ridge=1e-4,k=10):
        if kind not in ['mean','CORAL','kNN']:raise ValueError(kind)
        self.kind,self.cap,self.seed,self.ridge,self.k=kind,cap,seed,ridge,k
    @torch.no_grad()
    def fit(self,clinical,source):
        q=torch.as_tensor(clinical,dtype=torch.float32);x=torch.as_tensor(source,dtype=torch.float32,device=q.device)
        rg=np.random.default_rng(self.seed)
        self.fit_ids=rg.permutation(len(q))[:self.cap];self.source_ids=rg.permutation(len(x))[:self.cap]
        q=q[self.fit_ids];self.source=x[self.source_ids].clone()
        self.qmean=q.mean(0);self.smean=self.source.mean(0);self.matrix=None
        if self.kind=='CORAL':
            def power(c,p):
                e,v=torch.linalg.eigh(c);return (v*e.clamp_min(self.ridge).pow(p))@v.T
            eye=torch.eye(q.shape[1],device=q.device)*self.ridge
            self.matrix=power(torch.cov(q.T)+eye,-.5)@power(torch.cov(self.source.T)+eye,.5)
        return self
    @torch.no_grad()
    def transform(self,z,batch=256):
        z=torch.as_tensor(z,dtype=self.source.dtype,device=self.source.device);out=[]
        for start in range(0,len(z),batch):
            a=z[start:start+batch]
            if self.kind=='kNN':
                ids=torch.cdist(a,self.source).topk(min(self.k,len(self.source)),largest=False).indices
                mapped=self.source[ids].mean(1)
            else:
                mapped=a-self.qmean
                if self.matrix is not None:mapped=mapped@self.matrix
                mapped=mapped+self.smean
            out.append(F.normalize(mapped,dim=1))
        return torch.cat(out) if out else z.clone()
    def state(self):
        return dict(params=dict(kind=self.kind,cap=self.cap,seed=self.seed,ridge=self.ridge,k=self.k),
                    **{k:(getattr(self,k).cpu() if torch.is_tensor(getattr(self,k)) else getattr(self,k))
                       for k in ['source','qmean','smean','matrix','fit_ids','source_ids']})
    @classmethod
    def load(cls,state,device='cpu'):
        obj=cls(**state['params'])
        for k,v in state.items():
            if k!='params':setattr(obj,k,v.to(device) if torch.is_tensor(v) else v)
        return obj

class ConditionalUOT:
    """Three PBC-anchor maps; caller routes unseen cells using frozen head C."""
    def fit(self,clinical,source,fit_class,source_class):
        self.maps={}
        for c in range(3):
            q=torch.as_tensor(clinical)[np.asarray(fit_class)==c]
            x=torch.as_tensor(source)[np.asarray(source_class)==c]
            self.maps[c]=UOT(rho=.05*.91/(1-.91)).fit(q,x)
        return self
    def transform(self,z,routes):
        z=torch.as_tensor(z); out=torch.empty_like(z)
        routes=torch.as_tensor(routes,device=z.device)
        if not torch.isin(routes,torch.tensor([0,1,2],device=z.device)).all(): raise ValueError('invalid predicted route')
        for c,m in self.maps.items():
            mask=routes==c
            if mask.any(): out[mask]=m.transform(z[mask])
        return out
    def state(self): return {c:m.state() for c,m in self.maps.items()}
    @classmethod
    def load(cls,state,device='cpu'):
        obj=cls();obj.maps={int(c):UOT.load(s,device) for c,s in state.items()};return obj

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',required=True,help='NPZ z, ids, role=fit')
    p.add_argument('--source',required=True,help='NPZ z, ids, role=source_train')
    p.add_argument('--out',required=True)
    a=p.parse_args();q=np.load(a.fit);x=np.load(a.source)
    if str(q['role'])!='fit' or str(x['role'])!='source_train': raise ValueError('OT roles violated')
    model=UOT().fit(q['z'],x['z']); Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=model.state(),fit_ids=q['ids'],source_ids=x['ids']),a.out)
    print(json.dumps(model.diagnostics))
if __name__=='__main__': main()
