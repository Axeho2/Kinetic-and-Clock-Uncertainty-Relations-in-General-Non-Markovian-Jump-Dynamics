#!/usr/bin/env python3
"""Accelerated exact SSA for the eight-state catalytic gate.

Output defaults to os.getcwd(). Default run is the requested x10 workload:
960 random parameter positions, 3,840 trajectories per position, tau=20,000.
Use --preset validation for an explicitly smaller fresh validation run.

Core: Numba-compiled asynchronous NumPy broadcast batches + prange;
compact 2/4-state observed likelihood filter and exact untimed sequence reader.
Independent long SSA streams supply all three bound estimates (Eq19, Eq21ab).
All coefficients are sampled BEFORE the evaluation ensemble, from fixed
Gaussian laws; no pilot, no data-dependent whitening, no point rejection.

N,Sigma ordinate: mean(Q)^2 / [tau Var(Q)] (long-time form).
X ordinate: Cov(Q,Lambda_X)^2 / [tau Var(Q)] (time-tagged response).
No b rescaling of ordinates. No forced saturation or vertical/horizontal jitter.
"""
from __future__ import annotations
import os
for _v in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse, csv, hashlib, json, math, platform, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import scipy
from scipy.linalg import solve, expm
import numba
from numba import njit, prange

OUT = Path(os.getcwd())
OBS = np.array([0,0,1,1,1,1,2,2], dtype=np.int64)
FLIPS = np.array([4,2,1],dtype=np.int64)
EDGE_ID = np.full((3,3),-1,dtype=np.int64)
EDGE_ID[0,1]=0; EDGE_ID[1,0]=1; EDGE_ID[1,2]=2; EDGE_ID[2,1]=3
EDGES = [(0,1),(1,0),(1,2),(2,1)]

@dataclass
class Model:
    am:float; ap:float; b:float; eta:float; kappa:float
    rates:np.ndarray; pi:np.ndarray; G:np.ndarray; S:np.ndarray
    R:np.ndarray; K:np.ndarray; pars:np.ndarray
    activity:float; poisson:np.ndarray; eventmean:np.ndarray; TN_exact:float


def build_model(am,ap,b,eta,kappa):
    if not all(np.isfinite(v) and v>0 for v in (am,ap,b,eta,kappa)):
        raise ValueError('Rates and eta must be positive and finite.')
    rates=np.empty((8,3)); G=np.zeros((8,8)); S=np.zeros((8,8)); R=np.zeros((4,8,8))
    for h in range(8):
        s1=h>>2; s2=(h>>1)&1; z=h&1
        rates[h]=[(ap if z else am)*(eta if s1==0 else 1),
                  b*(eta if s2==0 else 1),kappa]
        G[h,h]=S[h,h]=-rates[h].sum()
        for ch in range(3):
            hh=h^int(FLIPS[ch]); G[h,hh]+=rates[h,ch]
            if ch==2:S[h,hh]+=kappa
            else:R[EDGE_ID[OBS[h],OBS[hh]],h,hh]=rates[h,ch]
    # Solve rather than assume the stationary distribution.
    A=G.T.copy(); rhs=np.zeros(8); A[-1]=1;rhs[-1]=1
    pi=solve(A,rhs)
    expected=.5*eta**OBS/(1+eta)**2
    if not np.allclose(pi,expected,rtol=1e-10,atol=1e-12):
        raise ArithmeticError('Stationary-law check failed.')
    if not np.allclose(pi[:,None]*G,(pi[:,None]*G).T,atol=1e-10):
        raise ArithmeticError('Detailed-balance check failed.')
    K=np.zeros((4,3,8,8))
    for e in range(4):
        K[e,0]=solve(-S,R[e]); K[e,1]=solve(-S,K[e,0]); K[e,2]=2*solve(-S,K[e,1])
    P=K[:,0].sum(axis=0); M=K[:,1].sum(axis=0)
    activity=float(pi@R.sum(axis=0)@np.ones(8)); rho=pi@R.sum(axis=0)/activity
    rr=np.column_stack((M.sum(axis=1),K[:2,0].sum(axis=(0,2))))
    eventmean=rho@rr
    H=solve(np.eye(8)-P+np.ones((8,1))*rho[None,:],rr-eventmean[None,:])
    C=np.zeros((2,2))
    for e in range(4):
        for i in range(8):
            for j in range(8):
                p,m,v=K[e,:,i,j]; dt,dl=H[j]-H[i]-eventmean
                dl+=float(e<2)
                C[0,0]+=rho[i]*(v+2*dt*m+dt*dt*p)
                C[0,1]+=rho[i]*dl*(m+dt*p)
                C[1,1]+=rho[i]*dl*dl*p
    TN=(C[0,0]-C[0,1]**2/C[1,1])/eventmean[0]
    pars=np.empty((4,3))
    for s in range(4):
        l0,l1=rates[2*s,:2].sum(),rates[2*s+1,:2].sum()
        mid=-(l0+l1)/2-kappa; delta=(l1-l0)/2
        pars[s]=mid,delta,np.hypot(delta,kappa)
    return Model(am,ap,b,eta,kappa,rates,pi,G,S,R,K,pars,activity,H,eventmean,float(TN))


def sequence_kernels(model):
    """Two-gate-state passage kernels; EXACT marginalization, no timed input."""
    P=np.empty((2,2,2,2)); M=np.empty_like(P); W=np.empty((2,2))
    for i in range(2):
        e=0 if i==0 else 3; ii=np.array([0,1]) if i==0 else np.array([6,7])
        W[i]=model.K[e,1][ii].sum(axis=1)
        for j in range(2):
            f=1 if j==0 else 2; jj=np.array([0,1]) if j==0 else np.array([6,7])
            P[i,j]=(model.K[e,0]@model.K[f,0])[np.ix_(ii,jj)]
            M[i,j]=(model.K[e,1]@model.K[f,0]+model.K[e,0]@model.K[f,1])[np.ix_(ii,jj)]
    init=np.empty((2,3));prior=model.pi*(OBS==1);prior/=prior.sum()
    for j,e in enumerate([1,2]):
        jj=np.array([0,1]) if j==0 else np.array([6,7])
        p=(prior@model.K[e,0])[jj]; f=(prior@model.K[e,1])[jj]; Z=p.sum()
        p/=Z;f/=Z;dm=f.sum()
        init[j]=p[0],f[0]-dm*p[0],dm
    return P,M,W,init

@njit(inline='always')
def drawstate(pi):
    u=np.random.random();cs=0.
    for i in range(8):
        cs+=pi[i]
        if u<cs:return i
    return 7

@njit(inline='always')
def conditional_init(pi,x,p):
    Z=0.
    for h in range(8):
        p[h]=pi[h] if OBS[h]==x else 0.;Z+=p[h]
    for h in range(8):p[h]/=Z

@njit(inline='always')
def filter_flow(p,d,wp,wd,w,pars,kappa):
    """Analytic two-by-two no-visible-event propagator AND its speed derivative.

    p,d updated in place. Returns the log-speed derivative of log survival.
    No realized hidden states, gate events or switch identities enter this filter.
    """
    shift=-1.e300
    for s in range(4):
        if p[2*s]+p[2*s+1]>0:shift=max(shift,pars[s,0]+pars[s,2])
    for h in range(8):wp[h]=0.;wd[h]=0.
    for s in range(4):
        i=2*s
        if p[i]+p[i+1]==0 and d[i]==0 and d[i+1]==0:continue
        mid,delta,nu=pars[s]
        ep=math.exp((mid+nu-shift)*w);em=math.exp((mid-nu-shift)*w)
        co=.5*(ep+em);si=(ep-em)/(2*nu)
        E00=co+si*delta;E01=si*kappa;E11=co-si*delta
        a0=E00*p[i]+E01*p[i+1];a1=E01*p[i]+E11*p[i+1]
        wp[i]=a0;wp[i+1]=a1
        wd[i]=E00*d[i]+E01*d[i+1]+w*((mid+delta)*a0+kappa*a1)
        wd[i+1]=E01*d[i]+E11*d[i+1]+w*(kappa*a0+(mid-delta)*a1)
    Z=0.;D=0.
    for h in range(8):Z+=wp[h];D+=wd[h]
    score=D/Z
    for h in range(8):p[h]=wp[h]/Z;d[h]=wd[h]/Z-score*p[h]
    return score

@njit(inline='always')
def filter_jump(p,d,wp,wd,x,y,rates):
    for h in range(8):wp[h]=0.;wd[h]=0.
    for h in range(8):
        if OBS[h]!=x:continue
        for ch in range(2):
            hh=h^FLIPS[ch]
            if OBS[hh]==y:
                r=rates[h,ch];wp[hh]+=r*p[h];wd[hh]+=r*(d[h]+p[h])
    Z=0.;D=0.
    for h in range(8):Z+=wp[h];D+=wd[h]
    alpha=D/Z
    for h in range(8):p[h]=wp[h]/Z;d[h]=wd[h]/Z-alpha*p[h]
    return alpha

@njit(inline='always')
def seq_mean_update(p0,f0,i,j,P,M):
    p1=1-p0
    v0=p0*P[i,j,0,0]+p1*P[i,j,1,0]
    v1=p0*P[i,j,0,1]+p1*P[i,j,1,1];Z=v0+v1
    a0=f0*(P[i,j,0,0]-P[i,j,1,0])+p0*M[i,j,0,0]+p1*M[i,j,1,0]
    a1=f0*(P[i,j,0,1]-P[i,j,1,1])+p0*M[i,j,0,1]+p1*M[i,j,1,1]
    dm=(a0+a1)/Z;pp=v0/Z
    return pp,a0/Z-dm*pp,dm

@njit(inline='always')
def qseq_final(q,p,ep,h,W):
    if OBS[h]==1 and ep>=0:return q+p*W[ep,0]+(1-p)*W[ep,1]
    return q

@njit(cache=True)
def reference_batch(rates,pi,pars,kappa,P,M,W,init,tau,n,seed):
    """Asynchronous exact SSA in NumPy-broadcasted independent lanes.

    Output: L,U,qSigma,Lambda_X,observed bracket,micro score,gate count.
    Half-duration data use the same nominal readout and include censoring.
    """
    np.random.seed(seed)
    h=np.empty(n,np.int64);initial=np.empty(n,np.int64)
    clock=np.zeros(n);last_time=np.zeros(n);active=np.ones(n,np.bool_);left=n
    half_saved=np.zeros(n,np.bool_);out=np.zeros((n,7));half=np.zeros((n,7))
    pp=np.zeros((n,8));dd=np.zeros_like(pp);wp=np.zeros_like(pp);wd=np.zeros_like(pp)
    hp=np.zeros(8);hd=np.zeros(8);hw=np.zeros(8);hv=np.zeros(8)
    sp=np.full(n,.5);sf=np.zeros(n);qs=np.zeros(n);ep=np.empty(n,np.int64)
    tot=rates.sum(axis=1);cuts=np.empty((8,2))
    for h0 in range(8):
        cuts[h0,0]=rates[h0,0];cuts[h0,1]=rates[h0,0]+rates[h0,1]
    for a in range(n):
        h[a]=drawstate(pi);initial[a]=h[a];x=OBS[h[a]]
        conditional_init(pi,x,pp[a]);ep[a]=-1 if x==1 else x//2
    while left:
        # Broadcasting (n,1) random thresholds against (n,2) cumulative rates.
        uu=np.random.random((n,2));escape=tot[h]
        wait=-np.log(np.maximum(uu[:,0],1.e-300))/escape
        kind=np.sum((uu[:,1]*escape)[:,None]>=cuts[h,:],axis=1)
        tn=clock+wait
        for a in range(n):
            if not active[a]:continue
            if not half_saved[a] and tn[a]>=tau/2:
                half[a]=out[a]
                half[a,2]=qseq_final(qs[a],sp[a],ep[a],h[a],W)
                for k in range(8):hp[k]=pp[a,k];hd[k]=dd[a,k]
                half[a,3]+=filter_flow(hp,hd,hw,hv,tau/2-last_time[a],pars,kappa)
                half[a,5]-=escape[a]*(tau/2-clock[a]);half_saved[a]=True
            if tn[a]>=tau:
                out[a,2]=qseq_final(qs[a],sp[a],ep[a],h[a],W)
                out[a,3]+=filter_flow(pp[a],dd[a],wp[a],wd[a],tau-last_time[a],pars,kappa)
                out[a,5]-=escape[a]*(tau-clock[a]);active[a]=False;left-=1
                continue
            out[a,5]+=1.-escape[a]*wait[a]
            clock[a]=tn[a];old=h[a];ch=kind[a];h[a]=old^FLIPS[ch]
            if ch==2:out[a,6]+=1;continue
            x=OBS[old];y=OBS[h[a]]
            if x==0 or y==0:out[a,0]+=1
            else:out[a,1]+=1
            w=tn[a]-last_time[a]
            out[a,3]+=filter_flow(pp[a],dd[a],wp[a],wd[a],w,pars,kappa)
            alpha=filter_jump(pp[a],dd[a],wp[a],wd[a],x,y,rates)
            out[a,3]+=alpha;out[a,4]+=alpha*alpha;last_time[a]=tn[a]
            if y!=1:
                j=y//2
                if ep[a]<0:sp[a],sf[a],qs[a]=init[j]
                else:
                    sp[a],sf[a],dm=seq_mean_update(sp[a],sf[a],ep[a],j,P,M)
                    qs[a]+=dm
                ep[a]=j
    return out,half,initial,h

@njit(parallel=True,cache=True)
def reference_ensemble(rates,pi,pars,kappa,P,M,W,init,tau,n,batch,seeds):
    out=np.empty((n,7));half=np.empty_like(out);ini=np.empty(n,np.int64);fin=np.empty(n,np.int64)
    for k in prange(len(seeds)):
        lo=k*batch;hi=min(lo+batch,n)
        a,b,c,d=reference_batch(rates,pi,pars,kappa,P,M,W,init,tau,hi-lo,seeds[k])
        out[lo:hi]=a;half[lo:hi]=b;ini[lo:hi]=c;fin[lo:hi]=d
    return out,half,ini,fin

@njit(inline='always')
def sequence_var_step(p,f,s,e,K,work):
    for j in range(8):
        a0=a1=a2=0.
        for i in range(8):
            k0=K[e,0,i,j]
            if k0==0:continue
            k1=K[e,1,i,j];k2=K[e,2,i,j]
            a0+=p[i]*k0;a1+=f[i]*k0+p[i]*k1
            a2+=s[i]*k0+2*f[i]*k1+p[i]*k2
        work[0,j]=a0;work[1,j]=a1;work[2,j]=a2
    Z=0.;fm=0.;sm=0.
    for j in range(8):Z+=work[0,j];fm+=work[1,j];sm+=work[2,j]
    dm=fm/Z;dv=sm/Z-dm*dm
    for j in range(8):
        p[j]=work[0,j]/Z;v=work[1,j]/Z
        s[j]=work[2,j]/Z-2*dm*v+(dm*dm-dv)*p[j]
        f[j]=v-dm*p[j]
    return dm,dv

@njit(cache=True)
def bound_stream(rates,pi,pars,kappa,K,H,eventmean,events,seed):
    """One true stationary SSA stream; all displayed bound statistics sampled."""
    np.random.seed(seed);h=drawstate(pi);prev=h
    p=np.zeros(8);conditional_init(pi,OBS[h],p);d=np.zeros(8)
    wp=np.zeros(8);wd=np.zeros(8);ps=p.copy();fs=np.zeros(8);ss=np.zeros(8);work=np.zeros((3,8))
    out=np.zeros((3,9));stop=np.array([events//4,events//2,events],np.int64)
    t=w=cm=cv=a2=ctt=ctl=cll=0.;n=idx=0
    while n<events:
        r0,r1,rz=rates[h];total=r0+r1+rz
        dt=-math.log(max(np.random.random(),1e-300))/total
        t+=dt;w+=dt;u=np.random.random()*total
        ch=0 if u<r0 else (1 if u<r0+r1 else 2)
        old=h;h^=FLIPS[ch]
        if ch==2:continue
        x=OBS[old];y=OBS[h];e=EDGE_ID[x,y]
        filter_flow(p,d,wp,wd,w,pars,kappa)
        alpha=filter_jump(p,d,wp,wd,x,y,rates);a2+=alpha*alpha
        dm,dv=sequence_var_step(ps,fs,ss,e,K,work);cm+=dm;cv+=dv
        rt=w-eventmean[0]+H[h,0]-H[prev,0]
        rl=(1. if e<2 else 0.)-eventmean[1]+H[h,1]-H[prev,1]
        ctt+=rt*rt;ctl+=rt*rl;cll+=rl*rl
        prev=h;w=0.;n+=1
        if n==stop[idx]:
            out[idx]=np.array([n,t,cm,cv,a2,ctt,ctl,cll,0.])
            idx+=1
    return out

@njit(parallel=True,cache=True)
def bound_ensemble(rates,pi,pars,kappa,K,H,em,events,seeds):
    a=np.empty((len(seeds),3,9))
    for k in prange(len(seeds)):
        a[k]=bound_stream(rates,pi,pars,kappa,K,H,em,events,seeds[k])
    return a


def seeds(seed,index,purpose,n):
    r=np.random.default_rng(np.random.SeedSequence([int(seed),int(index),int(purpose)]))
    return r.integers(0,2**32,n,dtype=np.uint32)


def reference_sample(model,n,tau,seed,index,purpose=1,batch=32):
    P,M,W,init=sequence_kernels(model)
    return reference_ensemble(model.rates,model.pi,model.pars,model.kappa,P,M,W,init,
                           tau,n,batch,seeds(seed,index,purpose,(n+batch-1)//batch))


def bound_values(raw):
    v=raw.sum(axis=0)
    _,t,_,cv,a2,tt,tl,ll,_=v
    # Eq21b: empirical quadratic variation of Poisson-corrected increments.
    TN=(tt-tl*tl/ll)/t
    return np.array([1/TN,t/cv,a2/t])


def mc_bounds(model,events,streams,seed,index):
    raw=bound_ensemble(model.rates,model.pi,model.pars,model.kappa,model.K,
          model.poisson,model.eventmean,events,seeds(seed,index,41,streams))
    vals=[];errs=[]
    for j in range(3):
        z=raw[:,j,:];v=bound_values(z)
        jk=np.array([bound_values(np.delete(z,k,axis=0)) for k in range(streams)])
        se=np.sqrt((streams-1)/streams*((jk-jk.mean(axis=0))**2).sum(axis=0))
        vals.append(v);errs.append(se)
    return np.array(vals),np.array(errs),raw


# ---------------------------------------------------------------------------
# Optimized conditional likelihood: only 2 states at x=0/2 and 4 states at x=1.
# Arrays do NOT contain the actual hidden microstate; they are observer beliefs.
# ---------------------------------------------------------------------------
@njit(inline='always')
def _flow_pair(p0,p1,d0,d1,w,mid,delta,nu,kappa,shift):
    # Common exponential factor exp(shift*w) cancels in normalized updates.
    # The derivative below uses the ORIGINAL generator, not a differentiated
    # numerical shift. This keeps the true log-speed likelihood derivative.
    ep=math.exp((mid+nu-shift)*w)
    em=math.exp((mid-nu-shift)*w)
    co=.5*(ep+em);si=(ep-em)/(2.*nu)
    E00=co+si*delta;E01=si*kappa;E11=co-si*delta
    a0=E00*p0+E01*p1; a1=E01*p0+E11*p1
    b0=E00*d0+E01*d1+w*((mid+delta)*a0+kappa*a1)
    b1=E01*d0+E11*d1+w*(kappa*a0+(mid-delta)*a1)
    return a0,a1,b0,b1

@njit(inline='always')
def compact_update(p,d,x,y,w,pars,kappa,am,ap,b,eta):
    """Combined survival/event update; y=-1 means terminal survival only.

    p,d length 4. x=1 positions (01-,01+,10-,10+); endpoints use positions 0,1.
    Returns log-likelihood score increment and event alpha=1+mu.
    """
    if x==1:
        shift=max(pars[1,0]+pars[1,2],pars[2,0]+pars[2,2])
        a0,a1,d0,d1=_flow_pair(p[0],p[1],d[0],d[1],w,
            pars[1,0],pars[1,1],pars[1,2],kappa,shift)
        a2,a3,d2,d3=_flow_pair(p[2],p[3],d[2],d[3],w,
            pars[2,0],pars[2,1],pars[2,2],kappa,shift)
    else:
        m=0 if x==0 else 3
        shift=pars[m,0]+pars[m,2]
        a0,a1,d0,d1=_flow_pair(p[0],p[1],d[0],d[1],w,
            pars[m,0],pars[m,1],pars[m,2],kappa,shift)
        a2=a3=d2=d3=0.
    surv=(d0+d1+d2+d3)/(a0+a1+a2+a3)
    if y<0:
        return surv,0.
    # Collapse survival and jump normalizations to a single normalization.
    if x==0:
        v0=a0*eta*b;v1=a1*eta*b;v2=a0*eta*am;v3=a1*eta*ap
        u0=(d0+a0)*eta*b;u1=(d1+a1)*eta*b
        u2=(d0+a0)*eta*am;u3=(d1+a1)*eta*ap
    elif x==2:
        v0=a0*am;v1=a1*ap;v2=a0*b;v3=a1*b
        u0=(d0+a0)*am;u1=(d1+a1)*ap
        u2=(d0+a0)*b;u3=(d1+a1)*b
    elif y==0:
        v0=a0*b+a2*am;v1=a1*b+a3*ap;v2=v3=0.
        u0=(d0+a0)*b+(d2+a2)*am;u1=(d1+a1)*b+(d3+a3)*ap;u2=u3=0.
    else:
        v0=eta*(a0*am+a2*b);v1=eta*(a1*ap+a3*b);v2=v3=0.
        u0=eta*((d0+a0)*am+(d2+a2)*b)
        u1=eta*((d1+a1)*ap+(d3+a3)*b);u2=u3=0.
    Z=v0+v1+v2+v3
    score=(u0+u1+u2+u3)/Z
    p[0]=v0/Z;p[1]=v1/Z;p[2]=v2/Z;p[3]=v3/Z
    d[0]=u0/Z-score*p[0];d[1]=u1/Z-score*p[1]
    d[2]=u2/Z-score*p[2];d[3]=u3/Z-score*p[3]
    return score,score-surv

@njit(cache=True)
def fast_batch(rates,pi,pars,kappa,P,M,W,init,tau,n,seed,am,ap,b,eta):
    """Exact, asynchronous SSA. Broadcasting handles n independent next events.

    No fixed-step time discretization. Each lane has an independent clock.
    Only end-point readouts are stored; no trajectory-sized array is allocated.
    """
    np.random.seed(seed)
    h=np.empty(n,np.int64);ini=np.empty(n,np.int64)
    pp=np.zeros((n,4));dd=np.zeros((n,4))
    t=np.zeros(n);last_time=np.zeros(n);active=np.ones(n,np.bool_)
    half_saved=np.zeros(n,np.bool_);out=np.zeros((n,7));half=np.zeros_like(out)
    sp=np.full(n,.5);sf=np.zeros(n);qs=np.zeros(n);ep=np.empty(n,np.int64)
    hp=np.zeros(4);hd=np.zeros(4)
    total=rates.sum(axis=1);cuts=np.empty((8,2))
    for m in range(8):
        cuts[m,0]=rates[m,0];cuts[m,1]=rates[m,0]+rates[m,1]
    for i in range(n):
        h[i]=drawstate(pi);ini[i]=h[i];x=OBS[h[i]]
        if x==1:
            pp[i,0]=.25;pp[i,1]=.25;pp[i,2]=.25;pp[i,3]=.25;ep[i]=-1
        else:pp[i,0]=.5;pp[i,1]=.5;ep[i]=x//2
    left=n
    while left:
        # Broadcast (n,1) thresholds against (n,2) cumulative rates.
        # Numba compiles these operations inside the event loop.
        uu=np.random.random((n,2))
        escape=total[h]
        waits=-np.log(np.maximum(uu[:,0],1.e-300))/escape
        channels=np.sum((uu[:,1]*escape)[:,None]>=cuts[h,:],axis=1)
        for i in range(n):
            if not active[i]:continue
            tn=t[i]+waits[i];x=OBS[h[i]]
            if not half_saved[i] and tn>=tau*.5:
                half[i]=out[i]
                half[i,2]=qseq_final(qs[i],sp[i],ep[i],h[i],W)
                for j in range(4):hp[j]=pp[i,j];hd[j]=dd[i,j]
                ds,_=compact_update(hp,hd,x,-1,tau*.5-last_time[i],pars,kappa,am,ap,b,eta)
                half[i,3]+=ds;half[i,5]-=escape[i]*(tau*.5-t[i])
                half_saved[i]=True
            if tn>=tau:
                out[i,2]=qseq_final(qs[i],sp[i],ep[i],h[i],W)
                ds,_=compact_update(pp[i],dd[i],x,-1,tau-last_time[i],pars,kappa,am,ap,b,eta)
                out[i,3]+=ds;out[i,5]-=escape[i]*(tau-t[i]);active[i]=False;left-=1
                continue
            out[i,5]+=1.-escape[i]*waits[i]
            t[i]=tn;ch=channels[i];h[i]^=FLIPS[ch]
            if ch==2:out[i,6]+=1;continue
            y=OBS[h[i]]
            if x==0 or y==0:out[i,0]+=1
            else:out[i,1]+=1
            ds,alpha=compact_update(pp[i],dd[i],x,y,tn-last_time[i],pars,kappa,am,ap,b,eta)
            out[i,3]+=ds;out[i,4]+=alpha*alpha;last_time[i]=tn
            if y!=1:
                j=y//2
                if ep[i]<0:sp[i],sf[i],qs[i]=init[j]
                else:
                    sp[i],sf[i],dm=seq_mean_update(sp[i],sf[i],ep[i],j,P,M)
                    qs[i]+=dm
                ep[i]=j
    return out,half,ini,h

@njit(parallel=True,cache=True)
def fast_ensemble(rates,pi,pars,kappa,P,M,W,init,tau,n,batch,seed_array,am,ap,b,eta):
    F=np.empty((n,7));half=np.empty_like(F);ini=np.empty(n,np.int64);fin=np.empty(n,np.int64)
    for j in prange(len(seed_array)):
        lo=j*batch;hi=min(lo+batch,n)
        a,c,i,f=fast_batch(rates,pi,pars,kappa,P,M,W,init,tau,hi-lo,seed_array[j],am,ap,b,eta)
        F[lo:hi]=a;half[lo:hi]=c;ini[lo:hi]=i;fin[lo:hi]=f
    return F,half,ini,fin

def fast_sample(model,n,tau,seed,index,purpose=1,batch=32):
    P,M,W,init=sequence_kernels(model)
    return fast_ensemble(model.rates,model.pi,model.pars,model.kappa,P,M,W,init,tau,n,batch,
        seeds(seed,index,purpose,(n+batch-1)//batch),model.am,model.ap,model.b,model.eta)

@njit(cache=True)
def fused_batch(rates,pi,pars,kappa,P,M,W,init,tau,n,seed,am,ap,b,eta):
    """Alternative JIT-fused scalar event kernel, one independent lane at a time.

    Useful when NumPy temporary arrays dominate. Same exact SSA law as the
    broadcast kernel, but different random-number consumption/order.
    """
    np.random.seed(seed)
    F=np.zeros((n,7));half=np.zeros_like(F);initial=np.empty(n,np.int64);final=np.empty(n,np.int64)
    p=np.empty(4);d=np.empty(4)
    for lane in range(n):
        h=drawstate(pi);initial[lane]=h;x=OBS[h]
        for j in range(4):p[j]=0.;d[j]=0.
        if x==1:
            for j in range(4):p[j]=.25
            ep=-1
        else:p[0]=.5;p[1]=.5;ep=x//2
        sp=.5;sf=0.;qs=0.;clock=0.;last=0.
        L=U=score=bracket=microscore=gate=0.;half_saved=False
        while True:
            r0,r1,rz=rates[h];total=r0+r1+rz
            wait=-math.log(max(np.random.random(),1.e-300))/total
            selector=np.random.random()*total
            tn=clock+wait
            if not half_saved and tn>=tau*.5:
                ds,_=compact_update(p,d,x,-1,tau*.5-last,pars,kappa,am,ap,b,eta)
                half[lane,0]=L;half[lane,1]=U
                half[lane,2]=qseq_final(qs,sp,ep,h,W)
                half[lane,3]=score+ds;half[lane,4]=bracket
                half[lane,5]=microscore-total*(tau*.5-clock);half[lane,6]=gate
                half_saved=True
            if tn>=tau:
                ds,_=compact_update(p,d,x,-1,tau-last,pars,kappa,am,ap,b,eta)
                F[lane,0]=L;F[lane,1]=U;F[lane,2]=qseq_final(qs,sp,ep,h,W)
                F[lane,3]=score+ds;F[lane,4]=bracket
                F[lane,5]=microscore-total*(tau-clock);F[lane,6]=gate
                final[lane]=h
                break
            microscore+=1.-total*wait;clock=tn
            if selector<r0:ch=0
            elif selector<r0+r1:ch=1
            else:ch=2
            h^=FLIPS[ch]
            if ch==2:gate+=1.;continue
            y=OBS[h]
            if x==0 or y==0:L+=1.
            else:U+=1.
            ds,alpha=compact_update(p,d,x,y,tn-last,pars,kappa,am,ap,b,eta)
            score+=ds;bracket+=alpha*alpha;last=tn;x=y
            if y!=1:
                j=y//2
                if ep<0:sp,sf,qs=init[j]
                else:
                    sp,sf,dm=seq_mean_update(sp,sf,ep,j,P,M);qs+=dm
                ep=j
    return F,half,initial,final

@njit(parallel=True,cache=True)
def fused_ensemble(rates,pi,pars,kappa,P,M,W,init,tau,n,batch,seed_array,am,ap,b,eta):
    F=np.empty((n,7));half=np.empty_like(F);ini=np.empty(n,np.int64);fin=np.empty(n,np.int64)
    for j in prange(len(seed_array)):
        lo=j*batch;hi=min(lo+batch,n)
        a,c,i,f=fused_batch(rates,pi,pars,kappa,P,M,W,init,tau,hi-lo,seed_array[j],am,ap,b,eta)
        F[lo:hi]=a;half[lo:hi]=c;ini[lo:hi]=i;fin[lo:hi]=f
    return F,half,ini,fin

def fused_sample(model,n,tau,seed,index,purpose=1,batch=8):
    P,M,W,init=sequence_kernels(model)
    return fused_ensemble(model.rates,model.pi,model.pars,model.kappa,P,M,W,init,tau,n,batch,
        seeds(seed,index,purpose,(n+batch-1)//batch),model.am,model.ap,model.b,model.eta)

# ---------------------------------------------------------------------------
# Many random observables per ensemble. Weights are independent of the data.
# ---------------------------------------------------------------------------
def observable_bases(F, eta, b):
    C=F[:,1]-eta*F[:,0]
    # b*qSigma is only a fixed choice of feature units, NOT a y-axis rescaling.
    return (F[:,:2], np.column_stack((b*F[:,2], C)),
            np.column_stack((F[:,3], C)))

def draw_weights(seed,index,sizes):
    result=[]
    for group,size in enumerate(sizes):
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,900+group]))
        w=rng.normal(size=(size,2))
        w/=np.linalg.norm(w,axis=1,keepdims=True)
        result.append(w)
    return result

def evaluate_many(F, weights, tau, eta, b):
    """NumPy broadcasting over trajectories AND observable coefficients.

    Each column is an independently chosen observable, not an independent
    trajectory ensemble. Standard errors account for ensemble fluctuations
    for each fixed readout; points using the same F have correlated errors.
    """
    result=[];score=F[:,3]-F[:,3].mean();n=len(F)
    for group,(base,W) in enumerate(zip(observable_bases(F,eta,b),weights)):
        # A (n_trajectories,1) column broadcasts against (1,n_observables).
        Q=base[:,0,None]*W[None,:,0]+base[:,1,None]*W[None,:,1]
        mean=Q.mean(axis=0);q=Q-mean[None,:]
        variance=(q*q).sum(axis=0)/(n-1)
        if not np.all(variance>0):
            raise ArithmeticError('A readout has zero variance; increase trajectories, do not fabricate points.')
        if group<2:
            response=mean
            influence=(2*mean/(tau*variance))[None,:]*q
            influence-=(mean**2/(tau*variance**2))[None,:]*((q*q)-(q*q).mean(axis=0))
        else:
            response=(q*score[:,None]).sum(axis=0)/(n-1)
            influence=(2*response/(tau*variance))[None,:]*((q*score[:,None])-(q*score[:,None]).mean(axis=0))
            influence-=(response**2/(tau*variance**2))[None,:]*((q*q)-(q*q).mean(axis=0))
        value=response**2/(tau*variance)
        se=influence.std(axis=0,ddof=1)/np.sqrt(n)
        result.append(np.column_stack((value,se,mean,variance,response)))
    return result

def write_csv(path, rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def atomic_npz(path, **arrays):
    path=Path(path);temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**arrays)
    os.replace(temporary,path)

def _warmup(model,kernel,threads):
    numba.set_num_threads(threads)
    fn=fast_sample if kernel=='broadcast' else fused_sample
    fn(model,64,5.,31999,0,batch=32)
    mc_bounds(model,32,2,31998,0)

def run_sampling(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    work=root/'checkpoints';work.mkdir(exist_ok=True)
    checks=root/'checks';checks.mkdir(exist_ok=True)
    threads=min(args.threads,numba.config.NUMBA_NUM_THREADS)
    numba.set_num_threads(threads)
    rng=np.random.default_rng(args.seed)
    # Parameter randomization precedes simulation; no post-hoc x-jitter.
    xs=np.sort(rng.uniform(args.xmin,args.xmax,args.points))
    bx=np.linspace(args.xmin,args.xmax,args.bound_points)
    sizes=(args.count_weights,args.sequence_weights,args.full_weights)
    config={k:v for k,v in vars(args).items() if k not in ('output','resume','benchmark','self_test')}
    config.update(actual_threads=threads,ordinate_N_Sigma='mean(Q)^2/(tau*Var(Q))',
        ordinate_X='Cov(Q,Lambda_X)^2/(tau*Var(Q))',no_b_rescaling=True,
        weights='independent fixed N(0,1) directions in (L,U),(b*qSigma,U-eta*L),(Lambda_X,U-eta*L)',
        no_empirical_weight_scaling=True,score_contains_terminal_survival=True,
        python=platform.python_version(),numpy=np.__version__,numba=numba.__version__)
    hashed={k:v for k,v in config.items() if k not in ('actual_threads','threads','python','numpy','numba','preset')}
    signature=hashlib.sha256(json.dumps(hashed,sort_keys=True).encode()).hexdigest()
    cfgfile=root/'run_config.json'
    if cfgfile.exists():
        old=json.loads(cfgfile.read_text())
        if not args.resume:
            raise FileExistsError('Output contains a run. Use --resume or a fresh --output directory.')
        if old.get('signature')!=signature:
            raise ValueError('Checkpoint parameters differ from requested run. Do not mix ensembles.')
    config.update(signature=signature,complete=False,parameter_values=xs.tolist())
    cfgfile.write_text(json.dumps(config,indent=2))
    np.savetxt(root/'parameter_values.dat',xs,header='actual_a_plus_over_b')
    fn=fast_sample if args.kernel=='broadcast' else fused_sample
    model=build_model(args.a_minus,((args.xmin+args.xmax)/2)*args.b,args.b,args.eta,args.kappa)
    t=time.perf_counter();_warmup(model,args.kernel,threads)
    print(f'JIT warm-up: {time.perf_counter()-t:.2f} s; threads={threads}',flush=True)
    start=time.perf_counter();bound_rows=[];conv=[]
    for k,x in enumerate(bx):
        path=work/f'bound_{k:04d}.npz'
        if args.resume and path.exists():
            with np.load(path) as z:vv,ee,raw,exact=z['values'],z['errors'],z['raw'],float(z['TN_exact'])
        else:
            m=build_model(args.a_minus,x*args.b,args.b,args.eta,args.kappa)
            vv,ee,raw=mc_bounds(m,args.events,args.streams,args.seed,k)
            exact=m.TN_exact
            atomic_npz(path,values=vv,errors=ee,raw=raw,TN_exact=exact,x=x)
        bound_rows.append([x,*vv[-1],*ee[-1]])
        for j in range(3):
            conv.append(dict(x=x,events=int(raw[0,j,0]),BN=vv[j,0],BS=vv[j,1],BX=vv[j,2],
                se_N=ee[j,0],se_Sigma=ee[j,1],se_X=ee[j,2],BN_matrix_check=1/exact))
        if (k+1)%8==0 or k==len(bx)-1:
            print(f'Bounds {k+1}/{len(bx)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    np.savetxt(root/'bounds_curves.dat',bound_rows,header='x BN BS BX seN seSigma seX',fmt='%.12g')
    write_csv(checks/'bound_convergence.csv',conv)
    data=[[],[],[]];fullrows=[];audits=[];family=('N','Sigma','X')
    filenames=('count_samples.dat','sequence_samples.dat','full_samples.dat')
    sample_start=time.perf_counter();total_events=0;new_events=0
    newly_completed=0
    for k,x in enumerate(xs):
        path=work/f'ensemble_{k:04d}.npz'
        W=draw_weights(args.seed,k,sizes)  # BEFORE the evaluation data exist.
        if args.resume and path.exists():
            with np.load(path) as z:
                results=[z[f'result_{g}'] for g in range(3)]
                halves=[z[f'half_result_{g}'] for g in range(3)]
                audit=json.loads(str(z['audit']))
        else:
            tic=time.perf_counter()
            m=build_model(args.a_minus,x*args.b,args.b,args.eta,args.kappa)
            F,HF,ini,fin=fn(m,args.trajectories,args.tau,args.seed,k,purpose=21,batch=args.batch_size)
            results=evaluate_many(F,W,args.tau,args.eta,args.b)
            halves=evaluate_many(HF,W,args.tau/2,args.eta,args.b)
            event_count=int(F[:,0].sum()+F[:,1].sum()+F[:,6].sum())
            audit=dict(parameter=k,x=float(x),trajectories=args.trajectories,tau=args.tau,
                microscopic_events=event_count,score_mean=float(F[:,3].mean()),
                score_mean_se=float(F[:,3].std(ddof=1)/np.sqrt(len(F))),
                score_variance_rate=float(F[:,3].var(ddof=1)/args.tau),
                bracket_rate=float(F[:,4].mean()/args.tau),
                count_L_rate=float(F[:,0].mean()/args.tau),count_U_rate=float(F[:,1].mean()/args.tau),
                sequence_mean_rate=float(F[:,2].mean()/args.tau),
                sequence_half_mean_rate=float(HF[:,2].mean()/(args.tau/2)),
                seconds=float(time.perf_counter()-tic))
            payload={'audit':np.array(json.dumps(audit)),'x':x,'feature_mean':F.mean(axis=0),
                'feature_covariance':np.cov(F,rowvar=False),'half_feature_mean':HF.mean(axis=0),
                'half_feature_covariance':np.cov(HF,rowvar=False)}
            for g in range(3):payload[f'result_{g}']=results[g];payload[f'half_result_{g}']=halves[g];payload[f'weights_{g}']=W[g]
            if args.save_ensembles:payload.update(F=F,half=HF,initial=ini,final=fin)
            atomic_npz(path,**payload)
            newly_completed+=1;new_events+=event_count
        total_events+=audit['microscopic_events'];audits.append(audit)
        for g in range(3):
            for j,r in enumerate(results[g]):
                # All actual values retained, including any statistical overshoot.
                data[g].append([x,r[0],r[1],halves[g][j,0],*W[g][j]])
                fullrows.append(dict(parameter=k,x=float(x),family=family[g],observable=j,
                    value=float(r[0]),se=float(r[1]),mean=float(r[2]),variance=float(r[3]),response=float(r[4]),
                    half_value=float(halves[g][j,0]),half_se=float(halves[g][j,1]),
                    w0=float(W[g][j,0]),w1=float(W[g][j,1])))
        if (k+1)%16==0 or k==0 or k==len(xs)-1:
            elapsed=time.perf_counter()-sample_start
            rate=new_events/max(elapsed,1e-9)
            for g in range(3):np.savetxt(root/filenames[g],data[g],header='x value se half_value w0 w1',fmt='%.12g')
            status=dict(complete=False,completed_parameters=k+1,target_parameters=len(xs),
                newly_completed=newly_completed,elapsed_seconds=elapsed,
                estimated_remaining_seconds=(len(xs)-k-1)*elapsed/max(newly_completed,1),events_per_second=rate)
            (checks/'progress.json').write_text(json.dumps(status,indent=2))
            print(f'Scatter {k+1}/{len(xs)}; {elapsed:.1f}s; {rate/1e6:.2f} M events/s',flush=True)
    write_csv(root/'all_observables.csv',fullrows);write_csv(checks/'trajectory_checks.csv',audits)
    summary=dict(complete=True,preset=args.preset,kernel=args.kernel,parameter_positions=len(xs),
        trajectories_per_position=args.trajectories,physical_duration=args.tau,kappa_tau=args.kappa*args.tau,
        total_trajectories=len(xs)*args.trajectories,total_microscopic_events=total_events,
        random_weights_per_parameter=dict(zip(family,sizes)),scatter_points=dict(zip(family,map(len,data))),
        below_ymin={family[g]:int(sum(r[1]<.01 for r in data[g])) for g in range(3)},
        finite_statistical_overshoots_not_removed=True,
        mean_score_z_max=float(max(abs(a['score_mean'])/a['score_mean_se'] for a in audits)),
        score_variance_over_bracket_median=float(np.median([a['score_variance_rate']/a['bracket_rate'] for a in audits])),
        elapsed_seconds_this_invocation=time.perf_counter()-start,
        ensemble_seconds_recorded=sum(float(a['seconds']) for a in audits),threads=threads)
    (checks/'summary.json').write_text(json.dumps(summary,indent=2))
    (checks/'progress.json').write_text(json.dumps(dict(complete=True,completed_parameters=len(xs)),indent=2))
    config['complete']=True;cfgfile.write_text(json.dumps(config,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

def benchmark(args):
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    threads=min(args.threads,numba.config.NUMBA_NUM_THREADS);numba.set_num_threads(threads)
    m=build_model(1.,6.5,1.,2.,.05)
    tests=[]
    for ap in (1.,3.,6.5,10.):
        mm=build_model(1.,ap,1.,2.,.05)
        a=reference_sample(mm,128,100.,617,0,batch=32)
        b=fast_sample(mm,128,100.,617,0,batch=32)
        differences=[float(np.max(abs(v-w))) for v,w in zip(a,b)]
        if not all(np.allclose(v,w,rtol=2e-10,atol=2e-9) for v,w in zip(a,b)):
            raise AssertionError(f'Pathwise regression failed: {ap}, {differences}')
        tests.append(dict(a_plus=ap,pathwise_max_differences=differences))
    (root/'regression_tests.json').write_text(json.dumps(tests,indent=2))
    results=[]
    for n,tau in ((384,2000.),(384,20000.),(3840,2000.)):
        row=dict(trajectories=n,tau=tau,threads=threads)
        for name,fn in [('reference',reference_sample),('broadcast',fast_sample),('fused',fused_sample)]:
            fn(m,64,10.,123,0,batch=32) # excludes JIT compilation from timing
            times=[]
            for repeat in range(2):
                t=time.perf_counter();F,*_=fn(m,n,tau,123,1,batch=32);times.append(time.perf_counter()-t)
            event_count=int(F[:,:2].sum()+F[:,6].sum());seconds=float(np.median(times))
            row[name]=dict(seconds=seconds,events=event_count,events_per_second=event_count/seconds)
        row['broadcast_speedup']=row['reference']['seconds']/row['broadcast']['seconds']
        row['fused_speedup']=row['reference']['seconds']/row['fused']['seconds']
        results.append(row);print(json.dumps(row),flush=True)
    expected_events=960*3840*20000*(2*(6.5+1+2)/3+.05)
    for row in results:
        for name in ('broadcast','fused'):
            row[name]['full_x10_hours_extrapolated']=expected_events/row[name]['events_per_second']/3600
    (root/'benchmark.json').write_text(json.dumps(dict(benchmarks=results,
        expected_full_x10_events=expected_events,compilation_excluded=True),indent=2))

def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output',default=os.getcwd())
    p.add_argument('--preset',choices=['full','validation'],default='full')
    p.add_argument('--points',type=int)
    p.add_argument('--trajectories',type=int)
    p.add_argument('--tau',type=float)
    p.add_argument('--count-weights',type=int,default=12)
    p.add_argument('--sequence-weights',type=int,default=12)
    p.add_argument('--full-weights',type=int,default=24)
    p.add_argument('--xmin',type=float,default=3.);p.add_argument('--xmax',type=float,default=10.)
    p.add_argument('--b',type=float,default=1.);p.add_argument('--a-minus',type=float,default=1.)
    p.add_argument('--eta',type=float,default=2.);p.add_argument('--kappa',type=float,default=.05)
    p.add_argument('--bound-points',type=int,default=25)
    p.add_argument('--events',type=int,default=50000)
    p.add_argument('--streams',type=int,default=6)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--kernel',choices=['broadcast','fused'],default='broadcast',
        help='broadcast=NumPy array batches; fused=JIT loop variant, same exact SSA law')
    p.add_argument('--seed',type=int,default=20261020)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--save-ensembles',action='store_true',help='save per-trajectory sufficient readouts in addition to checkpoints')
    p.add_argument('--benchmark',action='store_true')
    args=p.parse_args()
    defaults=(960,3840,20000.) if args.preset=='full' else (960,768,2000.)
    if args.points is None:args.points=defaults[0]
    if args.trajectories is None:args.trajectories=defaults[1]
    if args.tau is None:args.tau=defaults[2]
    if min(args.points,args.trajectories,args.bound_points,args.events,args.streams,args.batch_size,args.threads)<2:
        p.error('sample counts, streams, batch size and threads must be >=2')
    if min(args.count_weights,args.sequence_weights,args.full_weights)<1:p.error('weight counts must be positive')
    if not (0<args.xmin<args.xmax and args.tau>0):p.error('Require 0<xmin<xmax and tau>0')
    return args

if __name__=='__main__':
    args=parse_args()
    if args.benchmark:benchmark(args)
    else:run_sampling(args)
