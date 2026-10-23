#!/usr/bin/env python3
"""Barrier dependence of the three timing scales: exact 8-state Gillespie sampling.

Run: python sample_barrier_advantage.py --output . --threads 4
Then: lualatex -interaction=nonstopmode -halt-on-error barrier_advantage.tex

Only kappa changes: kappa = kappa0 * exp(-h), h=B/(k_B T).
Data use Eq.(19), Eq.(21a,b), NOT an optimization over random readouts.
Each independent stationary trajectory stops after m visible events, with
m = max(min_events, ceil(activity * gate_times / kappa)). Thus its mean duration
is of order gate_times/kappa or longer. Quarter/half/full checkpoints check
convergence. Error bars are one sample standard deviation of the single-stream
inverse-time estimates, with ddof=1. They are NOT standard errors or confidence
intervals. The central markers are pooled estimates, as in the preceding version.

The plotted bounds are Monte Carlo averages. Exact matrix quantities enter only
the conditional filters/moments and a Poisson-equation correction for Eq.(21b).
No analytical reference curves are drawn. The cap shape is controlled entirely
by the companion TikZ file; this revision leaves all sampling and numerical
estimation unchanged. The last three columns of barrier_data.dat are SDs in
linear y units, not logarithmic units.
Output defaults to the current working directory. numpy, scipy, numba required.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse, csv, json, math, time
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve
from scipy.integrate import quad
from numba import njit, prange, set_num_threads, config as numba_config

OBS = np.array([0,0,1,1,1,1,2,2], dtype=np.int64)
FLIPS = np.array([4,2,1], dtype=np.int64)
EDGE_ID = np.full((3,3), -1, dtype=np.int64)
EDGE_ID[0,1]=0; EDGE_ID[1,0]=1; EDGE_ID[1,2]=2; EDGE_ID[2,1]=3

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
    out=np.zeros((3,10));stop=np.array([events//4,events//2,events],np.int64)
    t=w=cm=cv=a2=ctt=ctl=cll=fast_time=0.;n=idx=gates=0
    while n<events:
        r0,r1,rz=rates[h];total=r0+r1+rz
        dt=-math.log(max(np.random.random(),1e-300))/total
        t+=dt;w+=dt
        if h&1: fast_time+=dt
        u=np.random.random()*total
        ch=0 if u<r0 else (1 if u<r0+r1 else 2)
        old=h;h^=FLIPS[ch]
        if ch==2:
            gates+=1
            continue
        x=OBS[old];y=OBS[h];e=EDGE_ID[x,y]
        filter_flow(p,d,wp,wd,w,pars,kappa)
        alpha=filter_jump(p,d,wp,wd,x,y,rates);a2+=alpha*alpha
        dm,dv=sequence_var_step(ps,fs,ss,e,K,work);cm+=dm;cv+=dv
        rt=w-eventmean[0]+H[h,0]-H[prev,0]
        rl=(1. if e<2 else 0.)-eventmean[1]+H[h,1]-H[prev,1]
        ctt+=rt*rt;ctl+=rt*rl;cll+=rl*rl
        prev=h;w=0.;n+=1
        if n==stop[idx]:
            out[idx]=np.array([n,t,cm,cv,a2,ctt,ctl,cll,gates,fast_time])
            idx+=1
    return out

@njit(parallel=True,cache=True)
def bound_ensemble(rates,pi,pars,kappa,K,H,em,events,seeds):
    a=np.empty((len(seeds),3,10))
    for k in prange(len(seeds)):
        a[k]=bound_stream(rates,pi,pars,kappa,K,H,em,events,seeds[k])
    return a



def estimates(raw):
    """Pool independent streams: columns are m,T,conditional mean/variance,
    score bracket, TT/TL/LL martingale sums, gate flips, fast-mode time.
    """
    _, elapsed, _, variance, bracket, tt, tl, ll, _, _ = raw.sum(axis=0)
    tn = (tt - tl*tl/ll)/elapsed
    if min(tn, variance, bracket, elapsed) <= 0:
        raise ArithmeticError('Nonpositive estimate; no points are discarded.')
    return np.array([1/tn, elapsed/variance, bracket/elapsed])


def with_errors(raw):
    """Return pooled estimate and the sample standard deviation across streams.

    Each independent stream r yields its own estimate estimates(raw[r:r+1]).
    The plotted error bar is the ordinary sample standard deviation of these
    per-stream estimates, with ddof=1. It is therefore a standard deviation
    across independently sampled Gillespie trajectories, not a standard error
    and not a confidence interval.
    """
    value = estimates(raw)
    per_stream = np.array([estimates(raw[i:i+1]) for i in range(len(raw))])
    std = per_stream.std(axis=0, ddof=1)
    return value, std


def frozen_reference(a, b, eta):
    """Ungated four-state system. Reference constants only, never the MC points.
    T_Sigma = mean conditional passage variance / mean passage duration.
    I_X rate = Fisher information of the timed passage / mean passage duration.
    """
    x=np.array([0,1,1,2]); rates=np.array([
        [a*(eta if not (s>>1) else 1),b*(eta if not (s&1) else 1)] for s in range(4)])
    lam=rates.sum(1); R=np.zeros((4,4,4))
    for s in range(4):
        for c,flip in enumerate([2,1]):
            q=s^flip;R[EDGE_ID[x[s],x[q]],s,q]=rates[s,c]
    K=np.empty((4,3,4,4))
    for e in range(4):
        K[e,0]=R[e]/lam[:,None];K[e,1]=K[e,0]/lam[:,None]
        K[e,2]=2*K[e,1]/lam[:,None]
    mean=variance=middle_info=0.
    for i,s in enumerate([0,3]):
        nu=(1. if i==0 else eta)/(1+eta);e=[0,3][i]
        for j,q in enumerate([0,3]):
            f=[1,2][j]
            p=(K[e,0]@K[f,0])[s,q]
            m=(K[e,1]@K[f,0]+K[e,0]@K[f,1])[s,q]
            v=(K[e,2]@K[f,0]+2*K[e,1]@K[f,1]+K[e,0]@K[f,2])[s,q]
            mean+=nu*m;variance+=nu*(v-m*m/p)
            coeff=K[e,0,s,1:3]*R[f,1:3,q]; middle_lam=lam[1:3]
            lmin=middle_lam.min()
            def integrand(w):
                weights=coeff*np.exp(-(middle_lam-lmin)*w);z=weights.sum()
                if z==0: return 0.
                score=1-w*(weights@middle_lam)/z
                return np.exp(-lmin*w)*z*score*score
            middle_info+=nu*quad(integrand,0,np.inf,epsabs=1e-11,epsrel=1e-10)[0]
    return float(variance/mean), float((1+middle_info)/mean)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output',default=os.getcwd())
    p.add_argument('--a-plus',type=float,default=6.);p.add_argument('--a-minus',type=float,default=1.)
    p.add_argument('--b',type=float,default=1.);p.add_argument('--eta',type=float,default=2.)
    p.add_argument('--kappa0',type=float,default=.5)
    p.add_argument('--barrier-max',type=float,default=9.)
    p.add_argument('--points',type=int,default=19)
    p.add_argument('--streams',type=int,default=16)
    p.add_argument('--gate-times',type=float,default=25.,help='target mean kappa*T_m')
    p.add_argument('--min-events',type=int,default=50000)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261021)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.streams<4 or args.points<2 or args.min_events<100 or args.gate_times<=0:
        p.error('Require streams>=4, points>=2, min-events>=100 and gate-times>0.')
    if not (args.a_plus>args.a_minus>0 and args.b>0 and args.eta>0 and args.kappa0>0):
        p.error('Positive model rates and a-plus>a-minus are required.')
    if np.isclose(args.a_plus*args.a_minus,args.b**2):
        p.error('The reference-limit example excludes a-plus*a-minus = b^2.')
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    (root/'raw').mkdir(exist_ok=True)
    set_num_threads(min(args.threads,numba_config.NUMBA_NUM_THREADS))
    saved={k:v for k,v in vars(args).items() if k not in ('output','resume','threads')}
    cfg=root/'run_config.json'
    if cfg.exists():
        if not args.resume: raise FileExistsError('Output already has a run; use --resume or a new folder.')
        if json.loads(cfg.read_text())!=saved: raise ValueError('Resume parameters differ.')
    cfg.write_text(json.dumps(saved,indent=2))
    hs=np.linspace(0,args.barrier_max,args.points)
    rows=[];checks=[];plot=[];start=time.perf_counter()
    # Warm up the JIT before timing the parameter scan.
    model=build_model(args.a_minus,args.a_plus,args.b,args.eta,args.kappa0)
    bound_ensemble(model.rates,model.pi,model.pars,model.kappa,model.K,model.poisson,
                   model.eventmean,100,np.array([17,31],dtype=np.uint32))
    r2=((args.a_plus-args.a_minus)/(args.a_plus+args.a_minus+2*args.b))**2
    for i,h in enumerate(hs):
        kappa=args.kappa0*np.exp(-h)
        model=build_model(args.a_minus,args.a_plus,args.b,args.eta,kappa)
        events=max(args.min_events,int(np.ceil(model.activity*args.gate_times/kappa)))
        path=root/'raw'/f'barrier_{i:03d}.npz'
        if args.resume and path.exists():
            raw=np.load(path)['raw']
        else:
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,i]))
            sd=rng.choice(2**32,size=args.streams,replace=False).astype(np.uint32)
            raw=bound_ensemble(model.rates,model.pi,model.pars,model.kappa,model.K,
                               model.poisson,model.eventmean,events,sd)
            np.savez_compressed(path,raw=raw,barrier=h,kappa=kappa,seeds=sd)
        for j in range(3):
            value,std=with_errors(raw[:,j]);end=raw[:,j]
            checks.append(dict(barrier=float(h),fraction=[.25,.5,1.][j],events=int(end[0,0]),
                mean_duration=float(end[:,1].mean()),mean_kappa_T=float(kappa*end[:,1].mean()),
                min_kappa_T=float(kappa*end[:,1].min()),mean_gate_flips=float(end[:,8].mean()),
                fast_time_fraction=float(end[:,9].sum()/end[:,1].sum()),
                BN=value[0],BS=value[1],BX=value[2],stdN=std[0],stdS=std[1],stdX=std[2]))
        plot.append([h,*value,*std])
        rows.append(dict(barrier=float(h),kappa=float(kappa),events=events,streams=args.streams,
            mean_duration=checks[-1]['mean_duration'],kappa_T=checks[-1]['mean_kappa_T'],
            invN=value[0],invSigma=value[1],invX=value[2],stdN=std[0],stdSigma=std[1],stdX=std[2],
            TN=1/value[0],TSigma=1/value[1],TX=1/value[2],
            counting_exact_check=1/model.TN_exact,counting_asymptote=kappa/r2))
        # Save after EACH completed barrier, not only at the end.
        np.savetxt(root/'barrier_data.dat',plot,fmt='%.12g',
                   header='barrier invN invSigma invX stdN stdSigma stdX')
        write_csv(root/'barrier_results.csv',rows);write_csv(root/'convergence.csv',checks)
        print(f'{i+1}/{len(hs)} h={h:.2f} m={events:,} kappa<T>={checks[-1]["mean_kappa_T"]:.1f} '
              f'invT={value} std={std} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    value=np.asarray(plot)[:,1:4];tail=hs>=max(0,args.barrier_max-3)
    summary=dict(complete=True,total_visible_events=int(sum(r['streams']*r['events'] for r in rows)),
        elapsed_seconds=time.perf_counter()-start,
        counting_tail_ln_slope=float(np.polyfit(hs[tail],np.log(value[tail,0]),1)[0]),
        last_ratio_TN_over_TSigma=float(value[-1,1]/value[-1,0]),
        max_relative_counting_error_vs_exact=float(max(abs(r['invN']/r['counting_exact_check']-1) for r in rows)),
        final_half_full_ratios=[(np.array([c['BN'],c['BS'],c['BX']])/np.array([r['invN'],r['invSigma'],r['invX']])).tolist()
            for c,r in zip(checks[1::3],rows)],
        note='Every marker is a fresh SSA estimate. Error bars are the sample standard deviation across independently sampled Gillespie streams for the corresponding single-stream estimator.')
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='final_half_full_ratios'},indent=2))


if __name__=='__main__':
    main()
