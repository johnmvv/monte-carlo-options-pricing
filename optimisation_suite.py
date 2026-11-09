"""Benchmark and validate European call control-variate implementations.

Run: python optimisation_suite.py --quick
     python optimisation_suite.py --full
Requires: numpy, scipy, numba.
"""
import argparse
import csv
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
from scipy.special import ndtr

S0, K, T, R, SIGMA = 100.0, 100.0, 1.0, 0.05, 0.2
PARAMS = (S0, K, T, R, SIGMA)


def analytical_price(s0, k, t, r, sigma):
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    return s0 * ndtr(d1) - k * np.exp(-r*t) * ndtr(d1-sigma*np.sqrt(t))


def optimal_beta(s0, k, t, r, sigma):
    v = sigma*np.sqrt(t)
    d1 = (np.log(s0/k)+(r+0.5*sigma*sigma)*t)/v
    c = analytical_price(s0,k,t,r,sigma)
    cov = s0*s0*np.exp(v*v)*ndtr(d1+v)-k*s0*np.exp(-r*t)*ndtr(d1)-c*s0
    return cov/(s0*s0*np.expm1(v*v))


@njit(cache=True)
def serial_moments(z, s0, k, t, r, sigma, beta):
    drift=(r-0.5*sigma*sigma)*t
    diffusion=sigma*np.sqrt(t)
    discount=np.exp(-r*t)
    total=0.0
    sq=0.0
    for i in range(z.size):
        st=s0*np.exp(drift+diffusion*z[i])
        x=discount*max(st-k,0.0)-beta*(discount*st-s0)
        total+=x
        sq+=x*x
    return total,sq


@njit(parallel=True,cache=True)
def parallel_moments(z, s0, k, t, r, sigma, beta):
    drift=(r-0.5*sigma*sigma)*t
    diffusion=sigma*np.sqrt(t)
    discount=np.exp(-r*t)
    total=0.0
    sq=0.0
    for i in prange(z.size):
        st=s0*np.exp(drift+diffusion*z[i])
        x=discount*max(st-k,0.0)-beta*(discount*st-s0)
        total+=x
        sq+=x*x
    return total,sq


def numpy_moments(z,s0,k,t,r,sigma,beta):
    st=s0*np.exp((r-0.5*sigma*sigma)*t+sigma*np.sqrt(t)*z)
    x=np.exp(-r*t)*np.maximum(st-k,0.0)-beta*(np.exp(-r*t)*st-s0)
    return float(np.sum(x)),float(np.dot(x,x))


def price_from_moments(total,sq,n):
    mean=total/n
    variance=max((sq-n*mean*mean)/(n-1),0.0)
    return mean,np.sqrt(variance/n)


def generate_chunks(rng,n,chunk_size):
    remaining=n
    while remaining:
        count=min(chunk_size,remaining)
        yield rng.standard_normal(count)
        remaining-=count


def price_chunked(n,chunk_size,seed=42,threads=1,kernel='parallel'):
    if not isinstance(n,int) or n<2 or not isinstance(chunk_size,int) or chunk_size<1:
        raise ValueError('n >= 2 and chunk_size >= 1 required')
    if threads<1:
        raise ValueError('threads must be positive')
    set_num_threads(threads)
    rng=np.random.default_rng(seed)
    beta=optimal_beta(*PARAMS)
    f={'parallel':parallel_moments,'serial':serial_moments,'numpy':numpy_moments}[kernel]
    total=sq=0.0
    for z in generate_chunks(rng,n,chunk_size):
        a,b=f(z,*PARAMS,beta)
        total+=a
        sq+=b
    return price_from_moments(total,sq,n)


def timed(fn, repeats=7, target_seconds=0.15):
    fn()
    t0=time.perf_counter()
    fn()
    estimate=max(time.perf_counter()-t0,1e-7)
    batch=max(1,min(500,int(target_seconds/estimate)))
    values=[]
    for _ in range(repeats):
        enabled=gc.isenabled()
        if enabled: gc.disable()
        try:
            t0=time.perf_counter()
            for _ in range(batch): fn()
            values.append((time.perf_counter()-t0)/batch*1000)
        finally:
            if enabled: gc.enable()
    return float(np.median(values)),float(np.percentile(values,25)),float(np.percentile(values,75))


def validate():
    beta=optimal_beta(*PARAMS)
    rng=np.random.default_rng(120)
    z=rng.standard_normal(100_000)
    ref=numpy_moments(z,*PARAMS,beta)
    for fn in (serial_moments,parallel_moments):
        got=fn(z,*PARAMS,beta)
        np.testing.assert_allclose(got,ref,rtol=1e-10,atol=1e-6)
    for threads in (1,min(4,get_num_threads())):
        result=price_chunked(100_000,17_000,seed=11,threads=threads)
        assert abs(result[0]-analytical_price(*PARAMS)) < 6*result[1]
    return True


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--full',action='store_true')
    p.add_argument('--quick',action='store_true')
    p.add_argument('--out',default='results/optimisation')
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    max_threads=get_num_threads()
    threads=[t for t in (1,2,4,6,12) if t<=max_threads]
    n=2_000_000 if args.full else 200_000
    repeats=11 if args.full else 5
    print('Validating kernels...',flush=True)
    validate()
    set_num_threads(max_threads)
    beta=optimal_beta(*PARAMS)
    z=np.random.default_rng(123).standard_normal(n)
    rows=[]
    print('Kernel-only comparison (pre-generated normals):',flush=True)
    for t in threads:
        set_num_threads(t)
        fn=lambda:parallel_moments(z,*PARAMS,beta)
        med,q25,q75=timed(fn,repeats)
        rows.append(dict(stage='kernel',threads=t,chunk_size=n,paths=n,median_ms=med,q25_ms=q25,q75_ms=q75))
        print(f'{t} threads: {med:.3f} ms',flush=True)
    best=min((r for r in rows if r['stage']=='kernel'),key=lambda x:x['median_ms'])['threads']
    print('End-to-end chunk comparison:',flush=True)
    chunk_sizes=([100_000,500_000,1_000_000,2_000_000] if args.full else [25_000,100_000,200_000])
    for t in sorted(set([1,best])):
        for chunk in chunk_sizes:
            fn=lambda t=t,chunk=chunk:price_chunked(n,chunk,seed=42,threads=t)
            med,q25,q75=timed(fn,repeats)
            rows.append(dict(stage='end_to_end',threads=t,chunk_size=chunk,paths=n,median_ms=med,q25_ms=q25,q75_ms=q75))
            print(f'{t} threads, chunk {chunk:,}: {med:.3f} ms',flush=True)
    with (out/'timings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Statistical checks use independent seeds. No claim of deterministic superiority.
    reps=120 if args.full else 30
    exact=analytical_price(*PARAMS)
    stat=[]
    for method in ('numpy','serial','parallel'):
        prices=[]
        for seed in range(reps):
            prices.append(price_chunked(20_000 if args.full else 5_000,5_000,seed=seed,threads=best,kernel=method)[0])
        stat.append(dict(method=method,repetitions=reps,rmse=float(np.sqrt(np.mean((np.array(prices)-exact)**2))),bias=float(np.mean(np.array(prices)-exact))))
    with (out/'accuracy.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stat[0]))
        writer.writeheader();writer.writerows(stat)
    metadata=dict(cpu=platform.processor(),platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,paths=n,threads=threads,repetitions=repeats,statistical_repetitions=reps,parameters=PARAMS,exact_price=exact)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Saved timings.csv, accuracy.csv and metadata.json to',out)

if __name__=='__main__':
    main()
