"""Exhaustive Student-t MARX(1,1,q) subset selection by BIC.
"""

from itertools import combinations
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

# Define log likelihood function
def _nll(theta, y, X):
    phi, psi = theta[:2]
    beta = theta[2:-2]
    sigma, nu = theta[-2:]
    if sigma <= 0 or nu <= 0 or abs(phi) >= 1 or abs(psi) >= 1:
        return 1e100
    residual = ((1+phi*psi)*y[1:-1] - phi*y[:-2] - psi*y[2:])
    if X.shape[1]:
        residual = residual - X[1:-1] @ beta
    const = gammaln((nu+1)/2)-gammaln(nu/2)-0.5*np.log(nu*np.pi)-np.log(sigma)
    val = -np.sum(const - (nu+1)/2 * np.log1p((residual/sigma)**2/nu))
    return float(val) if np.isfinite(val) else 1e100


#  Function for AML estimation
def _fit(ids, y, X, n_starts, seed, min_nu=0.1, maxiter=5000, warm=None):
    x = X[:, ids]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), *[i+1 for i in ids], 999]))
    scale = max(np.median(np.abs(np.diff(y))), 0.05)
    bounds = [(-.99,.99),(-.99,.99)] + [(-5.,5.)] * len(ids) + [(1e-6,30.),(min_nu, 40)]
    initials = [np.r_[.4,.5,np.zeros(len(ids)),scale,5.]]
    if warm is not None:
        initials.insert(0, warm)
    while len(initials)<n_starts:
        initials.append(np.r_[rng.uniform(-.85,.85,2),rng.normal(0,.1,len(ids)),
                              scale*rng.uniform(.5,2),rng.choice([.8,1.,1.8,3.,6.,10.])])
    fitted=[]
    for t0 in initials:
        try:
            r=minimize(_nll,t0,args=(y,x),method='L-BFGS-B',bounds=bounds,
                       options={'maxiter':maxiter,'ftol':1e-11, 'gtol': 1e-8,'maxls':50})
            if r.success and np.isfinite(r.fun) and r.fun<1e90:
                fitted.append(r)
        except (ValueError, FloatingPointError, OverflowError):
            pass
    if not fitted:
        return {'ids':ids,'success':False,'error':'No converged start','n_converged':0}
    best=min(fitted,key=lambda r:r.fun)
    ll=-float(best.fun)
    k=len(best.x)
    n=len(y)-2
    return {'ids':ids,'success':True,'theta':np.array(best.x),
            'loglik':ll,'AIC':2*k-2*ll,'BIC':k*np.log(n)-2*ll,
            'n_converged':len(fitted)}


# Function that fits all subsets of exogenous variables 
def exhaustive_bic_marx(y, X, names, n_starts=5, seed=42, min_nu=.1,
                         maxiter=2000, checkpoint='marx_exhaustive_checkpoint.pkl',
                         save_every=25, verbose=True, retry_failed=False):
    """Fit ALL 2**p subsets, including empty subset, with Student-t nu free.

    A checkpoint records fits and a hash of the exact input arrays and key
    settings, so resumed results cannot silently mix different data/settings.
    For reliability, rerun the best subsets using more starts afterward.
    """
    y=np.asarray(y,dtype=float).reshape(-1)
    X=np.asarray(X,dtype=float)
    names=list(names)
    if X.ndim!=2 or X.shape!=(len(y),len(names)) or not np.isfinite(y).all() or not np.isfinite(X).all():
        raise ValueError('y and X must be finite, correctly aligned arrays.')
    if len(y)<10 or len(set(names))!=len(names) or n_starts<1:
        raise ValueError('Insufficient observations, repeated names or n_starts < 1.')
    if not 0<min_nu<2:
        raise ValueError('min_nu must be positive and below 2.')
    settings=(names, n_starts, seed, min_nu, maxiter)
    h=hashlib.sha256()
    for arr in (y,X):
        h.update(str(arr.shape).encode());h.update(np.ascontiguousarray(arr).tobytes())
    h.update(json.dumps(settings).encode())
    fingerprint=h.hexdigest()
    path=Path(checkpoint) if checkpoint else None
    fits={}
    if path and path.exists():
        with path.open('rb') as f: saved=pickle.load(f)
        if saved['fingerprint']!=fingerprint:
            raise ValueError('Checkpoint belongs to different data/settings; use a new filename.')
        fits=saved['fits']
        if verbose: print(f'Resuming: {len(fits)} subsets already recorded.')
    def save():
        if path:
            tmp=path.with_suffix(path.suffix+'.tmp')
            with tmp.open('wb') as f: pickle.dump({'fingerprint':fingerprint,'fits':fits},f)
            tmp.replace(path)
    count=0
    p=len(names)
    try:
        for k in range(p+1):
            for ids in combinations(range(p),k):
                if ids in fits and not (retry_failed and not fits[ids]['success']):
                    continue
                fits[ids]=_fit(ids,y,X,n_starts,seed,min_nu,maxiter)
                count+=1
                if count % save_every==0:
                    save()
                if verbose and count%500==0:
                    print(f'New fits: {count} | recorded {len(fits)}/{2**p}',flush=True)
    finally:
        save()
    successful=[v for v in fits.values() if v['success']]
    if not successful: raise RuntimeError('No successful fits. Inspect optimizer settings.')
    rows=[]
    for ids,fit in fits.items():
        row={'regressors':', '.join(names[i] for i in ids) or '(none)',
             'ids':ids,'n_regressors':len(ids),'success':fit['success'],
             'loglik':fit.get('loglik',np.nan),'AIC':fit.get('AIC',np.nan),
             'BIC':fit.get('BIC',np.nan),'nu':fit['theta'][-1] if fit['success'] else np.nan,
             'n_converged':fit['n_converged']}
        rows.append(row)
    table=pd.DataFrame(rows).sort_values('BIC',na_position='last').reset_index(drop=True)
    best=min(successful,key=lambda v:v['BIC'])
    if verbose:
        print(f'Finished: {len(fits)}/{2**p} subsets; successes={len(successful)}, failures={len(fits)-len(successful)}')
        print('Best subset:',[names[i] for i in best['ids']])
        print(f"Best BIC={best['BIC']:.6f}; nu={best['theta'][-1]:.4f}")
        if len(successful)!=2**p:
            print('WARNING: some fits failed. Best-found BIC is not certified across all subsets.')
    return {'best':best,'selected_names':[names[i] for i in best['ids']],
            'selected_X':X[:,best['ids']],'table':table,'fits':fits,
            'n_total':2**p,'n_success':len(successful),'checkpoint':str(path) if path else None}


# Function that fits best subsets again 
def refit_top_subsets(selection, y, X, names, top_n=10, n_starts=40,
                      seed=2026, min_nu=.1, maxiter=5000):
    """Increase multistart effort for top-ranked subsets and update the table."""
    y=np.asarray(y,dtype=float).reshape(-1);X=np.asarray(X,dtype=float)
    fits=selection['fits']; names=list(names)
    for ids in selection['table'].loc[selection['table']['success'],'ids'].head(top_n):
        result=_fit(ids,y,X,n_starts,seed,min_nu,maxiter)
        if result['success'] and result['BIC'] < fits[ids].get('BIC',np.inf):
            fits[ids]=result
    rows=[]
    for ids,r in fits.items():
        rows.append({'regressors':', '.join(names[j] for j in ids) or '(none)',
                     'ids':ids,'n_regressors':len(ids),'success':r['success'],
                     'loglik':r.get('loglik',np.nan),'AIC':r.get('AIC',np.nan),
                     'BIC':r.get('BIC',np.nan),'nu':r['theta'][-1] if r['success'] else np.nan,
                     'n_converged':r['n_converged']})
    selection['table']=pd.DataFrame(rows).sort_values('BIC',na_position='last').reset_index(drop=True)
    selection['best']=min((r for r in fits.values() if r['success']),key=lambda r:r['BIC'])
    selection['selected_names']=[names[j] for j in selection['best']['ids']]
    selection['selected_X']=X[:,selection['best']['ids']]
    return selection
