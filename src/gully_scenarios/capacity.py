"""Depth-factor sensitivity with the original model8 recurrence and flux accounting.

Capacity is an experimental effective-depth proxy. No fitted coefficient is
changed. Excess means upper clipping after model losses, not routed discharge.
"""
from __future__ import annotations
import numpy as np


def simulate_budget(rain_mm, pet_mm, capacity_mm, alpha, k_daily, *, timestep_hours=24., initial=None):
    rain=np.asarray(rain_mm,dtype=float)
    pet=np.asarray(pet_mm,dtype=float)
    cap=np.atleast_1d(np.asarray(capacity_mm,dtype=float))
    if rain.ndim!=1 or pet.shape!=rain.shape:
        raise ValueError('Rain and PET must be matching one-dimensional step totals')
    if not np.isfinite(rain).all() or not np.isfinite(pet).all() or np.any(rain<0) or np.any(pet<0):
        raise ValueError('Rain and PET must be finite nonnegative totals')
    if cap.ndim!=1 or not np.isfinite(cap).all() or np.any(cap<=0):
        raise ValueError('Capacity must be a finite positive per-pixel vector')
    if not np.isfinite(alpha) or alpha<=0 or not 0<=k_daily<1 or not 0<timestep_hours<=24:
        raise ValueError('Invalid bucket parameters or timestep')
    k=k_daily if timestep_hours==24 else 1-(1-k_daily)**(timestep_hours/24)
    s=0.5*cap if initial is None else np.array(initial,dtype=float,copy=True)
    if s.shape!=cap.shape or not np.isfinite(s).all() or np.any(s<0) or np.any(s>cap):
        raise ValueError('Initial storage must lie within each cell capacity')
    names=['storage_mm','aet_mm','drainage_mm','excess_mm','floor_correction_mm','balance_residual_mm']
    out={name:np.empty((len(rain),len(cap)),dtype=float) for name in names}
    for t,(p,e) in enumerate(zip(rain,pet)):
        before=s.copy()
        wet=before+p
        aet=e*np.minimum(1.,wet/(alpha*cap))
        drainage=k*wet
        raw=wet-aet-drainage
        excess=np.maximum(raw-cap,0.)
        floor=np.maximum(-raw,0.)
        s=np.clip(raw,0.,cap)
        residual=s-before-p+aet+drainage+excess-floor
        for name,value in zip(names,[s,aet,drainage,excess,floor,residual]):out[name][t]=value
    out['initial_storage_mm']=0.5*cap if initial is None else np.array(initial,dtype=float,copy=True)
    out['capacity_mm']=cap.copy()
    return out


def aggregate_budget(budget, mask, cell_area_m2):
    """Integrate a fixed cell mask; relative fullness is capacity weighted."""
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=budget['capacity_mm'].shape or not mask.any():raise ValueError('Invalid region mask')
    if not np.isfinite(cell_area_m2) or cell_area_m2<=0:raise ValueError('Invalid cell area')
    out={name.replace('_mm','_m3'):budget[name][:,mask].sum(axis=1)*cell_area_m2/1000
         for name in ['storage_mm','aet_mm','drainage_mm','excess_mm','floor_correction_mm','balance_residual_mm']}
    capacity=float(budget['capacity_mm'][mask].sum()*cell_area_m2/1000)
    out['capacity_m3']=capacity
    out['fullness_pct']=100*out['storage_m3']/capacity
    out['at_capacity_area_pct']=100*np.mean(budget['storage_mm'][:,mask]>=budget['capacity_mm'][mask]-1e-9,axis=1)
    return out
