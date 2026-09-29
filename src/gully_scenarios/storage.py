"""Spatial integration of model8 readout and conceptual bucket storage.

A fixed planimetric footprint and assumed soil depth define each comparison.
Readout-derived volume is not the mass-conserving internal bucket state.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import distance_transform_edt
import rasterio
from .core import incision_weight


def volume_from_vwc(sm_pct, area_m2, soil_depth_m=1.):
    if soil_depth_m<=0:raise ValueError('Soil depth must be positive')
    sm=np.asarray(sm_pct,float);area=np.asarray(area_m2,float)
    if not np.isfinite(sm).all() or not np.isfinite(area).all():
        raise ValueError('Storage integration requires a fixed, finite common mask')
    return np.sum(sm/100*area*soil_depth_m,axis=-1)


def volume_from_bucket(storage_mm, area_m2):
    return np.sum(np.asarray(storage_mm,float)/1000*np.asarray(area_m2,float),axis=-1)


def mask_from_centreline(coords,shape,transform,half_width_m):
    """Reconstruct the original footprint, including its longitudinal taper."""
    rows,cols=rasterio.transform.rowcol(transform,np.asarray(coords)[:,0],np.asarray(coords)[:,1])
    path=list(zip(rows,cols));resolution=abs(transform.a)
    line=np.zeros(shape,bool);along=np.zeros(shape,float)
    lengths=np.r_[0,np.cumsum([np.hypot(a[0]-b[0],a[1]-b[1])*resolution for a,b in zip(path[:-1],path[1:])])]
    if lengths[-1]<=0:raise ValueError('Empty gully centreline')
    taper=min(50,lengths[-1]/3)
    for cell,s in zip(path,lengths):
        line[cell]=True;along[cell]=min(1,s/taper,(lengths[-1]-s)/taper)
    dist,nearest=distance_transform_edt(~line,sampling=resolution,return_indices=True)
    weight=incision_weight(dist,half_width_m)*along[tuple(nearest)]
    return weight>0
