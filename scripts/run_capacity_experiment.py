#!/usr/bin/env python3
"""Cross effective-capacity factors with all existing DEM incision scenarios."""
from __future__ import annotations
import os,sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/phenode_numba')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/phenode_mpl')
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
import joblib
from pyproj import Transformer
from gully_scenarios.capacity import simulate_budget,aggregate_budget
from gully_scenarios.core import file_hash,terrain_signature
from gully_scenarios.capacity_report import make_capacity_report,write_event_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--factors',type=float,nargs='+',help='Override configured capacity factors; must include 1')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text());out=ROOT/'outputs'
    factors=args.factors or cfg.get('capacity_depth_factors',[1.,.5,.25])
    factors=sorted(set(factors),reverse=True)
    if 1. not in factors or not all(np.isfinite(f) and f>0 for f in factors):raise ValueError('Factors must be finite positive values including the 1.0 control')
    signature=json.loads((out/'terrain_signature.json').read_text())
    if signature!=terrain_signature(cfg):raise ValueError('Terrain or sensor inputs changed: rerun scripts/run_analysis.py first')
    sys.path.insert(0,cfg['dmm_repo'])
    from emt.model7.model import _step_loop
    artifact=Path(cfg['dmm_repo'])/'data/models/model8.joblib'
    meta=json.loads((out/'model_metadata.json').read_text())
    if file_hash(artifact)!=meta['artifact_sha256']:raise ValueError('Frozen model artifact changed')
    model=joblib.load(artifact)
    if getattr(model,'_n_process',5)!=5:raise ValueError('Expected the five-parameter model8')
    with np.load(out/'gully_storage_masks.npz') as d:
        rr=d['rows'];cc=d['cols']
        area_table=pd.read_csv(out/'gully_storage_areas.csv')
        membership={r:d[r.replace('?','').replace(' ','_')].copy() for r in area_table.region}
    with rasterio.open(out/'terrain/baseline_dem_5m.tif') as s:
        area=abs(s.transform.a*s.transform.e-s.transform.b*s.transform.d)
        x,y=rasterio.transform.xy(s.transform,rr,cc);crs=s.crs
    for r,m in membership.items():
        expected=float(area_table.set_index('region').loc[r,'valid_area_m2'])
        if not np.isclose(m.sum()*area,expected):raise ValueError('Mask areas disagree with saved common footprint')
    lon,lat=Transformer.from_crs(crs,4326,always_xy=True).transform(x,y)
    soil_cache=sorted((ROOT/'data/cache').rglob('*slga.nc'))[0]
    with xr.open_dataset(soil_cache) as ds:
        if min(lon)<ds.x.min() or max(lon)>ds.x.max() or min(lat)<ds.y.min() or max(lat)>ds.y.max():raise ValueError('Soil cache does not cover the footprint')
        soil=ds.sel(x=xr.DataArray(lon,dims='pixel'),y=xr.DataArray(lat,dims='pixel'),method='nearest').load()
    n=len(rr);smax,alpha,k=model.bucket_params
    capacity_base=smax*soil.soil_awc.to_numpy()/model.cap_train_mean_
    fixed={}
    for scenario in cfg['incisions_m']:
        cols=[]
        for var in model._static_vars:
            if var in soil:a=soil[var].to_numpy()
            elif var=='aridity':a=np.full(n,meta['aridity'])
            else:
                with rasterio.open(out/f'terrain/{scenario}_{var}.tif') as s:a=s.read(1,masked=True).filled(np.nan)[rr,cc]
            cols.append(a)
        S=np.column_stack(cols)
        if not np.isfinite(S).all():raise ValueError('Nonfinite model inputs in common mask')
        fixed[scenario]=model.readout(np.zeros(n),S)
    weather=pd.read_csv(ROOT/'data/raw/silo_daily.csv',parse_dates=['YYYY-MM-DD']).rename(columns={'YYYY-MM-DD':'date'}).set_index('date').sort_index()
    if not weather.index.equals(pd.date_range(weather.index.min(),weather.index.max())):raise ValueError('Non-contiguous daily forcing')
    forcing=pd.read_csv(out/'hourly_experimental_forcing.csv')
    forcing.time=pd.to_datetime(forcing.time,utc=True)
    keep=weather.index>=pd.Timestamp('2025-03-01')
    frames=[];audit=[]
    def aggregate(budget,times,starts,series,event_id,policy,factor,rain,pet,dt):
        for region,mask in membership.items():
            values=aggregate_budget(budget,mask,area)
            area_total=float(mask.sum()*area)
            dynamic=budget['storage_mm'][:,mask].mean(axis=1)*float(model.params_['dtheta']/smax)
            for scenario,intercept in fixed.items():
                frames.append(pd.DataFrame(dict(time=times,step_start=starts,series=series,event_id=event_id,gap_policy=policy,region=region,dem_scenario=scenario,capacity_factor=factor,timestep_hours=dt,area_m2=area_total,rain_m3=np.asarray(rain)*area_total/1000,pet_m3=np.asarray(pet)*area_total/1000,legacy_sm_pct=intercept[mask].mean()+dynamic,**values)))
    for factor in factors:
        capacity=capacity_base*factor
        print(f'Capacity factor {factor:g}: independent daily spin-up over {len(weather)} days',flush=True)
        budget=simulate_budget(weather.daily_rain,weather.et_morton_potential,capacity,alpha,k)
        stock=_step_loop(np.repeat(weather.daily_rain.to_numpy()[:,None],n,axis=1),np.repeat(weather.et_morton_potential.to_numpy()[:,None],n,axis=1),capacity,alpha,k)
        parity=float(np.max(np.abs(stock-budget['storage_mm'])))
        if parity>1e-10:raise AssertionError('Budget tracker diverged from original daily recurrence')
        audit.append(dict(capacity_factor=factor,series='daily',event_id='',gap_policy='daily',max_daily_recurrence_error_mm=parity,max_balance_error_mm=float(np.max(np.abs(budget['balance_residual_mm']))),floor_correction_sum_mm=float(budget['floor_correction_mm'].sum())))
        subset={key:(a[keep] if isinstance(a,np.ndarray) and a.ndim==2 else a) for key,a in budget.items()}
        times=weather.index[keep].tz_localize(cfg['display_timezone']).tz_convert('UTC')
        aggregate(subset,times,times,'daily','','daily',factor,weather.daily_rain[keep],weather.et_morton_potential[keep],24)
        for (event,policy),f in forcing.groupby(['event_id','gap_policy']):
            f=f.sort_values('time')
            previous=f.time.min().tz_convert(cfg['display_timezone']).tz_localize(None).normalize()-pd.Timedelta(days=1)
            initial=budget['storage_mm'][weather.index.get_loc(previous)].copy()
            hourly=simulate_budget(f.rain_mm.to_numpy(),f.pet_mm.to_numpy(),capacity,alpha,k,timestep_hours=1,initial=initial)
            audit.append(dict(capacity_factor=factor,series='experimental_hourly',event_id=event,gap_policy=policy,max_daily_recurrence_error_mm=np.nan,max_balance_error_mm=float(np.max(np.abs(hourly['balance_residual_mm']))),floor_correction_sum_mm=float(hourly['floor_correction_mm'].sum())))
            aggregate(hourly,f.time+pd.Timedelta(hours=1),f.time,'experimental_hourly',event,policy,factor,f.rain_mm.to_numpy(),f.pet_mm.to_numpy(),1)
        del stock,budget,subset,hourly
    totals=pd.concat(frames,ignore_index=True)
    totals.to_csv(out/'capacity_depth_timeseries.csv',index=False)
    pd.DataFrame(audit).to_csv(out/'capacity_depth_budget_audit.csv',index=False)
    # All terrain rows should have identical process outputs at fixed capacity.
    group=['series','event_id','gap_policy','region','capacity_factor','time']
    maxima=totals.groupby(group,dropna=False).storage_m3.agg(['min','max'])
    assert np.max(maxima['max']-maxima['min'])<1e-9
    # Preserve the original baseline at factor one, daily AND hourly.
    old=pd.read_csv(out/'gully_storage_timeseries.csv').fillna({'event_id':''})
    old.time=pd.to_datetime(old.time,utc=True)
    control=totals[totals.capacity_factor==1.].rename(columns={'dem_scenario':'scenario'})
    keys=['time','series','event_id','gap_policy','region','scenario']
    paired=control.merge(old,on=keys,validate='one_to_one')
    if len(paired)!=len(control) or len(paired)!=len(old):raise AssertionError('Factor-one coverage differs from the earlier experiment')
    control_error=float(np.max(np.abs(paired.storage_m3-paired.bucket_storage_m3)))
    np.testing.assert_allclose(paired.storage_m3,paired.bucket_storage_m3,rtol=1e-12,atol=1e-9)
    np.testing.assert_allclose(paired.legacy_sm_pct,paired.mean_sm_pct,rtol=1e-12,atol=1e-9)
    events=pd.read_csv(out/'selected_events.csv',parse_dates=['date'])
    metrics=write_event_metrics(totals,events,cfg,out)
    provenance=dict(capacity_factors=factors,dem_scenarios=cfg['incisions_m'],factorial_combinations=len(factors)*len(cfg['incisions_m']),definition='capacity_i = frozen_model_smax * SLGA_AWC_i / training_mean_AWC * capacity_factor',depth_interpretation='relative effective-depth proxy, not measured soil depths or incision subtracted from a soil profile',spinup_start=str(weather.index.min().date()),initial_fraction=.5,independent_spinup_per_capacity=True,readout='legacy_sm_pct preserves original fitted global denominator; diagnostic only, not depth-corrected volumetric moisture',excess_definition='max(storage_after_rain_minus_AET_minus_drainage - capacity, 0); upper-clipped water, not routed runoff',time_conventions='Daily time is a SILO date label; hourly state is at interval end and step_start identifies the flux interval',model_sha256=file_hash(artifact),soil_sha256=file_hash(soil_cache),mask_sha256=file_hash(out/'gully_storage_masks.npz'),forcing_sha256=file_hash(ROOT/'data/raw/silo_daily.csv'),hourly_forcing_sha256=file_hash(out/'hourly_experimental_forcing.csv'),fixed_parameters=model.params_.to_dict(),max_factor_one_difference_m3=control_error,max_budget_residual_mm=max(a['max_balance_error_mm'] for a in audit),floor_correction_sum_mm=sum(a['floor_correction_sum_mm'] for a in audit),limitations=['No lateral routing or surface ponding','Fitted alpha and k held constant; capacity affects saturation and ET stress but k is not re-estimated','Hourly timestep, PET allocation and missing-rain assumptions remain experimental','DEM controls the additive readout only; there is no DEM-by-capacity interaction in the bucket dynamics'])
    (out/'capacity_depth_metadata.json').write_text(json.dumps(provenance,indent=2))
    make_capacity_report(ROOT,cfg,totals,metrics,events,provenance)
    print(out/'capacity_depth_dashboard.html')

if __name__=='__main__':main()
