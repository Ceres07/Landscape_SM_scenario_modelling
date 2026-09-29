#!/usr/bin/env python3
"""Predict every gully pixel and integrate storage over identical scenario masks."""
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
from gully_scenarios.storage import mask_from_centreline,volume_from_vwc,volume_from_bucket
from gully_scenarios.core import hourly_bucket,file_hash
from gully_scenarios.area_report import make_area_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--soil-depth-m',type=float,default=1.,help='Assumed uniform soil-layer depth; readout-equivalent volume scales linearly')
    args=parser.parse_args()
    if args.soil_depth_m<=0:raise ValueError('Soil depth must be positive')
    cfg=json.loads((ROOT/'config.json').read_text());out=ROOT/'outputs';terrain_dir=out/'terrain'
    sys.path.insert(0,cfg['dmm_repo'])
    from emt.model7.model import _step_loop
    model=joblib.load(Path(cfg['dmm_repo'])/'data/models/model8.joblib')
    if getattr(model,'_n_process',5)!=5:raise ValueError('This integration expects the five-parameter model8')
    with rasterio.open(terrain_dir/'baseline_dem_5m.tif') as s:
        dem=s.read(1,masked=True).filled(np.nan);profile=s.profile;transform=s.transform
        area=abs(transform.a*transform.e-transform.b*transform.d)
    features=json.loads((terrain_dir/'gully_centrelines.geojson').read_text())['features']
    masks={f['properties']['label']:mask_from_centreline(f['geometry']['coordinates'],dem.shape,transform,cfg['gully_half_width_m']) for f in features}
    union=np.logical_or.reduce(list(masks.values()))
    with rasterio.open(terrain_dir/'incision_weight.tif') as s:
        original=s.read(1)>0
        if not np.array_equal(union,original):raise AssertionError('Reconstructed individual masks differ from original incision footprint')
    rr,cc=np.where(union);xy=rasterio.transform.xy(transform,rr,cc)
    lon,lat=Transformer.from_crs(profile['crs'],4326,always_xy=True).transform(xy[0],xy[1])
    caches=list((ROOT/'data/cache').rglob('*slga.nc'))
    if not caches:raise FileNotFoundError('Run scripts/fetch_weather_soil.py first')
    with xr.open_dataset(caches[0]) as ds:
        if min(lon)<float(ds.x.min()) or max(lon)>float(ds.x.max()) or min(lat)<float(ds.y.min()) or max(lat)>float(ds.y.max()):
            raise ValueError('Soil cache does not cover all gully cells')
        sampled=ds.sel(x=xr.DataArray(lon,dims='pixel'),y=xr.DataArray(lat,dims='pixel'),method='nearest').load()
    soil={v:sampled[v].to_numpy() for v in ['soil_clay','soil_sand','soil_awc','soil_bdw']}
    weather=pd.read_csv(ROOT/'data/raw/silo_daily.csv',parse_dates=['YYYY-MM-DD']).rename(columns={'YYYY-MM-DD':'date'}).set_index('date').sort_index()
    meta=json.loads((out/'model_metadata.json').read_text())
    statics={};valid=np.isfinite(dem[rr,cc])
    for name in cfg['incisions_m']:
        cols=[]
        for var in model._static_vars:
            if var in soil:a=soil[var]
            elif var=='aridity':a=np.full(len(rr),meta['aridity'])
            else:
                with rasterio.open(terrain_dir/f'{name}_{var}.tif') as s:a=s.read(1,masked=True).filled(np.nan)[rr,cc]
            cols.append(a)
        statics[name]=np.column_stack(cols);valid &= np.isfinite(statics[name]).all(axis=1)
    valid &= soil['soil_awc']>0
    if not valid.any():raise ValueError('No valid gully pixels')
    masks['All gullies (union)']=union
    membership={name:m[rr,cc][valid] for name,m in masks.items()}
    coverage=[]
    for name,mask in masks.items():
        n=int(membership[name].sum())
        if not n:raise ValueError(f'No pixels for {name}')
        coverage.append(dict(region=name,footprint_pixels=int(mask.sum()),valid_pixels=n,footprint_area_m2=int(mask.sum())*area,valid_area_m2=n*area,excluded_area_m2=(int(mask.sum())-n)*area))
    pd.DataFrame(coverage).to_csv(out/'gully_storage_areas.csv',index=False)
    np.savez_compressed(out/'gully_storage_masks.npz',rows=rr[valid],cols=cc[valid],**{k.replace('?','').replace(' ','_'):v for k,v in membership.items()})
    S={name:arr[valid] for name,arr in statics.items()}
    audit=[]
    for name,a in S.items():
        for j,var in enumerate(model._static_vars):
            lo=float(model.static[var].min());hi=float(model.static[var].max())
            audit.append(dict(scenario=name,variable=var,training_min=lo,training_max=hi,gully_min=float(a[:,j].min()),gully_max=float(a[:,j].max()),fraction_cells_outside_training_range=float(np.mean((a[:,j]<lo)|(a[:,j]>hi)))))
    pd.DataFrame(audit).to_csv(out/'gully_storage_covariate_audit.csv',index=False)
    smax,alpha,k=model.bucket_params
    capacity=smax*soil['soil_awc'][valid]/model.cap_train_mean_
    n=int(valid.sum());nt=len(weather)
    print(f'Simulating {n} gully pixels over {nt} daily steps',flush=True)
    stor=_step_loop(np.repeat(weather.daily_rain.to_numpy()[:,None],n,axis=1),np.repeat(weather.et_morton_potential.to_numpy()[:,None],n,axis=1),capacity,alpha,k)
    fixed={name:model.readout(np.zeros(n),a) for name,a in S.items()}
    scale=float(model.params_['dtheta']/smax)
    # Linearity of model8's readout permits exact area aggregation without
    # allocating a time × pixel × scenario array. Check it against readout.
    for name,a in S.items():
        np.testing.assert_allclose(fixed[name]+scale*stor[-1],model.readout(stor[-1],a),rtol=1e-13)
    area_by_region={r['region']:r['valid_area_m2'] for r in coverage}
    rows=[]
    def aggregate(times,storage,series,event_id='',policy='daily'):
        for region,mask in membership.items():
            area_total=area_by_region[region]
            dynamic=storage[:,mask].mean(axis=1)*scale
            bucket=volume_from_bucket(storage[:,mask],area)
            for scenario,intercept in fixed.items():
                sm=intercept[mask].mean()+dynamic
                volume=sm/100*area_total*args.soil_depth_m
                rows.append(pd.DataFrame(dict(time=times,series=series,event_id=event_id,gap_policy=policy,region=region,scenario=scenario,mean_sm_pct=sm,sm_equivalent_m3=volume,bucket_storage_m3=bucket,area_m2=area_total,soil_depth_m=args.soil_depth_m)))
    use=weather.index>=pd.Timestamp('2025-03-01')
    aggregate(weather.index[use].tz_localize(cfg['display_timezone']).tz_convert('UTC'),stor[use],'daily')
    forcing=pd.read_csv(out/'hourly_experimental_forcing.csv',parse_dates=['time'])
    forcing['time']=pd.to_datetime(forcing.time,utc=True)
    for (event_id,policy),f in forcing.groupby(['event_id','gap_policy']):
        f=f.sort_values('time');begin=f.time.min().tz_convert(cfg['display_timezone']).tz_localize(None).normalize()
        initial=stor[weather.index.get_loc(begin-pd.Timedelta(days=1))]
        hourly=hourly_bucket(f.rain_mm.to_numpy(),f.pet_mm.to_numpy(),capacity,alpha,k,initial)
        aggregate(f.time+pd.Timedelta(hours=1),hourly,'experimental_hourly',event_id,policy)
    totals=pd.concat(rows,ignore_index=True)
    totals.to_csv(out/'gully_storage_timeseries.csv',index=False)
    compare=[]
    for (series,event_id,policy,region),g in totals.groupby(['series','event_id','gap_policy','region'],dropna=False):
        v=g.pivot(index='time',columns='scenario',values='sm_equivalent_m3')
        b=g.pivot(index='time',columns='scenario',values='bucket_storage_m3')
        for name in cfg['incisions_m']:
            delta=v[name]-v.baseline
            compare.append(dict(series=series,event_id=event_id,gap_policy=policy,region=region,scenario=name,mean_storage_difference_m3=float(delta.mean()),storage_difference_temporal_range_m3=float(np.ptp(delta)),max_bucket_difference_m3=float(np.max(np.abs(b[name]-b.baseline)))))
    checks=pd.DataFrame(compare);checks.to_csv(out/'gully_storage_scenario_comparison.csv',index=False)
    assert checks.storage_difference_temporal_range_m3.max()<1e-8
    assert checks.max_bucket_difference_m3.max()==0
    # Explicit test of actual volume integration, including per-pixel readout.
    for region,mask in membership.items():
        for name,a in S.items():
            v=volume_from_vwc(model.readout(stor[-1,mask],a[mask]),area,args.soil_depth_m)
            expected=totals[(totals.series=='daily')&(totals.region==region)&(totals.scenario==name)].iloc[-1].sm_equivalent_m3
            np.testing.assert_allclose(v,expected,rtol=1e-12)
    provenance=dict(soil_depth_m=args.soil_depth_m,cell_area_m2=area,soil_cache_sha256=file_hash(caches[0]),model_sha256=meta['artifact_sha256'],mask='Original incision footprint, constant across scenarios; union counts overlaps once',storage_definition='SM-derived equivalent: sum(SM_percent / 100 * horizontal_cell_area * assumed_soil_depth)',bucket_definition='sum(conceptual_bucket_storage_mm / 1000 * horizontal_cell_area)',assumptions=['Uniform layer below each scenario surface','Soil properties and depth fixed despite excavation','No surface water, lateral routing or groundwater inventory','SM readout-derived volume is not the model water-balance state'],max_offset_time_variation_m3=float(checks.storage_difference_temporal_range_m3.max()),max_bucket_scenario_difference_m3=float(checks.max_bucket_difference_m3.max()))
    (out/'gully_storage_metadata.json').write_text(json.dumps(provenance,indent=2))
    make_area_report(ROOT,cfg,totals,pd.DataFrame(coverage),checks,provenance)
    print(pd.DataFrame(coverage).to_string(index=False))
    print(checks[(checks.series=='daily')&(checks.scenario!='baseline')][['region','scenario','mean_storage_difference_m3']].to_string(index=False))
    print(out/'whole_gully_storage.html')

if __name__=='__main__':main()
