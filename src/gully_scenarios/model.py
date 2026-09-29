"""Frozen model8 daily predictions and a clearly separated hourly experiment."""
import sys
import json
import numpy as np
import pandas as pd
import joblib
from .core import hourly_bucket,select_events,event_metrics,file_hash


def run_models(root,cfg,points,terrain,silo,rain,observations):
    sys.path.insert(0,cfg['dmm_repo'])
    from emt.model7.model import _step_loop
    from pathlib import Path
    artifact=Path(cfg['dmm_repo'])/'data/models/model8.joblib'
    model=joblib.load(artifact)
    soil=pd.read_csv(root/'data/raw/soil_points.csv').set_index('device').loc[points.device]
    smax,alpha,k=model.bucket_params
    capacity=smax*soil.soil_awc.to_numpy()/model.cap_train_mean_
    daily_storage=_step_loop(np.repeat(silo.rain_mm.to_numpy()[:,None],len(points),axis=1),
                             np.repeat(silo.pet_mm.to_numpy()[:,None],len(points),axis=1),capacity,alpha,k)
    static={};audit=[]
    train=model.static.copy()  # training statics embedded in the frozen artifact
    for name in cfg['incisions_m']:
        tp=terrain[terrain.scenario==name].set_index('device').loc[points.device]
        inputs=pd.DataFrame(index=points.device)
        for var in model._static_vars:
            inputs[var]=soil[var].values if var in soil else (silo.rain_mm.mean()/silo.pet_mm.mean() if var=='aridity' else tp[var].values)
        S=inputs[model._static_vars].to_numpy()
        if not np.isfinite(S).all():raise ValueError('Missing terrain/soil values at sensors')
        static[name]=S
        z=(S-model._static_mean)/model._static_std
        for i,p in enumerate(points.itertuples()):
            for j,var in enumerate(model._static_vars):
                lo=float(train[var].min()) if var in train else np.nan
                hi=float(train[var].max()) if var in train else np.nan
                audit.append(dict(label=p.label,scenario=name,variable=var,value=float(S[i,j]),training_z=float(z[i,j]),training_min=lo,training_max=hi,outside_training_range=bool(S[i,j]<lo or S[i,j]>hi)))
    pd.DataFrame(audit).to_csv(root/'outputs/covariate_training_audit.csv',index=False)
    frames=[]
    for name,S in static.items():
        for i,p in enumerate(points.itertuples()):
            pred=model.readout(daily_storage[:,i],np.repeat(S[i:i+1],len(silo),axis=0))
            frames.append(pd.DataFrame(dict(date=silo.index,device=p.device,label=p.label,scenario=name,sm_pct=pred,storage_mm=daily_storage[:,i])))
    daily=pd.concat(frames,ignore_index=True)
    daily=daily[daily.date>=pd.Timestamp('2025-03-01')]
    daily.to_csv(root/'outputs/daily_predictions.csv',index=False)
    end=pd.Timestamp(cfg['analysis_date'])-pd.Timedelta(days=1)
    start=pd.Timestamp(cfg['analysis_date'])-pd.DateOffset(years=1)
    ranked=silo.loc[start:end,['rain_mm']].reset_index().sort_values(['rain_mm','date'],ascending=[False,True])
    ranked['rain_rank']=np.arange(1,len(ranked)+1);ranked.to_csv(root/'outputs/rainfall_past_year_ranked.csv',index=False)
    main=select_events(ranked,cfg['events_per_group'],cfg['event_separation_days']);main['group']='past_year_daily';main['rain_source']='SILO daily';main['coverage_hours']=24
    # Historical supplement uses observed gauge subtotals, never silently treats
    # absent hours as measured zero. Require at least 22 observed hours per day.
    local_index=rain.index.tz_convert(cfg['display_timezone'])
    gauge=rain.copy();gauge['date']=local_index.tz_localize(None).normalize()
    gd=gauge.groupby('date').agg(rain_mm=('rain_mm',lambda x:x.sum(min_count=1)),coverage_hours=('observed','sum')).reset_index()
    gd.to_csv(root/'outputs/rainfall_gauge_daily_subtotals.csv',index=False)
    valid_obs=observations[observations.sm_pct.notna() & observations.device.isin(points.loc[points.is_gully,'device'])]
    obs_min=valid_obs.time.min().tz_convert(cfg['display_timezone']).tz_localize(None)
    obs_max=valid_obs.time.max().tz_convert(cfg['display_timezone']).tz_localize(None)
    eligible=gd[(gd.coverage_hours>=22)&(gd.date>=obs_min+pd.Timedelta(days=cfg['pre_days']))&(gd.date<=min(obs_max,gauge.date.max())-pd.Timedelta(days=cfg['post_days']))]
    hist=select_events(eligible,cfg['events_per_group'],cfg['event_separation_days']);hist['group']='historical_hourly';hist['rain_source']='Gauge observed subtotal (incomplete days)'
    events=pd.concat([main,hist],ignore_index=True)
    events['event_id']=[f'{r.group}_{r.date:%Y%m%d}' for r in events.itertuples()]
    events.to_csv(root/'outputs/selected_events.csv',index=False)
    hourly_rows=[];forcing_rows=[]
    for event in events[events.group=='historical_hourly'].itertuples():
        begin=(event.date-pd.Timedelta(days=cfg['pre_days'])).tz_localize(cfg['display_timezone'])
        finish=(event.date+pd.Timedelta(days=cfg['post_days']+1)).tz_localize(cfg['display_timezone'])
        idx=pd.date_range(begin,finish,freq='h',inclusive='left').tz_convert('UTC')
        forcing=rain.reindex(idx).copy()
        # SILO daily PET is allocated evenly over each local day's actual hours.
        # This is an explicit approximation, not observed hourly evaporation.
        days=idx.tz_convert(cfg['display_timezone']).tz_localize(None).normalize()
        n_hours=np.asarray([(d.tz_localize(cfg['display_timezone'])+pd.DateOffset(days=1)-d.tz_localize(cfg['display_timezone'])).total_seconds()/3600 for d in days])
        pet=silo.pet_mm.reindex(days).to_numpy()/n_hours
        missing=forcing.rain_mm.isna().to_numpy()
        fallback=silo.rain_mm.reindex(days).to_numpy()/n_hours
        previous_day=begin.tz_localize(None).normalize()-pd.Timedelta(days=1)
        initial=daily_storage[silo.index.get_loc(previous_day)]
        for policy in ['silo_fraction','zero_missing']:
            pp=forcing.rain_mm.to_numpy().copy();pp[missing]=fallback[missing] if policy=='silo_fraction' else 0
            stor=hourly_bucket(pp,pet,capacity,alpha,k,initial)
            forcing_rows.append(pd.DataFrame(dict(event_id=event.event_id,time=idx,rain_mm=pp,pet_mm=pet,gauge_observed=~missing,gap_policy=policy)))
            for name,S in static.items():
                for i,p in enumerate(points.itertuples()):
                    pred=model.readout(stor[:,i],np.repeat(S[i:i+1],len(idx),axis=0))
                    hourly_rows.append(pd.DataFrame(dict(event_id=event.event_id,time=idx+pd.Timedelta(hours=1),device=p.device,label=p.label,scenario=name,gap_policy=policy,sm_pct=pred,storage_mm=stor[:,i])))
    hourly=pd.concat(hourly_rows,ignore_index=True)
    hourly.to_csv(root/'outputs/hourly_experimental_predictions.csv',index=False)
    pd.concat(forcing_rows,ignore_index=True).to_csv(root/'outputs/hourly_experimental_forcing.csv',index=False)
    metrics=[];invariance=[]
    for e in events.itertuples():
        ev=e.date.tz_localize(cfg['display_timezone'])
        if e.group=='past_year_daily':
            d=daily[(daily.date>=e.date-pd.Timedelta(days=cfg['pre_days']))&(daily.date<=e.date+pd.Timedelta(days=cfg['post_days']))].copy()
            d['time']=d.date.dt.tz_localize(cfg['display_timezone']);d['gap_policy']='daily'
        else:d=hourly[hourly.event_id==e.event_id].copy()
        for (device,policy),g in d.groupby(['device','gap_policy']):
            pivot=g.pivot(index='time',columns='scenario',values='sm_pct').sort_index()
            for name in cfg['incisions_m']:
                vals=pivot[name].values
                metrics.append(dict(event_id=e.event_id,device=device,label=points.set_index('device').loc[device,'label'],scenario=name,gap_policy=policy,**event_metrics(pivot.index,vals,ev)))
                difference=pivot[name]-pivot.baseline
                invariance.append(dict(event_id=e.event_id,device=device,scenario=name,gap_policy=policy,offset_pp=float(difference.mean()),offset_time_range_pp=float(difference.max()-difference.min())))
    metrics=pd.DataFrame(metrics);metrics.to_csv(root/'outputs/event_metrics.csv',index=False)
    inv=pd.DataFrame(invariance);inv.to_csv(root/'outputs/scenario_invariance.csv',index=False)
    if inv.offset_time_range_pp.max()>1e-9:raise AssertionError('Unexpected terrain-induced dynamics in additive model8')
    metadata=dict(artifact_sha256=file_hash(artifact),parameters=model.params_.to_dict(),static_variables=model._static_vars,
        aridity_window=[str(silo.index.min().date()),str(silo.index.max().date())],aridity=float(silo.rain_mm.mean()/silo.pet_mm.mean()),
        hourly_k=float(1-(1-k)**(1/24)),daily_k=float(k),max_scenario_offset_time_range_pp=float(inv.offset_time_range_pp.max()),
        past_year=[str(start.date()),str(end.date())],hourly_status='experimental timestep adaptation, not recalibrated or validated',
        hourly_gap_policies=['SILO daily rain / hours in day at missing gauge hours','zero at missing gauge hours'],
        hourly_initialization='daily SILO bucket storage on the day preceding the event window; approximate time alignment',
        rain_timezone_basis='UTC assumed from original R ymd_hms() default; CSV has no timezone metadata',
        elevation_crs_basis='LAZ has no CRS; EPSG:28355 assigned from neighbouring DEM and Brindabella source processing records')
    (root/'outputs/model_metadata.json').write_text(json.dumps(metadata,indent=2))
    return daily,hourly,events,metrics,metadata
