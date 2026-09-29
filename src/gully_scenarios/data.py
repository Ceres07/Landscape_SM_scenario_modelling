from pathlib import Path
import numpy as np
import pandas as pd
from .core import is_gully,file_hash


def load_phenode(cfg,root):
    observations=[];points=[];sources=[]
    for f in sorted(Path(cfg['phenode_dir']).glob('WS-*.csv')):
        d=pd.read_csv(f);t=pd.to_datetime(d.Timestamp,utc=True,errors='coerce')
        label=str(d['Device Name'].dropna().iloc[0]);sm=pd.to_numeric(d['Soil Moisture (VWC%)'],errors='coerce')
        sm=sm.where(sm.between(0,100))
        observations.append(pd.DataFrame(dict(device=f.stem,label=label,time=t,sm_pct=sm)))
        dt=t.sort_values().diff().dt.total_seconds()/3600
        points.append(dict(device=f.stem,label=label,is_gully=is_gully(label),label_uncertain='?' in label,
                           lon=float(d.Longitude.median()),lat=float(d.Latitude.median()),
                           median_interval_hours=float(dt.median()),n_valid_sm=int(sm.notna().sum()),
                           first=str(t.min()),last=str(t.max())))
        sources.append(dict(path=str(f),sha256=file_hash(f)))
    obs=pd.concat(observations,ignore_index=True)
    if obs.duplicated(['device','time']).any():raise ValueError('Duplicate Phenode timestamps')
    obs.to_csv(root/'outputs/phenode_observations.csv',index=False)
    return obs,pd.DataFrame(points),sources


def load_weather(cfg,root):
    s=pd.read_csv(root/'data/raw/silo_daily.csv').rename(columns={'YYYY-MM-DD':'date','daily_rain':'rain_mm','et_morton_potential':'pet_mm'})
    s['date']=pd.to_datetime(s.date);s=s.set_index('date').sort_index()
    if not s.index.equals(pd.date_range(s.index.min(),s.index.max(),freq='D')):raise ValueError('SILO missing dates')
    if not np.isfinite(s[['rain_mm','pet_mm']]).all().all():raise ValueError('SILO nonfinite forcing')
    d=pd.read_csv(cfg['rain_hourly']);t=pd.to_datetime(d.hour_time,errors='coerce')
    # Original R script used lubridate::ymd_hms(), whose default is UTC.
    t=t.dt.tz_localize(cfg['rain_timezone'],ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
    rain=pd.DataFrame(dict(time=t,rain_mm=pd.to_numeric(d.total_rainfall,errors='coerce'),n_obs=pd.to_numeric(d.n_obs_rain,errors='coerce'))).dropna(subset=['time'])
    if rain.time.duplicated().any():raise ValueError('Duplicate gauge hours')
    rain=rain.set_index('time').sort_index()
    rain['rain_mm']=rain.rain_mm.where((rain.rain_mm>=0)&(rain.n_obs>0))
    rain=rain.reindex(pd.date_range(rain.index.min().floor('D'),rain.index.max().ceil('D')-pd.Timedelta(hours=1),freq='h'))
    rain.index.name='time';rain['observed']=rain.rain_mm.notna()
    rain.to_csv(root/'outputs/hourly_rain_with_gaps.csv')
    return s,rain
