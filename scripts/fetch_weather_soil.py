"""Cache unmodified model8 forcing and soil inputs using configured PaddockTS access."""
import os,sys,json
from pathlib import Path
from datetime import date
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
settings=json.loads((ROOT/'config.json').read_text())
DMM=Path(settings['dmm_repo'])
sys.path.insert(0,str(DMM))
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/phenode_numba')
from PaddockTS.config import Config,config as original
from PaddockTS.query import Query
from PaddockTS.Environmental.SILO.download_silo import download_silo
from emt.slga import soil_covariates
from emt.covariates import sample_points
cfg=Config(out_dir=str(ROOT/'data/cache/out'),tmp_dir=str(ROOT/'data/cache/tmp'),email=original.email,tern_api_key=original.tern_api_key)
raw=ROOT/'data/raw'; raw.mkdir(parents=True,exist_ok=True)
bbox=settings['weather_bbox']
end=(pd.Timestamp(settings['analysis_date'])-pd.Timedelta(days=1)).date()
start=date.fromisoformat(settings['weather_start'])
q=Query(bbox=bbox,start=start,end=end,stub=f'phenode_{start}_{end}',config=cfg)
weather_current=False
if (raw/'silo_daily.csv').exists():
 existing=pd.read_csv(raw/'silo_daily.csv')
 weather_current=str(existing.iloc[0,0])==str(start) and str(existing.iloc[-1,0])==str(end)
if not weather_current:
 s=download_silo(q);s.to_csv(raw/'silo_daily.csv',index=False)
 print('Weather rows',len(s),'range',s.iloc[0,0],s.iloc[-1,0],flush=True)
if not (raw/'soil_points.csv').exists():
 soil=soil_covariates(q)
 records=[]
 for f in Path(settings['phenode_dir']).glob('WS-*.csv'):
  d=pd.read_csv(f);lon=float(d.Longitude.median());lat=float(d.Latitude.median())
  vals=sample_points(soil,lon,lat)
  records.append(dict(device=f.stem,label=d['Device Name'].iloc[0],lon=lon,lat=lat,**{v:float(vals[v]) for v in soil.data_vars}))
 pd.DataFrame(records).to_csv(raw/'soil_points.csv',index=False)
 print('Soil points cached',len(records),flush=True)
(raw/'forcing_provenance.json').write_text(json.dumps(dict(source='SILO DataDrill',bbox=bbox,requested_start=str(q.start),requested_end=str(q.end),soil='SLGA v2, root-zone 0–100 cm, thickness-weighted',weather_resolution='daily',weather_url='https://www.longpaddock.qld.gov.au/silo/'),indent=2))
