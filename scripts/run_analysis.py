#!/usr/bin/env python3
import os,sys,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/phenode_numba')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/phenode_mpl')
import pandas as pd
from gully_scenarios.data import load_phenode,load_weather
from gully_scenarios.model import run_models
from gully_scenarios.report import make_report
from gully_scenarios.core import file_hash,terrain_signature
cfg=json.loads((ROOT/'config.json').read_text());(ROOT/'outputs').mkdir(exist_ok=True)
obs,points,sources=load_phenode(cfg,ROOT)
silo,rain=load_weather(cfg,ROOT)
signature=ROOT/'outputs/terrain_signature.json'
if not (ROOT/'outputs/terrain_points.csv').exists() or not signature.exists() or json.loads(signature.read_text())!=terrain_signature(cfg):
    subprocess.run([sys.executable,str(ROOT/'scripts/build_dem.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/prepare_terrain.py')],check=True)
terrain=pd.read_csv(ROOT/'outputs/terrain_points.csv')
print('Running frozen daily model and experimental hourly events...',flush=True)
daily,hourly,events,metrics,metadata=run_models(ROOT,cfg,points,terrain,silo,rain,obs)
print('Rendering event dashboard...',flush=True)
make_report(ROOT,cfg,points,daily,hourly,events,obs,silo,rain,metadata)
for key in ['laz','rain_hourly']:sources.append(dict(path=cfg[key],sha256=file_hash(cfg[key])))
(ROOT/'outputs/input_manifest.json').write_text(json.dumps(dict(config=cfg,inputs=sources),indent=2))
print(events[['group','date','rain_mm','coverage_hours']].to_string(index=False))
print('Maximum variation of scenario offset:',metadata['max_scenario_offset_time_range_pp'])
subprocess.run([sys.executable,str(ROOT/'scripts/write_results.py')],check=True)
print(ROOT/'outputs/index.html')
