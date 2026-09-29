import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('NUMBA_CACHE_DIR','/private/tmp/phenode_numba')
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/phenode_mpl')
from gully_scenarios.data import load_phenode
from gully_scenarios.terrain import build_scenarios
from gully_scenarios.core import terrain_signature
cfg=json.loads((ROOT/'config.json').read_text())
obs,points,sources=load_phenode(cfg,ROOT)
build_scenarios(ROOT,cfg,points)
(ROOT/'outputs/terrain_signature.json').write_text(json.dumps(terrain_signature(cfg),indent=2))
print('Terrain scenarios ready',flush=True)
