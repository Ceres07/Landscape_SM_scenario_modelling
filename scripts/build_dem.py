"""Generate an aligned 5 m bare-earth mean DEM directly from classified LAZ."""
import json, subprocess, sys
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parents[1]/'src'))
from gully_scenarios.core import file_hash
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'config.json').read_text())
raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
out=raw/'ground_5m.tif'
signature={'laz_sha256':file_hash(cfg['laz']),'crs':cfg['dem_crs'],'resolution':cfg['resolution_m']}
stamp=raw/'dem_signature.json'
if not out.exists() or not stamp.exists() or json.loads(stamp.read_text())!=signature:
    if cfg['resolution_m'] != 5: raise ValueError('This source grid is fixed at 5 m; change the PDAL grid geometry to change resolution')
    pipeline=[{'type':'readers.las','filename':cfg['laz'],'override_srs':cfg['dem_crs']},
      {'type':'filters.range','limits':'Classification[2:2]'},
      {'type':'writers.gdal','filename':str(out),'resolution':5,'output_type':'mean',
       'binmode':True,'origin_x':674000,'origin_y':6112000,'width':1200,'height':800,
       'nodata':-9999,'data_type':'float','gdalopts':'COMPRESS=DEFLATE,TILED=YES'}]
    path=raw/'pdal_pipeline.json';path.write_text(json.dumps(pipeline,indent=2))
    subprocess.run(['pdal','pipeline',str(path)],check=True)
    stamp.write_text(json.dumps(signature,indent=2))
print(out)
