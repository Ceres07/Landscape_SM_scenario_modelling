"""Integration checks on real generated terrain and paired predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
ROOT=Path(__file__).resolve().parents[1]
with rasterio.open(ROOT/'outputs/terrain/baseline_dem_5m.tif') as src:
    base=src.read(1,masked=True);tx=src.transform;crs=src.crs
    assert src.res==(5,5)
with rasterio.open(ROOT/'outputs/terrain/incision_weight.tif') as src:w=src.read(1)
for name,depth in [('moderate_1m',1),('strong_4m',4)]:
    with rasterio.open(ROOT/f'outputs/terrain/{name}_dem_5m.tif') as src:
        a=src.read(1,masked=True)
        assert src.transform==tx and src.crs==crs and a.shape==base.shape
        np.testing.assert_allclose((base-a).compressed(),(depth*w)[~np.ma.getmaskarray(base)],atol=4e-5)
        np.testing.assert_array_equal(a.data[w==0],base.data[w==0])
        assert abs((base-a).max()-depth)<4e-5
inv=pd.read_csv(ROOT/'outputs/scenario_invariance.csv')
assert inv.offset_time_range_pp.max()<1e-9
metrics=pd.read_csv(ROOT/'outputs/event_metrics.csv')
for _,g in metrics.groupby(['event_id','device','gap_policy']):
    assert np.ptp(g.peak_rise_pp)<1e-9
    assert np.ptp(g.hours_to_peak)<1e-9
pred=pd.read_csv(ROOT/'outputs/daily_predictions.csv')
for _,g in pred.groupby(['date','device']):assert np.ptp(g.storage_mm)<1e-10
assert not pred.duplicated(['date','device','scenario']).any()
rank=pd.read_csv(ROOT/'outputs/rainfall_past_year_ranked.csv')
assert len(rank)==365
assert rank.rain_mm.is_monotonic_decreasing
with open(ROOT/'outputs/index.html') as f:page=f.read()
assert 'Plotly.react' in page and 'Experimental hourly' in page
summary=dict(status='passed',checks=['5 m grid and aligned scenarios','exact maximum 1 m and 4 m incision','no elevation edits outside footprint','no changes to bucket storage across scenarios','constant terrain offsets','invariant peak rises and peak timing','365 rainfall days ranked','standalone dashboard generated'],max_offset_variation_pp=float(inv.offset_time_range_pp.max()))
(ROOT/'outputs/verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
