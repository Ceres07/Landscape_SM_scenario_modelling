"""Check the exported spatial storage series independently of the plot code."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs'
d=pd.read_csv(out/'gully_storage_timeseries.csv').fillna({'event_id':''})
areas=pd.read_csv(out/'gully_storage_areas.csv').set_index('region')
keys=['series','event_id','gap_policy','region','scenario','time']
assert not d.duplicated(keys).any()
assert (d.area_m2>0).all() and (d.soil_depth_m>0).all()
np.testing.assert_allclose(d.sm_equivalent_m3,d.mean_sm_pct/100*d.area_m2*d.soil_depth_m,rtol=1e-12)
assert (areas.excluded_area_m2==0).all(), 'Current dataset should have complete common-mask coverage'
assert areas.loc['All gullies (union)','valid_pixels']==1501
for _,g in d.groupby(['series','event_id','gap_policy','region'],dropna=False):
    v=g.pivot(index='time',columns='scenario',values='sm_equivalent_m3')
    b=g.pivot(index='time',columns='scenario',values='bucket_storage_m3')
    for name in ['moderate_1m','strong_4m']:
        assert np.ptp(v[name]-v.baseline)<1e-8
        np.testing.assert_allclose(b[name],b.baseline,rtol=0,atol=0)
# In this particular dataset the per-gully masks are disjoint, so sum=union.
assert areas.drop('All gullies (union)').valid_area_m2.sum()==areas.loc['All gullies (union)','valid_area_m2']
for _,g in d.groupby(['series','event_id','gap_policy','scenario'],dropna=False):
    p=g.pivot(index='time',columns='region',values='sm_equivalent_m3')
    np.testing.assert_allclose(p[['EOG','AOG','AEG?']].sum(axis=1),p['All gullies (union)'],rtol=1e-12)
summary={'status':'passed','checks':['VWC units and soil-depth conversion','common mask covers 1501 cells','no duplicate region/scenario/timestamps','no double counting in combined total','area-integrated readout has constant scenario offsets','internal bucket totals identical across scenarios']}
(out/'gully_storage_verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
