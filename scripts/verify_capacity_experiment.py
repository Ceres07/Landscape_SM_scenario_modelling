"""Validate the exported factorial experiment, independently of plotting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs'
d=pd.read_csv(out/'capacity_depth_timeseries.csv').fillna({'event_id':''})
meta=json.loads((out/'capacity_depth_metadata.json').read_text())
d.time=pd.to_datetime(d.time,utc=True)
keys=['series','event_id','gap_policy','region','dem_scenario','capacity_factor','time']
assert not d.duplicated(keys).any()
assert len(d[['dem_scenario','capacity_factor']].drop_duplicates())==meta['factorial_combinations']
assert np.isfinite(d[['storage_m3','capacity_m3','rain_m3','aet_m3','drainage_m3','excess_m3']]).all().all()
assert (d.storage_m3>=-1e-9).all() and (d.storage_m3<=d.capacity_m3+1e-8).all()
np.testing.assert_allclose(d.fullness_pct,100*d.storage_m3/d.capacity_m3,atol=1e-10)
assert d.floor_correction_m3.max()==0, 'Unexpected lower clipping in this data; inspect loss demand'
assert d.balance_residual_m3.abs().max()<1e-8
max_balance=0.
process=d[d.dem_scenario=='baseline']
for _,g in process.groupby(['series','event_id','gap_policy','region','capacity_factor'],dropna=False):
    g=g.sort_values('time')
    residual=(g.storage_m3.diff()-g.rain_m3+g.aet_m3+g.drainage_m3+g.excess_m3-g.floor_correction_m3).iloc[1:]
    max_balance=max(max_balance,float(residual.abs().max()))
    assert residual.abs().max()<1e-8
    assert np.ptp(g.capacity_m3)<1e-8
for _,g in d.groupby(['series','event_id','gap_policy','region','capacity_factor','time'],dropna=False):
    for variable in ['storage_m3','excess_m3','fullness_pct']:
        assert np.ptp(g[variable])<1e-8
# Factor one matches the former experiment rather than changing its baseline.
old=pd.read_csv(out/'gully_storage_timeseries.csv').fillna({'event_id':''})
old.time=pd.to_datetime(old.time,utc=True)
join=['series','event_id','gap_policy','region','scenario','time']
control=d[d.capacity_factor==1].rename(columns={'dem_scenario':'scenario'}).merge(old,on=join,validate='one_to_one')
assert len(control)==len(old)
np.testing.assert_allclose(control.storage_m3,control.bucket_storage_m3,atol=1e-8,rtol=0)
# Capacity perturbations must have a time-varying effect, not a mere offset.
union=process[(process.region=='All gullies (union)')&(process.series=='daily')]
p=union.pivot(index='time',columns='capacity_factor',values='storage_m3')
for factor in meta['capacity_factors']:
    if factor!=1:assert np.ptp(p[factor]-p[1.])>1, 'Capacity factor did not change dynamics'
# This dataset has disjoint gully masks, so individual totals reproduce union.
for _,g in process.groupby(['series','event_id','gap_policy','capacity_factor','time'],dropna=False):
    np.testing.assert_allclose(g[g.region!='All gullies (union)'].storage_m3.sum(),g[g.region=='All gullies (union)'].storage_m3.iloc[0],atol=1e-8,rtol=0)
metrics=pd.read_csv(out/'capacity_depth_event_metrics.csv')
expected=len(pd.read_csv(out/'selected_events.csv'))*4*len(meta['dem_scenarios'])*len(meta['capacity_factors'])
# Daily events have one policy; historical hourly events have two.
assert len(metrics)==expected+len(pd.read_csv(out/'selected_events.csv').query("group == 'historical_hourly'"))*4*len(meta['dem_scenarios'])*len(meta['capacity_factors'])
summary=dict(status='passed',combinations=meta['factorial_combinations'],time_series_rows=len(d),event_metric_rows=len(metrics),max_exported_balance_residual_m3=max_balance,checks=['Full factorial coverage','Storage bounded by scenario capacity','Mass balance including explicit excess','No unreported lower-bound correction','Baseline reproduces previous daily and hourly outputs','Capacity changes dynamics','DEM leaves bucket dynamics unchanged at fixed capacity','Individual gully totals match union'])
(out/'capacity_depth_verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
