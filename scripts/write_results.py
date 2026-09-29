"""Write a short, reproducible results note from generated tables."""
from pathlib import Path
import pandas as pd,json
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'outputs'
points=pd.read_csv(out/'points.csv')
inv=pd.read_csv(out/'scenario_invariance.csv')
table=inv.groupby(['device','scenario']).offset_pp.mean().unstack().join(points.set_index('device')[['label','is_gully']])
lines=['# Results','', '[Open the interactive dashboard](index.html). [All sensors](all_sensors_daily.html).','',
       'Terrain changes the predicted moisture level but does not change peak rise, timing or recession in this frozen additive model8.', '',
       '| Gully | +1 m offset (percentage points) | +4 m offset (percentage points) |', '|---|---:|---:|']
for r in table[table.is_gully].itertuples():lines.append(f'| {r.label} | {r.moderate_1m:+.3f} | {r.strong_4m:+.3f} |')
lines+=['','The cuts above are maximum centreline depths; at EOG and AEG? the sensor-pixel cuts are 0.75 and 3 m. AOG is cut by the full 1 and 4 m. Control sensors are unchanged in these geometries.','',
        'The maximum temporal variation of any scenario-minus-baseline offset is '+f'{inv.offset_time_range_pp.max():.3g}'+' percentage points (floating-point noise).','',
        '## Selected rainfall days','', '| Series | Date | Rain (mm) |', '|---|---|---:|']
for e in pd.read_csv(out/'selected_events.csv').itertuples():lines.append(f'| {e.group} | {e.date} | {e.rain_mm:.1f} |')
lines+=['','Past-year events use original daily model8 and SILO. Historical hourly totals are incomplete gauge subtotals, after UTC-to-Sydney conversion, and the hourly model is an unvalidated timestep adaptation. The daily Phenode readings are plotted only where available.','',
        '## Material limitations','',
        '- Most 5 m slopes exceed model8 training ranges. Absolute predictions are exploratory.',
        '- Gully centrelines and widths are provisional; local source LAZ CRS was inferred from accompanying project data.',
        '- Missing gauge midnight hours require assumptions; the two displayed gap treatments are sensitivity cases, not confidence bounds.',
        '- Daily PET is distributed uniformly over each local day; hourly timing is unvalidated.',
        '- Sensor depth is unspecified here; model8 represents a root-zone response. No point calibration was applied.',
        '- Soil properties/depth are fixed despite the large incision. No lateral water routing is modelled.',
        '- Metrics span the displayed window and may include later rainfall; recession is not necessarily an isolated drydown.','']
(out/'RESULTS.md').write_text('\n'.join(lines))
