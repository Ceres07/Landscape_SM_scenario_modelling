# Results

[Open the interactive dashboard](index.html). [All sensors](all_sensors_daily.html).

Terrain changes the predicted moisture level but does not change peak rise, timing or recession in this frozen additive model8.

| Gully | +1 m offset (percentage points) | +4 m offset (percentage points) |
|---|---:|---:|
| EOG | -0.566 | -1.551 |
| AOG | +0.002 | +0.009 |
| AEG? | -0.384 | -2.273 |

The cuts above are maximum centreline depths; at EOG and AEG? the sensor-pixel cuts are 0.75 and 3 m. AOG is cut by the full 1 and 4 m. Control sensors are unchanged in these geometries.

The maximum temporal variation of any scenario-minus-baseline offset is 3.55e-15 percentage points (floating-point noise).

## Selected rainfall days

| Series | Date | Rain (mm) |
|---|---|---:|
| past_year_daily | 2026-03-03 | 36.0 |
| past_year_daily | 2025-12-22 | 28.1 |
| past_year_daily | 2026-02-08 | 27.5 |
| historical_hourly | 2025-07-26 | 19.3 |
| historical_hourly | 2025-07-06 | 12.6 |
| historical_hourly | 2025-06-24 | 10.2 |

Past-year events use original daily model8 and SILO. Historical hourly totals are incomplete gauge subtotals, after UTC-to-Sydney conversion, and the hourly model is an unvalidated timestep adaptation. The daily Phenode readings are plotted only where available.

## Material limitations

- Most 5 m slopes exceed model8 training ranges. Absolute predictions are exploratory.
- Gully centrelines and widths are provisional; local source LAZ CRS was inferred from accompanying project data.
- Missing gauge midnight hours require assumptions; the two displayed gap treatments are sensitivity cases, not confidence bounds.
- Daily PET is distributed uniformly over each local day; hourly timing is unvalidated.
- Sensor depth is unspecified here; model8 represents a root-zone response. No point calibration was applied.
- Soil properties/depth are fixed despite the large incision. No lateral water routing is modelled.
- Metrics span the displayed window and may include later rainfall; recession is not necessarily an isolated drydown.
