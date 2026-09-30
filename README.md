# Phenode gully incision scenarios

A separate, reproducible sensitivity experiment using frozen DownscalingMoistureModel model8, real Phenode locations and a 5 m DEM generated from the supplied classified Brindabella LiDAR.

Open **[outputs/capacity_depth_dashboard.html](outputs/capacity_depth_dashboard.html)** for the capacity-depth × DEM experiment, including hourly whole-gully storage, fullness and cumulative excess. See **[outputs/CAPACITY_DEPTH_RESULTS.md](outputs/CAPACITY_DEPTH_RESULTS.md)** for numerical results.

The earlier **[outputs/whole_gully_storage.html](outputs/whole_gully_storage.html)** shows whole-gully water-storage comparisons. The original sensor-pixel dashboard is **[outputs/index.html](outputs/index.html)**. It works offline. The dropdown switches between three past-year daily events and three older gauge-driven hourly experiments. Downloadable PNG/PDF figures and CSV predictions are beside it. **[outputs/all_sensors_daily.html](outputs/all_sensors_daily.html)** shows all seven sensors over the full period.

## What this experiment tests

At each original sensor position, change terrain only: original DEM, an additional **1 m** centreline incision, and an additional **4 m** centreline incision. Keep weather, soil properties, model parameters and sensor coordinates fixed. A label ending in G after removing punctuation denotes a gully; this includes EOG, AOG and AEG?. Question marks are retained as uncertain labels.

**Model8's terrain effect is additive. The terrain-only experiment changes absolute soil moisture but cannot change the amplitude, timing or recession of the rainfall response.** The new capacity experiment below changes the process reservoir and therefore can change the dynamics. The dashboard shows absolute predictions and pre-event-adjusted responses side by side. Their overlapping response curves are expected by construction, not evidence that real incision has no hydrological effect.

## Inputs and dates

- Phenode: `borevitz_projects/Data/Phenode_wireless_data/WS-*.csv`. Daily exports, not hourly. Valid soil moisture begins 10 June 2025 and extends to 1 July 2026; EOR? ends earlier. Units are taken from the source VWC% heading. Sensor depths/calibration are not independently verified here.
- LiDAR: `/Volumes/Dmitry_work/Honours2025/DEM_processing/merged.laz`, identified by the user as the local ELVIS-area LAZ. Existing processing JSON names Brindabella201802 NSW Spatial Services point clouds. No fresh download was necessary. The LAS creation date is a processing date, not the survey date. The local source has no embedded CRS; EPSG:28355 is assigned from adjacent DEM metadata and source file naming, and should be checked against original delivery metadata before publication.
- Hourly rain: `/Volumes/Dmitry_work/Honours2025/rain_hourly.csv`, March–August 2025. Each midnight hour is missing. Missing data are retained in the observation table. Hour presence and `n_obs_rain` are recorded; presence alone does not establish a complete rain-gauge hour.
- Daily rain and PET: cached SILO DataDrill data, 1 January 2023–28 September 2026, centred on the Phenode area. Daily SILO PET is Morton's potential ET, as used by model8. Daily rainfall dates are SILO date labels; they should not be interpreted as exact local midnight-to-midnight storm intervals.
- Soil: cached SLGA v2 clay, sand, AWC and bulk density, depth-weighted 0–100 cm using the existing DMM loader. Resolution is approximately 90 m, regardless of output DEM resolution.

“Past year” is explicitly **29 September 2025–28 September 2026**. All 365 days are ranked in `outputs/rainfall_past_year_ranked.csv`. The three highest separated days are 3 March 2026 (36.0 mm), 22 December 2025 (28.1 mm) and 8 February 2026 (27.5 mm). Events are separated by at least 10 days. The historical hourly supplement selects large observed gauge subtotals with >=22 reported hours/day and a full plotting window within the available gully-observation period. Historical events are **not** described as events from the past year.

## DEM alterations

PDAL reads classification 2 (ground) points and bins mean Z into aligned 5 m cells, with no canopy points and no interpolation from 30 m data. At most 10 m of local gap interpolation is permitted; larger missing areas stay nodata. D8 accumulation is computed after depression filling and flat resolution. The full 6 km × 4 km LiDAR extent is processed before sampling points.

Provisional centrelines snap within 20 m of each gully probe to a cell with high accumulation, then follow the principal upstream branch for up to 150 m and the downstream route for up to 300 m. A 30 m full-width cosine cross-section tapers to zero at the edges and over up to 50 m at either end. Overlapping incision masks use their maximum, not their sum. These are algorithmic scenario geometries, not surveyed gully bank boundaries; inspect `outputs/terrain_scenarios.png` and edit parameters as needed.

Cuts are exactly 1 m and 4 m at the centreline. At off-centre EOG and AEG? pixels they are 0.75/3 m, and at AOG they are 1/4 m. Original coordinates are retained. Both unconditioned scenario DEMs and the elevation raised by hydrological conditioning are saved. In the strong scenario, conditioning raises some footprint cells by up to about 1.73 m; this is visible in the QC rasters. It does not silently change the saved incision depths.

Slope and TWI follow DMM's conventions. TWI uses accumulation in **cell counts**, as in training. It is resolution dependent; changing to physical contributing area without refitting would also alter predictor meaning. Six of seven baseline slopes exceed the embedded training maximum of 14.73 degrees. `covariate_training_audit.csv` contains ranges and standardised values for every scenario. These are unvalidated extrapolations at 5 m.

## Whole-gully storage totals

The original time-series curves sample the 5 m cell containing each sensor. The storage extension instead predicts every valid cell in each **fixed original incision footprint**, then integrates over area. It includes EOG (12,000 m²), AOG (13,125 m²), AEG? (12,400 m²), and their 37,525 m² union. All 1,501 footprint cells have valid inputs in every scenario. Individual masks are disjoint here; the union calculation also handles overlap without counting it twice.

Two different quantities are plotted:

1. **SM-derived volume equivalent:** `sum(SM_percent / 100 × area_m2 × assumed_soil_depth_m)`. Default soil depth is explicitly assumed to be 1 m; use `--soil-depth-m` to change it. This means a uniform 1 m layer below each scenario surface, not the entire geological soil profile. Model8's fitted moisture readout is not a mass-conserving water-volume state. These totals are exploratory equivalents, not validated actual water inventories.
2. **Internal bucket storage volume:** `sum(storage_mm × area_m2 / 1000)`. This is the model's conceptual active reservoir; it is not total soil pore water. Terrain does not alter it, so all scenario curves overlap exactly.

The totals describe water present **at each time**. They are not a cumulative sum of hourly storage values, which would repeatedly count the same water. The right-hand panels show change from pre-event storage.

At the assumed 1 m depth, the combined SM-derived volume differs by approximately **−128 m³** for the moderate incision and **−539 m³** for the strong incision. These differences are constant through time; they do not represent extra water lost from the model's bucket. The calculation does not include surface runoff, ponding, groundwater, upstream catchment storage, or the soil volume physically removed by erosion. Soil depth and properties remain fixed across scenarios. Test these assumptions before treating the values as a physical erosion response.

Run the extension separately after the original analysis:

```bash
/opt/miniconda3/envs/paddockts/bin/python scripts/run_area_storage.py --soil-depth-m 1
/opt/miniconda3/envs/paddockts/bin/python scripts/verify_area_storage.py
```

It reuses the existing cached soil rasters, weather and DEMs without network access. Results are in `gully_storage_timeseries.csv`, `gully_storage_areas.csv`, `gully_storage_scenario_comparison.csv`, and `gully_storage_metadata.json`. The standalone report and PNG/PDF plots are named `whole_gully_*`.

## Capacity-depth factors crossed with incision

The new experiment crosses **1×, 0.5× and 0.25× effective capacity** with each original, +1 m and +4 m incision DEM: nine combinations. Each uses the same fixed gully footprint, weather and frozen fitted coefficients. Change `capacity_depth_factors` in `config.json`, or pass `--factors 1 0.5 0.25` to the script; the factor-one control is required.

At each cell, `capacity_mm = factor × fitted_smax × SLGA_AWC / training_mean_AWC`. This is a relative effective-depth hypothesis under unchanged soil water-holding properties. It is not a measured soil depth, and incision metres are not subtracted from an assumed soil profile. Unlike the earlier `--soil-depth-m` option, which only scales the SM-derived volume conversion, these factors enter the water-balance recurrence itself:

```text
wet = previous_storage + rain
AET = PET × min(1, wet / (alpha × capacity))
drainage = timestep_k × wet
raw_storage = wet - AET - drainage
excess = max(raw_storage - capacity, 0)
storage = clip(raw_storage, 0, capacity)
```

Every capacity receives its own daily spin-up from January 2023, initialised at half of its own capacity. Hourly events start from that capacity's previous daily state. The fitted evaporation coefficient and daily drainage coefficient remain fixed; the existing hourly drainage conversion is retained. Upper-clipped excess is recorded after the model's losses. Any lower clipping is recorded separately as a numerical floor correction; none occurs in these runs.

The dashboard plots total bucket water (m³), change from the preceding 24-hour mean, capacity-weighted fullness (%) and cumulative excess (m³), with rainfall alongside. Select individual gullies or their union. Volume describes water present at each time; only fluxes such as excess are accumulated through time. Fullness is storage divided by effective capacity, not volumetric SM. `legacy_sm_pct` remains in the CSV as a diagnostic of the original fitted readout, whose global denominator is unchanged.

The three DEM cases still have identical bucket dynamics at a fixed capacity. There is no terrain-to-capacity coupling, lateral routing or surface ponding. Smaller capacities can fill sooner, hold less water and reject more rainfall; this tests a capacity mechanism without claiming that DEM incision alone causes it. Excess is water removed by upper clipping, not routed channel runoff. Hourly and altered-capacity results are uncalibrated sensitivity experiments.

For the March 2026 event window, peak combined fullness is 51.7%, 83.4% and 100% for full, half and quarter capacity; cumulative excess is 0, 0 and approximately 654 m³ respectively. Later rainfall within a plotted window can affect its peak and recession metrics.

Run from the repository after the existing area analysis:

```bash
/opt/miniconda3/envs/paddockts/bin/python scripts/run_capacity_experiment.py
/opt/miniconda3/envs/paddockts/bin/python scripts/verify_capacity_experiment.py
```

`run_analysis.py` also runs this extension automatically. Outputs use the `capacity_depth_*` prefix: standalone HTML, hourly/daily PNG and PDF figures, time-series CSV, event-metrics CSV, budget audit and metadata. `capacity_depth_metadata.json` records factors, source hashes, fixed parameters and control parity. The independent exported-output verification checks the factorial coverage, water balance, storage bounds, union totals and unchanged factor-one control. Seventeen unit tests cover this and the earlier extensions.

## Daily versus hourly model

The daily branch calls the original `emt.model7.model._step_loop` and the frozen model8 artifact's `readout`, using point-specific AWC capacities. This corresponds to the point-series calculation; it does not interpolate a coarse raster bucket. A common cached SILO series forces all nearby probes. The common aridity offset is fixed over the cached 2023–2026 period. No fitting or local correction occurs.

The optional historical hourly branch is an **experimental timestep adaptation**, not original or validated hourly model8:

- Initialise each event from the previous day's daily SILO bucket storage, with three days before the chosen rainfall day; this is approximate timing alignment.
- Apply recorded hourly rain where available.
- Allocate daily SILO PET uniformly over the hours in the local calendar day.
- Convert daily drainage fraction `k` to `1 - (1-k)**(1/24)`. This preserves drainage over a rain-free, PET-free 24 hours; it does not guarantee daily equivalence with rainfall/ET or clipping.
- Principal curves fill missing gauge hours using that local day's SILO daily rainfall divided by the number of hours. Companion faint curves use zero instead. Both are explicit assumptions; their spread is not a confidence interval or rigorous bound.
- Preserve all scenario offsets and original fitted parameters.

The gauge CSV lacks timezone metadata. UTC is assumed because its originating R script used `lubridate::ymd_hms()` with the default UTC timezone. Plot labels use Australia/Sydney. Confirm the original logger timezone before interpreting hourly lags. Original Phenode timestamps explicitly carry Z and are converted from UTC. The daily sensor observations are never interpolated to create hourly observations.

The peak/recession metrics describe the plotted window relative to the selected rainfall-day start. Subsequent rainfall can influence the window maximum and recession; they are not estimates of isolated storm drainage constants.

## Run

Use the existing environment with PaddockTS/DMM dependencies and the PDAL executable:

```bash
cd /Volumes/Dmitry_work/borevitz_projects/phenode_gully_scenarios
/opt/miniconda3/envs/paddockts/bin/python scripts/fetch_weather_soil.py
/opt/miniconda3/envs/paddockts/bin/python scripts/build_dem.py
/opt/miniconda3/envs/paddockts/bin/python scripts/prepare_terrain.py
/opt/miniconda3/envs/paddockts/bin/python scripts/run_analysis.py
/opt/miniconda3/envs/paddockts/bin/python -m unittest discover -s tests -v
/opt/miniconda3/envs/paddockts/bin/python scripts/verify_outputs.py
```

`fetch_weather_soil.py` is the only network step; it uses existing PaddockTS credentials without storing them in this repository. The other steps run offline with cached inputs. Paths and scenario parameters are in `config.json`. Raw source inputs and caches remain ignored by git. Generated outputs, including terrain rasters, dashboards, figures and CSV tables, are included in the repository. The GitHub repository is [Ceres07/Landscape_SM_scenario_modelling](https://github.com/Ceres07/Landscape_SM_scenario_modelling). Download or clone the repository and open the HTML dashboards in a browser; GitHub displays their source rather than running them. Run the analysis with the required input data to regenerate the outputs. Numerical results and verification metadata are also included in git.

## Verification and provenance

`outputs/input_manifest.json` records input SHA-256 hashes and configuration; `model_metadata.json` records the frozen model hash, fitted parameters and assumptions. The unit checks cover gully labelling, event separation, incision shape, explicit missing forcing and timestep drainage conversion. Integration checks verify grid alignment, exact maximum incision, unchanged cells outside the footprint, unchanged scenario storage, invariant response metrics and the full rainfall ranking. Scenario offset variation through time is below 1e-9 percentage points.

Source services: [ELVIS](https://elevation.fsdf.org.au/), [GA elevation information](https://www.ga.gov.au/scientific-topics/national-location-information/digital-elevation-data), [SILO](https://www.longpaddock.qld.gov.au/silo/), [PDAL GDAL raster writer](https://pdal.io/en/2.8.4/stages/writers.gdal.html). Original data ownership/licensing remains with the custodians; keep delivery metadata with any published derivatives.
