# Capacity-depth sensitivity results

[Interactive dashboard](capacity_depth_dashboard.html) · [Hourly figure](capacity_depth_hourly.png) · [Daily figure](capacity_depth_daily.png)

Nine capacity/DEM combinations over 1,501 pixels. Model coefficients remain fixed. Factor-one control differs from the prior bucket total by at most 1.96e-10 m³.

Largest selected past-year event: 2026-03-03. Values below cover the selected event window, including subsequent rainfall.

| Capacity factor | Capacity (m³) | Peak storage (m³) | Rise above pre-event (m³) | Peak fullness (%) | Post-event excess (m³) |
|---|---:|---:|---:|---:|---:|
| 0.25 | 1699.7 | 1699.7 | 1166.6 | 100.0 | 654.5 |
| 0.5 | 3399.4 | 2833.9 | 1890.4 | 83.4 | 0.0 |
| 1 | 6798.9 | 3513.8 | 1960.1 | 51.7 | 0.0 |

The DEM variants have identical bucket dynamics at a fixed capacity factor. Their legacy fitted SM readouts retain terrain offsets. Shallower effective capacity is a hypothesis separate from gully incision; it is not evidence of the real hydrological effect of erosion.

The main quantities are conceptual bucket water volume and capacity-weighted fullness. Excess is post-loss clipping, not routed runoff. Fullness is not volumetric SM. Hourly runs retain the earlier timing and missing-rain assumptions.
