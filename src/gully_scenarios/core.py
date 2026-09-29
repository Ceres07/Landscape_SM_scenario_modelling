from __future__ import annotations
import hashlib
import re
import numpy as np
import pandas as pd


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def is_gully(label):
    return re.sub(r'[^A-Z]', '', str(label).upper()).endswith('G')


def select_events(daily, n=3, separation_days=10, date_column='date', rain_column='rain_mm'):
    """Highest totals with deterministic ties and a minimum peak-day separation."""
    candidates = daily.dropna(subset=[rain_column]).sort_values([rain_column, date_column], ascending=[False, True])
    selected = []
    for _, r in candidates.iterrows():
        t = pd.Timestamp(r[date_column])
        if r[rain_column] > 0 and all(abs((t - pd.Timestamp(s[date_column])).days) >= separation_days for s in selected):
            selected.append(r.to_dict())
        if len(selected) >= n:
            break
    return pd.DataFrame(selected)


def hourly_bucket(rain, pet, capacity, alpha, k_daily, initial):
    """Experimental hourly recurrence, preserving the no-input daily drainage factor.

    Rain and PET are mm per hour. Initial storage is supplied by the daily
    model at the start of each event. This is NOT a calibrated hourly model8.
    """
    rain, pet = np.asarray(rain, float), np.asarray(pet, float)
    if not np.isfinite(rain).all() or not np.isfinite(pet).all():
        raise ValueError('Forcing gaps must be resolved explicitly before simulation')
    if np.any(rain < 0) or np.any(pet < 0):
        raise ValueError('Negative rain or PET')
    capacity = np.asarray(capacity, float)
    k_hour = 1 - (1 - k_daily) ** (1 / 24)
    s = np.asarray(initial, float).copy()
    result = []
    for p, e in zip(rain, pet):
        wet = s + p
        aet = e * np.minimum(1., wet / (alpha * capacity))
        s = np.clip(wet - aet - k_hour * wet, 0, capacity)
        result.append(s.copy())
    return np.asarray(result)


def incision_weight(distance, half_width):
    """Compact cosine cross-section: exact depth at centre, zero outside footprint."""
    return np.where(distance < half_width, .5 * (1 + np.cos(np.pi * distance / half_width)), 0.)


def event_metrics(times, values, event_time):
    t = pd.DatetimeIndex(times)
    values = np.asarray(values, float)
    pre = (t < event_time) & (t >= event_time - pd.Timedelta(days=1))
    after = t >= event_time
    base = np.nanmean(values[pre])
    delta = values - base
    indices = np.flatnonzero(after)
    peak = indices[np.nanargmax(delta[after])]
    peak_rise = float(delta[peak])
    later = np.flatnonzero((np.arange(len(t)) > peak) & (delta <= peak_rise / 2))
    half = float((t[later[0]] - t[peak]).total_seconds() / 3600) if len(later) and peak_rise > 0 else np.nan
    step_h = np.diff(t.asi8) / 3.6e12
    return dict(pre_sm_pct=float(base), peak_sm_pct=float(values[peak]), peak_rise_pp=peak_rise,
                hours_to_peak=float((t[peak]-event_time).total_seconds()/3600),
                half_recession_hours=half, max_rise_pp_per_hour=float(np.nanmax(np.diff(values)/step_h)))


def terrain_signature(cfg):
    keys = ['dem_crs', 'resolution_m', 'incisions_m', 'gully_half_width_m',
            'gully_upstream_length_m', 'gully_downstream_length_m', 'snap_radius_m']
    inputs = {k: cfg[k] for k in keys}
    inputs['laz_sha256'] = file_hash(cfg['laz'])
    inputs['phenode'] = {p.name: file_hash(p) for p in sorted(__import__('pathlib').Path(cfg['phenode_dir']).glob('WS-*.csv'))}
    inputs['algorithm_version'] = 2
    return inputs
