"""Factorial capacity/DEM event metrics, scientific figures and offline dashboard."""
from __future__ import annotations
import json,html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

FACTOR_COLORS={1.:'#2563a6',.5:'#d48222',.25:'#b32342'}
DEM_LABELS={'baseline':'Original DEM','moderate_1m':'DEM incision +1 m','strong_4m':'DEM incision +4 m'}


def window(totals,event,cfg,region=None):
    ev=event.date.tz_localize(cfg['display_timezone'])
    lo=ev-pd.DateOffset(days=cfg['pre_days']);hi=ev+pd.DateOffset(days=cfg['post_days']+1)
    if event.group=='past_year_daily':d=totals[(totals.series=='daily')&(totals.time>=lo)&(totals.time<hi)]
    else:d=totals[totals.event_id==event.event_id]
    if region is not None:d=d[d.region==region]
    return ev,d


def write_event_metrics(totals,events,cfg,out):
    rows=[]
    for event in events.itertuples():
        ev,d=window(totals,event,cfg)
        for (region,dem,factor,policy),g in d.groupby(['region','dem_scenario','capacity_factor','gap_policy']):
            g=g.sort_values('time');pre=g[(g.time<ev)&(g.time>=ev-pd.Timedelta(hours=24))]
            post=g[g.time>=ev];flux=g[g.step_start>=ev]
            base=float(pre.storage_m3.mean());baseline_full=float(pre.fullness_pct.mean())
            peak=post.loc[post.storage_m3.idxmax()];rise=float(peak.storage_m3-base)
            later=post[(post.time>peak.time)&(post.storage_m3<=base+rise/2)]
            half=float((later.iloc[0].time-peak.time).total_seconds()/3600) if rise>0 and len(later) else np.nan
            dt=g.time.diff().dt.total_seconds()/3600;rate=g.storage_m3.diff()/dt
            rows.append(dict(event_id=event.event_id,date=str(event.date.date()),group=event.group,region=region,dem_scenario=dem,capacity_factor=factor,gap_policy=policy,capacity_m3=float(g.capacity_m3.iloc[0]),pre_storage_m3=base,peak_storage_m3=float(peak.storage_m3),peak_rise_m3=rise,pre_fullness_pct=baseline_full,peak_fullness_pct=float(post.fullness_pct.max()),peak_fullness_rise_pp=float(post.fullness_pct.max()-baseline_full),hours_to_peak=float((peak.time-ev).total_seconds()/3600),half_recession_hours=half,half_recession_censored=not np.isfinite(half),max_rise_m3_per_hour=float(rate[g.time>=ev].max()),post_rain_m3=float(flux.rain_m3.sum()),post_excess_m3=float(flux.excess_m3.sum()),post_drainage_m3=float(flux.drainage_m3.sum()),post_aet_m3=float(flux.aet_m3.sum()),post_max_at_capacity_area_pct=float(post.at_capacity_area_pct.max())))
    result=pd.DataFrame(rows);result.to_csv(out/'capacity_depth_event_metrics.csv',index=False)
    return result


def series_for(g,ev,quantity):
    x=(pd.DatetimeIndex(g.time)-ev).total_seconds()/3600
    if quantity=='rise_m3':
        v=g.storage_m3.to_numpy();return x,v-v[(x<0)&(x>=-24)].mean()
    if quantity=='cumulative_excess_m3':return x,np.where(g.step_start>=ev,g.excess_m3,0).cumsum()
    return x,g[quantity].to_numpy()


def make_capacity_report(root,cfg,totals,metrics,events,metadata):
    out=root/'outputs';factors=metadata['capacity_factors']
    colors={f:FACTOR_COLORS.get(f,['#6f4c9b','#20896f','#935b3b'][i%3]) for i,f in enumerate(factors)}
    regions=['All gullies (union)']+[r for r in totals.region.unique() if r!='All gullies (union)']
    quantities=[('storage_m3','Whole-gully bucket storage','Water (m³)'),('rise_m3','Increase above pre-event storage','Change (m³)'),('fullness_pct','Bucket fullness','Storage / capacity (%)'),('cumulative_excess_m3','Cumulative water exceeding capacity since selected day','Excess (m³)')]
    figures={}
    for event_index,event in enumerate(events.itertuples()):
        for region in regions:
            ev,d=window(totals,event,cfg,region)
            for dem in cfg['incisions_m']:
                gdem=d[d.dem_scenario==dem]
                fig=make_subplots(rows=3,cols=2,specs=[[{'colspan':2},None],[{},{}],[{},{}]],row_heights=[.18,.41,.41],subplot_titles=['Rain forcing (includes hourly gap estimates)']+[q[1] for q in quantities],vertical_spacing=.10,horizontal_spacing=.12)
                rainfall=gdem[(gdem.capacity_factor==1.)&(gdem.gap_policy!='zero_missing')].sort_values('time')
                rx=(pd.DatetimeIndex(rainfall.step_start)-ev).total_seconds()/3600
                fig.add_trace(go.Bar(x=rx,y=rainfall.rain_m3/rainfall.area_m2*1000,marker_color='#78a6bd',name='Rain forcing',width=20 if event.group=='past_year_daily' else .85),row=1,col=1)
                fig.update_yaxes(title_text='Rain (mm)',row=1,col=1)
                for factor in factors:
                    g=gdem[(gdem.capacity_factor==factor)&(gdem.gap_policy!='zero_missing')].sort_values('time')
                    z=gdem[(gdem.capacity_factor==factor)&(gdem.gap_policy=='zero_missing')].sort_values('time')
                    for i,(key,title,unit) in enumerate(quantities):
                        row,col=i//2+2,i%2+1;xx,yy=series_for(g,ev,key)
                        fig.add_trace(go.Scatter(x=xx,y=yy,name=f'{factor:g}× capacity',legendgroup=str(factor),showlegend=i==0,line=dict(color=colors[factor],width=2.5)),row=row,col=col)
                        if len(z):
                            zx,zy=series_for(z,ev,key)
                            fig.add_trace(go.Scatter(x=zx,y=zy,name=f'{factor:g}× · zero missing rain',legendgroup=str(factor),showlegend=False,opacity=.35,line=dict(color=colors[factor],width=1,dash='dot')),row=row,col=col)
                        fig.update_yaxes(title_text=unit,row=row,col=col)
                        fig.update_xaxes(title_text='Hours from rainfall-day start (Sydney)',row=row,col=col)
                fig.update_yaxes(range=[0,105],row=3,col=1)
                area=float(gdem.area_m2.iloc[0]);mode='daily recurrence' if event.group=='past_year_daily' else 'experimental hourly timestep'
                fig.update_layout(title=dict(text=f'{region} · {event.date:%d %b %Y} · {DEM_LABELS[dem]}<br><sup>{area:,.0f} m² fixed footprint · {mode} · independently spun-up capacities</sup>'),height=1030,template='plotly_white',hovermode='x unified',legend=dict(orientation='h',y=1.10),margin=dict(t=120,b=70,l=80,r=40))
                fig.update_xaxes(range=[-24*cfg['pre_days'],24*(cfg['post_days']+1)])
                figures[f'{event_index}|{region}|{dem}']=json.loads(fig.to_json())
    # Facets explicitly show all nine factor combinations; dynamics are identical
    # across DEM columns at fixed capacity, an expected structural result.
    for group,name in [('past_year_daily','capacity_depth_daily'),('historical_hourly','capacity_depth_hourly')]:
        event=next(events[events.group==group].itertuples());ev,d=window(totals,event,cfg,'All gullies (union)')
        fig,axes=plt.subplots(5,3,figsize=(15,14),sharex=True,sharey='row',gridspec_kw={'height_ratios':[.55,1,1,1,1]})
        for col,dem in enumerate(cfg['incisions_m']):
            axes[0,col].set_title(DEM_LABELS[dem])
            rainfall=d[(d.dem_scenario==dem)&(d.capacity_factor==1.)&(d.gap_policy!='zero_missing')].sort_values('time')
            rx=(pd.DatetimeIndex(rainfall.step_start)-ev).total_seconds()/3600
            axes[0,col].bar(rx,rainfall.rain_m3/rainfall.area_m2*1000,width=20 if group=='past_year_daily' else 1,color='#78a6bd')
            if col==0:axes[0,col].set_ylabel('Rain forcing (mm)')
            for factor in factors:
                g=d[(d.dem_scenario==dem)&(d.capacity_factor==factor)&(d.gap_policy!='zero_missing')].sort_values('time')
                z=d[(d.dem_scenario==dem)&(d.capacity_factor==factor)&(d.gap_policy=='zero_missing')].sort_values('time')
                for row,(quantity,title,unit) in enumerate(quantities):
                    xx,yy=series_for(g,ev,quantity);ax=axes[row+1,col]
                    ax.plot(xx,yy,color=colors[factor],lw=2,label=f'{factor:g}× capacity')
                    if len(z):
                        zx,zy=series_for(z,ev,quantity);ax.fill_between(xx,np.minimum(yy,zy),np.maximum(yy,zy),color=colors[factor],alpha=.1)
                    if col==0:ax.set_ylabel(('Storage (m³)','Storage increase (m³)','Fullness (%)','Cumulative excess (m³)')[row])
            axes[3,col].set_ylim(0,105)
            axes[-1,col].set_xlabel('Hours from rainfall-day start')
        for ax in axes.ravel():ax.grid(alpha=.18);ax.axvline(0,color='#999',lw=.8);ax.set_xlim(-24*cfg['pre_days'],24*(cfg['post_days']+1))
        axes[1,0].legend(fontsize=9)
        fig.suptitle(f'Capacity × DEM incision experiment · {event.date:%d %b %Y}\nCombined gully footprint · '+('daily model recurrence' if group=='past_year_daily' else 'experimental hourly timestep; gap treatments shaded'),fontsize=14)
        fig.text(.5,.012,'Each capacity uses its own antecedent-weather spin-up. Excess is upper-clipped bucket water, not routed runoff. DEM does not alter bucket dynamics.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.035,1,.94]);fig.savefig(out/f'{name}.png',dpi=170);fig.savefig(out/f'{name}.pdf');plt.close(fig)
    example=next(events[events.group=='past_year_daily'].itertuples())
    table=metrics[(metrics.event_id==example.event_id)&(metrics.region=='All gullies (union)')&(metrics.dem_scenario=='baseline')][['capacity_factor','capacity_m3','pre_storage_m3','peak_storage_m3','peak_rise_m3','peak_fullness_pct','post_excess_m3','half_recession_hours']].round(2)
    eopts=''.join(f'<option value="{i}">{e.date:%Y-%m-%d} · {e.group.replace("_"," ")}</option>' for i,e in enumerate(events.itertuples()))
    ropts=''.join(f'<option>{html.escape(r)}</option>' for r in regions)
    dopts=''.join(f'<option value="{k}">{v}</option>' for k,v in DEM_LABELS.items())
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Capacity depth × DEM incision</title>
<style>body{{font:16px/1.55 system-ui;color:#243142;background:#f6f8fa;margin:0}}main{{max-width:1300px;margin:auto;padding:26px}}.card{{background:white;border:1px solid #dce3e9;border-radius:10px;padding:20px;margin:18px 0}}select{{padding:9px;font:inherit;max-width:100%}}label{{display:inline-block;margin:5px 16px 5px 0}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #ddd}}a{{color:#2563a6}}.scroll{{overflow:auto}}</style><main>
<h1>Capacity depth × DEM incision</h1><p>{metadata['factorial_combinations']} combinations: effective capacity factors {', '.join(f'{f:g}×' for f in factors)} crossed with original, +1 m and +4 m DEM incision. The analysis uses the same 1,501 cells and 37,525 m² combined gully footprint.</p>
<div class="card"><strong>Capacity now changes the simulation itself.</strong><p>Each factor multiplies the existing soil-dependent bucket capacity, including the capacity used for evaporation stress and upper storage clipping. Every capacity is independently spun up from January 2023 with the same weather. The fitted drainage and evaporation coefficients remain unchanged.</p>
<p>Changing the DEM selector leaves these bucket curves unchanged: model8 still uses terrain only in its additive moisture readout. This experiment separates a capacity effect from a terrain effect rather than assuming that incision automatically removes a particular soil depth.</p></div>
<div class="card"><label>Rainfall event<br><select id="event">{eopts}</select></label><label>Area<br><select id="region">{ropts}</select></label><label>DEM scenario<br><select id="dem">{dopts}</select></label><div id="plot"></div>
<p>Faint hourly curves use zero for missing gauge rain; principal curves use the existing SILO-fraction gap treatment. Neither is an uncertainty bound. Past-year daily events have daily information even though their horizontal axis is in hours.</p></div>
<div class="card"><h2>Largest selected past-year event · {example.date:%d %b %Y}</h2><div class="scroll">{table.to_html(index=False)}</div><p>All volumes are m³. Capacity is the available space in the conceptual bucket; fullness is total storage divided by total capacity. Excess is summed after the selected rainfall-day start through the plotting window, and is not channel discharge. Blank recession times mean the half-recovery threshold was not reached in the window.</p></div>
<div class="card"><h2>What is held fixed</h2><ul><li>Rainfall, PET, soil AWC, gully footprint and fitted coefficients are identical across capacity cases. Only the effective capacity multiplier changes.</li><li>Factors 1, 0.5 and 0.25 are relative-depth hypotheses. They are not measured depths of 1, 0.5 and 0.25 m unless an independent 1 m effective-depth reference is justified.</li><li>Storage is aggregated from the internal bucket in millimetres × area / 1000. It does not require the assumed 1 m layer used by the earlier SM-derived volume plots.</li><li>Fullness is not volumetric soil moisture. The original fitted SM readout is retained only as the CSV diagnostic <code>legacy_sm_pct</code>; its global denominator has not been reinterpreted as a depth-aware moisture calibration.</li></ul></div>
<div class="card"><h2>Water balance and interpretation</h2><p>The original recurrence adds rain, removes model AET and drainage, then clips storage to capacity. The new accounting records that upper-clipped excess. If the lower bound ever clips a negative state, a separate floor-correction diagnostic reports it; it is not disguised as a physical input.</p><p>Factor 1 reproduces the earlier daily and hourly results. Maximum budget residual: {metadata['max_budget_residual_mm']:.2g} mm. No lateral routing, surface ponding or bedrock-depth model has been added. Altered-capacity and hourly responses remain sensitivity experiments without recalibration.</p><p>Event metrics are relative to the chosen rainfall-day start and can include later rainfall. They do not estimate an isolated drainage constant. The original experiment's uncertain gauge timezone, missing hours and coarse PET assumptions still apply.</p></div>
<div class="card"><h2>Files</h2><p><a href="capacity_depth_hourly.png">Hourly figure: all nine combinations</a> · <a href="capacity_depth_daily.png">Daily figure</a> · <a href="capacity_depth_hourly.pdf">Hourly PDF</a> · <a href="capacity_depth_daily.pdf">Daily PDF</a></p><p><a href="capacity_depth_timeseries.csv">All time series</a> · <a href="capacity_depth_event_metrics.csv">Event metrics</a> · <a href="capacity_depth_budget_audit.csv">Water-balance checks</a> · <a href="capacity_depth_metadata.json">Parameters and provenance</a></p><p><a href="whole_gully_storage.html">Earlier fixed-capacity area totals</a> · <a href="index.html">Original sensor-point plots</a></p></div></main>
<script>__PLOTLY__</script><script>const figures=__FIGURES__;function show(){{const key=document.getElementById('event').value+'|'+document.getElementById('region').value+'|'+document.getElementById('dem').value;const f=figures[key];Plotly.react('plot',f.data,f.layout,{{responsive:true,displaylogo:false}});}}for(const id of ['event','region','dem'])document.getElementById(id).addEventListener('change',show);show();</script></html>'''
    (out/'capacity_depth_dashboard.html').write_text(page.replace('__PLOTLY__',get_plotlyjs()).replace('__FIGURES__',json.dumps(figures)))
    lines=['# Capacity-depth sensitivity results','','[Interactive dashboard](capacity_depth_dashboard.html) · [Hourly figure](capacity_depth_hourly.png) · [Daily figure](capacity_depth_daily.png)','',f"Nine capacity/DEM combinations over 1,501 pixels. Model coefficients remain fixed. Factor-one control differs from the prior bucket total by at most {metadata['max_factor_one_difference_m3']:.3g} m³.",'',f'Largest selected past-year event: {example.date:%Y-%m-%d}. Values below cover the selected event window, including subsequent rainfall.','', '| Capacity factor | Capacity (m³) | Peak storage (m³) | Rise above pre-event (m³) | Peak fullness (%) | Post-event excess (m³) |','|---|---:|---:|---:|---:|---:|']
    for r in table.itertuples():lines.append(f'| {r.capacity_factor:g} | {r.capacity_m3:.1f} | {r.peak_storage_m3:.1f} | {r.peak_rise_m3:.1f} | {r.peak_fullness_pct:.1f} | {r.post_excess_m3:.1f} |')
    lines+=['','The DEM variants have identical bucket dynamics at a fixed capacity factor. Their legacy fitted SM readouts retain terrain offsets. Shallower effective capacity is a hypothesis separate from gully incision; it is not evidence of the real hydrological effect of erosion.','', 'The main quantities are conceptual bucket water volume and capacity-weighted fullness. Excess is post-loss clipping, not routed runoff. Fullness is not volumetric SM. Hourly runs retain the earlier timing and missing-rain assumptions.']
    (out/'CAPACITY_DEPTH_RESULTS.md').write_text('\n'.join(lines)+'\n')
