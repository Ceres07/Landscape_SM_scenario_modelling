"""Plots of spatially integrated gully storage, with explicit volume definitions."""
import html,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs
from .report import COLORS,LABELS


def select_window(totals,e,cfg,region):
    ev=e.date.tz_localize(cfg['display_timezone'])
    if e.group=='past_year_daily':
        lo=ev-pd.DateOffset(days=cfg['pre_days']);hi=ev+pd.DateOffset(days=cfg['post_days']+1)
        d=totals[(totals.series=='daily')&(totals.region==region)&(totals.time>=lo)&(totals.time<hi)]
    else:d=totals[(totals.event_id==e.event_id)&(totals.region==region)]
    return ev,d


def make_area_report(root,cfg,totals,areas,checks,metadata):
    out=root/'outputs';depth=metadata['soil_depth_m']
    events=pd.read_csv(out/'selected_events.csv',parse_dates=['date'])
    rain=pd.read_csv(out/'hourly_experimental_forcing.csv',parse_dates=['time'])
    rain.time=pd.to_datetime(rain.time,utc=True)
    silo=pd.read_csv(root/'data/raw/silo_daily.csv',parse_dates=['YYYY-MM-DD']).rename(columns={'YYYY-MM-DD':'date'})
    regions=['All gullies (union)']+[x for x in areas.region if x!='All gullies (union)']
    quantities={'sm_equivalent_m3':f'SM-derived water volume · assumed {depth:g} m soil layer','bucket_storage_m3':'Internal model bucket storage · conceptual reservoir'}
    specs={};examples={}
    for event_idx,e in enumerate(events.itertuples()):
        for region in regions:
            ev,d=select_window(totals,e,cfg,region)
            def x(t):return (pd.DatetimeIndex(t)-ev).total_seconds()/3600
            if e.group=='past_year_daily':
                r=silo[(silo.date>=e.date-pd.Timedelta(days=cfg['pre_days']))&(silo.date<=e.date+pd.Timedelta(days=cfg['post_days']))]
                rx=x(r.date.dt.tz_localize(cfg['display_timezone']));ry=r.daily_rain;missing=np.zeros(len(r),bool)
            else:
                r=rain[(rain.event_id==e.event_id)&(rain.gap_policy=='silo_fraction')]
                rx=x(r.time);ry=r.rain_mm.where(r.gauge_observed);missing=~r.gauge_observed.to_numpy()
            for quantity,label in quantities.items():
                f=make_subplots(rows=2,cols=2,specs=[[{'colspan':2},None],[{},{}]],row_heights=[.25,.75],subplot_titles=['Rainfall','Water stored across the footprint','Change from pre-event storage'],horizontal_spacing=.11,vertical_spacing=.16)
                f.add_trace(go.Bar(x=rx,y=ry,name='Rain',marker_color='#78a6bd',width=20 if e.group=='past_year_daily' else .85),row=1,col=1)
                if missing.any():f.add_trace(go.Scatter(x=np.asarray(rx)[missing],y=np.zeros(missing.sum()),mode='markers',marker=dict(symbol='x',color='#c02738'),name='Missing gauge hour'),row=1,col=1)
                for scenario,color in COLORS.items():
                    g=d[(d.scenario==scenario)&(d.gap_policy!='zero_missing')].sort_values('time')
                    xx=x(g.time);yy=g[quantity].to_numpy();before=yy[(xx<0)&(xx>=-24)].mean()
                    for col,y in [(1,yy),(2,yy-before)]:
                        f.add_trace(go.Scatter(x=xx,y=y,name=LABELS[scenario],legendgroup=scenario,showlegend=col==1,mode='lines',line=dict(color=color,width=2.4,dash={'baseline':'solid','moderate_1m':'dash','strong_4m':'dot'}[scenario])),row=2,col=col)
                    z=d[(d.scenario==scenario)&(d.gap_policy=='zero_missing')].sort_values('time')
                    if len(z):f.add_trace(go.Scatter(x=x(z.time),y=z[quantity],name='Zero missing rain: '+LABELS[scenario],legendgroup=scenario,showlegend=False,line=dict(color=color,width=1),opacity=.35),row=2,col=1)
                area=float(areas.set_index('region').loc[region,'valid_area_m2'])
                mode='daily model8' if e.group=='past_year_daily' else 'experimental hourly model8'
                f.update_layout(title=dict(text=f'{region} · {e.date:%d %b %Y} · {area:,.0f} m²<br><sup>{label}; {mode}</sup>'),height=650,template='plotly_white',hovermode='x unified',legend=dict(orientation='h',y=1.10),margin=dict(t=120,b=70,l=80,r=30))
                f.update_yaxes(title_text='Rain (mm)',row=1,col=1)
                for col in [1,2]:
                    f.update_yaxes(title_text='Water (m³)' if col==1 else 'Change (m³)',row=2,col=col)
                    f.update_xaxes(title_text='Hours from rainfall-day start (Sydney)',row=2,col=col)
                f.update_xaxes(range=[-24*cfg['pre_days'],24*(cfg['post_days']+1)])
                key=f'{event_idx}|{region}|{quantity}';specs[key]=json.loads(f.to_json())
            if region=='All gullies (union)' and e.Index in [0,len(events[events.group=='past_year_daily'])]:examples[e.group]=(e,ev,d,np.asarray(rx),np.asarray(ry),missing)
    for group,(e,ev,d,rx,ry,missing) in examples.items():
        fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True,gridspec_kw={'height_ratios':[.55,1,1]})
        for ax in axes[0]:
            ax.bar(rx,ry,width=20 if group=='past_year_daily' else 1,color='#78a6bd');ax.set_ylabel('Rain (mm)')
            if missing.any():ax.scatter(rx[missing],np.zeros(missing.sum()),c='#c02738',marker='x',s=15)
        for row,quantity in enumerate(quantities,1):
            for name,color in COLORS.items():
                g=d[(d.scenario==name)&(d.gap_policy!='zero_missing')].sort_values('time');xx=(pd.DatetimeIndex(g.time)-ev).total_seconds()/3600;yy=g[quantity].to_numpy();pre=yy[(xx<0)&(xx>=-24)].mean()
                for col,y in [(0,yy),(1,yy-pre)]:axes[row,col].plot(xx,y,color=color,lw=2,ls={'baseline':'-','moderate_1m':'--','strong_4m':':'}[name],label=LABELS[name])
                z=d[(d.scenario==name)&(d.gap_policy=='zero_missing')].sort_values('time')
                if len(z):axes[row,0].fill_between(xx,np.minimum(yy,z[quantity]),np.maximum(yy,z[quantity]),color=color,alpha=.12)
            axes[row,0].set_ylabel(('SM-derived equivalent\n' if row==1 else 'Internal model bucket\n')+'water (m³)');axes[row,1].set_ylabel('Change (m³)')
        axes[0,0].set_title('Total water storage');axes[0,1].set_title('Increase above pre-event storage')
        axes[1,0].legend(fontsize=8,ncol=1)
        for ax in axes.ravel():ax.grid(alpha=.2);ax.axvline(0,color='#999',lw=.8);ax.set_xlim(-24*cfg['pre_days'],24*(cfg['post_days']+1))
        for ax in axes[-1]:ax.set_xlabel('Hours from rainfall-day start (Australia/Sydney)')
        area=float(areas.set_index('region').loc['All gullies (union)','valid_area_m2'])
        mode='Original daily model8' if group=='past_year_daily' else 'Experimental hourly timestep; gap treatments shaded'
        fig.suptitle(f'Combined gully water storage · {e.date:%d %b %Y}\n{area:,.0f} m² fixed footprint · assumed {depth:g} m soil layer · {mode}',fontsize=12)
        fig.text(.5,.015,'SM-derived volume is a depth-dependent equivalent. Internal bucket storage is unchanged by terrain. No surface water is modelled.',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.04,1,.94]);name='whole_gully_hourly' if group=='historical_hourly' else 'whole_gully_daily'
        fig.savefig(out/f'{name}.png',dpi=180);fig.savefig(out/f'{name}.pdf');plt.close(fig)
    opts=''.join(f'<option value="{i}">{e.date:%Y-%m-%d} · {e.group.replace("_"," ")}</option>' for i,e in enumerate(events.itertuples()))
    ropts=''.join(f'<option>{html.escape(r)}</option>' for r in regions)
    qopts=''.join(f'<option value="{k}">{html.escape(v)}</option>' for k,v in quantities.items())
    cmp=checks[(checks.series=='daily')&(checks.scenario!='baseline')][['region','scenario','mean_storage_difference_m3']].copy()
    cmp.mean_storage_difference_m3=cmp.mean_storage_difference_m3.round(2)
    body=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Whole-gully water storage</title>
<style>body{{font:16px/1.55 system-ui;color:#243142;background:#f6f8fa;margin:0}}main{{max-width:1300px;margin:auto;padding:26px}}.card{{background:white;border:1px solid #dce3e9;border-radius:10px;padding:20px;margin:18px 0}}select{{padding:9px;font:inherit;max-width:100%}}label{{display:inline-block;margin:5px 16px 5px 0}}table{{border-collapse:collapse;font-size:14px}}th,td{{padding:8px;border-bottom:1px solid #ddd}}a{{color:#2563a6}}img{{max-width:100%}}</style><main>
<h1>Water stored across the gullies</h1><p>Spatial totals over the same 5 m cells in every scenario. Choose an individual gully or the combined footprint.</p>
<div class="card"><strong>These are area totals at each time, not sensor-point predictions and not a running sum through time.</strong>
<p>SM-derived water volume = Σ(SM% / 100 × cell area × soil depth). Here each cell is 25 m² and soil depth is explicitly assumed to be {depth:g} m. This is a storage equivalent inferred from the model's moisture readout, not a validated inventory of water in the gully.</p>
<p>The internal bucket quantity is a separate conceptual reservoir: Σ(bucket storage in mm × cell area / 1000). It has no terrain-dependent dynamics, so its three scenario curves overlap exactly.</p></div>
<div class="card"><label>Rainfall event<br><select id="event">{opts}</select></label><label>Area<br><select id="region">{ropts}</select></label><label>Quantity<br><select id="quantity">{qopts}</select></label><div id="plot"></div>
<p>Faint hourly curves compare the zero-missing-rain assumption with the principal SILO-fraction gap treatment. They are not confidence bounds. An axis labelled in hours does not give daily SILO events hourly resolution.</p></div>
<div class="card"><h2>Area actually included</h2>{areas.to_html(index=False)}<p>Areas are horizontal planimetric areas. A common valid-cell mask is used for all scenarios. Overlap is counted once in the combined union; individual gully totals should not be added if their footprints overlap.</p></div>
<div class="card"><h2>Change in SM-derived storage relative to baseline</h2>{cmp.to_html(index=False)}<p>Differences are in m³ for the assumed {depth:g} m soil layer. They remain constant through time because terrain changes only the additive readout. Increasing incision does not change rainfall response or internal water storage in this version of model8.</p></div>
<div class="card"><h2>Interpretation</h2><ul><li>This measures soil-water equivalents over the provisional gully incision footprints. It does not include surface ponding, runoff moving through the channel, groundwater, or the whole upstream catchment.</li><li>The same soil-layer depth is placed below each scenario's ground surface. This does not account for soil physically removed by erosion or changing bedrock depth, soil properties, infiltration, or channel drainage.</li><li>Fine-grid terrain inputs can exceed the original model's training range; the per-variable area audit is exported as gully_storage_covariate_audit.csv. The volume calculation adds a depth assumption to an already exploratory prediction.</li><li>Phenode sensor readings are local measurements and are not observations of these area totals.</li><li>Past-year events use daily SILO forcing. Historical gauge events are an unvalidated hourly timestep adaptation. Missing gauge hours and the assumed logger timezone remain as documented in the original methodology.</li></ul></div>
<div class="card"><h2>Files</h2><p><a href="whole_gully_hourly.png">Hourly storage figure</a> · <a href="whole_gully_daily.png">Daily storage figure</a> · <a href="gully_storage_timeseries.csv">All storage time series (CSV)</a> · <a href="gully_storage_scenario_comparison.csv">Scenario differences</a> · <a href="index.html">Original sensor-point dashboard</a></p></div>
<p>Volume conversion follows the definition of volumetric soil moisture: <a href="https://www.fao.org/4/R4082E/r4082e03.htm">FAO soil and water</a>. Soil depth is an explicit assumption.</p></main><script>__PLOTLY__</script><script>const figures=__FIGURES__;function show(){{let key=document.getElementById('event').value+'|'+document.getElementById('region').value+'|'+document.getElementById('quantity').value;let f=figures[key];Plotly.react('plot',f.data,f.layout,{{responsive:true,displaylogo:false}});}}for(const id of ['event','region','quantity'])document.getElementById(id).addEventListener('change',show);show();</script></html>'''
    (out/'whole_gully_storage.html').write_text(body.replace('__PLOTLY__',get_plotlyjs()).replace('__FIGURES__',json.dumps(specs)))
