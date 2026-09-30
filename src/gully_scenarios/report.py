"""Standalone event dashboard and exportable scientific figures."""
import json
import html
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import rasterio
from pyproj import Transformer
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS={'baseline':'#2563a6','moderate_1m':'#e48b21','strong_4m':'#b32342'}
LABELS={'baseline':'Original gully','moderate_1m':'Additional 1 m incision','strong_4m':'Additional 4 m incision'}


def event_data(e,cfg,daily,hourly,obs,silo,rain):
    ev=e.date.tz_localize(cfg['display_timezone'])
    lo=ev-pd.DateOffset(days=cfg['pre_days']);hi=ev+pd.DateOffset(days=cfg['post_days']+1)
    if e.group=='past_year_daily':
        d=daily[(daily.date>=lo.tz_localize(None))&(daily.date<hi.tz_localize(None))].copy()
        d['time']=d.date.dt.tz_localize(cfg['display_timezone']);d['gap_policy']='daily'
        rr=silo.loc[lo.tz_localize(None):hi.tz_localize(None)-pd.Timedelta(days=1),['rain_mm']].copy()
        rr.index=rr.index.tz_localize(cfg['display_timezone']);rr['observed']=True
    else:
        d=hourly[hourly.event_id==e.event_id].copy()
        rr=rain[(rain.index>=lo)&(rain.index<hi)].copy()
    o=obs[(obs.time>=lo)&(obs.time<hi)].copy()
    return ev,d,rr,o


def event_figure(e,cfg,points,daily,hourly,obs,silo,rain):
    ev,d,rr,o=event_data(e,cfg,daily,hourly,obs,silo,rain)
    gullies=points[points.is_gully]
    specs=[[{'colspan':2},None]]+[[{},{}] for _ in range(len(gullies))]
    titles=['Rainfall — '+e.rain_source]
    for p in gullies.itertuples():titles.extend([p.label+' · soil moisture',p.label+' · rise above pre-event level'])
    fig=make_subplots(rows=len(gullies)+1,cols=2,specs=specs,shared_xaxes=True,
                      row_heights=[.16]+[.28]*len(gullies),vertical_spacing=.07,subplot_titles=titles)
    def xs(index):return (pd.DatetimeIndex(index)-ev).total_seconds()/3600
    fig.add_trace(go.Bar(x=xs(rr.index),y=rr.rain_mm,name='Rainfall',marker_color='#78a6bd',width=20 if e.group=='past_year_daily' else .85),row=1,col=1)
    if (~rr.observed).any():
        fig.add_trace(go.Scatter(x=xs(rr.index[~rr.observed]),y=np.zeros((~rr.observed).sum()),mode='markers',marker=dict(symbol='x',color='#c02738',size=7),name='Gauge hour missing'),row=1,col=1)
    for row,p in enumerate(gullies.itertuples(),start=2):
        for name,color in COLORS.items():
            g=d[(d.device==p.device)&(d.scenario==name)&(d.gap_policy!='zero_missing')].sort_values('time')
            x=xs(g.time);v=g.sm_pct.to_numpy();pre=np.nanmean(v[(x<0)&(x>=-24)])
            for col,y in [(1,v),(2,v-pre)]:
                fig.add_trace(go.Scatter(x=x,y=y,name=LABELS[name],legendgroup=name,showlegend=row==2 and col==1,mode='lines',line=dict(color=color,width=2.2,dash={'baseline':'solid','moderate_1m':'dash','strong_4m':'dot'}[name])),row=row,col=col)
            if e.group=='historical_hourly':
                low=d[(d.device==p.device)&(d.scenario==name)&(d.gap_policy=='zero_missing')].sort_values('time')
                fig.add_trace(go.Scatter(x=xs(low.time),y=low.sm_pct,mode='lines',line=dict(color=color,width=1),opacity=.35,name=LABELS[name]+' · zero missing rain',showlegend=False,legendgroup=name),row=row,col=1)
        z=o[o.device==p.device]
        fig.add_trace(go.Scatter(x=xs(z.time),y=z.sm_pct,mode='markers',marker=dict(color='#24282d',size=6),name='Phenode observations (daily)',showlegend=row==2,legendgroup='obs'),row=row,col=1)
        fig.update_yaxes(title_text='VWC (%)',row=row,col=1)
        fig.update_yaxes(title_text='Change (percentage points)',row=row,col=2)
    title=f'{e.date:%d %b %Y} · {e.rain_mm:.1f} mm'
    subtitle='Original daily model8 · daily SILO rainfall' if e.group=='past_year_daily' else 'Experimental hourly timestep · recorded gauge hours + explicit gap assumptions'
    fig.update_layout(title=dict(text=title+'<br><sup>'+subtitle+'</sup>'),height=1050,template='plotly_white',margin=dict(t=110,b=65,l=70,r=35),hovermode='x unified',legend=dict(orientation='h',y=1.06,x=0),font=dict(family='Arial',size=12))
    fig.update_xaxes(range=[-24*cfg['pre_days'],24*(cfg['post_days']+1)])
    for col in [1,2]:fig.update_xaxes(title_text='Hours from rainfall-day start (Sydney time)',row=len(gullies)+1,col=col)
    fig.update_yaxes(title_text='Rain (mm)',row=1,col=1)
    return fig


def static_event(e,cfg,points,daily,hourly,obs,silo,rain,out):
    ev,d,rr,o=event_data(e,cfg,daily,hourly,obs,silo,rain)
    gullies=points[points.is_gully]
    fig,axes=plt.subplots(len(gullies)+1,2,figsize=(13,10),sharex=True,gridspec_kw={'height_ratios':[.7]+[1]*len(gullies)})
    def xs(t):return (pd.DatetimeIndex(t)-ev).total_seconds()/3600
    for ax in axes[0]:
        ax.bar(xs(rr.index),rr.rain_mm,width=20 if e.group=='past_year_daily' else 1,color='#78a6bd');ax.set_ylabel('Rain (mm)')
        if (~rr.observed).any():ax.scatter(xs(rr.index[~rr.observed]),np.zeros((~rr.observed).sum()),marker='x',s=16,c='#c02738')
    for row,p in enumerate(gullies.itertuples(),start=1):
        for name,color in COLORS.items():
            g=d[(d.device==p.device)&(d.scenario==name)&(d.gap_policy!='zero_missing')].sort_values('time')
            x=xs(g.time);v=g.sm_pct.to_numpy();base=np.nanmean(v[(x<0)&(x>=-24)])
            for col,y in [(0,v),(1,v-base)]:axes[row,col].plot(x,y,color=color,label=LABELS[name],linestyle={'baseline':'-','moderate_1m':'--','strong_4m':':'}[name],lw=2)
            if e.group=='historical_hourly':
                z=d[(d.device==p.device)&(d.scenario==name)&(d.gap_policy=='zero_missing')].sort_values('time')
                axes[row,0].fill_between(x,np.minimum(v,z.sm_pct),np.maximum(v,z.sm_pct),color=color,alpha=.1)
        z=o[o.device==p.device];axes[row,0].scatter(xs(z.time),z.sm_pct,s=13,color='black',label='Phenode (daily)',zorder=5)
        axes[row,0].set_ylabel(p.label+'\nVWC (%)');axes[row,1].set_ylabel('Change (pp)')
    axes[1,0].legend(fontsize=8,ncol=2)
    axes[0,0].set_title('Absolute soil moisture');axes[0,1].set_title('Response above pre-event level')
    for ax in axes.ravel():
        ax.axvline(0,color='#999',lw=.8);ax.grid(alpha=.18);ax.set_xlim(-24*cfg['pre_days'],24*(cfg['post_days']+1))
    for ax in axes[-1]:ax.set_xlabel('Hours from rainfall-day start (Australia/Sydney)')
    mode='Original daily model8' if e.group=='past_year_daily' else 'Experimental hourly model8; shaded ranges compare missing-rain assumptions'
    fig.suptitle(f'Phenode gully scenarios · {e.date:%d %b %Y} · {e.rain_mm:.1f} mm\n{mode}',fontsize=13)
    fig.text(.5,.012,'1 m / 4 m are maximum centreline cuts. Terrain changes moisture offsets; event-response curves overlap.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,.94]);fig.savefig(out,dpi=180);fig.savefig(out.with_suffix('.pdf'));plt.close(fig)


def geometry_figure(root,points):
    with rasterio.open(root/'outputs/terrain/baseline_dem_5m.tif') as src:
        dem=src.read(1,masked=True);extent=[src.bounds.left,src.bounds.right,src.bounds.bottom,src.bounds.top]
    with rasterio.open(root/'outputs/terrain/incision_weight.tif') as src:weight=src.read(1)
    fig,axes=plt.subplots(1,2,figsize=(13,6))
    hill=LightSource(315,40).hillshade(dem.filled(np.nan),vert_exag=1,dx=5,dy=5)
    for ax in axes:
        ax.imshow(hill,extent=extent,cmap='gray',origin='upper');ax.set_xlim(675500,677400);ax.set_ylim(6113650,6115450);ax.set_aspect('equal')
        for p in points.itertuples():
            x,y=Transformer.from_crs(4326,28355,always_xy=True).transform(p.lon,p.lat)
            ax.scatter(x,y,c='#b32342' if p.is_gully else '#2563a6',s=28)
            ax.annotate(p.label,(x,y),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.ticklabel_format(style='plain',useOffset=False);ax.set_xlabel('MGA55 easting (m)')
    masked=np.ma.masked_where(weight==0,weight*4)
    im=axes[1].imshow(masked,extent=extent,origin='upper',cmap='magma_r',vmin=0,vmax=4)
    axes[0].set_title('5 m DEM from classified ground LiDAR');axes[1].set_title('Provisional gully footprint · strong incision')
    axes[0].set_ylabel('MGA55 northing (m)');fig.colorbar(im,ax=axes[1],label='Additional incision (m)',shrink=.8)
    fig.tight_layout();fig.savefig(root/'outputs/terrain_scenarios.png',dpi=180);plt.close(fig)


def make_report(root,cfg,points,daily,hourly,events,obs,silo,rain,metadata):
    import plotly.io as pio
    from plotly.offline import get_plotlyjs
    plot_json=[]
    for e in events.itertuples():
        plot_json.append(json.loads(pio.to_json(event_figure(e,cfg,points,daily,hourly,obs,silo,rain))))
    for group,filename in [('past_year_daily','past_year_largest_event.png'),('historical_hourly','hourly_largest_event.png')]:
        e=next(events[events.group==group].itertuples())
        static_event(e,cfg,points,daily,hourly,obs,silo,rain,root/'outputs'/filename)
    geometry_figure(root,points)
    options=''.join(f'<option value="{i}">{html.escape(e.group.replace("_"," "))} · {e.date:%Y-%m-%d} · {e.rain_mm:.1f} mm</option>' for i,e in enumerate(events.itertuples()))
    offsets=pd.read_csv(root/'outputs/scenario_invariance.csv').groupby(['device','scenario']).offset_pp.mean().unstack().join(points.set_index('device')[['label']]).set_index('label')
    table=offsets[['moderate_1m','strong_4m']].round(4).rename(columns={'moderate_1m':'1 m: SM offset (pp)','strong_4m':'4 m: SM offset (pp)'}).to_html()
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phenode gully scenarios</title>
<style>body{font:16px/1.55 system-ui,sans-serif;color:#243142;background:#f6f8fa;margin:0}main{max-width:1300px;margin:0 auto;padding:28px}h1{font-size:32px;margin-bottom:8px}.card{background:white;border:1px solid #dce3e9;border-radius:10px;padding:20px;margin:20px 0}.notice{border-left:5px solid #c48731}select{font:inherit;padding:10px;max-width:100%}table{border-collapse:collapse;font-size:14px}th,td{padding:8px 14px;border-bottom:1px solid #ddd}img{max-width:100%}a{color:#2563a6}</style><main>
<h1>Phenode gully incision scenarios</h1><p><a href="capacity_depth_dashboard.html"><strong>Capacity-depth × DEM experiment: whole-gully dynamics</strong></a></p><p><a href="whole_gully_storage.html"><strong>View whole-gully water-storage totals</strong></a> · This page shows the original sensor-pixel curves.</p><p>Original terrain, additional 1 m incision and additional 4 m incision. Predictions at the original sensor positions on a 5 m LiDAR grid.</p>
<div class="card notice"><strong>Result: incision changes the moisture level, not event peakiness.</strong> Model8 applies terrain as a static additive offset. Its pre-event-adjusted curves overlap for every scenario; increasing incision cannot change that model structure.</div>
<div class="card"><label for="event"><strong>Select a rainfall event</strong></label><br><select id="event">OPTIONS</select><div id="plot"></div>
<p>Solid/dashed/dotted curves distinguish scenarios. Black points are the supplied daily Phenode observations. The right-hand panels remove the pre-event offset. Zoom and hover to inspect values.</p></div>
<div class="card"><strong>Time resolution and rainfall coverage</strong><ul>
<li>The requested past year is 29 September 2025–28 September 2026. Its events use SILO daily rainfall and the original daily model8 recurrence. An axis in hours does not imply hourly information.</li>
<li>The hourly gauge file only covers March–August 2025. Historical event selection is restricted to windows with valid gully soil-moisture observations (available from 10 June 2025). Those historical events are a separate experimental hourly adaptation using unchanged fitted parameters and drainage fraction 1−(1−k)<sup>1/24</sup>. Daily PET is distributed uniformly over the hours of each local day.</li>
<li>Every midnight hour is absent in the supplied gauge timestamps. Red crosses identify missing hours. The principal hourly curves substitute SILO daily rainfall divided by the hours in that day at missing hours. Faint curves use zero for missing hours; these are alternative assumptions, not uncertainty bounds.</li>
<li>Gauge event totals are observed subtotals and may underestimate actual rain. Gauge times are assumed UTC from the original R script; plots use Australia/Sydney. No subdaily soil-moisture observations were manufactured.</li>
</ul></div>
<div class="card"><strong>Scenario offsets</strong><p>Percentage points relative to the baseline. These offsets are constant through time. They are model sensitivity results, not estimated real erosion effects.</p>TABLE</div>
<div class="card"><strong>Terrain and sampling</strong><img src="terrain_scenarios.png" alt="LiDAR terrain and incision footprints"><p>The LAZ contains classified ground points, gridded by mean elevation into aligned 5 m cells. Its CRS is absent; MGA94 zone 55 is assigned using neighbouring DEM and Brindabella processing records. Gully centrelines are provisional drainage paths near G-labelled probes, with a 30 m full-width cosine incision and tapered ends. AEG? retains its uncertain label. Sensor locations remain fixed.</p>
<p>The 1 m and 4 m depths apply at centrelines; off-centre sensor cuts can be smaller. Soil properties and depth remain fixed, even under 4 m incision. Six of seven baseline sensor slopes exceed the maximum training slope (14.73°); these fine-resolution predictions are extrapolations. Terrain TWI retains the original cell-count convention, so transfer from the 30 m training scale remains unvalidated. Soil moisture is the model's root-zone readout; the supplied Phenode sensing depth is not documented here.</p></div>
<div class="card"><strong>Files</strong><ul><li><a href="past_year_largest_event.png">Largest past-year event: PNG</a> · <a href="past_year_largest_event.pdf">PDF</a></li><li><a href="hourly_largest_event.png">Largest historical gauge event: PNG</a> · <a href="hourly_largest_event.pdf">PDF</a></li><li><a href="daily_predictions.csv">Daily predictions</a> · <a href="hourly_experimental_predictions.csv">Experimental hourly predictions</a></li><li><a href="event_metrics.csv">Event metrics</a> · <a href="selected_events.csv">Selected rainfall events</a> · <a href="rainfall_past_year_ranked.csv">All past-year rainfall days</a></li><li><a href="terrain_points.csv">Terrain at sensors</a> · <a href="gully_geometry_qc.csv">Geometry checks</a> · <a href="covariate_training_audit.csv">Training-range checks</a> · <a href="all_sensors_daily.html">All seven sensors: daily time series</a></li></ul></div>
<p>Weather: <a href="https://www.longpaddock.qld.gov.au/silo/">SILO</a>. Elevation source: local Brindabella LiDAR and processing records; discovery portal <a href="https://elevation.fsdf.org.au/">ELVIS</a>. No recalibration or lateral routing was introduced.</p></main>
<script>PLOTLY_JS</script><script>const figures=FIGURES;function show(i){const f=figures[i];Plotly.react('plot',f.data,f.layout,{responsive:true,displaylogo:false});}document.getElementById('event').addEventListener('change',e=>show(Number(e.target.value)));show(0);</script></html>'''
    page=page.replace('OPTIONS',options).replace('TABLE',table).replace('PLOTLY_JS',get_plotlyjs()).replace('FIGURES',json.dumps(plot_json))
    (root/'outputs/index.html').write_text(page)
    # Full-period observations and all seven original sensor locations.
    full=make_subplots(rows=len(points),cols=1,shared_xaxes=True,subplot_titles=points.label.tolist(),vertical_spacing=.025)
    for r,p in enumerate(points.itertuples(),1):
        for name,color in COLORS.items():
            g=daily[(daily.device==p.device)&(daily.scenario==name)]
            full.add_trace(go.Scatter(x=g.date,y=g.sm_pct,name=LABELS[name],legendgroup=name,showlegend=r==1,line=dict(color=color,dash={'baseline':'solid','moderate_1m':'dash','strong_4m':'dot'}[name])),row=r,col=1)
        o=obs[obs.device==p.device]
        full.add_trace(go.Scatter(x=o.time.dt.tz_convert(cfg['display_timezone']).dt.tz_localize(None),y=o.sm_pct,mode='markers',marker=dict(size=3,color='black'),name='Phenode daily',showlegend=r==1),row=r,col=1)
        full.update_yaxes(title_text='VWC %',row=r,col=1)
    full.update_layout(height=1700,title='All Phenode sensors · original daily model8 on 5 m terrain',template='plotly_white')
    full.write_html(root/'outputs/all_sensors_daily.html',include_plotlyjs=True)
