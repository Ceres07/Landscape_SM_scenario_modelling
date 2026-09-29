"""5 m terrain covariates and reproducible, provisional gully centreline incision."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.fill import fillnodata
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation
from pyproj import Transformer
from .core import incision_weight

D8 = {64:(-1,0),128:(-1,1),1:(0,1),2:(1,1),4:(1,0),8:(1,-1),16:(0,-1),32:(-1,-1)}


def write_grid(path, data, profile):
    p=profile.copy();p.update(count=1,dtype='float32',nodata=-9999,compress='deflate',tiled=True)
    with rasterio.open(path,'w',**p) as out:
        out.write(np.where(np.isfinite(data),data,-9999).astype('float32'),1)


def terrain(path):
    if not hasattr(np,'in1d'): np.in1d=np.isin
    from pysheds.grid import Grid
    with rasterio.open(path) as src:
        dem=src.read(1,masked=True).filled(np.nan).astype(float)
        profile=src.profile
    grid=Grid.from_raster(str(path));r=grid.read_raster(str(path))
    filled=grid.fill_depressions(grid.fill_pits(r));conditioned=grid.resolve_flats(filled)
    fdir=grid.flowdir(conditioned);acc=grid.accumulation(fdir)
    dy,dx=np.gradient(dem,abs(profile['transform'].e),profile['transform'].a)
    slope=np.degrees(np.arctan(np.hypot(dx,dy)))
    # Preserve the training code's cell-count TWI convention. Do not silently
    # switch to physical contributing area without refitting its coefficients.
    with np.errstate(divide='ignore',invalid='ignore'):
        ratio=np.asarray(acc)/np.tan(np.radians(slope))
        twi=np.log(np.where(ratio<=0,1,ratio))
    valid=binary_erosion(np.isfinite(dem),iterations=2)
    twi[~valid | ~np.isfinite(twi)]=np.nan
    return dict(elevation=dem,slope=slope,twi=twi,accumulation=np.asarray(acc)), grid, fdir, np.asarray(filled)-dem


def trace_gully(seed,fdir,acc,up_m,down_m,res):
    h,w=fdir.shape
    def inside(p):return 0<=p[0]<h and 0<=p[1]<w
    def nextcell(p):
        d=D8.get(int(fdir[p]));return (p[0]+d[0],p[1]+d[1]) if d else None
    up=[seed];length=0
    while length<up_m:
        candidates=[]
        for dy,dx in D8.values():
            p=(up[-1][0]+dy,up[-1][1]+dx)
            if inside(p) and nextcell(p)==up[-1] and p not in up:
                candidates.append(p)
        if not candidates:break
        best=max(candidates,key=lambda p:acc[p])
        length+=res*np.hypot(best[0]-up[-1][0],best[1]-up[-1][1]);up.append(best)
    down=[seed];length=0
    while length<down_m:
        nxt=nextcell(down[-1])
        if nxt is None or not inside(nxt) or nxt in down:break
        length+=res*np.hypot(nxt[0]-down[-1][0],nxt[1]-down[-1][1]);down.append(nxt)
    return list(reversed(up[1:]))+down


def build_scenarios(root,cfg,points):
    out=root/'outputs/terrain';out.mkdir(parents=True,exist_ok=True)
    with rasterio.open(root/'data/raw/ground_5m.tif') as src:
        dem=src.read(1,masked=True).filled(np.nan);profile=src.profile
    original_valid=np.isfinite(dem)
    # Fill only within 10 m of observed cells; larger uncovered areas stay nodata.
    interp=fillnodata(np.where(original_valid,dem,-9999).astype('float32'),mask=original_valid.astype('uint8'),max_search_distance=2,smoothing_iterations=0)
    dem=np.where(interp==-9999,np.nan,interp)
    base=out/'baseline_dem_5m.tif';write_grid(base,dem,profile)
    print('Deriving baseline terrain...',flush=True)
    base_cov,grid,fdir,fill_delta=terrain(base)
    res=cfg['resolution_m'];tx=profile['transform']
    transformer=Transformer.from_crs(4326,profile['crs'],always_xy=True)
    xy=[transformer.transform(r.lon,r.lat) for r in points.itertuples()]
    rc=[rasterio.transform.rowcol(tx,x,y) for x,y in xy]
    points=points.copy();points['row']=[p[0] for p in rc];points['col']=[p[1] for p in rc]
    weights=np.zeros(dem.shape,dtype=float);features=[];point_qc=[]
    for p in points.itertuples():
        if not p.is_gully:continue
        rr,cc=np.indices(dem.shape);dist=np.hypot(rr-p.row,cc-p.col)*res
        nearby=(dist<=cfg['snap_radius_m']) & np.isfinite(dem)
        # Accumulation preference places the centreline on an existing drainage
        # route; retain original sensor cell for all moisture predictions.
        score=np.where(nearby,base_cov['accumulation'],-np.inf)
        seed=tuple(np.unravel_index(np.argmax(score),score.shape))
        path=trace_gully(seed,np.asarray(fdir),base_cov['accumulation'],cfg['gully_upstream_length_m'],cfg['gully_downstream_length_m'],res)
        line=np.zeros(dem.shape,bool);longitudinal=np.zeros(dem.shape)
        cumulative=np.r_[0,np.cumsum([res*np.hypot(a[0]-b[0],a[1]-b[1]) for a,b in zip(path[:-1],path[1:])])]
        taper_len=min(50,cumulative[-1]/3)
        for cell,s in zip(path,cumulative):
            line[cell]=True
            longitudinal[cell]=min(1.,s/taper_len,(cumulative[-1]-s)/taper_len)
        distance,nearest=distance_transform_edt(~line,sampling=res,return_indices=True)
        weight=incision_weight(distance,cfg['gully_half_width_m'])*longitudinal[tuple(nearest)]
        weights=np.maximum(weights,weight)
        coords=[list(rasterio.transform.xy(tx,*cell)) for cell in path]
        features.append(dict(type='Feature',properties=dict(label=p.label,provisional=True,snap_distance_m=float(dist[seed])),geometry=dict(type='LineString',coordinates=coords)))
        catch=grid.catchment(x=seed[1],y=seed[0],fdir=fdir,xytype='index')
        c=np.asarray(catch,dtype=bool)
        edge=bool(c[0,:].any() or c[-1,:].any() or c[:,0].any() or c[:,-1].any())
        point_qc.append(dict(label=p.label,snap_distance_m=float(dist[seed]),sensor_weight=float(weight[p.row,p.col]),catchment_touches_dem_edge=edge,catchment_touches_nodata=bool(np.any(c & binary_dilation(~np.isfinite(dem)))),catchment_area_m2=int(c.sum())*res**2))
    if not np.any(weights):raise ValueError('No incision footprint created')
    write_grid(out/'incision_weight.tif',weights,profile)
    (out/'gully_centrelines.geojson').write_text(json.dumps(dict(type='FeatureCollection',crs=dict(type='name',properties=dict(name=str(profile['crs']))),features=features),indent=2))
    rows=[];quality=[]
    for name,depth in cfg['incisions_m'].items():
        z=dem-depth*weights;path=out/f'{name}_dem_5m.tif';write_grid(path,z,profile)
        if depth==0:cov,fd,fill=base_cov,fdir,fill_delta
        else:
            print('Deriving terrain',name,flush=True)
            cov,_,fd,fill=terrain(path)
        for var,arr in cov.items():write_grid(out/f'{name}_{var}.tif',arr,profile)
        write_grid(out/f'{name}_conditioning_raise_m.tif',fill,profile)
        footprint=weights>0
        quality.append(dict(scenario=name,max_incision_m=float(np.nanmax(dem-z)),mean_incision_m=float(np.nanmean((dem-z)[footprint])),conditioning_raise_max_m=float(np.nanmax(fill[footprint])),conditioning_raise_mean_m=float(np.nanmean(fill[footprint]))))
        for p in points.itertuples():
            row=dict(device=p.device,label=p.label,is_gully=p.is_gully,scenario=name,incision_at_sensor_m=float(depth*weights[p.row,p.col]))
            row.update({v:float(a[p.row,p.col]) for v,a in cov.items()});rows.append(row)
    pd.DataFrame(point_qc).to_csv(root/'outputs/gully_geometry_qc.csv',index=False)
    pd.DataFrame(quality).to_csv(root/'outputs/dem_scenario_qc.csv',index=False)
    pd.DataFrame(rows).to_csv(root/'outputs/terrain_points.csv',index=False)
    points.to_csv(root/'outputs/points.csv',index=False)
    return pd.DataFrame(rows),points
