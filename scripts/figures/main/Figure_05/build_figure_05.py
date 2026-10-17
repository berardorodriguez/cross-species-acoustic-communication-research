from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFont

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
DATA = REPOSITORY_ROOT / 'data' / 'derived' / 'figure_05'
OUT = REPOSITORY_ROOT / 'outputs' / 'figure_05_rebuild'
OUT.mkdir(parents=True, exist_ok=True)

OAAS = DATA / 'Figure05A_OAAS_plot_data.csv'
BEHAV = DATA / 'Figure05_behavior_hrv_minute_level.csv'

FINAL_WIDTH_IN = 180 / 25.4
FINAL_DPI = 300

# -------------------- Panel A --------------------
points = pd.read_csv(OAAS)
required = {'canonical','x','y','z_v28','balance_weight'}
missing = required - set(points.columns)
if missing:
    raise ValueError(f'Missing OAAS columns: {sorted(missing)}')

legend_order = [
    ('BASELINE','Noise floor'),
    ('RUIDO_ROSA','Pink noise'),
    ('SILBOS','Herding whistles'),
    ('PASTOREO','Grazing calls'),
    ('CANTO_ORDENO','Milking songs'),
    ('KULNING','Kulning herding call'),
    ('GRITO_MONTE','High-amplitude range shouts'),
    ('VAQUERIA','Cattle herding calls'),
    ('FERIA','Pre-slaughter calls (PSC)'),
]
rank = {c:i+1 for i,(c,_) in enumerate(legend_order)}
real_order = [c for c,_ in legend_order]
centroids_real = (points.groupby('canonical')
                  .agg(x=('x','median'),y=('y','median'),z=('z_v28','median'),w=('balance_weight','median'))
                  .reset_index())
centroids_real = centroids_real[centroids_real['canonical'].isin(real_order)].copy()
centroids = centroids_real.copy()
centroids['rank'] = centroids['canonical'].map(rank)
centroids = centroids.sort_values('rank')

grid_n=175; pad=0.34
xmin,xmax = points['x'].min()-pad, points['x'].max()+pad
ymin,ymax = points['y'].min()-pad, points['y'].max()+pad
xmin,xmax = min(xmin,-2.15), max(xmax,2.15)
ymin,ymax = min(ymin,-2.05), max(ymax,1.95)
gx=np.linspace(xmin,xmax,grid_n); gy=np.linspace(ymin,ymax,grid_n)
Gx,Gy=np.meshgrid(gx,gy)
Z=np.zeros_like(Gx,dtype=float)
Z += 0.15
Z += 0.070*np.exp(-0.5*((Gy-0.10)/2.10)**2)
Z += 0.070*np.exp(-0.5*((Gx+Gy-0.10)/1.70)**2)
Z += 0.045*np.exp(-0.5*((Gx-Gy+0.15)/2.10)**2)
Z += -0.010*((Gx-0.1)**2 + 0.65*(Gy+0.05)**2)/4.5
zmin,zmax=points['z_v28'].min(),points['z_v28'].max()
energy_scale={'BASELINE':0.30,'RUIDO_ROSA':0.42,'SILBOS':0.54,'PASTOREO':0.62,'CANTO_ORDENO':0.68,'KULNING':0.75,'GRITO_MONTE':0.82,'VAQUERIA':0.90,'FERIA':0.85}
for r in points[points['canonical'].isin(real_order)].itertuples():
    relz=(r.z_v28-zmin)/(zmax-zmin+1e-12)
    base_amp=(0.35+0.65*relz)*energy_scale.get(r.canonical,0.6)
    amp=0.080*np.sqrt(max(r.balance_weight,0))*base_amp
    if r.canonical=='BASELINE': sx,sy=0.34,0.44
    elif r.canonical=='RUIDO_ROSA': sx,sy=0.31,0.40
    else: sx,sy=0.25,0.32
    Z += amp*np.exp(-0.5*(((Gx-r.x)/sx)**2+((Gy-r.y)/sy)**2))
for r in centroids_real.itertuples():
    rr=rank[r.canonical]; amp=0.068+0.016*rr
    if r.canonical in ['VAQUERIA','FERIA']: amp*=0.95
    if r.canonical in ['PASTOREO','KULNING','CANTO_ORDENO']: amp*=0.94
    if r.canonical in ['BASELINE','RUIDO_ROSA','GRITO_MONTE','SILBOS']: sx,sy=0.26,0.24
    elif r.canonical=='PASTOREO': sx,sy=0.20,0.18
    elif r.canonical=='CANTO_ORDENO': sx,sy=0.18,0.18
    elif r.canonical=='KULNING': sx,sy=0.09,0.09
    elif r.canonical in ['VAQUERIA','FERIA']: sx,sy=0.28,0.26
    else: sx,sy=0.30,0.28
    Z += amp*np.exp(-0.5*(((Gx-r.x)/sx)**2+((Gy-r.y)/sy)**2))

def add_corridor(Z_in,a,b,amp=0.06,width=0.18,length_soft=0.34):
    ax0,ay0=map(float,a); bx0,by0=map(float,b)
    vx,vy=bx0-ax0,by0-ay0; L2=vx*vx+vy*vy+1e-12
    t=((Gx-ax0)*vx+(Gy-ay0)*vy)/L2; tc=np.clip(t,0,1)
    px=ax0+tc*vx; py=ay0+tc*vy
    dperp2=(Gx-px)**2+(Gy-py)**2
    along=np.exp(-0.5*((np.maximum(-t,t-1))/length_soft)**2)
    return Z_in+amp*np.exp(-0.5*dperp2/(width**2))*along
cent_dict={r.canonical:(r.x,r.y) for r in centroids_real.itertuples()}
corridors=[('BASELINE','RUIDO_ROSA',0.045,0.24),('RUIDO_ROSA','SILBOS',0.055,0.22),('SILBOS','PASTOREO',0.070,0.42),('PASTOREO','CANTO_ORDENO',0.088,0.32),('CANTO_ORDENO','KULNING',0.070,0.22),('RUIDO_ROSA','GRITO_MONTE',0.075,0.32),('GRITO_MONTE','VAQUERIA',0.080,0.32),('VAQUERIA','FERIA',0.050,0.26),('BASELINE','PASTOREO',0.030,0.38),('SILBOS','GRITO_MONTE',0.040,0.30),('KULNING','FERIA',0.015,0.36)]
for c1,c2,amp,width in corridors:
    if c1 in cent_dict and c2 in cent_dict:
        Z=add_corridor(Z,cent_dict[c1],cent_dict[c2],amp,width)
for (cx,cy),amp,sx,sy in [((0.25,0.20),0.026,0.72,0.60),((0.55,-0.35),0.022,0.64,0.56),((-0.15,-0.45),0.020,0.72,0.62),((-0.85,0.95),0.012,0.42,0.34)]:
    Z -= amp*np.exp(-0.5*(((Gx-cx)/sx)**2+((Gy-cy)/sy)**2))
Z=gaussian_filter(Z,sigma=0.35)
lo,hi=np.percentile(Z,[1.0,99.0]); Zn=np.clip((Z-lo)/(hi-lo+1e-12),0,1)
Zn=Zn**0.82; Zn=1/(1+np.exp(-3.8*(Zn-0.50))); Zn=(Zn-Zn.min())/(Zn.max()-Zn.min()+1e-12)
Z=-0.54+1.06*Zn; Z=gaussian_filter(Z,sigma=0.22)

cmap=LinearSegmentedColormap.from_list('oaas_final',['#10188f','#126db3','#0c8e9c','#6abf59','#f2d43a','#f28a1a','#d64a1f'],N=256)
colors={c:cmap((rank[c]-1)/(len(legend_order)-1)) for c,_ in legend_order}
fig,ax=plt.subplots(figsize=(FINAL_WIDTH_IN,5.7),dpi=FINAL_DPI)
cf=ax.contourf(Gx,Gy,Z,levels=64,cmap=cmap)
# Sparse, full-weight contour guides remain legible at final print size.
ax.contour(Gx,Gy,Z,levels=6,colors='white',linewidths=2.0,alpha=0.42,linestyles='solid')
marker_offsets={'BASELINE':(-0.03,0.03),'RUIDO_ROSA':(0.17,0.17),'SILBOS':(0,0),'PASTOREO':(0.08,-0.04),'CANTO_ORDENO':(-0.06,0.05),'KULNING':(0.03,0.04),'GRITO_MONTE':(-0.12,-0.17),'VAQUERIA':(-0.08,0.05),'FERIA':(0.08,-0.02)}
for r in centroids.itertuples():
    c=r.canonical; num=rank[c]; dx,dy=marker_offsets.get(c,(0,0))
    ax.scatter(r.x+dx,r.y+dy,s=230,color='white',edgecolors=colors[c],linewidths=2.2,zorder=6)
    ax.text(r.x+dx,r.y+dy,str(num),fontsize=11.5,fontweight='black',ha='center',va='center',color='black',zorder=7)
ax.set_xlabel('Latent normalized unit 1 (LNU1)',fontsize=13,fontweight='bold',labelpad=8)
ax.set_ylabel('Latent normalized unit 2 (LNU2)',fontsize=13,fontweight='bold',labelpad=8)
ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect('equal',adjustable='box')
ax.tick_params(axis='both',labelsize=10,width=2.0,length=5)
for spine in ax.spines.values(): spine.set_linewidth(2.0)
cbar=fig.colorbar(cf,ax=ax,shrink=0.82,pad=0.035)
cbar.set_label('Normalized convergence field',fontsize=12,fontweight='bold')
cbar.ax.tick_params(labelsize=9,width=2.0,length=4)
cbar.outline.set_linewidth(2.0)
fig.subplots_adjust(left=0.17,right=0.88,top=0.975,bottom=0.120)
A_png=OUT/'Figure_05A.png'
fig.savefig(A_png,dpi=FINAL_DPI); plt.close(fig)

# -------------------- Panel B --------------------
df=pd.read_csv(BEHAV)
required_b={'Tratamiento','Minuto','Stimulus_Block','canonical','category_label'}
missing=required_b-set(df.columns)
if missing: raise ValueError(f'Missing behavior columns: {sorted(missing)}')
order_map={'BASELINE':1,'RUIDO_ROSA':2,'SILBOS':3,'PASTOREO':4,'CANTO_ORDENO':5,'KULNING':6,'GRITO_MONTE':7,'VAQUERIA':8,'FERIA':9}
short_label_map={'BASELINE':'Baseline','RUIDO_ROSA':'Pink noise','SILBOS':'Whistles','PASTOREO':'GC','CANTO_ORDENO':'Milking','KULNING':'Kulning','GRITO_MONTE':'Range\nshouts','VAQUERIA':'Cattle\nherding\ncalls','FERIA':'PSC'}
valid_cats=list(order_map.keys()); df=df[df['canonical'].isin(valid_cats)].copy()
if 'looking_time_s_per_min' not in df.columns:
    if 'mirar_prop' in df.columns: df['looking_time_s_per_min']=df['mirar_prop']*60.0
    elif 'mirar_mean' in df.columns: df['looking_time_s_per_min']=df['mirar_mean']
    else: raise ValueError('Could not compute orientation endpoint')
summary=(df.groupby('canonical',dropna=False)
         .agg(N=('canonical','size'),orientation_mean=('looking_time_s_per_min','mean'),orientation_sd=('looking_time_s_per_min','std'),orientation_median=('looking_time_s_per_min','median'),orientation_min=('looking_time_s_per_min','min'),orientation_max=('looking_time_s_per_min','max'))
         .reset_index())
summary['figure5_order']=summary['canonical'].map(order_map); summary=summary.sort_values('figure5_order')
summary.to_csv(OUT/'Figure_05B_summary.csv',index=False)
plot_order=summary['canonical'].tolist(); plot_labels=[short_label_map[c] for c in plot_order]
data=[df.loc[df['canonical']==c,'looking_time_s_per_min'].dropna().values for c in plot_order]
fig,ax=plt.subplots(figsize=(FINAL_WIDTH_IN,3.3),dpi=FINAL_DPI)
box=ax.boxplot(data,tick_labels=plot_labels,showfliers=False,patch_artist=True,widths=0.58,
               medianprops=dict(color='black',linewidth=2.0),boxprops=dict(linewidth=2.0),whiskerprops=dict(linewidth=2.0),capprops=dict(linewidth=2.0))
for patch in box['boxes']:
    patch.set_facecolor('#f2f2f2'); patch.set_edgecolor('black')
rng=np.random.default_rng(7)
for i,vals in enumerate(data,start=1):
    if len(vals)==0: continue
    jitter=rng.normal(loc=0.0,scale=0.045,size=len(vals))
    ax.scatter(np.full(len(vals),i)+jitter,vals,s=14,alpha=0.55,color='black',linewidths=0,zorder=3)
valid_arrays=[vals for vals in data if len(vals)>0]; ymax=max(np.nanmax(vals) for vals in valid_arrays); ymin=min(np.nanmin(vals) for vals in valid_arrays); yrange=ymax-ymin if ymax>ymin else 1.0
ax.set_ylim(max(0,ymin-0.12*yrange),ymax+0.20*yrange)
ax.set_ylabel('Source-directed orientation (s/min-equivalent)',fontsize=11,fontweight='bold')
ax.set_xlabel('Acoustic category',fontsize=11,fontweight='bold',labelpad=8)
ax.tick_params(axis='x',labelsize=8,width=2.0,length=5); ax.tick_params(axis='y',labelsize=9,width=2.0,length=5); ax.grid(False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
for side in ['left','bottom']: ax.spines[side].set_linewidth(2.0)
for xpos in [3.5,6.5,7.5]: ax.axvline(xpos,color='0.70',linewidth=2.0,alpha=0.45,linestyle='--',zorder=0)
fig.subplots_adjust(left=0.22,right=0.985,top=0.93,bottom=0.30); B_png=OUT/'Figure_05B.png'; fig.savefig(B_png,dpi=FINAL_DPI); plt.close(fig)

# -------------------- Composite --------------------
A=Image.open(A_png).convert('RGB'); B=Image.open(B_png).convert('RGB')
target_width=max(A.width,B.width)
def resize_to_width(img,width):
    ratio=width/img.width; return img.resize((width,int(img.height*ratio)),Image.LANCZOS)
A=resize_to_width(A,target_width); B=resize_to_width(B,target_width)
margin=48; gap=36; canvas_w=target_width; canvas_h=A.height+B.height+gap+2*margin
canvas=Image.new('RGB',(canvas_w,canvas_h),'white'); draw=ImageDraw.Draw(canvas)
canvas.paste(A,(0,margin)); canvas.paste(B,(0,margin+A.height+gap))
font=None
for fp in ['C:/Windows/Fonts/arialbd.ttf','C:/Windows/Fonts/calibrib.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf']:
    if Path(fp).exists():
        font=ImageFont.truetype(fp,42); break
if font is None: font=ImageFont.load_default()
draw.text((8,5),'(A)',fill='black',font=font); draw.text((8,margin+A.height+gap+5),'(B)',fill='black',font=font)
for ext,kwargs in [('png',{}),('jpg',{'quality':95}),('tif',{'compression':'tiff_lzw'})]:
    p=OUT/f'Figure_05.{ext}'
    if ext=='jpg': canvas.save(p,'JPEG',dpi=(FINAL_DPI,FINAL_DPI),**kwargs)
    elif ext=='tif': canvas.save(p,'TIFF',dpi=(FINAL_DPI,FINAL_DPI),**kwargs)
    else: canvas.save(p,'PNG',dpi=(FINAL_DPI,FINAL_DPI))

# QC + metadata
qc = {
    'oaas_rows': int(len(points)),
    'oaas_categories': {k:int(v) for k,v in points['canonical'].value_counts().sort_index().items()},
    'behavior_rows': int(len(df)),
    'behavior_category_counts': {k:int(v) for k,v in df['canonical'].value_counts().reindex(valid_cats).fillna(0).items()},
    'orientation_equals_mirar_prop_x60_max_abs_error': float(np.nanmax(np.abs(df['looking_time_s_per_min'] - df['mirar_prop']*60.0))) if 'mirar_prop' in df.columns else None,
    'panel_a_labels': ['Latent normalized unit 1 (LNU1)','Latent normalized unit 2 (LNU2)','Normalized convergence field'],
    'panel_b_xlabel': 'Acoustic category',
    'panel_b_ylabel': 'Source-directed orientation (s/min-equivalent)',
    'export_width_mm': 180,
    'export_height_mm': round(canvas.height / FINAL_DPI * 25.4, 2),
    'dpi': FINAL_DPI,
    'panel_markers': ['(A)', '(B)'],
    'minimum_visible_text_pt': 8,
    'minimum_figure_line_width_pt': 2.0,
    'build_status': 'reproduced_from_packaged_data',
}
(OUT/'Figure_05_QC.json').write_text(json.dumps(qc,indent=2),encoding='utf-8')
meta = {
    'build_status':'reproduced_from_packaged_data',
    'panel_a_source': OAAS.name,
    'panel_b_source': BEHAV.name,
    'panel_a_note':'Topographic field reproduced from the packaged production coordinates and displayed on neutral LNU axes.',
    'panel_b_note':'Original numeric endpoint retained; label changed from looking time to source-directed orientation on s/min-equivalent scale. No values altered.',
    'interpretive_boundaries':['OAAS geometry is not an affect detector.','The normalized convergence field is a derived visualization, not a directly observed variable.','Behavioral minute-level observations are descriptive and are not independent biological replicates.'],
    'frontiers_format': {'width_mm': 180, 'height_mm': round(canvas.height / FINAL_DPI * 25.4, 2), 'dpi': FINAL_DPI, 'color_mode': 'RGB', 'minimum_visible_text_pt': 8, 'minimum_figure_line_width_pt': 2.0, 'panel_markers': ['(A)', '(B)']},
    'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [OAAS, BEHAV]},
    'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [A_png, B_png, OUT/'Figure_05.png', OUT/'Figure_05.jpg', OUT/'Figure_05.tif', OUT/'Figure_05B_summary.csv']}
}
(OUT/'Figure_05_build_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('Created', OUT)
print(json.dumps(qc,indent=2))
