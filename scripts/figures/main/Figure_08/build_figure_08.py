"""New  graphic from unmodified recovered stem projection/distance values."""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image

REPO_ROOT=Path(__file__).resolve().parents[4]
ROOT=REPO_ROOT/'figures'/'main'/'Figure_08'
PROV=ROOT/'02_PROVENANCE'
INPUT=REPO_ROOT/'data'/'derived'/'figure_08'/'Figure08_configuration_coordinates_and_distances.csv'
FINAL=ROOT/'00_FINAL'
DPI=300;WIDTH=round(180/25.4*DPI);HEIGHT=round(165/25.4*DPI)
ORDER=['O1_full','O1_noVox','O1_vox_stem','O2_full','O2_noVox']
LABELS=['O1 full','O1 no-vocal','Processed vocal stem\n(isolated)','O2 full','O2 no-vocal']
COLORS=['#6b479f','#6b479f','#555555','#14834c','#14834c']
MARKERS=['o','s','D','o','s']


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    data=pd.read_csv(INPUT).set_index('version').loc[ORDER]
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.linewidth':2,
        'xtick.major.width':2,'ytick.major.width':2})
    fig=plt.figure(figsize=(WIDTH/DPI,HEIGHT/DPI),dpi=DPI,facecolor='white')
    fig.text(.018,.978,'(A)',fontsize=11,fontweight='bold',va='top')
    ax=fig.add_axes([.10,.59,.86,.34])
    for (_,row),color,marker in zip(data.iterrows(),COLORS,MARKERS):
        ax.scatter(row.PCA1,row.PCA2,c=color,marker=marker,s=70,edgecolor='black',linewidth=2,zorder=3)
    ax.set(xlim=(0,25),ylim=(-14.4,-8.2),xlabel='PCA1',ylabel='PCA2')
    ax.set_xticks([0,5,10,15,20,25]);ax.set_yticks([-14,-13,-12,-11,-10,-9])
    ax.spines[['top','right']].set_visible(False)
    ax.annotate('O1 full',(data.loc['O1_full','PCA1'],data.loc['O1_full','PCA2']),
                xytext=(6,-11.35),fontsize=9,arrowprops={'arrowstyle':'-','lw':2})
    ax.annotate('Processed vocal stem\n(isolated)',(data.loc['O1_vox_stem','PCA1'],data.loc['O1_vox_stem','PCA2']),
                xytext=(15.5,-13.95),fontsize=9,ha='center',arrowprops={'arrowstyle':'-','lw':2})
    # Same saved coordinates enlarged solely to separate adjacent point labels.
    detail=fig.add_axes([.61,.745,.31,.175])
    for i in [0,1,3,4]:
        row=data.iloc[i]
        detail.scatter(row.PCA1,row.PCA2,c=COLORS[i],marker=MARKERS[i],s=45,edgecolor='black',linewidth=2,zorder=3)
    detail.set(xlim=(2.3,4.55),ylim=(-11.5,-8.35))
    detail.set_xticks([2.5,3.5,4.5]);detail.set_yticks([-11,-10,-9]);detail.tick_params(labelsize=8)
    detail.set_title('Full/no-vocal detail',fontsize=9,pad=5)
    detail.set_xlabel('PCA1',fontsize=8,labelpad=1);detail.set_ylabel('PCA2',fontsize=8,labelpad=1)
    positions={0:(4,-10.85),1:(3.2,-9.2),3:(3.2,-8.63),4:(3.2,-9.8)}
    for i,position in positions.items():
        row=data.iloc[i]
        detail.annotate(LABELS[i],(row.PCA1,row.PCA2),xytext=position,fontsize=8,
            arrowprops={'arrowstyle':'-','lw':2},va='center')
    fig.text(.018,.485,'(B)',fontsize=11,fontweight='bold',va='top')
    bx=fig.add_axes([.245,.115,.715,.285])
    for i,(color,(_,row)) in enumerate(zip(COLORS,data.iterrows())):
        for field,offset,style,marker in [('GC',-.115,'-','o'),('LFC',.115,'--','D')]:
            value=row[f'dist_to_{field}']; y=i+offset
            bx.plot([0,value],[y,y],color=color,lw=2,ls=style)
            bx.plot(value,y,color=color,marker=marker,ms=6,mec='black',mew=2)
            bx.text(value+.5,y,f'{value:.2f}',fontsize=8,va='center')
    bx.set_xlim(0,30);bx.set_ylim(4.5,-.5)
    bx.set_yticks(np.arange(5),LABELS);bx.set_xticks([0,5,10,15,20,25,30])
    bx.set_xlabel('Distance to bovine reference centroid (3D PCA)',fontsize=9)
    bx.spines[['top','right']].set_visible(False)
    handles=[Line2D([0],[0],color='#444444',ls=style,lw=2,marker=marker,ms=6,mew=2,label=f'{label} reference')
             for label,style,marker in [('GC','-','o'),('LFC','--','D')]]
    bx.legend(handles=handles,loc='lower right',bbox_to_anchor=(1,1.06),ncol=2,frameon=False,fontsize=9,borderaxespad=0)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts)>=8
    for t in texts:
        box=t.get_window_extent(renderer)
        assert box.x0>=0 and box.y0>=0 and box.x1<=WIDTH and box.y1<=HEIGHT,(t.get_text(),box)
    buf=BytesIO();fig.savefig(buf,format='png',dpi=DPI);buf.seek(0)
    rgb=Image.open(buf).convert('RGB');assert rgb.size==(WIDTH,HEIGHT)
    outputs=[]
    for ext in ['png','jpg','tif']:
        p=FINAL/f'Figure_08.{ext}'
        opts={'quality':100,'subsampling':0} if ext=='jpg' else ({'compression':'tiff_lzw'} if ext=='tif' else {})
        rgb.save(p,dpi=(DPI,DPI),**opts)
        with Image.open(p) as im:
            assert im.mode=='RGB' and im.size==(WIDTH,HEIGHT)
            assert max(abs(float(v)-DPI) for v in im.info['dpi'])<.01
        outputs.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p)})
    metadata={'figure':8,'status':'reproduced_from_packaged_data',
        'pixels':[WIDTH,HEIGHT],'mode':'RGB','dpi':DPI,'nominal_mm':[180,165],
        'minimum_text_pt':8,'minimum_visible_stroke_pt':2,'panels':'A PCA1–PCA2 with enlarged detail using identical values; B GC/LFC distances in full 3D PCA space.',
        'values':'Unmodified recovered projection/distance table values; display rounded B annotations only.',
        'caption_note':'The distance interpretation is provided in the manuscript caption.',
        'scientific_limits':'Correspondence only; no emotional equivalence, causal component effect, emergence, or trajectories. Single processed stem is author/manuscript provenance, not a mixing-session verification.',
        'accessibility':'Direct labels, full-circle/no-vocal-square markers, shared-stem diamond; GC solid/circle and LFC dashed/diamond.',
        'new_scientific_analysis':False,'script_sha256':sha(Path(__file__)),
        'input_sha256':sha(INPUT),
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'exports':outputs}
    (PROV/'Figure_08_build_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    plt.close(fig);print(f'Exported Figure 8: {WIDTH} x {HEIGHT}, 300-dpi RGB')


if __name__=='__main__':main()
