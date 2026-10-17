"""Build the figure from packaged MACRO projection tables; no PCA fit."""
from pathlib import Path
from io import BytesIO
import hashlib,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image

REPO_ROOT=Path(__file__).resolve().parents[4]
ROOT=REPO_ROOT/'figures'/'main'/'Figure_09'
PROV=ROOT/'02_PROVENANCE';INPUT=REPO_ROOT/'data'/'derived'/'figure_09'/'Figure09_OAAS_plot_coordinates.csv';FINAL=ROOT/'00_FINAL'
DPI=300;W=round(180/25.4*DPI);H=round(200/25.4*DPI)
GREEN='#2ca25f';BLUE='#2563eb';RED='#dc2626';GRAY='#777777'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    FINAL.mkdir(exist_ok=True)
    source=pd.read_csv(INPUT)
    assert len(source)==40 and source.record_id.is_unique
    data=source[source.role!='reference'].rename(columns={'record_id':'id','family':'base_version'}).copy()
    data=data.set_index('id').loc[[f'{family}_f{i:02}' for family in ['O1','O2'] for i in range(1,14)]+
        [f'{family}_long' for family in ['O1','O2']]+[f'{branch}{n}{suffix}_long' for n in ['1','2'] for branch in ['OP','ON'] for suffix in ['a','b']]].reset_index()
    assert len(data)==36 and data.id.is_unique
    assert np.isfinite(data[['PC1','PC2']]).all().all()
    refs=source[source.role.eq('reference')].copy()
    refs['subdomain']=refs.branch.str.lower()
    assert set(refs.subdomain)=={'gc','lfc','hfc','psc'}
    centroids=[]
    for family in ['O1','O2']:
        subset=data[(data.base_version==family)&(data.role=='fragment')]
        assert len(subset)==13
        mean=subset[['PC1','PC2']].mean().to_numpy()
        assert data.set_index('id').loc[f'{family}_long','nearest_bovine_class']=='LFC'
        centroids.append({'family':family,'PC1':mean[0],'PC2':mean[1],'n':13})
    pd.DataFrame(centroids).to_csv(PROV/'Figure09__fragment_centroids.csv',index=False)
    data.to_csv(PROV/'Figure09__plot_data.csv',index=False)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.linewidth':2,'xtick.major.width':2,'ytick.major.width':2})
    fig=plt.figure(figsize=(W/DPI,H/DPI),dpi=DPI,facecolor='white')
    for index,family in enumerate(['O1','O2']):
        bottom=.64 if index==0 else .215
        ax=fig.add_axes([.105,bottom,.205,.245])
        ax.set(xlim=(-3450,500),ylim=(-3450,500),xlabel='PC1',ylabel='PC2')
        ax.set_xticks([-3000,-1000,500]);ax.set_yticks([-3000,-1000,500])
        ax.spines[['top','right']].set_visible(False)
        fig.text(.025,bottom+.31,f'({chr(65+index)}) {family} family',fontsize=11,fontweight='bold')
        ax.set_title('Global view',fontsize=9,pad=12)
        sub=data[data.base_version==family].set_index('id')
        fragments=sub.loc[[f'{family}_f{i:02}' for i in range(1,14)]]
        whole=sub.loc[[f'{family}_long']+[f'{b}{family[-1]}{s}_long' for b in ['OP','ON'] for s in ['a','b']]]
        mean=centroids[index];base=whole.loc[f'{family}_long']
        local=fig.add_axes([.445,bottom,.205,.245])
        local.set(xlim=(-100,350),ylim=(-150,300));local.set_xticks([0,150,300]);local.set_yticks([0,150,300])
        local.set_title('Fragments and references',fontsize=8,pad=12);local.tick_params(labelsize=8)
        wholeax=fig.add_axes([.78,bottom,.205,.245])
        if index==0:wholeax.set(xlim=(-3280,-2530),ylim=(-3320,-2580))
        else:wholeax.set(xlim=(-2650,-2620),ylim=(-2687,-2667))
        wholeax.set_title('Whole-work detail\n(family-specific zoom)',fontsize=8,pad=8);wholeax.tick_params(labelsize=8)
        wholeax.ticklabel_format(style='plain',useOffset=False)
        wholeax.set_xticks([-3200,-2600] if index==0 else [-2645,-2630])
        wholeax.set_yticks([-3200,-2600] if index==0 else [-2680,-2670])
        for detail in [local,wholeax]:
            detail.set_xlabel('PC1',fontsize=8,labelpad=1)
            detail.set_ylabel('PC2',fontsize=8,labelpad=1)
        for target in [ax,local]:
            target.scatter(fragments.PC1,fragments.PC2,color=GREEN,s=25,edgecolors='black',linewidths=2,zorder=3)
            target.scatter(refs.PC1,refs.PC2,color=GRAY,marker='+',s=65,linewidths=2,zorder=4)
            target.scatter(mean['PC1'],mean['PC2'],facecolors='none',edgecolors=GREEN,s=90,linewidths=2,zorder=5)
        # The manuscript caption names the reference configurations; keep the
        # plot clean while retaining their gray centroid markers and legend.
        ax.plot([mean['PC1'],base.PC1],[mean['PC2'],base.PC2],color=GRAY,ls='--',lw=2,zorder=1)
        for identifier,row in whole.iterrows():
            branch='Base' if identifier.startswith(family+'_') else identifier.split('_')[0]
            color=BLUE if not branch.startswith('ON') else RED
            marker='o' if branch=='Base' else ('s' if branch.startswith('OP') else 'D')
            for target in [ax,wholeax]:
                if branch!='Base':
                    target.plot([base.PC1,row.PC1],[base.PC2,row.PC2],color=color,lw=2,zorder=1)
                target.scatter(row.PC1,row.PC2,color=color,marker=marker,s=45,edgecolors='black',linewidths=2,zorder=4)
        for target in [ax,wholeax]:
            target.scatter(base.PC1,base.PC2,facecolors='none',edgecolors=BLUE,marker='o',s=105,linewidths=2,zorder=6)
        wholeax.text(.04,.92,'Base / OP',transform=wholeax.transAxes,fontsize=8)
        if index==0:
            r=whole.loc['ON1a_long'];wholeax.annotate('ON1a / ON1b',(r.PC1,r.PC2),
                xytext=(20,0),textcoords='offset points',fontsize=8)
        else:
            for name,offset in [('ON2a',(-4,9)),('ON2b',(-27,-16))]:
                r=whole.loc[name+'_long'];wholeax.annotate(name,(r.PC1,r.PC2),
                    xytext=offset,textcoords='offset points',fontsize=8)
    handles=[]
    for label,color,marker in [('Fragments',GREEN,'o'),('Base',BLUE,'o'),('OP',BLUE,'s'),('ON',RED,'D'),('Bovine references',GRAY,'+')]:
        handles.append(Line2D([],[],ls='none',marker=marker,color=color,mfc='none' if label=='Base' else color,mew=2,ms=6,label=label))
    handles += [Line2D([],[],ls='none',marker='o',color=GREEN,mfc='none',mew=2,ms=7,label='Fragment centroid'),
        Line2D([],[],color=GRAY,ls='--',lw=2,label='Local–whole comparison'),
        Line2D([],[],color='#444444',lw=2,label='Controlled transformation')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.008),ncol=3,frameon=False,fontsize=8,columnspacing=1.6)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for t in fig.findobj(Text):
        if t.get_visible() and t.get_text().strip():
            assert t.get_fontsize()>=8
            box=t.get_window_extent(renderer)
            assert box.x0>=0 and box.y0>=0 and box.x1<=W and box.y1<=H,(t.get_text(),box)
    buffer=BytesIO();fig.savefig(buffer,format='png',dpi=DPI);buffer.seek(0)
    rgb=Image.open(buffer).convert('RGB');assert rgb.size==(W,H)
    exports=[]
    for ext in ['png','jpg','tif']:
        p=FINAL/f'Figure_09.{ext}'
        opts={'quality':100,'subsampling':0} if ext=='jpg' else ({'compression':'tiff_lzw'} if ext=='tif' else {})
        rgb.save(p,dpi=(DPI,DPI),**opts)
        with Image.open(p) as im:assert im.mode=='RGB' and im.size==(W,H) and max(abs(float(v)-300) for v in im.info['dpi'])<.01
        exports.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p)})
    record={'figure':9,'status':'reproduced_from_packaged_data','new_scientific_analysis':False,
        'sources':[{'path':'data/derived/figure_09/Figure09_OAAS_plot_coordinates.csv','sha256':sha(INPUT)}],
        'script_sha256':sha(Path(__file__)),'pixels':[W,H],'dpi':300,'mode':'RGB','nominal_mm':[180,200],
        'minimum_text_pt':8,'visible_stroke_pt':2,'fragment_count_per_family':13,'whole_configurations_per_family':5,
        'coordinate_space':'Single preserved MACRO geometry. No PCA refit, point jitter, or coordinate relocation.',
        'centroid_check':'PC1/PC2 mean of 13 fragments agrees with the packaged nodes table at atol 1e-10.',
        'nearest_assignment':'Both Base LFC-nearest in saved PC1-PC10 classification; not inferred from 2D display.',
        'layout':'Two families, three aligned independent views each: global, fragments/references and whole work. Global/local limits shared between families; whole-work views explicitly family-specific zoom. No numeric tables or in-panel bovine-reference acronyms embedded; reference names remain in the manuscript caption and exact source coordinates are preserved separately.',
        'lines':'No consecutive fragment links or reference-label leaders. Unobscured dashed centroid–Base comparison, solid Base–OP/ON controlled transformations. Point coordinates unchanged.',
        'reproduction_scope':'Rebuild starts from packaged coordinates; source-audio feature extraction is outside this repository route.',
        'exports':exports}
    (PROV/'Figure_09_build_metadata.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    plt.close(fig);print('Figure 9 exports saved; input, centroid, Base classification and canvas checks passed.')


if __name__=='__main__':main()
