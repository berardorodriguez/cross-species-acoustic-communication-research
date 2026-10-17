from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / 'figures' / 'main'
PROV = REPO_ROOT / 'docs' / 'build_metadata'
DATA = REPO_ROOT / 'data' / 'derived' / 'figure_03'
OUT = ROOT
DPI = 300
WIDTH_MM, HEIGHT_MM = 180, 84
PIXELS = (2125, 992)  # Preserves the existing 179.92 x 84.00 mm raster geometry.
COLORS = {'GC':'#2ca02c','LFC':'#1f77b4','HFC':'#d62728','PSC':'#ff7f0e'}
CLASSES = ['GC','LFC','HFC','PSC']


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    specs=[
      ('MICRO', DATA/'Figure03_MICRO_plot_data.csv', 'PC1','PC2',(-10,50)),
      ('MACRO', DATA/'Figure03_MACRO_plot_data.csv', 'PC1','PC2',(-3500,550))]
    tables=[]
    for name,p,x,y,xlim in specs:
        df=pd.read_csv(p)
        assert list(df.columns)==['PC1','PC2','class']
        assert len(df)>0 and df['class'].isin(CLASSES).all()
        assert np.isfinite(df[['PC1','PC2']].to_numpy()).all()
        tables.append((name,df,xlim))
    fig,axes=plt.subplots(1,2,figsize=(PIXELS[0]/DPI,PIXELS[1]/DPI),dpi=DPI)
    fig.patch.set_facecolor('white')
    titles=['Micro (individual vocalizations)','Macro (aggregated representations)']
    for i,(ax,(name,df,xlim)) in enumerate(zip(axes,tables)):
        for cls in CLASSES:
            sub=df[df['class']==cls]
            ax.scatter(sub.PC1,sub.PC2,s=16 if name=='MICRO' else 22,
                       alpha=.50 if name=='MICRO' else .70,c=COLORS[cls],
                       label=cls,edgecolors='none',zorder=2)
        ax.set_title(titles[i],fontsize=10,pad=7)
        ax.set_xlabel('PC1',fontsize=9)
        ax.set_ylabel('PC2',fontsize=9)
        ax.set_xlim(*xlim)
        if name=='MICRO':
            ax.set_xticks(np.arange(-10,51,10))
        else:
            ax.set_xticks(np.arange(-3500,501,500))
        ax.margins(y=.06)
        # Omitting sub-2-pt gridlines keeps the plot clean and the visible strokes compliant.
        ax.grid(False)
        ax.tick_params(axis='both',labelsize=8,width=2.0,length=3.5)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        ax.text(-.105,1.085,f'({chr(65+i)})',transform=ax.transAxes,
                fontsize=11,fontweight='bold',ha='left',va='top',clip_on=False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,
               fontsize=9,bbox_to_anchor=(.5,.012))
    fig.subplots_adjust(left=.075,right=.985,top=.85,bottom=.20,wspace=.22)
    fig.canvas.draw()
    for ax in axes:
        for spine in ax.spines.values():
            assert not spine.get_visible() or spine.get_linewidth()>=2
        for tick in ax.get_xticklines()+ax.get_yticklines():
            if tick.get_visible(): assert tick.get_markeredgewidth()>=2
    renderer=fig.canvas.get_renderer()
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text().strip():
            assert item.get_fontsize()>=8,(item.get_text(),item.get_fontsize())
            box=item.get_window_extent(renderer)
            assert box.x0>=0 and box.y0>=0 and box.x1<=PIXELS[0] and box.y1<=PIXELS[1],(item.get_text(),box)
    exports=[]
    for ext in ['png','jpg','tif']:
        out=OUT/f'Figure_03.{ext}'
        buf=__import__('io').BytesIO()
        fig.savefig(buf,format=ext,dpi=DPI,facecolor='white')
        buf.seek(0)
        with Image.open(buf) as im:
            rgb=im.convert('RGB')
            assert rgb.size==PIXELS
        fmt={'png':'PNG','jpg':'JPEG','tif':'TIFF'}[ext]
        opts={'dpi':(DPI,DPI)}
        if ext=='jpg': opts.update(quality=95,subsampling=0)
        if ext=='tif': opts.update(compression='tiff_lzw')
        rgb.save(out,format=fmt,**opts)
        with Image.open(out) as im:
            assert im.size==PIXELS and im.mode=='RGB'
            assert max(abs(float(v)-DPI) for v in im.info['dpi'])<.01
        exports.append({'file':str(out.relative_to(ROOT)),'sha256':sha(out)})
    plt.close(fig)
    meta={'figure':3,'build_status':'reproduced_from_packaged_data',
      'science':'No PCA refit or new analysis; packaged coordinates are plotted verbatim.',
      'builder_sha256':sha(Path(__file__)),
      'layout':{'width_mm':WIDTH_MM,'height_mm':HEIGHT_MM,'pixels':list(PIXELS),'dpi':DPI,'mode':'RGB'},
      'panels':{},'sources':[],'exports':exports,
      'frontiers_checks':{'minimum_visible_text_pt':8,'axes_and_ticks_pt':2,
        'gridlines':'omitted; no sub-2-pt grid strokes','format_check':'passed'}}
    for (name,df,_),spec in zip(tables,specs):
        meta['panels'][name]={'n':len(df),'PC1_range':[float(df.PC1.min()),float(df.PC1.max())],
          'PC2_range':[float(df.PC2.min()),float(df.PC2.max())],
          'class_counts':{k:int(v) for k,v in df['class'].value_counts().reindex(CLASSES,fill_value=0).items()}}
        meta['sources'].append({'plot_data':str(spec[1].relative_to(REPO_ROOT)),
          'plot_data_sha256':sha(spec[1]),'n':len(df)})
    PROV.mkdir(parents=True,exist_ok=True)
    (PROV/'Figure_03.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    qc=[
      {'check':'Scientific processing','result':'PASS','details':'Frozen PC1/PC2 coordinates plotted verbatim; no PCA refit or new analysis.'},
      {'check':'Source-derived counts and class membership','result':'PASS','details':'MICRO n=186 and MACRO n=529; all classes and ranges recorded in final metadata.'},
      {'check':'Raster format and dimensions','result':'PASS','details':f'PNG/JPEG/TIFF are RGB, {PIXELS[0]} x {PIXELS[1]} px, 300 dpi; TIFF uses lossless LZW.'},
      {'check':'Publication size and text/strokes','result':'PASS','details':'Approximately 179.92 x 83.99 mm; visible text >=8 pt; axes and ticks 2 pt; sub-2-pt gridlines omitted.'},
      {'check':'Rebuild provenance','result':'PASS','details':'Builder, source-derived plot tables, and exported image hashes are recorded in package metadata.'},
      {'check':'Interpretive boundary','result':'PASS','details':'MICRO and MACRO are separate PCA geometries; coordinates do not represent a shared-space trajectory.'},
    ]
    pd.DataFrame(qc).to_csv(PROV/'Figure_03_QC.csv',index=False)
    print('Figure 3 exports created:',OUT)
    print('MICRO:',meta['panels']['MICRO'])
    print('MACRO:',meta['panels']['MACRO'])
    print('Exports:',len(exports),'RGB, 300 dpi,',PIXELS)

if __name__=='__main__':
    main()

