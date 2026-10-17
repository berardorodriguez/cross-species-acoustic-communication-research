"""Four-panel production: illustrative score and packaged acoustic panels.

Panels B/C/D use packaged derived arrays. Panel D uses the same Fourier
interpolation through a memory-bounded inverse transform. No values are inferred
from an image. Score fine-engraving certification remains open.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.patches import Rectangle
from PIL import Image
import PIL

REPO_ROOT=Path(__file__).resolve().parents[3]
ROOT=REPO_ROOT/'figures'/'main'
DATA=REPO_ROOT/'data'/'derived'/'figure_07'
SUPPORT=REPO_ROOT/'figures'/'support'/'figure_07'
FINAL=ROOT
DPI=300
WIDTH=round(180/25.4*DPI)
HEIGHT=round(250/25.4*DPI)


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    data=np.load(DATA/'acoustic_curves.npz')
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.linewidth':2,
                         'xtick.major.width':2,'ytick.major.width':2})
    fig=plt.figure(figsize=(WIDTH/DPI,HEIGHT/DPI),dpi=DPI,facecolor='white')
    score=Image.open(SUPPORT/'score_excerpt.png').convert('RGB')
    a=fig.add_axes([.04,.85,.94,.12]);a.imshow(score);a.axis('off')
    fig.text(.018,.988,'(A)',fontweight='bold',fontsize=10,va='top')
    fig.text(.13,.835,'Illustrative score excerpt',fontsize=9)
    fig.text(.018,.825,'(B)',fontweight='bold',fontsize=10,va='top')
    inputs=[DATA/'acoustic_curves.npz',SUPPORT/'score_excerpt.png']
    for row,material in enumerate(['O1','O2']):
        y=.69 if row==0 else .54
        fig.text(.047,y+.063,material,fontweight='bold',fontsize=9,va='center')
        for col,section in enumerate([1,7,13]):
            p=DATA/f'{material}_section_{section:02d}_spectrogram.npz'
            inputs.append(p)
            s=np.load(p)
            x=.13+col*.29
            ax=fig.add_axes([x,y,.25,.11])
            db=s['db'];sr=int(s['sample_rate']);hop=int(s['hop_length'])
            times=np.arange(db.shape[1])*hop/sr
            ax.pcolormesh(times,s['frequency_hz'],db,shading='auto',cmap='magma',vmin=-80,vmax=0,rasterized=True)
            duration=float(s['duration_s'])
            ax.set_xlim(0,duration);ax.set_ylim(0,9000)
            ax.set_yticks([0,2000,4000,6000,8000],['0','2','4','6','8'] if col==0 else ['']*5)
            ticks=[0,30,60] if duration<100 else ([0,90,180] if duration>170 else [0,60,120])
            ax.set_xticks(ticks,[str(v) for v in ticks])
            ax.tick_params(labelsize=8,length=3,width=2)
            if col==0:
                ax.set_ylabel('Frequency (kHz)',fontsize=9)
            if row==0:
                ax.set_title(f'Section {section}',fontsize=9,pad=8)
            else:
                ax.set_xlabel('Time (s)',fontsize=9,labelpad=3)
    fig.text(.018,.48,'(C)',fontweight='bold',fontsize=10,va='top')
    for panel,bottom,height,field,ylabel,ylim in [
        ('C',.32,.135,'loudness','Relative loudness',(0,1.05)),
        ('D',.055,.18,'roughness','Relative peak-pair dissonance',(0,1))]:
        if panel=='D':
            fig.text(.018,.27,'(D)',fontweight='bold',fontsize=10,va='top')
        ax=fig.add_axes([.091,bottom,.879,height])
        for material,color,style in [('O1','#1f77b4','-'),('O2','#ff7f0e','--')]:
            t=data[f'{material}_loudness_time_norm'] if field=='loudness' else data['roughness_time_norm']
            ax.plot(t,data[f'{material}_{field}_relative'],color=color,ls=style,lw=2,label=material,alpha=1)
        ax.set_xlim(0,1);ax.set_ylim(*ylim)
        ax.set_xticks([0,.2,.4,.6,.8,1])
        ax.set_yticks([0,.2,.4,.6,.8,1])
        ax.set_ylabel(ylabel,fontsize=9)
        ax.set_xlabel('Normalized time (0–1)',fontsize=9)
        ax.tick_params(labelsize=8,width=2,length=3)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',bbox_to_anchor=(1,1.025),ncol=2,fontsize=9,frameon=False,borderaxespad=0)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    assert min(t.get_fontsize() for t in texts)>=8
    for t in texts:
        b=t.get_window_extent(renderer)
        assert b.x0>=0 and b.y0>=0 and b.x1<=WIDTH and b.y1<=HEIGHT,(t.get_text(),b)
    buf=BytesIO();fig.savefig(buf,format='png',dpi=DPI,facecolor='white');buf.seek(0)
    rgb=Image.open(buf).convert('RGB')
    assert rgb.size==(WIDTH,HEIGHT)
    FINAL.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for ext in ['png','jpg','tif']:
        p=FINAL/f'Figure_07.{ext}'
        opts={'quality':100,'subsampling':0} if ext=='jpg' else ({'compression':'tiff_lzw'} if ext=='tif' else {})
        rgb.save(p,dpi=(DPI,DPI),**opts)
        with Image.open(p) as im:
            assert im.mode=='RGB' and im.size==(WIDTH,HEIGHT)
            assert max(abs(float(v)-DPI) for v in im.info['dpi'])<.01
            outputs.append({'file':str(p.relative_to(ROOT)),'sha256':sha(p),'pixels':list(im.size),'mode':im.mode,'dpi':[float(v) for v in im.info['dpi']]})
    metadata={'figure':7,'nominal_width_mm':180,'nominal_height_mm':250,
              'pixel_dimensions':[WIDTH,HEIGHT],'dpi':DPI,
              'input_sha256':{str(p.relative_to(REPO_ROOT)):sha(p) for p in inputs},
              'script_sha256':sha(Path(__file__)),'exports':outputs}
    metadata_path=REPO_ROOT/'docs'/'build_metadata'/'Figure_07.json'
    metadata_path.parent.mkdir(parents=True,exist_ok=True)
    metadata_path.write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    plt.close(fig)
    print(f'Figure 7 exported: {WIDTH} x {HEIGHT}, RGB, 300 dpi')


if __name__=='__main__':
    main()
