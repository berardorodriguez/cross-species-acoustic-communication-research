#!/usr/bin/env python3
"""
Figure 2 —  multiscale nearest-reference correspondence
Canonical reproducibility / production script.

Panels A-C:
- exact one-to-one MICRO/MACRO matching by id
- source domains: pastoral_human, non_pastoral_human, animal_comparison

Panel D:
- preferred source: 130-row exact-ID Obonuco fragment table with
  columns class_micro and class_macro
- precomputed aggregate fallback allowed only with explicit flag

Scientific boundary:
The figure shows descriptive nearest-reference correspondence of the same
identified acoustic units in independently fitted MICRO and MACRO geometries.
It does NOT represent physical movement, transition probability, temporal
trajectory, causal MICRO->MACRO transition, attractor evidence, or
aggregation-independent emergence.
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from PIL import Image, ImageOps

CLASSES = ["GC", "LFC", "HFC", "PSC"]
DPI = 300
FINAL_WIDTH_MM = 180.0
MM_PER_INCH = 25.4
FONT_MIN_PT = 8.0
FONT_AXIS_PT = 8.5
FONT_TITLE_PT = 9.0
FONT_PANEL_PT = 10.0
LINE_PT = 2.25

DOMAIN_SPECS = {
    "A": ("pastoral_human", "Pastoral human traditions"),
    "B": ("non_pastoral_human", "Non-pastoral human practices"),
    "C": ("animal_comparison", "Cross-species comparison"),
}

def clean_class(x):
    x = str(x).strip().upper()
    return x.replace("DIST_TO_", "").replace("NEAREST_", "").replace("CLASS_", "")

def standardize_domain(x):
    x = str(x).strip().lower()
    mapping = {
        "pastoral": "pastoral_human",
        "human_pastoral": "pastoral_human",
        "non-pastoral": "non_pastoral_human",
        "human_non_pastoral": "non_pastoral_human",
        "animal": "animal_comparison",
        "porcine": "animal_comparison",
        "pig": "animal_comparison",
        "cross_species": "animal_comparison",
        "cross-species": "animal_comparison",
    }
    return mapping.get(x, x)

def exact_match(micro, macro, domain):
    mi = micro.loc[micro["domain"] == domain, ["id","nearest_bovine_class"]].copy()
    ma = macro.loc[macro["domain"] == domain, ["id","nearest_bovine_class"]].copy()
    if mi["id"].duplicated().any() or ma["id"].duplicated().any():
        raise ValueError(f"Duplicate id(s) in {domain}")
    mi = mi.rename(columns={"nearest_bovine_class":"class_micro"})
    ma = ma.rename(columns={"nearest_bovine_class":"class_macro"})
    out = mi.merge(ma,on="id",how="outer",indicator=True,validate="one_to_one")
    if (out["_merge"]!="both").any():
        raise ValueError(f"Unmatched MICRO/MACRO ids in {domain}")
    return out.drop(columns="_merge")

def matrices(pairs):
    c = pd.crosstab(pairs["class_micro"],pairs["class_macro"]).reindex(
        index=CLASSES,columns=CLASSES,fill_value=0
    )
    r = c.sum(axis=1)
    p = c.div(r.replace(0,np.nan),axis=0)*100
    return c,r,p

def precomputed_d_counts():
    c=pd.DataFrame(0,index=CLASSES,columns=CLASSES,dtype=int)
    c.loc["GC","GC"]=1
    c.loc["HFC","GC"]=117
    c.loc["HFC","LFC"]=4
    c.loc["HFC","HFC"]=4
    c.loc["HFC","PSC"]=2
    c.loc["PSC","GC"]=2
    return c

def render_panel(counts,props,panel,title,subtitle,outfile):
    rows=counts.sum(axis=1)
    arr=np.ma.masked_invalid(props.to_numpy(dtype=float))
    fig=plt.figure(figsize=(3.25,3.08))
    ax=fig.add_axes([0.24,0.16,0.71,0.68])
    base_cmap=plt.get_cmap("viridis")
    cmap=LinearSegmentedColormap.from_list("viridis_uniform_black_text",base_cmap(np.linspace(0.50,1.00,256)))
    cmap.set_bad("0.88")
    ax.imshow(arr,vmin=0,vmax=100,aspect="equal",cmap=cmap)
    ax.set_xticks(range(4),CLASSES,fontsize=FONT_AXIS_PT)
    ax.set_yticks(range(4),[f"{r} (n={int(rows.loc[r])})" for r in CLASSES],
                  fontsize=FONT_AXIS_PT)
    ax.set_xlabel("MACRO nearest bovine reference",fontsize=FONT_AXIS_PT,labelpad=4)
    ax.set_ylabel("MICRO nearest bovine reference",fontsize=FONT_AXIS_PT,labelpad=5)
    ax.set_title(f"{title}\n{subtitle}",fontsize=FONT_TITLE_PT,pad=7)
    fig.text(0.025,0.965,f"({panel})",fontsize=FONT_PANEL_PT,fontweight="bold",
             ha="left",va="top")
    ax.set_xticks(np.arange(-0.5,4,1),minor=True)
    ax.set_yticks(np.arange(-0.5,4,1),minor=True)
    ax.grid(which="minor",linewidth=LINE_PT,color="0.72")
    ax.tick_params(which="minor",bottom=False,left=False)
    for s in ax.spines.values():
        s.set_linewidth(LINE_PT)
    for i,r in enumerate(CLASSES):
        if int(rows.loc[r])==0:
            continue
        for j,c in enumerate(CLASSES):
            n=int(counts.loc[r,c])
            if n>0:
                pct=float(props.loc[r,c])
                pt="100%" if np.isclose(pct,100) else f"{pct:.1f}%"
                ax.text(j,i,f"{n}\n({pt})",ha="center",va="center",
                        fontsize=FONT_MIN_PT,color="black")
    fig.savefig(outfile,dpi=DPI,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--micro",required=True)
    ap.add_argument("--macro",required=True)
    ap.add_argument("--obonuco-exact-id",default=None)
    ap.add_argument("--allow-precomputed-aggregate-fallback",action="store_true")
    ap.add_argument("--outdir",required=True)
    args=ap.parse_args()

    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    micro=pd.read_csv(args.micro); macro=pd.read_csv(args.macro)
    for df in (micro,macro):
        df.columns=[str(c).strip() for c in df.columns]
        df["domain"]=df["domain"].apply(standardize_domain)
        df["nearest_bovine_class"]=df["nearest_bovine_class"].apply(clean_class)

    panel_paths=[]
    for panel,(domain,title) in DOMAIN_SPECS.items():
        pairs=exact_match(micro,macro,domain)
        c,r,p=matrices(pairs)
        pairs.to_csv(out/f"Figure02_{panel}_{domain}_exact_id_matches.csv",index=False)
        c.to_csv(out/f"Figure02_{panel}_{domain}_counts.csv")
        p.to_csv(out/f"Figure02_{panel}_{domain}_within_micro_row_percent.csv")
        subtitle=f"n = {len(pairs)} matched units"
        pp=out/f"Figure02_{panel}.png"
        render_panel(c,p,panel,title,subtitle,pp)
        panel_paths.append(pp)

    if args.obonuco_exact_id:
        d=pd.read_csv(args.obonuco_exact_id)
        if not {"class_micro","class_macro"}.issubset(d.columns):
            raise ValueError("Obonuco exact-ID table needs class_micro and class_macro")
        if len(d)!=130:
            raise ValueError(f"Expected 130 Obonuco fragment rows; found {len(d)}")
        d["class_micro"]=d["class_micro"].apply(clean_class)
        d["class_macro"]=d["class_macro"].apply(clean_class)
        c,r,p=matrices(d)
    elif args.allow_precomputed_aggregate_fallback:
        c=precomputed_d_counts()
        r=c.sum(axis=1)
        p=c.div(r.replace(0,np.nan),axis=0)*100
    else:
        raise ValueError("Panel D requires the exact-ID table or the explicit precomputed-aggregate fallback flag.")

    pp=out/"Figure02_D.png"
    render_panel(c,p,"D","Obonuco fragments","n = 130 matched fragment units",pp)
    panel_paths.append(pp)

    # Shared colorbar
    fig=plt.figure(figsize=(6.4,0.72))
    cax=fig.add_axes([0.06,0.43,0.57,0.25])
    base_cmap=plt.get_cmap("viridis")
    cmap=LinearSegmentedColormap.from_list("viridis_uniform_black_text",base_cmap(np.linspace(0.50,1.00,256)))
    sm=ScalarMappable(norm=Normalize(0,100),cmap=cmap)
    cb=fig.colorbar(sm,cax=cax,orientation="horizontal")
    cb.set_label("Within-MICRO-row proportion (%)",fontsize=FONT_AXIS_PT,labelpad=3)
    cb.ax.tick_params(labelsize=FONT_MIN_PT,width=LINE_PT)
    cb.outline.set_linewidth(LINE_PT)
    from matplotlib.patches import Rectangle
    fig.patches.append(Rectangle((0.69,0.43),0.035,0.22,transform=fig.transFigure,facecolor="0.88",edgecolor="0.55",linewidth=0.7))
    fig.text(0.75,0.56,"No observations",fontsize=FONT_MIN_PT,ha="left",va="center")
    fig.text(0.75,0.40,"MICRO row n = 0",fontsize=FONT_MIN_PT,ha="left",va="center")
    cbp=out/"Figure02_shared_colorbar.png"
    fig.savefig(cbp,dpi=DPI,bbox_inches="tight",facecolor="white")
    plt.close(fig)

    # Composite at exact 180 mm width
    imgs=[Image.open(x).convert("RGB") for x in panel_paths]
    mw=max(x.width for x in imgs); mh=max(x.height for x in imgs)
    def pad(im,w,h):
        l=(w-im.width)//2; t=(h-im.height)//2
        return ImageOps.expand(im,border=(l,t,w-im.width-l,h-im.height-t),fill="white")
    imgs=[pad(x,mw,mh) for x in imgs]
    gap=18
    grid=Image.new("RGB",(2*mw+gap,2*mh+gap),"white")
    grid.paste(imgs[0],(0,0)); grid.paste(imgs[1],(mw+gap,0))
    grid.paste(imgs[2],(0,mh+gap)); grid.paste(imgs[3],(mw+gap,mh+gap))
    cbim=Image.open(cbp).convert("RGB")
    tw=int(grid.width*0.72)
    cbim=cbim.resize((tw,round(cbim.height*tw/cbim.width)),Image.Resampling.LANCZOS)
    canvas=Image.new("RGB",(grid.width,grid.height+cbim.height+14),"white")
    canvas.paste(grid,(0,0)); canvas.paste(cbim,((grid.width-cbim.width)//2,grid.height+4))
    target=round((FINAL_WIDTH_MM/MM_PER_INCH)*DPI)
    scale=target/canvas.width
    canvas=canvas.resize((target,round(canvas.height*scale)),Image.Resampling.LANCZOS)
    canvas.save(out/"Figure_02.png",dpi=(DPI,DPI))
    canvas.save(out/"Figure_02.jpg",quality=95,dpi=(DPI,DPI))
    canvas.save(out/"Figure_02.tif",compression="tiff_lzw",dpi=(DPI,DPI))

if __name__=="__main__":
    main()
