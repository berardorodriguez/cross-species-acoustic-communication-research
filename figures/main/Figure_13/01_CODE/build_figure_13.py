from pathlib import Path
import io
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, FancyArrowPatch, Rectangle, Arc, PathPatch
from matplotlib.lines import Line2D
from matplotlib.path import Path as MatplotlibPath
from PIL import Image

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[3]
FIGURE_DIR = REPOSITORY_ROOT / "figures" / "main"
OUT_DIR = FIGURE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)
STEM = "Figure_13"
OUT_SVG, OUT_PNG, OUT_TIF, OUT_QC = (OUT_DIR / f"{STEM}.{ext}" for ext in ("svg", "png", "tif", "QC.json"))
DPI, TIFF_DPI = 300, 600
WIDTH_MM, HEIGHT_MM = 179.9, 150
MIN_FONT_PT, MIN_LINE_PT = 8, 2
FONT_RENDER_SCALE = .75

INK, GRAY = "#171717", "#59616A"
GREEN, BLUE, PURPLE, CORAL = "#4E9A65", "#498BC6", "#8B62C5", "#D26F64"
PALE_GREEN, PALE_BLUE = "#F1F8F1", "#F1F7FC"
PALE_PURPLE, PALE_CORAL = "#F7F2FB", "#FCF3F1"

SUBMITTED_SOURCE = REPOSITORY_ROOT / "figures" / "support" / "figure_13" / "icon_source.tif"
SUBMITTED_CROPS = {
    "animal_vocalizations": (504, 231, 734, 300),
    "human_vocal_practice": (824, 231, 1061, 300),
    "musical_score": (1153, 231, 1358, 300),
    "soundscape": (1480, 231, 1678, 300),
    "bioacoustic_layer": (1648, 386, 1790, 458),
    "perceptual_layer": (1650, 548, 1795, 598),
    "musical_layer": (1648, 690, 1753, 739),
    "environment_layer": (1648, 838, 1795, 900),
    "playback_cow": (504, 231, 600, 300),
}
SUBMITTED_TIFF = Image.open(SUBMITTED_SOURCE).convert("L")

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "savefig.facecolor": "white",
                     "svg.hashsalt": "Figure_13_v1"})
fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=DPI, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def text(x, y, s, size=8, weight="normal", ha="center", va="center", color=INK, style="normal", linespacing=1.05):
    rendered_size=size*FONT_RENDER_SCALE
    if rendered_size < MIN_FONT_PT: raise ValueError(f"Text under 8 pt: {s}")
    return ax.text(x, y, s, fontsize=rendered_size, fontweight=weight, ha=ha, va=va, color=color,
                   style=style, linespacing=linespacing, family="DejaVu Sans", clip_on=False)

def line(x1, y1, x2, y2, color=INK, lw=MIN_LINE_PT, ls="-"):
    if lw < MIN_LINE_PT: raise ValueError("Stroke under 2 pt")
    ax.add_line(Line2D([x1,x2],[y1,y2],color=color,linewidth=lw,linestyle=ls,
                       solid_capstyle="round",dash_capstyle="round"))

def box(x,y,w,h,edge=INK,face="white",radius=.006,lw=MIN_LINE_PT):
    if lw < MIN_LINE_PT: raise ValueError("Box outline under 2 pt")
    p=FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad=0.003,rounding_size={radius}",
                     linewidth=lw,edgecolor=edge,facecolor=face)
    ax.add_patch(p); return p

def arrow(x1,y1,x2,y2,both=False,color=INK,lw=MIN_LINE_PT,ls="-"):
    if lw < MIN_LINE_PT: raise ValueError("Arrow under 2 pt")
    if both and abs(y2-y1) < 1e-12:
        tip1=x1+.006; tip2=x2-.006; head=.003
        line(tip1,y1,tip2,y2,color,lw,ls)
        line(tip1,y1,tip1+head,y1+head,color,lw,ls); line(tip1,y1,tip1+head,y1-head,color,lw,ls)
        line(tip2,y2,tip2-head,y2+head,color,lw,ls); line(tip2,y2,tip2-head,y2-head,color,lw,ls)
        return
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="<->" if both else "->",
        mutation_scale=9,linewidth=lw,linestyle=ls,color=color,shrinkA=0,shrinkB=0))

def dot(x,y,r,face="white",edge=INK,lw=MIN_LINE_PT):
    ax.add_patch(Circle((x,y),r,facecolor=face,edgecolor=edge,linewidth=lw))

def waveform(cx,cy,w=.050,h=.018,color=INK):
    vals=[.25,.42,.62,.86,.62,1,.62,.86,.62,.42,.25]
    for x,v in zip(np.linspace(cx-w/2,cx+w/2,len(vals)),vals): line(x,cy-h*v/2,x,cy+h*v/2,color,lw=MIN_LINE_PT)

def cow(cx,cy,s=.55,color=INK):
    ax.add_patch(Polygon([(cx-.03*s,cy-.007*s),(cx-.024*s,cy+.01*s),(cx+.012*s,cy+.01*s),
                          (cx+.021*s,cy+.004*s),(cx+.023*s,cy-.01*s),(cx-.025*s,cy-.01*s)],
                         closed=True,facecolor=color,edgecolor=color,linewidth=MIN_LINE_PT))
    dot(cx+.025*s,cy+.004*s,.008*s,color,color)
    for xx in [cx-.02*s,cx-.005*s,cx+.014*s]: line(xx,cy-.007*s,xx-.002*s,cy-.023*s,color)
    line(cx-.03*s,cy+.004*s,cx-.039*s,cy+.01*s,color)

def pig(cx,cy,s=.55,color=INK):
    ax.add_patch(Ellipse((cx,cy),.046*s,.026*s,facecolor=color,edgecolor=color,linewidth=MIN_LINE_PT))
    ax.add_patch(Polygon([(cx+.014*s,cy+.006*s),(cx+.019*s,cy+.020*s),(cx+.006*s,cy+.014*s)],
                         closed=True,facecolor=color,edgecolor=color,linewidth=MIN_LINE_PT))
    ax.add_patch(Ellipse((cx+.025*s,cy+.003*s),.015*s,.013*s,facecolor=color,edgecolor=color,linewidth=MIN_LINE_PT))
    for xx in [cx-.014*s,cx+.009*s]: line(xx,cy-.008*s,xx-.002*s,cy-.021*s,color)
    dot(cx+.027*s,cy+.006*s,.0018*s,"white","white",MIN_LINE_PT)

def person(cx,cy,s=.55,color=INK):
    dot(cx,cy+.008*s,.009*s,color,color)
    ax.add_patch(Polygon([(cx-.018*s,cy-.02*s),(cx-.013*s,cy-.003*s),(cx,cy+.001*s),
                          (cx+.013*s,cy-.003*s),(cx+.018*s,cy-.02*s)],closed=True,facecolor=color,edgecolor=color,lw=MIN_LINE_PT))

def music(cx,cy,s=.55,color=INK):
    for j in range(3): line(cx-.026*s,cy-.01*s+j*.009*s,cx+.026*s,cy-.01*s+j*.009*s,color)
    dot(cx-.012*s,cy-.002*s,.004*s,color,color); line(cx-.008*s,cy-.002*s,cx-.008*s,cy+.018*s,color)
    dot(cx+.010*s,cy+.005*s,.004*s,color,color); line(cx+.014*s,cy+.005*s,cx+.014*s,cy+.025*s,color)

def hills(cx,cy,s=.55,color=INK):
    ax.add_patch(Polygon([(cx-.032*s,cy-.02*s),(cx-.012*s,cy+.018*s),(cx+.002*s,cy-.002*s),
                          (cx+.017*s,cy+.015*s),(cx+.034*s,cy-.02*s)],closed=False,fill=False,edgecolor=color,lw=MIN_LINE_PT))
    line(cx-.033*s,cy-.021*s,cx+.035*s,cy-.021*s,color)

def brain(cx,cy,s=.55,color=INK):
    ax.add_patch(Circle((cx-.007*s,cy),.013*s,fill=False,edgecolor=color,lw=MIN_LINE_PT))
    ax.add_patch(Circle((cx+.007*s,cy),.013*s,fill=False,edgecolor=color,lw=MIN_LINE_PT))
    line(cx,cy-.012*s,cx,cy+.012*s,color)

def waves(cx,cy,s=.55,color=INK):
    for r in [.008,.015,.022]: ax.add_patch(Ellipse((cx,cy),2*r*s,2*r*s,fill=False,edgecolor=color,lw=MIN_LINE_PT))

def propagate(cx,cy,color=INK):
    """A small sound source emits expanding wavefronts into the environment."""
    ax.add_patch(Rectangle((cx-.016,cy-.006),.004,.012,facecolor=color,edgecolor=color,lw=MIN_LINE_PT))
    ax.add_patch(Polygon([(cx-.012,cy-.006),(cx-.004,cy-.011),(cx-.004,cy+.011),(cx-.012,cy+.006)],
                         closed=True,facecolor=color,edgecolor=color,lw=MIN_LINE_PT))
    for radius in [.012,.020]:
        ax.add_patch(Arc((cx-.006,cy),2*radius,2*radius,theta1=-58,theta2=58,
                         edgecolor=color,lw=MIN_LINE_PT))

def transduction(cx,cy,color=INK):
    """Show an acoustic waveform converted at a receptor into neural pulses."""
    xx=np.linspace(cx-.016,cx-.004,35)
    yy=cy+.006*np.sin(np.linspace(0,3*np.pi,35))
    points=np.column_stack((xx,yy))
    codes=np.full(len(points),MatplotlibPath.LINETO,dtype=np.uint8); codes[0]=MatplotlibPath.MOVETO
    ax.add_patch(PathPatch(MatplotlibPath(points,codes),fill=False,edgecolor=color,lw=MIN_LINE_PT,
                           capstyle="round",joinstyle="round"))
    line(cx-.001,cy-.012,cx-.001,cy+.012,color)
    line(cx+.003,cy,cx+.009,cy,color)
    for j in range(3):
        px=cx+.010+j*.003
        line(px,cy-.005,px,cy+.005,color)
        dot(px,cy+.006,.0015,color,color)

def speaker(cx,cy,s=.55,color=BLUE):
    ax.add_patch(Rectangle((cx-.025*s,cy-.013*s),.012*s,.026*s,facecolor=color,edgecolor=color,lw=MIN_LINE_PT))
    ax.add_patch(Polygon([(cx-.013*s,cy-.013*s),(cx+.008*s,cy-.029*s),(cx+.008*s,cy+.029*s),(cx-.013*s,cy+.013*s)],
                         closed=True,facecolor=color,edgecolor=color,lw=MIN_LINE_PT))
    for r in [.022,.034]: ax.add_patch(Arc((cx+.006*s,cy),2*r*s,2*r*s,theta1=-55,theta2=55,edgecolor=color,lw=MIN_LINE_PT))

def landscape(cx,cy,w=.10,h=.035,color=GRAY):
    """Draw a compact, colored pasture scene with sky, rolling fields, sun, and tree."""
    x=np.linspace(-.5,.5,48)
    back=.01+.20*np.sin((x+.5)*2*np.pi*1.35+.35)
    front=-.17+.075*np.sin((x+.5)*2*np.pi*1.05+1.0)
    def filled_land(y_curve,base,face):
        upper=np.column_stack((cx+x*w,cy+y_curve*h))
        lower=np.array([[cx+.5*w,cy+base*h],[cx-.5*w,cy+base*h]])
        points=np.vstack((upper,lower))
        codes=np.full(len(points),MatplotlibPath.LINETO,dtype=np.uint8)
        codes[0]=MatplotlibPath.MOVETO; codes[-1]=MatplotlibPath.CLOSEPOLY
        ax.add_patch(PathPatch(MatplotlibPath(points,codes),facecolor=face,edgecolor="none"))
    filled_land(back,-.36,"#B8CDD0")
    filled_land(front,-.36,"#85A975")
    # A small sun and tree silhouette make the scene read immediately as a pasture.
    sx,sy=cx+.35*w,cy+.25*h
    ax.add_patch(Circle((sx,sy),h*.105,facecolor="#E5B85C",edgecolor="none"))
    tx,ty=cx-.34*w,cy-.11*h
    ax.add_patch(Rectangle((tx-.009*w,ty-.16*h),.018*w,.22*h,facecolor="#526F4F",edgecolor="none"))
    ax.add_patch(Circle((tx,ty+.12*h),h*.14,facecolor="#526F4F",edgecolor="none"))
    # Fine foreground marks suggest grass without competing with the model labels.
    for u,v in [(-.06,-.19),(.02,-.22),(.12,-.18)]:
        gx,gy=cx+u*w,cy+v*h
        line(gx,gy,gx-.015*w,gy+.09*h,"#526F4F")
        line(gx,gy,gx+.018*w,gy+.07*h,"#526F4F")

def pasture_icon(cx,cy,s=.55,color=GREEN):
    landscape(cx,cy,.075*s,.043*s,color)

def submitted_icon(name,cx,cy,maxw,maxh,palette=None):
    """Trace a submitted pictogram into resolution-independent vector paths."""
    gray=np.asarray(SUBMITTED_TIFF.crop(SUBMITTED_CROPS[name]))
    border=np.concatenate((gray[0,:],gray[-1,:],gray[:,0],gray[:,-1]))
    background=float(np.median(border))
    thresholds=[max(0,int(background-offset)) for offset in (3,18,42,78,125)]
    masks=[cv2.dilate(np.where(gray<t,255,0).astype(np.uint8),np.ones((3,3),np.uint8),iterations=1)
           for t in thresholds]
    contours_by_level=[]
    for mask in masks:
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if not contours: contours_by_level.append(([],[])); continue
        contours_by_level.append((contours,hierarchy[0]))
    if not contours_by_level[0][0]: raise ValueError(f"No ink found in submitted icon crop: {name}")
    points=np.concatenate([c.reshape(-1,2) for c in contours_by_level[0][0]])
    minx,miny=points.min(axis=0); maxx,maxy=points.max(axis=0)
    spanx=max(1,int(maxx-minx)); spany=max(1,int(maxy-miny))
    icon_aspect=spanx/spany
    normalized_aspect=icon_aspect*HEIGHT_MM/WIDTH_MM
    if maxw/maxh > normalized_aspect:
        h=maxh; w=h*normalized_aspect
    else:
        w=maxw; h=maxw/normalized_aspect
    shades=palette or ["#b5b5b5","#8b8b8b","#626262","#393939",INK]
    for level,(contours,hierarchy) in enumerate(contours_by_level):
        if not contours: continue
        vertices=[]; codes=[]
        for i,contour in enumerate(contours):
            if name == "playback_cow" and cv2.contourArea(contour) < 10:
                continue
            pts=contour.reshape(-1,2).astype(float)
            if len(pts)<3: continue
            # Nested contours alternate orientation so transparent holes remain holes.
            depth=0; parent=hierarchy[i][3]
            while parent!=-1:
                depth+=1; parent=hierarchy[parent][3]
            xnorm=(pts[:,0]-minx)/spanx
            ynorm=(pts[:,1]-miny)/spany
            xy=np.column_stack((cx-w/2+xnorm*w,cy+h/2-ynorm*h))
            area=.5*np.sum(xy[:,0]*np.roll(xy[:,1],-1)-np.roll(xy[:,0],-1)*xy[:,1])
            want_positive=(depth%2==0)
            if (area>0) != want_positive: xy=xy[::-1]
            vertices.extend(xy.tolist()+[xy[0].tolist()])
            codes.extend([MatplotlibPath.MOVETO]+[MatplotlibPath.LINETO]*(len(xy)-1)+[MatplotlibPath.CLOSEPOLY])
        path=MatplotlibPath(np.asarray(vertices),np.asarray(codes,dtype=np.uint8))
        ax.add_patch(PathPatch(path,facecolor=shades[level],edgecolor="none",
                               antialiased=True,zorder=2.1+level*.1))

# Header and four domains: preserve the submitted row and its visual rhythm.
text(.5,.978,"AFFECTIVE INFORMATION TRANSMISSION",17,"bold")
text(.5,.944,"Cross-species acoustic communication: evidence for multiscale acoustic–affective organization",11,"normal",style="italic")
text(.5,.911,"BIOLOGICAL, CULTURAL AND ENVIRONMENTAL SOUND CONTEXTS",11,"bold")
domain_x=[.025,.272,.519,.766]; domain_w=.213; dy=.786; dh=.105
domains=[("ANIMAL VOCALS","Bovine reference calls","Pastoral settings",GREEN,PALE_GREEN,"cow"),
         ("HUMAN VOICES","Pastoral / non-pastoral","Songs / calls",CORAL,PALE_CORAL,"person"),
         ("MUSICAL SYSTEMS","Compositional","Obonuco FASE",BLUE,PALE_BLUE,"music"),
         ("SOUND ENVIRONMENTS","Ecological","Instrumental / pasture",PURPLE,PALE_PURPLE,"hills")]
for x,(title,sub,note,color,face,icon) in zip(domain_x,domains):
    box(x,dy,domain_w,dh,color,face)
    text(x+domain_w/2,dy+.092,title,11,"bold")
    text(x+domain_w/2,dy+.070,sub,11)
    if icon=="hills":
        landscape(x+domain_w/2,dy+.044,.115,.045)
    else:
        asset={"cow":"animal_vocalizations","person":"human_vocal_practice",
               "music":"musical_score"}[icon]
        submitted_icon(asset,x+domain_w/2,dy+.041,.150,.030)
    text(x+domain_w/2,dy+.017,note,11,"bold")
arrow(domain_x[0]+domain_w,dy+.052,domain_x[1],dy+.052,both=True)
line(domain_x[1]+domain_w,dy+.052,domain_x[2],dy+.052)
line(domain_x[2]+domain_w,dy+.052,domain_x[3],dy+.052)

# A full-width pathway makes the literature-grounded biological steps part of the model.
text(.5,.758,"ACOUSTIC COMMUNICATION PROCESS",12,"bold")
text(.5,.735,"A literature-grounded pathway engaged by each acoustic interaction",10.67,style="italic",color=GRAY)
centers=[.105,.3025,.500,.6975,.895]; node_y=.693
labels=["SOURCE","PROPAGATE","TRANSDUCE","INTEGRATE","RESPOND"]
subtexts=["Vocal / musical signal","Sound propagation","Sound to neural signal","Receiver context","Behavior / physiology"]
for i,cx in enumerate(centers):
    dot(cx,node_y,.020)
    if i==0: person(cx,node_y,.46)
    elif i==1: propagate(cx,node_y)
    elif i==2: transduction(cx,node_y)
    elif i==3: brain(cx,node_y,.45)
    else: submitted_icon("playback_cow",cx,node_y,.030,.022)
    if i<len(centers)-1: arrow(cx+.022,node_y,centers[i+1]-.022,node_y)
    text(cx,.658,labels[i],10.67,"bold")
    text(cx,.627,subtexts[i],10.67)

# Paired containers show temporal scales and FASE layers as integrated structures.
box(.025,.430,.455,.160,INK,"white",radius=.008)
text(.2525,.607,"TEMPORAL SCALES",11,"bold")
scale_cards=[(.043,"MICRO","Pitch / timbre","ms–s",GREEN,PALE_GREEN),
             (.191,"MESO","Motifs / calls","s–min",BLUE,PALE_BLUE),
             (.339,"MACRO","Integrated","min–h",PURPLE,PALE_PURPLE)]
for x,title,desc,span,color,face in scale_cards:
    box(x,.476,.132,.085,color,face,radius=.005)
    text(x+.066,.535,title,10.67,"bold")
    text(x+.066,.511,desc,10.67)
    text(x+.066,.488,span,10.67)
text(.2525,.447,"Complementary windows within one sound sequence",10.67,style="italic",color=GRAY)

box(.520,.430,.455,.160,PURPLE,"white",radius=.008)
text(.7475,.607,"FASE COMPOSITION",11,"bold",color=PURPLE)
fase_layers=[(.535,.511,"BIOACOUSTIC","Bovine calls via OAAS",GREEN,PALE_GREEN),
             (.755,.511,"PSYCHOACOUSTIC","Perceptual design",BLUE,PALE_BLUE),
             (.535,.447,"MUSICAL","Rhythm and melody",PURPLE,PALE_PURPLE),
             (.755,.447,"ECOACOUSTIC","Pasture soundscape",CORAL,PALE_CORAL)]
for x,y,title,body,color,face in fase_layers:
    box(x,y,.205,.057,color,face,radius=.004)
    text(x+.1025,y+.039,title,10.67,"bold")
    text(x+.1025,y+.015,body,10.67)

# The interpretive hypothesis is the synthesis between organized sound and convergent evidence.
ax.add_patch(Ellipse((.5,.352),.800,.120,facecolor=PALE_PURPLE,edgecolor=PURPLE,linewidth=MIN_LINE_PT,
                     linestyle=(0,(4,2))))
text(.5,.389,"CENTRAL HYPOTHESIS | MULTISCALE",11,"bold",color=PURPLE)
text(.5,.359,"Biologically contextualized multidimensional acoustic configurations",11)
text(.5,.337,"show scale-dependent organization across bovine and human acoustic domains",11)
text(.5,.315,"and remain functionally relevant to bovine receivers",11)

# Two distinct evidence streams converge on the hypothesis above.
text(.5,.274,"CONVERGENT STUDY EVIDENCE",12,"bold")
box(.025,.125,.462,.112,INK,"white")
box(.513,.125,.462,.112,INK,"white")
speaker(.080,.181,.75,BLUE)
submitted_icon("playback_cow",.132,.171,.031,.033,
               palette=["#B7D8BF","#92C6A0","#70B481",GREEN,"#367547"])
text(.310,.210,"PASTORAL + PRE-FASE EVIDENCE",11,"bold")
text(.310,.177,"Bovine calls + recorded human songs",11)
text(.310,.146,"Cattle behavior and HRV",11)
pasture_icon(.568,.181,.70,CORAL)
speaker(.607,.181,.48,BLUE)
text(.778,.210,"COMPOSED FASE PLAYBACK",11,"bold")
text(.778,.177,"Obonuco FASE presented to cattle",11)
text(.778,.146,"Behavior strongest; HRV complementary",11)
# Dashed evidence and analytic inputs meet the interpretive hypothesis.
arrow(.215,.237,.215,.316,color=PURPLE,lw=MIN_LINE_PT,ls="--")
arrow(.785,.237,.785,.316,color=PURPLE,lw=MIN_LINE_PT,ls="--")
arrow(.252,.430,.365,.410,color=PURPLE,lw=MIN_LINE_PT,ls="--")
arrow(.748,.430,.635,.410,color=PURPLE,lw=MIN_LINE_PT,ls="--")

# Compact symbol key; line semantics are explicit and non-causal where appropriate.
box(.025,.034,.950,.047,INK,"white",radius=.004)
legend=[(.075,"Flow","arrow"),(.300,"Exchange","both"),(.535,"Compare","solid"),
        (.765,"Scale / FASE","dash")]
for x,label,kind in legend:
    if kind=="arrow": arrow(x,.057,x+.035,.057)
    elif kind=="both": arrow(x,.057,x+.035,.057,both=True)
    elif kind=="solid": line(x,.057,x+.035,.057)
    elif kind=="dash": line(x,.057,x+.035,.057,ls="--")
    text(x+.043,.057,label,11,ha="left")

# Check actual text boxes after layout so collisions are caught before delivery.
fig.canvas.draw()
renderer=fig.canvas.get_renderer()
texts=[t for t in ax.texts if t.get_visible()]
collisions=[]
for i,a in enumerate(texts):
    ba=a.get_window_extent(renderer)
    for b in texts[i+1:]:
        bb=b.get_window_extent(renderer)
        if ba.overlaps(bb): collisions.append((a.get_text().replace("\n"," / "),b.get_text().replace("\n"," / ")))
if collisions:
    raise RuntimeError("Overlapping text labels: " + repr(collisions))
axes_bbox=ax.get_window_extent(renderer)
outside=[]
for t in texts:
    b=t.get_window_extent(renderer)
    if b.x0 < axes_bbox.x0 or b.x1 > axes_bbox.x1 or b.y0 < axes_bbox.y0 or b.y1 > axes_bbox.y1:
        outside.append(t.get_text().replace("\n"," / "))
if outside:
    raise RuntimeError("Text outside figure bounds: " + repr(outside))

fig.savefig(OUT_SVG,format="svg",facecolor="white",metadata={"Date": None})
fig.savefig(OUT_PNG,format="png",dpi=DPI,facecolor="white")
svg_data=OUT_SVG.read_text(encoding="utf-8")
buffer=io.BytesIO()
fig.savefig(buffer,format="tiff",dpi=TIFF_DPI,facecolor="white")
buffer.seek(0)
with Image.open(buffer) as im:
    im.convert("RGB").save(OUT_TIF,format="TIFF",compression="tiff_lzw",dpi=(TIFF_DPI,TIFF_DPI))
with Image.open(OUT_TIF) as im:
    qc={"figure":STEM,"source_script":Path(__file__).name,
        "canvas_mm":{"width":WIDTH_MM,"height":HEIGHT_MM},"dpi":TIFF_DPI,"preview_dpi":DPI,
        "pixel_dimensions":{"width":im.width,"height":im.height},"mode":im.mode,
        "format":im.format,"compression":im.info.get("compression","tiff_lzw"),
        "font_sizes_pt":sorted({float(t.get_fontsize()) for t in texts}),
        "minimum_font_pt":MIN_FONT_PT,"minimum_graphic_stroke_pt":MIN_LINE_PT,
        "text_collision_check":"pass","text_canvas_bounds_check":"pass",
        "illustration_rendering":"editable vector shapes; selected source icons traced; landscapes redrawn as colored vector scenes",
        "svg_path_count":svg_data.count("<path"),"embedded_svg_raster_images":svg_data.count("<image"),
        "editable_vector_svg":OUT_SVG.name,"raster_preview":OUT_PNG.name}
OUT_QC.write_text(json.dumps(qc,indent=2),encoding="utf-8")
plt.close(fig)
print(f"Created editable SVG, {DPI} dpi PNG preview, and {TIFF_DPI} dpi RGB/LZW TIFF at {WIDTH_MM} × {HEIGHT_MM} mm")
print(f"Files: {OUT_SVG.name}, {OUT_PNG.name}, {OUT_TIF.name}, {OUT_QC.name}")
