"""Regenerate every picture in the README from the raw parquet files.

Usage (from the repo root, after scripts/fetch_data.py):
    python docs/make_figures.py

Writes PNGs to docs/images/. Takes a few minutes, mostly reading the parquets.
The four result plots (10 to 13) are exported from 4Notebook_Modelling.ipynb, not made here.
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
os.chdir(Path(__file__).resolve().parent)          # work inside docs/
Path('images').mkdir(exist_ok=True)
Path('_cache').mkdir(exist_ok=True)

# ---------- step 1: per-pixel tables ----------
import pandas as pd, numpy as np
D=str(ROOT/'data')+'/'
EV={'Madrid':1960,'Amsterdam':1945}
for city,f in [('Madrid','madrid_train.parquet'),('Amsterdam','amsterdam_data.parquet')]:
    df=pd.read_parquet(D+f,columns=['px_key','py_key','year','coverage','weighted_mean_year','Red_1','NIR_1','SWIR1_1','qa_valid_1','Red_2','NIR_2','qa_valid_2','Red_3','NIR_3','qa_valid_3'])
    pix=df.drop_duplicates(['px_key','py_key'])[['px_key','py_key','coverage','weighted_mean_year']].copy()
    pix['age_class']=pd.cut(pix.weighted_mean_year,[-np.inf,EV[city],1984,2004,np.inf],labels=[1,2,3,4]).astype(int)
    pix.to_parquet(f'_cache/pix_{city}.parquet'); print(city,len(pix),pix.age_class.value_counts().sort_index().tolist(), pix.coverage.describe()[['min','50%','max']].tolist())
    # NDVI per pixel-year from first valid obs
    R=np.where(df.qa_valid_1.fillna(False),df.Red_1,np.where(df.qa_valid_2.fillna(False),df.Red_2,np.where(df.qa_valid_3.fillna(False),df.Red_3,np.nan)))
    N=np.where(df.qa_valid_1.fillna(False),df.NIR_1,np.where(df.qa_valid_2.fillna(False),df.NIR_2,np.where(df.qa_valid_3.fillna(False),df.NIR_3,np.nan)))
    ts=df[['px_key','py_key','year','weighted_mean_year']].copy(); ts['NDVI']=(N-R)/(N+R)
    ts.to_parquet(f'_cache/ts_{city}.parquet')

# ---------- step 2: figures ----------
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Patch
from scipy.spatial import cKDTree, ConvexHull
from sklearn.model_selection import GroupKFold

AGE={1:'#0d366b',2:'#256abf',3:'#5598e7',4:'#86b6ef'}
LAB={'Madrid':{1:'Pre-1960',2:'1960–1984',3:'1984–2004',4:'2004–2024'},
     'Amsterdam':{1:'Pre-1945',2:'1945–1984',3:'1984–2004',4:'2004–2024'}}
INK='#0b0b0b'; INK2='#52514e'; MUTED='#8a8984'; GRID='#e6e5e1'; BG='#ffffff'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK,'axes.labelcolor':INK2,
  'xtick.color':INK2,'ytick.color':INK2,'axes.edgecolor':GRID,'figure.facecolor':BG,
  'axes.facecolor':BG,'savefig.facecolor':BG,'font.size':11})
P={c:pd.read_parquet(f'_cache/pix_{c}.parquet') for c in ['Madrid','Amsterdam']}

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]

# ---------- 1. Hero: the two cities ----------
fig,axes=plt.subplots(1,2,figsize=(14,7.4),gridspec_kw={'width_ratios':[1.25,1]})
for ax,c in zip(axes,['Madrid','Amsterdam']):
    p=P[c].sample(frac=1,random_state=0)
    ax.scatter(p.px_key,p.py_key,c=p.age_class.map(AGE),s=0.35 if c=='Madrid' else 0.8,marker='s',linewidths=0)
    ax.set_aspect('equal'); clean(ax)
    ax.set_title(f'{c}  ·  {len(p):,} pixels', loc='left', fontsize=14, fontweight='bold', pad=8)
    h=[Patch(color=AGE[k],label=f'Class {k}  {LAB[c][k]}') for k in [1,2,3,4]]
    ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.01),ncol=2,frameon=False,fontsize=10)
    # 5 km scale bar (30 m pixels)
    x0,y0=p.px_key.max()-200,p.py_key.min()+15
    if c=='Amsterdam': x0=p.px_key.min()+10
    ax.plot([x0,x0+5000/30],[y0,y0],color=INK,lw=2.5); ax.text(x0+2500/30,y0+12,'5 km',ha='center',fontsize=9,color=INK2)
fig.suptitle('Every dot is a 30 m Landsat pixel, coloured by the age of the buildings inside it',x=0.01,ha='left',fontsize=15,color=INK2,y=0.99)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(bbox_inches='tight',pad_inches=0.2,fname='images/01_cities_by_age.png',dpi=130); plt.close()

# ---------- 2. A pixel is mostly not a building ----------
rng=np.random.default_rng(3)
fig,axes=plt.subplots(1,2,figsize=(11,5.4))
specs=[('A pixel at 15% coverage',[(0.08,0.62,0.30,0.25)],True),('A pixel at 80% coverage',[(0.04,0.42,0.42,0.54),(0.52,0.42,0.44,0.54),(0.04,0.04,0.92,0.30)],False)]
for ax,(t,blds,green) in zip(axes,specs):
    ax.add_patch(Rectangle((0,0),1,1,fc='#dfe9d6' if green else '#e9e7e1',ec='none'))
    if green:
        ax.add_patch(Rectangle((0,0.38),1,0.13,fc='#b9b7b0',ec='none'))  # road
        ax.plot([0,1],[0.445,0.445],color='white',lw=2,ls=(0,(6,6)))
        for _ in range(16):
            x,y=rng.uniform(.05,.95),rng.uniform(.05,.33); ax.add_patch(Circle((x,y),rng.uniform(.03,.06),fc='#7fa36b',ec='none',alpha=.9))
        for _ in range(5):
            x,y=rng.uniform(.45,.95),rng.uniform(.58,.95); ax.add_patch(Circle((x,y),rng.uniform(.03,.05),fc='#7fa36b',ec='none',alpha=.9))
    for (x,y,w,h) in blds:
        ax.add_patch(Rectangle((x+.015,y-.02),w,h,fc='#c9c7c0',ec='none'))  # shadow
        ax.add_patch(Rectangle((x,y),w,h,fc=AGE[2],ec='white',lw=1.5))
    ax.add_patch(Rectangle((0,0),1,1,fc='none',ec=INK,lw=2.5))
    ax.set_xlim(-.05,1.05); ax.set_ylim(-.12,1.05); ax.set_aspect('equal'); clean(ax)
    ax.set_title(t,fontsize=14,fontweight='bold',loc='left')
    ax.text(0.5,-0.08,'30 m × 30 m',ha='center',color=INK2,fontsize=10)
axes[0].text(0.23,0.745,'building',ha='center',va='center',color='white',fontsize=10,fontweight='bold')
axes[0].text(0.75,0.445,'road',ha='center',va='center',color=INK,fontsize=10,bbox=dict(fc='#b9b7b0',ec='none'))
axes[0].text(0.5,0.18,'gardens, trees, shadows',ha='center',color=INK,fontsize=10,bbox=dict(fc='white',ec='none',alpha=.8))
fig.text(0.01,0.01,'The satellite sees one blended colour per square. On the left, most of that colour is not building at all.',color=INK2,fontsize=10.5)
fig.tight_layout(rect=[0,0.04,1,1]); fig.savefig(bbox_inches='tight',pad_inches=0.2,fname='images/02_pixel_coverage.png',dpi=130); plt.close()

# ---------- 3. Coverage distribution ----------
fig,ax=plt.subplots(figsize=(10,4.2))
bins=np.arange(0.145,1.0051,0.05)
for c,col in [('Madrid','#2a78d6'),('Amsterdam','#eb6834')]:
    w=np.ones(len(P[c]))/len(P[c])*100
    ax.hist(P[c].coverage,bins=bins,weights=w,histtype='step',lw=2,color=col,label=f'{c} (median {P[c].coverage.median():.0%})')
ax.set_xlabel('Building coverage of the pixel'); ax.set_ylabel('% of pixels')
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis='y',color=GRID); ax.spines[['top','right']].set_visible(False)
ax.legend(frameon=False); ax.set_title('How much of each pixel is actually building?',loc='left',fontweight='bold',fontsize=13)
fig.tight_layout(); fig.savefig(bbox_inches='tight',pad_inches=0.2,fname='images/03_coverage_hist.png',dpi=130); plt.close()

# ---------- 4. The satellite watches construction (event-aligned NDVI) ----------
fig,axes=plt.subplots(1,2,figsize=(13,4.6),sharey=True)
for ax,c in zip(axes,['Madrid','Amsterdam']):
    ts=pd.read_parquet(f'_cache/ts_{c}.parquet')
    ts=ts[(ts.weighted_mean_year>=1992)&(ts.weighted_mean_year<=2016)&ts.NDVI.between(-1,1)]
    ts['rel']=ts.year-ts.weighted_mean_year.round()
    # normalise each pixel by its own pre-construction mean
    pre=ts[ts.rel.between(-8,-3)].groupby(['px_key','py_key']).NDVI.mean().rename('pre')
    ts=ts.join(pre,on=['px_key','py_key']).dropna(subset=['pre'])
    ts['d']=ts.NDVI-ts.pre
    g=ts[ts.rel.between(-8,10)].groupby('rel').d
    m,q1,q3=g.median(),g.quantile(.25),g.quantile(.75)
    ax.fill_between(m.index,q1,q3,color='#2a78d6',alpha=.15,lw=0,label='middle 50% of pixels')
    ax.plot(m.index,m.values,color='#2a78d6',lw=2,marker='o',ms=4,label='median pixel')
    ax.axvline(0,color=INK,lw=1,ls='--'); ax.axhline(0,color=MUTED,lw=.8)
    ax.text(0.3,ax.get_ylim()[1] if False else 0.012,'recorded build year',fontsize=9,color=INK2)
    ax.set_title(f'{c}: {ts[["px_key","py_key"]].drop_duplicates().shape[0]:,} pixels built 1992–2016',loc='left',fontweight='bold',fontsize=12)
    ax.set_xlabel('Years relative to recorded construction year'); ax.grid(color=GRID); ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Change in greenness (NDVI)\nvs. before construction'); axes[0].legend(frameon=False,loc='lower left')
fig.suptitle('Greenness drops when buildings go up. This is the signal the model hunts for.',x=0.01,ha='left',fontsize=13.5,color=INK2)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(bbox_inches='tight',pad_inches=0.2,fname='images/04_construction_signal.png',dpi=130); plt.close()

# ---------- 5. Neighbourhood scales ----------
m=P['Madrid']; xy=m[['px_key','py_key']].to_numpy(float); tree=cKDTree(xy)
cx,cy=np.median(xy[:,0]),np.median(xy[:,1])
# pick a pixel near centre-ish
q=xy[tree.query([cx+40,cy-20])[1]]
_,idx=tree.query(q,k=2048)
fig,ax=plt.subplots(figsize=(9,8.2))
win=70
sel=(abs(xy[:,0]-q[0])<win)&(abs(xy[:,1]-q[1])<win)
ax.scatter(xy[sel,0],xy[sel,1],s=6,marker='s',c='#d9d8d3',lw=0)
shade=['#0d366b','#1c5cab','#3987e5','#86b6ef','#cde2fb'][::-1]
ks=[2048,1024,256,64,16]
handles=[]
for k,col in zip(ks,shade):
    pts=xy[idx[:k]]; ax.scatter(pts[:,0],pts[:,1],s=6,marker='s',c=col,lw=0)
    hull=ConvexHull(pts); hp=pts[hull.vertices]; hp=np.vstack([hp,hp[0]])
    ax.plot(hp[:,0],hp[:,1],color=INK,lw=.8,alpha=.6)
    rad=np.max(np.hypot(*(pts-q).T))*30
    handles.append(Patch(fc=col,ec=INK,lw=.5,label=f'k = {k:>5,}   reaches ~{int(round(rad,-1)):,} m'))
ax.scatter([q[0]],[q[1]],s=90,marker='*',c='#eb6834',edgecolors='white',linewidths=.8,zorder=5)
ax.legend(handles=handles[::-1],loc='upper right',frameon=True,framealpha=.95,edgecolor=GRID,fontsize=10,prop={'family':'DejaVu Sans Mono','size':9.5},title='nearest neighbours',title_fontsize=10)
ax.annotate('the pixel we\nwant to date',xy=q,xytext=(q[0]+30,q[1]-48),va='center',fontsize=10,color=INK,arrowprops=dict(arrowstyle='->',color=INK,lw=1))
ax.set_xlim(q[0]-win,q[0]+win); ax.set_ylim(q[1]-win,q[1]+win); ax.set_aspect('equal'); clean(ax)
ax.plot([q[0]-win+5,q[0]-win+5+1000/30],[q[1]-win+5]*2,color=INK,lw=2.5); ax.text(q[0]-win+5+500/30,q[1]-win+7,'1 km',ha='center',fontsize=9,color=INK2)
ax.set_title('Asking the neighbours: five rings of nearest building pixels (real Madrid grid)',loc='left',fontsize=12.5,fontweight='bold')
fig.tight_layout(); fig.savefig(bbox_inches='tight',pad_inches=0.2,fname='images/05_neighbourhood_scales.png',dpi=130); plt.close()

# ---------- 6. Random split vs blocked split ----------
fig,axes=plt.subplots(1,2,figsize=(13,6.4))
win=60; c0=q
sel=(abs(xy[:,0]-c0[0])<win)&(abs(xy[:,1]-c0[1])<win)
sub=xy[sel]
r=np.random.default_rng(1).random(len(sub))<0.1
bx=np.floor(m.px_key.to_numpy()/10).astype(int); by=np.floor(m.py_key.to_numpy()/10).astype(int)
grp=np.char.add(np.char.add(bx.astype(str),'_'),by.astype(str))
fold=np.empty(len(m),int)
for f,(_,va) in enumerate(GroupKFold(10).split(xy,groups=grp)): fold[va]=f
b=fold[sel]==0
for ax,mask,t in [(axes[0],r,'Random pixel split (the trap)'),(axes[1],b,'300 m block split (what we used)')]:
    ax.scatter(sub[~mask,0],sub[~mask,1],s=9,marker='s',c='#c9d9ee',lw=0,label='training pixels')
    ax.scatter(sub[mask,0],sub[mask,1],s=9,marker='s',c='#eb6834',lw=0,label='validation pixels')
    ax.set_xlim(c0[0]-win,c0[0]+win); ax.set_ylim(c0[1]-win,c0[1]+win); ax.set_aspect('equal'); clean(ax)
    ax.set_title(t,loc='left',fontsize=13,fontweight='bold')
for gx in range(int((c0[0]-win)//10)*10,int(c0[0]+win)+10,10): axes[1].axvline(gx-.5,color='white',lw=.6)
for gy in range(int((c0[1]-win)//10)*10,int(c0[1]+win)+10,10): axes[1].axhline(gy-.5,color='white',lw=.6)
axes[0].legend(loc='lower left',frameon=True,framealpha=.95,edgecolor=GRID,markerscale=2.5)
axes[0].text(0.5,-0.06,'Every test pixel has a training twin next door. Easy marks, fake confidence.',transform=axes[0].transAxes,ha='center',fontsize=10,color=INK2)
axes[1].text(0.5,-0.06,'Whole neighbourhoods are hidden. The model has to genuinely generalise.',transform=axes[1].transAxes,ha='center',fontsize=10,color=INK2)
fig.tight_layout(rect=[0,0.03,1,1]); fig.savefig(bbox_inches='tight',pad_inches=0.2,fname='images/06_blocked_cv.png',dpi=130); plt.close()

# ---------- 7. Results bars ----------
metrics=['Madrid CV\n(10-fold, blocked)','Amsterdam\nzero-shot (0 labels)','Amsterdam\n25 labels per class']
vals={'Provided baseline':[0.6179,0.3427,0.5986],'Our checkpoint':[0.6849,0.4787,0.6340],'Final pipeline':[0.6976,0.5192,0.6499]}
cols={'Provided baseline':'#b9b7b0','Our checkpoint':'#86b6ef','Final pipeline':'#1c5cab'}
fig,ax=plt.subplots(figsize=(11,4.8)); w=0.26; x=np.arange(3)
for i,(k,v) in enumerate(vals.items()):
    bars=ax.bar(x+(i-1)*w,v,w-0.02,color=cols[k],label=k)
    for bb,val in zip(bars,v): ax.text(bb.get_x()+bb.get_width()/2,val+0.008,f'{val:.3f}',ha='center',fontsize=9.5,color=INK if k=='Final pipeline' else INK2,fontweight='bold' if k=='Final pipeline' else 'normal')
ax.set_xticks(x); ax.set_xticklabels(metrics); ax.set_ylim(0,0.8); ax.set_ylabel('Macro F1 (higher is better)')
ax.grid(axis='y',color=GRID); ax.set_axisbelow(True); ax.spines[['top','right','left']].set_visible(False)
ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.13))
ax.text(1+w,0.575,'+52% vs\nbaseline',ha='center',fontsize=9.5,fontweight='bold',color='#1c5cab')
fig.tight_layout(); fig.savefig(bbox_inches='tight',pad_inches=0.2,fname='images/07_results.png',dpi=130); plt.close()

# ---------- 8. Prototype transfer (illustration) ----------
rng=np.random.default_rng(7)
cent={1:(-2.2,1.4),2:(-0.6,-0.2),3:(1.0,1.2),4:(2.2,-1.0)}
fig,axes=plt.subplots(1,2,figsize=(13,5.6),sharex=True,sharey=True)
for ax,shift,t in [(axes[0],np.array([0,0]),'Madrid: the model learned what each age "looks like"'),(axes[1],np.array([0.9,-0.7]),'Amsterdam: same shapes, shifted. A few labels re-centre them')]:
    for k,(cx_,cy_) in cent.items():
        pts=rng.normal([cx_,cy_],0.55,(160,2))+shift
        ax.scatter(pts[:,0],pts[:,1],s=10,c=AGE[k],alpha=.3 if ax is axes[1] else .8,lw=0)
        if ax is axes[1]:
            sup=pts[:5]; ax.scatter(sup[:,0],sup[:,1],s=40,c=AGE[k],edgecolors='white',linewidths=1,zorder=3)
            pr=sup.mean(0); ax.scatter(*pr,s=380,marker='*',c=AGE[k],edgecolors=INK,linewidths=1.2,zorder=4)
            ax.scatter(cx_,cy_,s=160,marker='X',facecolors='none',edgecolors=MUTED,linewidths=1.2,zorder=3)
            ax.annotate('',xy=pr,xytext=(cx_,cy_),arrowprops=dict(arrowstyle='->',color=MUTED,lw=1.2))
    ax.set_title(t,loc='left',fontsize=12,fontweight='bold'); clean(ax); ax.add_patch(Rectangle((0,0),1,1,transform=ax.transAxes,fc='none',ec=GRID))
qpt=np.array([0.9,0.2]); axes[1].scatter(*qpt,s=110,marker='o',c='white',edgecolors='#eb6834',linewidths=2.2,zorder=5)
axes[1].annotate('unlabelled pixel → nearest star wins',xy=qpt,xytext=(qpt[0]-0.6,qpt[1]+2.1),fontsize=10,arrowprops=dict(arrowstyle='->',color=INK,lw=1))
h=[Patch(color=AGE[k],label=f'Class {k}') for k in [1,2,3,4]]+[plt.Line2D([],[],marker='*',ls='',ms=14,mfc='white',mec=INK,label='prototype (mean of 5 labels)'),plt.Line2D([],[],marker='X',ls='',ms=10,mfc='none',mec=MUTED,label="where Madrid's centre was")]
axes[1].legend(handles=h,loc='lower left',fontsize=9,frameon=True,framealpha=.95,edgecolor=GRID)
fig.text(0.01,0.01,'Illustration in 2D. The real thing happens in 204 dimensions, which is harder to draw and much harder to fit on a README.',color=INK2,fontsize=10)
fig.tight_layout(rect=[0,0.04,1,1]); fig.savefig(bbox_inches='tight',pad_inches=0.2,fname='images/08_prototype_transfer.png',dpi=130); plt.close()


# ---------- figure 9 ----------
fig,axes=plt.subplots(2,1,figsize=(12,6.6),sharex=True)
bins=np.arange(1850,2026,5)
for ax,c in zip(axes,['Madrid','Amsterdam']):
    p=P[c]
    for k in [1,2,3,4]:
        ax.hist(p.weighted_mean_year[p.age_class==k],bins=bins,color=AGE[k],label=f'Class {k}  {LAB[c][k]}',edgecolor='white',linewidth=.6)
    ax.axvspan(1945,1960,color='#fbe3d8',lw=0,zorder=0)
    ev=1960 if c=='Madrid' else 1945
    for b in [ev,1984,2004]: ax.axvline(b,color=INK,lw=1,ls='--')
    ax.set_title(f'{c}: Class 1 ends in {ev}',loc='left',fontweight='bold',fontsize=12)
    ax.legend(frameon=False,loc='upper left',fontsize=9.5); ax.set_ylabel('pixels')
    ax.grid(axis='y',color=GRID); ax.set_axisbelow(True); ax.spines[['top','right']].set_visible(False)
share=((P['Amsterdam'].weighted_mean_year>=1945)&(P['Amsterdam'].weighted_mean_year<1960)).mean()
axes[1].text(1952.5,axes[1].get_ylim()[1]*0.92,f'the 1945–1960 trap\n{share:.1%} of Amsterdam',ha='center',va='top',fontsize=10,color='#b8481d',fontweight='bold',bbox=dict(fc='white',ec='none',alpha=.85))
axes[1].set_xlabel('Area-weighted mean construction year of the pixel')
fig.suptitle('Same class names, different meanings: a 1950 building is Class 1 in Madrid but Class 2 in Amsterdam',x=0.01,ha='left',fontsize=13,color=INK2)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig('images/09_class_boundaries.png',dpi=130,bbox_inches='tight',pad_inches=0.2)


# ---------- figure 14: KD-tree splits ----------
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
INK='#0b0b0b'; INK2='#52514e'; GRID='#e6e5e1'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK})
m=pd.read_parquet('_cache/pix_Madrid.parquet'); xy=m[['px_key','py_key']].to_numpy(float)
c=np.median(xy,0)+[40,-20]; w=40
pts=xy[(abs(xy[:,0]-c[0])<w)&(abs(xy[:,1]-c[1])<w)]
DEPTH=4; cols=['#0d366b','#256abf','#5598e7','#86b6ef']; lws=[2.6,2.0,1.5,1.2]
fig,axes=plt.subplots(1,2,figsize=(13,6.4))
def split(ax,p,box,d,maxd):
    if d>=maxd or len(p)<2: return
    ax_i=d%2; med=np.median(p[:,ax_i])
    ln=med+0.5
    if ax_i==0: ax.plot([ln,ln],[box[2],box[3]],color=cols[d],lw=lws[d])
    else: ax.plot([box[0],box[1]],[ln,ln],color=cols[d],lw=lws[d])
    L=p[p[:,ax_i]<=med]; R=p[p[:,ax_i]>med]
    bl=list(box); br=list(box)
    if ax_i==0: bl[1]=med+.5; br[0]=med+.5
    else: bl[3]=med+.5; br[2]=med+.5
    split(ax,L,bl,d+1,maxd); split(ax,R,br,d+1,maxd)
box=[c[0]-w,c[0]+w,c[1]-w,c[1]+w]
for ax,md,t in [(axes[0],2,'Levels 1 and 2: split at the median x, then the median y'),(axes[1],DEPTH,'Levels 1 to 4: each box now holds about 1/16 of the pixels')]:
    ax.scatter(pts[:,0],pts[:,1],s=7,marker='s',c='#c9c7c0',lw=0)
    split(ax,pts,box,0,md)
    ax.set_xlim(box[0],box[1]); ax.set_ylim(box[2],box[3]); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([]); [s.set_color(GRID) for s in ax.spines.values()]
    ax.set_title(t,loc='left',fontsize=11.5,fontweight='bold')
# query example on right
tree=cKDTree(pts); q=pts[tree.query(c)[1]]; _,idx=tree.query(q,k=64)
axes[1].scatter(pts[idx,0],pts[idx,1],s=7,marker='s',c='#eb6834',lw=0,zorder=3)
axes[1].scatter(*q,s=80,marker='*',c=INK,zorder=4)
axes[1].annotate('query pixel and its 64 nearest\nneighbours (they cross box edges,\nso nearby boxes are checked too)',xy=q,xytext=(q[0]+8,q[1]+22),fontsize=10,arrowprops=dict(arrowstyle='->',color=INK),bbox=dict(fc='white',ec='none',alpha=.9))
h=[plt.Line2D([],[],color=cols[i],lw=lws[i],label=f'level {i+1} ({"x" if i%2==0 else "y"} split)') for i in range(DEPTH)]
axes[1].legend(handles=h,loc='lower right',frameon=True,framealpha=.95,edgecolor=GRID,fontsize=9)
axes[0].plot([box[0]+3,box[0]+3+500/30],[box[2]+3]*2,color=INK,lw=2.5); axes[0].text(box[0]+3+250/30,box[2]+4.5,'500 m',ha='center',fontsize=9,color=INK2)
fig.tight_layout(); fig.savefig('images/14_kdtree_splits.png',dpi=130,bbox_inches='tight',pad_inches=0.2)

print('Figures written to docs/images/')
