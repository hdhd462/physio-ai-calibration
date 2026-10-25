import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA_DIR, PROJ_DIR, FIG_DIR, SUB_DIR, PATCH_DIR, NHANES_DIR, PROJ1_DIR, SHAOXING_DIR, PTBXL_DIR
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

rcParams["font.family"]="Arial"; rcParams["pdf.fonttype"]=42

f=np.load(r"ecg_data\fusion3_probs.npz")
y_int,p_int=f["y_te"],f["int"]; y_ext,p_ext=f["y_ext"],f["ext"]
lg=lambda q: np.log(np.clip(q,1e-6,1-1e-6)/(1-np.clip(q,1e-6,1-1e-6)))
p_pl=LogisticRegression(C=1e6).fit(lg(p_ext).reshape(-1,1),y_ext).predict_proba(lg(p_ext).reshape(-1,1))[:,1]
fr,mp=calibration_curve(y_int,p_int,n_bins=10,strategy="quantile")
fr2,mp2=calibration_curve(y_ext,p_ext,n_bins=10,strategy="quantile")
fr3,mp3=calibration_curve(y_ext,p_pl,n_bins=10,strategy="quantile")

C_INT="#2C3E50"; C_EXT="#E67E22"; C_FIX="#27AE60"; C_BAD="#C0392B"; GREY="#7F8C8D"; BG="#F8F9F9"

fig=plt.figure(figsize=(12,5.2)); fig.patch.set_facecolor("white")
X=[0.030,0.267,0.504,0.741]; W=0.229; Y=0.315; H=0.585

def panel(x,t,sub,tc):
    ax=fig.add_axes([x,Y,W,H]); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02,0.02),0.96,0.96,boxstyle="round,pad=0.01,rounding_size=0.05",
                 fc=BG,ec="#D5DBDB",lw=1.0))
    ax.text(0.07,0.92,t,fontsize=11.5,fontweight="bold",color=tc)
    ax.text(0.07,0.845,sub,fontsize=9.5,color=GREY)
    return ax

def inset(ax,y,pred,obs,col,t):
    a=ax.inset_axes([0.62,0.05,0.33,0.28])
    a.plot([0,1],[0,1],ls="--",color="#BDC3C7",lw=0.9)
    a.plot(pred,obs,"-o",color=col,ms=2.0,lw=1.4)
    a.set_xlim(0,1); a.set_ylim(0,1); a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_color("#D5DBDB")
    a.set_title(t,fontsize=7.5,color=GREY,pad=2)

# 1
ax=panel(X[0],"1  Development (Germany)","PTB-XL, n = 2,189",C_INT)
ax.text(0.07,0.63,"AUC 0.987",fontsize=20,fontweight="bold",color=C_INT)
ax.text(0.07,0.46,"ECE 0.019",fontsize=20,fontweight="bold",color=C_INT)
ax.text(0.07,0.28,"ranking and probability\nscale both hold",fontsize=9.5,color="#34495E")
inset(ax,None,mp,fr,C_INT,"calibration")

# 2
ax=panel(X[1],"2  Deployment (China)","Chapman-Shaoxing, n = 22,190",C_EXT)
ax.text(0.07,0.63,"AUC 0.839",fontsize=20,fontweight="bold",color=C_INT)
ax.text(0.07,0.46,"ECE 0.358",fontsize=20,fontweight="bold",color=C_BAD)
ax.text(0.07,0.30,"39.8% scored \u22650.9\nvs 8.0% prevalence",fontsize=9.5,color="#34495E")
inset(ax,None,mp2,fr2,C_EXT,"calibration")

# 3
ax=panel(X[2],"3  What actually broke","decomposition of ECE",  "#34495E")
rows=[("score offset",0.90,C_BAD),("slope mismatch",0.09,C_EXT),("prevalence shift",0.01,GREY)]
yy=0.70
for lab,val,col in rows:
    ax.text(0.07,yy+0.055,lab,fontsize=9.5,color="#34495E")
    ax.text(0.93,yy+0.055,f"{int(round(val*100))}%",fontsize=9.5,color="#34495E",ha="right")
    ax.add_patch(plt.Rectangle((0.07,yy-0.012),0.86,0.045,fc="#EAECEE",ec="none"))
    ax.add_patch(plt.Rectangle((0.07,yy-0.012),max(0.86*val,0.012),0.045,fc=col,ec="none"))
    yy-=0.155
ax.text(0.07,0.20,"one scalar offset \u2014 removable\nwithout touching the ranking",fontsize=9.0,color=GREY,style="italic")

# 4
ax=panel(X[3],"4  Repair, and a control site","400 local labels",C_FIX)
ax.text(0.07,0.63,"ECE 0.007",fontsize=20,fontweight="bold",color=C_FIX)
ax.text(0.07,0.50,"from 0.358; AUC unchanged",fontsize=9.5,color="#34495E")
ax.text(0.07,0.37,"Brazil, same model:\nECE 0.027, no repair",fontsize=9.0,color="#34495E")
inset(ax,None,mp3,fr3,C_FIX,"after repair")

for i in range(3):
    fig.add_artist(FancyArrowPatch((X[i]+W+0.002,0.60),(X[i+1]-0.002,0.60),transform=fig.transFigure,
                 arrowstyle="-|>",mutation_scale=15,lw=1.9,color=GREY,shrinkA=0,shrinkB=0))

fig.text(0.5,0.245,"Calibration is local.",ha="center",fontsize=15,fontweight="bold",color="#1B2631")
fig.text(0.5,0.185,"A few hundred labels restore a usable probability scale \u2014 but every deployment site must fit its own map.",
         ha="center",fontsize=11,color="#34495E")
fig.text(0.5,0.125,"Development-stage calibration does not travel, and the drift tracks the site rather than the geographic distance or the prevalence shift.",
         ha="center",fontsize=9.5,color=GREY,style="italic")

PIC=os.path.join(SUB_DIR, "picture")
PDF=os.path.join(SUB_DIR, "Figure pdf")
fig.savefig(os.path.join(PIC,"Graphical_Abstract.png"),dpi=600,facecolor="white",bbox_inches="tight")
fig.savefig(os.path.join(PDF,"Graphical_Abstract.pdf"),facecolor="white",bbox_inches="tight")
print("saved")
from PIL import Image
im=Image.open(os.path.join(PIC,"Graphical_Abstract.png")).convert("RGB")
im.resize((1500,int(1500*im.size[1]/im.size[0])),Image.LANCZOS).save(r"C:\Users\COLORFUL\AppData\Local\Temp\opencode\ga2.png")
print(im.size)
