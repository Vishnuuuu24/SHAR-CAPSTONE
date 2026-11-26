import numpy as np, os, csv
os.environ["OMP_NUM_THREADS"]="4"
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
d=np.load(os.environ["SHAR_SCRATCH_DIR"]+"/tmp/vidmeans.npz"); X,y,vids=d["X"],d["y"],list(d["vids"]); pos={v:i for i,v in enumerate(vids)}
sp="Datasets/UCF Crime Official/raw/UCF_Crimes/Action_Regnition_splits"
def rd(f): return [os.path.basename(l.strip())[:-4] for l in open(f) if l.strip()]
F=[];A=[];allval=set();inner=[]
for k in range(1,5):
    trfull=[pos[v] for v in rd(f"{sp}/train_00{k}.txt")]
    val=set(r["source_video_id"] for r in csv.DictReader(open(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{k}/validation/rows.csv")))
    allval|=val
    tr=[i for i in trfull if vids[i] not in val]; va=[i for i in trfull if vids[i] in val]
    best=None
    for C in [0.001,0.003,0.01,0.03,0.1,0.3,1.0]:
        p=cross_val_predict(make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000)),X[tr],y[tr],cv=StratifiedKFold(5,shuffle=True,random_state=0))
        s=f1_score(y[tr],p,average="macro")
        if best is None or s>best[0]: best=(s,C)
    inner.append(round(best[0],3))
    p=make_pipeline(StandardScaler(),LogisticRegression(C=best[1],max_iter=3000)).fit(X[tr],y[tr]).predict(X[va])
    F.append(f1_score(y[va],p,average="macro")); A.append(accuracy_score(y[va],p))
print("mean-pool LR on the 56 validation videos/fold: macroF1",np.round(F,3),round(np.mean(F),4),"acc",round(np.mean(A),4),"| unique val videos",len(allval),"| inner-CV F1 on 476 train videos",inner)
