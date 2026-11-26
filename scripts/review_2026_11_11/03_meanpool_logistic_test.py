import numpy as np, os, csv, json
os.environ["OMP_NUM_THREADS"]="4"
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score
d=np.load(os.environ["SHAR_SCRATCH_DIR"]+"/tmp/vidmeans.npz"); X,y,vids=d["X"],d["y"],list(d["vids"])
pos={v:i for i,v in enumerate(vids)}
sp="Datasets/UCF Crime Official/raw/UCF_Crimes/Action_Regnition_splits"
def rd(f): return [os.path.basename(l.strip())[:-4] for l in open(f) if l.strip()]
Cs=[0.001,0.003,0.01,0.03,0.1,0.3,1.0]
out={}
for mode in ("train476","train532"):
  A=[];F=[];chosen=[];P=[];T=[];V=[]
  for k in range(1,5):
    te=[pos[v] for v in rd(f"{sp}/test_00{k}.txt")]
    trfull=[pos[v] for v in rd(f"{sp}/train_00{k}.txt")]
    val=set(r["source_video_id"] for r in csv.DictReader(open(f"results/p1c/p1c-dinov2-feature-cache-ar-fold{k}/validation/rows.csv")))
    tr=[i for i in trfull if vids[i] not in val] if mode=="train476" else trfull
    assert not set(tr)&set(te)
    best=None
    for C in Cs:  # C chosen by 5-fold CV inside the training videos only
        pipe=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=3000))
        p=cross_val_predict(pipe,X[tr],y[tr],cv=StratifiedKFold(5,shuffle=True,random_state=0))
        s=f1_score(y[tr],p,average="macro")
        if best is None or s>best[0]: best=(s,C)
    pipe=make_pipeline(StandardScaler(),LogisticRegression(C=best[1],max_iter=3000)).fit(X[tr],y[tr])
    p=pipe.predict(X[te]); A.append(accuracy_score(y[te],p)); F.append(f1_score(y[te],p,average="macro")); chosen.append(best)
    P+=list(p);T+=list(y[te]);V+=[vids[i] for i in te]
  print(mode,"n_train",len(tr),"test acc",np.round(A,3),round(np.mean(A),4),"macroF1",np.round(F,3),round(np.mean(F),4),"inner-CV (F1,C)",[(round(a,3),c) for a,c in chosen])
  out[mode]=dict(pred=P,truth=T,vids=V)
json.dump(out,open(os.environ["SHAR_SCRATCH_DIR"]+"/tmp/lr_preds.json","w"))
