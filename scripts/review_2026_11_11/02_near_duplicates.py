import numpy as np, os, collections
os.environ["OMP_NUM_THREADS"]="4"
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
d=np.load(os.environ["SHAR_SCRATCH_DIR"]+"/tmp/vidmeans.npz"); X,y,vids=d["X"],d["y"],list(d["vids"])
sp="Datasets/UCF Crime Official/raw/UCF_Crimes/Action_Regnition_splits"
def rd(f): return [os.path.basename(l.strip())[:-4] for l in open(f) if l.strip()]
pos={v:i for i,v in enumerate(vids)}
Xn=X/np.linalg.norm(X,axis=1,keepdims=True)
S=Xn@Xn.T; np.fill_diagonal(S,-1)
print("NN cosine percentiles",np.percentile(S.max(1),[50,90,95,99,100]).round(4))
pairs=[(S[i,j],vids[i],vids[j]) for i in range(len(vids)) for j in range(i+1,len(vids)) if S[i,j]>0.97]
pairs.sort(reverse=True); print("pairs >0.97:",len(pairs)); 
same=sum(y[pos[a]]==y[pos[b]] for _,a,b in pairs); print("same class among them",same)
for p in pairs[:25]: print(round(float(p[0]),4),p[1],p[2])
res=[]
for C in (0.01,0.1,1.0):
  accs=[];f1s=[];dupacc=[];nodupacc=[]
  for k in range(1,5):
    tr=[pos[v] for v in rd(f"{sp}/train_00{k}.txt")]; te=[pos[v] for v in rd(f"{sp}/test_00{k}.txt")]
    sc=StandardScaler().fit(X[tr]); clf=LogisticRegression(C=C,max_iter=3000).fit(sc.transform(X[tr]),y[tr])
    p=clf.predict(sc.transform(X[te])); accs.append(accuracy_score(y[te],p)); f1s.append(f1_score(y[te],p,average="macro"))
    mx=S[np.ix_(te,tr)].max(1); ok=(p==y[te])
    dupacc+=list(ok[mx>0.95]); nodupacc+=list(ok[mx<=0.95])
  print(f"C={C} acc {np.mean(accs):.4f} per fold {np.round(accs,3)} macroF1 {np.mean(f1s):.4f}; test videos with train NN>0.95: n={len(dupacc)} acc={np.mean(dupacc):.3f}; others n={len(nodupacc)} acc={np.mean(nodupacc):.3f}")
# 1-NN
for k in range(1,5):
    tr=[pos[v] for v in rd(f"{sp}/train_00{k}.txt")]; te=[pos[v] for v in rd(f"{sp}/test_00{k}.txt")]
    nn=np.array(tr)[S[np.ix_(te,tr)].argmax(1)]; res.append((y[nn]==y[te]).mean())
print("1-NN acc per fold",np.round(res,3),np.mean(res))
