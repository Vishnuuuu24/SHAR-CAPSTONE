import json, os, numpy as np, re
os.environ["OMP_NUM_THREADS"]="4"
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
cap={}
for f in ("Train","Val","Test"): cap.update(json.load(open(f"Datasets/UCA/UCFCrime_{f}.json")))
sp="Datasets/UCF Crime Official/raw/UCF_Crimes/Action_Regnition_splits"
def rd(f): return [l.strip() for l in open(f) if l.strip()]
names=sorted(set(l.split("/")[0] for l in rd(f"{sp}/train_001.txt")))
# words that directly name the class, to test how much is label leakage in captions
leak=r"\b(abus\w*|arrest\w*|arson\w*|assault\w*|burgl\w*|explo\w*|fight\w*|accident\w*|robb\w*|shoot\w*|shot|shoplift\w*|steal\w*|stole\w*|thief|thieves|theft|vandal\w*)\b"
def text(v,strip): 
    t=" ".join(cap[v]["sentences"]).lower()
    return re.sub(leak," ",t) if strip else t
for strip in (False,True):
  A=[];F=[];miss=0;T=[];P=[]
  for k in range(1,5):
    def load(f):
        vs=[(os.path.basename(l)[:-4],l.split("/")[0]) for l in rd(f)]
        return [(v,c) for v,c in vs if v in cap], sum(v not in cap for v,_ in vs)
    tr,m1=load(f"{sp}/train_00{k}.txt"); te,m2=load(f"{sp}/test_00{k}.txt"); miss+=m2
    pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2),LogisticRegression(C=10,max_iter=3000))
    pipe.fit([text(v,strip) for v,_ in tr],[c for _,c in tr]); p=pipe.predict([text(v,strip) for v,_ in te])
    t=[c for _,c in te]; A.append(accuracy_score(t,p)); F.append(f1_score(t,p,average="macro")); T+=t; P+=list(p)
  print("class-words stripped" if strip else "raw captions", "acc",np.round(A,3),round(np.mean(A),4),"macroF1",round(np.mean(F),4),"test videos without caption",miss)
  pc=f1_score(T,P,average=None,labels=names); print({n:round(float(x),2) for n,x in zip(names,pc)})
