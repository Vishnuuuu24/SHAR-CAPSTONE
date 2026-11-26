import itertools, json, numpy as np, sys
sys.path.insert(0,'.')
from core.confusion_groups import load_runs, macro_f1
classes, runs = load_runs('results/p1c/p1c-late-fusion/dinov2_probs.csv')
C=len(classes); ci={c:i for i,c in enumerate(classes)}
FAM={'Abuse':'violence','Assault':'violence','Fighting':'violence','Shooting':'violence',
 'Burglary':'theft','Robbery':'theft','Shoplifting':'theft','Stealing':'theft',
 'Arson':'fire','Explosion':'fire','Vandalism':'damage','RoadAccidents':'traffic','Arrest':'police','Normal_Videos_event':'normal'}
folds=sorted({f for f,s in runs}); seeds=sorted({s for f,s in runs})
valset={f:set(runs[(f,seeds[0])]['videos']) for f in folds}
def score(S, f, exclude=None):
    idx=np.array([ci[c] for c in S]); out=[]
    for s in seeds:
        r=runs[(f,s)]; t=np.asarray(r['truth']); p=np.asarray(r['probs'])
        m=np.isin(t,idx)
        if exclude is not None: m&=~np.isin(np.asarray(r['videos']),list(exclude))
        if m.sum()==0: return None
        sub=p[m][:,idx]; pred=idx[sub.argmax(1)]
        loc={g:k for k,g in enumerate(idx)}
        out.append(macro_f1(np.array([loc[x] for x in t[m]]),np.array([loc[x] for x in pred]),len(idx)))
    return float(np.mean(out))
subs=list(itertools.combinations(classes,7))
res=[]
for S in subs:
    pf={f:score(S,f) for f in folds}
    res.append((S,pf,np.mean(list(pf.values()))))
res.sort(key=lambda x:-x[2])
# LOFO selection: rank by other folds (excluding videos shared with scored fold)
lofo={}
for f in folds:
    best=None
    for S in subs:
        vals=[score(S,g,exclude=valset[f]) for g in folds if g!=f]
        vals=[v for v in vals if v is not None]
        m=np.mean(vals)
        if best is None or m>best[1]: best=(S,m)
    lofo[f]={'subset':best[0],'selection_score':best[1],'scored':score(best[0],f)}
json.dump({'ranked':[{'classes':S,'per_fold':pf,'mean':m,'families':sorted({FAM[c] for c in S})} for S,pf,m in res],
           'lofo':lofo},open('results/divide-and-conquer/subset7_scores.json','w'),indent=1)
print('in-sample top 10')
for S,pf,m in res[:10]: print(round(m,3),[round(v,3) for v in pf.values()],S, len({FAM[c] for c in S}))
print('median',round(np.median([m for *_,m in res]),3),'worst',round(res[-1][2],3))
for f,v in lofo.items(): print('LOFO',f,v['subset'],round(v['selection_score'],3),'->',round(v['scored'],3))

print('--- rationale subsets')
d={x['classes'] if False else tuple(x['classes']):x for x in json.load(open('results/divide-and-conquer/subset7_scores.json'))['ranked']}
ranks={tuple(x['classes']):i for i,x in enumerate(json.load(open('results/divide-and-conquer/subset7_scores.json'))['ranked'])}
cands={'one_per_family':('Arrest','Explosion','Fighting','Normal_Videos_event','RoadAccidents','Shoplifting','Vandalism'),
 'top7_by_class_f1':('Arson','Explosion','Normal_Videos_event','Shoplifting','Stealing','Vandalism','Arrest')}
from core.confusion_groups import per_class_f1
for n,S in cands.items():
    S=tuple(sorted(S,key=classes.index)); pf={f:score(S,f) for f in folds}
    print(n,round(np.mean(list(pf.values())),3),[round(v,3) for v in pf.values()],'rank',ranks[S]+1,'/',len(subs),'families',len({FAM[c] for c in S}))
# LOFO top-7 by per-class F1 from other folds
for f in folds:
    acc=[]
    for g in folds:
        if g==f: continue
        for s in seeds:
            r=runs[(g,s)]; m=~np.isin(np.asarray(r['videos']),list(valset[f]))
            t=np.asarray(r['truth'])[m]; p=np.asarray(r['pred'])[m]
            acc.append(per_class_f1(t,p,C))
    pc=np.mean(acc,0); S=tuple(classes[i] for i in sorted(np.argsort(-pc)[:7]))
    print('LOFO-perclass',f,[c[:6] for c in S],'->',round(score(S,f),3))
print('--- 7-family merge of all 14 classes')
fams=sorted(set(FAM.values())); fi={c:fams.index(FAM[c]) for c in classes}
M=np.zeros((C,len(fams)))
for c in classes: M[ci[c],fi[c]]=1
pf={}
for f in folds:
    v=[]
    for s in seeds:
        r=runs[(f,s)]; t=np.array([fi[classes[x]] for x in r['truth']]); p=(np.asarray(r['probs'])@M).argmax(1)
        v.append(macro_f1(t,p,len(fams)))
    pf[f]=np.mean(v)
print('family7',round(np.mean(list(pf.values())),3),[round(x,3) for x in pf.values()])
ok=[np.mean([x for x in r['per_fold'].values()]) for r in json.load(open('results/divide-and-conquer/subset7_scores.json'))['ranked']]
