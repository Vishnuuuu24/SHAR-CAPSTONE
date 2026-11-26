"""Separate RWF-2000 binary clip task. Official val is held out until final scoring."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def report(y, probabilities):
    pred = probabilities >= .5
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,average='macro')),
            'roc_auc':float(roc_auc_score(y,probabilities)), 'confusion_matrix_NonFight_Fight':confusion_matrix(y,pred,labels=[0,1]).tolist()}


def select_c(x,y,seed,groups):
    fit,held = next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(x,y,groups))
    scores = []
    for c in (.01,.1,1.):
        model = make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,random_state=seed))
        model.fit(x[fit],y[fit])
        scores.append(float(f1_score(y[held],model.predict(x[held]),average='macro')))
    return (.01,.1,1.)[int(np.argmax(scores))], scores


def bootstrap_interval(y,probabilities,groups,draws=2000,seed=0):
    """Cluster by original-video prefix inferred from upstream member names, not renamed files."""
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {g:np.flatnonzero(groups==g) for g in unique}
    values = []
    for _ in range(draws):
        rows = np.concatenate([indices[g] for g in rng.choice(unique,len(unique),replace=True)])
        if len(np.unique(y[rows]))==2:
            r = report(y[rows],probabilities[rows])
            values.append([r[k] for k in ('accuracy','macro_f1','roc_auc')])
    return {k:np.quantile(np.asarray(values)[:,i],[.025,.975]).tolist() for i,k in enumerate(('accuracy','macro_f1','roc_auc'))}
