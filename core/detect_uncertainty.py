"""Exact frame-AUC bootstrap using weighted score runs, clustered by source video.

The causal expander assigns one score to a run of source frames. Counting positive/negative
frames per run avoids sorting millions of repeated frame scores for every bootstrap draw.
"""
import numpy as np
from core.detect_metrics import source_frame_labels


def prepare(records):
    values, positives, negatives, owner = [],[],[],[]
    for video_id,v in enumerate(records):
        starts = np.asarray(v['source_frames'],dtype=np.int64).copy()
        starts[0] = 0
        stops = np.r_[starts[1:],v['total_frames']]
        labels = source_frame_labels(v['total_frames'],v['spans'])
        prefix = np.r_[0,np.cumsum(labels,dtype=np.int64)]
        pos = prefix[stops]-prefix[starts]
        values.extend(v['scores'])
        positives.extend(pos)
        negatives.extend(stops-starts-pos)
        owner.extend([video_id]*len(starts))
    values = np.asarray(values,dtype=np.float32)
    order = np.argsort(values,kind='stable')
    sorted_values = values[order]
    tie_starts = np.r_[0,np.flatnonzero(np.diff(sorted_values)!=0)+1]
    return tuple(np.asarray(x)[order] for x in (positives,negatives,owner))+(tie_starts,len(records))


def auc(prepared,multiplicity):
    positives,negatives,owner,tie_starts,_ = prepared
    weights = multiplicity[owner]
    pos = np.add.reduceat(positives*weights,tie_starts)
    neg = np.add.reduceat(negatives*weights,tie_starts)
    denominator = pos.sum()*neg.sum()
    if not denominator:return np.nan
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/denominator)


def paired_auc_intervals(rows,control='control',draws=2000,seed=0):
    """Rows map method to a list of seed record lists; bootstrap the mean per-seed AUC."""
    prepared = {name:[prepare(records) for records in seeds] for name,seeds in rows.items()}
    n = next(iter(prepared.values()))[0][-1]
    identities = [[r['stem'] for r in records] for seeds in rows.values() for records in seeds]
    if any(keys!=identities[0] for keys in identities):
        raise ValueError('row videos must match in order for a paired bootstrap')
    rng = np.random.default_rng(seed)
    samples = {name:[] for name in prepared}
    for _ in range(draws):
        weights = np.bincount(rng.integers(0,n,n),minlength=n)
        for name,data in prepared.items():samples[name].append(np.mean([auc(d,weights) for d in data]))
    baseline = np.asarray(samples[control])
    estimates = {name:float(np.mean([auc(d,np.ones(n,dtype=np.int64)) for d in data])) for name,data in prepared.items()}
    result = {name:{'mean_per_seed_frame_auc':estimates[name],
        'source_video_95_ci':np.nanquantile(samples[name],[.025,.975]).tolist(),
        'paired_delta_vs_control':estimates[name]-estimates[control],
        'paired_delta_95_ci':np.nanquantile(np.asarray(samples[name])-baseline,[.025,.975]).tolist(),
        'two_sided_bootstrap_p':min(1.,2*min((np.count_nonzero(np.asarray(samples[name])-baseline<=0)+1)/(draws+1),
                                           (np.count_nonzero(np.asarray(samples[name])-baseline>=0)+1)/(draws+1)))}
        for name in prepared}
    names = sorted((name for name in result if name!=control),key=lambda name:result[name]['two_sided_bootstrap_p'])
    running = 0.
    for rank,name in enumerate(names):
        running = max(running,min(1.,result[name]['two_sided_bootstrap_p']*(len(names)-rank)))
        result[name]['holm_adjusted_p'] = running
    return result
