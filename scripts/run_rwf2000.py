#!/usr/bin/env python3
"""Native-split RWF-2000 features and binary classification; independent of UCF labels."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time
from importlib.metadata import version

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from core.p1c_readiness import sha256_file
from scripts.run_detector_followup import event,write_json


def decode(path):
    """Decode to EOF; never trust metadata frame counts to choose frames."""
    import cv2
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'cannot open {path}')
    frames = []
    try:
        while True:
            ok,frame = cap.read()
            if not ok:
                break
            frames.append(cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),(224,224),interpolation=cv2.INTER_LINEAR))
    finally:
        cap.release()
    if not frames:
        raise ValueError(f'no decoded frames: {path}')
    selected = np.linspace(0,len(frames)-1,16).round().astype(int)
    # Resize on CPU one frame at a time to avoid holding whole high-resolution clips.
    pixels = np.stack([cv2.resize(frames[i],(224,224),interpolation=cv2.INTER_LINEAR) for i in selected])
    return pixels,selected.tolist(),len(frames)


def feature_pass(args,items):
    if (args.output/f'{args.stage}.json').exists():
        raise FileExistsError('completed feature/calibration artifact already exists; refusing overwrite')
    import yaml
    from scripts.build_combined_patch_cache import _load_backbone_and_preprocessing
    from scripts.build_p1c_feature_cache import _local_siglip
    dinov2,_,_ = _load_backbone_and_preprocessing(torch.device(args.device))
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/experiments/p1c/P1C-VIDEO-SIGLIP-AR-FOLD1.yaml').read_text())
    siglip = _local_siglip(cfg['fixed_protocol']['backbone']['weights']).to(args.device).eval()
    limit = 16 if args.stage=='calibrate' else len(items)
    todo = items[:limit]
    arrays = {name:np.lib.format.open_memmap(args.output/f'{name}-{args.stage}.npy',mode='w+',dtype=np.float32,shape=(len(todo),1536)) for name in ('dinov2b','siglip')}
    mean = torch.tensor([.485,.456,.406],device=args.device).view(1,3,1,1)
    std = torch.tensor([.229,.224,.225],device=args.device).view(1,3,1,1)
    started = time.time()
    decoded = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        # Keep at most four decoders ahead: executor.map would retain thousands of pixel arrays.
        pending = {}
        for i in range(min(4,len(todo))):
            pending[i] = pool.submit(decode,args.data_root/'Datasets/RWF-2000'/todo[i]['path'])
        for i,item in enumerate(todo):
            pixels,indices,count = pending.pop(i).result()
            if i+4<len(todo):
                pending[i+4] = pool.submit(decode,args.data_root/'Datasets/RWF-2000'/todo[i+4]['path'])
            x = torch.from_numpy(pixels).permute(0,3,1,2).float().to(args.device)/255
            with torch.inference_mode():
                b = dinov2.pooled_features((x-mean)/std).mean(0)
                s = siglip.pooled_features((x-.5)/.5).mean(0)
            arrays['dinov2b'][i] = b.cpu().numpy()
            arrays['siglip'][i] = s.cpu().numpy()
            decoded.append({'path':item['path'],'decoded_frames':count,'selected_source_frames':indices})
            if (i+1)%50==0:
                for a in arrays.values():a.flush()
                print(json.dumps({'clips':i+1,'seconds':time.time()-started}),flush=True)
    for a in arrays.values():a.flush()
    elapsed = time.time()-started
    write_json(args.output/f'{args.stage}.json',{'clips':len(todo),'seconds':elapsed,'clips_per_second':len(todo)/elapsed,
        'projected_all_minutes':elapsed/len(todo)*len(items)/60,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30,
        'decode':decoded,'sampling':'16 rounded linspace source indices after decoding to EOF; OpenCV INTER_LINEAR resize to 224',
        'arrays':{name:sha256_file(args.output/f'{name}-{args.stage}.npy') for name in arrays}})


def evaluate(args,items):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from core.rwf_evaluation import select_c, report, bootstrap_interval
    if (args.output/'test_read_started.json').exists():
        raise FileExistsError('native val scoring was already opened')
    arrays = {name:np.load(args.output/f'{name}-features.npy') for name in ('dinov2b','siglip')}
    train = np.array([i for i,v in enumerate(items) if v['path'].startswith('train/')])
    test = np.array([i for i,v in enumerate(items) if v['path'].startswith('val/')])
    y = np.array([int('/Fight/' in v['path']) for v in items])
    source_names = [re.sub(r'_\d+$','',Path(v['archive_member']).stem) for v in items]
    # Join clips sharing either a source-name prefix or byte content into one CV group.
    parent = list(range(len(items)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    seen = {}
    for i,item in enumerate(items):
        for key in (('source',source_names[i]),('content',item['sha256'])):
            if key in seen:parent[find(i)] = find(seen[key])
            else:seen[key] = i
    groups = np.array([find(i) for i in range(len(items))])
    group_overlap = sorted(set(groups[train]) & set(groups[test]))
    clean = ~np.isin(groups[test],groups[train])
    write_json(args.output/'source_group_overlap.json',{'overlap':list(map(int,group_overlap)),
        'excluded_terminal_paths':[items[i]['path'] for i in test[~clean]],
        'group_definition':'connected components of original name without final _integer and SHA256 identity'})
    matrices = {**arrays,'fusion':np.concatenate(list(arrays.values()),axis=1)}
    chosen = {}
    for name,x in matrices.items():
        chosen[name] = [select_c(x[train],y[train],seed,groups[train]) for seed in (0,1,2)]
    write_json(args.output/'selection.json',chosen)
    write_json(args.output/'test_read_started.json',{'rows':list(matrices),'seeds':[0,1,2],'native_val_as_terminal_test':True})
    reports = {}
    for name,x in matrices.items():
        rows,probabilities = [],[]
        for seed,(c,scores) in enumerate(chosen[name]):
            model = make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,random_state=seed))
            model.fit(x[train],y[train])
            prob = model.predict_proba(x[test])[:,1]
            rows.append({'seed':seed,'C':c,'heldout_train_macro_f1':scores,'native_split':report(y[test],prob),
                         'overlap_excluded_sensitivity':report(y[test][clean],prob[clean])})
            probabilities.append(prob)
            joblib.dump(model,args.output/f'{name}-seed{seed}.joblib')
        np.save(args.output/f'{name}-test-probabilities.npy',np.asarray(probabilities))
        reports[name] = {'per_seed':rows,'native_clustered_95_ci_seed_mean':bootstrap_interval(y[test],np.mean(probabilities,axis=0),groups[test]),
                        'overlap_excluded_clustered_95_ci_seed_mean':bootstrap_interval(y[test][clean],np.mean(probabilities,axis=0)[clean],groups[test][clean])}
    write_json(args.output/'test_summary.json',{'task':'RWF-2000 native binary clip recognition; no UCF label merging',
        'split':{'train':len(train),'terminal_val':len(test),'overlap_excluded_terminal':int(clean.sum())},
        'source_group_overlap':list(map(int,group_overlap)),'table':reports})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=('calibrate','features','evaluate'))
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--intake',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--code-revision',required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('requested CUDA unavailable')
    args.output.mkdir(parents=True,exist_ok=True)
    config = {'protocol':'docs/DETECTOR_FOLLOWUP_2026-11-12.md','code_revision':args.code_revision,
              'intake_sha256':sha256_file(args.intake),'frames_per_clip':16,'seeds':[0,1,2],
              'C_grid':[.01,.1,1.],'device':args.device,'torch':torch.__version__}
    config['packages'] = {name:version(name) for name in ('numpy','scikit-learn','transformers','tokenizers','safetensors','opencv-python-headless')}
    config['cuda'] = torch.version.cuda
    config['hardware'] = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
    p = args.output/'config.json'
    if p.exists() and json.loads(p.read_text())!=config:
        raise ValueError('configuration changed within a run directory')
    write_json(p,config)
    items = sorted(json.loads(args.intake.read_text())['videos'],key=lambda v:v['path'])
    event(args.output,stage=args.stage,status='started')
    try:
        if args.stage=='evaluate':evaluate(args,items)
        else:feature_pass(args,items)
        event(args.output,stage=args.stage,status='completed')
    except Exception as exc:
        event(args.output,stage=args.stage,status='failed',error=repr(exc))
        raise


if __name__=='__main__':main()
