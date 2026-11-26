#!/usr/bin/env python3
"""Detector follow-up: calibrate/train using training videos; evaluate fixed rows once.

Run from a frozen code snapshot with --data-root pointing at the main checkout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import platform
from importlib.metadata import version

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from core import detect_data, detect_model, detect_followup, detect_language
from core.detect_metrics import causal_mean, detection_report
from core.p1c_readiness import sha256_file


def write_json(path,value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=1,default=float)+'\n')
    temp.replace(path)


def event(root,**record):
    with (root/'attempts.jsonl').open('a') as f:
        f.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**record})+'\n')


def make_store(features,videos,fit,variant,similarity_path):
    store = detect_model.load_store(features,('dinov2b','siglip'),videos,standardise=False)
    if variant == 'language':
        store.arrays.append(np.load(similarity_path,mmap_mode='r'))
    return detect_followup.fit_statistics(store,fit)


def save_language(features,weights,output,device):
    embeddings,prompts = detect_language.encode_prompts(weights,device)
    np.save(output/'text_embeddings.npy',embeddings)
    image = np.load(features/'siglip.float16.npy',mmap_mode='r')
    values = np.lib.format.open_memmap(output/'similarities.npy',mode='w+',dtype=np.float32,shape=(len(image),len(embeddings)))
    for a in range(0,len(image),8192):
        values[a:a+8192] = detect_language.similarities(image[a:a+8192,:768],embeddings)
    values.flush()
    write_json(output/'prompts.json',{'prompts':prompts,'weights_sha256':sha256_file(weights/'model.safetensors'),
                                     'embeddings_sha256':sha256_file(output/'text_embeddings.npy'),
                                     'similarities_sha256':sha256_file(output/'similarities.npy')})


def evaluate(args,videos,fit,held,config):
    if (args.output/'test_summary.json').exists():
        raise FileExistsError('test evaluation is already closed; refusing another read')
    variants = list(detect_followup.VARIANTS)
    # Require every trained artifact before opening any temporal annotations.
    for variant in variants:
        for seed in args.seeds:
            for suffix in ('json','pt'):
                if not (args.output/f'{variant}-seed{seed}.{suffix}').exists():
                    raise FileNotFoundError(f'missing completed training artifact: {variant}, {seed}, {suffix}')
    if (args.output/'test_read_started.json').exists():
        raise RuntimeError('a prior test read was opened; inspect its attempts before an explicit recovery')
    write_json(args.output/'test_read_started.json',{'declared_rows':variants+['zero_shot_language'], 'seeds':args.seeds,
                                                   'protocol':config,'utc':datetime.now(timezone.utc).isoformat()})
    intervals = detect_data.load_test_intervals(args.data_root)
    test = [v for v in videos if v.split=='test']
    normal = [v for v in held if not v.anomaly]
    summary,bootstrap_rows = {},{}
    for variant in variants:
        store = make_store(args.features,videos,fit,variant,args.output/'similarities.npy')
        reports,bootstrap_rows[variant] = [],[]
        for seed in args.seeds:
            artifact = args.output/f'{variant}-seed{seed}.pt'
            meta = json.loads((args.output/f'{variant}-seed{seed}.json').read_text())
            if meta['checkpoint_sha256'] != sha256_file(artifact):
                raise ValueError(f'checkpoint changed: {artifact}')
            checkpoint = torch.load(artifact,map_location=args.device,weights_only=False)
            store.mean,store.std = checkpoint['mean'],checkpoint['std']
            model = detect_followup.FrameScorer(store.dim*2).to(args.device)
            model.load_state_dict(checkpoint['state'],strict=True)
            def scores(v):
                return causal_mean(detect_model.score_frames(model,store.frames(v.relative_path,'causal'),args.device),5)
            threshold = float(np.percentile(np.concatenate([scores(v) for v in normal]),99))
            records = [{'scores':scores(v),'source_frames':store.source_frames_of(v.relative_path),'total_frames':v.decoded_frames,
                        'fps':v.fps,'spans':intervals[v.stem],'anomaly':v.anomaly,'category':v.category,'stem':v.stem} for v in test]
            report = detection_report(records,threshold)
            bootstrap_rows[variant].append(records)
            reports.append({'seed':seed,**report})
            np.savez_compressed(args.output/f'test-{variant}-seed{seed}.npz',**{v.relative_path:r['scores'] for v,r in zip(test,records)})
            event(args.output,stage='test',variant=variant,seed=seed,status='completed')
        summary[variant] = reports
        write_json(args.output/f'test-{variant}.json',reports)
    # Fixed, untuned prompt baseline: max anomaly cosine minus normal cosine.
    store = detect_model.load_store(args.features,('siglip',),videos,standardise=False)
    similarity = np.load(args.output/'similarities.npy',mmap_mode='r')
    def zero(v):
        a,b = store.spans[v.relative_path]
        s = similarity[a:b]
        return causal_mean(1/(1+np.exp(-10*(s[:,1:].max(1)-s[:,0]))),5)
    threshold = float(np.percentile(np.concatenate([zero(v) for v in normal]),99))
    records = [{'scores':zero(v),'source_frames':store.source_frames_of(v.relative_path),
        'total_frames':v.decoded_frames,'fps':v.fps,'spans':intervals[v.stem],'anomaly':v.anomaly,'category':v.category,'stem':v.stem} for v in test]
    summary['zero_shot_language'] = [detection_report(records,threshold)]
    bootstrap_rows['zero_shot_language'] = [records]
    from core.detect_uncertainty import paired_auc_intervals
    intervals_report = paired_auc_intervals(bootstrap_rows)
    write_json(args.output/'test_summary.json',{'protocol':config,'table':summary,'paired_source_video_bootstrap':intervals_report,
                                               'task':'separate binary temporal VAD; not 14-class recognition'})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=('language','calibrate','train','evaluate'))
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--features',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--code-revision',required=True)
    args = ap.parse_args(argv)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no implicit CPU fallback')
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    args.features = args.features or args.data_root/'results/detect/features'
    args.output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((args.features/'manifest.json').read_text())
    config = {'protocol':'docs/DETECTOR_FOLLOWUP_2026-11-12.md','code_revision':args.code_revision,
              'feature_manifest':manifest,'seeds':args.seeds,'epochs':args.epochs,'device':args.device,
              'variants':list(detect_followup.VARIANTS),'holdout_fraction':.1,'holdout_seed':0,
              'torch':torch.__version__,'python':platform.python_version(),'threshold_percentile':99,'causal_smoothing':5}
    config['packages'] = {name:version(name) for name in ('numpy','scikit-learn','transformers','tokenizers','safetensors')}
    config['cuda'] = torch.version.cuda
    config['hardware'] = torch.cuda.get_device_name() if torch.cuda.is_available() else platform.processor()
    config['deterministic_algorithms'] = True
    config_path = args.output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('configuration changed within a run directory')
    write_json(config_path,config)
    event(args.output,stage=args.stage,status='started')
    started = time.time()
    try:
        if args.stage=='language':
            if not (args.output/'prompts.json').exists():
                save_language(args.features,args.data_root/'.cache/weights/siglip-base-patch16-224-full',args.output,args.device)
        else:
            videos = detect_data.load_split(args.data_root)
            fit,held = detect_model.split_heldout(videos,fraction=.1,seed=0)
            if args.stage=='evaluate':
                evaluate(args,videos,fit,held,config)
            else:
                for variant in (['control'] if args.stage=='calibrate' else detect_followup.VARIANTS):
                    store = make_store(args.features,videos,fit,variant,args.output/'similarities.npy')
                    for seed in ([0] if args.stage=='calibrate' else args.seeds):
                        meta_path = args.output/f'{variant}-seed{seed}.json'
                        if args.stage=='train' and meta_path.exists():
                            if json.loads(meta_path.read_text())['checkpoint_sha256'] != sha256_file(meta_path.with_suffix('.pt')):
                                raise ValueError('completed checkpoint digest mismatch')
                            continue
                        start = time.time()
                        event(args.output,stage=args.stage,variant=variant,seed=seed,status='started')
                        model,history,best = detect_followup.train(store,fit,held,variant=variant,seed=seed,device=args.device,
                            epochs=1 if args.stage=='calibrate' else args.epochs,
                            checkpoint_path=None if args.stage=='calibrate' else args.output/f'{variant}-seed{seed}-latest.pt',
                            log=lambda x:print(json.dumps(x),flush=True))
                        elapsed = time.time()-start
                        if args.stage=='calibrate':
                            write_json(args.output/'calibration.json',{'epoch_seconds':elapsed,'fit_videos':len(fit),'held_videos':len(held),
                                'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None,
                                'worst_case_training_hours':elapsed*args.epochs*len(detect_followup.VARIANTS)*len(args.seeds)/3600,
                                'test_read':False})
                        else:
                            path = meta_path.with_suffix('.pt')
                            torch.save({'state':model.state_dict(),'mean':store.mean,'std':store.std,'config':config},path)
                            write_json(meta_path,{'variant':variant,'seed':seed,'best_epoch':best,'history':history,'seconds':elapsed,
                                                  'checkpoint_sha256':sha256_file(path),'test_read':False})
                        event(args.output,stage=args.stage,variant=variant,seed=seed,status='completed',seconds=elapsed)
        event(args.output,stage=args.stage,status='completed',seconds=time.time()-started)
    except Exception as exc:
        event(args.output,stage=args.stage,status='failed',variant=locals().get('variant'),seed=locals().get('seed'),error=repr(exc),seconds=time.time()-started)
        raise


if __name__=='__main__':
    main()
