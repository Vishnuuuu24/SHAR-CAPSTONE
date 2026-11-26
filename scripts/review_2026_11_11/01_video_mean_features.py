import numpy as np, csv, os, sys, collections
os.environ["OMP_NUM_THREADS"]="4"
root="results/p1c/p1c-dinov2-feature-cache-ar-fold1"
vid_feat={}; vid_lab={}; vid_n={}
for split in ("train","validation","test"):
    F=np.load(f"{root}/{split}/features.float16.npy",mmap_mode="r")
    rows=list(csv.DictReader(open(f"{root}/{split}/rows.csv")))
    assert len(rows)==F.shape[0],(len(rows),F.shape)
    idx=collections.defaultdict(list)
    for i,r in enumerate(rows): idx[r["source_video_id"]].append(i); vid_lab[r["source_video_id"]]=r["label"]
    for v,ix in idx.items():
        assert v not in vid_feat, v
        a=np.asarray(F[ix[0]:ix[-1]+1],dtype=np.float32); assert len(a)==len(ix)
        vid_feat[v]=a.mean(0); vid_n[v]=len(ix)
    print(split,len(idx),F.shape,flush=True)
vids=sorted(vid_feat); X=np.stack([vid_feat[v] for v in vids]); y=np.array([vid_lab[v] for v in vids])
np.savez(os.environ["SHAR_SCRATCH_DIR"]+"/tmp/vidmeans.npz",X=X,y=y,vids=np.array(vids),n=np.array([vid_n[v] for v in vids]))
print(len(vids), np.percentile([vid_n[v] for v in vids],[0,25,50,75,100]))
