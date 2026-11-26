import json,csv,collections,random,sys,os,subprocess,statistics as st
import cv2,numpy as np
OUT=sys.argv[1]; root='results/p1c/p1c-roi-cache/'
lab={r['source_video_id']:r['label'] for r in csv.DictReader(open(root+'rows.csv'))}
vids={}
for p in subprocess.check_output(['find','Datasets/UCF Crime Official/raw/UCF_Crimes/Videos','-name','*.mp4'],text=True).splitlines():
    vids.setdefault(os.path.basename(p)[:-4],p)
EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
BUCKETS=[(0,40,'under40'),(40,60,'40-60'),(60,80,'60-80'),(80,120,'80-120'),(120,1e9,'over120')]
def bucket(h):
    for a,b,n in BUCKETS:
        if a<=h<b:return n
def ok(k,i):return k[i][2]>=0.5
def torso_legs(k):return (ok(k,5) or ok(k,6)) and (ok(k,11) or ok(k,12)) and (ok(k,13) or ok(k,14))
def full(k):return torso_legs(k) and (ok(k,15) or ok(k,16))
recs=[json.loads(l) for l in open(root+'detections.jsonl')]
# stats
bs=collections.defaultdict(lambda:[0,0,0,0]);pv=collections.defaultdict(lambda:collections.defaultdict(list))
for d in recs:
    v=d['source_video_id'];best=False;hb=False
    for p in d['persons']:
        h=p['box'][3]-p['box'][1];b=bs[bucket(h)];k=p['keypoints'];t=torso_legs(k)
        b[0]+=1;b[1]+=t;b[2]+=full(k);b[3]+=sum(s>=.5 for _,_,s in k);best=best or t
    pv[lab[v]][v].append((bool(d['persons']),best))
print('height persons torso+legs full-body meanVisibleJoints')
tot=sum(b[0] for b in bs.values())
for _,_,n in BUCKETS:
    b=bs[n];print(n,b[0],round(b[0]/tot,3),round(b[1]/b[0],3),round(b[2]/b[0],3),round(b[3]/b[0],1))
print('ALL persons torso+legs',round(sum(b[1] for b in bs.values())/tot,3))
print('class perVideoMean: anyPerson usablePerson')
A=[]
for c,vs in sorted(pv.items()):
    a=st.mean(sum(x for x,_ in f)/len(f) for f in vs.values());u=st.mean(sum(y for _,y in f)/len(f) for f in vs.values());A.append(u);print(c,round(a,3),round(u,3))
print('mean across classes usable',round(st.mean(A),3))
# sample
rng=random.Random(20261105);S=collections.defaultdict(list)
for d in recs:
    if d['source_video_id'] not in vids:continue
    if not d['persons']:S['none'].append(d);continue
    p=max(d['persons'],key=lambda p:p['box'][3]-p['box'][1]);S[bucket(p['box'][3]-p['box'][1])].append(d)
def grab(d):
    c=cv2.VideoCapture(vids[d['source_video_id']]);c.set(cv2.CAP_PROP_POS_FRAMES,d['source_frame_index']);o,f=c.read();c.release();return f if o else None
def draw(img,persons,s=1.0,ox=0,oy=0):
    im=img.copy()
    for p in persons:
        k=p['keypoints']
        for a,b in EDGES:
            if k[a][2]>=.3 and k[b][2]>=.3:
                cv2.line(im,(int((k[a][0]-ox)*s),int((k[a][1]-oy)*s)),(int((k[b][0]-ox)*s),int((k[b][1]-oy)*s)),(0,255,0),1,cv2.LINE_AA)
        for x,y,c in k[5:]:
            if c>=.3:cv2.circle(im,(int((x-ox)*s),int((y-oy)*s)),2,(0,0,255),-1)
    return im
for name,n,cols in [(x[2],6,2) for x in BUCKETS]+[('none',12,4)]:
    pool=S[name];rng.shuffle(pool);seen=set();cells=[]
    for d in pool:
        v=d['source_video_id']
        if v in seen:continue
        f=grab(d)
        if f is None:continue
        seen.add(v);tag=f"{lab[v]} {v.split('_')[0]} f{d['source_frame_index']}"
        if name=='none':
            cell=cv2.resize(f,(400,300));cv2.putText(cell,tag,(4,16),0,.45,(0,255,255),1,cv2.LINE_AA)
        else:
            P=max(d['persons'],key=lambda p:p['box'][3]-p['box'][1]);x0,y0,x1,y1=P['box'];m=.25*(y1-y0)
            X0,Y0=max(int(x0-m),0),max(int(y0-m),0);X1,Y1=min(int(x1+m),f.shape[1]),min(int(y1+m),f.shape[0])
            crop=f[Y0:Y1,X0:X1];s=240/max(crop.shape[0],1);w=max(int(crop.shape[1]*s),1);w=min(w,300)
            raw=cv2.resize(crop,(w,240),interpolation=cv2.INTER_CUBIC);sx=w/max(crop.shape[1],1)
            sk=raw.copy()
            for a,b in EDGES:
                k=P['keypoints']
                if k[a][2]>=.3 and k[b][2]>=.3:cv2.line(sk,(int((k[a][0]-X0)*sx),int((k[a][1]-Y0)*s)),(int((k[b][0]-X0)*sx),int((k[b][1]-Y0)*s)),(0,255,0),1,cv2.LINE_AA)
            full_=draw(f,d['persons']);cv2.rectangle(full_,(int(x0),int(y0)),(int(x1),int(y1)),(255,255,0),1)
            cv2.putText(full_,f"{tag} h={int(y1-y0)}",(3,14),0,.4,(0,255,255),1,cv2.LINE_AA)
            cell=np.zeros((240,320+w+w+8,3),np.uint8);cell[:240,:320]=full_[:240,:320];cell[:,324:324+w]=raw;cell[:,328+w:328+2*w]=sk
            canvas=np.zeros((240,950,3),np.uint8);canvas[:,:min(cell.shape[1],950)]=cell[:,:950];cell=canvas
        cells.append(cell)
        if len(cells)==n:break
    rows=[np.hstack(cells[i:i+cols]) for i in range(0,len(cells),cols) if len(cells[i:i+cols])==cols]
    cv2.imwrite(f"{OUT}/{name}.jpg",np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,90]);print('wrote',name,len(cells))
