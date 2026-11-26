"""Scientific boundary and loss checks for the detector-only follow-up."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from core import detect_followup as follow
from core.detect_model import FeatureStore
from core.detect_data import DetectVideo
from core.detect_language import similarities,token_ids
from core.rwf_evaluation import select_c
from core.detect_uncertainty import prepare,auc


class LossTests(unittest.TestCase):
    def test_padding_never_enters_top_or_bottom(self):
        values = torch.tensor([[2.,3.,999.],[4.,999.,999.]])
        mask = torch.tensor([[False,False,True],[False,True,True]])
        self.assertEqual(follow.selected_mean(values,values,mask).tolist(),[3.,4.])
        self.assertEqual(follow.selected_mean(values,values,mask,largest=False).tolist(),[2.,4.])

    def test_empty_bag_fails(self):
        with self.assertRaises(ValueError):
            follow.selected_mean(torch.ones(1,2),torch.ones(1,2),torch.ones(1,2,dtype=torch.bool))

    def test_magnitude_gradient_separates_classes(self):
        latent = torch.ones(2,3,4,requires_grad=True)
        logits = torch.zeros(2,3,requires_grad=True)
        value = follow.magnitude_loss(latent,logits,torch.zeros(2,3,dtype=torch.bool),torch.tensor([1.,0.]))
        value.backward()
        self.assertLess(latent.grad[0].sum(),0)
        self.assertGreater(latent.grad[1].sum(),0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_teacher_detached_padding_ignored(self):
        logits = torch.zeros(2,3,requires_grad=True)
        samples = torch.tensor([[[.99,.5,.99],[.5,.5,.5]]],requires_grad=True)
        mask = torch.tensor([[False,False,True],[False,False,True]])
        follow.pseudo_instance_loss(logits,samples,mask,torch.tensor([1.,0.])).backward()
        self.assertIsNone(samples.grad)
        self.assertLess(logits.grad[0,0],0)
        self.assertEqual(logits.grad[0,1],0)
        self.assertGreater(logits.grad[1,0],0)
        self.assertTrue((logits.grad[:,2]==0).all())

    def test_uncertainty_reduces_unstable_instance_weight(self):
        logits = torch.zeros(1,2,requires_grad=True)
        samples = torch.tensor([[[.5,.0]],[[.5,1.]]])
        follow.pseudo_instance_loss(logits,samples,torch.zeros(1,2,dtype=torch.bool),torch.tensor([0.]),True).backward()
        self.assertGreater(logits.grad[0,0],logits.grad[0,1])

    def test_statistics_exclude_holdout_and_test(self):
        array = np.array([[1.,3.],[3.,5.],[100.,100.],[1000.,1000.]],np.float32)
        store = FeatureStore([array],{'fit':(0,2),'held':(2,3),'test':(3,4)},np.arange(4),np.zeros(2),np.ones(2))
        fit = [DetectVideo('fit','train',True,'A',30.,0,2,2)]
        follow.fit_statistics(store,fit)
        np.testing.assert_array_equal(store.mean,[2.,4.])
        np.testing.assert_array_equal(store.std,[1.,1.])

    def test_language_cosine_uses_raw_pooler(self):
        got = similarities(np.array([[3.,4.]]),np.eye(2))
        np.testing.assert_allclose(got,[[.6,.8]],atol=1e-7)
        with self.assertRaises(ValueError):similarities(np.zeros((1,2)),np.eye(2))

    def test_all_methods_fit_tiny_bags_with_seeded_initialisation(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(0)
        arrays,spans,videos = [],{},[]
        for i in range(12):
            key = str(i)
            x = rng.normal(size=(8,4)).astype(np.float32) + (2 if i%2 else -2)
            spans[key]=(i*8,(i+1)*8)
            arrays.append(x)
            videos.append(DetectVideo(key,'train',bool(i%2),'A' if i%2 else 'Normal',30.,0,8,8))
        store = FeatureStore([np.concatenate(arrays)],spans,np.tile(np.arange(8),12),np.zeros(4,np.float32),np.ones(4,np.float32))
        for variant in follow.VARIANTS:
            model,history,_ = follow.train(store,videos[:8],videos[8:],variant=variant,seed=0,device='cpu',epochs=6,batch_videos=2,patience=10,log=lambda x:None)
            self.assertEqual(len(history),6)
            self.assertTrue(all(np.isfinite(h['loss']) for h in history))
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        a,_,_ = follow.train(store,videos[:8],videos[8:],variant='control',seed=0,device='cpu',epochs=1,batch_videos=2,log=lambda x:None)
        b,_,_ = follow.train(store,videos[:8],videos[8:],variant='control',seed=0,device='cpu',epochs=1,batch_videos=2,log=lambda x:None)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters())))

    def test_rwf_regularisation_chosen_without_terminal_labels(self):
        rng = np.random.default_rng(0)
        y = np.tile([0,1],30)
        x = rng.normal(size=(60,4))+y[:,None]
        c,scores = select_c(x,y,0,np.arange(60))
        self.assertIn(c,(.01,.1,1.))
        self.assertEqual(len(scores),3)

    def test_cluster_auc_matches_dense_with_ties_and_intervals(self):
        from core.detect_metrics import expand_to_source_frames,source_frame_labels
        from sklearn.metrics import roc_auc_score
        records = [{'scores':np.array([.2,.8,.8]),'source_frames':np.array([0,3,6]),'total_frames':8,'spans':[(2,4),(4,6)]},
                   {'scores':np.array([.2,.8]),'source_frames':np.array([0,3]),'total_frames':5,'spans':[]}]
        dense = np.concatenate([expand_to_source_frames(r['scores'],r['source_frames'],r['total_frames']) for r in records])
        labels = np.concatenate([source_frame_labels(r['total_frames'],r['spans']) for r in records])
        self.assertAlmostEqual(auc(prepare(records),np.array([1,1])),roc_auc_score(labels,dense))
        weights = np.r_[np.full(8,2),np.ones(5)]
        self.assertAlmostEqual(auc(prepare(records),np.array([2,1])),roc_auc_score(labels,dense,sample_weight=weights))

    def test_complete_checkpoint_resumes_without_retraining_or_rng_drift(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(64,4)).astype(np.float32)
        spans={str(i):(i*8,(i+1)*8) for i in range(8)}
        vs=[DetectVideo(str(i),'train',bool(i%2),'A',30.,0,8,8) for i in range(8)]
        store=FeatureStore([x],spans,np.tile(np.arange(8),8),np.zeros(4,np.float32),np.ones(4,np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'latest.pt'
            args=dict(variant='selftrain',seed=7,device='cpu',epochs=6,batch_videos=2,checkpoint_path=path,log=lambda x:None)
            a,history,best=follow.train(store,vs[:4],vs[4:],**args)
            b,other,epoch=follow.train(store,vs[:4],vs[4:],**args)
            self.assertEqual((history,best),(other,epoch))
            self.assertTrue(all(torch.equal(u,v) for u,v in zip(a.parameters(),b.parameters())))


if __name__=='__main__':unittest.main()
