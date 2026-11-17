#!/usr/bin/env python3
"""Scientific/data contracts, numerical OT gradient, and exact CPU resume test."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('n045', ROOT/'scripts/n045_sotalign.py')
m = importlib.util.module_from_spec(spec)
sys.modules['n045'] = m
spec.loader.exec_module(m)
UP = Path('/hnvme/workspace/v181bb10-bsccm/external/SOTAlign')
MatchingModel, CCA, AlignmentLayer = m.upstream(UP)
torch.set_num_threads(2)


class Contracts(unittest.TestCase):
    def test_retrieval_identity_permutation_and_collapse(self):
        x = np.eye(64, dtype=np.float32)
        self.assertEqual(m.retrieval(x,x)['R1'], 1.)
        self.assertEqual(m.retrieval(x,np.roll(x,1,axis=0))['R1'], 0.)
        collapsed = np.ones((64,8), dtype=np.float32)
        self.assertEqual(m.retrieval(collapsed,collapsed)['R10'], 0.)
        self.assertEqual(m.retrieval(x,x)['chance_R10'], 10/64)

    def test_cca_affinity_equals_original_injected_layer(self):
        torch.manual_seed(4)
        x, y = torch.randn(100,17), torch.randn(100,11)
        c = CCA(n_components=8, lam_x=.1, lam_y=.1, eps=1e-6)
        c.fit(x,y)
        layer = AlignmentLayer(vision_dimension=11, text_dimension=17,
                               target_dimension=8, linear_type='cca')
        layer.fit_affinity(x,y,method='cca',cca_lam_x=.1,cca_lam_y=.1)
        out = layer(text_features=x[:13],image_features=y[:19])
        k = F.normalize(out['text_features'],dim=1) @ F.normalize(out['image_features'],dim=1).T
        torch.testing.assert_close(k,c(x[:13],y[:19]),atol=3e-6,rtol=3e-5)

    def test_klot_gradient_matches_finite_difference(self):
        from optimal_transport.klot import KLOT
        torch.manual_seed(7)
        x = (torch.randn(4,5,dtype=torch.float64)*.1).requires_grad_()
        teacher = torch.randn(4,5,dtype=torch.float64)*.1
        obj = KLOT(epsilon_student=.2,epsilon_teacher=.2,n_iter=500)
        loss = obj(x,teacher)
        loss.backward()
        h = 1e-5
        plus, minus = x.detach().clone(), x.detach().clone()
        plus[1,2] += h; minus[1,2] -= h
        fd = (obj(plus,teacher)-obj(minus,teacher))/(2*h)
        self.assertAlmostEqual(x.grad[1,2].item(),fd.item(),places=6)
        self.assertAlmostEqual(float(obj(teacher,teacher)),0.,places=10)

    def test_real_partition_and_eval_separation(self):
        import json
        smoke = ROOT/'smoke'
        complete = json.loads((smoke/'completion.json').read_text())
        self.assertLessEqual(complete['sample_budget']['total_unique_cells'],2000)
        t = np.load('/hnvme/workspace/v181bb10-bsccm/targets_v5c.npz')
        h = np.load('/hnvme/workspace/v181bb10-bsccm/pack/hist_cells.npy')
        for path in smoke.glob('*/seed0/partitions.npz'):
            p = np.load(path)
            b = int(p['source_draw'])
            for pool in ('unpaired_y','head_train','head_val'):
                rows = p[pool]
                self.assertTrue(np.all(t['hist_row'][rows] < 0))
                self.assertTrue(np.all(t['batch'][rows] == b))
            for split,code in [('train',0),('val',1)]:
                rows = h[p[split]]
                self.assertTrue(np.all(t['batch'][rows] == b))
                self.assertTrue(np.all(t['split'][rows] == code))
            if path.parents[1].name.startswith('cross'):
                self.assertTrue(np.all(t['batch'][h[p['test']]] != b))
            self.assertEqual(len(np.intersect1d(p['pbc_a'],p['pbc_b'])),0)
            self.assertTrue(np.all(p['pbc_a']%2 == 0))
            self.assertTrue(np.all(p['pbc_b']%2 == 1))

    def test_exact_resume_after_checkpointed_interrupt(self):
        rng = np.random.default_rng(12)
        xx = rng.normal(size=(64,16)).astype('float32')
        yy = (xx @ rng.normal(size=(16,12))).astype('float32')
        # No labels/targets attribute: the alignment trainer cannot access labels.
        d = SimpleNamespace(G=xx,L=yy,P=rng.normal(size=(40,16)).astype('float32'),
                            X=lambda r: xx[r],Y=lambda r: yy[r])
        part = dict(train=np.arange(24),val=np.arange(24,40),pbc_a=np.arange(20),
                    unpaired_y=np.arange(40,64))
        a = SimpleNamespace(device='cpu',dim=8,steps=8,lr=1e-4,wd=1e-5,optimizer='lion',
            klot_weight=.001,sail_negatives='expanded',eps_student=.05,eps_teacher=.01,
            sinkhorn_iters=30,batch_pairs=12,batch_unpaired=12,eval_every=2,
            checkpoint_every=2,resume=False,fingerprint='synthetic_contract')
        c = CCA(n_components=8,lam_x=.1,lam_y=.1,eps=1e-6)
        c.fit(torch.from_numpy(xx[:24]),torch.from_numpy(yy[:24]))
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, contextlib.redirect_stdout(io.StringIO()):
            full, split = Path(tmp)/'full', Path(tmp)/'interrupted'
            full.mkdir(); split.mkdir()
            m.STOP = False
            m.train_model(d,part,a,'sotalign',0,full,c,MatchingModel,AlignmentLayer,None)
            original_save = m.atomic_torch
            def save_then_interrupt(path, obj):
                original_save(path,obj)
                if Path(path).name == 'last.pt' and obj['step'] == 4:
                    m.STOP = True
            with patch.object(m,'atomic_torch',side_effect=save_then_interrupt):
                with self.assertRaises(SystemExit) as exc:
                    m.train_model(d,part,a,'sotalign',0,split,c,MatchingModel,AlignmentLayer,None)
                self.assertEqual(exc.exception.code,75)
            m.STOP = False
            a.resume = True
            m.train_model(d,part,a,'sotalign',0,split,c,MatchingModel,AlignmentLayer,None)
            x = torch.load(full/'last.pt',weights_only=False)
            y = torch.load(split/'last.pt',weights_only=False)
            for k in x['model']:
                self.assertTrue(torch.equal(x['model'][k],y['model'][k]),k)
            self.assertEqual(x['best_step'],y['best_step'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
