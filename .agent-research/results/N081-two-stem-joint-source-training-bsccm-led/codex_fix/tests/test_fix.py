"""N081 fix-round CPU regressions: domain BN, donor names, training-only fits, RNG."""
import gc
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_ifc as prior
from n074_support import STAGE, REPO
import numpy as np
import torch

TMP = Path('/tmp/n081_fix/fix_tests')
TMP.mkdir(parents=True, exist_ok=True)
REAL_RUNS = ['sample0', 'sample1_early', 'sample1_late', 'sample2_early', 'sample2_late']


def real_names_pack(path):
    prior.make_ifc(path)
    with np.load(path/'labels.npz') as z: payload = dict(z)
    payload['sample'] = np.array(['sample0']*4 + ['sample1_early']*6 + ['sample1_late']*6 + ['sample2_early']*4 + ['sample2_late']*4)
    np.savez(path/'labels.npz', **payload)
    return payload


def buffers(ns, prefix=''):
    return {f'{i}.{key}': getattr(bn, prefix+key).clone()
            for i, bn in enumerate(ns['IFC_BNS']) for key in ns['BN_KEYS']}


def equal_dict(test, a, b):
    test.assertEqual(set(a), set(b))
    for key in a: test.assertTrue(torch.equal(a[key], b[key]), key)


class FixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prior.IFCTests.setUpClass()
        cls.realpack = TMP/'runs_pack'
        real_names_pack(cls.realpack)

    def tearDown(self): gc.collect()

    def test_a_bn_three_optimizer_steps_rng_matched_control(self):
        traces = []
        for enabled in (False, True):
            if enabled: ns = prior.harness(('--ext-w', '0'))
            else: ns = prior.n074.harness(STAGE/'train_lf.py', prior.SOURCE, ssl=prior.SSL)
            rng_init = torch.get_rng_state().clone()
            data_it = iter(ns['dtr']); rows, inputs, stats, rngs, predictions = [], [], [], [], []
            bns = [m for m in ns['model']['enc'].modules() if isinstance(m, torch.nn.BatchNorm2d)]
            for step in range(3):
                try: batch = next(data_it)
                except StopIteration: data_it = iter(ns['dtr']); batch = next(data_it)
                raw, ph, g, y, *_ = batch
                x = ns['prep'](raw, ph, g, True)
                p, *_ = ns['fwd'](x); bs_loss = .01 * (p-y).square().mean()
                rows.append(g.clone()); inputs.append(hashlib.sha256(x.numpy().tobytes()).hexdigest())
                ns['opt'].zero_grad(set_to_none=True)
                if enabled:
                    bs_before = buffers(ns); ifc_before = buffers(ns, 'ifc_')
                    ix, iy, im, _ = ns['next_ext_batch']()
                    ip = ns['fwd_ifc'](ns['prep_ifc'](ix, True))
                    equal_dict(self, bs_before, buffers(ns))
                    self.assertTrue(any(not torch.equal(v, buffers(ns, 'ifc_')[k]) for k, v in ifc_before.items()))
                    ns['source_backward'](bs_loss, ns['ifc_mse'](ip, iy, im))
                else: bs_loss.backward()
                joint_norm = float(torch.nn.utils.clip_grad_norm_(ns['model'].parameters(), 2.))
                self.assertLess(joint_norm, 2.)  # isolate BN/RNG from allowed joint clipping
                ns['opt'].step(); ns['sched'].step()
                stats.append({f'{i}.{k}':getattr(bn,k).clone() for i,bn in enumerate(bns) for k in ('running_mean','running_var','num_batches_tracked')})
                rngs.append(torch.get_rng_state().clone())
                ns['model'].eval()
                with torch.no_grad(): predictions.append(ns['fwd'](x)[0].clone())
                ns['model'].train()
            traces.append((rng_init,rows,inputs,stats,rngs,predictions))
            del ns, bns, bs_loss, p; gc.collect()
        a,b = traces
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertEqual(a[2], b[2])
        for step in range(3):
            self.assertTrue(torch.equal(a[1][step],b[1][step]))
            equal_dict(self,a[3][step],b[3][step])
            self.assertTrue(torch.equal(a[4][step],b[4][step]))
            self.assertTrue(torch.equal(a[5][step],b[5][step]))
        print('3 optimizer steps, ext_w=0: ALL 53 BN buffer triples, BSCCM rows/augmented inputs/eval predictions/global RNG equal to IFC-skipped control')

    def test_a2_nonzero_ifc_bn_matches_weight_replay_control(self):
        ns = prior.harness(('--dual-zoom','2'))
        control = prior.n074.harness(STAGE/'train_lf.py', prior.SOURCE, ssl=prior.SSL, extra=('--dual-zoom','2'))
        # Replay the joint model's weight trajectory into an IFC-skipped control.
        # Nonzero IFC gradients intentionally change weights; domain isolation means
        # the same BSCCM inputs at the same weights update exactly the same buffers.
        rng = torch.get_rng_state().clone()
        data_it = iter(ns['dtr'])
        for step in range(3):
            for name,p in control['model'].named_parameters():
                p.data.copy_(dict(ns['model'].named_parameters())[name].detach())
            try: raw,ph,g,y,*_ = next(data_it)
            except StopIteration:
                data_it=iter(ns['dtr']); raw,ph,g,y,*_=next(data_it)
            pre = torch.get_rng_state().clone()
            x=ns['prep'](raw,ph,g,True); p,*_=ns['fwd'](x)
            bs_loss=(p-y).square().mean()
            ix,iy,im,_=ns['next_ext_batch']()
            ip=ns['fwd_ifc'](ns['prep_ifc'](ix,True)); il=ns['ifc_mse'](ip,iy,im)
            after=torch.get_rng_state().clone()
            torch.set_rng_state(pre)
            cx=control['prep'](raw,ph,g,True)
            with torch.no_grad(): control['fwd'](cx)
            self.assertTrue(torch.equal(x,cx)); self.assertTrue(torch.equal(after,torch.get_rng_state()))
            cstats={f'{i}.{k}':getattr(bn,k).clone() for i,bn in enumerate(m for m in control['model']['enc'].modules() if isinstance(m,torch.nn.BatchNorm2d)) for k in ns['BN_KEYS']}
            equal_dict(self,buffers(ns),cstats)
            ns['opt'].zero_grad(set_to_none=True)
            _,gi=ns['source_backward'](bs_loss,il); self.assertGreater(gi,0.)
            torch.nn.utils.clip_grad_norm_(ns['model'].parameters(),2.)
            ns['opt'].step(); ns['sched'].step()
        print('3 joint optimizer steps, ext_w=.5, dual zoom: every BSCCM BN buffer equals IFC-skipped weight-replay control; IFC gradients nonzero')

    def test_b_ifc_bn_eval_exception_and_strict_state(self):
        ns = prior.harness(); model = ns['model']; model.train()
        x, *_ = next(iter(ns['ext_dtr']))
        before = buffers(ns)
        ns['fwd_ifc'](ns['prep_ifc'](x, False))
        equal_dict(self, before, buffers(ns))
        expected_counts = [bn.ifc_num_batches_tracked.clone() for bn in ns['IFC_BNS']]
        ns['evaluate_ifc']()
        equal_dict(self, before, buffers(ns))
        for bn,n in zip(ns['IFC_BNS'], expected_counts): self.assertTrue(torch.equal(n,bn.ifc_num_batches_tracked))
        model.eval()
        with torch.no_grad():
            first = ns['fwd_ifc'](ns['prep_ifc'](x,False))
            model['enc'].bn1.ifc_running_mean.add_(100)
            second = ns['fwd_ifc'](ns['prep_ifc'](x,False))
        self.assertFalse(torch.equal(first,second))
        equal_dict(self, before, buffers(ns))
        with mock.patch.object(model['enc'].layer1,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'): ns['fwd_ifc'](ns['prep_ifc'](x,False))
        equal_dict(self,before,buffers(ns))
        stream=io.BytesIO(); torch.save(model.state_dict(),stream); stream.seek(0)
        state=torch.load(stream,weights_only=True); model.load_state_dict(state,strict=True)
        self.assertEqual(len([k for k in state if '.ifc_running_' in k]),2*len(ns['IFC_BNS']))
        del state['enc.bn1.ifc_running_mean']
        with self.assertRaisesRegex(RuntimeError,'Missing key'): model.load_state_dict(state,strict=True)
        print('IFC eval uses private buffers; eval/exception restore BSCCM; both sets serialize; missing IFC buffer rejected')

    def test_c_real_run_donors_and_per_access_boundary(self):
        ns=prior.harness(('--ext-ifc',str(self.realpack)))
        pack=ns['ifc']
        self.assertEqual(set(pack.sample[pack.tr]),set(REAL_RUNS[:3]))
        self.assertEqual(set(pack.sample[pack.ev]),set(REAL_RUNS[3:]))
        self.assertEqual(set(pack.donor[pack.tr]),{'sample0','sample1'})
        for run in ('sample1_early','sample2_late'):
            with self.assertRaisesRegex(ValueError,'DONOR'): ns['IFCData'](str(self.realpack),'sample0,'+run,'sample2')
            with self.assertRaisesRegex(ValueError,'DONOR'): ns['IFCData'](str(self.realpack),'sample0',run)
        dataset=ns['IFCSet'](pack,True)
        for idx in pack.ev:
            dataset.rows=np.array([idx])
            with self.assertRaisesRegex(AssertionError,'donor'): dataset[0]
        other=ns['IFCData'](str(self.realpack),'sample0,sample1,sample2','sample1')
        self.assertFalse(np.isin(other.sample[other.tr],['sample1_early','sample1_late']).any())
        print('Real naming: early/late runs collapse to donors; sample2 early AND late excluded; run-name arguments rejected')

    def test_d_affine_masked_rows_and_optimizer_groups(self):
        path=TMP/'masked_pack'; payload=real_names_pack(path)
        payload['mask'][2,2]=False; payload['Y'][2,2]=1e12; payload['Y'][16:]=1e10
        np.savez(path/'labels.npz',**payload)
        source=prior.SOURCE
        with np.load(source/'targets_v5cx_v10.npz') as z: original=dict(z)
        altered={k:v.copy() for k,v in original.items()}
        altered['mask'][32,4]=False; altered['Y'][32,4]=1e12
        np.savez(source/'targets_v5cx_v10.npz',**altered)
        try: ns=prior.harness(('--ext-ifc',str(path),'--enc-lr-mult','.1'))
        finally: np.savez(source/'targets_v5cx_v10.npz',**original)
        for i,(marker,j) in enumerate(zip(ns['IFC_SHARED'],ns['IFC_REG_ROWS'])):
            b=ns['Y'][ns['tr'],j][ns['Mk'][ns['tr'],j]].astype('f8')
            k=ns['ifc'].names.index(marker); q=ns['ifc']; v=q.y[q.tr,k][q.mask[q.tr,k]].astype('f8')
            scale=v.std()/b.std(); offset=v.mean()-scale*b.mean()
            self.assertAlmostEqual(float(ns['model']['ifc_affine']['scale'][i].detach()),scale,places=6)
            self.assertAlmostEqual(float(ns['model']['ifc_affine']['offset'][i].detach()),offset,places=6)
        groups={id(p):g for g in ns['opt'].param_groups for p in g['params']}
        model=ns['model']
        self.assertEqual(groups[id(model['ifc_stem'].weight)]['max_lr'],ns['a'].lr*.1)
        self.assertEqual(groups[id(model['enc'].conv1.weight)]['max_lr'],ns['a'].lr*.1)
        self.assertEqual(groups[id(model['ifc_only'].weight)]['max_lr'],ns['a'].lr)
        for p in model['ifc_affine'].parameters(): self.assertEqual(groups[id(p)]['weight_decay'],0.)
        self.assertEqual(groups[id(model['enc'].bn1.weight)]['weight_decay'],.05)
        print('Masked train-only affine mean/SD ratios correct; held-out/hidden poison inert; stem encoder LR; affine no decay; baseline BN/bias decay policy retained')

    def test_e_source_gradient_norms_and_joint_gradients(self):
        ns=prior.harness(); ns['el']=0.; prior.n074.one_batch(ns)
        params=list(ns['model'].parameters())
        expected=torch.autograd.grad(ns['loss'],params,allow_unused=True,retain_graph=True)
        bs=torch.autograd.grad(ns['bs_loss'],params,allow_unused=True,retain_graph=True)
        ext=torch.autograd.grad(ns['a'].ext_w*ns['ext_loss'],params,allow_unused=True,retain_graph=True)
        norm=lambda grads: float(torch.stack([g.detach().float().norm().square() for g in grads if g is not None]).sum().sqrt())
        nb,ni=ns['source_backward'](ns['bs_loss'],ns['ext_loss'])
        self.assertAlmostEqual(nb,norm(bs),places=5); self.assertAlmostEqual(ni,norm(ext),places=5)
        for p,e in zip(params,expected):
            if e is None: self.assertIsNone(p.grad)
            else: torch.testing.assert_close(p.grad,e,rtol=2e-5,atol=1e-6)
        before=norm([p.grad for p in params]); result=float(torch.nn.utils.clip_grad_norm_(params,2.))
        self.assertAlmostEqual(result,before,places=4)
        self.assertLessEqual(norm([p.grad for p in params]),2.00001)
        print(f'Per-source norms match independent autograd, accumulated joint gradients agree; preclip norm={result:.4f}')

    def test_f_vectorized_mixed_d4_rng_isolation_adabn_rejection(self):
        ns=prior.harness(); x=torch.arange(8*3*12*12,dtype=torch.float32).reshape(8,3,12,12)
        shifts=torch.tensor([[-4,4],[4,-4],[0,0],[1,2],[-1,-2],[3,4],[0,-3],[4,1]])
        rot=torch.arange(8)%4; flips=torch.arange(8)//4
        state=torch.get_rng_state().clone()
        with mock.patch.object(torch,'randint',side_effect=[shifts,rot,flips]): got=ns['prep_ifc'](x,True)
        scaled=ns['prep_ifc'](x,False)
        want=torch.stack([torch.roll(torch.rot90(v,int(k),(-2,-1)).flip(-1) if f else torch.rot90(v,int(k),(-2,-1)),tuple(sh.tolist()),(-2,-1)) for v,k,f,sh in zip(scaled,rot,flips,shifts)])
        self.assertTrue(torch.equal(got,want))
        ns['prep_ifc'](x,True); ns['next_ext_batch'](); ns['evaluate_ifc']()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        with self.assertRaises(SystemExit): prior.harness(('--adabn','2'))

    def test_g_packer_training_donor_statistics_and_main_metadata(self):
        spec=importlib.util.spec_from_file_location('fix_packer',STAGE/'pack_sbiad452.py')
        pack=importlib.util.module_from_spec(spec); spec.loader.exec_module(pack)
        samples=np.repeat(REAL_RUNS,2); names,rows=pack.statistic_rows(samples,'sample0,sample1')
        np.testing.assert_array_equal(rows,np.arange(6)); self.assertEqual(names,['sample0','sample1'])
        rng=np.random.default_rng(42); x=rng.normal(size=(10,3,128,128)).astype('f4'); intensity=rng.normal(size=(10,9)).astype('f4')*10
        scales,c,rs=pack.fit_statistics(x,intensity,rows)
        x[6:]=np.nan; intensity[6:]=1e30
        s2,c2,rs2=pack.fit_statistics(x,intensity,rows)
        np.testing.assert_array_equal(scales,s2); np.testing.assert_array_equal(c,c2); np.testing.assert_array_equal(rs,rs2)
        manual=np.maximum(1.4826*np.median(np.abs(intensity[:6]-np.median(intensity[:6],0)),0),1.)
        np.testing.assert_array_equal(c,manual)
        for bad in ('sample1_late','sample9',''):
            with self.assertRaises(ValueError): pack.statistic_rows(samples,bad)
        # Execute main with tiny in-memory Zarr/FCS fixtures but real writes/fits.
        class Group:
            attrs={'object_number':[1,2],'shape':[(12,4,4)]*2}
            def __getitem__(self,i): return np.full(12*4*4,10+i,dtype='u2')
        class Zarr:
            class storage:
                @staticmethod
                def ZipStore(path,mode='r'): return path
            @staticmethod
            def open(*args,**kw): return Group()
        columns=['Object Number','Area M01']+['Intensity_MC_'+pack.FCS_KEY[m] for m in pack.MARKERS]
        def fcs(path):
            base=1000 if 'sample2' in path else 1
            return columns,np.array([[1,10]+[base+j for j in range(9)],[2,20]+[base+10+j for j in range(9)]],dtype='f4')
        def image(stack,scale):
            return np.tile(np.arange(3*128*128,dtype='f4').reshape(3,128,128)%17,(1,1,1))
        out=TMP/'packed'; parts=[f'/fake/zarr/{run}/part{i}.zarr.zip' for i,run in enumerate(REAL_RUNS)]
        with mock.patch.object(sys,'argv',['pack','--out',str(out),'--allow-partial']), mock.patch.dict(sys.modules,{'zarr':Zarr}), mock.patch.object(pack.glob,'glob',return_value=parts), mock.patch.object(pack,'read_fcs',side_effect=fcs), mock.patch.object(pack,'lf_image',side_effect=image): pack.main()
        metadata=json.loads((out/'pack.json').read_text())
        self.assertEqual(metadata['stat_donors'],['sample0','sample1']); self.assertEqual(metadata['n_stat_cells'],6)
        self.assertEqual(metadata['stat_runs'],REAL_RUNS[:3])
        with np.load(out/'labels.npz') as z:
            np.testing.assert_array_equal(z['cofactor'],pack.fit_statistics(np.load(out/'lf.npy',mmap_mode='r'),z['I_raw'],np.arange(6))[1])
        print('Packer main writes donor-fit provenance; test-donor NaN inputs/extreme labels cannot change fitted cofactor/channel scale')

    def test_h_real_v10_training_draw_affine_statistics(self):
        for protocol in ('b1to0','b0to1'):
            ns=prior.n074.harness(STAGE/'train_lf.py',prior.n074.SOURCE,ssl=prior.SSL,
                extra=('--ext-ifc',str(self.realpack),'--max-train','0','--protocol',protocol))
            for record,j in zip(ns['_affine_init'],ns['IFC_REG_ROWS']):
                values=ns['Y'][ns['tr'],j][ns['Mk'][ns['tr'],j]].astype('f8')
                self.assertAlmostEqual(record['sd_bsccm'],float(values.std()),places=12)
                self.assertEqual(record['n_bsccm'],len(values))
            print('REAL V10',protocol,json.dumps(ns['_affine_init']))
            del ns; gc.collect()


if __name__=='__main__': unittest.main(verbosity=2)
