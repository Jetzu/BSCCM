"""CPU-only <=64-cell checks against the current, unpatched working-copy harness."""
import ast
import contextlib
import copy
import gc
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

STAGE = Path(__file__).resolve().parents[1]
REPO = next(p for p in STAGE.parents if (p/'bsccm/data.py').is_file())
TMP = Path('/tmp/n074_codex/test')
TMP.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
import bsccm
bsccm.__path__.insert(0, str(STAGE/'bsccm'))
import numpy as np
import pandas as pd
import timm
import torch
from torch.nn import functional as F
from bsccm.dense_aux import (CHANNELS, DenseTargets, DenseDecoder, align_target, dense_mse,
                             disk, pooled_with_spatial, reflect, roll_batch)
import bsccm.data as data
import bsccm.gpu_norm as gpu_norm

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
SOURCE = Path('/hnvme/workspace/v181bb10-bsccm')
ORIGINAL = REPO/'scripts/train_lf.py'
PATCHED = STAGE/'train_lf.py'


def make_fixture():
    """Exactly 64 real fluorescence crops; all synthetic run artifacts stay in /tmp."""
    index = pd.read_parquet(SOURCE/'bsccm_index.parquet')
    with np.load(SOURCE/'targets_v5cx_v10.npz') as z:
        batch, split = z['batch'], z['split']
    rows = []
    for draw in (0, 1):
        for sp, unstained, count in ((0, True, 16), (0, False, 12), (2, False, 4)):
            select = (batch == draw) & (split == sp) & ((index.antibodies.values == 'unstained') == unstained)
            rows.extend(np.flatnonzero(select)[:count].tolist())
    rows = np.array(rows)
    assert len(rows) == 64 and len(np.unique(rows)) == 64
    source = TMP/'fixture'
    (source/'pack').mkdir(parents=True, exist_ok=True)
    idx = index.iloc[rows].copy().reset_index(drop=True)
    idx['global_index'] = np.arange(64)
    idx.to_parquet(source/'bsccm_index.parquet', index=False)
    for name in ('fluor', 'led_array', 'dpc'):
        raw = np.load(SOURCE/'pack'/f'{name}.npy', mmap_mode='r')
        np.save(source/'pack'/f'{name}.npy', raw[rows])
    rng = np.random.default_rng(74)
    names = [f'single:{m}' for m in ('CD45','CD123','CD19','CD56','CD14','CD16','HLA-DR','CD3')]
    names += ['grp:CD123/HLA-DR/CD14','grp:CD3/CD19/CD56','grp:CD45','grp:CD16','autofl']
    np.savez(source/'targets_v5cx_v10.npz', Y=rng.normal(size=(64,13)).astype('f4'),
             mask=np.ones((64,13), bool), cls3=np.arange(64)%3, cls10=np.arange(64)%3,
             split=split[rows], batch=batch[rows], hist_row=np.full(64,-1),
             slide=batch[rows], fov=np.arange(64), names=np.array(names),
             censored=rng.random((64,13))<.2)
    np.savez(source/'cellmeta.npz', pos_y=np.full(64,64), pos_x=np.full(64,64), slide=batch[rows])
    np.savez(source/'led_norm.npz', slide_scale=np.ones((2,23), 'f4'), s_c=np.ones(23,'f4'))
    np.save(source/'backgrounds_p50.npy', np.full((23,128,128), 200., 'f4'))
    (TMP/'real_rows.json').write_text(json.dumps(rows.tolist()))
    return source


def pack_fixture(source, name='fluor_aux64', workers=2):
    out = TMP/f'{name}.npy'
    # These are reproducible tests, not an accidental resume from an earlier code revision.
    for p in (out, out.with_suffix('.json'), out.with_suffix('.json.tmp')):
        p.unlink(missing_ok=True)
    command = [sys.executable, '-I', '-B', str(STAGE/'pack_fluor_aux.py'), '--source', str(source),
               '--out', str(out), '--smoke', '--workers', str(workers), '--chunk-size', '7',
               '--cofactor-cells', '12']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(result.stdout, end='')
    return out, command


def harness(path, source, ssl='', dense='', extra=()):
    data.W = gpu_norm.W = str(source)
    # Default pack was bound at class definition time; use the unchanged RawSet with an explicit fixture pack.
    original_set = gpu_norm.RawSet
    class FixtureRawSet(original_set):
        def __init__(self, rows, labels=None):
            super().__init__(rows, labels, pack=str(source/'pack'))
    tree = ast.parse(path.read_text(), filename=str(path))
    stop = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == 'ck' for t in n.targets))
    args = ['train_lf.py', '--out', str(TMP/'harness'), '--protocol', 'b1to0', '--arch', 'resnet50',
            '--epochs', '2', '--max-train', '4', '--bs', '2', '--workers', '0', '--device', 'cpu',
            '--targets', 'targets_v5cx_v10.npz', '--seed', '74']
    if ssl: args += ['--init-ssl', str(ssl)]
    if dense: args += ['--dense-aux', str(dense), '--dense-width', '8']
    args += list(extra)
    namespace = dict(__name__='n074_test_harness', __file__=str(path))
    with mock.patch.object(sys, 'argv', args), mock.patch.object(gpu_norm, 'RawSet', FixtureRawSet), \
            mock.patch.dict(sys.modules, {'wandb': None}):
        exec(compile(ast.Module(tree.body[:stop], type_ignores=[]), str(path), 'exec'), namespace)
    # Use the actual training-loop statements through construction of the final loss.
    ep_loop = next(n for n in tree.body if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'ep')
    batch_loop = next(n for n in ep_loop.body if isinstance(n, ast.For))
    end = next(i for i, n in enumerate(batch_loop.body) if isinstance(n, ast.Expr) and
               isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'zero_grad')
    namespace['_batch_code'] = compile(ast.Module(batch_loop.body[:end], type_ignores=[]), str(path), 'exec')
    return namespace


def one_batch(ns):
    torch.manual_seed(987)
    batch = next(iter(ns['dtr']))
    for key, value in zip(('raw','ph','g','y','m','k3','k10','wt'), batch[:8]):
        ns[key] = value
    ns['pz'] = batch[8:]
    exec(ns['_batch_code'], ns)


class DenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = make_fixture()
        cls.aux, cls.command = pack_fixture(cls.source)
        torch.manual_seed(74)
        ssl = timm.create_model('resnet50', pretrained=False, in_chans=48, num_classes=0)
        cls.ssl = TMP/'fixture_ssl.pt'
        torch.save(ssl.state_dict(), cls.ssl)
        del ssl
        print('fixture: 64 unique real cells, actual native fluorescence/LF/phase; synthetic scalar labels/calibration and SSL weights')

    def tearDown(self):
        gc.collect()

    def test_a_legacy_exact(self):
        for ssl in ('', self.ssl):
            for extra in ((), ('--diag-aug','1','--diag-tta','1','--drop-led','LED119','--censor-loss','1')):
                left = harness(ORIGINAL, self.source, ssl=ssl, extra=extra)
                right = harness(PATCHED, self.source, ssl=ssl, extra=extra)
                self.assertEqual(left['_cfg'](), right['_cfg']())
                self.assertEqual(left['a'].__dict__, right['_run_args']())
                self.assertEqual(list(left['model'].state_dict()), list(right['model'].state_dict()))
                for key, value in left['model'].state_dict().items():
                    self.assertTrue(torch.equal(value, right['model'].state_dict()[key]), key)
                one_batch(left)
                one_batch(right)
                for a, b in zip(left['_last_h'], right['_last_h']):
                    self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(left['loss'], right['loss']))
                left['loss'].backward()
                right['loss'].backward()
                for (key, a), (_, b) in zip(left['model'].named_parameters(), right['model'].named_parameters()):
                    if a.grad is not None:
                        self.assertTrue(torch.equal(a.grad, b.grad), key)
                # Same checkpoint content serializes byte-for-byte with the same archive name.
                def checkpoint(ns):
                    buffer = io.BytesIO()
                    torch.save(dict(m=ns['model'].state_dict(), o=ns['opt'].state_dict(),
                                    s=ns['sched'].state_dict(), ep=0, log=[], tr=ns['tr'], ev=ns['ev'], cfg=ns['_cfg']()), buffer)
                    return hashlib.sha256(buffer.getvalue()).hexdigest()
                self.assertEqual(checkpoint(left), checkpoint(right))
                print(f'legacy exact: SSL={bool(ssl)} N068/N073={bool(extra)} outputs/loss/grads/config/checkpoint bytes')
                del left, right
                gc.collect()
        old_eval = next(n for n in ast.parse(ORIGINAL.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        new_eval = next(n for n in ast.parse(PATCHED.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        self.assertEqual(ast.dump(old_eval), ast.dump(new_eval))

    def test_b_dense_backward_and_eval_isolation(self):
        ns = harness(PATCHED, self.source, ssl=SOURCE/'runs/n031_ssl_aug_ex0/encoder_final.pt', dense=self.aux,
                     extra=('--diag-aug','1','--diag-tta','1','--drop-led','LED119','--censor-loss','1', '--zoom','2'))
        one_batch(ns)
        self.assertTrue(torch.isfinite(ns['loss']))
        ns['dense_loss'].backward(retain_graph=True)
        grad = ns['model']['enc'].conv1.weight.grad
        self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)
        self.assertTrue(ns['model']['dense'].condition.weight.grad.abs().sum() > 0)
        ns['opt'].zero_grad(set_to_none=True)
        ns['loss'].backward()
        for p in ns['model'].parameters():
            if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all())
        ns['opt'].step()
        ns['model'].eval()
        x = ns['prep'](ns['raw'], ns['ph'], ns['g'], False)
        seen = []
        encoder_inputs = []
        hook = ns['model']['enc'].register_forward_pre_hook(lambda module, inputs: encoder_inputs.append(inputs[0]))
        with torch.no_grad():
            standard = ns['fwd'](x)
            captured = ns['fwd'](x, seen)
            for a, b in zip(standard, captured): self.assertTrue(torch.equal(a,b))
            output = ns['model']['dense'](seen[0], ns['dense_panel'])
            self.assertEqual(tuple(output.shape), (2,6,64,64))
        hook.remove()
        self.assertEqual(tuple(encoder_inputs[0].shape), (2,48,128,128))
        self.assertTrue((encoder_inputs[0][:,22] == 0).all())
        self.assertTrue((encoder_inputs[0][:,46] == 0).all())
        self.assertTrue((encoder_inputs[0][:,24:46] == 1).all())
        # Evaluation must neither read dense labels nor execute the decoder.
        with mock.patch.object(ns['dense_targets'], 'get', side_effect=AssertionError('eval read dense target')), \
                mock.patch.object(ns['model']['dense'], 'forward', side_effect=AssertionError('eval ran decoder')):
            ns['evaluate'](save=False)
        for rows in ([0], [-1], [64], [int(ns['tr'][-1])+1]):
            with self.assertRaises(AssertionError): ns['dense_targets'].get(torch.tensor(rows), 'cpu')
        print('dense: real n031 source-only SSL resnet50 48-channel forward/backward finite; aux-only encoder gradient; (2,6,64,64); pooled output exact; eval isolated; leakage rejected')

    def test_c_pack_hand_computation_and_resume(self):
        native = np.load(self.source/'pack/fluor.npy')
        y, x = np.indices((128,128))
        corners = (y-64)**2+(x-64)**2 >= 64**2
        hand = np.empty((64,6,64,64), np.float32)
        for i in range(64):
            for c in range(6):
                crop = native[i,5-c].astype(np.float32)
                crop = crop - np.median(crop[corners])
                hand[i,c] = (crop[0::2,0::2]+crop[0::2,1::2]+crop[1::2,0::2]+crop[1::2,1::2])/4
        meta = json.loads(self.aux.with_suffix('.json').read_text())
        idx = pd.read_parquet(self.source/'bsccm_index.parquet')
        t = np.load(self.source/'targets_v5cx_v10.npz')
        for draw in (0,1):
            rows = np.flatnonzero((t['batch']==draw)&(t['split']==0)&(idx.antibodies.values=='unstained'))
            rows = rows[np.linspace(0,len(rows)-1,min(12,len(rows)),dtype=int)]
            fg = np.flatnonzero(disk().ravel())
            pixels = fg[np.linspace(0,len(fg)-1,64,dtype=int)]
            sample = hand[rows].reshape(len(rows),6,-1)[:,:,pixels].transpose(1,0,2).reshape(6,-1)
            cofactor = np.maximum(1,1.4826*np.median(np.abs(sample-np.median(sample,axis=1)[:,None]),axis=1))
            np.testing.assert_array_equal(cofactor, meta['draws'][str(draw)]['cofactor'])
            sel = t['batch']==draw
            transformed = np.arcsinh(hand[sel]/cofactor[None,:,None,None])
            cal = transformed[t['split'][sel]==0].astype(np.float64)
            mean = cal.mean((0,2,3))
            sd = np.maximum(cal.std((0,2,3)),1e-6)
            np.testing.assert_allclose(mean, meta['draws'][str(draw)]['mean'], atol=1e-12)
            np.testing.assert_allclose(sd, meta['draws'][str(draw)]['std'], atol=1e-12)
            hand[sel]=(transformed-mean.astype('f4')[None,:,None,None])/sd.astype('f4')[None,:,None,None]
        np.testing.assert_array_equal(hand.astype('f2'), np.load(self.aux))
        original_bytes = self.aux.read_bytes()
        # Deliberately leave an uncommitted corrupted chunk and restart from the preceding durable row.
        meta['complete']=False
        meta['next_row']=21
        self.aux.with_suffix('.json').write_text(json.dumps(meta))
        out = np.load(self.aux,mmap_mode='r+')
        out[21:] = 0
        out.flush()
        del out
        result = subprocess.run(self.command,capture_output=True,text=True,check=True)
        self.assertEqual(original_bytes,self.aux.read_bytes())
        result = subprocess.run(self.command,capture_output=True,text=True,check=True)
        self.assertIn('already complete',result.stdout)
        leftovers = list(TMP.glob('fluor_aux64*'))
        self.assertEqual({p.suffix for p in leftovers},{'.npy','.json'})
        print('packer: all 64 real crops byte-identical to independent hand transform; durable-row restart byte-identical; completed restart no-op; no chunk files')

    def test_d_registration(self):
        tree = ast.parse(PATCHED.read_text())
        prep = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='prep')
        torch.manual_seed(21)
        # Six distinct spectral channels, with strongly asymmetric features and cyclic edge content.
        target = torch.randn(2,6,64,64)
        high = F.interpolate(target, scale_factor=2, mode='nearest')
        permutation = torch.tensor([1,0,3,2,4,5])
        ns = dict(torch=torch, np=np, F=F, DEVICE='cpu', DENSE=True, CH=[], phys_aug=None,
                  roll_batch=roll_batch, align_target=align_target,
                  diag=lambda z: reflect(z)[:,permutation],
                  gnorm=lambda raw,ph,g,ch,phase,shift: raw.clone())
        exec(compile(ast.Module([prep],type_ignores=[]),str(PATCHED),'exec'),ns)
        from types import SimpleNamespace
        for zoom in (1,2):
            ns['a']=SimpleNamespace(phase=0,lowpass=0,zoom=zoom,diag_aug=.5)
            for dy in range(-4,5):
                for dx in range(-4,5):
                    shifts=torch.tensor([[dy,dx],[-dy,-dx]])
                    # First sample reflects, second does not. Tests the actual prep implementation.
                    with mock.patch.object(torch,'randint',return_value=shifts.T), \
                            mock.patch.object(torch,'rand',return_value=torch.tensor([0.,1.])):
                        xx, yy, weights=ns['prep'](high,torch.empty(2),torch.arange(2),True,target)
                    expected=F.avg_pool2d(xx,2)
                    expected[0]=expected[0,permutation]  # undo LF directional swaps only
                    self.assertTrue(torch.equal(expected,yy),(zoom,dy,dx))
                    self.assertTrue((weights>=.25).all() and (weights<=1).all())
        print('registration: actual prep tested at all 81 integer shifts, with/without reflection, zoom 1/2; spectral channels unswapped')

    def test_e_architecture_support(self):
        for arch in ('dinov3_s','dinov3_b','dinobloom_b'):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as error:
                harness(PATCHED,self.source,dense=self.aux,extra=('--arch',arch))
            self.assertIn('spatial map',error.getvalue())
        # Actual locally stored Cytoland weights and the unmodified LFBackbone wrapper.
        from bsccm.encoders import LFBackbone
        wrapper=LFBackbone('cytoland',24,drop_path=0.)
        wrapper.eval()
        x=torch.randn(2,24,128,128)
        with torch.no_grad():
            a=wrapper(x)
            b,spatial=pooled_with_spatial(wrapper,x)
        self.assertTrue(torch.equal(a,b))
        self.assertEqual(spatial.shape[1],wrapper.num_features)
        self.assertNotIn('forward_features',wrapper.net.__dict__)
        print('archs: dinov3/dinobloom fail clearly before loading weights; actual pretrained Cytoland LFBackbone gate/resize/pooling preserved')

    def test_f_signal_restart(self):
        out = TMP/'signal_aux.npy'
        for p in (out, out.with_suffix('.json'), out.with_suffix('.json.tmp')):
            p.unlink(missing_ok=True)
        command = [sys.executable, '-I', '-B', str(STAGE/'tests/preempt_probe.py'),
                   '--source', str(self.source), '--out', str(out), '--smoke',
                   '--workers', '1', '--chunk-size', '7', '--cofactor-cells', '12']
        import signal
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in proc.stdout:
            if line.startswith('committed '):
                proc.send_signal(signal.SIGTERM)
                break
        stdout, stderr = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 143, stderr)
        state=json.loads(out.with_suffix('.json').read_text())
        self.assertFalse(state['complete'])
        self.assertTrue(0 < state['next_row'] < 64)
        command[3] = str(STAGE/'pack_fluor_aux.py')
        result=subprocess.run(command, capture_output=True, text=True, check=True)
        self.assertEqual(self.aux.read_bytes(), out.read_bytes())
        print('SIGTERM: partial pack exited 143 after a durable chunk; real CLI restart exactly reproduced all 64 rows')

    def test_g_full_loop_checkpoint_resume(self):
        # Complete harness execution, including evaluate, optimizer/scheduler steps,
        # checkpoint writing, strict resume and result JSON. Four source rows / four eval rows.
        data.W = gpu_norm.W = str(self.source)
        original_set = gpu_norm.RawSet
        class FixtureRawSet(original_set):
            def __init__(self, rows, labels=None):
                super().__init__(rows, labels, pack=str(self_source/'pack'))
        self_source=self.source
        def run(path, out, dense):
            args=['train_lf.py','--out',str(out),'--protocol','b1to0','--arch','resnet18',
                  '--epochs','2','--max-train','4','--bs','2','--workers','0','--device','cpu',
                  '--targets','targets_v5cx_v10.npz','--seed','74','--censor-loss','1',
                  '--diag-aug','1','--drop-led','LED119','--diag-tta','1']
            if dense: args+=['--dense-aux',str(self.aux),'--dense-width','8']
            ns=dict(__name__='n074_full_loop',__file__=str(path))
            with mock.patch.object(sys,'argv',args), mock.patch.object(gpu_norm,'RawSet',FixtureRawSet), \
                    mock.patch.dict(sys.modules,{'wandb':None}):
                exec(compile(path.read_text(),str(path),'exec'),ns)
            return ns
        import shutil
        for name in ('legacy_full','patched_full','dense_full'):
            out=TMP/name
            if out.exists(): shutil.rmtree(out)
        original=run(ORIGINAL,TMP/'legacy_full',False)
        patched=run(PATCHED,TMP/'patched_full',False)
        for key,value in original['model'].state_dict().items():
            self.assertTrue(torch.equal(value,patched['model'].state_dict()[key]),key)
        a=np.load(TMP/'legacy_full/eval_preds.npz')
        b=np.load(TMP/'patched_full/eval_preds.npz')
        for key in a.files: np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(original['_cfg'](),patched['_cfg']())
        del original,patched
        ns=run(PATCHED,TMP/'dense_full',True)
        first=torch.load(TMP/'dense_full/ckpt.pt',map_location='cpu',weights_only=False)
        self.assertEqual(first['ep'],2)
        self.assertIn('dense_setup',first['cfg'])
        self.assertTrue(any(k.startswith('dense.') for k in first['m']))
        self.assertTrue(all(np.isfinite(row['train_loss']) for row in first['log']))
        del ns
        ns=run(PATCHED,TMP/'dense_full',True)
        self.assertEqual(ns['ep0'],2)
        self.assertEqual(first['cfg'],ns['_cfg']())
        print('full loop: flag-off final weights/eval predictions exactly match legacy; flag-on completes 4 steps, writes checkpoint and resumes strictly')


if __name__=='__main__':
    unittest.main(verbosity=2)
