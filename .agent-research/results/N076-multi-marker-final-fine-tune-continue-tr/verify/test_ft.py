"""N076 flag-on smoke on Codex's N074 fixture: (1) base run 2 epochs; (2) --init-full base/ckpt.pt --train-panel CD45,CD19 for 2 epochs:
strict full-state load succeeds, training rows restricted to the panel, losses finite, checkpoint written; args dump keeps the flags."""
import sys, unittest, json, numpy as np, torch
from unittest import mock
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import test_dense_aux as T
class FTTests(T.DenseTests):
    def test_init_full_train_panel(self):
        gpu_norm = T.gpu_norm; T.data.W = gpu_norm.W = str(self.source); src = self.source
        class FixtureRawSet(gpu_norm.RawSet):
            def __init__(s, rows, labels=None): super().__init__(rows, labels, pack=str(src/'pack'))
        def run(out, extra):
            args = ['train_lf.py', '--out', str(out), '--protocol', 'b1to0', '--arch', 'resnet18', '--epochs', '2', '--bs', '2', '--workers', '0',
                    '--device', 'cpu', '--targets', 'targets_v5cx_v10.npz', '--seed', '74'] + extra
            ns = dict(__name__='n076_ft', __file__=str(T.PATCHED))
            with mock.patch.object(sys, 'argv', args), mock.patch.object(gpu_norm, 'RawSet', FixtureRawSet), mock.patch.dict(sys.modules, {'wandb': None}):
                exec(compile(T.PATCHED.read_text(), str(T.PATCHED), 'exec'), ns)
            return ns
        base = run(T.TMP/'ft_base', [])
        ft = run(T.TMP/'ft_tuned', ['--init-full', str(T.TMP/'ft_base/ckpt.pt'), '--train-panel', 'CD45,CD19', '--lr', '1e-4'])
        import pandas as pd
        ab = pd.read_parquet(src/'bsccm_index.parquet').antibodies.to_numpy()
        self.assertTrue(set(ab[ft['tr']]) <= {'CD45', 'CD19'}); self.assertLess(len(ft['tr']), len(base['tr']))
        ck = torch.load(T.TMP/'ft_tuned/ckpt.pt', map_location='cpu', weights_only=False); self.assertTrue(all(np.isfinite(r['train_loss']) for r in ck['log']))
        args = json.load(open(T.TMP/'ft_tuned/result.json'))['args']; self.assertEqual(args['train_panel'], 'CD45,CD19'); self.assertIn('init_full', args)
        b0 = torch.load(T.TMP/'ft_base/ckpt.pt', map_location='cpu', weights_only=False)['m']
        moved = sum(float((ck['m'][k].float() - v.float()).abs().sum()) for k, v in b0.items() if v.dtype.is_floating_point)
        print('ft ok: rows', len(base['tr']), '->', len(ft['tr']), 'param L1 change', round(moved, 4)); self.assertGreater(moved, 0)
if __name__ == '__main__':
    r = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([FTTests('test_init_full_train_panel')])); sys.exit(not r.wasSuccessful())
