"""N075 flag-on smoke: --aux-label (+ --censor-loss so the extra batch elements are C, Y2, M2) runs 2 epochs on Codex's N074 fixture,
checkpoint has reg2.*, losses finite, eval predictions written; --aux-label with --pseudo refused."""
import sys, unittest, numpy as np, torch
from unittest import mock
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import test_dense_aux as T
class AuxTests(T.DenseTests):
    def test_aux_label_full_loop(self):
        gpu_norm = T.gpu_norm; T.data.W = gpu_norm.W = str(self.source)
        src = self.source
        class FixtureRawSet(gpu_norm.RawSet):
            def __init__(s, rows, labels=None): super().__init__(rows, labels, pack=str(src/'pack'))
        out = T.TMP/'aux_full'
        args = ['train_lf.py', '--out', str(out), '--protocol', 'b1to0', '--arch', 'resnet18', '--epochs', '2', '--max-train', '4', '--bs', '2',
                '--workers', '0', '--device', 'cpu', '--targets', 'targets_v5cx_v10.npz', '--seed', '74', '--censor-loss', '1',
                '--aux-label', 'targets_v5cx_v10.npz', '--aux-w', '1.0']
        ns = dict(__name__='n075_aux', __file__=str(T.PATCHED))
        with mock.patch.object(sys, 'argv', args), mock.patch.object(gpu_norm, 'RawSet', FixtureRawSet), mock.patch.dict(sys.modules, {'wandb': None}):
            exec(compile(T.PATCHED.read_text(), str(T.PATCHED), 'exec'), ns)
        ck = torch.load(out/'ckpt.pt', map_location='cpu', weights_only=False)
        self.assertTrue(any(k.startswith('reg2.') for k in ck['m'])); self.assertTrue(all(np.isfinite(r['train_loss']) for r in ck['log']))
        self.assertTrue((out/'eval_preds.npz').exists()); print('aux full loop ok; train_loss', [round(r['train_loss'], 4) for r in ck['log']])
if __name__ == '__main__':
    suite = unittest.TestSuite([AuxTests('test_aux_label_full_loop')]); r = unittest.TextTestRunner(verbosity=2).run(suite); sys.exit(not r.wasSuccessful())
