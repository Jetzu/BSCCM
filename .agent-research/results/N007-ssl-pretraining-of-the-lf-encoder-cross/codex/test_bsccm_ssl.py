"""Small CPU checks: python3 -B -m unittest -v test_bsccm_ssl.

Keep this beside bsccm_ssl.py when testing the writer delivery.
"""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from bsccm_ssl import make_views, sigreg, ssl_loss


class SSLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Small tensor tests: thread startup/oversubscription costs more than work.
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(123)

    def test_gaussian_beats_collapse_and_mean_shift(self):
        z = torch.randn(2048, 16)
        gaussian = sigreg(z, n_dirs=64)
        collapsed = sigreg(torch.zeros_like(z), n_dirs=64)
        shifted = sigreg(z + 4, n_dirs=64)
        self.assertLess(gaussian.item(), collapsed.item() * 0.03)
        self.assertLess(gaussian.item(), shifted.item() * 0.03)

    def test_sigreg_matches_scalar_integral_and_batch_scaling(self):
        # At zero, the empirical CF is identically 1, independent of direction.
        z = torch.zeros(31, 7, dtype=torch.float64)
        t = torch.linspace(-5, 5, 17, dtype=torch.float64)
        target = torch.exp(-t.square() / 2)
        expected = torch.trapezoid((1 - target).square() * target, t)
        self.assertTrue(torch.allclose(sigreg(z, n_dirs=9), 31 * expected))
        self.assertTrue(torch.allclose(sigreg(z, n_dirs=9, scale_by_batch=False), expected))

    def assert_valid_view(self, view):
        self.assertEqual(tuple(view.shape), (48, 128, 128))
        data, mask = view[:24], view[24:]
        self.assertEqual(view.dtype, torch.float32)
        self.assertTrue(torch.isfinite(view).all())
        self.assertTrue(((mask == 0) | (mask == 1)).all())
        self.assertTrue((mask == mask[:, :1, :1]).all())
        keep = mask[:, 0, 0].bool()
        self.assertGreater(int(keep.sum()), 0)
        self.assertTrue((data[~keep] == 0).all())
        self.assertTrue((data[keep].abs().sum((1, 2)) > 0).all())
        return keep

    def test_xillum_partition_and_presence(self):
        x = torch.randn(24, 128, 128)
        for n in range(2, 7):
            views = make_views(x, 'xillum', n, torch.Generator().manual_seed(n))
            self.assertEqual(len(views), n)
            keep = torch.stack([self.assert_valid_view(v) for v in views])
            self.assertTrue((keep.sum(0) == 1).all())
            self.assertTrue((keep.sum(1) >= 4).all())
            self.assertEqual(int(keep[:, 23].sum()), 1)

    def test_aug_dropout_and_reproducibility(self):
        x = torch.randn(24, 128, 128)
        views = make_views(x, 'aug', 8, torch.Generator().manual_seed(99))
        repeat = make_views(x, 'aug', 8, torch.Generator().manual_seed(99))
        keep = torch.stack([self.assert_valid_view(v) for v in views])
        self.assertTrue((~keep).any())
        self.assertTrue((keep[0] != keep[1:]).any())
        self.assertFalse(torch.equal(views[0], views[1]))
        for v, w in zip(views, repeat):
            self.assertTrue(torch.equal(v, w))

    def test_spatial_transform_is_positive_isotropic_translation_only(self):
        y, x = torch.meshgrid(torch.arange(128.), torch.arange(128.), indexing='ij')
        image = (x + 2 * y)[None].expand(24, -1, -1).contiguous()
        with patch.object(F, 'affine_grid', wraps=F.affine_grid) as affine, \
             patch.object(F, 'grid_sample', wraps=F.grid_sample) as sample, \
             patch.object(torch, 'flip', side_effect=AssertionError('flip')), \
             patch.object(torch, 'rot90', side_effect=AssertionError('rotation')), \
             patch.object(torch, 'roll', side_effect=AssertionError('wraparound')):
            views = make_views(image, 'aug', 4, torch.Generator().manual_seed(2))
        self.assertEqual(affine.call_count, 4)
        for call in affine.call_args_list:
            theta = call.args[0][0]
            self.assertEqual(theta[0, 1].item(), 0)
            self.assertEqual(theta[1, 0].item(), 0)
            self.assertGreaterEqual(theta[0, 0].item(), 1 / 1.1 - 1e-6)
            self.assertLessEqual(theta[0, 0].item(), 1 / 0.9 + 1e-6)
            self.assertEqual(theta[0, 0].item(), theta[1, 1].item())
            output_shift = -theta[:, 2] / theta[:, :2].diag() * 128 / 2
            self.assertTrue((output_shift.abs() <= 6 + 1e-6).all())
        for call in sample.call_args_list:
            self.assertEqual(call.kwargs['padding_mode'], 'reflection')
        for view in views:
            keep = self.assert_valid_view(view)
            crop = view[:24][keep, 16:-16, 16:-16]
            self.assertGreater((crop[:, :, -1] - crop[:, :, 0]).mean().item(), 0)
            self.assertGreater((crop[:, -1, :] - crop[:, 0, :]).mean().item(), 0)

    def test_dropout_never_removes_every_channel(self):
        original_rand = torch.rand

        def force_dropout(*args, **kwargs):
            if args == (24,):
                return torch.zeros(24, device=kwargs.get('device'))
            return original_rand(*args, **kwargs)

        with patch.object(torch, 'rand', side_effect=force_dropout):
            views = make_views(torch.zeros(24, 128, 128), 'aug', 2, torch.Generator().manual_seed(4))
        for view in views:
            keep = self.assert_valid_view(view)
            self.assertEqual(int(keep.sum()), 1)

    def test_invalid_view_parameters(self):
        x = torch.zeros(24, 128, 128)
        for arm, n in [('xillum', 7), ('aug', 1), ('wrong', 2)]:
            with self.assertRaises(ValueError):
                make_views(x, arm, n, torch.Generator())
        with self.assertRaises(ValueError):
            make_views(x[:23], 'aug', 2, torch.Generator())

    def test_loss_is_finite_and_backpropagates(self):
        zs = [torch.randn(32, 16, requires_grad=True) for _ in range(3)]
        loss, pred, reg = ssl_loss(zs)
        self.assertTrue(torch.isfinite(torch.stack((loss, pred, reg))).all())
        self.assertTrue(torch.allclose(loss, 0.95 * pred + 0.05 * reg))
        centre = torch.stack(zs).mean(0)
        reference = torch.stack([(z - centre).square().sum(1).mean() for z in zs]).mean()
        self.assertTrue(torch.allclose(pred, reference))
        loss.backward()
        for z in zs:
            self.assertIsNotNone(z.grad)
            self.assertTrue(torch.isfinite(z.grad).all())
            self.assertGreater(z.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
