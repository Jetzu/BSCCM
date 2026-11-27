import functools
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import n087_image_ae as ae
import n087_extract as extract
import bsccm.gpu_norm as gpu_norm


@pytest.fixture
def pack(tmp_path, monkeypatch):
    # Exercise the real RawSet + GPUNorm on a small LF pack including nontrivial flats.
    rng = np.random.default_rng(23)
    batch = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    slides, fov = batch * 2, np.arange(len(batch)) // 2
    np.savez(tmp_path / 'targets_v3.npz', batch=batch, fov=fov, slide=slides,
             split=np.arange(len(batch)) % 3)
    np.savez(tmp_path / 'cellmeta.npz', slide=slides, pos_y=np.arange(len(batch)) + 63,
             pos_x=np.arange(len(batch)) + 60)
    np.savez(tmp_path / 'led_norm.npz', slide_scale=np.linspace(.8, 1.3, 3*23,
             dtype=np.float32).reshape(3, 23), s_c=np.linspace(.7, 2., 23, dtype=np.float32))
    bg = rng.uniform(250, 650, (23, 128, 128)).astype(np.float32)
    np.save(tmp_path / 'backgrounds_p50.npy', bg)
    packed = tmp_path / 'pack'
    packed.mkdir()
    np.save(packed / 'led_array.npy', rng.integers(80, 1100, (10, 23, 128, 128), dtype=np.uint16))
    np.save(packed / 'dpc.npy', rng.normal(size=(10, 128, 128)).astype(np.float32))
    for mod in [ae, extract]:
        monkeypatch.setattr(mod, 'W', str(tmp_path))
        monkeypatch.setattr(mod, 'RawSet', functools.partial(gpu_norm.RawSet, pack=str(packed)))
    monkeypatch.setattr(gpu_norm, 'W', str(tmp_path))
    return tmp_path


def train_options(out):
    return ae.parse_args(['--protocol', 'b1to0', '--dim', '4', '--out', str(out),
                          '--epochs', '1', '--bs', '2', '--workers', '0', '--device', 'cpu',
                          '--l1', '.05', '--no-wandb'])


@pytest.mark.parametrize('sig', [signal.SIGUSR1, signal.SIGTERM])
def test_signal_resume_matches_uninterrupted_and_rejects_wrong_fold(pack, monkeypatch, sig):
    torch.set_num_threads(1)
    base, resumed = train_options(pack/'complete'), train_options(pack/'resume')
    ae.train(base)
    original = ae.normalise

    def signal_once(*args):
        monkeypatch.setattr(ae, 'normalise', original)
        os.kill(os.getpid(), sig)
        return original(*args)

    monkeypatch.setattr(ae, 'normalise', signal_once)
    ae.train(resumed)
    s = torch.load(resumed.out/'ckpt.pt', weights_only=True)
    assert (s['state']['steps'], s['state']['batch'], s['state']['phase']) == (1, 1, 'train')
    assert not (resumed.out/'encoder_final.pt').exists()
    # Poison every held-out raw image. Resuming on source rows must reproduce the whole run.
    mem = np.load(pack/'pack/led_array.npy', mmap_mode='r+')
    mem[::2] = 2047
    del mem
    ae.train(resumed)
    log = json.loads((resumed.out/'log.json').read_text())
    assert log == json.loads((base.out/'log.json').read_text())
    assert log[0]['steps'] == 2 and min(log[0]['train_mse'], log[0]['val_mse']) > 0
    before = torch.load(base.out/'ckpt.pt', weights_only=True)
    after = torch.load(resumed.out/'ckpt.pt', weights_only=True)
    for k, v in before['model'].items():
        torch.testing.assert_close(v, after['model'][k], atol=0, rtol=0)
    resumed.protocol = 'b0to1'
    with pytest.raises(ValueError, match='config'):
        ae.train(resumed)


def test_partition_mask_and_ae_features(pack):
    torch.set_num_threads(1)
    for source in [0, 1]:
        train, val = ae.source_partition(pack/'targets_v3.npz', f'b{source}to{1-source}', 4)
        assert not set(train) & set(val)
        assert set(np.r_[train, val]) == set(range(source, 10, 2))
    x = torch.arange(2*24*128*128, dtype=torch.float32).reshape(2, 24, 128, 128) + 1
    x[0, 0] = float('nan')
    xi, target, mask = ae.masked_input(x, .25, 0)
    assert torch.isfinite(xi).all() and torch.isfinite(target).all()
    assert not mask[0, 0].any() and mask[1].sum() == 24*128*128//4
    assert torch.all(xi[mask] == 0) and torch.equal(xi[~mask], target[~mask])
    a = train_options(pack/'ae')
    ae.train(a)
    checkpoint = a.out/'encoder_final.pt'
    g = np.array([8, 5, 0, 3, 7])
    flags = (g % 2 == 0).astype(np.int8)
    names = np.asarray(['a', 'b'])
    ref = pack/'reference.npz'
    np.savez(ref, H=np.zeros((5, 17)), P=np.zeros((5, 2)), G=g, is_eval=flags, names=names)
    out = pack/'frozen.npz'
    args = extract.parse_args(['--encoder', 'ae', '--ckpt', str(checkpoint), '--rows-from', str(ref),
                               '--out', str(out), '--workers', '0', '--device', 'cpu', '--bs', '2'])
    extract.extract(args)
    model, _ = extract.load_encoder('ae', checkpoint, 0, None)
    raw, ph, row = next(iter(torch.utils.data.DataLoader(ae.RawSet(g), batch_size=len(g))))
    xx = gpu_norm.GPUNorm('cpu')(raw, ph, row, 'all', True, shift=0)
    with torch.no_grad():
        expected = model.eval()(xx).numpy()
    with np.load(out) as z:
        assert set(z.files) == {'H', 'G', 'is_eval', 'names'}
        assert z['H'].shape == (5, 4) and z['H'].dtype == np.float16
        np.testing.assert_allclose(z['H'], expected, atol=.001, rtol=.001)
        for k, v in [('G', g), ('is_eval', flags), ('names', names)]:
            np.testing.assert_array_equal(z[k], v)
    with pytest.raises(ValueError, match='protocol'):
        extract.load_encoder('ae', checkpoint, 1, None)


def test_frozen_ssl_full_presence_and_exclusion(pack):
    import timm
    torch.set_num_threads(1)
    model = timm.create_model('resnet50', pretrained=False, in_chans=48, num_classes=0).eval()
    # Repo's full trainer checkpoints contain both config and the actual row list.
    proof = dict(model=model.state_dict(), rows=np.array([1, 3, 5, 7, 9]),
                 config=dict(arch='resnet50', exclude_batch=[0]))
    ckpt = pack/'ckpt.pt'
    torch.save(proof, ckpt)
    g = np.array([8, 3, 9])
    ref = pack/'reference.npz'
    np.savez(ref, G=g, names=['a'], is_eval=[1, 0, 0])
    out = pack/'ssl.npz'
    a = extract.parse_args(['--encoder', 'ssl', '--ckpt', str(ckpt), '--rows-from', str(ref),
                            '--out', str(out), '--device', 'cpu', '--workers', '0', '--bs', '2'])
    extract.extract(a)
    raw, ph, row = next(iter(torch.utils.data.DataLoader(ae.RawSet(g), batch_size=len(g))))
    xx = gpu_norm.GPUNorm('cpu')(raw, ph, row, 'all', True, shift=0)
    with torch.no_grad():
        expected = model(torch.cat([xx, torch.ones_like(xx)], 1)).numpy()
    with np.load(out) as z:
        assert z['H'].shape == (3, 2048)
        np.testing.assert_allclose(z['H'], expected, atol=.001, rtol=.001)
    proof['rows'][0] = 0
    torch.save(proof, ckpt)
    with pytest.raises(ValueError, match='provenance'):
        extract.extract(a)
