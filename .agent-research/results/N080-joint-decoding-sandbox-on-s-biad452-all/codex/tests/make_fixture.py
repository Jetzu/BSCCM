"""Create a tiny, entirely synthetic three-donor pack; never modifies a real pack."""
import argparse
from pathlib import Path

import numpy as np

MARKERS = ['CD15', 'Siglec8', 'CD14', 'CD19', 'CD3', 'CD45', 'CD4', 'CD56', 'CD8']


def make_fixture(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if (path / 'lf.npy').exists() or (path / 'labels.npz').exists():
        raise FileExistsError(f'Refusing to overwrite an existing pack: {path}')
    rng = np.random.default_rng(13)
    samples = np.array(['sample0'] * 100 + ['sample1_early'] * 50 + ['sample1_late'] * 50 +
                       ['sample2_early'] * 50 + ['sample2_late'] * 50)
    y = rng.normal(size=(len(samples), 9)).astype(np.float32)
    raw = (50 * np.sinh(.5 * y + 1)).astype(np.float32)
    images = np.lib.format.open_memmap(path / 'lf.npy', mode='w+', dtype=np.float16,
                                      shape=(len(y), 3, 128, 128))
    for i in range(len(y)):
        images[i] = rng.normal(0, .1, (3, 128, 128)) + y[i, :3, None, None] * .1
    images.flush()
    np.savez(path / 'labels.npz', Y=y, I_raw=raw, names=MARKERS, sample=samples,
             chan_scale=np.array([.2, .3, .4]), part=np.repeat('part0', len(y)),
             object_number=np.arange(len(y)))
    return path


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    print(make_fixture(ap.parse_args().out))
