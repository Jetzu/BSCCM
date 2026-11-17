"""Optional source split-0 fit visualisation (matplotlib; no fitting or LF access)."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='/hnvme/workspace/v181bb10-bsccm/targets_v5.npz')
    ap.add_argument('--diagnostics', type=Path, default=Path('diagnostics'))
    a = ap.parse_args()
    with np.load(a.targets) as t:
        y, mask, split, batch, hist, names = (t[k] for k in ('Y', 'mask', 'split', 'batch', 'hist_row', 'names'))
    fig, axes = plt.subplots(8, 2, figsize=(11, 19), constrained_layout=True)
    for b in (0, 1):
        meta = json.loads((a.diagnostics / f'pospos_draw{b}.json').read_text())
        for i, (name, f) in enumerate(meta['fits'].items()):
            j = list(names).index(name)
            v = y[(batch == b) & (split == 0) & (hist < 0) & mask[:, j], j]
            ax = axes[i, b]; ax.hist(v, bins=100, density=True, color='#bac7d8', label='split-0 labels')
            x = np.linspace(v.min(), v.max(), 1200)
            mu = np.array(f['mu']); var = np.array(f['variance']); q = f['q_observed']
            d = np.exp(-.5 * (x[:, None] - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)
            ax.plot(x, (1-q)*d[:, 0], color='#426b8a', label='negative emission')
            ax.plot(x, q*d[:, 1], color='#bf603d', label='positive emission')
            ax.plot(x, d @ np.array([1-q, q]), color='#1d2732', lw=1, label='mixture')
            ax.set_title(f"Draw {b} {name.removeprefix('single:')}: d′={f['d_prime']:.2f}, π={f['pi']:.3f}", fontsize=10)
            ax.set_xlabel('curated v5 intensity'); ax.set_ylabel('density')
            if i == 0: ax.legend(fontsize=7)
    fig.suptitle('N049: source-draw clean split-0 Gaussian fits\nEta/phi and true prevalence are prior-dependent', fontsize=14)
    fig.savefig(a.diagnostics / 'mixture_fits.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__': main()
