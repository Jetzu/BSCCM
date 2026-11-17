"""Fit source-only N049 noise-aware soft labels, CPU / NumPy only.

Z=true positive, S=observed-positive component, Y=curated intensity:
  Z~Bernoulli(pi); P(S=1|Z=1)=eta; P(S=1|Z=0)=phi;
  Y|S=s ~ Normal(mu_s, sigma**2) (shared variance, monotone posterior), mu_0 < mu_1.
The marginal identifies q=pi*eta+(1-pi)*phi, NOT pi,eta,phi separately.
Truncated Beta priors regularise this ridge; eta/phi are not measured error rates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bsccm.n049_pospos import source_digest

ETA_BOUNDS = (0.60, 0.99)
PHI_BOUNDS = (0.001, 0.15)
ETA_BETA = (18., 2.)
PHI_BETA = (2., 48.)


def _log_components(y, mu, var):
    return -0.5 * (np.log(2 * np.pi * var) + (y[:, None] - mu) ** 2 / var)


def _prior(eta, phi):
    return ((ETA_BETA[0] - 1) * np.log(eta) + (ETA_BETA[1] - 1) * np.log1p(-eta)
            + (PHI_BETA[0] - 1) * np.log(phi) + (PHI_BETA[1] - 1) * np.log1p(-phi))


def _estep(y, mu, var, pi, eta, phi):
    q = np.clip(pi * eta + (1 - pi) * phi, 1e-12, 1 - 1e-12)
    lp = _log_components(y, mu, var) + np.log([1 - q, q])
    norm = logsumexp(lp, axis=1)
    r = np.exp(lp - norm[:, None])  # P(S | Y)
    # Joint posterior sufficient counts (Z,S). Y depends on S only.
    ez1s1 = r[:, 1].sum() * pi * eta / q
    ez1s0 = r[:, 0].sum() * pi * (1 - eta) / (1 - q)
    ez0s1 = r[:, 1].sum() * (1 - pi) * phi / q
    ez0s0 = r[:, 0].sum() * (1 - pi) * (1 - phi) / (1 - q)
    return r, float(norm.sum()), (ez1s1, ez1s0, ez0s1, ez0s0)


def fit_noise_mixture(y, max_iter=600, tol=1e-8):
    """Deterministic multi-start MAP EM; tied variance gives location populations.

    The shared variance prevents a broad negative tail from becoming a second
    component and guarantees P(Z=1|Y=y) increases with y (eta > phi).
    """
    y = np.asarray(y, np.float64)
    if y.ndim != 1 or len(y) < 32 or not np.isfinite(y).all():
        raise ValueError('fit requires >=32 finite 1D labels')
    n = len(y); v = float(y.var())
    if v < 1e-12: raise ValueError('constant marker: a two-component fit is undefined')
    floor = max(0.01 * v, 1e-8)  # component SD >= 0.1 total SD
    best = None
    for quantiles in ((20, 80), (10, 60), (50, 90), (5, 95)):
        mu = np.percentile(y, quantiles); var = np.full(2, v * .5)
        eta, phi = 17 / 18, 1 / 48  # truncated Beta modes
        pi = .5; previous = -np.inf; trace = []
        converged = False
        for it in range(max_iter):
            r, ll, counts = _estep(y, mu, var, pi, eta, phi)
            objective = ll + _prior(eta, phi)
            trace.append(objective)
            if it and abs(objective - previous) <= tol * (1 + abs(previous)):
                converged = True
                break
            previous = objective
            mass = np.maximum(r.sum(0), 1e-12)
            mu = (r * y[:, None]).sum(0) / mass
            # Tied-variance Gaussian responsibilities preserve ordered means.
            if mu[0] > mu[1] + 1e-12: raise RuntimeError('ordered means crossed')
            shared_var = max(float((r * (y[:, None] - mu) ** 2).sum() / n), floor)
            var = np.full(2, shared_var)
            a, b, c, d = counts
            pi = float(np.clip((a + b) / n, 1e-8, 1 - 1e-8))
            eta = float(np.clip((a + ETA_BETA[0] - 1) / (a + b + sum(ETA_BETA) - 2), *ETA_BOUNDS))
            phi = float(np.clip((c + PHI_BETA[0] - 1) / (c + d + sum(PHI_BETA) - 2), *PHI_BOUNDS))
            # MAP ascent along the unidentified ridge, keeping q (and likelihood) fixed.
            # Independent Beta modes projected onto phi <= q <= eta maximise the prior.
            q = pi * eta + (1 - pi) * phi
            eta = float(np.clip(max(17 / 18, q), *ETA_BOUNDS))
            phi = float(np.clip(min(1 / 48, q), *PHI_BOUNDS))
            pi = float(np.clip((q - phi) / (eta - phi), 0., 1.))
        r, ll, _ = _estep(y, mu, var, pi, eta, phi)
        objective = ll + _prior(eta, phi)
        trace.append(objective)
        candidate = dict(mu=mu.tolist(), variance=var.tolist(), pi=pi, eta=eta, phi=phi,
                         q_observed=float(pi * eta + (1 - pi) * phi),
                         log_likelihood=ll, map_objective=float(objective), n_fit=n,
                         n_iter=it + 1, converged=converged,
                         min_objective_step=float(np.min(np.diff(trace))),
                         variance_floor=floor, init_quantiles=list(quantiles))
        if best is None or objective > best['map_objective']: best = candidate
    ll1 = float((-0.5 * (np.log(2 * np.pi * v) + (y - y.mean()) ** 2 / v)).sum())
    best.update(d_prime=float((best['mu'][1] - best['mu'][0]) / np.sqrt(np.mean(best['variance']))),
                bic_1=float(2 * np.log(n) - 2 * ll1),
                bic_2=float(4 * np.log(n) - 2 * best['log_likelihood']))
    best['delta_bic_1_minus_2'] = best['bic_1'] - best['bic_2']
    best['warnings'] = []
    if not best['converged']: best['warnings'].append('EM iteration limit')
    if best['d_prime'] < 1: best['warnings'].append('weak separation: d_prime < 1')
    if best['pi'] < .01 or best['pi'] > .99:
        best['warnings'].append('true prevalence near boundary: prior-sensitive or constant soft targets')
    if best['q_observed'] < .01 or best['q_observed'] > .99:
        best['warnings'].append('tiny observed component: possible tail/outlier fit')
    if best['delta_bic_1_minus_2'] <= 0: best['warnings'].append('one component preferred by BIC')
    if min(best['variance']) <= floor * 1.001: best['warnings'].append('variance floor active')
    if min(abs(best['eta'] - x) for x in ETA_BOUNDS) < .001: best['warnings'].append('eta bound active')
    if min(abs(best['phi'] - x) for x in PHI_BOUNDS) < .001: best['warnings'].append('phi bound active')
    return best


def posterior(y, params):
    pi, eta, phi = (params[k] for k in ('pi', 'eta', 'phi'))
    q = pi * eta + (1 - pi) * phi
    lp = _log_components(np.asarray(y, np.float64), np.array(params['mu']), np.array(params['variance']))
    lp += np.log([1 - q, q])
    r = np.exp(lp - logsumexp(lp, axis=1)[:, None])
    return r[:, 1] * pi * eta / q + r[:, 0] * pi * (1 - eta) / (1 - q)


def generate(targets, source_draw, out, max_iter=600):
    start = time.monotonic()
    with np.load(targets, allow_pickle=False) as t:
        T = {k: t[k] for k in ('Y', 'mask', 'split', 'batch', 'hist_row', 'names')}
    cols = np.array([i for i, n in enumerate(T['names']) if str(n).startswith('single:')])
    if not len(cols): raise ValueError('no single markers')
    g = np.flatnonzero((T['batch'] == source_draw) & (T['hist_row'] < 0))
    mask = T['mask'][g][:, cols].astype(bool)
    fm = mask & (T['split'][g, None] == 0)
    if not np.isfinite(T['Y'][g][:, cols][mask]).all():
        raise ValueError('non-finite observed source-draw labels')
    p = np.zeros(mask.shape, np.float32); fits = {}; summaries = {}
    for k, col in enumerate(cols):
        rows = g[fm[:, k]]
        fit = fit_noise_mixture(T['Y'][rows, col], max_iter=max_iter)
        fit['fit_rows_sha256'] = hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest()
        fit['n_targets'] = int(mask[:, k].sum())
        p[mask[:, k], k] = posterior(T['Y'][g[mask[:, k]], col], fit).astype(np.float32)
        values = p[mask[:, k], k]
        summaries[str(T['names'][col])] = dict(posterior_min=float(values.min()),
            posterior_max=float(values.max()), posterior_mean=float(values.mean()))
        fits[str(T['names'][col])] = fit
        print(f"draw={source_draw} {T['names'][col]} n={len(rows)} d'={fit['d_prime']:.3f} "
              f"eta={fit['eta']:.4f} phi={fit['phi']:.4f} delta_BIC={fit['delta_bic_1_minus_2']:.1f}", flush=True)
    meta = dict(schema='n049-v1', source_draw=int(source_draw), source_sha256=source_digest(T, g, cols),
                model='Z -> S -> Y; two tied-variance Gaussian emissions, joint MAP EM + ridge ascent',
                eta_bounds=list(ETA_BOUNDS), phi_bounds=list(PHI_BOUNDS),
                eta_beta=list(ETA_BETA), phi_beta=list(PHI_BETA), pi_prior='Uniform(0,1)',
                identifiability='Only q=pi*eta+(1-pi)*phi is identified; eta/phi/pi are prior-dependent.',
                bic_note='BIC uses 4 identifiable marginal parameters versus 2 for one Gaussian; descriptive MAP plug-in.',
                fit_selection='batch=source_draw AND split=0 AND hist_row<0 AND single-marker mask',
                target_selection='same source draw, hist_row<0, original single-marker mask; all splits',
                fits=fits, target_diagnostics=summaries)
    out = Path(out)
    if out.suffix != '.npz': raise ValueError('--out must end with .npz')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, G=g, P=p, mask=mask, fit_mask=fm,
                        names=T['names'][cols].astype(str), metadata=np.array(json.dumps(meta, sort_keys=True)))
    report = dict(meta, elapsed_seconds=time.monotonic() - start, targets_path=str(targets))
    out.with_suffix('.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--targets', default='/hnvme/workspace/v181bb10-bsccm/targets_v5.npz')
    ap.add_argument('--source-draw', type=int, choices=(0, 1), required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-iter', type=int, default=600)
    a = ap.parse_args()
    if a.max_iter < 2: ap.error('--max-iter must be >=2')
    generate(a.targets, a.source_draw, a.out, a.max_iter)


if __name__ == '__main__': main()
