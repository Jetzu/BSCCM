"""N096 A+B (pre-registered in the N096 item): BSCCM marker-marker correlation error.
A: correlation error of the N085 record ensemble E8, its members and the n051 single model on every co-measured target pair.
B: decoders on frozen n051_v10_base features: linear MSE control vs MSE + lambda x correlation-matching loss (target = SOURCE-draw
   label correlation of each co-measured pair), plus a training-free recolouring of the control's predictions.
Correlation error of a pair = r(pred_a, pred_b) - r(label_a, label_b) on the test cells where both are measured.
Usage: python3 -I n096_bsccm.py [--quick]   (writes CSVs next to this file)"""
import argparse, itertools, os
import numpy as np, pandas as pd, torch
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
W = '/hnvme/workspace/v181bb10-bsccm'
T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True)
Y, M = T['Y'].astype(np.float32), T['mask'].astype(bool)
names = [str(n) for n in T['names']]
c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]
POOL = [names.index(n) for n in ('grp:CD123/HLA-DR/CD14', 'grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]
AF = names.index('autofl'); SINGLE = list(range(8))
PAIRS = list(itertools.combinations(POOL, 2)) + [(j, AF) for j in SINGLE]
cls3, cls3n = T['cls3'], [str(x) for x in T['cls3_names']]
SLIDE, FOV = T['slide'], T['fov']
MIN = 500   # pairs with fewer co-measured test cells are not scored
LAMBDAS = (0., 1., 10., 100.)


def pr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b)); return float(a @ b / d) if d > 1e-12 else np.nan


def goal_dir(P, G):
    return float(np.mean([pr(P[M[G, j], j], Y[G[M[G, j]], j]) for j in c11]))


def per_target(P, G):
    return {names[j]: pr(P[M[G, j], j], Y[G[M[G, j]], j]) for j in c11}


def corr_rows(P, G, label, direction, within=True):
    rows = []
    for a, b in PAIRS:
        sel = M[G, a] & M[G, b]
        if sel.sum() < MIN:
            continue
        groups = [('all', sel)]
        if within:
            groups += [(n, sel & (cls3[G] == k)) for k, n in enumerate(cls3n)]
        for g, s in groups:
            if s.sum() < MIN:
                continue
            rt, rp = pr(Y[G[s], a], Y[G[s], b]), pr(P[s, a], P[s, b])
            rows.append(dict(model=label, direction=direction, pair=f'{names[a]} ~ {names[b]}',
                             kind='pooled' if b != AF else 'single~autofl', group=g, n=int(s.sum()), r_true=rt, r_pred=rp, err=rp - rt))
    return rows


# ------------------------------------------------------------------ A: record ensemble and members (verbatim N085 member definitions)
L = lambda r, d: np.load(f'{W}/runs/{r}_{d}/eval_preds.npz')
pooled3 = [names.index(n) for n in ('grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16')]


def comp(tp, pa, d):
    a, b = L(tp, d), L(pa, d); assert np.array_equal(a['G'], b['G']); P = a['P'].copy(); P[:, pooled3] = b['P'][:, pooled3]; return P, a['G']


def members(d):
    m = {}
    for r in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland'):
        z = L(r, d); m[r] = z['P']; G = z['G']
    m['comp_base'] = np.mean([comp(f'n063_tpfov_s{s}', f'n063_panel_s{s}', d)[0] for s in (2, 3)], 0)
    m['comp_z2'] = comp('n065_tpfov_z2', 'n065_panel_z2', d)[0]; m['comp_cyto'] = comp('n065_tpfov_cyto', 'n065_panel_cyto', d)[0]
    plain = ('n085_cnxs120_plain_s0', 'n085_cnxs120_plain_s1') if d == 'b10' else ('n083_cnxs120_s0', 'n084_cnxs120_s1')
    m['cnxs_plain'] = np.mean([L(r, d)['P'] for r in plain], 0)
    m['cnxs_comp'] = comp('n085_cnxs120_tpfov_s0', 'n085_cnxs120_panel_s0', d)[0]
    for r in plain + ('n085_cnxs120_tpfov_s0', 'n063_tpfov_s2', 'n065_tpfov_z2', 'n065_tpfov_cyto'):
        assert np.array_equal(L(r, d)['G'], G), r
    return m, G


def part_a():
    rows, goals = [], []
    for d in ('b10', 'b01'):
        m, G = members(d)
        E8 = np.mean([m[k] for k in m], 0)
        for label, P in [('E8 record', E8)] + list(m.items()):
            rows += corr_rows(P, G, label, d); goals.append(dict(model=label, direction=d, GOAL_dir=goal_dir(P, G)))
    return rows, goals


# ------------------------------------------------------------------ B: decoders on frozen features
def inner_split(rows_g, seed=0):
    _, grp = np.unique(np.column_stack((SLIDE[rows_g], FOV[rows_g])), axis=0, return_inverse=True)
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=seed).split(grp, groups=grp))
    return fit, val


def label_corr(Gs):
    """Source-draw label correlation of every co-measured pair (only pairs with >= MIN co-measured source cells)."""
    out = {}
    for a, b in PAIRS:
        s = M[Gs, a] & M[Gs, b]
        if s.sum() >= MIN:
            out[(a, b)] = pr(Y[Gs[s], a], Y[Gs[s], b])
    return out


def torch_corr(pa, pb):
    pa = pa - pa.mean(); pb = pb - pb.mean()
    return (pa * pb).sum() / (pa.square().sum().sqrt() * pb.square().sum().sqrt() + 1e-8)


def train_decoder(X, Gs, fit, lam, C, seed, epochs, bs, corr_bs, lr=1e-3):
    """Linear 2048 -> 13. MSE over observed entries; corr term = mean over pairs of (r_batch(pred_a, pred_b) - C_ab)^2, each pair
    evaluated on its own batch of co-measured fit rows (corr_bs rows per pair, resampled every step)."""
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    net = torch.nn.Linear(X.shape[1], 13)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    y = torch.from_numpy(Y[Gs]); m = torch.from_numpy(M[Gs])
    pair_rows = {k: torch.from_numpy(fit[M[Gs[fit], k[0]] & M[Gs[fit], k[1]]]) for k in C}
    steps = epochs * int(np.ceil(len(fit) / bs)); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    fit_t = torch.from_numpy(fit)
    for ep in range(epochs):
        perm = fit_t[torch.randperm(len(fit_t), generator=g)]
        for rows in perm.split(bs):
            p = net(X[rows]); mm = m[rows]
            loss = ((p - y[rows]).square() * mm).sum() / mm.sum().clamp_min(1)
            if lam > 0:
                cl = []
                for (a, b), pool in pair_rows.items():
                    r = pool[torch.randint(len(pool), (corr_bs,), generator=g)]
                    q = net(X[r]); cl.append((torch_corr(q[:, a], q[:, b]) - C[(a, b)]) ** 2)
                loss = loss + lam * torch.stack(cl).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return net


def recolour(Ps, Gs, fit, C, Pt):
    """Training-free: pooled block -> ZCA whiten with the predicted correlation, recolour to the source label correlation (rows where
    every block target is measured); single:X -> anchored on autofl: z_X' = r_t z_A + sqrt(1-r_t^2) (z_X - r_p z_A)/sqrt(1-r_p^2).
    Everything is fitted on SOURCE fit rows; the transform is applied to the target-draw predictions Pt."""
    out = Pt.copy()
    blk = [k for k in POOL if M[Gs[fit], k].sum() >= MIN]
    s = fit[M[Gs[fit]][:, blk].all(1)]
    mu, sd = Ps[s][:, blk].mean(0), Ps[s][:, blk].std(0)
    Rp = np.corrcoef(Ps[s][:, blk].T); Ry = np.corrcoef(Y[Gs[s]][:, blk].T)
    def msqrt(A, inv=False):
        w, V = np.linalg.eigh(A); w = np.clip(w, 1e-6, None); return (V * (w ** (-.5 if inv else .5))) @ V.T
    A = msqrt(Rp, inv=True) @ msqrt(Ry)
    out[:, blk] = (((Pt[:, blk] - mu) / sd) @ A) * sd + mu
    for j in SINGLE:
        sj = fit[M[Gs[fit], j] & M[Gs[fit], AF]]
        if len(sj) < MIN:
            continue
        mx, sx, ma, sa = Ps[sj, j].mean(), Ps[sj, j].std(), Ps[sj, AF].mean(), Ps[sj, AF].std()
        rp, rt = pr(Ps[sj, j], Ps[sj, AF]), pr(Y[Gs[sj], j], Y[Gs[sj], AF])
        zx, za = (Pt[:, j] - mx) / sx, (Pt[:, AF] - ma) / sa
        out[:, j] = (rt * za + np.sqrt(1 - rt ** 2) * (zx - rp * za) / np.sqrt(1 - rp ** 2)) * sx + mx
    return out


def summarise(rows):
    df = pd.DataFrame(rows); a = df[df.group == 'all']
    return dict(err_all=a.err.abs().mean(), err_pooled=a[a.kind == 'pooled'].err.abs().mean(),
                err_single_autofl=a[a.kind == 'single~autofl'].err.abs().mean())


def part_b(quick):
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    rows, summ, pt = [], [], []
    for d in ('b10', 'b01'):
        z = np.load(f'{W}/runs/n051_v10_base_{d}/features.npz')
        H, G, ev = z['H'], z['G'], z['is_eval'].astype(bool)
        if quick:
            keep = np.random.default_rng(0).random(len(G)) < .15; H, G, ev = H[keep], G[keep], ev[keep]
        Gs, Gt = G[~ev], G[ev]
        fit, val = inner_split(Gs)
        Xs = torch.from_numpy(H[~ev].astype(np.float32)); Xt = torch.from_numpy(H[ev].astype(np.float32))
        mu, sd = Xs[fit].mean(0), Xs[fit].std(0).clamp_min(1e-6); Xs = (Xs - mu) / sd; Xt = (Xt - mu) / sd
        C = label_corr(Gs[fit])
        # Floor for any inductive matching: a predictor that reproduces the SOURCE-draw label correlation exactly still differs from
        # the target-draw label correlation by |C_src - C_tgt| (pairs measured in both draws).
        Ct = label_corr(Gt)
        fl = [dict(model='source-draw labels', direction=d, pair=f'{names[a]} ~ {names[b]}', kind='pooled' if b != AF else 'single~autofl',
                   group='all', n=0, r_true=Ct[(a, b)], r_pred=C[(a, b)], err=C[(a, b)] - Ct[(a, b)]) for a, b in C if (a, b) in Ct]
        rows += fl; summ.append(dict(arm='source-draw label corr (floor)', lam=np.nan, direction=d, val_mean_r=np.nan, val_err=np.nan,
                                     GOAL_dir=np.nan, **summarise(fl)))
        val_lab = [(k, M[Gs[val], k[0]] & M[Gs[val], k[1]]) for k in C]
        nets = {}
        for lam in LAMBDAS:
            net = train_decoder(Xs, Gs, fit, lam, C, seed=0, epochs=2 if quick else 10, bs=1024, corr_bs=1024)
            with torch.no_grad():
                Pv, Pt = net(Xs[val]).numpy(), net(Xt).numpy()
                Ps = net(Xs).numpy() if lam == 0 else None
            vr = float(np.mean([pr(Pv[M[Gs[val], j], j], Y[Gs[val][M[Gs[val], j]], j]) for j in c11]))
            verr = float(np.mean([abs(pr(Pv[s, a], Pv[s, b]) - pr(Y[Gs[val][s], a], Y[Gs[val][s], b])) for (a, b), s in val_lab if s.sum() >= 50]))
            arm = f'corr lambda {lam:g}' if lam else 'control (MSE)'
            nets[lam] = vr
            r_ = corr_rows(Pt, Gt, arm, d); rows += r_
            summ.append(dict(arm=arm, lam=lam, direction=d, val_mean_r=vr, val_err=verr, GOAL_dir=goal_dir(Pt, Gt), **summarise(r_)))
            pt += [dict(arm=arm, direction=d, target=k, r=v) for k, v in per_target(Pt, Gt).items()]
            if lam == 0:
                Prc = recolour(Ps, Gs, fit, C, Pt); r_ = corr_rows(Prc, Gt, 'recolour (post hoc)', d); rows += r_
                summ.append(dict(arm='recolour (post hoc)', lam=np.nan, direction=d, val_mean_r=np.nan, val_err=np.nan, GOAL_dir=goal_dir(Prc, Gt), **summarise(r_)))
                pt += [dict(arm='recolour (post hoc)', direction=d, target=k, r=v) for k, v in per_target(Prc, Gt).items()]
            print(d, arm, summ[-1] if lam else summ[-2], flush=True)
        ok = [l for l in LAMBDAS if l > 0 and nets[l] >= nets[0.] - .005]
        sel = max(ok) if ok else None
        for s_ in summ:
            if s_['direction'] == d:
                s_['selected'] = (s_['lam'] == sel)
    return rows, summ, pt


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); ap.add_argument('--skip-a', action='store_true')
    a = ap.parse_args(); sfx = '_quick' if a.quick else ''
    if a.quick:
        MIN = 100   # smoke test only: the 15 % subset has < 500 co-measured cells per pair
    if not a.skip_a:
        ra, ga = part_a()
        pd.DataFrame(ra).to_csv(f'{HERE}/a_corr_errors{sfx}.csv', index=False); pd.DataFrame(ga).to_csv(f'{HERE}/a_goal{sfx}.csv', index=False)
        da = pd.DataFrame(ra); print(da[(da.model == 'E8 record') & (da.group == 'all')].round(3).to_string(index=False))
    rb, sb, pt = part_b(a.quick)
    pd.DataFrame(rb).to_csv(f'{HERE}/b_corr_errors{sfx}.csv', index=False); pd.DataFrame(pt).to_csv(f'{HERE}/b_per_target{sfx}.csv', index=False)
    S = pd.DataFrame(sb); S.to_csv(f'{HERE}/b_summary{sfx}.csv', index=False)
    G = S.groupby('arm')[['GOAL_dir', 'err_all', 'err_pooled', 'err_single_autofl']].mean().round(4); print(G.to_string())
