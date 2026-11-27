"""N087 arm E: conditional flow-matching (rectified flow) decoder of the marker vector from FROZEN image features vs MSE decoders.
E1 sandbox: S-BIAD452 'full' regime (9 markers on every cell), each donor held out once; features = pooled ResNet-50 features of the
   N080 full-regime model (best_state). Decoders trained on the training donor only: model's own linear head (mu), MLP mean (MSE),
   conditional flow (same MLP size). Test-donor metrics: per-marker r (flow: mean of K samples and one sample), whole-donor 36-pair
   correlation error, within-population pair correlations (n080_eval gating on the donor's own truth), per-marker W1 and sd ratio.
E2 BSCCM: frozen n051_v10_base features (seed 0, both directions); 1-D conditional flow per target (shared net + target embedding, only
   observed targets) vs masked-MSE MLP vs the model's own predictions; GOAL common-11 (asserts the model's P reproduces .6867 / .6592).
usage: python3 -I n087_flow_decoder.py OUT_DIR [--part sandbox|bsccm|both] [--K 16]"""
import sys, os, json, math, argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
REPO = '/home/hpc/v181bb/v181bb10/BSCCM'; sys.path.insert(0, REPO + '/scripts'); sys.path.insert(0, REPO)
W = '/hnvme/workspace/v181bb10-bsccm'; RUNS = W + '/runs'
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--part', default='both'); ap.add_argument('--K', type=int, default=16)
ap.add_argument('--steps', type=int, default=32); ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--hidden', type=int, default=512)
ap.add_argument('--max-train', type=int, default=0); ap.add_argument('--device', default='cuda')
A = ap.parse_args(); OUT = Path(A.out); OUT.mkdir(parents=True, exist_ok=True); DEV = A.device
torch.manual_seed(0); np.random.seed(0)


def pr(a, b):
    a, b = a - a.mean(), b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d > 1e-12 else np.nan


class Cond(nn.Module):   # feature -> conditioning vector
    def __init__(s, dc, h): super().__init__(); s.f = nn.Sequential(nn.Linear(dc, h), nn.GELU(), nn.Linear(h, h))
    def forward(s, c): return s.f(c)


class MeanMLP(nn.Module):
    def __init__(s, dc, dy, h): super().__init__(); s.c = Cond(dc, h); s.o = nn.Sequential(nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, dy))
    def forward(s, c): return s.o(s.c(c))


class Vel(nn.Module):   # v(x_t, t | features [, target id]) ; dy = output dims of x
    def __init__(s, dc, dy, h, n_ids=0):
        super().__init__(); s.c = Cond(dc, h); s.e = nn.Embedding(n_ids, 32) if n_ids else None
        s.n = nn.Sequential(nn.Linear(dy + 1 + h + (32 if n_ids else 0), h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, dy))
    def forward(s, x, t, c, ids=None):
        z = [x, t[:, None], s.c(c)] + ([s.e(ids)] if s.e is not None else [])
        return s.n(torch.cat(z, 1))


def batches(n, bs, ep):
    g = np.random.default_rng(1000 + ep); p = g.permutation(n); return [p[i:i + bs] for i in range(0, n, bs)]


def fit(model, loss_fn, n, epochs, bs=1024, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4); tot = epochs * math.ceil(n / bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=tot); model.train(); hist = []
    for ep in range(epochs):
        s = 0.0
        for b in batches(n, bs, ep):
            l = loss_fn(torch.from_numpy(b).to(DEV)); opt.zero_grad(set_to_none=True); l.backward(); opt.step(); sch.step(); s += float(l) * len(b)
        hist.append(s / n)
    model.eval(); return hist


@torch.no_grad()
def sample(vel, C, dy, K, steps, ids=None, bs=4096):   # Heun integration of dx/dt = v from N(0, I) at t=0 to t=1
    outs = []
    for i in range(0, len(C), bs):
        c = C[i:i + bs]; idb = None if ids is None else ids[i:i + bs]
        cK = c.repeat_interleave(K, 0); iK = None if idb is None else idb.repeat_interleave(K, 0)
        x = torch.randn(len(cK), dy, device=DEV); dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((len(x),), k * dt, device=DEV); v1 = vel(x, t, cK, iK)
            xp = x + dt * v1; v2 = vel(xp, t + dt, cK, iK); x = x + dt * 0.5 * (v1 + v2)
        outs.append(x.view(len(c), K, dy).cpu().numpy())
    return np.concatenate(outs)


def flow_loss(vel, Y, C, ids=None):
    def f(b):
        y = Y[b]; c = C[b]; x0 = torch.randn_like(y); t = torch.rand(len(b), device=DEV); xt = (1 - t[:, None]) * x0 + t[:, None] * y
        return ((vel(xt, t, c, None if ids is None else ids[b]) - (y - x0)) ** 2).mean()
    return f


rows_out = []
# ---------------------------------------------------------------- E1 sandbox
if A.part in ('sandbox', 'both'):
    import train_ifc_sandbox as S
    sys.argv = ['x']; ev = {}
    exec(open(sorted(Path(REPO + '/.agent-research/results').glob('N080-*/n080_eval.py'))[0]).read().split('LYM = (')[0].split('RUNS, PACK, OUT = sys.argv[1:4]')[0])   # imports + pr
    src = open(sorted(Path(REPO + '/.agent-research/results').glob('N080-*/n080_eval.py'))[0]).read()
    gsrc = src[src.index('def valley(x):'):src.index('LYM = (')]
    M9 = S.MARKERS; ix = {m: i for i, m in enumerate(M9)}
    PACK = Path(W + '/external/sbiad452_pack')
    with np.load(PACK / 'labels.npz', allow_pickle=False) as d: labels = {k: d[k] for k in d.files}
    raw = labels['I_raw'].astype(np.float32); images = np.load(PACK / 'lf.npy', mmap_mode='r')
    g_ns = {'np': np, 'ix': ix, 'GaussianMixture': __import__('sklearn.mixture', fromlist=['GaussianMixture']).GaussianMixture}; exec(gsrc, g_ns); gate = g_ns['gate']
    for d in (0, 1, 2):
        args = S.parse_args(['--out', str(OUT / f'tmp_d{d}'), '--test-donor', str(d), '--regime', 'full']); args.device = DEV
        tr, va, te = S.donor_split(labels['sample'], d); masks, _ = S.reveal_masks(np.arange(len(raw)), 'full', args.seed)
        _tr = np.zeros(len(raw), bool); _tr[tr] = True; c_tr = np.ones(9, np.float32)
        for j in range(9):
            v = raw[_tr & masks[:, j], j]; c_tr[j] = max(1.4826 * float(np.median(np.abs(v - np.median(v)))), 1.0)
        _yt = np.arcsinh(raw / c_tr); samp = labels['sample'].astype(str); y = np.zeros_like(_yt); is_te = S.donor_ids(labels['sample']) == d
        for sv in np.unique(samp):
            k = samp == sv
            for j in range(9):
                ref = _yt[k, j] if is_te[k].all() else _yt[k & masks[:, j], j]; y[k, j] = (_yt[k, j] - ref.mean()) / (ref.std() + 1e-6)
        _rs = np.sort(np.random.default_rng(0).choice(tr, min(len(tr), 5000), replace=False)); _cen = np.asarray(images[_rs, :, 32:96, 32:96], dtype=np.float32)
        cs = np.array([max(float(np.std(_cen[:, c][_cen[:, c] != 0])), 1e-3) for c in range(3)], np.float32)
        scale = torch.tensor(cs, device=DEV)[None, :, None, None]
        ck = torch.load(f'{RUNS}/n080_full_d{d}/ckpt.pt', map_location='cpu', weights_only=False)
        assert np.array_equal(ck['tr'], tr) and np.array_equal(ck['te'], te)
        model = S.SandboxModel(0, 'none', None, False); model.load_state_dict(ck['best_state']); model.to(DEV)
        ep = np.load(f'{RUNS}/n080_full_d{d}/eval_preds.npz'); assert np.array_equal(ep['rows'], te)
        assert np.abs(ep['Y'] - y[te]).max() < 1e-3, 'target normalisation differs from the N080 run'
        if A.max_train: tr = np.sort(np.random.default_rng(0).choice(tr, min(A.max_train, len(tr)), replace=False))
        mu_te, f_te, _ = S.predict(model, PACK, te, scale, args); _, f_tr, _ = S.predict(model, PACK, tr, scale, args)
        assert min(pr(mu_te[:, j], ep['P'][:, j]) for j in range(9)) > .995, 'features/predictions differ from the N080 run (bf16 tolerance)'
        m_, s_ = f_tr.mean(0), f_tr.std(0) + 1e-6
        Ctr = torch.tensor((f_tr - m_) / s_, device=DEV); Cte = torch.tensor((f_te - m_) / s_, device=DEV); Ytr = torch.tensor(y[tr], device=DEV)
        mlp = MeanMLP(Ctr.shape[1], 9, A.hidden).to(DEV); fit(mlp, lambda b: ((mlp(Ctr[b]) - Ytr[b]) ** 2).mean(), len(tr), A.epochs)
        with torch.no_grad(): p_mlp = mlp(Cte).cpu().numpy()
        vel = Vel(Ctr.shape[1], 9, A.hidden).to(DEV); hist = fit(vel, flow_loss(vel, Ytr, Ctr), len(tr), A.epochs)
        smp = sample(vel, Cte, 9, A.K, A.steps); p_fm = smp.mean(1); p_fs = smp[:, 0]
        yt = y[te]; tt = gate(raw[te]); Tm = np.isin(tt, ('T4', 'T8', 'Tother')); Lm = np.isin(tt, ('T4', 'T8', 'Tother', 'B', 'NK', 'lym_other'))
        C_true = np.corrcoef(yt.T); iu = np.triu_indices(9, 1)
        for name, P in (('linear head (model)', mu_te), ('MLP mean (MSE)', p_mlp), ('flow, mean of K samples', p_fm), ('flow, one sample', p_fs)):
            r = {m: pr(P[:, ix[m]], yt[:, ix[m]]) for m in M9}
            pe = float(np.abs(np.corrcoef(P.T)[iu] - C_true[iu]).mean())
            pairs = {f'{a}-{b}@{g}': (pr(P[s, ix[a]], P[s, ix[b]]), pr(yt[s, ix[a]], yt[s, ix[b]])) for g, s, a, b in
                     (('T', Tm, 'CD4', 'CD8'), ('lymph', Lm, 'CD3', 'CD19'), ('lymph', Lm, 'CD3', 'CD56'), ('lymph', Lm, 'CD19', 'CD56'))}
            from scipy.stats import wasserstein_distance as w1
            rows_out.append(dict(part='sandbox', test_donor=d, decoder=name, macro_r=float(np.mean(list(r.values()))), pair36_abs_err=pe,
                                 within_pair_abs_err=float(np.mean([abs(p - q) for p, q in pairs.values()])),
                                 CD4_CD8_in_T_pred=pairs['CD4-CD8@T'][0], CD4_CD8_in_T_true=pairs['CD4-CD8@T'][1],
                                 W1_mean=float(np.mean([w1(P[:, j], yt[:, j]) for j in range(9)])), sd_ratio_mean=float(np.mean(P.std(0) / yt.std(0))),
                                 **{f'r_{m}': v for m, v in r.items()}, **{f'pair_{k}': v[0] for k, v in pairs.items()}))
        json.dump(dict(flow_loss=hist), open(OUT / f'flow_hist_sandbox_d{d}.json', 'w'))
        print('sandbox donor', d, 'done', flush=True)
        import pandas as pd; pd.DataFrame(rows_out).to_csv(OUT / 'flow_decoder.csv', index=False)

# ---------------------------------------------------------------- E2 BSCCM (GOAL)
if A.part in ('bsccm', 'both'):
    T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, MK = T['Y'].astype(np.float32), T['mask']
    names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]; Kt = len(names)
    REC = {'b10': .6867, 'b01': .6592}; goal = {}
    for dname in ('b10', 'b01'):
        F = np.load(f'{RUNS}/n051_v10_base_{dname}/features.npz'); H = F['H'].astype(np.float32); G = F['G']; E = F['is_eval'].astype(bool)
        def score(P):   # GOAL per direction: mean over common-11 of Pearson r on eval rows with an observed label
            return float(np.mean([pr(P[MK[G[E], j], j], Y[G[E]][MK[G[E], j], j]) for j in c11]))
        g_model = score(F['P'][E]); assert abs(g_model - REC[dname]) < 2e-4, (dname, g_model)
        tri = np.where(~E)[0]
        if A.max_train: tri = np.sort(np.random.default_rng(0).choice(tri, min(A.max_train, len(tri)), replace=False))
        m_, s_ = H[tri].mean(0), H[tri].std(0) + 1e-6
        Ctr = torch.tensor((H[tri] - m_) / s_, device=DEV); Cev = torch.tensor((H[E] - m_) / s_, device=DEV)
        Ytr = torch.tensor(Y[G[tri]], device=DEV); Mtr = torch.tensor(MK[G[tri]].astype(np.float32), device=DEV)
        mlp = MeanMLP(Ctr.shape[1], Kt, A.hidden).to(DEV)
        fit(mlp, lambda b: (((mlp(Ctr[b]) - Ytr[b]) ** 2) * Mtr[b]).sum() / Mtr[b].sum().clamp(min=1), len(tri), A.epochs)
        with torch.no_grad(): p_mlp = mlp(Cev).cpu().numpy()
        ci, cj = np.nonzero(MK[G[tri]])   # one (cell, observed target) pair per observation
        CI = torch.tensor(ci, device=DEV); CJ = torch.tensor(cj, device=DEV); YV = Ytr[CI, CJ][:, None]
        vel = Vel(Ctr.shape[1], 1, A.hidden, n_ids=Kt).to(DEV)
        def lf(b):
            y = YV[b]; c = Ctr[CI[b]]; x0 = torch.randn_like(y); t = torch.rand(len(b), device=DEV); xt = (1 - t[:, None]) * x0 + t[:, None] * y
            return ((vel(xt, t, c, CJ[b]) - (y - x0)) ** 2).mean()
        hist = fit(vel, lf, len(ci), A.epochs)
        P_fm = np.zeros((E.sum(), Kt), np.float32); P_fs = np.zeros_like(P_fm)
        for j in c11:
            smp = sample(vel, Cev, 1, A.K, A.steps, ids=torch.full((len(Cev),), j, device=DEV, dtype=torch.long))[:, :, 0]
            P_fm[:, j] = smp.mean(1); P_fs[:, j] = smp[:, 0]
        goal[dname] = dict(model=g_model, mlp=score(p_mlp), flow_mean=score(P_fm), flow_one_sample=score(P_fs))
        json.dump(dict(flow_loss=hist), open(OUT / f'flow_hist_bsccm_{dname}.json', 'w')); print('bsccm', dname, goal[dname], flush=True)
    for k in ('model', 'mlp', 'flow_mean', 'flow_one_sample'):
        rows_out.append(dict(part='bsccm', decoder=k, b10=goal['b10'][k], b01=goal['b01'][k], GOAL=(goal['b10'][k] + goal['b01'][k]) / 2))
    import pandas as pd; pd.DataFrame(rows_out).to_csv(OUT / 'flow_decoder.csv', index=False)
print('done', flush=True)
