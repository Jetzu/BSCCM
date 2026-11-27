"""N098 arm 'ensemble heads' (user 11-27: "test them on the full ensemble"; pre-registered in the N098 item before any dump finished).
For every single-marker model of the 8 N085 E8 members (composites: their twopart/fov-re model; 2-seed members: both seeds):
per lineage marker a logistic positive-vs-negative head on that model's frozen pooled features, trained ONLY on cells of the marker's
own single-stain slide in the SOURCE draw (N097 gate rule on source labels), class-balanced, C fixed at 1e-3 (the value the N098
single-model CV chose in 11 of 12 fits). Test logits are z-scored over all test cells, averaged over seeds within a member, then
over the 8 members (equal weight, like E8), and re-z-scored. Pair score = z_a - z_b. Baseline = E8 regression outputs (same pairs).
Sanity: every dump's eval-row predictions must equal the archived eval_preds.npz of that run (same model rebuilt)."""
import glob, itertools, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import n098_lineage_heads as H
B, L, LIN = H.B, H.L, H.LIN
W = B.W + '/runs/'
MEM = {'b10': {'n048_v9_base': ['n048_v9_base_b10'], 'n048_v9_zoom2': ['n048_v9_zoom2_b10'], 'n053_cytoland': ['n053_cytoland_b10'],
               'comp_base': ['n063_tpfov_s2_b10', 'n063_tpfov_s3_b10'], 'comp_z2': ['n065_tpfov_z2_b10'], 'comp_cyto': ['n065_tpfov_cyto_b10'],
               'cnxs_plain': ['n085_cnxs120_plain_s0_b10', 'n085_cnxs120_plain_s1_b10'], 'cnxs_comp': ['n085_cnxs120_tpfov_s0_b10']},
       'b01': {'n048_v9_base': ['n048_v9_base_b01'], 'n048_v9_zoom2': ['n048_v9_zoom2_b01'], 'n053_cytoland': ['n053_cytoland_b01'],
               'comp_base': ['n063_tpfov_s2_b01', 'n063_tpfov_s3_b01'], 'comp_z2': ['n065_tpfov_z2_b01'], 'comp_cyto': ['n065_tpfov_cyto_b01'],
               'cnxs_plain': ['n083_cnxs120_s0_b01', 'n084_cnxs120_s1_b01'], 'cnxs_comp': ['n085_cnxs120_tpfov_s0_b01']}}


def fit(Xs, y, Xt):
    with threadpool_limits(8):
        return LogisticRegression(C=1e-3, class_weight='balanced', max_iter=500).fit(Xs, y).decision_function(Xt)


def zs(v):
    return (v - v.mean()) / v.std()


def run_logits(run, Gt_ref):
    z = np.load(W + run + '/features.npz'); Hh, G, ev, P = z['H'], z['G'], z['is_eval'].astype(bool), z['P']
    ep = np.load(W + run + '/eval_preds.npz')
    assert np.array_equal(G[ev], ep['G']) and np.array_equal(G[ev], Gt_ref), run
    dev = float(np.abs(P[ev] - ep['P']).max()); assert dev < 1e-2, (run, dev)   # same model rebuilt (bf16/fp32 tolerance)
    Xs, Gs, Xt = Hh[~ev].astype(np.float32), G[~ev], Hh[ev].astype(np.float32)
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6; Xs = (Xs - mu) / sd; Xt = (Xt - mu) / sd
    jobs = []
    for _, x in LIN:
        s, ypos = H.gate(Gs, x); jobs.append((x, s, ypos))
    out = Parallel(n_jobs=len(jobs))(delayed(fit)(Xs[s], yp, Xt) for _, s, yp in jobs)
    print(run, 'max |P - archived|', round(dev, 5), flush=True)
    return {x: zs(o) for (x, _, _), o in zip(jobs, out)}


def main():
    rows = []
    for d in ('b10', 'b01'):
        m, Gt = B.members(d); E8 = np.mean([m[k] for k in m], 0)
        member_z = []
        for mem, runs in MEM[d].items():
            per = [run_logits(r, Gt) for r in runs]
            member_z.append({x: np.mean([p[x] for p in per], 0) for _, x in LIN})
        Zh = {x: zs(np.mean([mz[x] for mz in member_z], 0)) for _, x in LIN}
        Zr = (E8 - E8.mean(0)) / E8.std(0)
        gates = {x: H.gate(Gt, x) for _, x in LIN}
        for (na, a), (nb, b) in itertools.combinations(LIN, 2):
            sa, pa = gates[a]; sb, pb = gates[b]
            for arm, sc in (('E8 regression', Zr[:, L.col[a]] - Zr[:, L.col[b]]), ('E8 heads', Zh[a] - Zh[b])):
                rows.append(dict(direction=d, A=na, B=nb, arm=arm, n_A=int(pa.sum()), n_B=int(pb.sum()),
                                 auroc=H.auc(sc[sa[pa]], sc[sb[pb]]), slide_control=H.auc(sc[sa], sc[sb])))
    df = pd.DataFrame(rows); df.to_csv(f'{HERE}/n098_ensemble_pairs.csv', index=False)
    print(df.pivot_table(index=['A', 'B'], columns=['direction', 'arm'], values='auroc').round(3).to_string())
    print(df.groupby(['arm', 'direction'])[['auroc', 'slide_control']].agg(['mean', 'min', 'max']).round(3).to_string())
    tb = df[(df.A == 'T') & (df.B == 'B')].pivot_table(index='arm', columns='direction', values='auroc')
    tb['mean'] = tb.mean(1); print(tb.round(3).to_string())


if __name__ == '__main__':
    main()
