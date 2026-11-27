"""N097 (user 11-27: "why can you see no anticorrelation in your correlation plots?"): per-cell predicted CD3 and CD19 (record ensemble E8)
for every BSCCM test cell, with flags: gated lymphocyte (draw-0 'all' slide cls3), true T (CD3+ on the CD3 slide), true B (CD19+ on
the CD19 slide). Writes cd3cd19_bsccm_cells.csv. Measured CD3 vs CD19 on the same cell exists only in the sandbox (sandbox_lymph_cells.csv)."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); import n097_lineage as L
B = L.B; out = []
for d in ('b10', 'b01'):
    m, G = B.members(d); P = np.mean([m[k] for k in m], 0)
    lin = np.array(['other'] * len(G), dtype=object)
    for x, name in (('CD3', 'T'), ('CD19', 'B')):
        s = np.where(B.M[G, L.col[x]])[0]; y = B.Y[G[s], L.col[x]]; t = L.valley(y); med = np.median(y)
        t = max(t, med + 3 * 1.4826 * np.median(np.abs(y - med))); lin[s[y > t]] = name
    out.append(pd.DataFrame(dict(direction=d, pred_CD3=P[:, L.col['CD3']], pred_CD19=P[:, L.col['CD19']], lymphocyte=B.cls3[G] == 0, lineage=lin)))
df = pd.concat(out); df.to_csv(f'{HERE}/cd3cd19_bsccm_cells.csv', index=False)
for d in ('b10', 'b01'):
    g = df[df.direction == d]
    print(d, 'all', round(B.pr(g.pred_CD3, g.pred_CD19), 3), 'lymph', round(B.pr(g[g.lymphocyte].pred_CD3, g[g.lymphocyte].pred_CD19), 3) if g.lymphocyte.sum() > 50 else '-',
          'T+B', round(B.pr(g[g.lineage != 'other'].pred_CD3, g[g.lineage != 'other'].pred_CD19), 3), dict(g.lineage.value_counts()))
