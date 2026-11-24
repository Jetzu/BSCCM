"""N080 pre-flight: true population fractions per donor under the evaluator's valley thresholds (fitted on each donor itself here,
descriptive) -- whole blood should give neutrophils ~50-70 %, lymphocytes ~20-40 %, monocytes ~2-10 %, eosinophils ~1-6 %."""
import re, numpy as np, pandas as pd, importlib.util, sys
spec = importlib.util.spec_from_file_location('ev', 'n080_eval.py'); src = open('n080_eval.py').read()
src = src[:src.index('RUNS, PACK, OUT = sys.argv[1:4]')] + src[src.index("M = ['CD15'"):src.index('PAIRS = ')]
ns = {'PACK': '/hnvme/workspace/v181bb10-bsccm/external/sbiad452_pack'}; exec(compile(src, 'n080_eval_defs', 'exec'), ns)
lab = np.load('/hnvme/workspace/v181bb10-bsccm/external/sbiad452_pack/labels.npz', allow_pickle=True); I = lab['I_raw'].astype(np.float64)
donor = np.array([int(re.match(r'sample(\d+)', s).group(1)) for s in lab['sample'].astype(str)]); rows = []
for d in (0, 1, 2):
    m = donor == d; c = np.maximum(1.4826 * np.median(np.abs(I[m] - np.median(I[m], 0)), 0), 1.0); Z = np.arcsinh(I[m] / c); Yb = (Z - Z.mean(0)) / Z.std(0)
    t = ns['gate'](I[m])
    fr = pd.Series(t).value_counts(normalize=True); fr['n'] = m.sum(); fr.name = f'donor{d}'; rows.append(fr)
print(pd.concat(rows, axis=1).round(3).to_string())
