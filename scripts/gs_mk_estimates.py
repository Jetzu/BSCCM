"""Transitive Giemsa->marker estimates (N017 method) from cross-slide GS students' result.json, plus the one real check: the same product
estimate for autofl, r_est = r(student, teacher) * r(teacher, truth on the GS cells), vs the student's MEASURED autofl truth r.
Markers: r_est = r(S,T) on the held-out GS slide * r(T,M) on held-out stained cells of that batch (bound/<m>.teacher_r)."""
import argparse, json, csv
ap = argparse.ArgumentParser(); ap.add_argument('--runs', nargs='+', required=True, help='label=runs_dir_name'); ap.add_argument('--out', required=True); a = ap.parse_args()
W = '/hnvme/workspace/v181bb10-bsccm/runs'; rows = []
for spec in a.runs:
    lab, run = spec.split('=')
    d = json.load(open(f'{W}/{run}/result.json')); f = d['final']; h = d['args']['holdout']; direction = {'batch1': 'b0->b1', 'batch0': 'b1->b0'}[h]
    for k, v in f.items():
        if k.startswith('bound/'):
            m = k[6:]; rst = f[f'student_vs_teacher_r/{m}']
            rows.append(dict(chain=lab, direction=direction, marker=m, r_student_teacher=rst, r_teacher_truth=v['teacher_r'], r_est=rst * v['teacher_r'], r_measured=''))
    rst, rtt, rs = f['student_vs_teacher_r/autofl'], f['autofl_truth_r/teacher_LF'], f['autofl_truth_r/student_GS']
    rows.append(dict(chain=lab, direction=direction, marker='autofl (GS truth)', r_student_teacher=rst, r_teacher_truth=rtt, r_est=rst * rtt, r_measured=rs))
with open(a.out, 'w') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['chain']:4s} {r['direction']} {r['marker']:24s} r(S,T) {r['r_student_teacher']:.2f} r(T,M) {r['r_teacher_truth']:.2f} est {r['r_est']:.2f} {('measured %.2f' % r['r_measured']) if r['r_measured'] != '' else ''}")
