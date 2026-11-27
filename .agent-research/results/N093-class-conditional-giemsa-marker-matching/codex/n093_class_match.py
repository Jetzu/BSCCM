#!/usr/bin/env -S python3 -I -B
"""Score class-conditioned GS students and paired-LF routing upper bounds with the registered N058 clinical contrasts."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from n093_data import PBC, Teacher, archived_ridge, embed_mll, limit_per_class, partition, targets
from n093_match import CLASSES, LF_CLASSES, PairedGSRouter, Students, sample

DATA = Path('/hnvme/workspace/v181bb10-bsccm')
HARNESS = HERE.parents[1]/'N058-confirm-hub-tt6c-on-mll23-fine-contrasts/code'


def load_harness(directory):
    spec = importlib.util.spec_from_file_location('n058_eval', directory/'n056_eval.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.tr.ANCH) != CLASSES:
        raise ValueError('registered PBC anchors changed')
    return module


def correlation(p, y):
    ok = np.isfinite(p) & np.isfinite(y)
    x, y = p[ok].astype(np.float64), y[ok].astype(np.float64)
    r = None if len(x) < 3 or min(x.std(), y.std()) < 1e-10 else float(np.corrcoef(x, y)[0, 1])
    return r, len(x)


def macro(rows, arm, teacher, seed):
    out = dict(arm=arm, teacher=teacher, seed=seed)
    for key, selected, expected in (
        ('MLL23_15', [r for r in rows if r['domain'] == 'MLL23'], 15),
        ('primary16', [r for r in rows if r['primary']], 16),
        ('PBC10', [r for r in rows if r['domain'] == 'PBC'], 10),
        ('all25', rows, 25),
    ):
        if len(selected) != expected or len({(r['domain'], r['contrast']) for r in selected}) != expected:
            raise ValueError(f'incomplete contrast dictionary: {key}')
        scores = [r['auroc'] for r in selected]
        if any(x is None or not np.isfinite(x) for x in scores):
            raise ValueError(f'missing contrast support: {key}')
        out[key] = float(np.mean(scores))
    return dict(**out, n=len(rows))


def csv_file(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest_ids(ids):
    return hashlib.sha256('\n'.join(map(str, ids)).encode()).hexdigest()


def eval_features(a, harness, pbc, fit_ids):
    clinical, audit = {}, {}
    for domain in ('PBC', 'MLL23'):
        records = harness.clinical_manifest(a.data, domain, 'eval')
        ids = harness.record_ids(records)
        if len(set(ids)) != len(ids) or set(ids).intersection(fit_ids):
            raise ValueError('duplicate evaluation ID or anchor overlap')
        # Compare the complete evaluation universe to the actual registered T run.
        with np.load(a.data/f'runs/n058_main/t0/T_s0/{domain}_T_pred.npz') as ref:
            if not np.array_equal(ids, ref['ids']):
                raise ValueError(f'{domain} manifest differs from N058')
        selected = limit_per_class(records, 2 if a.smoke else None, 93)
        x = pbc.get(selected) if domain == 'PBC' else embed_mll(a.data, selected)
        clinical[domain] = x, np.array([r[2] for r in selected])
        audit[domain] = dict(registered_n=len(ids), scored_n=len(x), registered_sha256=digest_ids(ids),
                             scored_sha256=digest_ids(harness.record_ids(selected)))
    return clinical, audit


ORACLE_HARD = 'F1-oracle-hard_UPPER_BOUND'
ORACLE_SOFT = 'F1-oracle-soft_UPPER_BOUND'


def gs_predictions(student, router, features):
    pred, pbc = student.predict_with_probabilities(features)
    paired = router.probabilities(features)
    pred['F1-bsccm'] = paired @ student.mu
    weights = {'F1': pbc, 'F2': np.eye(3)[pbc.argmax(axis=1)], 'F1-bsccm': paired}
    return pred, weights


def add_oracles(pred, weights, mu, lf_prob):
    hard = lf_prob.argmax(axis=1)
    pred[ORACLE_HARD] = mu[hard]
    pred[ORACLE_SOFT] = lf_prob @ mu
    weights[ORACLE_HARD] = np.eye(3)[hard]
    weights[ORACLE_SOFT] = lf_prob


def routing_fractions(fields, domain, group, weights, indices, gated, gate_source, lf_prob=None):
    """Compare every router over all selected cells and over the covered reference subset."""
    gates = gated[indices]
    mapped = gates >= 0
    ref = None if lf_prob is None else lf_prob[indices]
    n, n_gated = len(gates), int(mapped.sum())
    result = []
    for arm, probs in weights.items():
        p = probs[indices]
        hard = p.argmax(axis=1)
        for c, name in enumerate(LF_CLASSES):
            selected = hard == c
            result.append(dict(**fields, domain=domain, population=group, arm=arm, cls3=name,
                n=n, routed_n=int(selected.sum()), routed_fraction=selected.mean() if n else None,
                mean_route_weight=p[:, c].mean() if n else None, gate_source=gate_source,
                n_gated=n_gated, gated_fraction=(gates[mapped] == c).mean() if n_gated else None,
                routed_fraction_on_gated=selected[mapped].mean() if n_gated else None,
                lf_teacher_fraction=(ref.argmax(axis=1) == c).mean() if ref is not None and n else None,
                lf_teacher_mean_probability=ref[:, c].mean() if ref is not None and n else None))
    return result


def n058_sanity(values, truth, mask):
    # n056_eval.sanity evaluates one held-out slide, split 2, observed v6c column -1.
    p, y = values[mask, -1], truth[mask]
    r = float(np.corrcoef(p, y)[0, 1]) if len(p) > 2 and p.std() > 0 and y.std() > 0 else None
    return r, len(p)


def run(a, out):
    harness = load_harness(a.harness)
    t = targets(a.data)
    names = list(t['names'])
    pairs = {p['teacher']: p for p in json.loads((a.harness/'teachers.json').read_text()) if p['seed'] == 0}
    if set(pairs) != {'t0', 't1', 't2'}:
        raise ValueError('incomplete seed-0 teacher registry')
    gs_feats = np.load(a.data/'emb/giemsa_dinobloom_B.npy', mmap_mode='r')
    if gs_feats.shape != (15712, 768):
        raise ValueError('unexpected GS feature schema')
    pbc = PBC(a.data, harness)
    fit = [r for r in harness.clinical_manifest(a.data, 'PBC', 'fit') if r[2] in CLASSES]
    fit = limit_per_class(fit, 16 if a.smoke else None, a.seed)
    fit_ids = harness.record_ids(fit)
    anchors = pbc.get(fit)
    labels = np.array([CLASSES.index(r[2]) for r in fit])

    archive = archived_ridge(a.data)
    bundles, audit = {}, dict(mode='CPU_SMOKE' if a.smoke else 'FULL', seed=a.seed, anchor_n=len(anchors),
                              anchor_ids_sha256=digest_ids(fit_ids), teachers={}, folds={})
    for teacher in a.teachers:
        lf = Teacher(pairs[teacher], a.data, t, archive)
        audit['teachers'][teacher] = lf.audit
        for holdout in a.holdouts:
            gs, bank, test = partition(t, holdout)
            rng = np.random.default_rng(93 + holdout)
            gs = sample(gs, 32 if a.smoke else None, rng)
            bank = sample(bank, 96 if a.smoke else None, rng)
            test = sample(test, 24 if a.smoke else None, rng)
            marker, lf_class = lf.predict(bank)
            train_gs = gs_feats[t['hist_row'][gs]]
            student = Students(anchors, labels, a.seed).fit(train_gs, marker, lf_class)
            # The same global IDs select the GS features (via hist_row) and their LF targets.
            paired_class = lf.class_probabilities(gs).argmax(axis=1)
            router = PairedGSRouter(train_gs, paired_class, anchors, labels, a.seed)
            reference, _ = lf.predict(test)
            bundles[teacher, holdout] = student, router, test, reference, lf.class_probabilities(test)
            audit['folds'][f'{teacher}/holdout{holdout}'] = dict(holdout_slide=int(t['slide'][test[0]]),
                training_slide=int(t['slide'][gs[0]]), training_gs=len(gs),
                bsccm_router=dict(bsccm_weight=.5, pbc_weight=.5, cls3=list(LF_CLASSES),
                                  paired_class_counts=np.bincount(paired_class, minlength=3).tolist(),
                                  anchor_class_counts=np.bincount(labels, minlength=3).tolist()),
                training_lf=len(bank), test_gs=len(test), gs_sha256=digest_ids(gs),
                bank_sha256=digest_ids(bank), test_sha256=digest_ids(test), classes=student.audit)

    clinical, audit['clinical'] = eval_features(a, harness, pbc, fit_ids)
    for holdout in a.holdouts:
        folder = out/f'holdout{holdout}'
        folder.mkdir()
        per_run, contrasts, sanity, sanity_split2, agreement, fractions = [], [], [], [], [], []
        for teacher in a.teachers:
            student, router, rows, reference, lf_prob = bundles[teacher, holdout]
            slide = int(t['slide'][rows[0]])
            fields = dict(teacher=teacher, seed=a.seed, holdout=holdout, slide=slide)
            scores = {arm: [] for arm in ('F1', 'F2', 'F1-bsccm')}
            for domain, (x, y) in clinical.items():
                predictions, weights = gs_predictions(student, router, x)
                mapped = np.array([harness.CLASS.get(label, -1) for label in y])
                groups = [('all', np.arange(len(y)))] + [(label, np.flatnonzero(y == label)) for label in np.unique(y)]
                for group, indices in groups:
                    fractions.extend(routing_fractions(fields, domain, group, weights, indices, mapped,
                                                       'N058 clinical morphology mapping'))
                for arm, pred in predictions.items():
                    for row in harness.score_contrasts(pred, y, names, domain):
                        scores[arm].append(dict(**row, arm=arm, teacher=teacher, seed=a.seed))
            for arm, clinical_rows in scores.items():
                per_run.append(macro(clinical_rows, arm, teacher, a.seed))
                contrasts.extend(clinical_rows)

            pred, weights = gs_predictions(student, router, gs_feats[t['hist_row'][rows]])
            add_oracles(pred, weights, student.mu, lf_prob)
            observed, truth, split = t['observed'][rows], t['autofl'][rows], t['split'][rows]
            n058_mask = observed & (split == 2)
            populations = [('all', np.ones(len(rows), bool)), ('autofl_observed', observed),
                           ('n058_split2_observed', n058_mask)]
            lf_route = lf_prob.argmax(axis=1)
            populations.extend((f'lf_{name}', lf_route == c) for c, name in enumerate(LF_CLASSES))
            for group, mask in populations:
                fractions.extend(routing_fractions(fields, 'BSCCM', group, weights, np.flatnonzero(mask),
                                                   t['cls3'][rows], 'v4c measured cls3', lf_prob))
            for arm, values in pred.items():
                r, n = correlation(values[observed, -1], truth[observed])
                sanity.append(dict(arm=arm, **fields, target='autofl', n=n, autofl_r=r))
                r, n = n058_sanity(values, truth, n058_mask)
                sanity_split2.append(dict(arm=arm, **fields, target='autofl', n=n, autofl_r=r))
                for j, name in enumerate(names[:-1]):
                    r, n = correlation(values[:, j], reference[:, j])
                    agreement.append(dict(arm=arm, **fields, marker=name, n=n, r_student_teacher=r))
            np.savez_compressed(folder/f'{teacher}_paired.npz', global_row=rows, hist_row=t['hist_row'][rows],
                                names=t['names'], **pred, route=weights['F1'].argmax(axis=1), teacher=reference,
                                pbc_prob=weights['F1'], bsccm_prob=weights['F1-bsccm'], lf_prob=lf_prob,
                                class_means=student.mu, measured_cls3=t['cls3'][rows], split=split,
                                autofl=truth, autofl_observed=observed)
        prefix = 'smoke_' if a.smoke else 'n093_'
        csv_file(folder/(prefix+'per_run.csv'), per_run)
        csv_file(folder/'contrasts.csv', contrasts)
        csv_file(folder/'sanity.csv', sanity)
        csv_file(folder/'agreement.csv', agreement)
        csv_file(folder/'sanity_n058.csv', sanity_split2)
        csv_file(folder/'router_fractions.csv', fractions)
    (out/'audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--harness', type=Path, default=HARNESS)
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    parser.add_argument('--teachers', choices=('t0', 't1', 't2'), nargs='+')
    parser.add_argument('--holdouts', choices=(0, 1), type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--smoke', action='store_true', help='cap students and evaluation samples, use the registered LF teachers')
    a = parser.parse_args()
    a.teachers = a.teachers or (['t0'] if a.smoke else ['t0', 't1', 't2'])
    a.holdouts = a.holdouts or ([1] if a.smoke else [1, 0])
    if len(set(a.teachers)) != len(a.teachers) or len(set(a.holdouts)) != len(a.holdouts):
        parser.error('duplicate teachers/holdouts')
    threads = a.threads if a.threads is not None else (4 if a.smoke else 48)
    if threads < 1 or a.seed < 0:
        parser.error('threads must be positive and seed nonnegative')
    if a.out.exists():
        parser.error('choose a new output directory')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.set_num_threads(threads)
    with threadpool_limits(limits=threads), tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
        stage = Path(tmp)/'run'
        stage.mkdir()
        audit = run(a, stage)
        stage.rename(a.out)
    print(f"{audit['mode']} passed: {len(a.teachers)} teachers, holdouts {a.holdouts}, {a.out}")


if __name__ == '__main__':
    main()
