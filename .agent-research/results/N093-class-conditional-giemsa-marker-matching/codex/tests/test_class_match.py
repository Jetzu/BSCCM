"""CPU fixture controls and an independent check of the capped real CLI output."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import ot
import torch
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import n093_class_match as run
from n093_data import partition, replay_markers, routing_head
from n093_match import CLASSES, LF_CLASSES, ClassTransport, PairedGSRouter, Students, ranks


class MatchingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(19)
        self.c = np.repeat(np.arange(3), 15)
        self.anchors = rng.normal(size=(45, 7)) + np.eye(3, 7)[self.c] * 4
        self.gs = rng.normal(size=(30, 7))
        self.marker = rng.normal(size=(45, 4)) + 50*self.c[:, None]

    def fit(self):
        return Students(self.anchors, self.c, 1).fit(self.gs, self.marker, self.c, cap=9)

    def test_class_mean_soft_and_transport_class_exclusion(self):
        student = self.fit()
        x = self.gs[:10]
        out, route = student.predict(x)
        p = student.classifier.predict_proba(student.gs_scale.transform(x))
        means = np.stack([self.marker[self.c == c].mean(0) for c in range(3)])
        np.testing.assert_allclose(out['F1'], p @ means)
        self.assertGreater(np.max(abs(out['F1'] - means[route])), 1.)
        for c in range(3):
            markers = self.marker[self.c == c]
            predictions = out['F2'][route == c]
            self.assertTrue(np.all(predictions >= markers.min(0) - 1e-8))
            self.assertTrue(np.all(predictions <= markers.max(0) + 1e-8))
            self.assertEqual(student.audit[c]['lf_n'], len(markers))
            self.assertEqual(student.audit[c]['lf_ot_n'], 9)

    def test_seed_and_eval_batch_invariance(self):
        x = self.anchors[[4, 18, 42]]
        a = self.fit().predict(x)[0]
        b = self.fit().predict(np.vstack((self.gs * 1e3, x[::-1])))[0]
        for arm in a:
            np.testing.assert_allclose(a[arm], b[arm][-3:][::-1], atol=1e-10)

    def test_paired_router_learns_shifted_domain_and_equal_domain_weights(self):
        # PBC separates classes in the first three features. BSCCM separates them in the last three.
        ca = np.repeat(np.arange(3), 12)
        cg = np.repeat(np.arange(3), [10, 60, 20])
        anchors = np.column_stack((np.eye(3)[ca], np.zeros((len(ca), 3))))
        gs = np.column_stack((np.tile([0., 0., 4.], (len(cg), 1)), np.eye(3)[cg]))
        x = gs[[0, 40, 80]]
        old = Students(anchors, ca, 0)
        np.testing.assert_equal(old.classifier.predict(old.gs_scale.transform(x)), [2, 2, 2])
        model = PairedGSRouter(gs, cg, anchors, ca, 0)
        np.testing.assert_allclose(model.scaler.mean_, (gs.mean(0) + anchors.mean(0)) / 2)
        p = model.probabilities(x)
        np.testing.assert_equal(p.argmax(axis=1), [0, 1, 2])
        np.testing.assert_equal(model.probabilities(anchors).argmax(axis=1), ca)
        repeat = PairedGSRouter(gs, cg, anchors, ca, 0)
        np.testing.assert_allclose(repeat.probabilities(x[::-1])[::-1], p, atol=1e-12)
        shuffled = PairedGSRouter(gs, 2-cg, anchors, ca, 0)
        np.testing.assert_equal(shuffled.probabilities(x).argmax(axis=1), [2, 1, 0])

    def test_oracles_use_exact_cell_lf_probabilities_and_shared_profiles(self):
        mu = np.arange(12).reshape(3, 4)
        p = np.array([[.1, .3, .6], [.9, .04, .06], [.31, .38, .31]])
        pred, weights = {}, {}
        run.add_oracles(pred, weights, mu, p)
        np.testing.assert_equal(pred[run.ORACLE_HARD], mu[[2, 0, 1]])
        np.testing.assert_allclose(pred[run.ORACLE_SOFT], sum(p[:, c, None]*mu[c] for c in range(3)))
        np.testing.assert_equal(weights[run.ORACLE_HARD], np.eye(3)[[2, 0, 1]])
        np.testing.assert_equal(weights[run.ORACLE_SOFT], p)

    def test_barycentres_equal_explicit_coupling(self):
        q = np.column_stack((np.arange(9), np.arange(9)[::-1], np.ones(9)))
        x = np.column_stack((np.arange(12), np.arange(12)[::-1], np.ones(12)))
        y = np.column_stack((np.arange(12), np.arange(12)**2)).astype(float)
        model = ClassTransport.fit(q, x, y)
        cost = cdist(ranks(q, np.sort(q, axis=0)), ranks(x, np.sort(x, axis=0)), 'sqeuclidean') / 3
        plan = ot.sinkhorn(ot.unif(len(q)), ot.unif(len(x)), cost, model.epsilon, stopThr=1e-12)
        expected = (plan @ y) / plan.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(model.predict(q), expected, atol=1e-7)
        self.assertGreater(expected[-1, 0] - expected[0, 0], 6)
        self.assertLess(model.marginal_error, 1e-7)

    def test_tied_support_and_missing_class(self):
        model = ClassTransport.fit(np.ones((3, 3)), np.ones((5, 3)), np.arange(5)[:, None])
        np.testing.assert_allclose(model.predict(np.zeros((2, 3))), 2.)
        with self.assertRaisesRegex(ValueError, 'three anchored classes'):
            Students(self.anchors, self.c, 0).fit(self.gs, self.marker, np.minimum(self.c, 1))


class HarnessTests(unittest.TestCase):
    def test_slide_partition_and_reverse_coverage(self):
        t = dict(hist_row=np.array([2, 0, 3, 1, 4, -1, -1, -1, -1]), batch=np.array([0, 0, 1, 1, 0, 0, 1, 1, 0]),
                 split=np.array([0, 1, 2, 0, 2, 0, 0, 1, 2]), slide=np.array([2, 2, 19, 19, 2, 0, 14, 15, 8]))
        gs, lf, held = partition(t, 1)
        np.testing.assert_array_equal(gs, [0])
        np.testing.assert_array_equal(lf, [5])
        np.testing.assert_array_equal(held, [2, 3])
        gs, lf, other = partition(t, 0)
        np.testing.assert_array_equal(gs, [3])
        np.testing.assert_array_equal(lf, [6])
        np.testing.assert_array_equal(np.sort(np.r_[held, other]), np.arange(5))
        t['slide'][6] = 2
        with self.assertRaisesRegex(ValueError, 'held-out slide'):
            partition(t, 0)

    def test_marker_ridge_replay_and_exclusion(self):
        rng = np.random.default_rng(93)
        x = rng.normal(size=(32, 512)).astype(np.float16)
        mask = rng.uniform(size=(32, 13)) > .3
        y = rng.normal(size=mask.shape)
        t = dict(hist_row=np.r_[-np.ones(27), np.arange(5)], split=np.r_[np.zeros(22), np.ones(10)],
                 slide=np.repeat([1, 8], [27, 5]))
        mask[27:] = True
        y[22:] = 1e9
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d)/'targets_v4c.npz', mask=mask, Y=y)
            mu, sd, weight = replay_markers(x, Path(d), t)
        expected_x = x[:22].astype(float)
        np.testing.assert_allclose(mu, expected_x.mean(0))
        np.testing.assert_allclose(sd, expected_x.std(0, ddof=1)+1e-6)
        for j in range(12):
            r = np.flatnonzero(mask[:22, j])
            z = (x[r]-mu)/sd
            yc = y[r, j]-y[r, j].mean()
            zc = z-z.mean(0)
            w = zc.T @ np.linalg.solve(zc @ zc.T + 10*np.eye(len(r)), yc)
            np.testing.assert_allclose(weight[:-1, j], w, atol=1e-10)
            self.assertAlmostEqual(weight[-1, j], y[r, j].mean()-z.mean(0) @ w)

    def test_lf_class_source_and_fallback(self):
        cls = np.tile(np.arange(3), 5)
        x = np.eye(3)[cls]
        ck = dict(m={'c3.weight': torch.eye(3), 'c3.bias': torch.zeros(3)}, cfg={'w_cls': .5})
        (w, b), source = routing_head(ck, x, None, None)
        np.testing.assert_equal((x @ w.T + b).argmax(1), cls)
        self.assertEqual(source['source'], 'checkpoint c3')
        t = dict(batch=np.zeros(15), split=np.zeros(15), hist_row=np.full(15, -1), slide=np.zeros(15))
        t['batch'][9], t['split'][10], t['hist_row'][11], t['slide'][11] = 1, 2, 0, 19
        labels = cls.copy()
        labels[9:] = [9,9,9,9,-1,-1]
        mask = np.ones((15, 13), bool)
        mask[12, 8:12] = False
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d)/'targets_v4c.npz', cls3=labels, mask=mask)
            (w, b), source = routing_head(dict(m={}), x, Path(d), t)
        self.assertEqual(source['n'], 9)
        np.testing.assert_array_equal((x @ w.T + b).argmax(1), cls)

    def test_reference_coverage_and_n058_subset(self):
        p = np.array([[.7,.2,.1], [.1,.8,.1], [.3,.3,.4], [.1,.7,.2]])
        fields = run.routing_fractions({}, 'MLL23', 'all', {'F1':p}, np.arange(4),
                                       np.array([-1, 0, -1, 1]), 'morphology')[1]
        self.assertEqual((fields['n'], fields['n_gated'], fields['routed_n']), (4, 2, 2))
        self.assertEqual(fields['routed_fraction_on_gated'], 1)
        self.assertEqual(fields['gated_fraction'], .5)
        self.assertIsNone(fields['lf_teacher_fraction'])
        small = 1e-14*np.arange(6)[:, None]
        truth = np.array([0., 100., 2., -100., 4., -100.])
        mask = np.array([True, False, True, False, True, False])
        r, n = run.n058_sanity(small, truth, mask)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(r, 1.)
        self.assertEqual(run.n058_sanity(small*0, truth, mask), (None, 3))

    def test_registered_signs_support_and_macro(self):
        harness = run.load_harness(run.HARNESS)
        names = sorted({v[0] for cc in (harness.contrasts.C, harness.contrasts.MC) for v in cc.values()})
        rows = []
        for domain, cc in [('PBC', harness.contrasts.C), ('MLL23', harness.contrasts.MC)]:
            labels = np.array(sorted({c for v in cc.values() for c in v[1]+v[2]}))
            p = np.zeros((len(labels), len(names)))
            if domain == 'MLL23':
                p[labels == 'plasma_cell', names.index('single:CD45')] = -1
            rows.extend(harness.score_contrasts(p, labels, names, domain))
        score = run.macro(rows, 'F1', 't2', 0)
        self.assertEqual(score['n'], 25)
        self.assertEqual(score['PBC10'], .5)
        self.assertAlmostEqual(score['MLL23_15'], .5 + .5/15)
        self.assertAlmostEqual(score['primary16'], .5 + .5/16)
        self.assertAlmostEqual(score['all25'], .5 + .5/25)
        rows[0]['auroc'] = None
        with self.assertRaisesRegex(ValueError, 'missing contrast'):
            run.macro(rows, 'F1', 't0', 0)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def verify_routing(rows, domain, group, teacher, probs, gates, lf=None):
    rr = [r for r in rows if (r['domain'], r['population'], r['teacher']) == (domain, group, teacher)]
    assert len(rr) == 3 * len(probs)
    assert {(r['arm'], r['cls3']) for r in rr} == {(a, c) for a in probs for c in LF_CLASSES}
    mapped = gates >= 0
    for row in rr:
        p = probs[row['arm']]
        c = LF_CLASSES.index(row['cls3'])
        hits = p.argmax(1) == c
        assert int(row['n']) == len(p) and int(row['routed_n']) == hits.sum()
        assert int(row['n_gated']) == mapped.sum()
        for key, values in [('mean_route_weight', p[:, c]), ('routed_fraction', hits),
                            ('gated_fraction', gates[mapped] == c), ('routed_fraction_on_gated', hits[mapped]),
                            ('lf_teacher_fraction', [] if lf is None else lf.argmax(1) == c),
                            ('lf_teacher_mean_probability', [] if lf is None else lf[:, c])]:
            if len(values):
                np.testing.assert_allclose(float(row[key]), np.mean(values))
            else:
                assert row[key] == ''


def verify_real(out):
    audit = json.loads((out/'audit.json').read_text())
    assert audit['mode'] == 'CPU_SMOKE' and audit['anchor_n'] == 48
    assert audit['clinical']['PBC']['registered_n'] == 8538
    assert audit['clinical']['MLL23']['registered_n'] == 7265
    t = run.targets(run.DATA)
    harness = run.load_harness(run.HARNESS)
    pbc = run.PBC(run.DATA, harness)
    fit = [r for r in harness.clinical_manifest(run.DATA, 'PBC', 'fit') if r[2] in CLASSES]
    fit = run.limit_per_class(fit, 16, 0)
    anchors = pbc.get(fit)
    anchor_cls = [CLASSES.index(r[2]) for r in fit]
    gs = np.load(run.DATA/'emb/giemsa_dinobloom_B.npy', mmap_mode='r')
    for key, fold in audit['folds'].items():
        teacher, subdir = key.split('/')
        holdout = int(subdir[-1])
        folder = out/subdir
        assert [fold[k] for k in ('training_lf', 'training_gs', 'test_gs')] == [96, 32, 24]
        summary = [r for r in read_csv(folder/'smoke_per_run.csv') if r['teacher'] == teacher]
        assert [r['arm'] for r in summary] == ['F1', 'F2', 'F1-bsccm']
        clinical = read_csv(folder/'contrasts.csv')
        assert sum(r['teacher'] == teacher for r in clinical) == 75
        for record in summary:
            rows = [r for r in clinical if r['teacher'] == teacher and r['arm'] == record['arm']]
            assert len(rows) == int(record['n']) == 25
            for metric, selected in [('all25', rows), ('PBC10', [r for r in rows if r['domain'] == 'PBC']),
                                     ('MLL23_15', [r for r in rows if r['domain'] == 'MLL23']),
                                     ('primary16', [r for r in rows if r['primary'] == 'True'])]:
                np.testing.assert_allclose(float(record[metric]), np.mean([float(r['auroc']) for r in selected]))

        # Independent replay from raw, paired global IDs and the registered checkpoint.
        source = np.flatnonzero((t['hist_row'] >= 0) & (t['batch'] != holdout) & (t['split'] == 0))
        train = np.sort(np.random.default_rng(93 + holdout).choice(source, 32, replace=False))
        assert run.digest_ids(train) == fold['gs_sha256']
        primary = Path(audit['teachers'][teacher]['primary'])
        ck = torch.load(primary.parent/'ckpt.pt', map_location='cpu', weights_only=False)['m']
        lf_features = np.load(primary/'lf_feats_all.npy', mmap_mode='r')
        def probabilities(ids):
            logits = lf_features[ids].astype('f4') @ ck['c3.weight'].numpy().T + ck['c3.bias'].numpy()
            return softmax(logits.astype(float), axis=1)

        x = np.vstack((gs[t['hist_row'][train]], anchors)).astype(float)
        w = np.repeat([len(x)/(2*len(train)), len(x)/(2*len(anchors))], [len(train), len(anchors)])
        center = np.average(x, axis=0, weights=w)
        sd = np.sqrt(np.average((x-center)**2, axis=0, weights=w))
        sd[sd == 0] = 1.
        model = LogisticRegression(C=.1, max_iter=1000, random_state=0).fit(
            (x-center)/sd, np.r_[probabilities(train).argmax(1), anchor_cls], sample_weight=w)
        fractions = read_csv(folder/'router_fractions.csv')
        with np.load(folder/f'{teacher}_paired.npz') as d:
            if teacher == 't0' and holdout == 1:
                golden = ROOT/'tests/real_smoke/holdout1'
                with np.load(golden/'t0_paired.npz') as legacy:
                    for field in legacy.files:
                        if legacy[field].dtype.kind == 'f':
                            np.testing.assert_allclose(d[field], legacy[field], rtol=1e-6, atol=1e-8)
                        else:
                            np.testing.assert_array_equal(d[field], legacy[field])
                for old in read_csv(golden/'smoke_per_run.csv'):
                    current = next(r for r in summary if r['arm'] == old['arm'])
                    assert current == old
            ids, mu = d['global_row'], d['class_means']
            assert not np.intersect1d(ids, train).size and np.all(t['batch'][ids] == holdout)
            lf = probabilities(ids)
            bs = model.predict_proba((gs[t['hist_row'][ids]].astype(float)-center)/sd)
            np.testing.assert_allclose(d['lf_prob'], lf)
            np.testing.assert_allclose(d['bsccm_prob'], bs, atol=1e-8)
            np.testing.assert_allclose(d['F1-bsccm'], bs @ mu, atol=1e-8)
            np.testing.assert_allclose(d[run.ORACLE_HARD], mu[lf.argmax(1)])
            np.testing.assert_allclose(d[run.ORACLE_SOFT], lf @ mu)
            probs = {'F1':d['pbc_prob'], 'F2':np.eye(3)[d['route']], 'F1-bsccm':bs,
                     run.ORACLE_HARD:np.eye(3)[lf.argmax(1)], run.ORACLE_SOFT:lf}
            observed = t['observed'][ids]
            split2 = observed & (t['split'][ids] == 2)
            populations = [('all', np.ones(len(ids), bool)), ('autofl_observed', observed),
                           ('n058_split2_observed', split2)]
            populations.extend((f'lf_{name}', lf.argmax(1) == c) for c, name in enumerate(LF_CLASSES))
            for group, mask in populations:
                verify_routing(fractions, 'BSCCM', group, teacher, {a:p[mask] for a,p in probs.items()},
                               t['cls3'][ids][mask], lf[mask])
            for filename, mask in [('sanity.csv', observed), ('sanity_n058.csv', split2)]:
                rows = [r for r in read_csv(folder/filename) if r['teacher'] == teacher]
                assert {r['arm'] for r in rows} == set(probs)
                for r in rows:
                    p, y = d[r['arm']][mask, -1], t['autofl'][ids[mask]]
                    assert int(r['n']) == len(y)
                    if len(p)>2 and np.ptp(y) and np.ptp(p):
                        np.testing.assert_allclose(float(r['autofl_r']), np.corrcoef(p,y)[0,1])
                    else:
                        assert r['autofl_r'] == ''
            rows = [r for r in read_csv(folder/'agreement.csv') if r['teacher'] == teacher]
            assert len(rows) == 60
            for r in rows:
                j = list(d['names']).index(r['marker'])
                p = d[r['arm']][:, j]
                expected = np.corrcoef(p, d['teacher'][:, j])[0,1]
                np.testing.assert_allclose(float(r['r_student_teacher']), expected)
        for domain in ('MLL23', 'PBC'):
            rr = [r for r in fractions if r['teacher'] == teacher and r['domain'] == domain]
            assert {r['arm'] for r in rr} == {'F1', 'F2', 'F1-bsccm'}
            for arm in ('F1', 'F2', 'F1-bsccm'):
                overall = [r for r in rr if r['arm'] == arm and r['population'] == 'all']
                by_group = [r for r in rr if r['arm'] == arm and r['population'] != 'all']
                assert sum(int(r['routed_n']) for r in overall) == audit['clinical'][domain]['scored_n']
                for r in overall:
                    subs = [v for v in by_group if v['cls3'] == r['cls3']]
                    assert sum(int(v['routed_n']) for v in subs) == int(r['routed_n'])
                    assert sum(int(v['n_gated']) for v in subs) == int(r['n_gated'])
                    for v in subs:
                        label = harness.CLASS.get(v['population'])
                        if label is None:
                            assert v['gated_fraction'] == '' and int(v['n_gated']) == 0
                        else:
                            assert float(v['gated_fraction']) == (LF_CLASSES.index(v['cls3']) == label)
        print(f'{key} passed: paired-ID classifier replay, oracle profiles, 75 contrasts, '
              '60 marker correlations, both autofl endpoints and router denominators.')


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--verify-real':
        verify_real(Path(sys.argv[2]))
    else:
        unittest.main(verbosity=2)
