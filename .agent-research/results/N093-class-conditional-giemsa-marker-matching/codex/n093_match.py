"""Inductive class means and balanced entropic transport on class-referenced distances."""
from dataclasses import dataclass

import numpy as np
import ot
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


CLASSES = ('lymphocyte', 'neutrophil', 'monocyte')
LF_CLASSES = ('lymphocyte', 'granulocyte', 'monocyte')


def means(x, classes):
    if not np.array_equal(np.unique(classes), np.arange(len(CLASSES))):
        raise ValueError('training support must contain all three anchored classes')
    return np.stack([x[classes == c].mean(axis=0, dtype=np.float64) for c in range(len(CLASSES))])


def sample(rows, cap, rng):
    return rows if cap is None or len(rows) <= cap else np.sort(rng.choice(rows, cap, replace=False))


def ranks(values, ordered):
    # Mid-distribution rank treats tied and constant training coordinates symmetrically.
    return np.column_stack([
        (np.searchsorted(col, v, side='left') + np.searchsorted(col, v, side='right')) / (2 * len(col))
        for col, v in zip(ordered.T, values.T)
    ])


@dataclass
class ClassTransport:
    gs_sorted: np.ndarray
    lf_coords: np.ndarray
    log_v: np.ndarray
    marker: np.ndarray
    epsilon: float
    marginal_error: float

    @classmethod
    def fit(cls, gs_dist, lf_dist, marker, epsilon=.035):
        gs_sorted = np.sort(gs_dist, axis=0)
        q = ranks(gs_dist, gs_sorted)
        x = ranks(lf_dist, np.sort(lf_dist, axis=0))
        cost = cdist(q, x, 'sqeuclidean') / q.shape[1]
        a, b = ot.unif(len(q)), ot.unif(len(x))
        coupling, info = ot.sinkhorn(a, b, cost, epsilon, numItermax=1500, stopThr=1e-10, log=True)
        error = max(abs(coupling.sum(1) - a).max(), abs(coupling.sum(0) - b).max())
        v = info['v']
        if not np.isfinite(error) or error > 1e-7 or not np.all(np.isfinite(v) & (v > 0)):
            raise ValueError('invalid within-class Sinkhorn solution')
        return cls(gs_sorted, x, np.log(v), marker.copy(), epsilon, float(error))

    def predict(self, distances):
        q = ranks(distances, self.gs_sorted)
        cost = cdist(q, self.lf_coords, 'sqeuclidean') / q.shape[1]
        weights = softmax(self.log_v - cost / self.epsilon, axis=1)
        return weights @ self.marker


class Students:
    def __init__(self, anchors, labels, seed):
        self.seed = seed
        self.gs_scale = StandardScaler().fit(anchors)
        self.anchors = self.gs_scale.transform(anchors)
        self.centres = means(self.anchors, labels)
        self.labels = labels.copy()
        self.classifier = LogisticRegression(C=.1, max_iter=1000, random_state=seed).fit(self.anchors, labels)

    def fit(self, train_gs, marker, lf_class, cap=1024):
        """No fit argument contains paired GS-LF rows, clinical test features or evaluation labels."""
        rng = np.random.default_rng(self.seed)
        self.mu = means(marker, lf_class)
        lf_scale = StandardScaler().fit(marker)
        lf_centres = lf_scale.transform(self.mu)
        gs = self.gs_scale.transform(train_gs)
        train_route = self.classifier.predict(gs)
        gs = np.vstack((gs, self.anchors))
        gs_class = np.concatenate((train_route, self.labels))
        self.maps, self.audit = [], []
        for c in range(len(CLASSES)):
            qi = sample(np.flatnonzero(gs_class == c), cap, rng)
            xi = sample(np.flatnonzero(lf_class == c), cap, rng)
            gd = cdist(gs[qi], self.centres)
            ld = cdist(lf_scale.transform(marker[xi]), lf_centres)
            match = ClassTransport.fit(gd, ld, marker[xi])
            self.maps.append(match)
            self.audit.append(dict(cls=CLASSES[c], lf_n=int((lf_class == c).sum()),
                                   gs_slide_n=int((train_route == c).sum()), anchors=int((self.labels == c).sum()),
                                   gs_ot_n=len(qi), lf_ot_n=len(xi), marginal_error=match.marginal_error))
        return self

    def predict_with_probabilities(self, features):
        gs = self.gs_scale.transform(features)
        p = self.classifier.predict_proba(gs)
        f1 = p @ self.mu
        f2 = np.empty_like(f1)
        route = p.argmax(axis=1)
        for c, match in enumerate(self.maps):
            rows = np.flatnonzero(route == c)
            for start in range(0, len(rows), 256):
                r = rows[start:start+256]
                f2[r] = match.predict(cdist(gs[r], self.centres))
        return {'F1': f1, 'F2': f2}, p

    def predict(self, features):
        profiles, probabilities = self.predict_with_probabilities(features)
        return profiles, probabilities.argmax(axis=1)


class PairedGSRouter:
    """One half of the fit weight per domain, uniform within each domain."""
    def __init__(self, train_gs, paired_class, anchors, anchor_class, seed):
        if not len(train_gs) or not len(anchors):
            raise ValueError('both router training domains must have support')
        x = np.vstack((train_gs, anchors)).astype(np.float64)
        y = np.concatenate((paired_class, anchor_class))
        weights = np.r_[np.full(len(train_gs), .5 * len(x) / len(train_gs)),
                        np.full(len(anchors), .5 * len(x) / len(anchors))]
        self.scaler = StandardScaler().fit(x, sample_weight=weights)
        self.classifier = LogisticRegression(C=.1, solver='lbfgs', max_iter=1000, random_state=seed)
        self.classifier.fit(self.scaler.transform(x), y, sample_weight=weights)
        if not np.array_equal(self.classifier.classes_, np.arange(len(CLASSES))):
            raise ValueError('paired GS router requires the three registered classes')
        if self.classifier.n_iter_.max() >= self.classifier.max_iter:
            raise ValueError('paired GS router did not converge')

    def probabilities(self, features):
        return self.classifier.predict_proba(self.scaler.transform(np.asarray(features, dtype=np.float64)))
