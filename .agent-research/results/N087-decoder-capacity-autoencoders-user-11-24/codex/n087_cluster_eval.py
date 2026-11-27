"""Evaluate draw-1-trained representations on class-labelled cells of draw 0."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, normalized_mutual_info_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

DEFAULT_TARGETS = "/hnvme/workspace/v181bb10-bsccm/targets_v5cx_v10.npz"
PCA_DIMS = (16, 64, 256)


def labelled_rows(dump, targets):
    g, is_eval = dump["G"], dump["is_eval"]
    if (g.ndim != 1 or g.dtype.kind not in "iu" or not len(g)
            or g.max() >= len(targets["batch"]) or g.min() < 0 or len(np.unique(g)) != len(g)):
        raise ValueError("invalid or duplicate global row IDs")
    if is_eval.shape != g.shape or not np.array_equal(np.unique(is_eval), [0, 1]):
        raise ValueError("expected train and evaluation partitions in is_eval")
    held = is_eval == 1
    if not (np.all(targets["batch"][g[~held]] == 1) and np.all(targets["batch"][g[held]] == 0)):
        raise ValueError("clustering requires b10 features (train draw 1, evaluate draw 0)")
    eligible = np.flatnonzero(held & (targets["cls10"][g] >= 0))
    # Stable row order makes seed and group partitions comparable across encoders.
    take = eligible[np.argsort(g[eligible])]
    rows = g[take]
    if len(rows) < 10:
        raise ValueError("fewer than ten labelled b10 evaluation rows")
    y = targets["cls10"][rows]
    if not np.isin(y, np.arange(10)).all():
        raise ValueError("cls10 outside 0..9")
    groups = np.stack((targets["slide"][rows], targets["fov"][rows]), axis=1)
    _, group_id = np.unique(groups, axis=0, return_inverse=True)
    return rows, take, y, group_id


def cluster_scores(x, y, folds, seed):
    classes = KMeans(n_clusters=10, n_init=10, random_state=seed).fit_predict(x)
    oof = np.empty_like(y)
    fold_scores = []
    for train, test in folds:
        knn = KNeighborsClassifier(n_neighbors=15, metric="cosine", algorithm="brute")
        oof[test] = knn.fit(x[train], y[train]).predict(x[test])
        fold_scores.append(balanced_accuracy_score(y[test], oof[test]))
    return dict(nmi=normalized_mutual_info_score(y, classes),
                ari=adjusted_rand_score(y, classes),
                knn_balanced_accuracy=balanced_accuracy_score(y, oof),
                knn_mean_fold_balanced_accuracy=np.mean(fold_scores))


def evaluate(path, name, targets, seed, expected_rows):
    with np.load(path) as dump:
        rows, take, y, groups = labelled_rows(dump, targets)
        if expected_rows is not None and not np.array_equal(rows, expected_rows):
            raise ValueError(f"{name} has a different labelled draw-0 evaluation cohort")
        h = dump["H"]
        if h.ndim != 2 or len(h) != len(dump["G"]) or not h.shape[1]:
            raise ValueError(f"{path}: H/G shape mismatch")
        x = np.asarray(h[take], dtype=np.float32)
    del h
    if not np.isfinite(x).all():
        raise ValueError(f"{path}: nonfinite evaluated features")
    folds = list(GroupKFold(n_splits=5).split(x, y, groups))
    if min(len(train) for train, _ in folds) < 15:
        raise ValueError("a grouped fold has fewer than 15 training cells")

    result = []
    for dim in (None, *PCA_DIMS):
        record = dict(feature_set=name, representation="native" if dim is None else f"pca{dim}",
                      n_components=x.shape[1] if dim is None else dim, n_cells=len(x),
                      n_fovs=len(np.unique(groups)), status="ok")
        if dim is not None and dim > min(x.shape[1], len(x) - 1):
            record["status"] = "insufficient dimensions"
        else:
            features = x if dim is None else PCA(n_components=dim, random_state=seed).fit_transform(x)
            record.update(cluster_scores(features, y, folds, seed))
        result.append(record)
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", nargs="+", metavar="[NAME=]FILE")
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with np.load(args.targets) as t:
        targets = {k: t[k] for k in ("batch", "cls10", "slide", "fov")}
    records, seen, rows = [], set(), None
    for arg in args.features:
        name, sep, file = arg.partition("=")
        path = Path(file if sep else arg)
        if not sep:
            name = path.parent.name if path.stem == "features" else path.stem
            name = name or path.stem
        if not name or name in seen:
            raise ValueError("use distinct NAME=FILE labels for the input feature sets")
        seen.add(name)
        values, rows = evaluate(path, name, targets, args.seed, rows)
        records.extend(values)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.out / "cluster.csv", index=False)


if __name__ == "__main__":
    main()
