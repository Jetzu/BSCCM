import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "n087_cluster_eval.py"
spec = importlib.util.spec_from_file_location("n087_cluster_eval", SCRIPT)
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)


def make_dump(tmp_path, width=260):
    rng = np.random.default_rng(91)
    # Same FOV identifiers recur on different slides, as in the real pack.
    y = np.r_[np.full(12, -1), np.tile(np.arange(10), 32), [-1, -1]]
    hold = np.arange(len(y)) >= 12
    targets = dict(batch=(~hold).astype(int), cls10=y,
                   slide=np.arange(len(y)) // 100, fov=(np.arange(len(y)) // 10) % 10)
    x = rng.normal(size=(10, width))[y] * 3 + rng.normal(0, .05, (len(y), width))
    x[y < 0] = np.nan  # Unlabelled and source rows must be excluded before scoring.
    perm = rng.permutation(len(y))
    arrays = dict(H=x[perm].astype("float16"), G=perm, is_eval=hold[perm].astype("int8"))
    path = tmp_path / "features.npz"
    np.savez(path, **arrays)
    np.savez(tmp_path / "targets.npz", **targets)
    return path, targets, arrays


def test_cluster_cli_all_spaces(tmp_path):
    path, _, _ = make_dump(tmp_path)
    subprocess.run([sys.executable, "-IB", SCRIPT, f"fixture={path}", "--out", tmp_path,
                    "--targets", tmp_path / "targets.npz"], check=True)
    result = pd.read_csv(tmp_path / "cluster.csv").set_index("representation")
    assert set(result.index) == {"native", "pca16", "pca64", "pca256"}
    assert (result.feature_set == "fixture").all()
    assert (result.n_cells == 320).all()
    np.testing.assert_allclose(result[["nmi", "ari", "knn_balanced_accuracy", "knn_mean_fold_balanced_accuracy"]], 1.0)


def test_low_dim_and_fov_folds(tmp_path):
    path, targets, arrays = make_dump(tmp_path, width=8)
    result, ids = cluster.evaluate(path, "ae", targets, 0, None)
    assert [r["status"] for r in result] == ["ok"] + ["insufficient dimensions"] * 3
    assert "ari" not in result[1]
    rows, _, y, groups = cluster.labelled_rows(arrays, targets)
    for train, test in cluster.GroupKFold(5).split(rows, y, groups):
        assert not set(groups[train]).intersection(groups[test])
    with pytest.raises(ValueError, match="different labelled"):
        cluster.evaluate(path, "ae", targets, 0, ids[1:])


def test_rejects_wrong_draw_mixed_folds_and_duplicate_ids(tmp_path):
    _, targets, arrays = make_dump(tmp_path, width=8)
    with pytest.raises(ValueError, match="requires b10"):
        cluster.labelled_rows(dict(arrays, is_eval=1 - arrays["is_eval"]), targets)
    eval_g = arrays["G"][arrays["is_eval"] == 1][0]
    targets["batch"][eval_g] = 1
    with pytest.raises(ValueError, match="requires b10"):
        cluster.labelled_rows(arrays, targets)
    arrays["G"][0] = arrays["G"][1]
    with pytest.raises(ValueError, match="duplicate"):
        cluster.labelled_rows(arrays, targets)
