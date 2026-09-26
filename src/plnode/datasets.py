"""Frozen benchmark datasets (identical to the memoria; seeds fixed).

Both benchmarks share d1 = 2, d' = 5, N = 400, T = 1, K = 10, no augmentation.

1. make_moons classification, MSE with anchors (+-1, 0):
       L(x, y_emb) = 0.5 |x - y_emb|^2,  y_emb = (2y - 1, 0),
   prediction 1{X_1(T) > 0}.

2. California Housing top-2 regression (MedInc, AveOccup), scalar MSE:
       L(x, y) = 0.5 (x_1 - y)^2,
   x_2 is a free coordinate (zero terminal gradient).
"""
import numpy as np
from sklearn.datasets import fetch_california_housing, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

CALIFORNIA_FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def get_moons(n=400, noise=0.12, seed=SEED):
    """Standardized make_moons.  Returns (X, y) with X (n, 2) float64, y {0,1}."""
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    X = StandardScaler().fit_transform(X)
    return X.astype(float), y.astype(float)


def embed_labels(y):
    """y in {0,1} -> anchors (-1, 0) / (+1, 0) in R^2."""
    emb = np.zeros((len(y), 2))
    emb[:, 0] = 2.0 * y - 1.0
    return emb


def moons_problem(n=400, noise=0.12, seed=SEED):
    """Benchmark 1 packaged for the solver: (x0, y_emb, grad_fn, terminal_fn).

    grad_fn/terminal_fn are None: full MSE to the anchors is the solver default.
    """
    X, y = get_moons(n, noise, seed)
    y_emb = embed_labels(y)

    def accuracy(X_T):
        pred = (X_T[:, 0] > 0).astype(float)
        return float((pred == y).mean())

    return {"name": "moons", "x0": X, "y": y_emb, "labels": y,
            "grad_fn": None, "terminal_fn": None, "metric": accuracy,
            "metric_name": "accuracy"}


def get_california_top2(n_train=400, n_test=200, seed=SEED, features=None):
    """California Housing, top-2 features by |corr| with the target
    (MedInc, AveOccup on the frozen subsample), X and y standardized.

    features: optional list of two feature names to use instead of the
    top-2 selection (keeps the regression problem fixed when only the
    subsample changes, e.g. in the genericity probe).
    """
    data = fetch_california_housing()
    X_full = data.data.astype(np.float64)
    y_full = data.target.astype(np.float64)

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_full), size=n_train + n_test, replace=False)
    X_s, y_s = X_full[idx], y_full[idx]

    X_std = StandardScaler().fit_transform(X_s)
    y_std = StandardScaler().fit_transform(y_s.reshape(-1, 1)).ravel()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_std, y_std, train_size=n_train, test_size=n_test, random_state=seed)

    corrs = np.array([np.corrcoef(X_tr[:, j], y_tr)[0, 1]
                      for j in range(X_tr.shape[1])])
    if features is None:
        cols = np.argsort(-np.abs(corrs))[:2]
    else:
        cols = np.array([CALIFORNIA_FEATURES.index(f) for f in features])
    names = [CALIFORNIA_FEATURES[i] for i in cols]
    return (X_tr[:, cols], y_tr, X_te[:, cols], y_te, names)


def housing_problem(n_train=400, n_test=200, seed=SEED, features=None):
    """Benchmark 2 packaged for the solver (scalar MSE on coordinate 1)."""
    X_tr, y_tr, X_te, y_te, names = get_california_top2(n_train, n_test, seed,
                                                        features)
    y_dummy = np.zeros((len(y_tr), 2))   # placeholder, unused with grad_fn

    def grad_scalar(X_T):
        g = np.zeros_like(X_T)
        g[:, 0] = X_T[:, 0] - y_tr
        return g

    def term_scalar(X_T):
        return 0.5 * ((X_T[:, 0] - y_tr) ** 2).mean()

    def r2_train(X_T):
        ss_res = ((y_tr - X_T[:, 0]) ** 2).sum()
        ss_tot = ((y_tr - y_tr.mean()) ** 2).sum()
        return float(1.0 - ss_res / ss_tot)

    return {"name": "housing", "x0": X_tr, "y": y_dummy, "targets": y_tr,
            "X_test": X_te, "y_test": y_te, "features": names,
            "grad_fn": grad_scalar, "terminal_fn": term_scalar,
            "metric": r2_train, "metric_name": "R2_train"}


PROBLEMS = {"moons": moons_problem, "housing": housing_problem}
