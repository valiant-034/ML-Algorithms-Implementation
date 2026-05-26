"""Dataset generator for ML Assignment 2 (NumPy only)."""

import numpy as np

SEED = 34  # last 3 digits of roll number


def _shuffle(X, y, rng):
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def low_noise_dataset(n=1200, d_informative=10, d_noisy=10, seed=SEED):
    """Binary, well-separated classes with noise features."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0.0, 0.8, size=(n // 2, d_informative))
    X1 = rng.normal(5.0, 0.8, size=(n // 2, d_informative))
    X_info = np.vstack([X0, X1])
    X_noise = rng.normal(0.0, 1.0, size=(n, d_noisy))
    X = np.hstack([X_info, X_noise])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return _shuffle(X, y, rng)


def high_noise_dataset(n=1500, d_informative=5, d_noisy=15,
                       label_noise=0.15, seed=SEED):
    """Noisy features + label flips."""
    rng = np.random.default_rng(seed)
    X0 = rng.normal(0.0, 2.0, size=(n // 2, d_informative))
    X1 = rng.normal(1.5, 2.0, size=(n // 2, d_informative))
    X_info = np.vstack([X0, X1])
    X_noise = rng.normal(0.0, 5.0, size=(n, d_noisy))
    X = np.hstack([X_info, X_noise])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    flip_idx = rng.choice(n, size=int(label_noise * n), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    return _shuffle(X, y, rng)


def high_dimensional_dataset(n=5000, d_informative=10, d_noisy=50,
                             seed=SEED):
    """n>=5000, d>=50, non-linear boundary + noise."""
    rng = np.random.default_rng(seed)
    X_info = rng.normal(0.0, 1.0, size=(n, d_informative))
    y = (X_info[:, 0] * X_info[:, 1] > 0).astype(int)
    y = np.where(X_info[:, 2] > 0.5, y, 1 - y)
    X_noise = rng.normal(0.0, 1.0, size=(n, d_noisy))
    X = np.hstack([X_info, X_noise])
    return _shuffle(X, y, rng)


def kmeans_friendly_dataset(n_per_cluster=500, k=4, seed=SEED):
    """Spherical, well-separated clusters (k=4 by default)."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 10, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 10, 10, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 10, 10, 0, 0, 0, 0, 0, 0],
    ], dtype=float)[:k]

    X_list, y_list = [], []
    for i, center in enumerate(centers):
        X_list.append(rng.normal(center, 0.6, size=(n_per_cluster, 10)))
        y_list.append(np.full(n_per_cluster, i))

    X_clusters = np.vstack(X_list)
    y = np.concatenate(y_list)
    X_noise = rng.normal(0, 1, size=(len(y), 5))
    X = np.hstack([X_clusters, X_noise])
    return _shuffle(X, y.astype(int), rng)


def kmeans_adversarial_dataset(n=2000, seed=SEED):
    """Ellipses + rings in 4D to break k-means assumptions."""
    rng = np.random.default_rng(seed)
    half = n // 2

    A1 = rng.normal([0, 0], [5, 0.3], size=(half // 2, 2))
    A2 = rng.normal([0, 4], [0.3, 5], size=(half // 2, 2))
    X_ellipse = np.vstack([A1, A2])
    y_ellipse = np.array([0] * (half // 2) + [1] * (half // 2))

    angles = rng.uniform(0, 2 * np.pi, size=half)
    r = np.concatenate([
        rng.normal(2, 0.2, size=half // 2),
        rng.normal(5, 0.2, size=half // 2),
    ])
    X_rings = np.column_stack([r * np.cos(angles), r * np.sin(angles)])

    X = np.hstack([X_ellipse, X_rings])
    return _shuffle(X, y_ellipse, rng)


def nb_correlated_dataset(n=2000, seed=SEED):
    """Highly correlated features (NB independence violated)."""
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n)
    X2 = X1 + rng.normal(0, 0.1, n)
    X3 = rng.normal(0, 1, n)
    X_noise = rng.normal(0, 1, (n, 7))
    X = np.column_stack([X1, X2, X3, X_noise])
    y = ((X1 + X3) > 0).astype(int)
    return _shuffle(X, y, rng)


def nb_success_dataset(n=2000, seed=SEED):
    """Correlated but symmetric across classes (NB still works)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    base0 = rng.normal(0, 1, (half, 5))
    base1 = rng.normal(4, 1, (half, 5))

    def _corr_block(base):
        return np.column_stack([
            base[:, 0],
            base[:, 0] * 0.9 + rng.normal(0, 0.1, half),
            base[:, 1],
            base[:, 1] * 0.9 + rng.normal(0, 0.1, half),
            base[:, 2],
        ])

    X_corr = np.vstack([_corr_block(base0), _corr_block(base1)])
    X_noise = rng.normal(0, 1, (n, 10))
    X = np.hstack([X_corr, X_noise])
    y = np.array([0] * half + [1] * half)
    return _shuffle(X, y, rng)


def nb_failure_dataset(n=2000, seed=SEED):
    """XOR-like boundary (NB fails)."""
    rng = np.random.default_rng(seed)
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    y = ((X1 > 0) ^ (X2 > 0)).astype(int)
    X_noise = rng.normal(0, 1, (n, 13))
    X = np.column_stack([X1, X2, X_noise])
    return _shuffle(X, y, rng)


def dt_greedy_counterexample(n=1200, seed=SEED):
    """Feature A looks good by IG but is globally worse than B."""
    rng = np.random.default_rng(seed)
    B = rng.normal(0, 1, n)
    A = rng.normal(0, 3, n)
    y_base = (B > 0).astype(int)
    y = np.where(np.abs(A) > 2.5, 1 - y_base, y_base)
    X_noise = rng.normal(0, 1, (n, 13))
    X = np.column_stack([A, B, X_noise])
    return _shuffle(X, y, rng)


def dt_noisy_dataset(n=1500, d_informative=8, d_noisy=10, seed=SEED):
    """Clean base for noise sensitivity (noise added later)."""
    rng = np.random.default_rng(seed)
    X_info = rng.normal(0, 1, (n, d_informative))
    y = (
        (X_info[:, 0] > 0).astype(int)
        + (X_info[:, 1] > 0.5).astype(int)
        + (X_info[:, 2] < -0.5).astype(int)
    )
    y = (y >= 2).astype(int)
    X_noise = rng.normal(0, 1, (n, d_noisy))
    X = np.hstack([X_info, X_noise])
    return _shuffle(X, y, rng)


def summarize(name, X, y):
    classes, counts = np.unique(y, return_counts=True)
    balance = {int(c): int(n) for c, n in zip(classes, counts)}
    print(f"\n{name}")
    print(f"  shape={X.shape} features={X.shape[1]} samples={X.shape[0]}")
    print(f"  class_balance={balance} mean={X.mean():.4f} std={X.std():.4f}")


if __name__ == "__main__":
    datasets = [
        ("Low-Noise", low_noise_dataset()),
        ("High-Noise", high_noise_dataset()),
        ("High-Dimensional", high_dimensional_dataset()),
        ("k-Means Friendly", kmeans_friendly_dataset()),
        ("k-Means Adversarial", kmeans_adversarial_dataset()),
        ("NB Correlated", nb_correlated_dataset()),
        ("NB Success", nb_success_dataset()),
        ("NB Failure", nb_failure_dataset()),
        ("DT Greedy", dt_greedy_counterexample()),
        ("DT Noisy", dt_noisy_dataset()),
    ]

    print("\nML Assignment 2 | Dataset Generator")
    print(f"Seed = {SEED}")
    for name, (X, y) in datasets:
        summarize(name, X, y)
    print("\nAll datasets generated successfully.\n")
