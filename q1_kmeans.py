"""Q1: k-means from scratch (NumPy only)."""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_generator import kmeans_adversarial_dataset, kmeans_friendly_dataset, SEED

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PALETTE = ["#E94560", "#0F3460", "#16213E", "#533483",
           "#2EC4B6", "#FF9F1C", "#CBFF8C", "#E71D36"]


class KMeans:
    """Simple, vectorized k-means with random or kmeans++ init."""

    def __init__(self, k=3, max_iter=300, tol=1e-4, init="random", random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.history_ = []
        self.n_iter_ = 0

    def _init_centroids(self, X, rng):
        n = X.shape[0]
        if self.init == "kmeans++":
            idx = rng.integers(0, n)
            centroids = [X[idx]]
            for _ in range(1, self.k):
                C = np.array(centroids)
                D2 = np.min(np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2), axis=1)
                probs = D2 / D2.sum()
                centroids.append(X[rng.choice(n, p=probs)])
            return np.array(centroids)
        indices = rng.choice(n, size=self.k, replace=False)
        return X[indices].copy()

    @staticmethod
    def _compute_distances(X, centroids):
        return np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)

    @staticmethod
    def _assign_clusters(distances):
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        k, d = self.k, X.shape[1]
        new_centroids = np.zeros((k, d))
        counts = np.bincount(labels, minlength=k).astype(float)
        np.add.at(new_centroids, labels, X)
        empty = counts == 0
        if empty.any():
            rng = np.random.default_rng(self.random_state)
            new_centroids[empty] = X[rng.integers(0, len(X), size=int(empty.sum()))]
            counts[empty] = 1.0
        return new_centroids / counts[:, None]

    @staticmethod
    def _compute_inertia(X, labels, centroids):
        return float(np.sum((X - centroids[labels]) ** 2))

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        self.centroids_ = self._init_centroids(X, rng)
        self.history_ = []

        for i in range(self.max_iter):
            distances = self._compute_distances(X, self.centroids_)
            labels = self._assign_clusters(distances)
            new_centroids = self._update_centroids(X, labels)

            self.history_.append(self._compute_inertia(X, labels, new_centroids))
            shift = np.max(np.linalg.norm(new_centroids - self.centroids_, axis=1))
            self.centroids_ = new_centroids
            if shift < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter

        self.labels_ = self._assign_clusters(self._compute_distances(X, self.centroids_))
        self.inertia_ = self._compute_inertia(X, self.labels_, self.centroids_)
        return self

    def predict(self, X):
        return self._assign_clusters(self._compute_distances(X, self.centroids_))


def plot_clusters(ax, X, labels, centroids, title, dims=(0, 1)):
    d0, d1 = dims
    for i in range(len(centroids)):
        mask = labels == i
        ax.scatter(X[mask, d0], X[mask, d1], s=18, alpha=0.55,
                   color=PALETTE[i % len(PALETTE)], label=f"C{i}")
    ax.scatter(centroids[:, d0], centroids[:, d1], marker="X", s=140,
               c="white", edgecolors="black", linewidths=1.0, label="Centroids")
    ax.set_title(title)
    ax.set_xlabel(f"Feature {d0}")
    ax.set_ylabel(f"Feature {d1}")
    ax.grid(alpha=0.3)


def _cluster_accuracy(y_true, y_pred):
    from itertools import permutations

    uniq_true = np.unique(y_true)
    uniq_pred = np.unique(y_pred)
    if len(uniq_true) != len(uniq_pred):
        return 0.0
    best = 0.0
    for perm in permutations(uniq_true):
        mapping = dict(zip(uniq_pred, perm))
        mapped = np.vectorize(mapping.get)(y_pred)
        best = max(best, np.mean(mapped == y_true))
    return best


def part_b_adversarial():
    print("\nPart B: adversarial datasets")
    X_f, y_f = kmeans_friendly_dataset(seed=SEED)
    km_f = KMeans(k=len(np.unique(y_f)), init="kmeans++", random_state=SEED).fit(X_f)

    X_a, y_a = kmeans_adversarial_dataset(seed=SEED)
    km_a_raw = KMeans(k=2, init="kmeans++", random_state=SEED).fit(X_a)

    mu, sigma = X_a.mean(axis=0), X_a.std(axis=0)
    sigma[sigma == 0] = 1
    X_a_scaled = (X_a - mu) / sigma
    km_a_scaled = KMeans(k=2, init="kmeans++", random_state=SEED).fit(X_a_scaled)

    acc_f = _cluster_accuracy(y_f, km_f.labels_)
    acc_raw = _cluster_accuracy(y_a, km_a_raw.labels_)
    acc_scaled = _cluster_accuracy(y_a, km_a_scaled.labels_)

    print(f"  Friendly: WCSS={km_f.inertia_:.1f} acc={acc_f:.3f}")
    print(f"  Adversarial raw: WCSS={km_a_raw.inertia_:.1f} acc={acc_raw:.3f}")
    print(f"  Adversarial scaled: WCSS={km_a_scaled.inertia_:.1f} acc={acc_scaled:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    plot_clusters(axes[0, 0], X_f, km_f.labels_, km_f.centroids_,
                  f"Friendly (acc={acc_f:.3f})")
    plot_clusters(axes[0, 1], X_f, y_f, km_f.centroids_, "Friendly (true)")
    plot_clusters(axes[1, 0], X_a, km_a_raw.labels_, km_a_raw.centroids_,
                  f"Adversarial raw (acc={acc_raw:.3f})")
    plot_clusters(axes[1, 1], X_a_scaled, km_a_scaled.labels_, km_a_scaled.centroids_,
                  f"Adversarial scaled (acc={acc_scaled:.3f})")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "q1_partB_adversarial.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def part_c_initialization():
    print("\nPart C: initialization sensitivity")
    X, y = kmeans_friendly_dataset(seed=SEED)
    k = len(np.unique(y))
    n_runs = 20

    models = []
    for run in range(n_runs):
        km = KMeans(k=k, init="random", max_iter=200, random_state=SEED + run * 17)
        km.fit(X)
        models.append(km)
        print(f"  Run {run + 1:2d} | WCSS={km.inertia_:10.2f} | iters={km.n_iter_}")

    inertias = np.array([m.inertia_ for m in models])
    best = int(np.argmin(inertias))
    worst = int(np.argmax(inertias))
    print(f"  Best run: {best + 1}  Worst run: {worst + 1}  Std={inertias.std():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, m in enumerate(models):
        style = "-" if i == best else "--" if i == worst else "-"
        alpha = 0.9 if i in (best, worst) else 0.4
        axes[0].plot(m.history_, linestyle=style, alpha=alpha)
    axes[0].set_title("Convergence (WCSS per iter)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("WCSS")
    axes[0].grid(alpha=0.3)

    axes[1].bar(range(1, n_runs + 1), inertias)
    axes[1].set_title("Final WCSS per run")
    axes[1].set_xlabel("Run")
    axes[1].set_ylabel("WCSS")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "q1_partC_initialization.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("\nQ1: k-means clustering")

    X_f, y_f = kmeans_friendly_dataset(seed=SEED)
    k = len(np.unique(y_f))
    km_pp = KMeans(k=k, init="kmeans++", max_iter=300, random_state=SEED).fit(X_f)
    km_rnd = KMeans(k=k, init="random", max_iter=300, random_state=SEED).fit(X_f)

    print(f"  k-means++ WCSS={km_pp.inertia_:.2f} iters={km_pp.n_iter_}")
    print(f"  random   WCSS={km_rnd.inertia_:.2f} iters={km_rnd.n_iter_}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(km_pp.history_, label="k-means++")
    ax.plot(km_rnd.history_, linestyle="--", label="random")
    ax.set_title("Part A: WCSS vs iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("WCSS")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(OUT_DIR, "q1_partA_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    part_b_adversarial()
    part_c_initialization()
    print("\nQ1 complete. Plots saved.")


if __name__ == "__main__":
    main()
