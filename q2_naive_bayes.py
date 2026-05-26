"""Q2: Gaussian Naive Bayes from scratch (NumPy only)."""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from dataset_generator import (
    high_noise_dataset,
    low_noise_dataset,
    nb_correlated_dataset,
    nb_failure_dataset,
    nb_success_dataset,
    SEED,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


class GaussianNaiveBayes:
    """Gaussian NB with log-probabilities for stability."""

    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.log_priors_ = None
        self.means_ = None
        self.variances_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.log_priors_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.log_priors_[idx] = np.log(len(X_c) / len(y))
            self.means_[idx] = X_c.mean(axis=0)
            self.variances_[idx] = X_c.var(axis=0)

        max_var = self.variances_.max()
        self.variances_ += self.var_smoothing * max_var
        return self

    def _log_likelihood(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_likelihood = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            mu = self.means_[idx]
            var = self.variances_[idx]
            log_norm = -0.5 * np.log(2 * np.pi * var)
            log_gauss = -0.5 * ((X - mu) ** 2) / var
            log_likelihood[:, idx] = (log_norm + log_gauss).sum(axis=1)

        return log_likelihood

    def _log_posterior(self, X):
        return self._log_likelihood(X) + self.log_priors_[None, :]

    def predict(self, X):
        log_post = self._log_posterior(X)
        return self.classes_[np.argmax(log_post, axis=1)]

    def predict_proba(self, X):
        log_post = self._log_posterior(X)
        log_post = log_post - log_post.max(axis=1, keepdims=True)
        exp_post = np.exp(log_post)
        return exp_post / exp_post.sum(axis=1, keepdims=True)


def _calibration_curve(y_true, y_pred, y_proba, n_bins=10):
    max_conf = y_proba.max(axis=1)
    correct = y_pred == y_true
    bins = np.linspace(0, 1, n_bins + 1)
    mean_conf, mean_acc = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (max_conf >= lo) & (max_conf < hi)
        if mask.any():
            mean_conf.append(max_conf[mask].mean())
            mean_acc.append(correct[mask].mean())
    return np.array(mean_conf), np.array(mean_acc), max_conf, correct


def part_b_correlated():
    print("\nPart B: correlated features")
    X, y = nb_correlated_dataset(seed=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    gnb = GaussianNaiveBayes().fit(X_tr, y_tr)
    y_pred = gnb.predict(X_te)
    y_proba = gnb.predict_proba(X_te)

    acc = accuracy_score(y_te, y_pred)
    mean_conf, mean_acc, max_conf, correct = _calibration_curve(y_te, y_pred, y_proba)

    print(f"  Accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred))
    print(f"  Mean conf (correct): {max_conf[correct].mean():.4f}")
    print(f"  Mean conf (wrong):   {max_conf[~correct].mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(max_conf[correct], bins=30, alpha=0.7, label="Correct")
    axes[0].hist(max_conf[~correct], bins=30, alpha=0.7, label="Wrong")
    axes[0].set_title("Confidence distribution")
    axes[0].set_xlabel("Max class probability")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes[1].plot(mean_conf, mean_acc, "o-", label="GNB")
    axes[1].set_title("Calibration curve")
    axes[1].set_xlabel("Mean predicted confidence")
    axes[1].set_ylabel("Fraction correct")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "q2_partB_correlated.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def part_c_counterexample():
    print("\nPart C: counterexamples")

    X_s, y_s = nb_success_dataset(seed=SEED)
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_s, y_s, test_size=0.3, random_state=SEED, stratify=y_s
    )
    gnb_s = GaussianNaiveBayes().fit(X_tr_s, y_tr_s)
    acc_s = accuracy_score(y_te_s, gnb_s.predict(X_te_s))
    print(f"  Success dataset accuracy: {acc_s:.4f}")

    X_f, y_f = nb_failure_dataset(seed=SEED)
    X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(
        X_f, y_f, test_size=0.3, random_state=SEED, stratify=y_f
    )
    gnb_f = GaussianNaiveBayes().fit(X_tr_f, y_tr_f)
    acc_f = accuracy_score(y_te_f, gnb_f.predict(X_te_f))
    print(f"  Failure dataset accuracy: {acc_f:.4f}")

    def decision_boundary_2d(X_full, y_true, ax, title, feat=(0, 1)):
        X2 = X_full[:, list(feat)]
        gnb2 = GaussianNaiveBayes().fit(X2, y_true)

        h = 0.04
        x_min, x_max = X2[:, 0].min() - 1, X2[:, 0].max() + 1
        y_min, y_max = X2[:, 1].min() - 1, X2[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = gnb2.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.35, cmap="coolwarm")
        for cls, col in zip([0, 1], ["#2EC4B6", "#E94560"]):
            mask = y_true == cls
            ax.scatter(X2[mask, 0], X2[mask, 1], s=12, alpha=0.6, color=col)
        ax.set_title(title)
        ax.set_xlabel(f"Feature {feat[0]}")
        ax.set_ylabel(f"Feature {feat[1]}")
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    decision_boundary_2d(X_s, y_s, axes[0], f"Success (acc={acc_s:.3f})")
    decision_boundary_2d(X_f, y_f, axes[1], f"Failure (acc={acc_f:.3f})")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "q2_partC_boundaries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def part_d_conceptual():
    print("\nPart D: conceptual notes")
    print("  NB can outperform complex models on small data because it has few\n"
          "  parameters (low variance) even though it is biased.")
    print("  Correlated features double-count evidence, so posteriors become\n"
          "  overconfident even if accuracy stays similar.")


def main():
    print("\nQ2: Gaussian Naive Bayes")
    for name, dataset_fn in [("Low-Noise", low_noise_dataset), ("High-Noise", high_noise_dataset)]:
        X, y = dataset_fn(seed=SEED)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=SEED, stratify=y
        )
        gnb = GaussianNaiveBayes().fit(X_tr, y_tr)
        acc = accuracy_score(y_te, gnb.predict(X_te))
        proba = gnb.predict_proba(X_te)
        print(f"  {name}: acc={acc:.4f} avg_conf={proba.max(axis=1).mean():.4f}")

    part_b_correlated()
    part_c_counterexample()
    part_d_conceptual()
    print("\nQ2 complete. Plots saved.")


if __name__ == "__main__":
    main()
