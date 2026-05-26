"""Q3: C4.5 decision tree from scratch (NumPy only)."""

import os
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from dataset_generator import (
    dt_greedy_counterexample,
    dt_noisy_dataset,
    high_dimensional_dataset,
    low_noise_dataset,
    SEED,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def entropy(y):
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return float(-np.sum(probs * np.log2(probs)))


def information_gain(y, y_left, y_right):
    n = len(y)
    nl = len(y_left)
    nr = len(y_right)
    if nl == 0 or nr == 0:
        return 0.0
    return entropy(y) - (nl / n) * entropy(y_left) - (nr / n) * entropy(y_right)


def split_information(n, n_left, n_right):
    if n_left == 0 or n_right == 0:
        return 0.0
    p_l = n_left / n
    p_r = n_right / n
    return float(-(p_l * np.log2(p_l) + p_r * np.log2(p_r)))


def gain_ratio(ig, si):
    return 0.0 if si < 1e-10 else ig / si


class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.ig = 0.0
        self.gr = 0.0
        self.depth = 0
        self.n_samples = 0
        self.n_classes = {}

    @property
    def is_leaf(self):
        return self.value is not None


class DecisionTreeC45:
    """Binary C4.5 tree (gain ratio, continuous features)."""

    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_gain_ratio=0.0, n_thresholds=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_gain_ratio = min_gain_ratio
        self.n_thresholds = n_thresholds
        self.root_ = None
        self.n_features_ = None
        self.classes_ = None
        self.feature_ig_ = None
        self.feature_gr_ = None

    def _get_thresholds(self, x):
        unique_vals = np.unique(x)
        if len(unique_vals) < 2:
            return np.array([])
        midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        if self.n_thresholds is not None and len(midpoints) > self.n_thresholds:
            idx = np.linspace(0, len(midpoints) - 1, self.n_thresholds, dtype=int)
            midpoints = midpoints[idx]
        return midpoints

    def _best_split(self, X, y):
        n, d = X.shape
        best = {"gr": self.min_gain_ratio, "ig": 0.0, "si": 0.0,
                "feature": None, "threshold": None}

        for feat in range(d):
            for thresh in self._get_thresholds(X[:, feat]):
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                y_left = y[left_mask]
                y_right = y[right_mask]
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                ig = information_gain(y, y_left, y_right)
                si = split_information(n, len(y_left), len(y_right))
                gr = gain_ratio(ig, si)
                if gr > best["gr"]:
                    best.update({"gr": gr, "ig": ig, "si": si,
                                 "feature": feat, "threshold": thresh})

        return best if best["feature"] is not None else None

    @staticmethod
    def _leaf_value(y):
        return int(np.argmax(np.bincount(y)))

    def _build_tree(self, X, y, depth):
        node = Node()
        node.depth = depth
        node.n_samples = len(y)
        node.n_classes = Counter(y.tolist())

        pure = len(np.unique(y)) == 1
        too_deep = (self.max_depth is not None) and (depth >= self.max_depth)
        too_small = len(y) < self.min_samples_split
        if pure or too_deep or too_small:
            node.value = self._leaf_value(y)
            return node

        split = self._best_split(X, y)
        if split is None:
            node.value = self._leaf_value(y)
            return node

        feat = split["feature"]
        thresh = split["threshold"]
        if depth == 0 and self.feature_ig_ is None:
            self._record_root_metrics(X, y)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        node.feature = feat
        node.threshold = thresh
        node.ig = split["ig"]
        node.gr = split["gr"]
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _record_root_metrics(self, X, y):
        n, d = X.shape
        igs = np.zeros(d)
        grs = np.zeros(d)
        for feat in range(d):
            best_ig = 0.0
            best_gr = 0.0
            for thresh in self._get_thresholds(X[:, feat]):
                lm = X[:, feat] <= thresh
                rm = ~lm
                yl, yr = y[lm], y[rm]
                if len(yl) < 1 or len(yr) < 1:
                    continue
                ig = information_gain(y, yl, yr)
                si = split_information(n, len(yl), len(yr))
                gr = gain_ratio(ig, si)
                best_ig = max(best_ig, ig)
                best_gr = max(best_gr, gr)
            igs[feat] = best_ig
            grs[feat] = best_gr
        self.feature_ig_ = igs
        self.feature_gr_ = grs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.feature_ig_ = None
        self.feature_gr_ = None
        self.root_ = self._build_tree(X, y.astype(int), depth=0)
        if self.feature_ig_ is None:
            self._record_root_metrics(X, y.astype(int))
        return self

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root_) for x in X])

    def get_depth(self):
        def _depth(node):
            if node is None or node.is_leaf:
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root_)

    def get_n_leaves(self):
        def _count(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count(node.left) + _count(node.right)
        return _count(self.root_)

    def print_tree(self, node=None, indent="", max_depth=5):
        if node is None:
            node = self.root_
        if node.depth > max_depth:
            return
        if node.is_leaf:
            print(f"{indent}[leaf] class={node.value} n={node.n_samples} dist={dict(node.n_classes)}")
        else:
            print(f"{indent}[feat {node.feature} <= {node.threshold:.4f}] IG={node.ig:.4f} GR={node.gr:.4f}")
            self.print_tree(node.left, indent + "  L-", max_depth)
            self.print_tree(node.right, indent + "  R-", max_depth)


def part_b_gain_ratio():
    print("\nPart B: gain ratio vs information gain")
    X, y = high_dimensional_dataset(seed=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    dt = DecisionTreeC45(max_depth=8, min_samples_split=10, n_thresholds=30)
    dt.fit(X_tr, y_tr)

    ig_vals = dt.feature_ig_
    gr_vals = dt.feature_gr_
    ig_rank = np.argsort(-ig_vals)
    gr_rank = np.argsort(-gr_vals)

    print("  Top-5 by IG:", ig_rank[:5].tolist())
    print("  Top-5 by GR:", gr_rank[:5].tolist())

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(ig_vals, gr_vals, s=20, alpha=0.7)
    ax.set_title("IG vs GR (root)")
    ax.set_xlabel("Information gain")
    ax.set_ylabel("Gain ratio")
    ax.grid(alpha=0.3)
    path = os.path.join(OUT_DIR, "q3_partB_gain_ratio.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def part_c_overfitting():
    print("\nPart C: overfitting vs depth")
    X, y = high_dimensional_dataset(seed=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    depths = [1, 2, 3, 4, 5, 7, 10, 15, 20, None]
    train_accs, val_accs = [], []
    for d in depths:
        dt = DecisionTreeC45(max_depth=d, min_samples_split=5, min_samples_leaf=2, n_thresholds=20)
        dt.fit(X_tr, y_tr)
        train_accs.append(accuracy_score(y_tr, dt.predict(X_tr)))
        val_accs.append(accuracy_score(y_te, dt.predict(X_te)))
        label = str(d) if d is not None else "full"
        print(f"  depth={label:4s} train={train_accs[-1]:.4f} val={val_accs[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_accs, "o-", label="train")
    ax.plot(val_accs, "s-", label="val")
    ax.set_title("Accuracy vs max_depth")
    ax.set_xlabel("depth index")
    ax.set_ylabel("accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(OUT_DIR, "q3_partC_overfitting.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def part_d_greedy():
    print("\nPart D: greedy counterexample")
    X, y = dt_greedy_counterexample(seed=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    dt = DecisionTreeC45(max_depth=6, min_samples_split=20, min_samples_leaf=15, n_thresholds=50)
    dt.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, dt.predict(X_te))
    print(f"  val_acc={acc:.4f} depth={dt.get_depth()} leaves={dt.get_n_leaves()}")
    print("  Tree (first 3 levels):")
    dt.print_tree(max_depth=3)

    ig_A, ig_B = dt.feature_ig_[0], dt.feature_ig_[1]
    gr_A, gr_B = dt.feature_gr_[0], dt.feature_gr_[1]
    preferred = "Feature 0" if gr_A > gr_B else "Feature 1"
    print(f"  IG(A)={ig_A:.6f} IG(B)={ig_B:.6f}")
    print(f"  GR(A)={gr_A:.6f} GR(B)={gr_B:.6f} -> {preferred}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls, col in zip([0, 1], ["#2EC4B6", "#E94560"]):
        mask = y_te == cls
        ax.scatter(X_te[mask, 0], X_te[mask, 1], s=14, alpha=0.6, color=col)
    ax.axvline(0, color="#FF9F1C", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="#2EC4B6", linestyle="-", linewidth=1.2)
    ax.set_title("Feature A vs B (true labels)")
    ax.set_xlabel("Feature A")
    ax.set_ylabel("Feature B")
    ax.grid(alpha=0.3)
    path = os.path.join(OUT_DIR, "q3_partD_greedy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def part_e_noise():
    print("\nPart E: noise sensitivity")
    X_clean, y_clean = dt_noisy_dataset(seed=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=SEED, stratify=y_clean
    )

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    rng = np.random.default_rng(SEED)

    train_accs, val_accs = [], []
    for frac in noise_levels:
        y_tr_noisy = y_tr.copy()
        n_flip = int(frac * len(y_tr_noisy))
        if n_flip > 0:
            flip_idx = rng.choice(len(y_tr_noisy), size=n_flip, replace=False)
            y_tr_noisy[flip_idx] = 1 - y_tr_noisy[flip_idx]

        X_tr_noisy = X_tr.copy()
        n_out = max(1, int(frac * 0.5 * len(X_tr_noisy)))
        if n_out > 0:
            out_idx = rng.choice(len(X_tr_noisy), size=n_out, replace=False)
            X_tr_noisy[out_idx] = rng.normal(0, 10, (n_out, X_tr_noisy.shape[1]))

        dt = DecisionTreeC45(max_depth=10, min_samples_split=5, n_thresholds=20)
        dt.fit(X_tr_noisy, y_tr_noisy)
        train_accs.append(accuracy_score(y_tr_noisy, dt.predict(X_tr_noisy)))
        val_accs.append(accuracy_score(y_te, dt.predict(X_te)))
        print(f"  noise={frac:.0%} train={train_accs[-1]:.4f} val={val_accs[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(noise_levels, train_accs, "o-", label="train")
    ax.plot(noise_levels, val_accs, "s-", label="val")
    ax.set_title("Accuracy vs noise")
    ax.set_xlabel("noise fraction")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    path = os.path.join(OUT_DIR, "q3_partE_noise.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("\nQ3: C4.5 decision tree")
    X, y = low_noise_dataset(seed=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    dt = DecisionTreeC45(max_depth=5, min_samples_split=5, n_thresholds=25)
    dt.fit(X_tr, y_tr)
    y_pred = dt.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"  depth={dt.get_depth()} leaves={dt.get_n_leaves()} acc={acc:.4f}")
    print(classification_report(y_te, y_pred))
    dt.print_tree(max_depth=3)

    part_b_gain_ratio()
    part_c_overfitting()
    part_d_greedy()
    part_e_noise()
    print("\nQ3 complete. Plots saved.")


if __name__ == "__main__":
    main()
