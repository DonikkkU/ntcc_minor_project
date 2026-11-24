"""Discrimination, calibration and stability measures used in PD validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score, roc_curve)


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def summary(y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {
        "n": int(len(y)),
        "default_rate": float(np.mean(y)),
        "mean_pd": float(np.mean(p)),
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
    }


def bootstrap_auc(y, p, n_boot: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        aucs.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def hosmer_lemeshow(y, p, groups: int = 10):
    df = pd.DataFrame({"y": y, "p": p})
    df["g"] = pd.qcut(df["p"].rank(method="first"), groups, labels=False)
    g = df.groupby("g").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"), mean_p=("p", "mean"))
    hl = (((g.obs - g.exp) ** 2) / (g.exp * (1 - g.mean_p))).sum()
    pval = 1 - stats.chi2.cdf(hl, groups - 2)
    return float(hl), float(pval), g


def psi(expected, actual, bins: int = 10, edges=None) -> float:
    """Population Stability Index"""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if edges is None:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges = edges.copy()
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def binomial_test_grades(y, p, grades) -> pd.DataFrame:
    df = pd.DataFrame({"y": y, "p": p, "grade": grades})
    g = df.groupby("grade").agg(n=("y", "size"), defaults=("y", "sum"), pd_mean=("p", "mean"))
    g["odr"] = g.defaults / g.n
    g["p_value"] = [1 - stats.binom.cdf(d - 1, n, q) for d, n, q in zip(g.defaults, g.n, g.pd_mean)]
    g["critical_defaults_99"] = [stats.binom.ppf(0.99, n, q) for n, q in zip(g.n, g.pd_mean)]
    g["result"] = np.where(g.defaults > g.critical_defaults_99, "red", "green")
    return g.reset_index()
