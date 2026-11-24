"""Master scale, expected loss, Basel IRB capital and Vasicek stress formulas."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

#
GRADE_BOUNDS = [0.005, 0.01, 0.015, 0.02, 0.03, 0.045, 0.07, 0.10, 0.20, 1.0]
GRADE_NAMES = [f"R{i}" for i in range(1, len(GRADE_BOUNDS) + 1)]
DECISION = {**{g: "Approve" for g in GRADE_NAMES[:5]},
            **{g: "Manual review" for g in GRADE_NAMES[5:7]},
            **{g: "Decline" for g in GRADE_NAMES[7:]}}

LGD_BASE = 0.60      # assumed LGD for unsecured retail exposures
LGD_DOWNTURN = 0.70  # assumed downturn LGD used in severe scenarios


def grade(pd_values):
    pd_values = np.atleast_1d(np.asarray(pd_values, dtype=float))
    idx = np.searchsorted(GRADE_BOUNDS, pd_values, side="left")
    idx = np.clip(idx, 0, len(GRADE_NAMES) - 1)
    return np.array(GRADE_NAMES)[idx]


def grade_lower_bounds():
    return [0.0] + GRADE_BOUNDS[:-1]


def retail_correlation(pd_values):
    """Basel II/III asset correlation for exposures"""
    pd_values = np.asarray(pd_values, dtype=float)
    w = (1 - np.exp(-35 * pd_values)) / (1 - np.exp(-35))
    return 0.03 * w + 0.16 * (1 - w)


def vasicek_conditional_pd(pd_values, q, rho=None):
    """PD conditional on the systematic factor being at its q-quantile of stress.

    q = 0.5 gives the median year, q = 0.999 gives the Basel IRB stress level.
    """
    pd_values = np.clip(np.asarray(pd_values, dtype=float), 1e-6, 1 - 1e-6)
    rho = retail_correlation(pd_values) if rho is None else rho
    return norm.cdf((norm.ppf(pd_values) + np.sqrt(rho) * norm.ppf(q)) / np.sqrt(1 - rho))


def irb_capital(pd_values, lgd=LGD_BASE):
    """Basel IRB capital requirement K per unit of EAD for other retail (no maturity adj.)."""
    pd_values = np.clip(np.asarray(pd_values, dtype=float), 0.0003, 1 - 1e-6)
    return lgd * vasicek_conditional_pd(pd_values, 0.999) - lgd * pd_values


def expected_loss(pd_values, lgd=LGD_BASE, ead=1.0):
    return np.asarray(pd_values) * lgd * ead


def lifetime_pd(pd12, years: int = 3):
    """Cumulative PD over ``years`` assuming a flat marginal (conditional) PD."""
    return 1 - (1 - np.asarray(pd12, dtype=float)) ** years


def estimate_rho_from_default_rates(default_rates, n_per_period):

    dr = np.asarray(default_rates, dtype=float)
    n = np.asarray(n_per_period, dtype=float)
    p = dr.mean()
    total_var = dr.var(ddof=1)
    sampling_var = np.mean(p * (1 - p) / n)
    sys_var = max(total_var - sampling_var, 1e-12)
    s2 = sys_var / norm.pdf(norm.ppf(p)) ** 2
    return float(s2 / (1 + s2)), float(total_var), float(sampling_var)
