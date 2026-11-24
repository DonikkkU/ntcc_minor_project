
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_COL = "id"
RN_COL = "rn"

PAYM_COLS = [f"enc_paym_{i}" for i in range(25)]
SHIFTED_PAYM = [11, 20, 24]
ORDINAL_COLS = [
    "pre_since_opened", "pre_since_confirmed", "pre_pterm", "pre_fterm",
    "pre_till_pclose", "pre_till_fclose", "pre_loans_credit_limit",
    "pre_loans_next_pay_summ", "pre_loans_outstanding",
    "pre_loans_max_overdue_sum", "pre_loans_credit_cost_rate",
    "pre_loans5", "pre_loans530", "pre_loans3060", "pre_loans6090",
    "pre_loans90", "pre_util", "pre_over2limit", "pre_maxover2limit",
]
FLAG_COLS = [
    "is_zero_loans5", "is_zero_loans530", "is_zero_loans3060",
    "is_zero_loans6090", "is_zero_loans90", "is_zero_util",
    "is_zero_over2limit", "is_zero_maxover2limit", "pclose_flag", "fclose_flag",
]
LAST_COLS = [
    "pre_since_opened", "pre_loans_credit_limit", "pre_util", "pre_over2limit",
    "pre_maxover2limit", "pre_loans_outstanding", "pre_loans_next_pay_summ",
    "enc_loans_credit_type", "enc_loans_credit_status", "is_zero_loans5",
    "is_zero_loans530", "enc_paym_0", "enc_paym_1", "enc_paym_2",
]
PAY_WINDOWS = (3, 6, 12, 25)
PAY_STATUSES = (0, 1, 2, 3)


@dataclass
class FeatureSpec:

    categories: dict[str, list[int]] = field(default_factory=dict)

    @property
    def raw_columns(self) -> list[str]:
        return list(self.categories)

    def count_feature_names(self) -> list[str]:
        return [f"cnt__{c}_{v}" for c, vals in self.categories.items() for v in vals]

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.categories, f, indent=1)

    @classmethod
    def load(cls, path) -> "FeatureSpec":
        with open(path, encoding="utf-8") as f:
            return cls({k: [int(x) for x in v] for k, v in json.load(f).items()})

    def update(self, df: pd.DataFrame) -> None:
        for c in df.columns:
            if c in (ID_COL, RN_COL):
                continue
            vals = set(self.categories.get(c, [])) | set(np.unique(df[c].to_numpy()).tolist())
            self.categories[c] = sorted(int(v) for v in vals)


def build_features(loans: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    loans = loans.sort_values([ID_COL, RN_COL], kind="stable")
    ids, inv = np.unique(loans[ID_COL].to_numpy(), return_inverse=True)
    n = len(ids)
    out: dict[str, np.ndarray] = {}

    n_loans = np.bincount(inv, minlength=n)
    out["agg__n_loans"] = n_loans.astype(np.uint8 if n_loans.max() < 256 else np.uint16)

    # one-hot counts (baseline representation)
    for c, vals in spec.categories.items():
        x = np.clip(loans[c].to_numpy().astype(np.int64), -1, None)
        lut = np.full(max(max(vals), int(x.max())) + 2, -1, dtype=np.int64)
        lut[np.asarray(vals)] = np.arange(len(vals))
        codes = lut[x]  # x == -1 maps to the last slot, which stays -1
        ok = codes >= 0
        k = len(vals)
        counts = np.bincount(inv[ok] * k + codes[ok], minlength=n * k).reshape(n, k)
        counts = counts.astype(np.uint8 if counts.max() < 256 else np.uint16)
        for j, v in enumerate(vals):
            out[f"cnt__{c}_{v}"] = counts[:, j]

    for c in ORDINAL_COLS:
        x = loans[c].to_numpy().astype(np.float32)
        out[f"agg__{c}_mean"] = (np.bincount(inv, weights=x, minlength=n) / n_loans).astype(np.float32)
        mx = np.zeros(n, dtype=np.float32)
        np.maximum.at(mx, inv, x)
        out[f"agg__{c}_max"] = mx
    for c in FLAG_COLS:
        x = loans[c].to_numpy().astype(np.float32)
        out[f"agg__{c}_share"] = (np.bincount(inv, weights=x, minlength=n) / n_loans).astype(np.float32)

    paym = loans[PAYM_COLS].to_numpy().astype(np.int16)
    paym[:, SHIFTED_PAYM] = np.clip(paym[:, SHIFTED_PAYM] - 1, 0, 3)
    for s in PAY_STATUSES:
        hit = (paym == s)
        cum = np.cumsum(hit, axis=1)
        for w in PAY_WINDOWS:
            cnt = np.bincount(inv, weights=cum[:, w - 1], minlength=n)
            out[f"pay__s{s}_last{w}"] = cnt.astype(np.float32)
            out[f"pay__s{s}_last{w}_share"] = (cnt / (n_loans * w)).astype(np.float32)

    late = (paym == 1) | (paym == 2)
    first_late = np.where(late.any(axis=1), late.argmax(axis=1), 99).astype(np.float32)
    msl = np.full(n, 99, dtype=np.float32)
    np.minimum.at(msl, inv, first_late)
    out["pay__months_since_late"] = msl

    last_idx = np.r_[np.flatnonzero(np.diff(inv)), len(inv) - 1]
    for c in LAST_COLS:
        out[f"last__{c}"] = loans[c].to_numpy()[last_idx].astype(np.uint8)

    feats = pd.DataFrame(out)
    feats.insert(0, ID_COL, ids)
    return feats


def feature_names(spec: FeatureSpec) -> list[str]:
    """Column order produced by :func:`build_features` (without ``id``)."""
    dummy = {c: [0] for c in spec.raw_columns}
    dummy[ID_COL], dummy[RN_COL] = [0], [1]
    return [c for c in build_features(pd.DataFrame(dummy), spec).columns if c != ID_COL]


def baseline_feature_names(spec: FeatureSpec) -> list[str]:
    return spec.count_feature_names()
