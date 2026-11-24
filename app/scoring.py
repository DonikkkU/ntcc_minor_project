"""Model loading, scoring and explanation used by the Streamlit app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pd_scoring.config import MODELS_DIR, RESULTS_DIR, SPEC_PATH  # noqa: E402
from pd_scoring.features import PAYM_COLS, SHIFTED_PAYM, FeatureSpec, build_features  # noqa: E402
from pd_scoring.labels import describe  # noqa: E402
from pd_scoring.risk import DECISION, grade  # noqa: E402

APP_ASSETS = ROOT / "artifacts" / "app"

THEMES = {
    "Payment discipline": ("enc_paym", "pay__", "is_zero_loans", "pre_loans5", "pre_loans530",
                           "pre_loans3060", "pre_loans6090", "pre_loans90"),
    "Overdue amounts": ("pre_loans_total_overdue", "pre_loans_max_overdue_sum", "over2limit"),
    "Credit utilisation": ("pre_util", "is_zero_util", "pre_loans_outstanding", "pre_loans_credit_limit",
                           "pre_loans_next_pay_summ"),
    "Credit mix and status": ("enc_loans", "pre_loans_credit_cost_rate"),
    "History length and terms": ("pre_since", "pre_pterm", "pre_fterm", "pre_till", "close_flag", "n_loans"),
}


def theme_of(feature: str) -> str:
    for theme, keys in THEMES.items():
        if any(k in feature for k in keys):
            return theme
    return "Other"


def to_raw(loans: pd.DataFrame) -> pd.DataFrame:
    """Convert user-facing payment codes (0-3 in every month) to the stored encoding."""
    raw = loans.copy()
    for i in SHIFTED_PAYM:
        raw[PAYM_COLS[i]] = raw[PAYM_COLS[i]] + 1
    return raw


def from_raw(loans: pd.DataFrame) -> pd.DataFrame:
    friendly = loans.copy()
    for i in SHIFTED_PAYM:
        friendly[PAYM_COLS[i]] = np.clip(friendly[PAYM_COLS[i]] - 1, 0, 3)
    return friendly


class Scorer:
    def __init__(self):
        meta = json.load(open(APP_ASSETS / "final_model.json"))
        self.name = meta["name"]
        self.meta = meta
        self.spec = FeatureSpec.load(SPEC_PATH)
        self.cols = json.load(open(MODELS_DIR / "selected_features.json"))
        self.model = CatBoostClassifier()
        self.model.load_model(str(MODELS_DIR / f"{self.name}.cbm"))
        self.defaults = json.load(open(APP_ASSETS / "defaults.json"))

    def features(self, raw_loans: pd.DataFrame) -> pd.DataFrame:
        df = raw_loans.copy()
        for c in self.spec.raw_columns:
            if c not in df:
                df[c] = self.defaults[c]
        df = df[["id", "rn"] + self.spec.raw_columns].astype("int32")
        return build_features(df, self.spec)

    def score(self, raw_loans: pd.DataFrame) -> pd.DataFrame:
        f = self.features(raw_loans)
        p = self.model.predict_proba(f[self.cols].to_numpy(np.float32))[:, 1]
        g = grade(p)
        return pd.DataFrame({"id": f["id"].to_numpy(), "pd": p, "grade": g,
                             "decision": [DECISION[x] for x in g]})

    def explain(self, raw_loans: pd.DataFrame):
        """SHAP contributions (log-odds) for a single applicant."""
        f = self.features(raw_loans)
        X = f[self.cols].to_numpy(np.float32)[:1]
        sv = self.model.get_feature_importance(Pool(X, feature_names=self.cols), type="ShapValues")[0]
        base, contrib = sv[-1], sv[:-1]
        df = pd.DataFrame({"feature": self.cols, "value": X[0], "shap": contrib})
        df["description"] = df.feature.map(describe)
        df["theme"] = df.feature.map(theme_of)
        return float(base), df.reindex(df.shap.abs().sort_values(ascending=False).index)


def load_results() -> dict:
    out = {}
    for name in ("exp1_baseline", "exp3_models", "exp4_validation", "exp6_stress"):
        p = RESULTS_DIR / f"{name}.json"
        if p.exists():
            out[name] = json.load(open(p))
    return out
