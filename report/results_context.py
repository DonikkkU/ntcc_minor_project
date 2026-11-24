"""Collect every number the report text needs into one dictionary."""
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "artifacts" / "results"


def _j(name):
    p = RES / f"{name}.json"
    return json.load(open(p)) if p.exists() else {}


def load_context():
    R = {
        "exp1": _j("exp1_baseline"), "exp2": _j("exp2_feature_selection"), "exp3": _j("exp3_models"),
        "exp4": _j("exp4_validation"), "exp5": _j("exp5_shap"), "exp6": _j("exp6_stress"),
        "eda": _j("eda_stats"),
    }
    dr = pd.read_csv(RES / "default_rate_by_partition.csv")
    R["dr_parts"] = dr.to_dict(orient="records")
    tgt = pd.read_csv(ROOT / "data" / "train_target.csv")
    R["n_apps"] = len(tgt)
    R["n_defaults"] = int(tgt.flag.sum())
    R["default_rate"] = float(tgt.flag.mean())
    R["n_rows"] = 26_162_717
    spec = json.load(open(ROOT / "artifacts" / "feature_spec.json"))
    R["n_count_features"] = sum(len(v) for v in spec.values())
    imp = pd.read_csv(RES / "feature_importance_lgb.csv")
    R["n_features_all"] = len(imp)
    R["K"] = R["exp3"].get("k", 200)
    R["final"] = R["exp4"].get("final_model", R["exp3"].get("best_catboost"))
    R["ver"] = {k: version(m) for k, m in {
        "pandas": "pandas", "numpy": "numpy", "pyarrow": "pyarrow", "sklearn": "scikit-learn",
        "lightgbm": "lightgbm", "catboost": "catboost", "shap": "shap", "scipy": "scipy",
        "matplotlib": "matplotlib", "streamlit": "streamlit", "plotly": "plotly"}.items()}
    sel = json.load(open(ROOT / "artifacts" / "models" / "selected_features.json"))
    R["sel_groups"] = {
        "count": sum(c.startswith("cnt__") for c in sel),
        "pay": sum(c.startswith("pay__") for c in sel),
        "agg": sum(c.startswith("agg__") for c in sel),
        "last": sum(c.startswith("last__") for c in sel),
    }
    for f in ("master_scale_oot", "auc_by_block", "calibration_deciles"):
        p = RES / f"{f}.csv"
        if p.exists():
            R[f] = pd.read_csv(p)
    p = RES / f"shap_importance_{R['final']}.csv"
    if p.exists():
        R["shap_imp"] = pd.read_csv(p)
    return R
