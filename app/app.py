"""PD Scoring Studio - Streamlit front end for the CatBoost PD model.

Run from the project root:  streamlit run app/app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoring import APP_ASSETS, ROOT, Scorer, from_raw, load_results, to_raw  # noqa: E402

sys.path.insert(0, str(ROOT))
from pd_scoring.features import PAYM_COLS  # noqa: E402
from pd_scoring.risk import (DECISION, GRADE_BOUNDS, GRADE_NAMES, LGD_BASE, grade,  # noqa: E402
                             grade_lower_bounds, irb_capital, lifetime_pd, retail_correlation,
                             vasicek_conditional_pd)

BLUE, ORANGE, AQUA, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#898781", "#e1e0d9"
GRADE_COLORS = ["#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#6da7ec", "#eda100", "#eb6834", "#e34948", "#c23a39", "#8f2726"]
DECISION_STYLE = {"Approve": ("#0ca30c", "✔"), "Manual review": ("#fab219", "●"), "Decline": ("#d03b3b", "✖")}
PAY_LABELS = {0: "On time", 1: "Late", 2: "Seriously late", 3: "No data / closed"}
PAY_COLORS = {0: "#1baf7a", 1: "#eda100", 2: "#e34948", 3: "#e1e0d9"}
FLAG_FIELDS = [  # (raw column, user label) - raw value 0 means "had overdue"
    ("is_zero_loans5", "Overdue < 5 days"),
    ("is_zero_loans530", "Overdue 5-30 days"),
    ("is_zero_loans3060", "Overdue 30-60 days"),
    ("is_zero_loans6090", "Overdue 60-90 days"),
    ("is_zero_loans90", "Overdue 90+ days"),
]

st.set_page_config(page_title="PD Scoring Studio", page_icon="📉", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.kpi {border: 1px solid rgba(11,11,11,0.10); border-radius: 10px; padding: 14px 16px; background: var(--secondary-background-color);}
.kpi .label {font-size: 0.78rem; color: #898781; text-transform: uppercase; letter-spacing: .04em;}
.kpi .value {font-size: 1.9rem; font-weight: 650; line-height: 1.15;}
.kpi .sub {font-size: 0.82rem; color: #898781;}
.chip {display:inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 600; font-size: .85rem; color: white;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_scorer() -> Scorer:
    return Scorer()


@st.cache_data
def get_sample():
    loans = pd.read_parquet(APP_ASSETS / "sample_loans.parquet")
    target = pd.read_parquet(APP_ASSETS / "sample_target.parquet")
    return loans, target


@st.cache_data
def score_sample():
    loans, target = get_sample()
    return get_scorer().score(loans).merge(target, on="id", how="left")


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def fig_layout(fig, h=300, **kw):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=36, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(size=12),
                      legend=dict(orientation="h", y=-0.18), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ------------------------------------------------------------------ applicant editor helpers
def blank_loan(scorer: Scorer, rn: int, history: str = "0" * 12 + "3" * 13) -> dict:
    row = {"id": 0, "rn": rn, **scorer.defaults}
    for i, ch in enumerate(history[:25].ljust(25, "3")):
        row[PAYM_COLS[i]] = int(ch)
    for c, _ in FLAG_FIELDS:
        row[c] = 1
    return row


def template(name: str, scorer: Scorer) -> pd.DataFrame:
    examples = json.load(open(APP_ASSETS / "examples.json"))
    if name.startswith("Data example"):
        tag = name.split("(")[1].split(" ")[0]
        loans, _ = get_sample()
        df = from_raw(loans[loans.id == examples[tag]].copy())
        df["id"] = 0
        return df.sort_values("rn").reset_index(drop=True)
    if name == "Thin file: one recent loan":
        rows = [blank_loan(scorer, 1, "0" * 4 + "3" * 21)]
    elif name == "Recent delinquencies":
        rows = [blank_loan(scorer, 1, "1120000000000" + "3" * 12), blank_loan(scorer, 2, "1100000000" + "3" * 15),
                blank_loan(scorer, 3, "0" * 24 + "3")]
        rows[0]["is_zero_loans5"] = rows[0]["is_zero_loans530"] = 0
        rows[1]["is_zero_loans5"] = 0
    else:  # clean history
        rows = [blank_loan(scorer, i + 1, "0" * (24 - 4 * i) + "3" * (1 + 4 * i)) for i in range(4)]
    return pd.DataFrame(rows)


def to_editor(loans: pd.DataFrame) -> pd.DataFrame:
    ed = pd.DataFrame({
        "Product": loans["rn"].astype(int).to_numpy(),
        "Credit type": loans["enc_loans_credit_type"].astype(int).to_numpy(),
        "Credit status": loans["enc_loans_credit_status"].astype(int).to_numpy(),
        "Payment history (latest → oldest)": ["".join(str(int(v)) for v in r) for r in loans[PAYM_COLS].to_numpy()],
    })
    for c, lbl in FLAG_FIELDS:
        ed[lbl] = loans[c].to_numpy() == 0
    ed["Zero utilisation"] = loans["is_zero_util"].to_numpy() == 1
    return ed


def _int(v, default, hi):
    try:
        if v is None or pd.isna(v):
            return int(default)
        return int(np.clip(int(v), 0, hi))
    except (TypeError, ValueError):
        return int(default)


def from_editor(ed: pd.DataFrame, base: pd.DataFrame, scorer: Scorer) -> pd.DataFrame:
    rows = []
    for i, r in ed.reset_index(drop=True).iterrows():
        row = base.iloc[i].to_dict() if i < len(base) else blank_loan(scorer, i + 1)
        row["rn"] = i + 1
        row["enc_loans_credit_type"] = _int(r.get("Credit type"), row["enc_loans_credit_type"], 7)
        row["enc_loans_credit_status"] = _int(r.get("Credit status"), row["enc_loans_credit_status"], 6)
        h = r.get("Payment history (latest → oldest)")
        hist = "" if h is None or (isinstance(h, float) and np.isnan(h)) else str(h)
        hist = "".join(ch for ch in hist if ch in "0123") or "0" * 12
        for j, ch in enumerate(hist[:25].ljust(25, "3")):
            row[PAYM_COLS[j]] = int(ch)
        for c, lbl in FLAG_FIELDS:
            row[c] = 0 if bool(r.get(lbl)) else 1
        row["is_zero_util"] = 1 if bool(r.get("Zero utilisation")) else 0
        row["id"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def history_heatmap(loans: pd.DataFrame):
    z = loans[PAYM_COLS].to_numpy()
    colorscale = []
    for k in range(4):
        colorscale += [[k / 4, PAY_COLORS[k]], [(k + 1) / 4, PAY_COLORS[k]]]
    text = [[PAY_LABELS[int(v)] for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, zmin=-0.5, zmax=3.5, colorscale=colorscale, showscale=False,
                               x=[f"-{i}" for i in range(25)], y=[f"Product {int(r)}" for r in loans.rn],
                               text=text, hovertemplate="%{y}, month %{x}: %{text}<extra></extra>",
                               xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Months ago (0 = latest)")
    return fig_layout(fig, h=70 + 26 * len(loans), title="Payment history map")


def master_scale_strip(p: float):
    lows = grade_lower_bounds()
    fig = go.Figure()
    for i, (g, lo, hi) in enumerate(zip(GRADE_NAMES, lows, GRADE_BOUNDS)):
        fig.add_trace(go.Bar(x=[1], y=[g], orientation="h", marker_color=GRADE_COLORS[i],
                             hovertemplate=f"{g}: PD {lo:.1%} - {hi:.1%}<br>{DECISION[g]}<extra></extra>",
                             showlegend=False, width=0.86))
    gi = GRADE_NAMES.index(grade(p)[0])
    fig.add_annotation(x=1.02, y=GRADE_NAMES[gi], text="◀ applicant", showarrow=False, xanchor="left",
                       font=dict(size=12, color=INK2))
    fig.update_xaxes(visible=False, range=[0, 1.6])
    fig.update_yaxes(autorange="reversed")
    return fig_layout(fig, h=320, title="Master scale", bargap=0.08)


# ------------------------------------------------------------------ pages
def page_applicant(scorer: Scorer):
    st.subheader("Applicant assessment")
    st.caption("Describe the applicant's credit products as they appear in the bureau report, then read the PD, "
               "rating grade and the factors behind it. Payment codes: 0 on time, 1 late, 2 seriously late, "
               "3 no data / closed.")
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Start from", ["Clean history: four products", "Thin file: one recent loan",
                                         "Recent delinquencies", "Data example (low risk)",
                                         "Data example (medium risk)", "Data example (high risk)"])
    if st.session_state.get("template") != choice:
        st.session_state.template = choice
        st.session_state.loans = template(choice, scorer)
        st.session_state.editor_ver = st.session_state.get("editor_ver", 0) + 1
    loans = st.session_state.loans

    ed = st.data_editor(
        to_editor(loans), num_rows="dynamic", width="stretch", hide_index=True,
        key=f"editor_{st.session_state.editor_ver}",
        column_config={
            "Product": st.column_config.NumberColumn(disabled=True, width="small"),
            "Credit type": st.column_config.NumberColumn(min_value=0, max_value=7, step=1, help="Encoded bureau credit type, 0-7"),
            "Credit status": st.column_config.NumberColumn(min_value=0, max_value=6, step=1, help="Encoded bureau credit status, 0-6"),
            "Payment history (latest → oldest)": st.column_config.TextColumn(
                width="large", max_chars=25, help="Up to 25 digits, first digit = latest month. Missing months are filled with 3."),
        })
    loans = from_editor(ed, loans, scorer)
    if len(loans) == 0:
        st.info("Add at least one credit product to score the applicant.")
        return

    with st.expander("Advanced bureau fields (binned values, 0-19)"):
        st.caption("These are the anonymised bins of the original bureau fields. Defaults are the most common values.")
        adv_cols = [c for c in scorer.spec.raw_columns if c not in PAYM_COLS
                    and c not in [f for f, _ in FLAG_FIELDS] + ["is_zero_util", "enc_loans_credit_type", "enc_loans_credit_status"]]
        adv = st.data_editor(loans[["rn"] + adv_cols], hide_index=True, width="stretch",
                             disabled=["rn"], key=f"adv_{st.session_state.editor_ver}_{len(loans)}")
        loans[adv_cols] = adv[adv_cols].to_numpy()

    raw = to_raw(loans)
    res = scorer.score(raw).iloc[0]
    p, g = float(res.pd), res.grade
    colr, icon = DECISION_STYLE[res.decision]

    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "Probability of default", f"{p:.2%}", "12-month horizon, 90+ days past due")
    kpi(k2, "Rating grade", g, f"{grade_lower_bounds()[GRADE_NAMES.index(g)]:.1%} - {GRADE_BOUNDS[GRADE_NAMES.index(g)]:.1%} PD band")
    kpi(k3, "Decision", f'<span class="chip" style="background:{colr}">{icon} {res.decision}</span>',
        "Approve R1-R5, review R6-R7, decline R8-R10")
    kpi(k4, "Lifetime PD (3 years)", f"{float(lifetime_pd(p, 3)):.2%}", "flat marginal PD assumption")

    left, right = st.columns([3, 2])
    with left:
        base, ex = scorer.explain(raw)
        top = ex.head(10).iloc[::-1]
        fig = go.Figure(go.Bar(x=top.shap, y=top.description, orientation="h",
                               marker_color=[ORANGE if v > 0 else BLUE for v in top.shap],
                               customdata=top.value, hovertemplate="%{y}<br>value: %{customdata}<br>"
                               "contribution: %{x:+.3f} log-odds<extra></extra>"))
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        st.plotly_chart(fig_layout(fig, h=380, title="What drives this PD (orange raises risk, blue lowers it)"),
                        width="stretch")
        th = ex.groupby("theme").shap.sum().sort_values()
        fig = go.Figure(go.Bar(x=th.values, y=th.index, orientation="h",
                               marker_color=[ORANGE if v > 0 else BLUE for v in th.values],
                               hovertemplate="%{y}: %{x:+.3f} log-odds<extra></extra>"))
        st.plotly_chart(fig_layout(fig, h=250, title="Contribution by risk theme"), width="stretch")
    with right:
        st.plotly_chart(master_scale_strip(p), width="stretch")
        st.markdown("**Loss estimate for the requested loan**")
        a1, a2 = st.columns(2)
        ead = a1.number_input("Loan amount", min_value=1000, value=50_000, step=5000)
        lgd = a2.slider("LGD", 0.1, 1.0, LGD_BASE, 0.05)
        st.write(f"Expected loss = PD × LGD × EAD = **{p * lgd * ead:,.0f}**  ·  "
                 f"IRB capital ≈ **{float(irb_capital(p, lgd)) * ead:,.0f}**")
        qs = {"Baseline": None, "Adverse (1-in-20)": 0.95, "Severe (1-in-100)": 0.99, "Basel (1-in-1000)": 0.999}
        sp = [p if q is None else float(vasicek_conditional_pd(p, q)) for q in qs.values()]
        fig = go.Figure(go.Bar(x=list(qs), y=sp, marker_color=[BLUE, AQUA, ORANGE, "#e34948"],
                               text=[f"{v:.1%}" for v in sp], textposition="outside",
                               hovertemplate="%{x}: PD %{y:.2%}<extra></extra>"))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig_layout(fig, h=260, title="Applicant PD under macro stress"), width="stretch")

    st.plotly_chart(history_heatmap(loans), width="stretch")


def page_batch(scorer: Scorer):
    st.subheader("Batch scoring")
    st.caption("Upload loan-level bureau records (CSV or Parquet) with columns id, rn and the original fields. "
               "Missing fields are filled with their most common value. An optional 'flag' column enables validation.")
    up = st.file_uploader("Loan records", type=["csv", "parquet"])
    if up is None:
        st.info("No file uploaded - showing the bundled sample of 3,000 out-of-time applicants.")
        res = score_sample()
    else:
        df = pd.read_parquet(up) if up.name.endswith(".parquet") else pd.read_csv(up)
        missing = {"id", "rn"} - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(sorted(missing))}")
            return
        with st.spinner("Scoring..."):
            res = scorer.score(df)
        if "flag" in df.columns:
            res = res.merge(df.groupby("id").flag.first().reset_index(), on="id", how="left")

    k = st.columns(4)
    kpi(k[0], "Applicants", f"{len(res):,}")
    kpi(k[1], "Mean PD", f"{res.pd.mean():.2%}", f"median {res.pd.median():.2%}")
    kpi(k[2], "Approval rate", f"{(res.decision == 'Approve').mean():.1%}", "grades R1-R5")
    if "flag" in res and res.flag.notna().all() and res.flag.nunique() == 2:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(res.flag, res.pd)
        kpi(k[3], "ROC AUC on file", f"{auc:.3f}", f"Gini {2 * auc - 1:.3f}, default rate {res.flag.mean():.2%}")
    else:
        kpi(k[3], "Expected loss rate", f"{(res.pd * LGD_BASE).mean():.2%}", f"LGD {LGD_BASE:.0%}")

    c1, c2 = st.columns(2)
    mix = res.grade.value_counts().reindex(GRADE_NAMES).fillna(0)
    fig = go.Figure(go.Bar(x=GRADE_NAMES, y=mix.values, marker_color=GRADE_COLORS,
                           hovertemplate="%{x}: %{y} applicants<extra></extra>"))
    c1.plotly_chart(fig_layout(fig, title="Applicants by rating grade"), width="stretch")
    if "flag" in res and res.flag.notna().any():
        t = res.groupby("grade").agg(pd_mean=("pd", "mean"), odr=("flag", "mean")).reindex(GRADE_NAMES)
        fig = go.Figure([go.Bar(x=GRADE_NAMES, y=t.pd_mean, name="Mean PD", marker_color=BLUE),
                         go.Bar(x=GRADE_NAMES, y=t.odr, name="Observed default rate", marker_color=ORANGE)])
        fig.update_yaxes(tickformat=".0%")
        c2.plotly_chart(fig_layout(fig, title="Predicted vs observed by grade"), width="stretch")
    else:
        fig = go.Figure(go.Histogram(x=res.pd, nbinsx=60, marker_color=BLUE))
        c2.plotly_chart(fig_layout(fig, title="PD distribution"), width="stretch")

    st.dataframe(res.sort_values("pd", ascending=False).style.format({"pd": "{:.2%}"}),
                 width="stretch", height=280, hide_index=True)
    buf = io.StringIO()
    res.to_csv(buf, index=False)
    st.download_button("Download scores (CSV)", buf.getvalue(), "pd_scores.csv", "text/csv")


def page_stress(scorer: Scorer):
    st.subheader("Stress lab")
    st.caption("Stress the bundled out-of-time portfolio. Top-down: the Vasicek one-factor model moves every PD to "
               "the chosen quantile of the systematic factor. Bottom-up: payment records are shocked and re-scored.")
    res = score_sample()
    p0 = res.pd.to_numpy()
    c1, c2, c3, c4 = st.columns(4)
    q = c1.slider("Severity (factor quantile)", 0.50, 0.999, 0.95, 0.001, help="0.95 = 1-in-20 year downturn")
    lgd = c2.slider("LGD", 0.2, 1.0, 0.6, 0.05)
    rho_mode = c3.selectbox("Asset correlation", ["Basel retail formula", "Custom"])
    rho = None if rho_mode.startswith("Basel") else c3.number_input("rho", 0.001, 0.5, 0.05, 0.005, format="%.3f")
    shock = c4.slider("Bottom-up: share of on-time loans turned late", 0.0, 0.5, 0.0, 0.05)

    if shock > 0:
        loans, _ = get_sample()
        lo = loans.copy()
        rng = np.random.default_rng(1)
        hit = (lo.enc_paym_0.to_numpy() == 0) & (rng.random(len(lo)) < shock)
        lo.loc[hit, "enc_paym_0"] = 1
        lo.loc[hit, "is_zero_loans5"] = 0
        p_b = scorer.score(lo).pd.to_numpy()
    else:
        p_b = p0
    p_s = vasicek_conditional_pd(p_b, q, rho)

    k = st.columns(4)
    kpi(k[0], "Mean PD", f"{p_s.mean():.2%}", f"baseline {p0.mean():.2%}")
    kpi(k[1], "Expected loss rate", f"{(p_s * lgd).mean():.2%}", f"baseline {(p0 * lgd).mean():.2%}")
    kpi(k[2], "Approval rate", f"{(p_s < 0.03).mean():.1%}", f"baseline {(p0 < 0.03).mean():.1%}")
    kpi(k[3], "Decline grades R8-R10", f"{(p_s >= 0.07).mean():.1%}", f"baseline {(p0 >= 0.07).mean():.1%}")

    a, b = st.columns(2)
    m0 = pd.Series(grade(p0)).value_counts(normalize=True).reindex(GRADE_NAMES).fillna(0)
    m1 = pd.Series(grade(p_s)).value_counts(normalize=True).reindex(GRADE_NAMES).fillna(0)
    fig = go.Figure([go.Bar(x=GRADE_NAMES, y=m0, name="Baseline", marker_color=BLUE),
                     go.Bar(x=GRADE_NAMES, y=m1, name="Stressed", marker_color=ORANGE)])
    fig.update_yaxes(tickformat=".0%")
    a.plotly_chart(fig_layout(fig, title="Grade migration"), width="stretch")

    limit = b.number_input("Reverse stress: loss limit (% of exposure)", 1.0, 20.0, 5.0, 0.5) / 100
    qs = np.linspace(0.5, 0.9995, 150)
    els = np.array([(vasicek_conditional_pd(p_b, x, rho) * lgd).mean() for x in qs])
    fig = go.Figure(go.Scatter(x=qs, y=els, mode="lines", line=dict(color=BLUE, width=2),
                               hovertemplate="q=%{x:.3f}: loss %{y:.2%}<extra></extra>", name="Expected loss"))
    fig.add_hline(y=limit, line_dash="dash", line_color="#d03b3b")
    if els.max() >= limit:
        q_star = float(np.interp(limit, els, qs))
        fig.add_vline(x=q_star, line_dash="dot", line_color=MUTED)
        b.markdown(f"Loss limit is breached at factor quantile **{q_star:.3f}** "
                   f"(roughly a **1-in-{1 / (1 - q_star):.0f}** year event).")
    else:
        b.markdown("The loss limit is not reached even at the 99.95% quantile.")
    fig.update_yaxes(tickformat=".1%")
    b.plotly_chart(fig_layout(fig, h=260, title="Loss rate vs severity"), width="stretch")
    st.caption(f"Mean Basel retail correlation for this portfolio: {retail_correlation(p0).mean():.3f}.")


def page_model(scorer: Scorer):
    st.subheader("Model card")
    r = load_results()
    v = r.get("exp4_validation", {})
    st.markdown(f"**Model:** CatBoost gradient boosting (`{scorer.name}`), {len(scorer.cols)} features built "
                "from bureau credit history. **Target:** 90+ days past due. **Development:** applications 0-2.0M; "
                "**validation:** 2.0-2.5M; **out-of-time test:** 2.5-3.0M.")
    if "discrimination_oot" in v:
        names = {"lgb_baseline": "LightGBM baseline", "logreg": "Logistic regression",
                 "lgb_tuned": "LightGBM (new features)", scorer.name: "CatBoost (final)"}
        rows = [{"Model": names.get(k, k), "AUC": m["auc"], "Gini": m["gini"], "KS": m["ks"], "Brier": m["brier"],
                 "AUC 95% CI": f"{m['auc_ci'][0]:.4f} - {m['auc_ci'][1]:.4f}"}
                for k, m in v["discrimination_oot"].items()]
        st.dataframe(pd.DataFrame(rows).style.format({"AUC": "{:.4f}", "Gini": "{:.4f}", "KS": "{:.4f}", "Brier": "{:.5f}"}),
                     hide_index=True, width="stretch")
    figs = ROOT / "artifacts" / "figures"
    cols = st.columns(2)
    for i, (f, cap) in enumerate([("roc_oot.png", "ROC curves, OOT"), ("calibration_deciles.png", "Calibration"),
                                  ("shap_bar.png", "Global importance (SHAP)"), ("auc_by_block.png", "Stability over time")]):
        if (figs / f).exists():
            cols[i % 2].image(str(figs / f), caption=cap, width="stretch")
    st.markdown("**Limitations.** The bureau fields are anonymised bins, so the inputs on the assessment page are "
                "codes rather than amounts. LGD and EAD are assumptions, not estimates. Stress scenarios use the "
                "regulatory one-factor model rather than a fitted macro-economic link, because the data has no dates.")


def main():
    scorer = get_scorer()
    with st.sidebar:
        st.markdown("## 📉 PD Scoring Studio")
        page = st.radio("Section", ["Applicant assessment", "Batch scoring", "Stress lab", "Model card"],
                        label_visibility="collapsed")
        st.divider()
        oot = scorer.meta.get("oot")
        if oot:
            st.metric("Out-of-time ROC AUC", f"{oot['auc']:.4f}")
            st.metric("Gini", f"{oot['gini']:.4f}")
        st.caption("Course project: probability of default model on credit bureau history.")
    {"Applicant assessment": page_applicant, "Batch scoring": page_batch,
     "Stress lab": page_stress, "Model card": page_model}[page](scorer)


main()
