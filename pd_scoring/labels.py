"""Readable descriptions of feature names."""
import re

RAW = {
    "pre_since_opened": "days since opening (bin)",
    "pre_since_confirmed": "days since info confirmed (bin)",
    "pre_pterm": "planned term (bin)",
    "pre_fterm": "actual term (bin)",
    "pre_till_pclose": "days to planned close (bin)",
    "pre_till_fclose": "days to actual close (bin)",
    "pre_loans_credit_limit": "credit limit (bin)",
    "pre_loans_next_pay_summ": "next payment amount (bin)",
    "pre_loans_outstanding": "outstanding amount (bin)",
    "pre_loans_total_overdue": "current overdue amount (bin)",
    "pre_loans_max_overdue_sum": "max overdue amount (bin)",
    "pre_loans_credit_cost_rate": "full cost of credit (bin)",
    "pre_loans5": "overdues <5 days (bin)",
    "pre_loans530": "overdues 5-30 days (bin)",
    "pre_loans3060": "overdues 30-60 days (bin)",
    "pre_loans6090": "overdues 60-90 days (bin)",
    "pre_loans90": "overdues 90+ days (bin)",
    "is_zero_loans5": "no overdues <5 days",
    "is_zero_loans530": "no overdues 5-30 days",
    "is_zero_loans3060": "no overdues 30-60 days",
    "is_zero_loans6090": "no overdues 60-90 days",
    "is_zero_loans90": "no overdues 90+ days",
    "pre_util": "utilisation (bin)",
    "pre_over2limit": "overdue / limit (bin)",
    "pre_maxover2limit": "max overdue / limit (bin)",
    "is_zero_util": "zero utilisation",
    "is_zero_over2limit": "zero overdue / limit",
    "is_zero_maxover2limit": "zero max overdue / limit",
    "enc_loans_account_holder_type": "account holder type",
    "enc_loans_credit_status": "credit status",
    "enc_loans_account_cur": "currency",
    "enc_loans_credit_type": "credit type",
    "pclose_flag": "planned close undefined",
    "fclose_flag": "actual close undefined",
}
PAY_STATUS = {0: "on time", 1: "late (status 1)", 2: "late (status 2)", 3: "no data / closed (status 3)"}


def _raw(col: str) -> str:
    m = re.fullmatch(r"enc_paym_(\d+)", col)
    if m:
        return f"payment status, month -{m.group(1)}"
    return RAW.get(col, col)


def describe(name: str) -> str:
    if name == "agg__n_loans":
        return "Number of credit products"
    if name == "pay__months_since_late":
        return "Months since latest late payment (99 = never)"
    m = re.fullmatch(r"cnt__(.+)_(\d+)", name)
    if m:
        return f"# products with {_raw(m.group(1))} = {m.group(2)}"
    m = re.fullmatch(r"agg__(.+)_(mean|max|share)", name)
    if m:
        what = {"mean": "Mean", "max": "Max", "share": "Share of products:"}[m.group(2)]
        return f"{what} {_raw(m.group(1))}"
    m = re.fullmatch(r"pay__s(\d)_last(\d+)(_share)?", name)
    if m:
        s, w, share = int(m.group(1)), m.group(2), m.group(3)
        base = f"{PAY_STATUS[s]} payments, last {w} months"
        return ("Share of " if share else "# ") + base
    m = re.fullmatch(r"last__(.+)", name)
    if m:
        return f"Latest product: {_raw(m.group(1))}"
    return name
