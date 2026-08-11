"""
lagged_ar_regression.py
------------------------
Extends the lagged model in sobel_mediation.py with an autoregressive term
and condition/model fixed effects:

    contribution_{t+1} = b*IN_t + c'*DN_t + gamma*contribution_t + controls

where controls = condition fixed effects (BASELINE/NO_SELECTION/NO_DISCUSSION/
FULL) + model-family fixed effects (GPT, Llama-7B, ...). IN_t and DN_t are
elicited after round t's contribution (see sobel_mediation.py docstring), so
every predictor on the right-hand side is measured no later than t, and the
outcome is t+1 - the same elicitation-order argument as the lagged mediation
model there.

All variables are z-scored (within the full lagged sample) before fitting.
OLS with SEs clustered by run_id (model x seed x condition).

Output: exports/lagged_ar_regression_summary.txt
"""

import os
import warnings

import statsmodels.formula.api as smf

from sobel_mediation import load_all, add_lags, zscale, stars

warnings.filterwarnings("ignore")

OUT_PATH = "exports/lagged_ar_regression_summary.txt"

FORMULA = (
    "contribution_lead1_z ~ IN_z + DN_z + contribution_z "
    '+ C(condition, Treatment(reference="BASELINE")) '
    '+ C(family, Treatment(reference="GPT"))'
)


def run():
    print("Loading data …")
    raw = load_all()
    print(f"  Raw: {len(raw):,} obs | {raw['run_id'].nunique()} run_ids")

    df = add_lags(raw)
    df = df.dropna(subset=["contribution_lead1"]).copy()
    df = zscale(df, ["IN", "DN", "contribution", "contribution_lead1"])

    print("Fitting contribution_{t+1} ~ IN_t + DN_t + contribution_t + condition FE + family FE …")
    model = smf.ols(FORMULA, data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["run_id"]}
    )
    return model, df


def summarize(model, df):
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append("LAGGED AUTOREGRESSIVE MODEL")
    lines.append("contribution_{t+1} = b*IN_t + c'*DN_t + gamma*contribution_t + controls")
    lines.append("controls = condition FE (ref=BASELINE) + model-family FE (ref=GPT)")
    lines.append(sep)
    lines.append(f"\nN = {int(model.nobs):,}   clusters (run_id) = {df['run_id'].nunique()}")
    lines.append("SEs clustered by run_id.\n")

    lines.append("-" * 70)
    lines.append("Focal coefficients")
    lines.append("-" * 70)
    for name, label in [("IN_z", "b   (IN_t   -> contribution_t+1)"),
                        ("DN_z", "c'  (DN_t   -> contribution_t+1)"),
                        ("contribution_z", "gamma (contribution_t -> contribution_t+1)")]:
        beta = model.params[name]
        se   = model.bse[name]
        p    = model.pvalues[name]
        lines.append(f"  {label:45s} beta={beta:7.4f}  SE={se:6.4f}  p={p:7.4f}  {stars(p)}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("Condition fixed effects (ref = BASELINE)")
    lines.append("-" * 70)
    for name in model.params.index:
        if "condition" in name:
            beta, se, p = model.params[name], model.bse[name], model.pvalues[name]
            lines.append(f"  {name:45s} beta={beta:7.4f}  SE={se:6.4f}  p={p:7.4f}  {stars(p)}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("Model-family fixed effects (ref = GPT)")
    lines.append("-" * 70)
    for name in model.params.index:
        if "family" in name:
            beta, se, p = model.params[name], model.bse[name], model.pvalues[name]
            lines.append(f"  {name:45s} beta={beta:7.4f}  SE={se:6.4f}  p={p:7.4f}  {stars(p)}")

    lines.append("")
    lines.append(f"R-squared = {model.rsquared:.4f}   Adj. R-squared = {model.rsquared_adj:.4f}")
    lines.append(sep)
    return "\n".join(lines) + "\n"


def main():
    model, df = run()
    text = summarize(model, df)
    print(text)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        f.write(text)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
