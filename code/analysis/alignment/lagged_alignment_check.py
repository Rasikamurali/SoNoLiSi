"""
lagged_alignment_check.py
--------------------------
Item (d) of the perception/alignment consolidation (2026-08-24): lagged
respecification of gap_based_alignment.py's primary result -- a supporting
robustness check, not the headline number. Supersedes lagged_ar_regression.py
(archived to code/analysis/archive/alignment/), which this file ports
wholesale for the statistical logic; only tier dispatch, output-path
routing, and the new community/group-size tiers are new.

    contribution_{t+1} = b*IN_t + c'*DN_t + gamma*contribution_t + controls

where controls = condition FE + model-family FE. IN_t and DN_t are elicited
after round t's contribution, so every predictor on the right-hand side is
measured no later than t, and the outcome is t+1 -- the same
elicitation-order argument as gap_based_alignment.py's primary model, just
respecified in raw levels (no gap/differencing) with an explicit
autoregressive term. All variables are z-scored (within each tier's sample)
before fitting. OLS, SEs clustered by run_id.

Tiers (--tier flag, default 7b): 7b / s2 / pooled / community / group --
same family lists and community/group data sources as gap_based_alignment.py
(COMMUNITY_SOURCES / GROUP_N12_SOURCES / GROUP_N16_SOURCES are imported from
there rather than redefined, since these two files are siblings in the same
4-file consolidation, not independent scripts).

Output: figures/SUPPORTING_MAIN_RESULTS/4_alignment_cross_checks/
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_specs import load_all, add_lags, zscale, stars, MODEL_SPECS  # noqa: E402
from gap_based_alignment import (  # noqa: E402
    TIER_SPECS, STRUCTURAL_SETTINGS, build_base_dataset_from_sources,
)

warnings.filterwarnings("ignore")

BASE = "/data3/rasimura/social-norm-evo"
PUBLISH_ROOT = f"{BASE}/figures/SUPPORTING_MAIN_RESULTS/4_alignment_cross_checks"
EXPORT_ROOT = f"{BASE}/code/analysis/exports"


def formula_for(family_ref):
    return (
        "contribution_lead1_z ~ IN_z + DN_z + contribution_z "
        '+ C(condition, Treatment(reference="BASELINE")) '
        f'+ C(family, Treatment(reference="{family_ref}"))'
    )


def fit(df, family_ref):
    df = add_lags(df)
    df = df.dropna(subset=["contribution_lead1", "IN", "DN", "contribution"]).copy()
    df = zscale(df, ["IN", "DN", "contribution", "contribution_lead1"])
    model = smf.ols(formula_for(family_ref), data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["run_id"]}
    )
    return model, df


def summarize(model, df, label):
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append(f"LAGGED AUTOREGRESSIVE MODEL -- {label}")
    lines.append("contribution_{t+1} = b*IN_t + c'*DN_t + gamma*contribution_t + controls")
    lines.append("controls = condition FE (ref=BASELINE) + model-family FE")
    lines.append(sep)
    lines.append(f"\nN = {int(model.nobs):,}   clusters (run_id) = {df['run_id'].nunique()}")
    lines.append("SEs clustered by run_id.\n")

    lines.append("-" * 70)
    lines.append("Focal coefficients")
    lines.append("-" * 70)
    for name, term_label in [("IN_z", "b   (IN_t   -> contribution_t+1)"),
                             ("DN_z", "c'  (DN_t   -> contribution_t+1)"),
                             ("contribution_z", "gamma (contribution_t -> contribution_t+1)")]:
        beta, se, p = model.params[name], model.bse[name], model.pvalues[name]
        lines.append(f"  {term_label:45s} beta={beta:7.4f}  SE={se:6.4f}  p={p:7.4f}  {stars(p)}")

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
    lines.append("Model-family fixed effects")
    lines.append("-" * 70)
    for name in model.params.index:
        if "family" in name:
            beta, se, p = model.params[name], model.bse[name], model.pvalues[name]
            lines.append(f"  {name:45s} beta={beta:7.4f}  SE={se:6.4f}  p={p:7.4f}  {stars(p)}")

    lines.append("")
    lines.append(f"R-squared = {model.rsquared:.4f}   Adj. R-squared = {model.rsquared_adj:.4f}")
    lines.append(sep)
    return "\n".join(lines) + "\n"


def _write(text, label, out_subdir):
    export_dir = os.path.join(EXPORT_ROOT, "lagged_alignment_check", out_subdir)
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "lagged_ar_regression_summary.txt")
    with open(export_path, "w") as f:
        f.write(text)

    publish_dir = os.path.join(PUBLISH_ROOT, out_subdir) if out_subdir else PUBLISH_ROOT
    os.makedirs(publish_dir, exist_ok=True)
    publish_path = os.path.join(publish_dir, "lagged_ar_regression_summary.txt")
    with open(publish_path, "w") as f:
        f.write(text)
    print(f"[{label}] Saved -> {export_path}\n[{label}] Published -> {publish_path}")


def run_family_tier(tier):
    spec = TIER_SPECS[tier]
    families = spec["families"]

    print(f"[{tier}] Loading data …")
    raw = load_all()
    if families is not None:
        raw = raw[raw["family"].isin(families)].copy()
    print(f"[{tier}] {len(raw):,} obs | {raw['run_id'].nunique()} run_ids")

    families_present = sorted(raw["family"].unique())
    family_ref = "GPT" if "GPT" in families_present else families_present[0]

    model, df = fit(raw, family_ref)
    text = summarize(model, df, tier)
    print(text)

    out_subdir = "" if tier == "7b" else (
        f"subset_{'_'.join(f.lower().replace('-', '') for f in families)}" if families is not None
        else "subset_all_families"
    )
    _write(text, tier, out_subdir)


def run_structural_tier(axis):
    for stratum, val, sources in STRUCTURAL_SETTINGS[axis]:
        label = f"{axis}_{stratum}_{val}"
        raw = build_base_dataset_from_sources(sources, axis_label=f"{stratum}{val}")
        if raw.empty:
            print(f"[{label}] [WARN] no data assembled, skipping.")
            continue
        families_present = sorted(raw["family"].unique())
        family_ref = "Mistral-7B" if "Mistral-7B" in families_present else families_present[0]

        model, df = fit(raw, family_ref)
        text = summarize(model, df, label)
        print(text)
        _write(text, label, os.path.join(axis, label))


def main():
    parser = argparse.ArgumentParser(description="Lagged autoregressive respecification (item d).")
    parser.add_argument("--tier", choices=["7b", "s2", "13b", "70b", "pooled", "community", "group", "all"],
                         default="7b")
    args = parser.parse_args()

    tiers = ["7b", "s2", "13b", "70b", "pooled", "community", "group"] if args.tier == "all" else [args.tier]
    for tier in tiers:
        if tier in TIER_SPECS:
            run_family_tier(tier)
        else:
            run_structural_tier(tier)


if __name__ == "__main__":
    main()
