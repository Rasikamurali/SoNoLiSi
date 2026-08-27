"""
community_ss_mechanism_table.py
---------------------------------
Compact robustness table for the main social-selection pathway
(community size only -- N=12/16/20): does each pathway link retain its
expected direction/strength as the community expands relative to the N=12
default?

Reuses social_selection_feedback_analysis.py (ssfa)'s existing, unmodified
main-paper specifications for the three pathway links -- no changes to
variable definitions, condition restrictions, controls, clustering, or the
linear-probability specification:
  1. Undercontribution_t -> Evaluation_t          (ssfa.step5_analysis)
  2. Evaluation_t -> TieWeightChange_t             (ssfa.step3_validation)
  3. TieWeight_t -> SeedAccess_{t+1}               (ssfa.step7_analysis)
ssfa.configure_tier("community", f"N{N}") sets MODEL_SPECS/N_AGENTS/
GROUP_SIZE for each N; ssfa.build_core_datasets() then loads exactly the
same edge_df/agent_df these three steps already expect.

Each N is fit as an independent model (there is no single pooled model
spanning N=12/16/20 for this pathway, unlike the contribution/gap tables'
Condition*N interaction models), so the N-vs-12 difference test is a
standard two-independent-estimates test: diff = beta_N - beta_12,
se = sqrt(se_N^2 + se_12^2) -- no shared covariance matrix exists across
three separately-fit models to exploit.

Output: figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/community/
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as scipy_stats

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import social_selection_feedback_analysis as ssfa  # noqa: E402  (the existing, unmodified pathway specs)
from model_specs import stars  # noqa: E402

# BASE is the root of this release, computed from this file's own location
# (three levels up from code/analysis/selection/) so paths below still work
# if the release is moved or copied elsewhere.
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
COMMUNITY_OUT = f"{BASE}/figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/community"

PATHWAYS = [
    ("undercontribution_evaluation", r"Under-contribution $\rightarrow$ evaluation"),
    ("evaluation_tieweight",         r"Evaluation $\rightarrow$ tie-weight change"),
    ("tieweight_seedaccess",         r"Tie weight $\rightarrow$ seed access"),
]
N_LEVELS = ["12", "16", "20"]


def get_pathway_coefficients(N, log):
    label = ssfa.configure_tier("community", f"N{N}")
    log(f"\n{'='*72}\nN={N} (community size {label})\n{'='*72}")
    runs, edge_df, agent_df = ssfa.build_core_datasets()
    log(f"  {len(runs)} runs, {len(edge_df):,} evaluation events, {len(agent_df):,} agent-round rows")

    _, step3_reg, step3_text = ssfa.step3_validation(edge_df)
    log(step3_text)

    _, step5_reg, step5_text = ssfa.step5_analysis(agent_df)
    log(step5_text)
    b5 = step5_reg[step5_reg["model"] == "pooled_ols_with_FE"].iloc[0]

    trans = ssfa.build_transitions(agent_df)
    step7_results, step7_text = ssfa.step7_analysis(trans)
    log(step7_text)

    return {
        "undercontribution_evaluation": {"coef": float(b5["estimate"]), "se": float(b5["se"]),
                                         "p": float(b5["p_value"]), "n": int(b5["n"])},
        "evaluation_tieweight": {"coef": float(step3_reg["estimate"]), "se": float(step3_reg["se"]),
                                 "p": float(step3_reg["p_value"]), "n": int(step3_reg.get("n", np.nan))
                                 if "n" in step3_reg else np.nan},
        "tieweight_seedaccess": {"coef": float(step7_results["seed_access_coef"]),
                                 "se": float(step7_results["seed_access_se"]),
                                 "p": float(step7_results["seed_access_p"]),
                                 "n": int(step7_results.get("seed_access_n", np.nan))},
    }


def difference_test(coef_a, se_a, coef_b, se_b):
    diff = coef_b - coef_a
    se = float(np.sqrt(se_a ** 2 + se_b ** 2))
    z = diff / se if se > 0 else np.nan
    p = 2 * (1 - scipy_stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return diff, se, z, p


def beta_se_stars(coef, se, p):
    s = stars(p)
    return rf"{coef:.3f} ({se:.3f})$^{{{s}}}$" if s else rf"{coef:.3f} ({se:.3f})"


def make_table(coefs_by_n, diff_rows, out_path):
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r" & $N=12$ & $N=16$ & $N=20$ \\", r"\midrule",
    ]
    for key, label in PATHWAYS:
        cells = [beta_se_stars(coefs_by_n[n][key]["coef"], coefs_by_n[n][key]["se"], coefs_by_n[n][key]["p"])
                 for n in N_LEVELS]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{4}{l}{\textit{Change relative to $N=12$}} \\")
    for key, label in PATHWAYS:
        cells = ["--"]
        for n in ["16", "20"]:
            row = diff_rows[(diff_rows.pathway == key) & (diff_rows.N == n)].iloc[0]
            cells.append(f"{row['diff']:+.3f} ({row['se']:.3f}){stars(row['p_value'])}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        (r"\caption{\textbf{Community-size robustness: social-selection pathway.} "
         r"Each row is one link in the contribution $\to$ evaluation $\to$ network weight $\to$ "
         r"selection/exclusion pathway, using the unmodified main-paper specification for that link "
         r"(no changes to variable definitions, controls, clustering, or the linear-probability spec). "
         r"$N=12/16/20$ are fit as independent models (no pooled Condition$\times N$ interaction exists "
         r"for this pathway); \textit{Change relative} rows test $\beta_N-\beta_{12}$ directly "
         r"(SE $=\sqrt{SE_N^2+SE_{12}^2}$, independent-estimates test -- these are three separately "
         r"fit models, so there is no shared covariance matrix to exploit as in the contribution/gap "
         r"tables). $^{\dagger}p{<}0.10$, $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$.}"),
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(text)
    return text


def main():
    os.makedirs(COMMUNITY_OUT, exist_ok=True)
    log_lines = []
    def log(msg):
        log_lines.append(str(msg))
        print(msg)

    coefs_by_n = {n: get_pathway_coefficients(n, log) for n in N_LEVELS}

    coef_rows = []
    for n in N_LEVELS:
        for key, label in PATHWAYS:
            c = coefs_by_n[n][key]
            coef_rows.append({"N": n, "pathway": key, "label": label,
                              "coef": c["coef"], "se": c["se"], "p_value": c["p"], "n": c["n"]})
    coef_df = pd.DataFrame(coef_rows)
    coef_df.to_csv(os.path.join(COMMUNITY_OUT, "ss_mechanism_coefficients.csv"), index=False)

    diff_rows = []
    for key, label in PATHWAYS:
        c12 = coefs_by_n["12"][key]
        for n in ["16", "20"]:
            cn = coefs_by_n[n][key]
            diff, se, z, p = difference_test(c12["coef"], c12["se"], cn["coef"], cn["se"])
            diff_rows.append({"pathway": key, "label": label, "N": n,
                              "diff": diff, "se": se, "z": z, "p_value": p,
                              "ci_lower": diff - 1.96 * se, "ci_upper": diff + 1.96 * se})
    diff_df = pd.DataFrame(diff_rows)
    diff_df.to_csv(os.path.join(COMMUNITY_OUT, "ss_mechanism_difference_tests.csv"), index=False)

    tex = make_table(coefs_by_n, diff_df, os.path.join(COMMUNITY_OUT, "ss_mechanism_table.tex"))

    with open(os.path.join(COMMUNITY_OUT, "ss_mechanism_qc_and_stats.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

    print(f"\n{'=' * 74}\nSOCIAL-SELECTION MECHANISM TABLE\n{'=' * 74}")
    print(tex)
    print(f"\nOutputs -> {COMMUNITY_OUT}")
    return coefs_by_n, diff_df


if __name__ == "__main__":
    main()
