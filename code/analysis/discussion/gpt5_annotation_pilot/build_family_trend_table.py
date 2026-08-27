"""
build_family_trend_table.py
------------------------------------------------------------
One combined table: counterproposal / maintenance / agreement rate,
each as an Early -> Middle -> Late trend string, by model family plus
an Overall row. Outputs a markdown version (for chat/write-up) and a
LaTeX version matching the booktabs style used in
exports/discussion_mechanism/discussion_mechanism_table.tex.
"""

import os
import re

import pandas as pd

# Reorg (2026-08-11): see build_pattern_tables.py for why this anchor exists.
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)

# 2026-08-17: repointed at the per-turn, row-level-shot, corrected-codebook run
# (the earlier full_corpus_run/8_example_discussions_annotated.csv was per-group
# mode with discussion-level shots and the pre-fix codebook -- see
# project_social_norm_evo_discourse_coding_pipeline memory for why per-turn/
# row-shots is the validated combination, 93.2% vs. 81.8% held-out accuracy).
RUN_DIR = os.path.join(_ANALYSIS_DIR, "exports", "gpt5_annotation_pilot", "full_corpus_annotated")
IN_PATH = os.path.join(RUN_DIR, "8_example_rows_annotated.csv")

ROUND_RE = re.compile(r"_R(\d+)_G\d+$")
PERIOD_ORDER = ["Early", "Middle", "Late"]
FAMILY_ORDER = ["GPT", "Llama-7B", "Mistral-7B", "Qwen-7B"]


def recover_round(discussion_id: str) -> int:
    return int(ROUND_RE.search(discussion_id).group(1))


def period_of(round_num: int) -> str:
    if round_num <= 7:
        return "Early"
    if round_num <= 14:
        return "Middle"
    return "Late"


def pct(series: pd.Series, value) -> float:
    return (series == value).mean() * 100


def trend_str(sub: pd.DataFrame, col: str, value: str, arrow: str, pct_sign: str, decimals: int = 1) -> str:
    vals = sub.groupby("period_n")[col].apply(lambda s: pct(s, value)).reindex(PERIOD_ORDER)
    return arrow.join(f"{v:.{decimals}f}" for v in vals) + pct_sign


def build_rows(df: pd.DataFrame, arrow: str = " → ", pct_sign: str = "%") -> dict:
    followers = df[df.turn != 0]
    labeled_dir = df[df.LLM_Directionality != ""]

    rows = {}
    for fam in FAMILY_ORDER:
        fam_followers = followers[followers.family == fam]
        fam_dir = labeled_dir[labeled_dir.family == fam]
        rows[fam] = {
            "Counterproposal rate": trend_str(fam_followers, "LLM_Speech_Act", "Counterproposal", arrow, pct_sign),
            "Agreement rate": trend_str(fam_followers, "LLM_Agreement", "Yes", arrow, pct_sign),
            "Maintenance rate": trend_str(fam_dir, "LLM_Directionality", "Maintain", arrow, pct_sign),
            "Increase rate": trend_str(fam_dir, "LLM_Directionality", "Increase", arrow, pct_sign),
            "Decrease rate": trend_str(fam_dir, "LLM_Directionality", "Decrease", arrow, pct_sign),
        }
    rows["Overall"] = {
        "Counterproposal rate": trend_str(followers, "LLM_Speech_Act", "Counterproposal", arrow, pct_sign),
        "Agreement rate": trend_str(followers, "LLM_Agreement", "Yes", arrow, pct_sign),
        "Maintenance rate": trend_str(labeled_dir, "LLM_Directionality", "Maintain", arrow, pct_sign),
        "Increase rate": trend_str(labeled_dir, "LLM_Directionality", "Increase", arrow, pct_sign),
        "Decrease rate": trend_str(labeled_dir, "LLM_Directionality", "Decrease", arrow, pct_sign),
    }
    return rows


def to_markdown(rows: dict) -> str:
    cols = ["Counterproposal rate", "Agreement rate", "Maintenance rate", "Increase rate", "Decrease rate"]
    lines = [
        "| Model family | " + " | ".join(cols) + " |",
        "| --- | " + " | ".join(["---:"] * len(cols)) + " |",
    ]
    for fam in FAMILY_ORDER:
        lines.append(f"| {fam} | " + " | ".join(rows[fam][c] for c in cols) + " |")
    bold = lambda s: f"**{s}**"
    lines.append(f"| **Overall** | " + " | ".join(bold(rows['Overall'][c]) for c in cols) + " |")
    return "\n".join(lines)


def to_latex(rows: dict) -> str:
    cols = ["Counterproposal rate", "Agreement rate", "Maintenance rate", "Increase rate", "Decrease rate"]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        " & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for fam in FAMILY_ORDER:
        vals = " & ".join(rows[fam][c] for c in cols)
        lines.append(f"  {fam} & {vals} \\\\")
    lines.append(r"\midrule")
    vals = " & ".join(r"\textbf{" + rows["Overall"][c] + "}" for c in cols)
    lines.append(r"  \textbf{Overall} & " + vals + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Discussion-mechanism trends by model family, Early (rounds 1-7) "
        r"$\rightarrow$ Middle (rounds 8-14) $\rightarrow$ Late (rounds 15-20). "
        r"Counterproposal rate and agreement rate are shares of follower turns "
        r"(turn $>0$ within a group's round); maintenance/increase/decrease rates "
        r"are shares of directionality-labeled turns. LLM-coded "
        r"(gpt-5.4-mini, per-turn annotation, 8 row-level hand-coded shot examples "
        r"from a second coder's blind recode); see "
        r"Section~\ref{sec:annotation-validation} for accuracy/$\kappa$/$\alpha$ "
        r"against hand-coded ground truth (93.2\% row-exact-match on held-out rows).}",
        r"\label{tab:family_trend}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    # keep_default_na=False: "N/A" is a real Human_Agreement/LLM_Agreement
    # category value, not a missing-value marker -- pandas' default would
    # silently collapse it into NaN otherwise.
    df = pd.read_csv(IN_PATH, keep_default_na=False)
    df["round_n"] = df.discussion_id.apply(recover_round)
    df["period_n"] = df.round_n.apply(period_of)

    md = to_markdown(build_rows(df, arrow=" → ", pct_sign="%"))
    tex = to_latex(build_rows(df, arrow=r" $\rightarrow$ ", pct_sign=r"\%"))

    md_path = os.path.join(RUN_DIR, "family_trend_table.md")
    tex_path = os.path.join(RUN_DIR, "family_trend_table.tex")
    with open(md_path, "w") as f:
        f.write(md + "\n")
    with open(tex_path, "w") as f:
        f.write(tex + "\n")

    print(md)
    print()
    print(tex)
    print(f"\nSaved to {md_path} and {tex_path}")


if __name__ == "__main__":
    main()
