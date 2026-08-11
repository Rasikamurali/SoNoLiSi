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

RUN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "exports", "gpt5_annotation_pilot", "full_corpus_run",
)
IN_PATH = os.path.join(RUN_DIR, "8_example_discussions_annotated.csv")

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
    labeled_dir = df[df.LLM_Directionality.notna()]

    rows = {}
    for fam in FAMILY_ORDER:
        rows[fam] = {
            "Counterproposal rate": trend_str(followers[followers.family == fam], "LLM_Speech_Act", "Counterproposal", arrow, pct_sign),
            "Maintenance rate": trend_str(labeled_dir[labeled_dir.family == fam], "LLM_Directionality", "Maintain", arrow, pct_sign),
            "Agreement rate": trend_str(followers[followers.family == fam], "LLM_Agreement", "Yes", arrow, pct_sign),
        }
    rows["Overall"] = {
        "Counterproposal rate": trend_str(followers, "LLM_Speech_Act", "Counterproposal", arrow, pct_sign),
        "Maintenance rate": trend_str(labeled_dir, "LLM_Directionality", "Maintain", arrow, pct_sign),
        "Agreement rate": trend_str(followers, "LLM_Agreement", "Yes", arrow, pct_sign),
    }
    return rows


def to_markdown(rows: dict) -> str:
    cols = ["Counterproposal rate", "Maintenance rate", "Agreement rate"]
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
    cols = ["Counterproposal rate", "Maintenance rate", "Agreement rate"]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lccc}",
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
        r"(turn $>0$ within a group's round); maintenance rate is the share of "
        r"turns with a directionality label. LLM-coded (gpt-5.4-mini, per-group "
        r"annotation, 8 cross-family hand-coded shot examples); see "
        r"Section~\ref{sec:annotation-validation} for accuracy/$\kappa$ against "
        r"hand-coded ground truth.}",
        r"\label{tab:family_trend}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    df = pd.read_csv(IN_PATH)
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
