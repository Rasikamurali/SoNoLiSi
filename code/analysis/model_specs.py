"""
model_specs.py
---------------
Canonical registry + shared data-loading utilities for the local-variant
analysis pipeline: which results directory, data variant, and seed list
each of the 10 model families (GPT + 3x7B + 3x13B/14B + 2x70B/72B +
GPT-5-mini) uses, plus the IN/DN perception-and-contribution panel loader
built on top of it. GPT-5-mini was added 2026-08-18 for the S2 supplementary
replication (GPT-5-mini + Llama-70B + Mistral-13B + Qwen-72B) — see
PAPER_RESULTS_INVENTORY.md.

Extracted from sobel_mediation.py on 2026-08-17 when that script (whose own
Sobel-mediation-test analysis has been superseded — see
code/analysis/perception/mediation_llama_mistral_qwen.py) was archived to
archive/sobel_mediation.py. MODEL_SPECS/CONDITIONS/BASE and the load_all/
add_lags/zscale/stars helpers had all become shared dependencies for several
unrelated scripts (social_selection_feedback_analysis.py,
selection_mechanism_llama_mistral_qwen.py, discussion_talk_vs_behavior.py,
build_agent_round_panel.py, exclusion_diagnostics.py,
gap_based_alignment_test.py, discussion_mechanism_analysis.py,
lagged_ar_regression.py), and this file's presence at code/analysis/ is also
used by many scripts throughout code/analysis/ as a directory anchor (walk
up parent directories until this file is found) to locate ANALYSIS_DIR for
sys.path setup — so this module, not sobel_mediation.py, must stay at
code/analysis/ going forward.
"""

import os
import json
import glob
import pandas as pd

BASE    = "/data3/rasimura/social-norm-evo"
OUT_DIR = f"{BASE}/figures/2026-03-22/paper_stats"

MODEL_SPECS = [
    ("gpt",         "GPT",         f"{BASE}/results",      "local", list(range(43, 53))),
    ("llama",       "Llama-7B",    f"{BASE}/results",      "local", list(range(43, 53))),
    ("mistral",     "Mistral-7B",  f"{BASE}/results",      "local", list(range(43, 53))),
    ("qwen",        "Qwen-7B",     f"{BASE}/results",      "local", list(range(43, 53))),
    ("llama_13b",   "Llama-13B",   f"{BASE}/results",      "local", list(range(42, 52))),
    ("mistral_13b", "Mistral-13B", f"{BASE}/results",      "local", list(range(42, 52))),
    ("qwen_14b",    "Qwen-14B",    f"{BASE}/results",      "local", list(range(42, 52))),
    ("llama_70b",   "Llama-70B",   f"{BASE}/code/results", "local", list(range(42, 52))),
    ("qwen_72b",    "Qwen-72B",    f"{BASE}/code/results", "local", list(range(42, 52))),
    ("gpt-5-mini",  "GPT-5-mini",  f"{BASE}/code/results", "local", list(range(42, 52))),
]
CONDITIONS = ["BASELINE", "NO_SELECTION", "NO_DISCUSSION", "FULL"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""

SIG_TEX = {
    "***": r"$^{***}$", "**": r"$^{**}$",
    "*":   r"$^{*}$",   "†":  r"$^{\dagger}$", "": "",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all():
    """One row per (family, condition, seed, round, agent) with contribution, IN, DN."""
    rows = []
    for model_key, family, results_dir, variant, seeds in MODEL_SPECS:
        for seed in seeds:
            pattern = os.path.join(results_dir, model_key, variant,
                                   f"seed{seed}", "log_*.json")
            cond_data = {}
            for path in sorted(glob.glob(pattern)):
                try:
                    d = json.load(open(path))
                    cond_data[d["condition"]] = d
                except Exception:
                    continue

            for cond in CONDITIONS:
                d = cond_data.get(cond)
                if d is None:
                    continue
                run_id = f"{family}_s{seed}_{cond}"
                for r in d["round_logs"]:
                    contribs = {int(k): float(v)
                                for k, v in r["contributions"].items()}
                    percs = {int(k): v
                             for k, v in (r.get("perceptions") or {}).items()}
                    for aid, perc in percs.items():
                        if not perc:
                            continue
                        actual = contribs.get(int(aid))
                        inj    = perc.get("injunctive_norm")
                        desc   = perc.get("descriptive_norm")
                        if actual is None or inj is None or desc is None:
                            continue
                        rows.append({
                            "family":       family,
                            "model":        model_key,
                            "condition":    cond,
                            "seed":         seed,
                            "round":        r["round"],
                            "agent_id":     int(aid),
                            "run_id":       run_id,
                            "contribution": float(actual),
                            "IN":           float(inj),
                            "DN":           float(desc),
                        })
    return pd.DataFrame(rows)


def add_lags(df):
    df = df.sort_values(["run_id", "agent_id", "round"]).copy()
    grp = df.groupby(["run_id", "agent_id"])
    df["contribution_lead1"] = grp["contribution"].shift(-1)
    return df


def zscale(df, cols):
    for col in cols:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
    return df
