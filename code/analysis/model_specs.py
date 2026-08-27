"""
model_specs.py
---------------
Shared registry + data-loading helpers imported by every other script in
this release, so the model list, paths, and seed ranges stay in one place.

  MODEL_SPECS: one row per model family (10 total: GPT + 3 families each
    at 7B/13B-14B/70B-72B + GPT-5-mini), with results dir, variant, seeds.
  CONDITIONS: the 4 main experimental conditions.
  load_all(): flattens every family's raw JSON logs into one table, one
    row per (family, condition, seed, round, agent), with contribution
    and the two elicited expectations (IN = normative, DN = empirical).
  add_lags()/zscale()/stars()/SIG_TEX: shared formatting/transform helpers.

Inputs: results/ and code/results/ (see MODEL_SPECS). Writes nothing.

Also a directory landmark: other scripts walk up their own path to find
this file, then add every sibling subfolder to sys.path -- so it must
stay directly inside code/analysis/.
"""

import os
import json
import glob
import pandas as pd

# BASE is the root of this release (the folder containing "code/"), computed
# from this file's own location so the release still works if it's moved or
# copied elsewhere -- it does not depend on any fixed install path.
BASE    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    # Converts a p-value into the usual significance-star notation
    # (*** < .001, ** < .01, * < .05, dagger < .10, nothing otherwise).
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return ""

# LaTeX versions of the same stars, for building table cells directly.
SIG_TEX = {
    "***": r"$^{***}$", "**": r"$^{**}$",
    "*":   r"$^{*}$",   "†":  r"$^{\dagger}$", "": "",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all():
    """One row per (family, condition, seed, round, agent) with contribution, IN, DN."""
    # Walk every model family, seed, and condition listed in MODEL_SPECS,
    # open each run's raw JSON log, and flatten it into one long table with
    # one row per agent per round -- this is the common starting point most
    # of the other analysis scripts build on.
    rows = []
    for model_key, family, results_dir, variant, seeds in MODEL_SPECS:
        for seed in seeds:
            pattern = os.path.join(results_dir, model_key, variant,
                                   f"seed{seed}", "log_*.json")
            # A run can have more than one log file on disk (e.g. from a
            # rerun); keep only the latest one per condition.
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
                        # Skip an agent-round if any of the three values we
                        # need (actual contribution, normative expectation,
                        # empirical expectation) is missing.
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
    # Adds "contribution_lead1": each agent's contribution in the NEXT
    # round, aligned onto the current round's row. Used to model how an
    # agent's contribution changes from one round to the next.
    df = df.sort_values(["run_id", "agent_id", "round"]).copy()
    grp = df.groupby(["run_id", "agent_id"])
    df["contribution_lead1"] = grp["contribution"].shift(-1)
    return df


def zscale(df, cols):
    # Standardizes each listed column to mean 0, standard deviation 1
    # (adds a new "<col>_z" column; the original column is left unchanged).
    for col in cols:
        df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
    return df
