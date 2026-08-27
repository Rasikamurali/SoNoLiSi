"""
run_s2_analysis.py
-------------------
Runs the main-paper analysis pipeline (minus the conversational/social-
learning annotation piece) for the S2 supplementary replication set:
GPT-5-mini, Llama-70B, Mistral-13B, Qwen-72B — local variant, seeds 42-51.
See PAPER_RESULTS_INVENTORY.md for what S2 covers.

Covers:
  1. Mean contribution per condition, pooled across the 10 seeds, per model
  2. Pairwise condition contrasts (OLS + Wald, Bonferroni) via behavior_quantified.py
  3. DN/IN convergence (cross-agent SD) via perception_consensus.py --tier s2 (subprocess, 2026-08-24)

(Gap-based alignment and the social-selection per-link OLS are handled by
alignment/gap_based_alignment.py --tier s2 and
selection/selection_mechanism_gpt5mini_llama70b_mistral13b_qwen72b.py
respectively, not here -- both already support this exact model subset.)

Same monkey-patch-then-call pattern as run_70b_analysis.py /
run_13b_analysis.py (see CODEBASE_INVENTORY.md §0).

Data-root note: Mistral-13B's 10-seed local data lives under the top-level
results/ folder (results/mistral_13b/local/), unlike the other three models
here which live under code/results/. Every script this orchestrator patches
(stability_analysis.py, behavior_quantified.py, this file's own
load_latest_log) assumes a single RESULTS root shared by all MODELS -- that
assumption is baked deep enough into the shared pipeline (6+ modules) that
generalizing it to per-model roots isn't worth the risk here. Instead,
code/results/mistral_13b/local/seed{42..51} are symlinks to the real seed
directories under results/mistral_13b/local/ (added 2026-08-18, data itself
untouched/not duplicated) so a single RESULTS="code/results" works for all
four models, matching every other orchestrator in this codebase.

Output: figures/local/s2_models/
"""

import subprocess
import sys
import os
import glob
import json
import warnings

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────

RESULTS  = "/data3/rasimura/social-norm-evo/code/results"
FIG_ROOT = "/data3/rasimura/social-norm-evo/figures/local/s2_models"
CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reorg (2026-08-11): see run_13b_analysis.py for why this block exists.
ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(ANALYSIS_DIR, "model_specs.py")):
    ANALYSIS_DIR = os.path.dirname(ANALYSIS_DIR)
SCRIPT_SUBFOLDER = {"behavior_quantified.py": "behavioral",
                    "contribution_all_conditions_plot.py": "behavioral"}

MODELS = ["gpt-5-mini", "llama_70b", "mistral_13b", "qwen_72b"]

MODEL_LABELS = {
    "gpt-5-mini":  "gpt-5-mini",
    "llama_70b":   "llama_70b",
    "mistral_13b": "mistral_13b",
    "qwen_72b":    "qwen_72b",
}

DISPLAY_LABELS = {
    "gpt-5-mini":  "GPT-5-mini",
    "llama_70b":   "Llama 3.1-70B",
    "mistral_13b": "Mistral-13B",
    "qwen_72b":    "Qwen 2.5-72B",
}

# Per-model colors/markers for temp_eval_alignment2.py's cross-model plot
# (item 5, perception_action_gap_across_models). llama_70b/mistral_13b/qwen_72b
# reuse the colors already assigned to them in network_development_all_models.py;
# gpt-5-mini has no prior color anywhere in the codebase so gets a fresh one.
MODEL_COLORS_S2 = {
    "gpt-5-mini":  "#d62728",
    "llama_70b":   "#bcbd22",
    "mistral_13b": "#8c564b",
    "qwen_72b":    "#17becf",
}
MODEL_MARKERS_S2 = {
    "gpt-5-mini":  "o",
    "llama_70b":   "s",
    "mistral_13b": "^",
    "qwen_72b":    "D",
}

VARIANT = "local"
SEEDS   = list(range(42, 52))

CONDITIONS      = ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE", "PURE_BASELINE"]

os.makedirs(FIG_ROOT, exist_ok=True)
os.makedirs(os.path.join(FIG_ROOT, "cross_model"), exist_ok=True)

for _p in [ANALYSIS_DIR] + [
    os.path.join(ANALYSIS_DIR, d) for d in os.listdir(ANALYSIS_DIR)
    if os.path.isdir(os.path.join(ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ─── Shared loader ────────────────────────────────────────────────────────────

def load_latest_log(model, seed, condition):
    pattern = os.path.join(RESULTS, model, VARIANT, f"seed{seed}", "log_*.json")
    best = {}
    for p in sorted(glob.glob(pattern)):
        try:
            with open(p) as f:
                d = json.load(f)
            best[d["condition"]] = d
        except Exception:
            continue
    return best.get(condition)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. COMPLETENESS CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_completeness():
    print("=" * 60)
    print("COMPLETENESS CHECK — S2 models / local")
    print("=" * 60)
    all_ok = True
    for model in MODELS:
        print(f"\n  {DISPLAY_LABELS[model]}")
        for cond in CONDITIONS:
            found   = [s for s in SEEDS if load_latest_log(model, s, cond) is not None]
            missing = [s for s in SEEDS if s not in found]
            status  = "✓" if len(found) == len(SEEDS) else f"✗ {len(found)}/{len(SEEDS)}"
            msg = f"    {cond:20s}: {status}"
            if missing:
                msg += f"  missing: {missing}"
                all_ok = False
            print(msg)
    print()
    if all_ok:
        print(f"  All models have {len(SEEDS)} seeds for all conditions.")
    else:
        print("  WARNING: some seeds are missing.")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. STABILITY ANALYSIS (DN/IN convergence, item 3)
# ═══════════════════════════════════════════════════════════════════════════════

def run_stability():
    """Superseded 2026-08-24 by perception_consensus.py --tier s2 (item a of
    the perception/alignment consolidation) -- now also produces the
    early/late-OLS panel that this block never generated for S2."""
    print("=" * 60)
    print("PERCEPTUAL CONSENSUS (perception_consensus.py) — s2 tier")
    print("=" * 60)
    script_path = os.path.join(ANALYSIS_DIR, "perception", "perception_consensus.py")
    result = subprocess.run([sys.executable, script_path, "--tier", "s2"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] perception_consensus.py exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1000:])
    else:
        for l in [l for l in result.stdout.splitlines() if l.strip()][-10:]:
            print(f"    {l}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONTRIBUTION TRAJECTORIES (item 1: mean contribution per condition,
#    pooled across the 10 seeds, per model)
# ═══════════════════════════════════════════════════════════════════════════════

def run_contribution_plots():
    print("=" * 60)
    print("CONTRIBUTION TRAJECTORIES — S2 models")
    print("=" * 60)
    out_dir = "/data3/rasimura/social-norm-evo/figures/SUPPLEMENTARY_RESULTS/2_bigger_models/1_contribution_trajectories"
    script_path = os.path.join(ANALYSIS_DIR, SCRIPT_SUBFOLDER["contribution_all_conditions_plot.py"],
                               "contribution_all_conditions_plot.py")
    cmd = ([sys.executable, script_path,
            "--variant", VARIANT, "--models"] + MODELS
           + ["--seeds"] + [str(s) for s in SEEDS]
           + ["--results-dir", RESULTS,
              "--out-dir", out_dir,
              "--out-name", "all_conditions_s2",
              "--model-labels"] + [f"{k}={v}" for k, v in DISPLAY_LABELS.items()])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] contribution_all_conditions_plot.py exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1500:])
    else:
        for line in [l for l in result.stdout.splitlines() if l.strip()]:
            print(f"    {line}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PERCEPTION-ACTION GAP (perception_action_gap_plot.py, 2026-08-24) -- item 5,
#    the S2 analog of MAIN_RESULTS/5_perception_action_gap_evolution/perception_action_gap_across_models.
# ═══════════════════════════════════════════════════════════════════════════════

def run_alignment_plots():
    """Superseded 2026-08-24 by perception_action_gap_plot.py --tier s2
    (item b of the perception/alignment consolidation)."""
    print("=" * 60)
    print("PERCEPTION-ACTION GAP (perception_action_gap_plot.py) — s2 tier")
    print("=" * 60)
    script_path = os.path.join(ANALYSIS_DIR, "alignment", "perception_action_gap_plot.py")
    result = subprocess.run([sys.executable, script_path, "--tier", "s2"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] perception_action_gap_plot.py exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1000:])
    else:
        for l in [l for l in result.stdout.splitlines() if l.strip()][-10:]:
            print(f"    {l}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OLS + WALD STATS (item 2)
# ═══════════════════════════════════════════════════════════════════════════════

def run_stats():
    print("=" * 60)
    print("PAIRWISE OLS + WALD — S2 models")
    print("=" * 60)
    stats_out = os.path.join(FIG_ROOT, "paper_stats")
    os.makedirs(stats_out, exist_ok=True)
    seeds_str = [str(s) for s in SEEDS]

    script = "behavior_quantified.py"
    extra = (["--variant", VARIANT, "--models"] + MODELS
             + ["--seeds"] + seeds_str + ["--out-dir", stats_out])
    script_path = os.path.join(ANALYSIS_DIR, SCRIPT_SUBFOLDER.get(script, ""), script)
    cmd = [sys.executable, script_path] + extra
    print(f"\n  Running: {script}")
    env = os.environ.copy()
    env["SNLS_RESULTS_DIR"] = RESULTS
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"  [WARN] {script} exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1500:])
    else:
        for line in [l for l in result.stdout.splitlines() if l.strip()][-10:]:
            print(f"    {line}")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    check_completeness()
    run_stability()
    run_contribution_plots()
    run_alignment_plots()
    run_stats()
    print(f"\nAll outputs -> {FIG_ROOT}")
