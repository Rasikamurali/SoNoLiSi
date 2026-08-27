"""
selection_mechanism_llama_mistral_qwen.py
-------------------------------------------
Runs social_selection_feedback_analysis.py's full mechanism-chain analysis
(contribution -> evaluation -> network weight -> group formation/selection
consequences -> contribution adjustment/exclusion, one OLS per link, SEs
clustered by run) restricted to llama/mistral/qwen (7B, "local" dataset).

Seeds come from TWO locations, merged (50 seeds/condition once both are
populated) by giving each model TWO MODEL_SPECS entries under the same
family label (run_id = family_seed_condition stays unique since the two
seed ranges don't overlap, so load_canonical_runs() merges them for free):
  - 10 canonical seeds (43-52):   results/{model}/local/seed{s}/log_*.json
  - 40 additional seeds (53-92):  code/results/{model}/local/additional_runs/seed{s}/log_*.json

social_selection_feedback_analysis.py does `from model_specs import
MODEL_SPECS` at its own import time, so MODEL_SPECS must be patched on the
model_specs module BEFORE social_selection_feedback_analysis is
imported (that import statement copies the current value, it is not a live
reference back to model_specs) — see CODEBASE_INVENTORY.md for the same
monkey-patch pattern used by the 13B/70B variant runners.

Output: figures/2026-03-22/additional_runs_results/04_selection_mechanism/
"""

import os
import sys

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "model_specs.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_specs as sm  # noqa: E402

BASE = "/data3/rasimura/social-norm-evo"
_base_specs = [spec for spec in sm.MODEL_SPECS if spec[0] in ("llama", "mistral", "qwen")]
_additional_specs = [
    (model_key, family, f"{BASE}/code/results", "local/additional_runs", list(range(53, 93)))
    for model_key, family, _results_dir, _variant, _seeds in _base_specs
]
sm.MODEL_SPECS = _base_specs + _additional_specs

import social_selection_feedback_analysis as ssfa  # noqa: E402  (picks up patched MODEL_SPECS above)

ssfa.OUT_DIR = (f"{BASE}/figures/2026-03-22/additional_runs_results/"
               "04_selection_mechanism")
ssfa.REF_FAMILY = "Llama-7B"   # "GPT" (the script's default) isn't in the restricted MODEL_SPECS

if __name__ == "__main__":
    print(f"Restricted MODEL_SPECS (picked up by social_selection_feedback_analysis): "
         f"{[s[1] for s in ssfa.MODEL_SPECS]}")
    print(f"OUT_DIR: {ssfa.OUT_DIR}")
    ssfa.main()
