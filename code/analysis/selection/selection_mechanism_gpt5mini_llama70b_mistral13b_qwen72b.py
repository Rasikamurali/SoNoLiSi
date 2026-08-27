"""
selection_mechanism_gpt5mini_llama70b_mistral13b_qwen72b.py
--------------------------------------------------------------
Runs social_selection_feedback_analysis.py's full mechanism-chain analysis
(contribution -> evaluation -> network weight -> group formation/selection
consequences -> contribution adjustment/exclusion, one OLS per link, SEs
clustered by run) restricted to the S2 supplementary replication set:
GPT-5-mini, Llama-70B, Mistral-13B, Qwen-72B (10 seeds each, "local" variant).
See PAPER_RESULTS_INVENTORY.md for what S2 covers and why.

Same monkey-patch pattern as selection_mechanism_llama_mistral_qwen.py and
the 13B/70B variant runners (see CODEBASE_INVENTORY.md): patch
model_specs.MODEL_SPECS to the 4-family subset BEFORE importing
social_selection_feedback_analysis (its `from model_specs import
MODEL_SPECS` copies the value at that moment, it is not a live reference).

REF_FAMILY is also repointed: the script's default reference family for the
family fixed effect is "GPT", which isn't present in this subset (GPT-5-mini
is a different family label) -- patsy would raise "specified level 'GPT' not
found" otherwise.

Output: code/analysis/exports/social_selection_feedback_s2/
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
_S2_FAMILIES = ("gpt-5-mini", "llama_70b", "mistral_13b", "qwen_72b")
sm.MODEL_SPECS = [spec for spec in sm.MODEL_SPECS if spec[0] in _S2_FAMILIES]

import social_selection_feedback_analysis as ssfa  # noqa: E402  (picks up patched MODEL_SPECS above)

ssfa.OUT_DIR = f"{BASE}/code/analysis/exports/social_selection_feedback_s2"
ssfa.REF_FAMILY = "GPT-5-mini"  # "GPT" (the script's default) isn't in this subset

if __name__ == "__main__":
    print(f"Restricted MODEL_SPECS (picked up by social_selection_feedback_analysis): "
          f"{[s[1] for s in ssfa.MODEL_SPECS]}")
    print(f"OUT_DIR: {ssfa.OUT_DIR}")
    ssfa.main()
