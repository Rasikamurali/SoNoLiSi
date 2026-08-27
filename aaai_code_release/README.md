# AAAI Code Release — Analysis & Simulation Code

This folder is a clean, minimal subset of the full research repository,
containing the code needed to reproduce five reported analyses:

1. **Contribution trajectories** — mean contribution over rounds, by condition.
2. **OLS + Wald pairwise-contrast tests** on contribution, seed-clustered,
   Bonferroni-corrected.
3. **Perceptual convergence** — cross-agent SD of injunctive/descriptive
   norm perceptions over rounds (early-vs-late OLS + continuous slope).
4. **Perception-alignment regression** (expectation-gap effect on
   contribution) and its **lagged specification**.
5. **Social-selection mechanism analysis** — the per-link feedback chain
   from undercontribution through evaluation, network-weight update,
   selection consequences, and behavioral adjustment.

Each family includes its main (7B/8B) result plus every robustness tier
that was run for the paper: bigger open/proprietary models (s2), 13B and
70B model tiers, community-size / group-size structural sweeps, and (for
families 1 and 5 only, since it's the only arm that exists for them) the
no-expectations ablation.



## Directory layout

```
code/experiments/
  SoNoLiSi_v5_local.py               the simulation itself -- produces every
                                      log this release's analyses read
                                      (results/, code/results/). See
                                      "Running the simulation" below.
  new_group_IN_SoNoLiSi_os_local.py  supplementary adversarial-agent-injection
                                      experiment (mid-run agent replacement +
                                      recovery) -- included for transparency;
                                      not consumed by any of the five analyses
                                      above (its own analysis is out of scope
                                      for this release, see "What's
                                      intentionally excluded" below).
code/analysis/
  model_specs.py                          shared model/family registry + helpers
  behavioral/
    behavior_quantified.py                (2) OLS+Wald, main 7B/8B tier
    behavior_quantified_noexpect.py       (2) OLS+Wald, no-expectations arm
    contribution_all_conditions_plot.py   (1) trajectory figure, all tiers via CLI args
    community_group_contribution_tables.py(2) community/group structural-sweep tables --
                                           NOTE: unlike every other family, this does
                                           NOT call behavior_quantified.py's own Q1
                                           function for its regression; it's a separate
                                           implementation (via analyze_structural_
                                           robustness.py) of a similarly-shaped question.
                                           Known architectural inconsistency, left as-is
                                           (not a duplicate to remove -- see "What's
                                           intentionally excluded" below).
    analyze_structural_robustness.py      shared community/group data-build + OLS/Wald
                                           machinery that (2)'s structural-sweep table
                                           reuses (relocated here from a misplaced path
                                           in the original repo; see note below)
  perception/
    perception_consensus.py               (3) convergence, all tiers via --tier
  alignment/
    gap_based_alignment.py                (4) primary regression + internal lagged
                                           respecification, all tiers via --tier
    lagged_alignment_check.py             (4) independent lagged-level cross-check,
                                           all tiers via --tier
    build_agent_round_panel.py            builds this family's input panel (moved
                                           here from a "selection/" location it
                                           had no functional tie to; only
                                           gap_based_alignment.py reads its output)
  selection/
    social_selection_feedback_analysis.py (5) main mechanism script, all tiers
                                           (7b/s2/community/group/mcpr) via --tier
    social_selection_feedback_analysis_noexpect.py (5) no-expectations arm
    community_ss_mechanism_table.py       (5) compact community-sweep summary table,
                                           built on the main script's own functions
                                           via its configure_tier(), not a separate
                                           implementation
```


## Running the simulation

`code/experiments/SoNoLiSi_v5_local.py` produced every `log_*.json` file the
five analyses above read. Requirements:

- `OPENAI_API_KEY` in the environment, for any OpenAI model (e.g.
  `gpt-4o-mini`, `gpt-5-mini`). The `openai`/`python-dotenv` packages are
  only imported if an OpenAI model is actually requested.
- `vllm` installed and a GPU, for any open-weight model (Llama/Mistral/Qwen).
  Only imported if a local model is actually requested.
- `HF_TOKEN` in the environment, for gated Hugging Face model weights.
- Optionally `HF_MODEL_CACHE_DIR`, to point the vLLM download cache
  somewhere other than `~/.cache/huggingface/hub`.

Example (one OpenAI model, one open-weight model, one seed, all 5 standard
conditions):
```
python3 code/experiments/SoNoLiSi_v5_local.py \
    --model gpt-4o-mini llama --seeds 43 --device 0
```
Full CLI: `--help`. Output layout
(`results/{model}/{variant}/seed{N}/log_*.json` and `conversations_*.json`)
matches what every analysis script's `model_specs.py` registry expects.

`code/experiments/new_group_IN_SoNoLiSi_os_local.py` re-runs the same simulation but silently replaces a subset of agents mid-run with adversarial ones (self-described low-cooperation newcomers who inherit the displaced agents' reputation-network edges) to test how quickly an emerged cooperation norm recovers from an internal shock; see the script's own module docstring for the full design. Same environment requirements as above (open-weight models only; no OpenAI backend). Example:
```
python3 code/experiments/new_group_IN_SoNoLiSi_os_local.py \
    --model llama --seeds 43 --device 0
```

## Running each analysis

### 1. Contribution trajectories
```
python3 code/analysis/behavioral/contribution_all_conditions_plot.py \
    --variant local --models gpt llama mistral qwen --out-dir <dir>
```
Re-invoke with `--models llama_13b mistral_13b qwen_14b`, `--models
llama_70b qwen_72b`, or `--models gpt-5-mini llama_70b mistral_13b
qwen_72b` for the 13B/70B/s2 tiers. Community/group sweeps: pass
`--variant local/N16_G4` (etc.) — see the script's `--help`.

### 2. OLS + Wald pairwise contrasts
```
python3 code/analysis/behavioral/behavior_quantified.py            # main 7B/8B tier
python3 code/analysis/behavioral/behavior_quantified_noexpect.py   # no-expectations arm
python3 code/analysis/behavioral/community_group_contribution_tables.py  # community/group sweeps
```

### 3. Perceptual convergence
```
python3 code/analysis/perception/perception_consensus.py --tier 7b
python3 code/analysis/perception/perception_consensus.py --tier {s2,13b,70b,pooled,community,group}
python3 code/analysis/perception/perception_consensus.py --tier all   # every tier in one call
```

### 4. Perception-alignment regression + lagged specification
```
python3 code/analysis/alignment/gap_based_alignment.py --tier 7b
python3 code/analysis/alignment/gap_based_alignment.py --tier {s2,13b,70b,pooled,community,group}
python3 code/analysis/alignment/lagged_alignment_check.py --tier <same choices>
```
`gap_based_alignment.py`'s own `run_lagged_level_model()` (inside the
same file) is the primary lagged specification; `lagged_alignment_check.py`
is an independent cross-check on raw (non-gap) IN/DN — report the former
as the headline number, per the paper's convention.

### 5. Social-selection mechanism
```
python3 code/analysis/selection/social_selection_feedback_analysis.py --tier 7b
python3 code/analysis/selection/social_selection_feedback_analysis.py --tier {s2,community,group,mcpr,all}
python3 code/analysis/selection/social_selection_feedback_analysis_noexpect.py
python3 code/analysis/selection/community_ss_mechanism_table.py
```

