# Codebase Inventory

Purpose: catalog of every `.py` file in this repo — what it does, what it reads, what it writes — as groundwork for reorganizing into subfolders. Built 2026-08-10 by walking all 101 files (docstrings + import graph + `read_csv`/`to_csv`/`savefig`/`json.load` call sites), then read through and organized by hand. Treat the "Purpose" column as reliable (mostly straight from each file's own docstring); treat file groupings as a first-pass judgment call, not final.

**Before moving anything: read the "Import graph" section below.** Several scripts `import` each other directly by filename, assuming they all live flat in the same directory (no packages, no relative-import syntax). Moving a shared module into a subfolder without updating its importers will break them silently until run.

---

## 1. Top-level findings

- **101 `.py` files total.** 58 in `code/analysis/` alone (flat, no subfolders except the `gpt5_annotation_pilot/` one from this session) — that one directory is most of the clutter. `code/experiments/` (5), `code/robustness/` (10), `code/stability test/` (8), `code/` top-level (5), `archive/` (9).
- **87/101 files have a module docstring**; 14 don't. Of those 14, 9 are simulation-engine files (`SoNoLiSi_v5*.py` and the `stability test` engine variants) that are self-explanatory by name/location but weren't deeply read here — worth a closer look before any of them move. The other 5 are old `archive/` files, already quarantined.
- **`code/stability test/` has a literal space in the directory name** (needs `"quoting"` in every shell command — already bit us once this session) and mixes simulation-engine variants with their analysis scripts, unlike `code/experiments/` vs `code/analysis/` elsewhere. Candidate to split and rename.
- **Duplicate-variant clusters** (same analysis, different model-size/scope slice) — good candidates for consolidating into one parameterized script rather than N copies:
  - `behavioral_statistical.py` / `_13b.py` / `_70b.py` (the latter two import and reuse the base)
  - `run_local_analysis.py` / `run_13b_analysis.py` / `run_70b_analysis.py` (all three import the same 4 shared modules and monkey-patch their globals — see below)
  - `selection_precision.py` / `_avg7b.py`
  - `perception_consensus_combined_small_with_avg.py` / `perception_consensus_avg7b.py`
- **Naming is misleading in one spot**: `temp_eval_*.py` / `temp_evals_*.py` (7 files) sound like throwaway scratch scripts, but 4 of the 7 are actually load-bearing (imported by the `run_*_analysis.py` orchestrators or the `perception_consensus_*` pair) — see import graph. The other 3 (`temp_eval_alignment.py` *without* the "2", `temp_evals_discussion.py`, `temp_evals_selection.py`) aren't imported by anything found here, which makes them candidates for archiving, but "not imported" isn't the same as "dead" — they may be invoked directly (`python temp_evals_discussion.py`). Worth confirming with you before touching them.
- **One real bug found and fixed in passing this session**: `human_annotation_stratified_sample.py`'s `r.round` attribute access silently returned a bound pandas method instead of the round number (fixed to `r["round"]`) — corrupted `round` columns exist in already-generated files (`stratified_sample_for_coding.csv`, `full_corpus_eval_data.csv`); their `discussion_id` column is unaffected and usable as a workaround.

---

## 2. Import graph — load-bearing modules (do not relocate without fixing importers)

| Module | Imported by |
|---|---|
| `sobel_mediation.py` | `build_agent_round_panel`, `discussion_mechanism_analysis`, `discussion_talk_vs_behavior`, `exclusion_diagnostics`, `gap_based_alignment_test`, `lagged_ar_regression`, `social_selection_feedback_analysis` (7) |
| `discussion_talk_vs_behavior.py` | `discussion_mechanism_analysis`, `human_annotation_stratified_sample`, `sample_conversations_for_review`, `gpt5_annotation_pilot/build_full_corpus_eval_data`, `gpt5_annotation_pilot/estimate_full_corpus_cost` (5) |
| `stability_analysis.py`, `temp_eval_alignment2.py`, `temp_eval_network.py`, `temp_evals_behavior.py` | each imported by all three of `run_13b_analysis`, `run_70b_analysis`, `run_local_analysis`, which patch module-level globals (`VARIANT`, `SEEDS`, `FIG_ROOT`) before calling into them rather than duplicating logic — fragile but intentional per their docstrings |
| `repeated_phrases.py` | `discussion_mechanism_analysis`, `discussion_phrase_context` |
| `behavioral_statistical.py` | `behavioral_statistical_13b`, `behavioral_statistical_70b` |
| `selection_precision.py` | `selection_precision_avg7b` |
| `temp_eval_perception.py` | `perception_consensus_avg7b`, `perception_consensus_combined_small_with_avg` |
| `perception_consensus_avg7b.py` | `perception_consensus_combined_small_with_avg` |
| `discussion_talk_vs_behavior.py` + `human_annotation_stratified_sample.py` | `gpt5_annotation_pilot/build_full_corpus_eval_data.py` |
| `gpt_5_annotation_sl.py` | `gpt5_annotation_pilot/estimate_full_corpus_cost.py` |

Everything not listed here is a leaf script (nothing else imports it) — safe to move on its own once its *own* imports are fixed.

---

## 3. `code/experiments/` — simulation engines (5 files, no docstrings, not deeply read)

| File | Purpose (from name/context established this session) |
|---|---|
| `SoNoLiSi_v5.py` | Base simulation engine, presumably the "global" info-structure variant (discuss-then-form-groups) |
| `SoNoLiSi_v5_local.py` | "Local" info-structure variant, OpenAI-backed |
| `SoNoLiSi_v5_os.py` | Open-source/local-inference (vLLM) backend, global variant |
| `SoNoLiSi_v5_os_local.py` | Open-source backend, local variant — the one referenced directly this session (`form_groups()`, `run_sim()`, `LLM_discuss()`, group-composition-per-round via network weights) |
| `SoNoLiSi_v5_os_local_groupsizevary.py` | Local/OS variant allowing group size to vary (used for the MCPR=0.5/0.8 experiments per `mcpr_lmm_analysis.py`'s docstring) |

These are the actual simulators; everything else in the repo consumes their JSON log outputs. Worth a real read-through (not just filename inference) before deciding where they belong or whether they're all still current.

## 4. `code/` top-level (5 files)

| File | Purpose | Reads | Writes |
|---|---|---|---|
| `evaluation_outcomes.py` | Analyzes SoNoLiSi_v5 logs: behavior/expectations over time, perception convergence, discussion-vs-behavior alignment, selection enforcement, network assortativity | `results/` logs | figures |
| `extract_agent_data.py` | Extracts agent-level per-round data across all models/variants/seeds/conditions | `results/` logs | `results/agent_level_data.csv` |
| `internalization.py` | Bicchieri-style norm-internalization index per agent | logs | CSV + figures under `figures/.../internalization` |
| `plot_beautifier.py` | Combines per-model plots into composite publication figures | existing per-model figures | `figures/.../combined/` |
| `quant_behavioral.py` | Run-level behavioral metrics + bootstrap comparisons (across-condition, global-vs-local) | logs | `figures/.../behavioral_tables/*.csv` + bar plots |

## 5. `code/analysis/` — grouped by theme (58 files)

### 5a. Core shared utilities
| File | Purpose |
|---|---|
| `discussion_talk_vs_behavior.py` | Defines `load_messages()` — the canonical discussion-transcript loader used everywhere discussion data is needed. Also: talk-vs-behavior correspondence, first-proposal/anchoring, cross-group heterogeneity |
| `sobel_mediation.py` | Sobel mediation test (DN→IN→contribution); also the source of the shared IN/DN/contribution panel-loading logic reused by 6 other scripts |

### 5b. Behavioral statistics
| File | Purpose |
|---|---|
| `behavioral_statistical.py` | Formal stats on contribution: level/slope models, cross-model omnibus, dispersion (main 7B+GPT experiment) |
| `behavioral_statistical_13b.py` / `_70b.py` | Same 5 tests, 13B/70B model families |
| `behavior_quantified.py` | OLS w/ seed-clustered SEs, contribution+payoff, within- and cross-model tables |
| `paper_tables_behavioral.py` | Paper-ready tables from `quant_behavioral.py`'s CSVs |

### 5c. Perception / alignment statistics
| File | Purpose |
|---|---|
| `perception_quantified.py` | OLS: IN ~ DN × model × condition |
| `perception_dn_in.py` | Reverse regression: DN ~ IN × condition |
| `perception_consensus_avg7b.py` / `perception_consensus_combined_small_with_avg.py` | Pooled-across-7B-models version of an existing consensus table, plus a side-by-side per-model+pooled table |
| `alignment_quantified.py` | OLS on IN_gap/DN_gap (perception − behavior) |
| `gap_based_alignment_test.py` | Do agents shift next-round contribution toward their own stated IN/DN gap? |
| `gap_convergence_did.py` | DiD: does IN-gap close faster than DN-gap? |
| `in_dn_predictive.py` | Does IN or DN better predict contribution? |
| `timestep_did.py` | Lagged-predictor tests of DN→IN→contribution temporal ordering |
| `lagged_ar_regression.py` | Autoregressive extension of the lagged mediation model + FE controls |
| `mechanism_interaction.py` | Hindrance/synergy test between Selection/Discussion/Perception mechanisms |

### 5d. Network / selection
| File | Purpose |
|---|---|
| `eval_network_quantified.py` | OLS on network metrics (alignment, gini, top_share) early-vs-late + reward-loop lagged models |
| `network_development_all_models.py` | Network trajectories (gini, top_share) across all model sizes |
| `selection_precision.py` / `_avg7b.py` | Does group-formation selection actually favor top-quartile prior contributors? Per-model / pooled |
| `lockin_check.py` | Do agent pairs get "locked in" to repeated group co-membership? |
| `social_selection_feedback_analysis.py` | Full feedback-chain reconstruction: contribution → evaluation → weight → selection → next contribution (has a companion audit doc, `exports/social_selection_feedback/mechanism_audit.md`, worth reading first per its own docstring) |
| `weight_update_check.py` | Self-contained demo/animation of the network weight-update rule |
| `exclusion_diagnostics.py` | When does each agent first get excluded from group formation? |
| `build_agent_round_panel.py` | Builds the long-format agent×round panel (`exports/round_level_agent_panel.csv`) other scripts consume |

### 5e. Discussion / discourse + LLM annotation
| File | Purpose |
|---|---|
| `discussion_mechanism_analysis.py` | The big one — local-anchoring→collective-ratification analysis (talk-behavior correspondence, first-proposal anchoring, cross-group convergence, conversation-function lexicon) |
| `discussion_phrase_context.py` | Repeated-phrase threshold sweep + context tagging (round/speaker-position) |
| `repeated_phrases.py` | N-gram (2-5) frequency finder across discussion transcripts |
| `sample_conversations_for_review.py` | Samples whole-run transcripts for qualitative review |
| `human_annotation_stratified_sample.py` | Draws the 80-discussion cross-family stratified sample for hand-coding (has the `r.round` bug, now fixed) |
| `gpt_5_annotation_sl.py` | The LLM-as-judge annotation pipeline built this session (codebook-driven, shot-condition testing, now supports per-group batched annotation + external shot/eval data) |
| `gpt5_annotation_pilot/` (6 files) | Everything built this session on top of `gpt_5_annotation_sl.py`: `select_shot_discussions.py`, `estimate_full_corpus_cost.py`, `build_full_corpus_eval_data.py`, `analyze_full_corpus_patterns.py`, `build_pattern_tables.py`, `build_family_trend_table.py` — this subfolder is itself a template for how the rest of `code/analysis/` could be organized (one topic, one folder) |

### 5f. MCPR mixed-effects models
| File | Purpose |
|---|---|
| `mcpr_lmm_analysis.py` | Mixed-effects models of MCPR effects on contribution (7B only) |
| `mcpr_lmm_alignment_network.py` | Same, for alignment (IN/DN gap) and network (spearman/gini/top_share) outcomes |

### 5g. Shock / intervention experiments
| File | Purpose |
|---|---|
| `shock_analysis.py` | DiD: adversarial-agent introduction effect on other agents, round-10 and round-20 injections |
| `its_shock_round20.py` | Interrupted-time-series version of the round-20 shock (DiD isn't identified there — no post-shock control) |
| `new_intro_analysis.py` (in `code/analysis/`) | New-agent-introduction behavior/perception/alignment trajectories |
| `new_intro_adversarial_plots.py` | Cross-model trajectory plots for the adversarial new-intro experiments |
| `plot_dissociation_belief_vs_contribution.py` | The figure we edited this session — belief vs. behavior recovery around a group-composition injection (regenerated from saved aggregate; original plotting script is lost) |

### 5h. Cross-model / cross-size orchestrators
| File | Purpose |
|---|---|
| `compare_7b_13b.py` | 7B vs ~13B within-family comparison |
| `compare_global_local.py` | Global vs local info-structure comparison |
| `run_local_analysis.py` / `run_13b_analysis.py` / `run_70b_analysis.py` | Orchestrators that monkey-patch and re-run the shared `stability_analysis`/`temp_eval_alignment2`/`temp_eval_network`/`temp_evals_behavior` modules per model tier |
| `robustness_analysis.py` / `robustness_split.py` | Reproduces the standard figure set for each robustness experiment (N=16/20 agents, group size variants), and splits those by model size/family |
| `subsample_stability.py` | Are main conclusions robust to seed subsampling (5-8 of 10 seeds)? |
| `stability_analysis.py` | Cross-agent SD convergence rate (contribution/IN/DN) — also imported by the `run_*_analysis` orchestrators |

### 5i. Plotting / table utilities
| File | Purpose |
|---|---|
| `plot_prompt_boxes.py` | Renders the 5 LLM prompt templates as paper-style figure boxes |

### 5j. "temp_eval*/temp_evals_*" building blocks — misleadingly named, mixed status
| File | Status |
|---|---|
| `temp_eval_alignment2.py`, `temp_eval_network.py`, `temp_evals_behavior.py` | **Imported** by all 3 `run_*_analysis.py` orchestrators — load-bearing despite the name |
| `temp_eval_perception.py` | **Imported** by `perception_consensus_avg7b.py` / `perception_consensus_combined_small_with_avg.py` — load-bearing |
| `temp_eval_alignment.py` (no "2") | Not imported anywhere found — appears to be an earlier/superseded standalone version of `temp_eval_alignment2.py`; candidate for archiving, but confirm first |
| `temp_evals_discussion.py`, `temp_evals_selection.py` | Not imported anywhere found — standalone scripts (discussion lexical/log-odds features; weight-based selection metrics); likely fine to rename out of the "temp" namespace rather than archive, since neither looks throwaway in content |

---

## 6. `code/robustness/` (10 files) — already a reasonably contained cluster

| File | Purpose |
|---|---|
| `temperature_sweep.py` / `analyze_temperature_sweep.py` | Run + analyze: baseline stability across decision temperatures |
| `prompt_variants.py` / `analyze_prompt_variants.py` | Run + analyze: sensitivity to structural system-prompt framing |
| `prompt_sensitivity.py` / `analyze_prompt_sensitivity.py` | Run + analyze: word-level system-prompt sensitivity |
| `parameter_sweep.py` | OFAT sweep over `SimConfig` parameters (network update rate, tie-removal threshold, etc.) |
| `analyze_cross_model.py` | Cross-model summary combining all three robustness tests above |
| `analyze_mechanism_cross_model.py` / `analyze_mechanism_vs_pretraining.py` | Is FULL-vs-baseline divergence mechanism-driven (builds over rounds) or a pre-training artifact (present at round 1)? |

Note the run/analyze pairing convention here (`prompt_variants.py` runs it, `analyze_prompt_variants.py` analyzes it) — a cleaner pattern than most of `code/analysis/`, possibly worth adopting elsewhere.

## 7. `code/stability test/` (8 files, space in dirname) — mixes engines + analysis

| File | Purpose |
|---|---|
| `logit_SoNoLiFi_v5.py` | Engine variant capturing token-level logprob distributions (no docstring) |
| `logit_analysis.py` | Analyzes the above: point-estimate vs. logprob-distribution comparison, entropy trajectories |
| `new_group_SoNoLiSi_os_local.py` / `new_group_IN_SoNoLiSi_os_local.py` | Engine variants for group-composition-injection experiments (no docstring) |
| `new_intro_SoNoLiFi_v5.py` / `new_intro_SoNoLiSi_v5_local.py` / `new_intro_SoNoLiSi_v5_os_local.py` | Engine variants for new-agent-introduction experiments, 3 backend/variant flavors (no docstring) |
| `new_intro_analysis.py` | Analyzes the new-intro experiment: extension-round trajectories, pre/post comparison |

Recommend: split engines into `code/experiments/`, analysis into `code/analysis/`, and rename the directory (or retire it once its outputs are folded into the main experiment/analysis split) — the space in the path alone makes it error-prone from the shell.

## 8. `archive/` (9 files) — already quarantined, low priority

Old simulation-engine versions (`SoNoLiSi_v4.py`, `SoNoLiSi_v5_fixed*.py`, `SoNoLiSi_collab_write.py` — a Google-Docs collaborative-writing variant, looks like an abandoned direction) and old analysis scripts (`CT.py`, `appendix_evaluations.py`, `causal_checks.py`, `coop_vs_viol.py`, `diffbackbone.py`). These are already out of the way; probably don't need attention in this pass.

---

## Suggested next step

Given the import graph in §2, the lowest-risk reorg order is probably: (1) confirm/archive the 3 orphaned `temp_eval*` files, (2) split `code/stability test/` into engines vs. analysis and fix its 8 files' imports, (3) tackle the big flat `code/analysis/` folder using the thematic groupings in §5 as candidate subfolders (mirroring how `gpt5_annotation_pilot/` already works) — moving one theme at a time and fixing that theme's imports before moving the next, rather than one big move.
