# Codebase Inventory

Catalog of every `.py` file in this repo — what it does, what it reads, what it writes. Built 2026-08-10 as groundwork for reorganizing `code/analysis/`; **the reorg described here was executed on 2026-08-11.** Section 0 covers the new layout and how to run things now. Sections 1-8 are the original per-file catalog, left in place as reference (file purposes/imports are still accurate; a few file *paths* referenced in prose now need their theme subfolder prefixed — see the mapping table in §0).

---

## 0. Reorg status: DONE (2026-08-11)

`code/analysis/`'s 58 flat files were split into 9 theme subfolders, git-tracked via `git mv` (history preserved). Two files stayed at the top level of `code/analysis/` because they're imported across many themes: ~~**`sobel_mediation.py`**~~ and **`discussion_talk_vs_behavior.py`**.

**Update (2026-08-24):** `perception/` and `alignment/` were consolidated from
~15 overlapping scripts down to 4 canonical, tier-parametrized files (see
`MAIN_PAPER_RESULTS.md`'s "Perception/alignment consolidation" section for
the full mapping and rationale):
`perception/perception_consensus.py`, `alignment/perception_action_gap_plot.py`,
`alignment/gap_based_alignment.py`, `alignment/lagged_alignment_check.py`.
The superseded originals (`temp_eval_perception.py`, `perception_consensus_avg7b.py`,
`perception_consensus_combined_small_with_avg.py`, `perception_consensus_combined_s2.py`,
`perception_quantified.py`, `perception_dn_in.py`, `in_dn_predictive.py`,
`timestep_did.py`, `expectation_gap_evolution_plots.py`,
`mediation_llama_mistral_qwen.py`, `alignment_quantified.py`,
`gap_based_alignment_test.py`, `lagged_ar_regression.py`, `temp_eval_alignment.py`,
`temp_eval_alignment2.py`, `cross-model/stability_analysis.py`) were archived
to `archive/{perception,alignment,cross-model}/`. `gap_convergence_did.py` and
`mechanism_interaction.py` were untouched (different questions, not part of
this consolidation). The `perception/` and `alignment/` rows in the table
below, and the 13-line per-file catalog in Sections 1-8, now describe the
pre-2026-08-24 state for the archived files — read `MAIN_PAPER_RESULTS.md`
for the current canonical files.

**Update (2026-08-17):** `sobel_mediation.py` was archived to `archive/sobel_mediation.py` — its own Sobel-mediation-test analysis had been superseded by `perception/mediation_llama_mistral_qwen.py`'s direct lagged-gap regression, but the file had accumulated unrelated shared infra (`MODEL_SPECS`/`CONDITIONS`/`BASE` registry, plus the `load_all`/`add_lags`/`zscale`/`stars` panel-loading helpers) that 7 other scripts depended on, and its presence at `code/analysis/` was also used as a directory anchor by ~19 scripts. That shared infra was extracted into a new **`model_specs.py`**, which now plays both roles (shared config/utilities + anchor file) that `sobel_mediation.py` used to play. All import statements and anchor-walk loops below were updated accordingly; references to `sobel_mediation.py` in the rest of this section describe the pre-2026-08-17 state and are kept for history — read `model_specs.py` wherever the anchor/shared-infra role is meant.

### New layout

| Folder | Files |
|---|---|
| `behavioral/` | `behavioral_statistical.py`, `behavioral_statistical_13b.py`, `behavioral_statistical_70b.py`, `behavior_quantified.py`, `paper_tables_behavioral.py`, `temp_evals_behavior.py` |
| `perception/` | `perception_consensus.py` (as of 2026-08-24; superseded files listed in the 2026-08-24 update note above, now in `archive/perception/`) |
| `alignment/` | `gap_based_alignment.py`, `perception_action_gap_plot.py`, `lagged_alignment_check.py`, `gap_convergence_did.py`, `mechanism_interaction.py` (as of 2026-08-24; superseded files now in `archive/alignment/`) |
| `selection/` | `eval_network_quantified.py`, `network_development_all_models.py`, `selection_precision.py`, `selection_precision_avg7b.py`, `lockin_check.py`, `social_selection_feedback_analysis.py`, `weight_update_check.py`, `exclusion_diagnostics.py`, `build_agent_round_panel.py`, `temp_eval_network.py`, `temp_evals_selection.py` — network-related files folded in here, no separate "network" folder was requested |
| `discussion/` | `discussion_mechanism_analysis.py`, `discussion_phrase_context.py`, `repeated_phrases.py`, `sample_conversations_for_review.py`, `human_annotation_stratified_sample.py`, `gpt_5_annotation_sl.py`, `temp_evals_discussion.py`, and the whole `gpt5_annotation_pilot/` subfolder |
| `shock/` | `shock_analysis.py`, `its_shock_round20.py`, `new_intro_analysis.py`, `new_intro_adversarial_plots.py`, `plot_dissociation_belief_vs_contribution.py` |
| `mcpr/` | `mcpr_lmm_analysis.py`, `mcpr_lmm_alignment_network.py` |
| `cross-model/` | `compare_7b_13b.py`, `compare_global_local.py`, `run_local_analysis.py`, `run_13b_analysis.py`, `run_70b_analysis.py`, `robustness_analysis.py`, `robustness_split.py`, `subsample_stability.py`, `evaluation_multi.py` (`stability_analysis.py` archived 2026-08-24, superseded by `perception/perception_consensus.py`) |
| `plotting/` | `plot_prompt_boxes.py` |
| *(top level)* | `sobel_mediation.py`, `discussion_talk_vs_behavior.py` — shared infra, imported across most of the above |

### Judgment calls made (flagged per your instruction, not deleted, nothing hidden)

- **`in_dn_predictive.py` / `timestep_did.py`** → put in `perception/`, but they're borderline `alignment/` too (both test IN/DN's relationship to contribution, not exactly a "gap").
- **`mechanism_interaction.py`** → put in `alignment/` (its moderator is IN_z), but it genuinely tests Selection×Discussion×Perception interaction — doesn't belong cleanly anywhere.
- **`plot_dissociation_belief_vs_contribution.py`** → put in `shock/` (it's about the injection-recovery finding specifically) rather than `plotting/` (which I read as generic/topic-agnostic plot utilities, of which there's only `plot_prompt_boxes.py`).
- **`temp_eval_alignment.py`** (no "2") → moved into `alignment/` alongside its likely successor `temp_eval_alignment2.py`. Still not imported by anything — possibly safe to archive later, not touched further here.
- **`temp_evals_discussion.py`, `temp_evals_selection.py`** → moved by topic despite the misleading "temp" name; not imported anywhere, standalone scripts, content doesn't look like scratch work.
- **`evaluation_multi.py`** → put in `cross-model/` (docstring: "multi-seed, multi-model evaluation") but it's really a foundational metrics-building script rather than a themed analysis; didn't fit any of the 9 requested folders cleanly.
- **`sobel_mediation.py`, `discussion_talk_vs_behavior.py`** → deliberately NOT moved into any theme folder; they're imported by 7 and 5 other files respectively, spanning `selection/`, `alignment/`, `discussion/`. Putting them in any one theme would be arbitrary.

### How imports were kept working

Nothing here is a Python package (no `__init__.py`, no relative-import syntax) — every local import is a bare `import other_module`, which only resolves if `other_module.py` is on `sys.path`. Moving files into subfolders broke this for every cross-folder import. Fixed by inserting this pattern at the top of the ~19 affected files, right before their local imports:

```python
_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ANALYSIS_DIR, "sobel_mediation.py")):
    _ANALYSIS_DIR = os.path.dirname(_ANALYSIS_DIR)
for _p in [_ANALYSIS_DIR] + [
    os.path.join(_ANALYSIS_DIR, d) for d in os.listdir(_ANALYSIS_DIR)
    if os.path.isdir(os.path.join(_ANALYSIS_DIR, d)) and not d.startswith((".", "__"))
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

This walks up from the file's own location until it finds `code/analysis/` (using `sobel_mediation.py` as a marker, since that file's presence reliably identifies the directory), then adds that directory plus every one of its subfolders to `sys.path`. Robust to nesting depth, doesn't require hardcoding which theme a given module ended up in, and self-maintaining if folders are renamed later. `run_13b_analysis.py` / `_70b_` / `_local_` additionally needed a `SCRIPT_SUBFOLDER` dict since they invoke `behavior_quantified.py`/`perception_quantified.py` as **subprocesses** (`os.path.join(CODE_DIR, script)`), which needed the correct new subfolder spliced into the constructed path, not just a `sys.path` fix.

Files that only import *same-folder* siblings (e.g. `behavioral_statistical_13b.py` importing `behavioral_statistical.py`, now both in `behavioral/`) needed no change — Python auto-adds a script's own directory to `sys.path` when it's run directly.

### Data/output paths: mostly untouched, one invocation-convention note

Most scripts read/write via bare relative strings like `OUT_DIR = "exports"` or `PANEL_PATH = "exports/round_level_agent_panel.csv"`. These resolve against the process's **current working directory at runtime**, not the `.py` file's location — so moving the file itself doesn't break them, *as long as you keep invoking scripts with `code/analysis/` as your cwd* (as this session always did: `cd code/analysis && python3 <script>.py`). With the reorg, that becomes `cd code/analysis && python3 behavioral/behavioral_statistical.py` (subfolder-qualified script path, same cwd as always) — nothing about the bare `"exports/..."` strings inside those scripts needed to change.

The exception is anything that was **already `__file__`-anchored** with a fixed hop-count assuming the old (shallower) location — those needed the hop-count increased or replaced with the `_ANALYSIS_DIR` anchor above. Fixed in: `evaluation_multi.py` (`PROJECT_DIR`, +1 `dirname()` hop), the 3 `run_*_analysis.py` orchestrators (subprocess script paths), and 6 files under `discussion/gpt5_annotation_pilot/` (`RUN_DIR`/`IN_PATH`/`OUT_PATH` 2-hop → anchor; `select_shot_discussions.py`'s bare `"../exports/..."` → anchor since a literal `..` count is exactly the kind of thing that silently breaks with nesting depth).

One file needed a real fix beyond hop-counting: **`gpt_5_annotation_sl.py`** had two constants making *different* cwd assumptions in the same file — `PILOT_DIR = "gpt5_annotation_pilot"` (assumed cwd = own directory) and `OUT_DIR = "exports/gpt5_annotation_pilot"` (assumed cwd = `code/analysis/`). These happened to be compatible before the move only because the file lived directly in `code/analysis/`; moving it into `discussion/` (while `gpt5_annotation_pilot/` moved with it as a sibling) broke that coincidence. Fixed both to be independently `__file__`-anchored.

### Verification performed

- `python3 -m py_compile` on all 56 moved files — clean.
- Module-level exec-import test (imports + top-level constants, not `main()`) on all 14 files with content edits, plus the 5 "same-folder, should need no change" files, plus all 3 `run_*_analysis.py` + `evaluation_multi.py` — all clean.
- Explicitly re-tested the 4 function-local deferred imports in `run_local_analysis.py` (`stability_analysis`, `temp_eval_alignment2`, `temp_eval_network`, `temp_evals_behavior`) and the 2 subprocess-invoked scripts (`behavior_quantified.py`, `perception_quantified.py`) resolve to real files — all clean.
- Did **not** run any script's actual `main()` / full pipeline (several hit real data/API costs) — only import-time and path-construction correctness is verified. If something breaks deeper inside a `main()` body, it predates this reorg or is a genuinely new issue worth reporting.

---

## 1. Top-level findings (original, pre-reorg)

- **101 `.py` files total.** 58 in `code/analysis/` alone (flat, no subfolders except the `gpt5_annotation_pilot/` one from this session) — that one directory is most of the clutter. `code/experiments/` (5), `code/robustness/` (10), `code/stability test/` (8), `code/` top-level (5), `archive/` (9).
- **87/101 files have a module docstring**; 14 don't. Of those 14, 9 are simulation-engine files (`SoNoLiSi_v5*.py` and the `stability test` engine variants) that are self-explanatory by name/location but weren't deeply read here — worth a closer look before any of them move. The other 5 are old `archive/` files, already quarantined.
- **`code/stability test/` has a literal space in the directory name** (needs `"quoting"` in every shell command — already bit us once this session) and mixes simulation-engine variants with their analysis scripts, unlike `code/experiments/` vs `code/analysis/` elsewhere. Candidate to split and rename. *(Not yet done — this reorg pass covered `code/analysis/` only.)*
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

Everything not listed here is a leaf script (nothing else imports it) — safe to move on its own once its *own* imports are fixed. **(As of §0, all of the above have been re-verified working post-move.)**

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

## 5. `code/analysis/` — grouped by theme (58 files; **now the actual folder layout, see §0**)

### 5a. Core shared utilities (stayed at `code/analysis/` top level)
| File | Purpose |
|---|---|
| `discussion_talk_vs_behavior.py` | Defines `load_messages()` — the canonical discussion-transcript loader used everywhere discussion data is needed. Also: talk-vs-behavior correspondence, first-proposal/anchoring, cross-group heterogeneity |
| `sobel_mediation.py` | Sobel mediation test (DN→IN→contribution); also the source of the shared IN/DN/contribution panel-loading logic reused by 6 other scripts |

### 5b. Behavioral statistics (`behavioral/`)
| File | Purpose |
|---|---|
| `behavioral_statistical.py` | Formal stats on contribution: level/slope models, cross-model omnibus, dispersion (main 7B+GPT experiment) |
| `behavioral_statistical_13b.py` / `_70b.py` | Same 5 tests, 13B/70B model families |
| `behavior_quantified.py` | OLS w/ seed-clustered SEs, contribution+payoff, within- and cross-model tables |
| `paper_tables_behavioral.py` | Paper-ready tables from `quant_behavioral.py`'s CSVs |

### 5c. Perception / alignment statistics (`perception/` + `alignment/`)
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

### 5d. Network / selection (`selection/`)
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

### 5e. Discussion / discourse + LLM annotation (`discussion/`)
| File | Purpose |
|---|---|
| `discussion_mechanism_analysis.py` | The big one — local-anchoring→collective-ratification analysis (talk-behavior correspondence, first-proposal anchoring, cross-group convergence, conversation-function lexicon) |
| `discussion_phrase_context.py` | Repeated-phrase threshold sweep + context tagging (round/speaker-position) |
| `repeated_phrases.py` | N-gram (2-5) frequency finder across discussion transcripts |
| `sample_conversations_for_review.py` | Samples whole-run transcripts for qualitative review |
| `human_annotation_stratified_sample.py` | Draws the 80-discussion cross-family stratified sample for hand-coding (had the `r.round` bug, now fixed) |
| `gpt_5_annotation_sl.py` | The LLM-as-judge annotation pipeline built this session (codebook-driven, shot-condition testing, per-group batched annotation, external shot/eval data support) |
| `gpt5_annotation_pilot/` (6 scripts + codebook + hand-coded CSVs) | Everything built this session on top of `gpt_5_annotation_sl.py`: `select_shot_discussions.py`, `estimate_full_corpus_cost.py`, `build_full_corpus_eval_data.py`, `analyze_full_corpus_patterns.py`, `build_pattern_tables.py`, `build_family_trend_table.py` — this subfolder was the original template for the whole `discussion/` reorg |

### 5f. MCPR mixed-effects models (`mcpr/`)
| File | Purpose |
|---|---|
| `mcpr_lmm_analysis.py` | Mixed-effects models of MCPR effects on contribution (7B only) |
| `mcpr_lmm_alignment_network.py` | Same, for alignment (IN/DN gap) and network (spearman/gini/top_share) outcomes |

### 5g. Shock / intervention experiments (`shock/`)
| File | Purpose |
|---|---|
| `shock_analysis.py` | DiD: adversarial-agent introduction effect on other agents, round-10 and round-20 injections |
| `its_shock_round20.py` | Interrupted-time-series version of the round-20 shock (DiD isn't identified there — no post-shock control) |
| `new_intro_analysis.py` (in `code/analysis/`) | New-agent-introduction behavior/perception/alignment trajectories |
| `new_intro_adversarial_plots.py` | Cross-model trajectory plots for the adversarial new-intro experiments |
| `plot_dissociation_belief_vs_contribution.py` | The figure we edited this session — belief vs. behavior recovery around a group-composition injection (regenerated from saved aggregate; original plotting script is lost) |

### 5h. Cross-model / cross-size orchestrators (`cross-model/`)
| File | Purpose |
|---|---|
| `compare_7b_13b.py` | 7B vs ~13B within-family comparison |
| `compare_global_local.py` | Global vs local info-structure comparison |
| `run_local_analysis.py` / `run_13b_analysis.py` / `run_70b_analysis.py` | Orchestrators that monkey-patch and re-run the shared `stability_analysis`/`temp_eval_alignment2`/`temp_eval_network`/`temp_evals_behavior` modules per model tier |
| `robustness_analysis.py` / `robustness_split.py` | Reproduces the standard figure set for each robustness experiment (N=16/20 agents, group size variants), and splits those by model size/family |
| `subsample_stability.py` | Are main conclusions robust to seed subsampling (5-8 of 10 seeds)? |
| `stability_analysis.py` | Cross-agent SD convergence rate (contribution/IN/DN) — also imported by the `run_*_analysis` orchestrators |
| `evaluation_multi.py` | Foundational multi-seed/multi-model metrics builder (round_metrics.csv etc.) — see §0 flag on its placement |

### 5i. Plotting / table utilities (`plotting/`)
| File | Purpose |
|---|---|
| `plot_prompt_boxes.py` | Renders the 5 LLM prompt templates as paper-style figure boxes |

### 5j. "temp_eval*/temp_evals_*" building blocks — misleadingly named, mixed status (now distributed by topic: `behavioral/`, `perception/`, `alignment/`, `selection/`, `discussion/`)
| File | Status |
|---|---|
| `temp_eval_alignment2.py`, `temp_eval_network.py`, `temp_evals_behavior.py` | **Imported** by all 3 `run_*_analysis.py` orchestrators — load-bearing despite the name |
| `temp_eval_perception.py` | **Imported** by `perception_consensus_avg7b.py` / `perception_consensus_combined_small_with_avg.py` — load-bearing |
| `temp_eval_alignment.py` (no "2") | Not imported anywhere found — appears to be an earlier/superseded standalone version of `temp_eval_alignment2.py`; candidate for archiving, but confirm first |
| `temp_evals_discussion.py`, `temp_evals_selection.py` | Not imported anywhere found — standalone scripts (discussion lexical/log-odds features; weight-based selection metrics); likely fine to rename out of the "temp" namespace rather than archive, since neither looks throwaway in content |

---

## 6. `code/robustness/` (10 files) — already a reasonably contained cluster, not touched in this reorg

| File | Purpose |
|---|---|
| `temperature_sweep.py` / `analyze_temperature_sweep.py` | Run + analyze: baseline stability across decision temperatures |
| `prompt_variants.py` / `analyze_prompt_variants.py` | Run + analyze: sensitivity to structural system-prompt framing |
| `prompt_sensitivity.py` / `analyze_prompt_sensitivity.py` | Run + analyze: word-level system-prompt sensitivity |
| `parameter_sweep.py` | OFAT sweep over `SimConfig` parameters (network update rate, tie-removal threshold, etc.) |
| `analyze_cross_model.py` | Cross-model summary combining all three robustness tests above |
| `analyze_mechanism_cross_model.py` / `analyze_mechanism_vs_pretraining.py` | Is FULL-vs-baseline divergence mechanism-driven (builds over rounds) or a pre-training artifact (present at round 1)? |

Note the run/analyze pairing convention here (`prompt_variants.py` runs it, `analyze_prompt_variants.py` analyzes it) — a cleaner pattern than most of `code/analysis/` pre-reorg, possibly worth adopting elsewhere.

## 7. `code/stability test/` (8 files, space in dirname) — mixes engines + analysis, not touched in this reorg

| File | Purpose |
|---|---|
| `logit_SoNoLiFi_v5.py` | Engine variant capturing token-level logprob distributions (no docstring) |
| `logit_analysis.py` | Analyzes the above: point-estimate vs. logprob-distribution comparison, entropy trajectories |
| `new_group_SoNoLiSi_os_local.py` / `new_group_IN_SoNoLiSi_os_local.py` | Engine variants for group-composition-injection experiments (no docstring) |
| `new_intro_SoNoLiFi_v5.py` / `new_intro_SoNoLiSi_v5_local.py` / `new_intro_SoNoLiSi_v5_os_local.py` | Engine variants for new-agent-introduction experiments, 3 backend/variant flavors (no docstring) |
| `new_intro_analysis.py` | Analyzes the new-intro experiment: extension-round trajectories, pre/post comparison |

Recommend: split engines into `code/experiments/`, analysis into `code/analysis/` (probably `shock/`, alongside the other new-intro files), and rename the directory (or retire it once its outputs are folded into the main experiment/analysis split) — the space in the path alone makes it error-prone from the shell. **Not done in this pass** — this reorg covered `code/analysis/` only, per your original ask.

## 8. `archive/` (9 files) — already quarantined, low priority, not touched

Old simulation-engine versions (`SoNoLiSi_v4.py`, `SoNoLiSi_v5_fixed*.py`, `SoNoLiSi_collab_write.py` — a Google-Docs collaborative-writing variant, looks like an abandoned direction) and old analysis scripts (`CT.py`, `appendix_evaluations.py`, `causal_checks.py`, `coop_vs_viol.py`, `diffbackbone.py`). Already out of the way; also now excluded from git (see `.gitignore`) pending a data-release process.

---

## Next steps not yet done

1. Confirm/archive the 3 orphaned `temp_eval*` files (§5j) — waiting on your call.
2. Split `code/stability test/` into engines (→ `code/experiments/`) vs. analysis (→ `code/analysis/shock/`), fix its imports, rename the directory.
3. Consider consolidating the duplicate-variant clusters from §1 into single parameterized scripts.
