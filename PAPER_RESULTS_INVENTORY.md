# Paper Results Inventory

Tracking doc for which code/data/output belongs to which paper result —
main paper vs. supplementary vs. dead. Built 2026-08-18 in a session that
went through `code/analysis/` and `figures/` result-by-result against the
list of main and supplementary results as specified by the author. See
`CODEBASE_INVENTORY.md` for the general per-file catalog of the codebase,
and `code/analysis/MAIN_PAPER_RESULTS.md` / `MAIN_PAPER_RESULTS_NOEXPECT.md`
for a deeper dive specifically on the alignment result (item 4 below).

Scope note: all main-paper results below are GPT-4o-mini / Llama-7B /
Mistral-7B / Qwen-7B, local variant, the original 10 seeds (42-51), unless
stated otherwise.

Decision on physical file moves (2026-08-18): **not** reorganizing
`code/analysis/`'s theme folders around main/supplementary — that would
fight the 2026-08-11 reorg (see `CODEBASE_INVENTORY.md` §0) and risks
breaking the `sys.path`-anchoring pattern ~19 scripts rely on. This doc is
the main/supplementary source of truth instead; only genuinely dead files
get physically moved, to the existing `archive/` (`analysis_code/`,
`experiment_code/`, `figures_plots_tables/`, `results_archive/`) — not a
new "supplementary" archive bucket.

---

## Main paper

### 1. Contribution trajectories (1×4)
- Code: `code/analysis/cross-model/run_local_analysis.py` (`plot_contribution_all_conditions()`)
- Data: `results/{gpt,llama,mistral,qwen}/local/` — 10 seeds × 5 conditions
- Output: `figures/local/cross_model/all_conditions_local.pdf/.png`

### 2. Pairwise OLS + Wald (main 5 conditions + noexpect)
- Code: `code/analysis/behavioral/behavior_quantified.py`, `code/analysis/behavioral/behavior_quantified_noexpect.py`
- Data: main arm `results/{model}/local/`; noexpect `code/results/{llama,mistral,qwen}/local_noexpect/` + `code/results/gpt-4o-mini/local_noexpect/`
- Output: `figures/2026-03-22/paper_stats/` + `figures/2026-03-22/paper_stats/noexpect/`

### 3. DN/IN convergence (mean cross-agent SD)
- Code: `code/analysis/cross-model/stability_analysis.py`
- Data: same as #1
- Output: `figures/local/cross_model/stability_analysis.png/.pdf`

### 4. Gap-based alignment
- Code: `code/analysis/alignment/gap_based_alignment_test.py` (canonical per `MAIN_PAPER_RESULTS.md`)
- Data: same as #1
- Output: `code/analysis/exports/alignment_by_gap/`

### 5. EE/NE-contribution evolution (perception_action_gap_across_models)
- Code: `code/analysis/alignment/temp_eval_alignment2.py`, invoked via `run_local_analysis.py`
- Output: `figures/local/cross_model/perception_action_gap_across_models.pdf/.png`

### 6. Social selection (per-link OLS)
- Code: `code/analysis/selection/social_selection_feedback_analysis.py`
- Output: `code/analysis/exports/social_selection_feedback/`

### 7. Social learning (GPT-5 annotations)
- Code: `code/analysis/discussion/gpt_5_annotation_sl.py` + `code/analysis/discussion/gpt5_annotation_pilot/social_learning_annotation_codebook.json`
- Ground truth: `code/analysis/exports/human_annotation_stratified_sample/`
- Canonical output (per-turn, row-shot, corrected codebook — see `build_family_trend_table.py`'s 2026-08-17 repoint comment):
  `code/analysis/exports/gpt5_annotation_pilot/full_corpus_annotated/8_example_rows_annotated.csv`,
  plus downstream `analyze_full_corpus_patterns.py`, `build_pattern_tables.py`, `build_family_trend_table.py`

### 8. Shock treatment (2×2: round 10/20 × adversarial-gap/untouched-level, FULL condition)
- Code: `code/analysis/shock/plot_dissociation_belief_vs_contribution.py` (figure),
  `build_dissociation_round10.py`, `build_untouched_agents_levels.py` (data builders).
  This is the "new_group" (multi-agent replacement/hijack) design — the
  project's only shock methodology going forward as of 2026-08-18 (see
  Archive section below).
- Data: `code/results/{llama,mistral,qwen}/local_newgroup/replace{10,20}_random_adversarial/` + `results/{model}/local/` (control)
- Output: `figures/2026-03-22/paper_stats/dissociation_belief_vs_contribution.png/.pdf`
- **Note (2026-08-18):** `MAIN_RESULTS/8_shock_treatment/` still contains
  `shock_magnitude_round{10,20}.tex`, `shock_recovery_round{10,20}.tex`, and
  `its_shock_round20.tex` — these were produced by `shock_analysis.py`/
  `its_shock_round20.py`, now archived (they used the *other*,
  single-new-agent shock design — see Archive). Left in place since nothing
  gets deleted without being asked, but flagged: these 5 files' source
  scripts no longer exist in `code/analysis/`; ask before treating them as
  current.

### Cross-cutting shared infra
`code/analysis/model_specs.py` (shared MODEL_SPECS/CONDITIONS registry + panel-loading helpers),
`code/analysis/discussion_talk_vs_behavior.py` (canonical discussion-transcript loader) —
stay at `code/analysis/` top level, imported across most themes above.

---

## Supplementary

### S1. Prompt boxes
- Code: `code/analysis/plotting/plot_prompt_boxes.py`
- Output: `figures/appendix and tables/prompt_boxes.pdf/.tex`
- Status: done.

### S2. GPT-5-mini / Llama-70B / Mistral-13B / Qwen-72B replication
Same main-paper results, minus the conversational/social-learning annotation
piece (item 7). Built 2026-08-18.

- Data (complete, 10 seeds × 5 conditions each):
  `code/results/gpt-5-mini/local/`, `code/results/llama_70b/local/`,
  `results/mistral_13b/local/`, `code/results/qwen_72b/local/`.
  `gpt-5-mini` added as a 10th entry to `model_specs.py`'s `MODEL_SPECS`
  registry. Mistral-13B's data lives under the top-level `results/` folder
  (unlike the other three, under `code/results/`); rather than generalizing
  the single-RESULTS-root assumption baked into 6+ shared analysis modules,
  `code/results/mistral_13b/local/seed{42..51}` are symlinks to the real
  seed directories under `results/mistral_13b/local/` (data itself
  untouched, not duplicated) so every script's existing single-root
  patching still works unmodified.
- Code:
  - Items 1 (contribution trajectories) + 3 (DN/IN convergence):
    `code/analysis/cross-model/run_s2_analysis.py` — new orchestrator,
    modeled on `run_70b_analysis.py`'s monkey-patch pattern.
  - Item 2 (pairwise OLS + Wald): reuses `behavioral/behavior_quantified.py`
    directly via CLI (`--models gpt-5-mini llama_70b mistral_13b qwen_72b`),
    invoked by the orchestrator above. Required two small fixes for any
    model/family outside the original 4-9: `--models` argparse `choices`
    didn't include `gpt-5-mini`, and the `MODEL_TEX` LaTeX-label dict was
    missing an entry for it (the reference-model logic itself was already
    dynamic, no fix needed there).
  - Item 4 (gap-based alignment): reuses `alignment/gap_based_alignment_test.py`
    directly via its existing `--families` CLI flag — no new code needed
    beyond the `model_specs.py` registry entry. Fixed a real bug found
    while running this subset: the family fixed-effect's reference level
    was hardcoded to `"GPT"` in 4 formula strings + 2 print statements,
    which crashed (`patsy.PatsyError: specified level 'GPT' not found`) for
    any `--families` subset that excludes GPT. Replaced with a `FAMILY_REF`
    global, set dynamically in `main()` (prefers `"GPT"` when present,
    else the alphabetically-first family in the subset) — general fix,
    not S2-specific, benefits any future family-subset run.
  - Item 6 / main item 5 (social selection per-link OLS): new wrapper
    `code/analysis/selection/selection_mechanism_gpt5mini_llama70b_mistral13b_qwen72b.py`,
    modeled directly on the existing `selection_mechanism_llama_mistral_qwen.py`
    template (monkey-patches `model_specs.MODEL_SPECS` + `REF_FAMILY` before
    importing `social_selection_feedback_analysis.py`).
  - Item 6/main item 8 (shock, "10 vs 20, random replacement, FULL
    condition"): **blocked on a data gap, not code.** That methodology
    reads `local_newgroup/replace{10,20}_random_adversarial/`, which only
    exists for the 7B trio (llama/mistral/qwen). Llama-70B/Mistral-13B/
    Qwen-72B only have `local_newintro/intro{10,20}_adversarial/` — the
    *other* shock experiment (new-agent-introduction DiD/ITS via
    `shock_analysis.py`/`its_shock_round20.py`), not the replacement one.
    Real simulation runs are needed to close this, or substitute the
    new-intro version, or skip — pending author decision.
- Raw pipeline output (written by the scripts above, same convention as
  every other orchestrator — e.g. `run_70b_analysis.py` writes to
  `figures/local/70b_models/`):
  - `figures/local/s2_models/cross_model/all_conditions_s2.pdf/.png` (item 1)
  - `figures/local/s2_models/cross_model/stability_analysis.pdf/.png` (item 3)
  - `figures/local/s2_models/paper_stats/` (item 2, Q1/Q2 OLS+Wald tables)
  - `code/analysis/exports/alignment_by_gap/subset_gpt5mini_llama70b_mistral13b_qwen72b/` (item 4)
  - `code/analysis/exports/social_selection_feedback_s2/` (item 6/main-5)
- Curated copy (paper-ready subset, same organization as `MAIN_RESULTS/`):
  `figures/SUPPLEMENTARY_RESULTS/2_bigger_models/{1_contribution_trajectories,
  2_pairwise_ols_wald,3_dn_in_convergence,4_gap_based_alignment,
  6_social_selection_per_link}/`

### S3. Contribution trajectories, backup/extended-seed version
- Code: `code/analysis/behavioral/contribution_trajectories_all_runs.py`
  (1×3, llama/mistral/qwen only, up to 50 seeds — 10 canonical + 40 additional)
- Output: `figures/2026-03-22/additional_runs_results/01_contribution_trajectories/`

### S4. Implementation robustness (temperature / prompt formulation / prompt sensitivity / learning parameter)
Original pipeline code (`code/robustness/{temperature_sweep,analyze_temperature_sweep,
prompt_variants,analyze_prompt_variants,prompt_sensitivity,analyze_prompt_sensitivity,
parameter_sweep,build_supplementary_tables,render_supplementary_tables}.py`) plus a new
supplementary-analysis layer built 2026-08-18, `analyze_implementation_robustness.py`,
which reframes each sweep around whether the *substantive experimental contrast* holds
(not just whether the implementation knob has a main effect).

- Design-inventory findings (empirical, not assumed — see the script's own module
  docstring for the full account):
  - Temperature (T=0.3/0.5/0.7/1.0, PURE_BASELINE/BASELINE) and prompt formulation
    (standard/minimal/tendency_free, PURE_BASELINE/FULL) match the originally assumed
    design exactly, all 4 backends (GPT-4o-mini, Llama, Mistral, Qwen) complete.
  - What was originally called "prompt paraphrase sensitivity" **is**
    `prompt_sensitivity.py` (4 word-level variants: standard/verb_only/anchor_only/
    verb_anchor) — confirmed by the author 2026-08-18, analyzed as its own Part III
    with a full distribution-summary (median/mean/min/max effect, sign-consistency %,
    CI-crosses-zero %, heterogeneity test).
  - What was originally called "learning parameter" **is** `parameter_sweep.py` —
    confirmed by the author 2026-08-18. Mechanism footnote kept for accuracy: the
    parameter actually on disk is a positive/negative network-tie-update-rate pair
    (`pos0.2_neg0.8` reference, `pos0.4_neg0.6`, `pos0.5_neg0.5`), FULL condition only
    (no control arm), 7B trio only (no GPT) — not the `norm_internalization_rate` OFAT
    design `parameter_sweep.py`'s own module docstring originally described (never run;
    see `analyze_parameter_sweep.py`'s docstring, cross-checked against
    `results/robustness/parameter_sweep/`).
- **Headline finding: prompt formulation shows a genuine sign reversal**, not just loss
  of significance. FULL − PURE_BASELINE: standard = +1.35 (SE 0.48, CI excludes 0);
  minimal = −0.77 (SE 0.91); tendency_free = −0.57 (SE 0.63) — both structural
  alternatives flip sign. Condition×PromptVariant interaction p=0.0015 (formally
  significant, not sampling noise). By contrast: temperature is clean and stable
  (ΔE = 1.93/1.80/1.86/1.85 across T, no interaction); prompt *sensitivity* (word-level)
  stays directionally robust (4/4 variants positive, no interaction, p=0.63); learning
  parameter shows no strong effect either way (both non-reference pairs ≈0, wide CIs).
- Code: `figures/SUPPLEMENTARY_RESULTS/4_robustness_sweeps/outputs/analyze_implementation_robustness.py`
- Output (raw pipeline, same location for both script generations):
  `figures/robustness/{temperature_sweep,prompt_variants,prompt_sensitivity,parameter_sweep}/`
  + `figures/appendix and tables/table_{temperature_robustness,prompt_variants,prompt_sensitivity,parameter_sweep}.tex`
- Curated copy (paper-ready, all in `figures/SUPPLEMENTARY_RESULTS/4_robustness_sweeps/`):
  - `outputs/` — the new analysis: `implementation_robustness_design_inventory.csv`,
    `implementation_robustness_qc_report.txt`, `temperature_{run_level,summary,contrasts,variability}.csv`,
    `prompt_{run_level,summary,contrasts,model_specific}.csv`,
    `prompt_sensitivity_{run_level,summary_by_backend,contrasts,summary}.csv`,
    `learning_parameter_{run_level,summary,contrasts,mechanism}.csv`,
    `figure_R1_implementation_effects.pdf/.png` (4-panel effect-size summary),
    `figure_R2_implementation_raw_means.pdf/.png`, `figure_R3_learning_mechanism.pdf/.png`,
    `implementation_robustness_{contrasts,interactions}.csv/.tex`,
    `implementation_robustness_statistical_results.txt`, `implementation_robustness_interpretation.md`
  - `{temperature_sweep,prompt_variants,prompt_sensitivity,parameter_sweep}/` — copies of
    the original per-sweep figures/CSVs
  - `appendix_tables/` — the S1-S4 LaTeX tables + per-model CSVs from `build_supplementary_tables.py`
- Status: presentation of the *original* per-sweep tables still TBD (author reviewing);
  the new contrast-focused analysis above is complete.

### S5. Community size / group size / MCPR
- Code: `code/experiments/SoNoLiSi_v5_os_local_groupsizevary.py`
- Data: N16/N20 under `code/results/{model}/local/N{16,20}_G4/`;
  G3/G6/G8 (MCPR-constant design) + MCPR0.5/0.8 under
  `code/results/{llama,mistral,qwen}/local_groupsizevary/` — 7B trio only,
  not yet extended to 13B/70B/mini tiers
- Output: `figures/local/{16agents,20agents,groupsize,mcpr}/`
- Note: actual swept values are N=12(main)/16/20 and G=3/4(main)/6/8 —
  no N=8, G=2, G=5, or G=10 exist anywhere in the repo (checked 2026-08-18),
  despite that being the author's initial recollection.
- Curated copy (paper-ready, all in `figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/`):
  - `16agents/`, `20agents/`, `groupsize/{G3,G6,G8}/`, `mcpr/{MCPR0.5,MCPR0.8,stats}/` —
    copies of the original per-sweep figures/CSVs (as produced by the existing
    per-tier/per-model orchestrators).
  - `mcpr/mcpr_robustness/` — new MPCR-focused supplementary analysis built 2026-08-18,
    `analyze_mpcr_robustness.py`. Tests whether condition rankings/contrasts hold across
    MPCR=0.4/0.5/0.8 (N=12, G=4). **Data-coverage caveat**: GPT has MPCR=0.4 only (no
    0.5/0.8 anywhere); Llama's MPCR=0.8 directory exists but is empty — only
    Mistral/Qwen have all three MPCR levels, so the "pooled" panel is model-balanced
    among whichever models are present at each MPCR (4→3→2), annotated with `n=` per
    point. No significant Condition×MPCR interaction found (pooled p=0.82); condition
    rankings look stable in the data that exists. Outputs: `mpcr_run_level_means.csv`,
    `mpcr_summary.csv`, `mpcr_condition_contrasts.csv`, `mpcr_robustness.pdf/.png`,
    `mpcr_summary_table.tex`, `mpcr_condition_contrasts.tex`,
    `mpcr_statistical_results.txt`, `mpcr_interpretation.md`.
  - `community_group_robusntess/` — new structural-robustness supplementary analysis
    built 2026-08-18, `analyze_structural_robustness.py`. Covers community size
    (N=12/16/20, all 8 model variants + 7B-trio-only pooling) and group size
    (N=12: G=3/4/6; N=16: G=4/8, 7B trio only) across three layers: behavior (condition
    contrasts), convergence (late-round contrib/IN/DN SD, rounds 16-20), and mechanism
    (community: social-selection "undercontribution→evaluation" coefficient via
    `social_selection_feedback_analysis.py`'s Step 5, monkey-patched per N; group size:
    social-learning InGap/DnGap→shift coefficient via `gap_based_alignment_test.py`'s
    primary specification, self-contained loader since that script's own panel doesn't
    cover these variant runs). Two real path bugs found and fixed during this build
    (N16/N20 community dirs always live under `code/results/`, not each model's own
    default root; `social_selection_feedback_analysis.py` hardcodes `N_AGENTS=12`,
    needed monkey-patching per N) — see that script's own module docstring. Headline:
    behavior and both mechanisms hold robustly across every structural setting tested
    (no significant Condition×N or Condition×G interaction anywhere); convergence is
    flat across community size but genuinely tightens with larger group size.
    Outputs: `structural_qc_report.txt`, `{community,group_size}_run_level_means.csv`,
    `{community,group_size}_convergence_run_level.csv`, `{community,group_size}_summary.csv`,
    `structural_contrasts.csv/.tex`, `{community,group_size}_mechanism_coefficients.csv`,
    `figure_C1_structural_contributions.pdf/.png`, `figure_C2_structural_convergence.pdf/.png`,
    `figure_C3_mechanism_robustness.pdf/.png`, `community_model_tier_validation.pdf/.png`,
    `structural_summary_table.tex`, `structural_statistical_results.txt`,
    `structural_interpretation.md`.

### S6. Annotation pilot (kept whole, pending author review)
- All of `code/analysis/exports/gpt5_annotation_pilot/` — canonical run
  (`full_corpus_annotated/`) plus every iteration subfolder
  (`full_corpus_run/`, `haewoon_*`, `stratified_sample_run*`,
  `zero_shot_annotated.csv`, `codebook_examples_annotated.csv`, etc.).
  Not sorted further until the author has looked at it more closely.

### S7. Shock variants (kept, marked backup)
- Code: `code/analysis/shock/build_condition_variants.py`,
  `build_top_replacement_variant.py`, `plot_dissociation_condition_variants.py`,
  `plot_dissociation_top_replacement.py`
- Output: `dissociation_belief_vs_contribution_{baseline,no_selection,
  no_social_learning,top_replacement}.png/.pdf` in `figures/2026-03-22/paper_stats/`

---

## Archive (agreed 2026-08-18)

**Stale alignment scripts** → `archive/analysis_code/`:
- `alignment_quantified.py`, `temp_eval_alignment.py`
(already flagged stale in `MAIN_PAPER_RESULTS.md` — no live output, only
existed under the frozen `figures/final_quals_results/`-style snapshot)

**Perception/alignment consolidation (2026-08-24)** → `code/analysis/archive/{perception,alignment,cross-model}/`:
`temp_eval_perception.py`, `perception_consensus_avg7b.py`,
`perception_consensus_combined_small_with_avg.py`, `perception_consensus_combined_s2.py`,
`perception_quantified.py`, `perception_dn_in.py`, `in_dn_predictive.py`,
`timestep_did.py`, `expectation_gap_evolution_plots.py`,
`mediation_llama_mistral_qwen.py`, `gap_based_alignment_test.py`,
`lagged_ar_regression.py`, `temp_eval_alignment2.py`, `stability_analysis.py`
— superseded by 4 new tier-parametrized files
(`perception/perception_consensus.py`, `alignment/perception_action_gap_plot.py`,
`alignment/gap_based_alignment.py`, `alignment/lagged_alignment_check.py`); see
`code/analysis/MAIN_PAPER_RESULTS.md` for the full mapping and the redefined
alignment headline (7B-only, not pooled-all-families).

**Single-new-agent / adversarial-agent-introduction experiment** — author
decided 2026-08-18 the project is only pursuing the "new_group" (multi-agent
replacement/hijack) shock design going forward (see main result #8); this
whole parallel experiment is retired:
- Code → `archive/analysis_code/`: `shock_analysis.py`, `its_shock_round20.py`,
  `new_intro_adversarial_plots.py`, `new_intro_analysis.py` (the
  `code/analysis/shock/` copy — analyzes the non-adversarial single-new-agent
  variant)
- Code → `archive/experiment_code/`: `new_intro_analysis.py` (the
  `code/stability test/` copy — a *different* script despite the same name,
  analyzes `new_intro_SoNoLiFi_v5`; kept in a separate archive subfolder to
  avoid a filename collision with the one above), `new_intro_SoNoLiFi_v5.py`,
  `new_intro_SoNoLiSi_v5_local.py`, `new_intro_SoNoLiSi_v5_os_local.py`
  (the 3 simulation-engine variants)
- Data → `archive/results_archive/local_newintro/{model}/` (2.4GB, all 8
  models that had it: llama/mistral/qwen 7B, 13B/14B, and llama_70b/qwen_72b)
- Figures → `archive/figures_plots_tables/new_intro/`

All moves were plain `mv` (data preserved, not deleted) — see the note
under main result #8 for one loose end this created (5 orphaned `.tex`
files still sitting in `MAIN_RESULTS/8_shock_treatment/`).

---

## Open items

1. Decide whether to prune the 5 orphaned `.tex` files (now-archived
   `shock_analysis.py`/`its_shock_round20.py` output) out of
   `MAIN_RESULTS/8_shock_treatment/`.
2. S2's shock item (main result #8 for Llama-70B/Mistral-13B/Qwen-72B) has
   no path forward now that the single-new-agent design is retired — the
   "new_group" replacement design has no data for these 3 tiers either.
   New `local_newgroup` simulation runs are the only way to close this.
3. Decide S4's final presentation (which robustness result(s) to show).
4. Review S6 (`gpt5_annotation_pilot/`) and sort canonical vs. archive-candidate
   iteration folders once the author has looked more closely.
