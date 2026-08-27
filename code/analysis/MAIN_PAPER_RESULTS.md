# Main-paper results (with-expectation arm)

Tracking doc for which script is the canonical/primary analysis for a given
paper theme, in a codebase where multiple scripts often test closely related
questions (see `CODEBASE_INVENTORY.md` for the full per-file catalog; this
file exists so "which one is the real result" doesn't have to be re-derived
each time). Mirrors `MAIN_PAPER_RESULTS_NOEXPECT.md`'s role for the
no-expectation arm.

## Perception/alignment consolidation (2026-08-24)

The perception (`code/analysis/perception/`) and alignment
(`code/analysis/alignment/`) folders previously held ~15 overlapping
scripts. These were consolidated into exactly 4 canonical files, each
tier-parametrized (`--tier 7b|s2|13b|70b|pooled|community|group`) rather
than duplicated per tier:

- **(a) `perception/perception_consensus.py`** — cross-agent SD (consensus)
  of IN/DN, two panels: early-vs-late OLS (Holm-corrected, per family +
  pooled) and continuous per-seed slope + final-round level (+ mixed-effects
  robustness). Supersedes `temp_eval_perception.py`,
  `perception_consensus_avg7b.py`, `perception_consensus_combined_small_with_avg.py`,
  `perception_consensus_combined_s2.py`, `cross-model/stability_analysis.py`.
- **(b) `alignment/perception_action_gap_plot.py`** — descriptive IN/DN-gap
  trajectory plots. Supersedes `alignment/temp_eval_alignment2.py`.
- **(c) `alignment/gap_based_alignment.py`** — primary alignment result (see
  below). Supersedes `alignment/gap_based_alignment_test.py`.
- **(d) `alignment/lagged_alignment_check.py`** — supporting lagged
  respecification of (c). Supersedes `alignment/lagged_ar_regression.py`.

Old files archived to `code/analysis/archive/{perception,alignment,cross-model}/`.
`perception_quantified.py`, `perception_dn_in.py`, `in_dn_predictive.py`,
`timestep_did.py`, `expectation_gap_evolution_plots.py`,
`mediation_llama_mistral_qwen.py` (IN~DN predictive-relationship questions,
not part of this consolidation) were also archived — dropped from all 4
`run_*_analysis.py` orchestrators' pipelines, no replacement.
`alignment/gap_convergence_did.py`, `alignment/mechanism_interaction.py` test
different questions (does the IN-gap close faster than the DN-gap; 3-way
Selection×Discussion×Perception interaction) and were left untouched.

## Alignment (IN/DN expectation-gap effect on contribution)

**Primary: `code/analysis/alignment/gap_based_alignment.py --tier 7b`**
(redefined 2026-08-24 from "pooled across all 9-10 families" to **7B-only**
— gpt, llama-7b, mistral-7b, qwen-7b). Tests whether the gap between an
agent's own elicited injunctive/descriptive expectation and its
contribution predicts the shift into next-round contribution:

    InGap_t = IN_t - Contribution_t
    DnGap_t = DN_t - Contribution_t
    Shift_t = Contribution_{t+1} - Contribution_t
    Shift_t = b_IN*InGap_t + b_DN*DnGap_t + ConditionFE + FamilyFE + RoundFE

**Headline (N=33,485, 160 run_id clusters, SEs clustered by run):**
b_IN=0.182 (SE 0.020, p<.001), b_DN=0.194 (SE 0.026, p<.001). Wald test
b_IN=b_DN: stat=0.14, p=.705 — **cannot reject equality; do not claim IN
pulls behavior more strongly than DN for the 7B-only headline** (this
reverses the old pooled-all-families headline's claim, where the same test
was significant, p=.038 — a direct consequence of narrowing from 9-10
families to 4, not a change in any family's own estimate). Robust to a
lagged-level respecification (no differencing) and a within-agent
respecification (absorbs condition/family FE, isolates each agent's own
round-to-round variation) — both retain positive, significant IN/DN
coefficients (lagged-level: theta_IN=0.129, theta_DN=0.142, both p<.001;
within-agent: InGap=0.424, DnGap=0.249, both p<.001).

Output: `code/analysis/exports/alignment_by_gap/tier_7b/`, published to
`figures/MAIN_RESULTS/4_gap_based_alignment/`.

**Per-family breakdown** (`gap_based_alignment_by_family.tex`): refit the
primary formula separately per model family, plus a pooled InGap/DnGap ×
family interaction model for a joint heterogeneity test. Direction is
nearly universal — InGap is positive and significant in 3/4 families
(flat/null only for GPT: b=0.008, p=.54, consistent with its
separately-flagged elevated IN-null rate), DnGap is positive and
significant in all 4/4 — but **magnitude is not uniform**: point estimates
range from 0.16 to 0.66 for InGap (excl. GPT) and 0.10 to 0.33 for DnGap,
and the joint Wald test rejects slope-equality for both (InGap×family:
p=3.3e-57; DnGap×family: p=3.3e-10). So: **directionally universal, not
magnitude-universal** among the 7B tier too.

**Cross-checks, not primary:**
- `gap_based_alignment.py --tier pooled` — the OLD primary (all 9-10
  families pooled): b_IN=0.210 (SE 0.013), b_DN=0.169 (SE 0.014), Wald
  p=.038 (significant — IN > DN at this pooled scope). Kept as a
  cross-check now that 7b is primary, not deleted.
  Output: `figures/SUPPORTING_MAIN_RESULTS/4_alignment_cross_checks/subset_all_families/`.
- `gap_based_alignment.py --tier s2` — GPT-5-mini/Llama-70B/Mistral-13B/Qwen-72B
  replication. Output: `figures/SUPPLEMENTARY_RESULTS/2_bigger_models/4_gap_based_alignment/`.
- `gap_based_alignment.py --tier community` / `--tier group` — population-size
  (N∈{12,16,20}) and interaction-group-size (G∈{3,4,6,8}) structural
  robustness, new 2026-08-24. Output under
  `figures/SUPPLEMENTARY_RESULTS/5_community_group_mcpr/{community,group}/4_gap_based_alignment/`.
- `lagged_alignment_check.py` — closely related lagged-level spec on raw
  (non-gap) IN/DN. Corroborates `gap_based_alignment.py`'s own lagged-level
  robustness check — a useful independent cross-check, not a competing
  headline number. Output: `figures/SUPPORTING_MAIN_RESULTS/4_alignment_cross_checks/`.
- `perception_action_gap_plot.py` — descriptive trajectory plots (gap over
  rounds), not a statistical test. Output:
  `figures/MAIN_RESULTS/5_perception_action_gap_evolution/` (7b) and the
  SUPPLEMENTARY_RESULTS / community / group equivalents.
- `alignment/gap_convergence_did.py`, `alignment/mechanism_interaction.py` —
  live/current but test different questions (does the IN-gap close faster
  than the DN-gap over time; three-way Selection×Discussion×Perception
  interaction), not alternatives to the primary alignment result. Untouched
  by the 2026-08-24 consolidation.
- `alignment/alignment_quantified.py`, `alignment/temp_eval_alignment.py` —
  stale, archived (no live output; flagged 2026-08-17).

## Perceptual consensus (cross-agent SD of IN/DN)

**Primary: `code/analysis/perception/perception_consensus.py --tier 7b`**
(new as a designated primary, 2026-08-24 — this result previously existed
in two non-overlapping, never-reconciled forms; see below). Two panels,
both computed from cross-agent SD of IN/DN per round per seed:

- **Panel 1 (early-vs-late OLS):** SD ~ period (early = rounds 1-3, late =
  last 3 rounds), Holm-corrected across the 4 conditions, per family and
  pooled. Late-round SD is significantly lower than early-round SD in
  every condition for every 7B family (consensus strengthens over time).
- **Panel 2 (continuous slope + final level):** per-seed OLS slope of SD ~
  round, plus mean SD over the final 5 rounds, plus a mixed-effects
  (random intercept per seed) robustness check on the slope.

Output: `code/analysis/exports/perception_consensus/tier_7b/`, published to
`figures/MAIN_RESULTS/3_dn_in_convergence/`. Also available for `--tier
s2/13b/70b/pooled/community/group`.

**Note on provenance:** this consolidates two previously-separate,
never-reconciled pipelines that computed cross-agent SD differently —
`stability_analysis.py`'s per-seed continuous-slope approach (which was the
one actually promoted to `MAIN_RESULTS/3_dn_in_convergence/` before
2026-08-24, as `stability_final_sd_local.tex`) and
`temp_eval_perception.py`/`perception_consensus_avg7b.py`'s explicit
early-vs-late dummy OLS (which existed only in the `paper_stats/` scratch
folder and was never promoted). Both are now panels of the same table.
