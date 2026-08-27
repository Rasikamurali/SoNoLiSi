# Main-paper results: no-expectation (SoNoLiSi_..._noexpect) arm

Tracking note only — nothing in `code/analysis/` was reorganized to produce
this. Purpose: when `code/analysis/` next gets cleaned up / reorganized
(see `CODEBASE_INVENTORY.md` §0 for the last such pass), the three items
below should be recognized as **main paper results for the no-expectation
arm** and treated with the same care as the corresponding with-expectation
scripts, not swept into `archive/` as one-off/temp scripts.

Data: `code/results/{llama,mistral,qwen}/local_noexpect/seed{42..51}/log_*.json`.
No GPT noexpect run exists. Conditions actually present:
`FULL_NO_EXPECT`, `DISCUSSION_ONLY_NO_EXPECT`, `SELECTION_ONLY_NO_EXPECT`
(config source: `code/experiments/SoNoLiSi_v5_local_noexpect.py::make_config()`).

## Condition-name mapping (noexpect -> main-paper analogue)

| noexpect condition | discussion_on | selection_on | perception_on | main-paper analogue |
|---|---|---|---|---|
| `FULL_NO_EXPECT` | True | True | False | `FULL` |
| `DISCUSSION_ONLY_NO_EXPECT` | True | False | False | `NO_SELECTION` |
| `SELECTION_ONLY_NO_EXPECT` | False | True | False | `NO_DISCUSSION` |

No `BASELINE_NO_EXPECT` exists and none is needed: `BASELINE` already has
`discussion_on=False`, and perception/expectation formation is mechanically
gated by discussion being on, so plain `BASELINE` already *is* the no-expect
baseline (nothing for `perception_on` to toggle without discussion).

The script comments in `SoNoLiSi_v5_local_noexpect.py` (lines 9-10) label
`DISCUSSION_ONLY_NO_EXPECT` as "social learning only" and
`SELECTION_ONLY_NO_EXPECT` as "selection only" — i.e. **"no social learning"
= `SELECTION_ONLY_NO_EXPECT`** (discussion off) and **"no selection" =
`DISCUSSION_ONLY_NO_EXPECT`** (selection off). This is the mapping used
below and in each script's docstring.

## 1. Within-model condition contrasts for contribution (OLS + Wald, Bonferroni)

**Script:** `code/analysis/behavioral/behavior_quantified_noexpect.py`
**Mirrors:** `behavioral/behavior_quantified.py`'s Q1 methodology exactly
(same formula shape, same Wald-contrast machinery via `wald_contrast()`,
same Bonferroni correction), scoped to all C(4,2)=6 condition pairs (no
directional subset — "compare each pair").
**Models:** all 4 (gpt, llama, mistral, qwen). GPT-4o-mini noexpect data
(added 2026-08-15) lives under `code/results/gpt-4o-mini/local_noexpect/` —
directory name `gpt-4o-mini`, not `gpt` like the with-expect dataset — and
uses seed range 43-52 rather than 42-51 (same off-by-one as the with-expect
local-variant `results/gpt/local/`). See `MODEL_DIR`/`PB_MODEL_DIR`/
`MODEL_SEEDS` in the script for the per-model mapping.
**Conditions:** `PURE_BASELINE` (pulled from the main with-expect dataset,
`results/{model}/local/` — there's no noexpect-arm PURE_BASELINE run because
none is needed: PURE_BASELINE already has `perception_on=False`, so it's
mechanically identical to what a "PURE_BASELINE_NO_EXPECT" would be; seeds
matched per-model to the noexpect seed range above) plus the 3 noexpect
conditions.
**Output:** `figures/2026-03-22/paper_stats/noexpect/`
(`q1_all_rounds_contribution.csv`, `q1_final5r_contribution.csv`, + `.tex`)
**Metric:** contribution only (task scope; payoff not run).

## 2. Feedback-chain per-link test (contribution -> evaluation -> network update -> selection consequences -> contribution adjustment/exclusion), OLS clustered by run

**Script:** `code/analysis/selection/social_selection_feedback_analysis_noexpect.py`
**Condition:** `SELECTION_ONLY_NO_EXPECT` ("no social learning" — discussion
off, selection on; the condition where the chain's mechanics read cleanest).
**Design note:** this is a *per-link* design (one OLS per link, SEs
clustered by run = family x seed), not the pooled selection_on x
discussion_on interaction design of the original
`selection/social_selection_feedback_analysis.py` — that design doesn't
apply here because each noexpect arm only has one condition with
evaluation_on=True, so there's no within-arm interaction to estimate.
**For direct comparability**, the script re-fits the identical per-link
formulas on the matched with-expectation condition (`NO_DISCUSSION`, same 3
models, same seed range 42-51) in the same run — so the paper comparison
should use this script's `links_expect_matched.csv` output, **not** the
numbers in `exports/social_selection_feedback/` (different model shape,
not cell-comparable).
**Output:** `code/analysis/exports/social_selection_feedback_noexpect/`
(`links_noexpect.csv`, `links_expect_matched.csv`,
`feedback_chain_comparison.tex`, `feedback_chain_comparison_summary.md`)

## 3. Discussion read-through vs. NO_SELECTION (preliminary, qualitative)

**Condition:** `DISCUSSION_ONLY_NO_EXPECT` ("no selection" — discussion on,
selection off).
**Status:** explicitly lighter-weight than items 1-2 per the task as given
("for now just go through the discussion and compare with no selection
condition") — a manual transcript read, not a re-run of
`discussion/discussion_mechanism_analysis.py`'s regex/statistical pipeline.
**Output:** `code/analysis/exports/discussion_mechanism_noexpect/discussion_noexpect_readthrough.md`
**Headline (qualitative, not yet quantified):** the proposal -> uptake/
agreement -> reference-to-prior-round sequence structure looks intact
without expectations; explicit expectation/norm language ("expecting similar
contributions from others", "what others expect from me") is absent from
every no-expect sample read, consistent with `perception_on=False` removing
that channel. If this needs to become a real test later (comparable to
item 1-2's rigor), the natural next step is pointing
`discussion_mechanism_analysis.py`'s machinery at the noexpect corpus and
diffing against `exports/discussion_mechanism/*.csv`, the same way item 2
did for the mechanism chain — not yet done.
