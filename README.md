# SoNoLiSi — Social Norm Learning and Selection

Simulation code and analysis pipeline for studying how **social norms emerge, stabilise,
and are enforced among LLM agents** in a repeated public-goods game.

Groups of large-language-model agents repeatedly decide how much of a private endowment
to contribute to a shared fund. On top of that economic game we switch two social
mechanisms on and off — **social learning** (pre-decision group discussion) and
**social selection** (reputation-weighted group formation, up to and including exclusion)
— and elicit each agent's **normative** and **empirical expectations** after every round.
The design lets us ask which mechanism actually does the work of turning individual
behaviour into a shared, self-enforcing standard, and whether the answer is stable across
model families, model scales, community sizes, and adversarial shocks.

---

## The simulation

**Environment.** `N` agents (default 12) are partitioned each round into groups of
`GROUP_SIZE` (default 4). Each agent chooses a contribution `c ∈ [0, 10]` from a
10-token endowment; the group fund is multiplied by 1.6 and split equally among the
group. All interaction is **group-scoped** — agents observe and respond only to their
current groupmates, never the whole community. Runs last 20 rounds.

**Mechanisms.**

| Mechanism | What it adds |
|---|---|
| Social learning (**SL**) | A pre-decision discussion round among groupmates. |
| Social selection (**SS**) | Group formation follows a directed, weighted reputation network; sufficiently weak ties produce exclusion. Post-contribution peer evaluations update the tie weights. |
| Expectation elicitation (**E**) | Post-round reports of normative expectation (NE/IN — what agents think *should* be contributed), empirical expectation (EE/DN — what they think others *do* contribute), and partner preferences. These reports enter agent memory. |

**Conditions.** Five standard conditions cross the two mechanisms on top of elicitation:

```
FULL           = E + SL + SS
NO_DISCUSSION  = E + SS
NO_SELECTION   = E + SL
BASELINE       = E
PURE_BASELINE  = ∅
```

Three additional ablations isolate the mechanisms *without* expectation elicitation,
to test whether measurement itself drives the effect:
`FULL_NO_EXPECT` (SL+SS), `DISCUSSION_ONLY_NO_EXPECT` (SL), `SELECTION_ONLY_NO_EXPECT` (SS).

**Round order.** (1) group formation → (2) discussion → (3) contribution → (4) payoff →
(5) evaluation → (6) expectation/perception update → (7) memory update.

**Models.** Ten model families across three scale tiers, run through either the OpenAI
API or a local vLLM server:

| Tier | Families |
|---|---|
| Main (7B + API) | GPT-4o-mini, Llama-3.1-8B, Mistral-7B-v0.3, Qwen2.5-7B |
| 13B/14B | Llama-2-13B, Mistral-Nemo, Qwen2.5-14B |
| 70B+ / larger API | Llama-3.1-70B, Qwen2.5-72B, GPT-5-mini |

Ten random seeds per model × condition.

---

## Repository layout

```
code/
  experiments/                 Simulation engines
    SoNoLiSi_v5_local.py             main engine (OpenAI API + local vLLM)
    new_group_SoNoLiSi_os_local.py   adversarial-replacement shock variant
    new_group_IN_SoNoLiSi_os_local.py  shock variant w/ calibrated adversarial prior
  analysis/                    Statistical analyses, by theme
    model_specs.py                   shared model/seed/path registry + log loader
    discussion_talk_vs_behavior.py   canonical discussion-transcript loader
    behavioral/  perception/  alignment/  selection/  discussion/
    shock/  mcpr/  cross-model/  plotting/  supplementary/
  robustness/                  Temperature, prompt-variant, and parameter sweeps
  extract_agent_data.py, plot_beautifier.py
```


### Documentation in this repo

| File | What it is |
|---|---|
| [CODEBASE_INVENTORY.md](CODEBASE_INVENTORY.md) | Per-file catalog: what every `.py` does, what it reads, what it writes |
| [code/analysis/MAIN_PAPER_RESULTS.md](code/analysis/MAIN_PAPER_RESULTS.md) | Which script is *canonical* for each theme where several scripts test related questions, with the headline numbers. |

---

## Setup

```bash
git clone https://github.com/Rasikamurali/SoNoLiSi.git
cd SoNoLiSi
pip install -r requirement.txt
```

Running open-weight models additionally needs `vllm` (not pinned in `requirement.txt`,
since the right build depends on your CUDA version).

Configuration is entirely through environment variables — nothing is hard-coded:

| Variable | Needed for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | GPT-4o-mini / GPT-5-mini runs | Read from `code/.env` if present. |
| `HF_TOKEN` | Gated Hugging Face weights (Llama) | No default; unset means unauthenticated downloads. |
| `HF_MODEL_CACHE_DIR` | Local model weights cache | set this on any other machine. |

---

## Running simulations

```bash
cd code/experiments

# Main experiment: one API model, 5 standard conditions, 10 seeds
python SoNoLiSi_v5_local.py --model gpt-4o-mini

# Open-weight model on GPU 0
python SoNoLiSi_v5_local.py --model qwen --device 0

# Several models, one condition, explicit seeds
python SoNoLiSi_v5_local.py -m llama mistral qwen -c FULL -s 43 44 45

# No-expectation ablation arm
python SoNoLiSi_v5_local.py -m llama -c FULL_NO_EXPECT DISCUSSION_ONLY_NO_EXPECT SELECTION_ONLY_NO_EXPECT

# Structural robustness: larger community, or constant marginal per-capita return
python SoNoLiSi_v5_local.py -m mistral --n-agents 20 --group-size 4
python SoNoLiSi_v5_local.py -m mistral --group-size 6 --mcpr 0.4
```

Key flags: `--model/-m`, `--conditions/-c`, `--seeds/-s` or `--n`/`--start-seed`,
`--device/-d`, `--n-agents`, `--group-size`, `--mcpr`, `--out-subdir`.
Any `--model` value not in the local-model registry is treated as an OpenAI model name.

**Adversarial shock runs** use the `new_group_*` engines, which replace `GROUP_SIZE`
incumbents in place after a given round. The replacements inherit the full reputation
edges of whoever they displaced, so they start with the displaced agents' social capital:

```bash
python new_group_SoNoLiSi_os_local.py -m llama -c FULL \
    --replace-mode random --first-inject 10 --second-inject 20
```

### Output

Each run writes two JSON files per (model, condition, seed):

```
results/{model}/{variant}/[{N..._G...}/]seed{N}/
    log_v5os_local_{model}_{timestamp}_{CONDITION}_seed{N}.json     # round-by-round state
    conversations_v5os_local_{model}_{timestamp}_{CONDITION}_seed{N}.json   # discussion transcripts
```

`{variant}` is `local`, `local_noexpect`, or `local_groupsizevary` depending on the flags.
Every analysis script consumes these logs; nothing downstream re-runs the simulation.

---

## Running the analyses

Analysis scripts resolve their own location but read and write through paths relative to
the working directory, so **run them from `code/analysis/`**:

```bash
cd code/analysis
python behavioral/behavior_quantified.py
```

The main-paper analyses, numbered as in `PAPER_RESULTS_INVENTORY.md`:

| Result | Command |
|---|---|
| 1. Contribution trajectories | `python cross-model/run_local_analysis.py` |
| 2. Pairwise OLS + Wald tests | `python behavioral/behavior_quantified.py` (`behavior_quantified_noexpect.py` for the ablation arm) |
| 3. Perceptual consensus (cross-agent SD of IN/DN) | `python perception/perception_consensus.py --tier 7b` |
| 4. Gap-based alignment | `python alignment/gap_based_alignment.py --tier 7b` |
| 5. Perception–action gap evolution | `python alignment/perception_action_gap_plot.py --tier 7b` |
| 6. Social-selection feedback chain | `python selection/social_selection_feedback_analysis.py` |
| 7. Social-learning annotation | `python discussion/gpt_5_annotation_sl.py` (API cost — see below) |
| 8. Adversarial shock | `python shock/plot_dissociation_belief_vs_contribution.py` |

The tier-parametrised scripts take `--tier` (`7b`, `s2`, `13b`, `70b`, `community`, `group`,
and on some scripts `pooled` — run with `--help` for each one's exact set). That flag is how
the scale and structural-robustness replications are produced from the same code rather than
from per-tier copies.

Result 8 needs its two data builders run first: `shock/build_dissociation_round10.py` and
`shock/build_untouched_agents_levels.py`.
`cross-model/run_13b_analysis.py`, `run_70b_analysis.py`, and `run_s2_analysis.py` are the
per-tier orchestrators.

Result 7 (social-learning annotation) calls the OpenAI API to code discussion transcripts
against `discussion/gpt5_annotation_pilot/social_learning_annotation_codebook.json` — it
costs real money on a full corpus; see `estimate_full_corpus_cost.py` first.


```
