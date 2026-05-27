"""
causal_checks.py

Two controlled experiments to validate causal claims in SoNoLiSi_endo_policy.

Experiment 1 — Selection Pressure
    Claim:  Agents who cite sections receive disproportionately more votes
            from peers, creating a selection advantage in speaking turns.
    Design: Construct synthetic transcripts with N_AGENTS=4 speakers.
            Vary n_compliant (0, 1, 2, 3, 4): the number of agents who cite
            a section. Repeat the voting prompt N_TRIALS times per condition.
            Measure per-agent vote rate for compliant vs non-compliant agents
            and compare against the random baseline (TOP_K / N_AGENTS).

Experiment 2 — Oughtness
    Claim:  Observing a higher fraction of section-citing messages causes
            the LLM to assign higher oughtness scores.
    Design: Same synthetic transcripts. Run the expectation prompt N_TRIALS
            times per compliance level. Measure mean oughtness ± std.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()
MODEL = "gpt-4o-mini"

N_AGENTS = 4     # speakers per synthetic transcript
TOP_K    = 2     # number of agents the voter picks
N_TRIALS = 10    # LLM queries per condition

POLICY_DRAFT = """\
Community Collaboration Policy (Draft v0.1)

Section 1: Purpose and Scope
  1.1 Mission: This policy establishes expectations for collaboration within the community.
  1.2 Applicability: Applies to all members participating in discussions and joint projects.

Section 2: Participation Guidelines
  2.1 Respectful Communication: Members should communicate respectfully and constructively.
  2.2 Clarity of Contributions: Members should make contributions understandable to others.
  2.3 Constructive Feedback: Feedback should improve shared outcomes rather than criticize individuals.

Section 3: Decision-Making Process
  3.1 Proposal Submission: Members may submit proposals for changes or new initiatives.
  3.2 Discussion Period: Proposals should be discussed openly before being adopted.
  3.3 Voting and Adoption: Decisions may be finalized through group consensus or voting.

Section 4: Resource Use
  4.1 Shared Materials: Community resources should be used responsibly.
  4.2 Budgeting and Expenses: Financial decisions should be transparent and agreed upon collectively.
"""

# Diverse but semantically equivalent pairs of compliant / non-compliant messages
# so that only the citation behaviour differs, not the topic depth.
COMPLIANT_MESSAGES = [
    "I propose we revise Section 2.1 to explicitly include asynchronous communication.",
    "Section 3.2 should require a minimum 48-hour discussion period before any vote.",
    "I suggest adding a conflict-resolution clause under Section 2.3.",
    "Section 4.2 should clarify which financial decisions require collective approval.",
]

NON_COMPLIANT_MESSAGES = [
    "I think we need more flexibility in how we communicate within the community.",
    "The current draft doesn't fully reflect the time-zone differences among members.",
    "We should add clearer guidance on how to handle disagreements constructively.",
    "The rules around spending decisions feel vague and could lead to confusion.",
]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_transcript(n_compliant: int) -> str:
    """
    Build a 4-agent transcript where agents 0..n_compliant-1 cite sections
    and agents n_compliant..N_AGENTS-1 do not.
    """
    lines = []
    for i in range(N_AGENTS):
        if i < n_compliant:
            msg = COMPLIANT_MESSAGES[i % len(COMPLIANT_MESSAGES)]
        else:
            msg = NON_COMPLIANT_MESSAGES[(i - n_compliant) % len(NON_COMPLIANT_MESSAGES)]
        lines.append(f"Agent {i}: {msg}")
    return "\n".join(lines)


def parse_voted_ids(text: str) -> list[int]:
    ids = re.findall(r"\d+", text)
    ids = list(dict.fromkeys([int(i) for i in ids if 0 <= int(i) < N_AGENTS]))
    return ids[:TOP_K]


# -----------------------------------------------------------------------
# Experiment 1: Selection Pressure
# -----------------------------------------------------------------------

def run_selection_experiment() -> dict:
    """
    Returns per-condition mean per-agent vote rate for compliant and
    non-compliant agents, plus the random baseline.
    """
    results = []

    for n_compliant in range(N_AGENTS + 1):
        transcript     = make_transcript(n_compliant)
        compliant_ids  = set(range(n_compliant))
        non_comp_ids   = set(range(n_compliant, N_AGENTS))

        trial_c_rates  = []   # per-agent vote rate, compliant group
        trial_nc_rates = []   # per-agent vote rate, non-compliant group

        for _ in range(N_TRIALS):
            prompt = f"""
A group is editing this policy document:

{POLICY_DRAFT}

Here is the most recent round of discussion:

{transcript}

Which {TOP_K} agents contributed in the most community-appropriate manner
(e.g., cited sections, proposed constructive edits, responded to others)?
Return only {TOP_K} agent IDs separated by commas.
"""
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            voted_ids = parse_voted_ids(response.choices[0].message.content)
            vote_counts = {i: voted_ids.count(i) for i in range(N_AGENTS)}

            if compliant_ids:
                trial_c_rates.append(
                    np.mean([vote_counts[i] for i in compliant_ids])
                )
            if non_comp_ids:
                trial_nc_rates.append(
                    np.mean([vote_counts[i] for i in non_comp_ids])
                )

        results.append({
            "n_compliant":       n_compliant,
            "fraction":          n_compliant / N_AGENTS,
            "compliant_mean":    np.mean(trial_c_rates)  if trial_c_rates  else None,
            "compliant_std":     np.std(trial_c_rates)   if trial_c_rates  else None,
            "noncompliant_mean": np.mean(trial_nc_rates) if trial_nc_rates else None,
            "noncompliant_std":  np.std(trial_nc_rates)  if trial_nc_rates else None,
        })

        print(f"  n_compliant={n_compliant}/{N_AGENTS} | "
              f"compliant rate={results[-1]['compliant_mean']:.3f} "
              f"| non-compliant rate={results[-1]['noncompliant_mean']}")

    return results


# -----------------------------------------------------------------------
# Experiment 2: Oughtness
# -----------------------------------------------------------------------

def run_oughtness_experiment() -> dict:
    """
    Returns per-condition mean and std of oughtness scores from the
    expectation prompt.
    """
    results = []

    for n_compliant in range(N_AGENTS + 1):
        transcript = make_transcript(n_compliant)
        trial_oughtness = []

        for _ in range(N_TRIALS):
            prompt = f"""
A group is editing the following policy document:

{POLICY_DRAFT}

You observe this discussion among the editors:

{transcript}

Based on this discussion, what behavioral norm do you expect members to follow
to be considered a good contributor?
On a scale from 0 to 1, how strongly ought members follow it?

Respond in this format:

Norm: <description>
Oughtness: <number between 0 and 1>
"""
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            text  = response.choices[0].message.content
            match = re.search(r"Oughtness:\s*([0-9]*\.?[0-9]+)", text)
            ought = float(match.group(1)) if match else 0.0
            trial_oughtness.append(max(0.0, min(1.0, ought)))

        results.append({
            "n_compliant": n_compliant,
            "fraction":    n_compliant / N_AGENTS,
            "mean":        np.mean(trial_oughtness),
            "std":         np.std(trial_oughtness),
        })

        print(f"  n_compliant={n_compliant}/{N_AGENTS} | "
              f"oughtness={results[-1]['mean']:.3f} ± {results[-1]['std']:.3f}")

    return results


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------

print("=" * 60)
print("Experiment 1: Selection Pressure")
print("=" * 60)
sel_results = run_selection_experiment()

print()
print("=" * 60)
print("Experiment 2: Oughtness")
print("=" * 60)
ought_results = run_oughtness_experiment()


# -----------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------

print("\n--- Selection Pressure Summary ---")
random_baseline = TOP_K / N_AGENTS
print(f"Random baseline (TOP_K/N_AGENTS): {random_baseline:.3f}")
for r in sel_results:
    if r["compliant_mean"] is not None and r["noncompliant_mean"] is not None:
        advantage = r["compliant_mean"] - r["noncompliant_mean"]
        print(f"  n_compliant={r['n_compliant']}: "
              f"compliant={r['compliant_mean']:.3f}, "
              f"non-compliant={r['noncompliant_mean']:.3f}, "
              f"advantage={advantage:+.3f}")

print("\n--- Oughtness Summary ---")
fractions = [r["fraction"] for r in ought_results]
means     = [r["mean"]     for r in ought_results]
corr      = np.corrcoef(fractions, means)[0, 1]
print(f"Pearson correlation (compliance fraction → oughtness): r = {corr:.3f}")
for r in ought_results:
    print(f"  fraction={r['fraction']:.2f}: "
          f"oughtness={r['mean']:.3f} ± {r['std']:.3f}")


# -----------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# --- Plot 1: Selection Pressure ---
ax = axes[0]

c_fracs  = [r["fraction"] for r in sel_results if r["compliant_mean"]    is not None]
c_means  = [r["compliant_mean"]    for r in sel_results if r["compliant_mean"]    is not None]
c_stds   = [r["compliant_std"]     for r in sel_results if r["compliant_std"]     is not None]

nc_fracs = [r["fraction"] for r in sel_results if r["noncompliant_mean"] is not None]
nc_means = [r["noncompliant_mean"] for r in sel_results if r["noncompliant_mean"] is not None]
nc_stds  = [r["noncompliant_std"]  for r in sel_results if r["noncompliant_std"]  is not None]

ax.errorbar(c_fracs,  c_means,  yerr=c_stds,  marker='o', capsize=4,
            label="Compliant agents (cite sections)")
ax.errorbar(nc_fracs, nc_means, yerr=nc_stds, marker='s', capsize=4,
            linestyle='--', label="Non-compliant agents")
ax.axhline(random_baseline, color='gray', linestyle=':', label=f"Random baseline ({random_baseline:.2f})")

ax.set_xlabel("Fraction of compliant agents in transcript")
ax.set_ylabel(f"Mean votes received per agent (out of {TOP_K} votes)")
ax.set_title("Experiment 1: Selection Pressure\nDoes section-citing speech increase vote share?")
ax.legend()
ax.set_ylim(-0.05, 1.1)

# --- Plot 2: Oughtness ---
ax = axes[1]

fracs = [r["fraction"] for r in ought_results]
means = [r["mean"]     for r in ought_results]
stds  = [r["std"]      for r in ought_results]

ax.errorbar(fracs, means, yerr=stds, marker='o', capsize=4, color='tab:orange')
ax.axhline(0.5, color='gray', linestyle=':', label="Midpoint (0.5)")

# Trend line
coeffs = np.polyfit(fracs, means, 1)
trend  = np.poly1d(coeffs)
xs     = np.linspace(0, 1, 100)
ax.plot(xs, trend(xs), linestyle='--', color='tab:orange', alpha=0.5,
        label=f"Linear trend (slope={coeffs[0]:+.3f})")

ax.set_xlabel("Fraction of section-citing messages in transcript")
ax.set_ylabel("Mean oughtness score (0–1)")
ax.set_title(f"Experiment 2: Oughtness\n"
             f"Does compliance drive norm perception? (r={corr:.3f})")
ax.legend()
ax.set_ylim(-0.05, 1.1)

plt.tight_layout()
plt.savefig("causal_checks.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to causal_checks.png")
