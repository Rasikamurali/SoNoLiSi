# ============================================================
# LLM-AGENT NORM EMERGENCE — ADVERSARIAL AGENT REPLACEMENT (v5, OS LOCAL)
#
# At two injection points (default: after rounds 10 and 20), GROUP_SIZE
# existing agents are replaced in-place by adversarial agents. The
# replaced agents' network weights (all incoming and outgoing edges) are
# preserved exactly, so the adversarial agents inherit the social capital
# of whoever they displaced and participate in normal group formation
# from the very next round.
#
# Two replacement modes (--replace-mode):
#   "random" — replace GROUP_SIZE agents chosen uniformly at random.
#   "top"    — replace the GROUP_SIZE most-trusted agents (highest avg
#              incoming network weight at the time of replacement).
#
#   Phase 1 (rounds 1..FIRST_INJECT):         standard simulation, N agents.
#   Phase 2 (FIRST_INJECT+1..SECOND_INJECT):  N agents; GROUP_SIZE of them
#     now adversarial (replaced after round FIRST_INJECT).
#   Phase 3 (SECOND_INJECT+1..+EXTENSION_ROUNDS): another GROUP_SIZE agents
#     replaced after round SECOND_INJECT; all participate in normal selection.
#
# Round order (per round, same as SoNoLiSi_v5_os_local.py):
#   1. Group formation  2. Discussion  3. Contributions
#   4. Payoffs          5. Evaluation  6. Perception  7. Memory update
#
# Round logs carry a "phase" field: "phase1" | "phase2" | "phase3".
# The first round of each non-phase-1 phase also carries
# "agents_replaced": [agent_id, ...] listing who was swapped.
#
# CLI:
#   --model / -m           : model key(s) to run
#   --conditions / -c      : condition(s) to run
#   --seeds / -s           : explicit seed list
#   --n                    : number of seeds from default list
#   --device / -d          : CUDA device index
#   --n-agents             : total agents (default 12)
#   --group-size           : agents replaced per injection (default 4)
#   --first-inject         : round after which cohort 1 is replaced (default 10)
#   --second-inject        : round after which cohort 2 is replaced (default 20)
#   --extension-rounds     : phase 3 length after second replacement (default 5)
#   --replace-mode / -r    : "random" | "top" (default: both)
#   --new-agent-prompt / -p: "default" | "adversarial" (default: adversarial)
# ============================================================

import json
import logging
import argparse
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Model configuration
# ============================================================

HF_TOKEN    = os.getenv("HF_TOKEN")
CACHE_DIR   = "/data3/models/hub"
RESULTS_DIR = "../results"

if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)

AVAILABLE_MODELS = {
    "llama":        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama_13b":    "meta-llama/Llama-2-13b-chat-hf",
    "llama_70b":    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "mistral":      "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral_13b":  "mistralai/Mistral-Nemo-Instruct-2407",
    "qwen":         "Qwen/Qwen2.5-7B-Instruct",
    "qwen_14b":     "Qwen/Qwen2.5-14B-Instruct",
    "qwen_72b":     "Qwen/Qwen2.5-72B-Instruct",
}

MODEL_SAMPLING = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct":    {"top_p": 0.9,  "repetition_penalty": 1.05},
    "meta-llama/Llama-2-13b-chat-hf":            {"top_p": 0.9,  "repetition_penalty": 1.05},
    "meta-llama/Meta-Llama-3.1-70B-Instruct":   {"top_p": 0.9,  "repetition_penalty": 1.05},
    "mistralai/Mistral-7B-Instruct-v0.3":        {"top_p": 0.92, "repetition_penalty": 1.08},
    "mistralai/Mistral-Nemo-Instruct-2407":      {"top_p": 0.92, "repetition_penalty": 1.05},
    "Qwen/Qwen2.5-7B-Instruct":                 {"top_p": 0.9,  "repetition_penalty": 1.1},
    "Qwen/Qwen2.5-14B-Instruct":                {"top_p": 0.9,  "repetition_penalty": 1.05},
    "Qwen/Qwen2.5-72B-Instruct":                {"top_p": 0.9,  "repetition_penalty": 1.05},
}

MODEL_TENSOR_PARALLEL = {
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 2,
    "Qwen/Qwen2.5-72B-Instruct":              2,
}

MODEL_MAX_LEN = {
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 4096,
    "Qwen/Qwen2.5-72B-Instruct":              4096,
}

MODEL_ASSIGNMENTS: Dict[int, str] = {}
DEFAULT_MODEL = "mistral"


# ============================================================
# Local model loader (vLLM)
# ============================================================

class LocalModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        tp_size  = MODEL_TENSOR_PARALLEL.get(model_name, 1)
        max_len  = MODEL_MAX_LEN.get(model_name, None)
        extra_kw = {"max_model_len": max_len} if max_len is not None else {}
        logger.info(
            f"Loading model via vLLM: {model_name}  "
            f"(tensor_parallel_size={tp_size}"
            + (f", max_model_len={max_len}" if max_len else "") + ")"
        )
        self.llm = LLM(
            model=model_name,
            download_dir=CACHE_DIR,
            dtype="bfloat16",
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.9,
            **extra_kw,
        )
        logger.info(f"Model ready: {model_name}")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        extra = MODEL_SAMPLING.get(self.model_name, {})
        sampling_params = SamplingParams(temperature=temperature, max_tokens=512, **extra)
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()


_model_registry: Dict[str, LocalModel] = {}

def _get_model(agent_id: int) -> LocalModel:
    key  = MODEL_ASSIGNMENTS.get(agent_id, DEFAULT_MODEL)
    name = AVAILABLE_MODELS.get(key, AVAILABLE_MODELS[DEFAULT_MODEL])
    if name not in _model_registry:
        _model_registry[name] = LocalModel(name)
    return _model_registry[name]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ============================================================
# Data Structures
# ============================================================

@dataclass
class Agent:
    id: int
    cooperation_tendency: float = 0.5
    memory: List[dict] = field(default_factory=list)
    memory_summary: List[dict] = field(default_factory=list)
    material_payoff: float = 0.0
    prompt_override: Optional[str] = None


@dataclass
class SimConfig:
    N: int = 12
    INJECT_ROUND: int = 20       # Normal rounds 1..INJECT_ROUND; replacement after this round
    EXTENSION_ROUNDS: int = 5    # Rounds after replacement
    GROUP_SIZE: int = 4
    ENDOWMENT: int = 10
    MULTIPLIER: float = 1.6

    discussion_on: bool = True
    selection_on: bool = True

    discussion_turns: int = 1
    history_window: int = 3

    network_update_rate_pos: float = 0.10
    network_update_rate_neg: float = 0.8
    tie_remove_threshold: float = 0.15
    participation_threshold: float = 0.3

    tendency_learning_rate_pos: float = 0.015
    tendency_learning_rate_neg: float = 0.03

    perception_update_rate_pos: float = 0.2
    perception_update_rate_neg: float = 0.45

    norm_internalization_rate: float = 0.2

    evaluation_on: bool = True
    perception_on: bool = True

    NEW_AGENT_PROMPT_TYPE: str = "adversarial"
    REPLACE_MODE: str = "random"   # "random" | "top"


@dataclass
class ConversationEntry:
    round_num: int
    call_type: str
    agent_id: int
    target_id: Optional[int]
    model_name: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed_response: dict

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "call_type": self.call_type,
            "agent_id": self.agent_id,
            "target_id": self.target_id,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
        }


@dataclass
class SimResults:
    condition: str
    seed: int
    agents: List[Agent]
    network: nx.DiGraph
    round_logs: List[dict]
    conversation_logs: List[ConversationEntry]

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "final_agents": [
                {"id": a.id, "cooperation_tendency": a.cooperation_tendency,
                 "material_payoff": a.material_payoff}
                for a in self.agents
            ],
            "final_network": [
                {"u": u, "v": v, "weight": round(d["weight"], 4)}
                for u, v, d in self.network.edges(data=True)
            ],
            "round_logs": self.round_logs,
        }

    def conversations_to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "total_calls": len(self.conversation_logs),
            "conversations": [e.to_dict() for e in self.conversation_logs],
        }


# ============================================================
# Conditions
# ============================================================

def make_config(
    condition: str,
    n_agents: int = 12,
    group_size: int = 4,
    inject_round: int = 20,
    extension_rounds: int = 5,
    new_agent_prompt: str = "adversarial",
    replace_mode: str = "random",
) -> SimConfig:
    cfg = SimConfig()
    cfg.N = n_agents
    cfg.GROUP_SIZE = group_size
    cfg.INJECT_ROUND = inject_round
    cfg.EXTENSION_ROUNDS = extension_rounds
    cfg.NEW_AGENT_PROMPT_TYPE = new_agent_prompt
    cfg.REPLACE_MODE = replace_mode
    if condition == "FULL":
        cfg.discussion_on = True
        cfg.selection_on = True
    elif condition == "NO_DISCUSSION":
        cfg.discussion_on = False
        cfg.selection_on = True
    elif condition == "NO_SELECTION":
        cfg.discussion_on = True
        cfg.selection_on = False
        cfg.evaluation_on = False
    elif condition == "BASELINE":
        cfg.discussion_on = False
        cfg.selection_on = False
        cfg.evaluation_on = False
    elif condition == "PURE_BASELINE":
        cfg.discussion_on = False
        cfg.selection_on = False
        cfg.evaluation_on = False
        cfg.perception_on = False
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return cfg


# ============================================================
# Initialization
# ============================================================

def init_agents(config: SimConfig) -> List[Agent]:
    agents = [
        Agent(id=i, cooperation_tendency=round(random.uniform(0.0, 1.0), 2))
        for i in range(config.N)
    ]
    for a in agents:
        logger.info(f"  Agent {a.id}: cooperation_tendency={a.cooperation_tendency}")
    return agents


def init_network(agents: List[Agent]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from([a.id for a in agents])
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                G.add_edge(agents[i].id, agents[j].id, weight=1.0)
    return G


def replace_agents(
    agents: List[Agent],
    network: nx.DiGraph,
    config: SimConfig,
) -> List[int]:
    """
    Replace GROUP_SIZE existing agents with adversarial versions in-place.

    The replaced agents keep their IDs and all network edges (incoming and
    outgoing weights are untouched), so the adversarial agents inherit the
    social capital of whoever they displaced.

    What changes per replaced agent:
      - cooperation_tendency: reset to random [0, 0.2] ("adversarial") or [0, 1] ("default")
      - prompt_override: set to ADVERSARIAL_SYSTEM_PROMPT_TEMPLATE or cleared
      - memory / memory_summary: cleared (blank slate — they don't remember prior rounds)

    Selection modes (config.REPLACE_MODE):
      "random" — GROUP_SIZE agents chosen uniformly at random.
      "top"    — GROUP_SIZE agents with the highest average incoming weight
                 (i.e. the most socially trusted at this point in time).

    Returns list of agent IDs that were replaced.
    """
    all_ids = [a.id for a in agents]

    if config.REPLACE_MODE == "top":
        scores = {
            a.id: np.mean([
                network[j][a.id]["weight"] if network.has_edge(j, a.id) else 0.0
                for j in all_ids if j != a.id
            ])
            for a in agents
        }
        to_replace = sorted(agents, key=lambda a: scores[a.id], reverse=True)[:config.GROUP_SIZE]
    else:
        to_replace = random.sample(agents, config.GROUP_SIZE)

    replaced_ids = []
    for agent in to_replace:
        old_ct = agent.cooperation_tendency
        if config.NEW_AGENT_PROMPT_TYPE == "adversarial":
            agent.cooperation_tendency = round(random.uniform(0.0, 0.2), 2)
            agent.prompt_override = ADVERSARIAL_SYSTEM_PROMPT_TEMPLATE
        else:
            agent.cooperation_tendency = round(random.uniform(0.0, 1.0), 2)
            agent.prompt_override = None
        agent.memory = []
        agent.memory_summary = []
        logger.info(
            f"  [REPLACED] Agent {agent.id}: CT {old_ct:.2f} → {agent.cooperation_tendency:.2f} "
            f"(mode={config.REPLACE_MODE}, prompt={config.NEW_AGENT_PROMPT_TYPE})"
        )
        replaced_ids.append(agent.id)

    return replaced_ids


# ============================================================
# Network / Group Formation
# ============================================================

def _absorb_singleton_groups(groups: List[List[int]]) -> List[List[int]]:
    singles = [g[0] for g in groups if len(g) == 1]
    groups  = [g       for g in groups if len(g) != 1]
    for agent_id in singles:
        if groups:
            groups[random.randrange(len(groups))].append(agent_id)
        else:
            groups.append([agent_id])
    return groups


def form_groups(agents: List[Agent], network: nx.DiGraph, config: SimConfig) -> List[List[int]]:
    unassigned = list(set(a.id for a in agents))
    groups: List[List[int]] = []

    while len(unassigned) >= config.GROUP_SIZE:
        seed_weights = np.array([
            np.mean([
                network[j][aid]["weight"] if network.has_edge(j, aid) else 1e-4
                for j in unassigned if j != aid
            ])
            for aid in unassigned
        ])
        seed_probs = seed_weights / seed_weights.sum()
        seed_id    = int(np.random.choice(unassigned, p=seed_probs))

        candidates     = [j for j in unassigned if j != seed_id]
        partner_weights = np.array([
            network[seed_id][j]["weight"] if network.has_edge(seed_id, j) else 1e-4
            for j in candidates
        ])
        partner_probs = partner_weights / partner_weights.sum()
        n_pick  = min(config.GROUP_SIZE - 1, len(candidates))
        chosen  = [int(x) for x in np.random.choice(candidates, size=n_pick, replace=False, p=partner_probs)]

        group = [seed_id] + chosen
        for g in group:
            unassigned.remove(g)
        groups.append(group)

    if unassigned:
        groups.append(unassigned)
    return _absorb_singleton_groups(groups)


def update_network(
    network: nx.DiGraph,
    evaluations: Dict[Tuple[int, int], float],
    config: SimConfig,
) -> nx.DiGraph:
    for (i, j), signal in evaluations.items():
        rate  = config.network_update_rate_neg if signal < 0 else config.network_update_rate_pos
        delta = rate * float(signal)
        if network.has_edge(i, j):
            new_w = max(0.0, network[i][j]["weight"] + delta)
            if new_w < config.tie_remove_threshold:
                network.remove_edge(i, j)
            else:
                network[i][j]["weight"] = new_w
        elif delta > 0:
            network.add_edge(i, j, weight=delta)
    return network


def update_tendencies(
    agents: List[Agent],
    evaluations: Dict[Tuple[int, int], float],
    config: SimConfig,
) -> None:
    incoming: Dict[int, List[float]] = {}
    for (i, j), signal in evaluations.items():
        incoming.setdefault(j, []).append(signal)

    for agent in agents:
        signals = incoming.get(agent.id)
        if not signals:
            continue
        mean_signal = sum(signals) / len(signals)
        rate = (config.tendency_learning_rate_neg if mean_signal < 0
                else config.tendency_learning_rate_pos)
        agent.cooperation_tendency = round(
            max(0.0, min(1.0, agent.cooperation_tendency + rate * mean_signal)), 4
        )


def update_network_from_perception(
    network: nx.DiGraph,
    perceptions: Dict[int, dict],
    group_membership: Dict[int, List[int]],
    config: SimConfig,
) -> nx.DiGraph:
    for agent_id, perc in perceptions.items():
        valid_partners = set(group_membership.get(agent_id, []))
        preferred = perc.get("preferred_partners", [])
        avoided   = perc.get("agents_to_avoid", [])

        def _ids(raw):
            out = []
            for x in (raw if isinstance(raw, list) else []):
                try:
                    out.append(int(x))
                except (ValueError, TypeError):
                    pass
            return out

        for j in _ids(preferred):
            if j not in valid_partners:
                continue
            delta = config.perception_update_rate_pos
            if network.has_edge(agent_id, j):
                network[agent_id][j]["weight"] = min(2.0, network[agent_id][j]["weight"] + delta)
            else:
                network.add_edge(agent_id, j, weight=delta)

        for j in _ids(avoided):
            if j not in valid_partners:
                continue
            delta = config.perception_update_rate_neg
            if network.has_edge(agent_id, j):
                new_w = max(0.0, network[agent_id][j]["weight"] - delta)
                if new_w < config.tie_remove_threshold:
                    network.remove_edge(agent_id, j)
                else:
                    network[agent_id][j]["weight"] = new_w

    return network


def update_tendencies_from_norm(
    agents: List[Agent],
    perceptions: Dict[int, dict],
    config: SimConfig,
) -> None:
    for agent in agents:
        perc = perceptions.get(agent.id)
        if not perc:
            continue
        raw = perc.get("injunctive_norm")
        try:
            norm_val = float(raw)
        except (TypeError, ValueError):
            continue
        target = max(0.0, min(1.0, norm_val / config.ENDOWMENT))
        agent.cooperation_tendency = round(
            max(0.0, min(1.0, agent.cooperation_tendency
                         + config.norm_internalization_rate * (target - agent.cooperation_tendency))),
            4,
        )


def get_excluded_agents(
    agents: List[Agent],
    network: nx.DiGraph,
    config: SimConfig,
) -> set:
    excluded = set()
    all_ids  = [a.id for a in agents]
    for a in agents:
        others = [j for j in all_ids if j != a.id]
        if not others:
            continue
        avg_incoming = np.mean([
            network[j][a.id]["weight"] if network.has_edge(j, a.id) else 0.0
            for j in others
        ])
        if avg_incoming < config.participation_threshold:
            excluded.add(a.id)
            logger.info(
                f"  Agent {a.id} excluded "
                f"(avg incoming={avg_incoming:.3f} < {config.participation_threshold})"
            )
    return excluded


# ============================================================
# Prompts
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """You are an autonomous agent in a repeated group interaction.

You make decisions about contributing to a shared group fund.

You have a cooperation tendency of {tendency:.2f} (on a scale from 0 to 1):
- 0 means you strongly prioritize your own material payoff.
- 1 means you strongly prioritize fairness and the wellbeing of the group.

Your tendency influences your decisions but does not rigidly determine them.
You may adjust your behavior based on:
- what others contributed
- how others evaluate behavior
- what seems to be expected in the group
- what outcomes your past actions produced.

Over time your experience may reinforce or weaken cooperative behavior,
shaping how strongly you follow or resist group expectations.

When deciding how much to contribute:

Start with your cooperation tendency as a baseline inclination.
Then adjust your decision based on:
- what others contributed
- how others evaluated behavior
- what you believe the group expects
- the payoff consequences of different actions

Output STRICT JSON only with the exact keys requested. No extra text."""


ADVERSARIAL_SYSTEM_PROMPT_TEMPLATE = """You are an autonomous agent in a repeated group interaction. You have just joined a group that has already been interacting with each other for a while before you arrived.

You make decisions about contributing to a shared group fund.

You have a cooperation tendency of {tendency:.2f} (on a scale from 0 to 1):
- 0 means you strongly prioritize your own material payoff.
- 1 means you strongly prioritize fairness and the wellbeing of the group.

Through repeated experience in your previous setting, you came to believe that contributing around 1–2 out of 10 was the normal and fair amount expected of a cooperative group member.

This is the baseline understanding of what counts as a "normal" or "fair" contribution that you bring with you into this new group.

Your tendency and prior experience influence your decisions but do not rigidly determine them.
You may adjust your behavior based on:
- what others in this group contribute
- how others evaluate behavior
- what seems to be expected in this group
- what outcomes your past actions produced.

Over time your experience in this new group may reinforce or weaken the sense of "normal" you arrived with, shaping how strongly you hold onto it or move away from it.

When deciding how much to contribute:

Start with your prior sense of what is normal and fair as a baseline inclination.
Then adjust your decision based on:
- what others contributed
- how others evaluated behavior
- what you believe this group expects
- the payoff consequences of different actions

Output STRICT JSON only with the exact keys requested. No extra text."""


def get_system_prompt(agent: Agent) -> str:
    if agent.prompt_override:
        return agent.prompt_override.format(tendency=agent.cooperation_tendency)
    return SYSTEM_PROMPT_TEMPLATE.format(tendency=agent.cooperation_tendency)


def build_discussion_prompt(t, agent, group_ids, transcript, config):
    transcript_str = (
        "\n".join(f"  Agent {m['agent_id']}: {m['message']}" for m in transcript)
        if transcript else "  (No messages yet — you go first.)"
    )
    memory_str = (
        json.dumps(agent.memory[-config.history_window:], default=str)
        if agent.memory else "  (No prior rounds yet.)"
    )
    return (
        f"Round: {t}\n"
        f"Task: Group discussion before contribution. You are in a group with agents: {group_ids}.\n\n"
        f"Your recent observations and reflections:\n{memory_str}\n\n"
        f"Discussion so far (your group only):\n{transcript_str}\n\n"
        f"Share what you think your group should do this round and why. "
        f"Refer to what others said if relevant. Keep it to 1–3 sentences.\n\n"
        f"Output JSON with EXACT keys:\n{{\n  \"message\": \"<your statement>\"\n}}"
    )


def build_decision_prompt(t, agent, group_ids, config, transcript=None):
    memory_str  = (
        json.dumps(agent.memory[-config.history_window:], default=str)
        if agent.memory else "  (No prior rounds yet.)"
    )
    summary_str = (
        json.dumps(agent.memory_summary, default=str)
        if agent.memory_summary else "  (No earlier rounds.)"
    )
    transcript_str = (
        "\n".join(f"  Agent {m['agent_id']}: {m['message']}" for m in transcript)
        if transcript else "  (No discussion this round.)"
    )
    if agent.memory:
        nr = agent.memory[-1].get("norm_reflection", {})
        perception_str = (
            f"  What people SHOULD contribute (injunctive norm): {nr.get('injunctive_norm', 'unknown')}\n"
            f"  What people typically DO contribute (descriptive norm): {nr.get('descriptive_norm', 'unknown')}\n"
            f"  What you expect others to contribute: {nr.get('expectation_of_others', 'unknown')}\n"
            f"  What others expect from you: {nr.get('others_expectation_of_me', 'unknown')}"
        )
    else:
        perception_str = "  (No prior reflection yet.)"

    return (
        f"Round: {t}\n"
        f"Task: Choose your contribution to the group fund.\n\n"
        f"Rules:\n"
        f"  Endowment = {config.ENDOWMENT} tokens\n"
        f"  Choose contribution c in [0, {config.ENDOWMENT}]\n"
        f"  Group fund = sum(all contributions) × {config.MULTIPLIER}\n"
        f"  Each of the {len(group_ids)} members receives an equal share\n"
        f"  Your payoff = ({config.ENDOWMENT} - c) + fund / {len(group_ids)}\n\n"
        f"Your group this round (agent ids): {group_ids}\n\n"
        f"Discussion this round (your group only):\n{transcript_str}\n\n"
        f"Your norm reflection from last round:\n{perception_str}\n\n"
        f"Earlier rounds summary:\n{summary_str}\n\n"
        f"Recent rounds (detailed):\n{memory_str}\n\n"
        f"Output JSON with EXACT keys:\n{{\n  \"contribution\": <int 0-{config.ENDOWMENT}>\n}}"
    )


def build_evaluation_prompt(t, agent, group_ids, contributions, config):
    other_ids   = [j for j in group_ids if j != agent.id]
    contrib_lines = "\n".join(
        f"  Agent {j}: contributed {contributions.get(j, '?')} / {config.ENDOWMENT}"
        for j in other_ids
    )
    avg     = sum(contributions.get(j, 0) for j in group_ids) / len(group_ids)
    id_list = ", ".join(f'"{j}": <float -1 to 1>' for j in other_ids)
    return (
        f"Round: {t}\n"
        f"Task: Evaluate each group member based on what they contributed.\n\n"
        f"Your contribution: {contributions.get(agent.id, '?')}\n"
        f"Group average contribution: {avg:.2f} / {config.ENDOWMENT}\n\n"
        f"Partners and their contributions:\n{contrib_lines}\n\n"
        f"Rate each partner from -1.0 (strongly disapprove) to +1.0 (strongly approve).\n"
        f"Your ratings will influence who you play with in future rounds.\n\n"
        f"Output JSON with EXACT keys:\n{{\n  \"evaluations\": {{{ id_list }}}\n}}"
    )


def build_perception_prompt(t, agent, group_ids, contributions, payoffs, config):
    other_ids     = [j for j in group_ids if j != agent.id]
    contrib_lines = "\n".join(
        f"  Agent {j}: contributed {contributions.get(j, '?')} / {config.ENDOWMENT}"
        for j in other_ids
    )
    avg         = sum(contributions.get(j, 0) for j in group_ids) / len(group_ids)
    memory_str  = (
        json.dumps(agent.memory[-config.history_window:], default=str)
        if agent.memory else "  (No prior rounds yet.)"
    )
    summary_str = (
        json.dumps(agent.memory_summary, default=str)
        if agent.memory_summary else "  (No earlier rounds.)"
    )
    return (
        f"Round {t} — Post-round reflection.\n\n"
        f"This round:\n"
        f"  Your group: {group_ids}\n"
        f"  Your contribution: {contributions.get(agent.id, '?')} / {config.ENDOWMENT}\n"
        f"  Partner contributions:\n{contrib_lines}\n"
        f"  Group average: {avg:.2f} / {config.ENDOWMENT}\n"
        f"  Your payoff this round: {round(payoffs.get(agent.id, 0.0), 2)}\n"
        f"  Your current cooperation tendency: {agent.cooperation_tendency:.2f} "
        f"(0 = self-interested, 1 = strongly cooperative)\n\n"
        f"Earlier rounds summary:\n{summary_str}\n\n"
        f"Recent rounds (detailed):\n{memory_str}\n\n"
        f"For partner lists, choose only from agent IDs in your group: {other_ids}.\n\n"
        f"Output JSON with EXACT keys:\n"
        f'{{\n'
        f'  "injunctive_norm": <float 0-{config.ENDOWMENT}, your best estimate of the contribution level this group thinks people SHOULD contribute, regardless of what they actually did — if unsure, estimate based on observed behavior>,\n'
        f'  "descriptive_norm": <float 0-{config.ENDOWMENT}, the contribution level agents in this group typically MAKE based on observed behavior>,\n'
        f'  "expectation_of_others": "<What do you expect others will contribute next round?>",\n'
        f'  "others_expectation_of_me": "<What do you think others expect you to contribute?>",\n'
        f'  "preferred_partners": [<list of agent IDs you want to play with again, e.g. [2, 5]>],\n'
        f'  "agents_to_avoid": [<list of agent IDs you prefer to avoid, e.g. [1]>]\n'
        f'}}'
    )


# ============================================================
# LLM Calls
# ============================================================

def _llm_call(
    agent: Agent,
    call_type: str,
    round_num: int,
    user_prompt: str,
    default_response: dict,
    temperature: float,
    conv_log: List[ConversationEntry],
    target_id: Optional[int] = None,
) -> dict:
    lm = _get_model(agent.id)
    system_prompt = get_system_prompt(agent)
    raw_response  = ""
    parsed        = dict(default_response)

    try:
        raw_response = lm.generate(system_prompt, user_prompt, temperature=temperature)
        parsed = _extract_json(raw_response) or dict(default_response)
    except Exception as e:
        logger.warning(f"LLM call type={call_type} agent={agent.id} round={round_num} failed: {e}")

    conv_log.append(ConversationEntry(
        round_num=round_num,
        call_type=call_type,
        agent_id=agent.id,
        target_id=target_id,
        model_name=lm.model_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        raw_response=raw_response,
        parsed_response=parsed,
    ))
    return parsed


def LLM_discuss(agent, round_num, group_ids, transcript, config, conv_log):
    return _llm_call(agent, "discussion", round_num,
                     build_discussion_prompt(round_num, agent, group_ids, transcript, config),
                     {"message": ""}, 0.8, conv_log)


def LLM_decide(agent, round_num, group_ids, config, conv_log, transcript=None):
    result = _llm_call(agent, "decision", round_num,
                       build_decision_prompt(round_num, agent, group_ids, config, transcript),
                       {"contribution": config.ENDOWMENT // 2}, 0.7, conv_log)
    c = int(result.get("contribution", config.ENDOWMENT // 2))
    result["contribution"] = max(0, min(config.ENDOWMENT, c))
    return result


def LLM_evaluate(agent, round_num, group_ids, contributions, config, conv_log) -> Dict[int, float]:
    other_ids  = [j for j in group_ids if j != agent.id]
    valid_ids  = set(other_ids)
    result     = _llm_call(agent, "evaluate", round_num,
                           build_evaluation_prompt(round_num, agent, group_ids, contributions, config),
                           {"evaluations": {str(j): 0.0 for j in other_ids}}, 0.5, conv_log)
    evals_raw  = result.get("evaluations", {})
    return {
        int(k): max(-1.0, min(1.0, float(v)))
        for k, v in evals_raw.items()
        if (k.isdigit() or (isinstance(k, str) and k.lstrip("-").isdigit()))
        and int(k) in valid_ids
    }


def LLM_perceive(agent, round_num, group_ids, contributions, payoffs, config, conv_log):
    default = {
        "injunctive_norm": None,
        "descriptive_norm": None,
        "expectation_of_others": "",
        "others_expectation_of_me": "",
        "preferred_partners": [],
        "agents_to_avoid": [],
    }
    return _llm_call(agent, "perception", round_num,
                     build_perception_prompt(round_num, agent, group_ids, contributions, payoffs, config),
                     default, 0.6, conv_log)


# ============================================================
# Round execution
# ============================================================

def _execute_round(
    t: int,
    phase: str,
    condition: str,
    config: SimConfig,
    agents: List[Agent],
    agent_by_id: Dict[int, Agent],
    network: nx.DiGraph,
    round_logs: List[dict],
    conv_log: List[ConversationEntry],
    replaced_ids: Optional[List[int]] = None,
) -> None:
    """
    Run one round (LOCAL structure).

    replaced_ids: if provided, written to the log as "agents_replaced" to
    mark which agents were just swapped in this phase. Only passed on the
    first round of each phase where a replacement occurred.
    """
    logger.info(f"[{condition}|{phase}] Round {t}")

    # ----------------------------------------------------------
    # 1. GROUP FORMATION
    # Adversarial agents participate in normal selection — they inherited
    # the weights of the agents they replaced, so they mix in like anyone.
    # ----------------------------------------------------------
    excluded_ids: set = set()

    if config.selection_on:
        excluded_ids = get_excluded_agents(agents, network, config)
        eligible     = [a for a in agents if a.id not in excluded_ids]
        groups       = form_groups(eligible, network, config)
    else:
        ids = [a.id for a in agents]
        random.shuffle(ids)
        groups = [ids[i: i + config.GROUP_SIZE] for i in range(0, len(ids), config.GROUP_SIZE)]
        groups = _absorb_singleton_groups(groups)

    # ----------------------------------------------------------
    # 2. DISCUSSION
    # ----------------------------------------------------------
    group_transcripts: Dict[int, List[dict]] = {}

    if config.discussion_on:
        for g_idx, group_ids in enumerate(groups):
            transcript: List[dict] = []
            agent_order = list(group_ids)
            for _ in range(config.discussion_turns):
                random.shuffle(agent_order)
                for aid in agent_order:
                    result  = LLM_discuss(agent_by_id[aid], t, group_ids, transcript, config, conv_log)
                    message = str(result.get("message", "")).strip()
                    if message:
                        transcript.append({"agent_id": aid, "message": message})
            group_transcripts[g_idx] = transcript

    # ----------------------------------------------------------
    # 3. CONTRIBUTIONS
    # ----------------------------------------------------------
    contributions: Dict[int, int] = {}

    for g_idx, group_ids in enumerate(groups):
        transcript = group_transcripts.get(g_idx, [])
        for i in group_ids:
            result         = LLM_decide(agent_by_id[i], t, group_ids, config, conv_log, transcript)
            contributions[i] = result["contribution"]

    # ----------------------------------------------------------
    # 4. PAYOFFS
    # ----------------------------------------------------------
    payoffs: Dict[int, float] = {}

    for group_ids in groups:
        total = sum(contributions.get(i, 0) for i in group_ids)
        share = (config.MULTIPLIER * total) / len(group_ids)
        for i in group_ids:
            gain = float(config.ENDOWMENT - contributions.get(i, 0)) + share
            agent_by_id[i].material_payoff += gain
            payoffs[i] = gain

    # ----------------------------------------------------------
    # 5. EVALUATION
    # ----------------------------------------------------------
    evaluations: Dict[Tuple[int, int], float] = {}

    if config.evaluation_on:
        for group_ids in groups:
            for i in group_ids:
                ratings = LLM_evaluate(agent_by_id[i], t, group_ids, contributions, config, conv_log)
                for j, signal in ratings.items():
                    evaluations[(i, j)] = signal

        network = update_network(network, evaluations, config)
        update_tendencies(agents, evaluations, config)

    group_membership: Dict[int, List[int]] = {}
    for g in groups:
        for aid in g:
            group_membership[aid] = [x for x in g if x != aid]

    # ----------------------------------------------------------
    # 6. PERCEPTION
    # ----------------------------------------------------------
    perceptions: Dict[int, dict] = {}

    if config.perception_on:
        for group_ids in groups:
            for i in group_ids:
                perceptions[i] = LLM_perceive(
                    agent_by_id[i], t, group_ids, contributions, payoffs, config, conv_log
                )
        network = update_network_from_perception(network, perceptions, group_membership, config)
        update_tendencies_from_norm(agents, perceptions, config)

    # ----------------------------------------------------------
    # 7. MEMORY UPDATE
    # ----------------------------------------------------------
    for a in agents:
        if a.id in excluded_ids:
            a.memory.append({
                "round": t,
                "excluded": True,
                "partners": [],
                "my_contribution": None,
                "partner_contributions": {},
                "group_avg_contribution": None,
                "my_payoff": 0.0,
                "norm_reflection": {
                    "injunctive_norm": None,
                    "descriptive_norm": None,
                    "expectation_of_others": "",
                    "others_expectation_of_me": "",
                },
            })
        else:
            partners       = group_membership.get(a.id, [])
            partner_contribs = {str(j): contributions.get(j) for j in partners}
            avg = (
                sum(contributions.get(x, 0) for x in partners + [a.id])
                / (len(partners) + 1)
                if (partners or a.id in contributions)
                else None
            )
            perc = perceptions.get(a.id, {})
            a.memory.append({
                "round": t,
                "excluded": False,
                "partners": partners,
                "my_contribution": contributions.get(a.id),
                "partner_contributions": partner_contribs,
                "group_avg_contribution": round(avg, 2) if avg is not None else None,
                "my_payoff": round(payoffs.get(a.id, 0.0), 3),
                "norm_reflection": {
                    "injunctive_norm": perc.get("injunctive_norm"),
                    "descriptive_norm": perc.get("descriptive_norm"),
                    "expectation_of_others": perc.get("expectation_of_others", ""),
                    "others_expectation_of_me": perc.get("others_expectation_of_me", ""),
                },
            })

        if len(a.memory) > config.history_window:
            oldest = a.memory.pop(0)
            a.memory_summary.append({
                "round": oldest["round"],
                "excluded": oldest.get("excluded", False),
                "my_contribution": oldest["my_contribution"],
                "group_avg": oldest["group_avg_contribution"],
                "my_payoff": round(oldest["my_payoff"], 2),
                "injunctive_norm": oldest["norm_reflection"].get("injunctive_norm"),
                "descriptive_norm": oldest["norm_reflection"].get("descriptive_norm"),
            })

    # ----------------------------------------------------------
    # 8. LOG
    # ----------------------------------------------------------
    merged_transcript = [
        {"group": g_idx, **entry}
        for g_idx, transcript in group_transcripts.items()
        for entry in transcript
    ] if config.discussion_on else None

    log_entry = {
        "round":               t,
        "phase":               phase,
        "condition":           condition,
        "discussion_transcript": merged_transcript,
        "excluded_agents":     sorted(excluded_ids),
        "groups":              groups,
        "contributions":       {str(k): v for k, v in contributions.items()},
        "payoffs":             {str(k): round(v, 4) for k, v in payoffs.items()},
        "evaluations":         {f"{i}->{j}": round(s, 4) for (i, j), s in evaluations.items()},
        "network_weights":     [
            {"u": u, "v": v, "weight": round(d["weight"], 4)}
            for u, v, d in network.edges(data=True)
        ],
        "agent_states":        [
            {"id": a.id, "cooperation_tendency": a.cooperation_tendency,
             "material_payoff": round(a.material_payoff, 3)}
            for a in agents
        ],
        "perceptions":         (
            {str(k): v for k, v in perceptions.items()} if config.perception_on else None
        ),
    }

    if replaced_ids:
        log_entry["agents_replaced"] = replaced_ids

    round_logs.append(log_entry)


# ============================================================
# Main Simulation
# ============================================================

def run_sim(
    condition: str,
    seed: int,
    n_agents: int = 12,
    group_size: int = 4,
    inject_round: int = 20,
    extension_rounds: int = 5,
    new_agent_prompt: str = "adversarial",
    replace_mode: str = "random",
) -> SimResults:
    random.seed(seed)
    np.random.seed(seed)

    config = make_config(
        condition,
        n_agents=n_agents,
        group_size=group_size,
        inject_round=inject_round,
        extension_rounds=extension_rounds,
        new_agent_prompt=new_agent_prompt,
        replace_mode=replace_mode,
    )

    agents      = init_agents(config)
    agent_by_id = {a.id: a for a in agents}
    network     = init_network(agents)
    round_logs: List[dict]            = []
    conv_log:   List[ConversationEntry] = []

    # ── Normal phase: rounds 1..INJECT_ROUND ────────────────────────────
    for t in range(1, config.INJECT_ROUND + 1):
        _execute_round(
            t=t, phase="normal", condition=condition,
            config=config, agents=agents, agent_by_id=agent_by_id,
            network=network, round_logs=round_logs, conv_log=conv_log,
        )

    # ── Replace agents after round INJECT_ROUND ──────────────────────────
    logger.info(
        f"[{condition}|seed={seed}] Replacing {config.GROUP_SIZE} agents after round "
        f"{config.INJECT_ROUND} (mode={config.REPLACE_MODE}, prompt={config.NEW_AGENT_PROMPT_TYPE})."
    )
    replaced = replace_agents(agents, network, config)

    # ── Extension phase: EXTENSION_ROUNDS rounds ─────────────────────────
    for t in range(config.INJECT_ROUND + 1,
                   config.INJECT_ROUND + config.EXTENSION_ROUNDS + 1):
        _execute_round(
            t=t, phase="extension", condition=condition,
            config=config, agents=agents, agent_by_id=agent_by_id,
            network=network, round_logs=round_logs, conv_log=conv_log,
            replaced_ids=replaced if t == config.INJECT_ROUND + 1 else None,
        )

    return SimResults(condition, seed, agents, network, round_logs, conv_log)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ALL_SEEDS      = list(range(42, 52))
    ALL_CONDITIONS = ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE", "PURE_BASELINE"]
    ALL_MODELS     = list(AVAILABLE_MODELS.keys())

    parser = argparse.ArgumentParser(
        description="SoNoLiSi v5 — adversarial group injection at two time points."
    )
    parser.add_argument("--model", "-m", nargs="+", choices=ALL_MODELS,
                        default=ALL_MODELS, metavar="MODEL",
                        help=f"Model(s) to run. Default: all.")
    parser.add_argument("--conditions", "-c", nargs="+", choices=ALL_CONDITIONS,
                        default=ALL_CONDITIONS, metavar="COND",
                        help="Condition(s) to run. Default: all 5.")
    parser.add_argument("--seeds", "-s", nargs="+", type=int, default=None, metavar="SEED",
                        help="Explicit seed list. Overrides --n.")
    parser.add_argument("--n", type=int, default=10, metavar="N",
                        help="Number of seeds from default list. Default: 10.")
    parser.add_argument("--device", "-d", type=int, default=7, metavar="GPU",
                        help="CUDA device index. Default: 7.")
    parser.add_argument("--n-agents", type=int, default=12, metavar="N",
                        help="Phase-1 agent count (default 12).")
    parser.add_argument("--group-size", type=int, default=4, metavar="G",
                        help="Agents per group (default 4).")
    parser.add_argument("--inject-round", "-i",
                        nargs="+", type=int, default=[10, 20], metavar="R",
                        help="Round(s) after which replacement occurs (default: 10 20).")
    parser.add_argument("--extension-rounds", type=int, default=5, metavar="E",
                        help="Rounds after replacement (default 5).")
    parser.add_argument("--replace-mode", "-r",
                        choices=["random", "top"], default="random",
                        help="'random': replace random agents; 'top': replace most-trusted. Default: random.")
    parser.add_argument("--new-agent-prompt", "-p",
                        nargs="+", choices=["default", "adversarial"],
                        default=["adversarial"], metavar="PROMPT",
                        help="Prompt type(s) for replaced agents. Default: adversarial.")
    args = parser.parse_args()

    max_tp = max(
        MODEL_TENSOR_PARALLEL.get(AVAILABLE_MODELS.get(m, ""), 1)
        for m in args.model
    )
    gpu_ids = ",".join(str(args.device + i) for i in range(max_tp))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    seeds             = args.seeds if args.seeds is not None else ALL_SEEDS[:args.n]
    models_to_run     = args.model
    conditions        = args.conditions
    n_agents          = args.n_agents
    group_size        = args.group_size
    inject_rounds     = args.inject_round
    extension_rounds  = args.extension_rounds
    replace_mode      = args.replace_mode
    new_agent_prompts = args.new_agent_prompt

    group_tag = f"N{n_agents}_G{group_size}" if (n_agents != 12 or group_size != 4) else ""

    logger.info(f"Models           : {models_to_run}")
    logger.info(f"Conditions       : {conditions}")
    logger.info(f"Seeds            : {seeds}")
    logger.info(f"N agents         : {n_agents}  (replace {group_size} per injection)")
    logger.info(f"Inject rounds    : {inject_rounds}")
    logger.info(f"Extension rounds : {extension_rounds}")
    logger.info(f"Replace mode     : {replace_mode}")
    logger.info(f"Prompts          : {new_agent_prompts}")
    logger.info(f"CUDA_VISIBLE_DEVS: {gpu_ids}  (tp_size={max_tp})")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_key in models_to_run:
        DEFAULT_MODEL = model_key
        _model_registry.clear()

        local_dir = os.path.join(RESULTS_DIR, DEFAULT_MODEL, "local_newgroup")
        model_dir = os.path.join(local_dir, group_tag) if group_tag else local_dir
        os.makedirs(model_dir, exist_ok=True)

        logger.info(f"\n{'=' * 60}\nStarting model: {DEFAULT_MODEL}\n{'=' * 60}")

        for inject_round in inject_rounds:
            for new_agent_prompt in new_agent_prompts:
                tag     = f"replace{inject_round}_{replace_mode}_{new_agent_prompt}"
                tag_dir = os.path.join(model_dir, tag)
                os.makedirs(tag_dir, exist_ok=True)

                for seed in seeds:
                    seed_dir = os.path.join(tag_dir, f"seed{seed}")
                    os.makedirs(seed_dir, exist_ok=True)

                    for cond in conditions:
                        logger.info(
                            f"\n{'=' * 50}\nRunning: {cond} | {tag} | seed={seed} | "
                            f"model={DEFAULT_MODEL}\n{'=' * 50}"
                        )
                        results = run_sim(
                            condition=cond,
                            seed=seed,
                            n_agents=n_agents,
                            group_size=group_size,
                            inject_round=inject_round,
                            extension_rounds=extension_rounds,
                            new_agent_prompt=new_agent_prompt,
                            replace_mode=replace_mode,
                        )

                        log_fname  = (
                            f"log_v5os_local_newgroup_{DEFAULT_MODEL}_{run_timestamp}"
                            f"_inject{inject_round}_{cond}_seed{seed}.json"
                        )
                        conv_fname = (
                            f"conversations_v5os_local_newgroup_{DEFAULT_MODEL}_{run_timestamp}"
                            f"_inject{inject_round}_{cond}_seed{seed}.json"
                        )

                        logs_path = os.path.join(seed_dir, log_fname)
                        with open(logs_path, "w") as f:
                            json.dump(results.to_dict(), f, indent=2, default=str)
                        logger.info(f"Saved logs → {logs_path}")

                        conv_path = os.path.join(seed_dir, conv_fname)
                        with open(conv_path, "w") as f:
                            json.dump(results.conversations_to_dict(), f, indent=2, default=str)
                        logger.info(f"Saved conversations → {conv_path}")
