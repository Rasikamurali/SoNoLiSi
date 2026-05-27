# ============================================================
# LLM-AGENT NORM EMERGENCE — NEW AGENT INTRODUCTION (v5)
#
# Based on SoNoLiSi_v5.py with one structural addition:
#
#   Phase 1 (rounds 1–20): standard simulation as in v5.
#   Phase 2 (rounds 21–25): one new agent is introduced into the
#     established society and participates in 5 additional rounds.
#
# The new agent starts with:
#   - Random cooperation_tendency (same initialisation as original agents)
#   - Empty memory and memory_summary  (no prior observations)
#   - No accumulated norm reflections  (blank perception module)
#   - Equal network edges to all existing agents (weight = 1.0)
#
# In PURE_BASELINE perception is off for everyone, so the new agent
# enters without any perception-module activity — consistent with that
# condition. In all other conditions the new agent participates fully
# but begins with no prior beliefs.
#
# Round logs include a "phase" field: "normal" | "extension".
# Extension logs also carry "new_agent_id" to mark the introduction.
# ============================================================

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


@dataclass
class SimConfig:
    N: int = 12
    ROUNDS: int = 20
    EXTENSION_ROUNDS: int = 5          # extra rounds after new-agent introduction
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


@dataclass
class ConversationEntry:
    round_num: int
    call_type: str
    agent_id: int
    target_id: Optional[int]
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


def make_config(condition: str) -> SimConfig:
    cfg = SimConfig()
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


def introduce_new_agent(
    agents: List[Agent],
    agent_by_id: Dict[int, Agent],
    network: nx.DiGraph,
) -> Agent:
    """
    Add one new agent to the simulation after the normal phase ends.

    The new agent:
      - ID = current max agent ID + 1
      - Random cooperation_tendency
      - Empty memory and memory_summary (blank slate)
      - Connected to all existing agents with weight 1.0 (both directions)
    """
    new_id = max(a.id for a in agents) + 1
    new_agent = Agent(
        id=new_id,
        cooperation_tendency=round(random.uniform(0.0, 1.0), 2),
    )
    logger.info(
        f"  [NEW AGENT] Introduced agent {new_id} "
        f"(CT={new_agent.cooperation_tendency:.2f}, empty memory)"
    )

    agents.append(new_agent)
    agent_by_id[new_id] = new_agent

    network.add_node(new_id)
    for a in agents:
        if a.id != new_id:
            network.add_edge(new_id, a.id, weight=1.0)
            if not network.has_edge(a.id, new_id):
                network.add_edge(a.id, new_id, weight=1.0)

    return new_agent


# ============================================================
# Network / Group Formation
# ============================================================


def form_groups(
    agents: List[Agent],
    network: nx.DiGraph,
    config: SimConfig,
) -> List[List[int]]:
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
        seed_id = int(np.random.choice(unassigned, p=seed_probs))

        candidates = [j for j in unassigned if j != seed_id]
        partner_weights = np.array([
            network[seed_id][j]["weight"] if network.has_edge(seed_id, j) else 1e-4
            for j in candidates
        ])
        partner_probs = partner_weights / partner_weights.sum()

        n_pick = min(config.GROUP_SIZE - 1, len(candidates))
        chosen = [int(x) for x in np.random.choice(
            candidates, size=n_pick, replace=False, p=partner_probs
        )]

        group = [seed_id] + chosen
        for g in group:
            unassigned.remove(g)
        groups.append(group)

    if unassigned:
        groups.append(unassigned)

    return [g for g in groups if len(g) >= 2]


def update_network(
    network: nx.DiGraph,
    evaluations: Dict[Tuple[int, int], float],
    config: SimConfig,
) -> nx.DiGraph:
    for (i, j), signal in evaluations.items():
        rate = config.network_update_rate_neg if signal < 0 else config.network_update_rate_pos
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
        new_tendency = agent.cooperation_tendency + rate * mean_signal
        agent.cooperation_tendency = round(max(0.0, min(1.0, new_tendency)), 4)


def update_network_from_perception(
    network: nx.DiGraph,
    perceptions: Dict[int, dict],
    config: SimConfig,
) -> nx.DiGraph:
    for agent_id, perc in perceptions.items():
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
            delta = config.perception_update_rate_pos
            if network.has_edge(agent_id, j):
                network[agent_id][j]["weight"] = min(2.0, network[agent_id][j]["weight"] + delta)
            else:
                network.add_edge(agent_id, j, weight=delta)

        for j in _ids(avoided):
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
            max(0.0, min(1.0,
                agent.cooperation_tendency
                + config.norm_internalization_rate * (target - agent.cooperation_tendency)
            )),
            4,
        )


def get_excluded_agents(
    agents: List[Agent],
    network: nx.DiGraph,
    config: SimConfig,
) -> set:
    excluded = set()
    all_ids = [a.id for a in agents]
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


def get_system_prompt(agent: Agent) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(tendency=agent.cooperation_tendency)


def build_discussion_prompt(
    t: int,
    agent: Agent,
    transcript: List[dict],
    config: SimConfig,
) -> str:
    if transcript:
        transcript_str = "\n".join(
            f"  Agent {m['agent_id']}: {m['message']}" for m in transcript
        )
    else:
        transcript_str = "  (No messages yet — you go first.)"

    memory_str = (
        json.dumps(agent.memory[-config.history_window:], default=str)
        if agent.memory else "  (No prior rounds yet.)"
    )

    return (
        f"Round: {t}\n"
        f"Task: Group discussion before partner selection and contribution.\n\n"
        f"Your recent observations and reflections:\n{memory_str}\n\n"
        f"Discussion so far:\n{transcript_str}\n\n"
        f"Share what you think the group should do this round and why. "
        f"Refer to what others said if relevant. Keep it to 1–3 sentences.\n\n"
        f"Output JSON with EXACT keys:\n"
        f'{{\n'
        f'  "message": "<your statement>"\n'
        f'}}'
    )


def build_decision_prompt(
    t: int,
    agent: Agent,
    group_ids: List[int],
    config: SimConfig,
    transcript: List[dict] = None,
) -> str:
    memory_str = (
        json.dumps(agent.memory[-config.history_window:], default=str)
        if agent.memory else "  (No prior rounds yet.)"
    )
    summary_str = (
        json.dumps(agent.memory_summary, default=str)
        if agent.memory_summary else "  (No earlier rounds.)"
    )

    if transcript:
        transcript_str = "\n".join(
            f"  Agent {m['agent_id']}: {m['message']}" for m in transcript
        )
    else:
        transcript_str = "  (No discussion this round.)"

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
        f"Discussion this round:\n{transcript_str}\n\n"
        f"Your norm reflection from last round:\n{perception_str}\n\n"
        f"Earlier rounds summary:\n{summary_str}\n\n"
        f"Recent rounds (detailed):\n{memory_str}\n\n"
        f"Output JSON with EXACT keys:\n"
        f'{{\n'
        f'  "contribution": <int 0-{config.ENDOWMENT}>\n'
        f'}}'
    )


def build_evaluation_prompt(
    t: int,
    agent: Agent,
    group_ids: List[int],
    contributions: Dict[int, int],
    config: SimConfig,
) -> str:
    other_ids = [j for j in group_ids if j != agent.id]
    contrib_lines = "\n".join(
        f"  Agent {j}: contributed {contributions.get(j, '?')} / {config.ENDOWMENT}"
        for j in other_ids
    )
    avg = sum(contributions.get(j, 0) for j in group_ids) / len(group_ids)
    id_list = ", ".join(f'"{j}": <float -1 to 1>' for j in other_ids)

    return (
        f"Round: {t}\n"
        f"Task: Evaluate each group member based on what they contributed.\n\n"
        f"Your contribution: {contributions.get(agent.id, '?')}\n"
        f"Group average contribution: {avg:.2f} / {config.ENDOWMENT}\n\n"
        f"Partners and their contributions:\n{contrib_lines}\n\n"
        f"Rate each partner from -1.0 (strongly disapprove) to +1.0 (strongly approve).\n"
        f"Your ratings will influence who you play with in future rounds.\n\n"
        f"Output JSON with EXACT keys:\n"
        f'{{\n'
        f'  "evaluations": {{{ id_list }}}\n'
        f'}}'
    )


def build_perception_prompt(
    t: int,
    agent: Agent,
    group_ids: List[int],
    contributions: Dict[int, int],
    payoffs: Dict[int, float],
    config: SimConfig,
) -> str:
    other_ids = [j for j in group_ids if j != agent.id]
    contrib_lines = "\n".join(
        f"  Agent {j}: contributed {contributions.get(j, '?')} / {config.ENDOWMENT}"
        for j in other_ids
    )
    avg = sum(contributions.get(j, 0) for j in group_ids) / len(group_ids)
    memory_str = (
        json.dumps(agent.memory[-config.history_window:], default=str)
        if agent.memory else "  (No prior rounds yet.)"
    )
    summary_str = (
        json.dumps(agent.memory_summary, default=str)
        if agent.memory_summary else "  (No earlier rounds.)"
    )
    all_ids = [j for j in group_ids if j != agent.id]

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
        f"For partner lists, choose only from agent IDs in your group: {all_ids}.\n\n"
        f"Note: If you have limited experience with this group, the injunctive norm may be uncertain. "
        f"Only report a confident value if the group's shared expectations are reasonably clear to you.\n\n"
        f"Output JSON with EXACT keys:\n"
        f'{{\n'
        f'  "injunctive_norm": <float 0-{config.ENDOWMENT}, the contribution level agents in this group think people SHOULD contribute, regardless of what they actually did — or null if unclear>,\n'
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
    system_prompt = get_system_prompt(agent)
    raw_response = ""
    parsed = dict(default_response)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw_response = resp.choices[0].message.content
        parsed = json.loads(raw_response)
    except Exception as e:
        logger.warning(
            f"LLM call type={call_type} agent={agent.id} round={round_num} failed: {e}."
        )

    conv_log.append(
        ConversationEntry(
            round_num=round_num,
            call_type=call_type,
            agent_id=agent.id,
            target_id=target_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw_response,
            parsed_response=parsed,
        )
    )
    return parsed


def LLM_discuss(agent, round_num, transcript, config, conv_log):
    return _llm_call(
        agent=agent, call_type="discussion", round_num=round_num,
        user_prompt=build_discussion_prompt(round_num, agent, transcript, config),
        default_response={"message": ""}, temperature=0.8, conv_log=conv_log,
    )


def LLM_decide(agent, round_num, group_ids, config, conv_log, transcript=None):
    default = {"contribution": config.ENDOWMENT // 2}
    result = _llm_call(
        agent=agent, call_type="decision", round_num=round_num,
        user_prompt=build_decision_prompt(round_num, agent, group_ids, config, transcript),
        default_response=default, temperature=0.7, conv_log=conv_log,
    )
    c = int(result.get("contribution", config.ENDOWMENT // 2))
    result["contribution"] = max(0, min(config.ENDOWMENT, c))
    return result


def LLM_evaluate(agent, round_num, group_ids, contributions, config, conv_log):
    other_ids = [j for j in group_ids if j != agent.id]
    default = {"evaluations": {str(j): 0.0 for j in other_ids}}
    result = _llm_call(
        agent=agent, call_type="evaluate", round_num=round_num,
        user_prompt=build_evaluation_prompt(round_num, agent, group_ids, contributions, config),
        default_response=default, temperature=0.5, conv_log=conv_log,
    )
    evals_raw = result.get("evaluations", {})
    return {
        int(k): max(-1.0, min(1.0, float(v)))
        for k, v in evals_raw.items()
        if k.isdigit() or (isinstance(k, str) and k.lstrip("-").isdigit())
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
    return _llm_call(
        agent=agent, call_type="perception", round_num=round_num,
        user_prompt=build_perception_prompt(
            round_num, agent, group_ids, contributions, payoffs, config
        ),
        default_response=default, temperature=0.6, conv_log=conv_log,
    )


# ============================================================
# Round execution (shared by both phases)
# ============================================================


def _execute_round(
    t: int,
    phase: str,           # "normal" | "extension"
    condition: str,
    config: SimConfig,
    agents: List[Agent],
    agent_by_id: Dict[int, Agent],
    network: nx.DiGraph,
    round_logs: List[dict],
    conv_log: List[ConversationEntry],
    new_agent_id: Optional[int] = None,
) -> None:
    """Run one round and append its log entry."""

    logger.info(f"[{condition}|phase={phase}] Round {t}")

    # 1. DISCUSSION
    transcript: List[dict] = []
    if config.discussion_on:
        agent_order = list(range(len(agents)))
        random.shuffle(agent_order)
        for _ in range(config.discussion_turns):
            for idx in agent_order:
                a = agents[idx]
                result = LLM_discuss(a, t, transcript, config, conv_log)
                message = str(result.get("message", "")).strip()
                if message:
                    transcript.append({"agent_id": a.id, "message": message})

    # 2. GROUP FORMATION
    excluded_ids: set = set()
    if config.selection_on:
        excluded_ids = get_excluded_agents(agents, network, config)
        eligible = [a for a in agents if a.id not in excluded_ids]
        groups = form_groups(eligible, network, config)
    else:
        ids = [a.id for a in agents]
        random.shuffle(ids)
        groups = [
            ids[i : i + config.GROUP_SIZE]
            for i in range(0, len(ids), config.GROUP_SIZE)
        ]
        groups = [g for g in groups if len(g) >= 2]

    # 3. CONTRIBUTIONS
    contributions: Dict[int, int] = {}
    for group_ids in groups:
        for i in group_ids:
            a = agent_by_id[i]
            result = LLM_decide(a, t, group_ids, config, conv_log, transcript)
            contributions[i] = result["contribution"]

    # 4. PAYOFFS
    payoffs: Dict[int, float] = {}
    for group_ids in groups:
        total = sum(contributions.get(i, 0) for i in group_ids)
        share = (config.MULTIPLIER * total) / len(group_ids)
        for i in group_ids:
            gain = float(config.ENDOWMENT - contributions.get(i, 0)) + share
            agent_by_id[i].material_payoff += gain
            payoffs[i] = gain

    # 5. EVALUATION
    evaluations: Dict[Tuple[int, int], float] = {}
    if config.evaluation_on:
        for group_ids in groups:
            for i in group_ids:
                a = agent_by_id[i]
                ratings = LLM_evaluate(a, t, group_ids, contributions, config, conv_log)
                for j, signal in ratings.items():
                    evaluations[(i, j)] = signal
        network = update_network(network, evaluations, config)

    # 6. PERCEPTION
    perceptions: Dict[int, dict] = {}
    if config.perception_on:
        for group_ids in groups:
            for i in group_ids:
                a = agent_by_id[i]
                perceptions[i] = LLM_perceive(
                    a, t, group_ids, contributions, payoffs, config, conv_log
                )
        network = update_network_from_perception(network, perceptions, config)
        update_tendencies_from_norm(agents, perceptions, config)

    # 7. MEMORY UPDATE
    for a in agents:
        partners: List[int] = []
        for g in groups:
            if a.id in g:
                partners = [x for x in g if x != a.id]
                break

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
                "my_contribution": oldest["my_contribution"],
                "group_avg": oldest["group_avg_contribution"],
                "my_payoff": round(oldest["my_payoff"], 2),
                "injunctive_norm": oldest["norm_reflection"].get("injunctive_norm"),
                "descriptive_norm": oldest["norm_reflection"].get("descriptive_norm"),
            })

    # 8. LOG
    network_snapshot = [
        {"u": u, "v": v, "weight": round(d["weight"], 4)}
        for u, v, d in network.edges(data=True)
    ]

    log_entry = {
        "round": t,
        "phase": phase,
        "condition": condition,
        "discussion_transcript": transcript if config.discussion_on else None,
        "excluded_agents": sorted(excluded_ids),
        "groups": groups,
        "contributions": {str(k): v for k, v in contributions.items()},
        "payoffs": {str(k): round(v, 4) for k, v in payoffs.items()},
        "evaluations": {
            f"{i}->{j}": round(s, 4) for (i, j), s in evaluations.items()
        },
        "network_weights": network_snapshot,
        "agent_states": [
            {
                "id": a.id,
                "cooperation_tendency": a.cooperation_tendency,
                "material_payoff": round(a.material_payoff, 3),
            }
            for a in agents
        ],
        "perceptions": (
            {str(k): v for k, v in perceptions.items()}
            if config.perception_on else None
        ),
    }

    # Mark the round on which the new agent first appears
    if new_agent_id is not None and t == config.ROUNDS + 1:
        log_entry["new_agent_introduced"] = new_agent_id

    round_logs.append(log_entry)


# ============================================================
# Main Simulation
# ============================================================


def run_sim(condition: str, seed: int) -> SimResults:
    random.seed(seed)
    np.random.seed(seed)

    config  = make_config(condition)
    agents  = init_agents(config)
    agent_by_id: Dict[int, Agent] = {a.id: a for a in agents}
    network = init_network(agents)
    round_logs: List[dict] = []
    conv_log: List[ConversationEntry] = []

    # ── Phase 1: normal rounds (1 … ROUNDS) ─────────────────────────────
    for t in range(1, config.ROUNDS + 1):
        _execute_round(
            t=t, phase="normal", condition=condition,
            config=config, agents=agents, agent_by_id=agent_by_id,
            network=network, round_logs=round_logs, conv_log=conv_log,
        )

    # ── New-agent introduction ───────────────────────────────────────────
    logger.info(f"[{condition}|seed={seed}] Introducing new agent after round {config.ROUNDS}.")
    new_agent = introduce_new_agent(agents, agent_by_id, network)
    new_agent_id = new_agent.id

    # ── Phase 2: extension rounds (ROUNDS+1 … ROUNDS+EXTENSION_ROUNDS) ──
    for t in range(config.ROUNDS + 1, config.ROUNDS + config.EXTENSION_ROUNDS + 1):
        _execute_round(
            t=t, phase="extension", condition=condition,
            config=config, agents=agents, agent_by_id=agent_by_id,
            network=network, round_logs=round_logs, conv_log=conv_log,
            new_agent_id=new_agent_id,
        )

    return SimResults(
        condition=condition,
        seed=seed,
        agents=agents,
        network=network,
        round_logs=round_logs,
        conversation_logs=conv_log,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    conditions = ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE", "PURE_BASELINE"]
    seeds = [46, 47, 48, 49, 50, 51, 52]

    out_base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "../results/gpt/global"
    )

    for seed in seeds:
        seed_dir = os.path.join(out_base, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for cond in conditions:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Running NEW-INTRO condition: {cond} | seed={seed}")
            logger.info(f"{'=' * 50}")

            results = run_sim(condition=cond, seed=seed)

            logs_path = os.path.join(
                seed_dir,
                f"log_v5_newintro_{run_timestamp}_{cond}_seed{seed}.json",
            )
            with open(logs_path, "w") as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved simulation logs → {logs_path}")

            conv_path = os.path.join(
                seed_dir,
                f"conversations_v5_newintro_{run_timestamp}_{cond}_seed{seed}.json",
            )
            with open(conv_path, "w") as f:
                json.dump(results.conversations_to_dict(), f, indent=2, default=str)
            logger.info(
                f"Saved {len(results.conversation_logs)} conversation entries → {conv_path}"
            )
