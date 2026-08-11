"""
prompt_variants.py
------------------
Robustness check: sensitivity of baseline and full-model behavior to system
prompt framing.

Three system prompt variants are tested:

  standard      — current prompt: cooperation_tendency shown as an explicit
                  numeric value (0.0–1.0). This is what the main sim uses.

  minimal       — bare-bones: only the game role is described. No tendency,
                  no personality framing. Tests whether the LLM's default
                  behavior (without any internal-state guidance) differs
                  systematically from the standard condition.

  tendency_free — same structure as standard but the tendency is described
                  qualitatively ("you lean toward cooperation / self-interest")
                  rather than numerically. Tests whether the numeric value
                  drives behavior, or whether the framing alone is sufficient.

If PURE_BASELINE cooperation is consistent across all three variants, the
results are not an artifact of the prompt framing. If FULL consistently
outperforms PURE_BASELINE across all three, the social mechanisms are robust
to the system prompt used.

Conditions : PURE_BASELINE, FULL
Variants   : standard, minimal, tendency_free
Seeds      : 43–47 (5 seeds)
Backend    : "openai" (GPT-4o-mini) or "local" (vLLM, default: Qwen)

Output: results/robustness/prompt_variants/{backend}/{variant}/seed{s}/
"""

from __future__ import annotations

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Robustness config ────────────────────────────────────────────────────────

PROMPT_VARIANTS = ["standard", "minimal", "tendency_free"]
CONDITIONS      = ["PURE_BASELINE", "FULL"]
ALL_SEEDS       = list(range(43, 48))

AVAILABLE_MODELS = {
    "llama":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
}
MODEL_SAMPLING = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"top_p": 0.9,  "repetition_penalty": 1.05},
    "mistralai/Mistral-7B-Instruct-v0.3":     {"top_p": 0.92, "repetition_penalty": 1.08},
    "Qwen/Qwen2.5-7B-Instruct":               {"top_p": 0.9,  "repetition_penalty": 1.1},
}

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../results/robustness/prompt_variants",
)

# ─── Backend state (populated by _init_backend before run_sim is called) ─────

_BACKEND    = None
_MODEL_NAME = None
_client     = None
_registry: dict = {}


def _init_backend(backend: str, model_key: str = "qwen", device: int = 7) -> None:
    """Configure the LLM backend. Must be called once before run_sim."""
    global _BACKEND, _MODEL_NAME, _client

    _BACKEND = backend

    if backend == "openai":
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
        _client     = OpenAI()
        _MODEL_NAME = "gpt-4o-mini"

    elif backend == "local":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))
        _MODEL_NAME = AVAILABLE_MODELS[model_key]
        logger.info(f"Backend: local | model={model_key} | device={device}")

    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'openai' or 'local'.")


def _get_local_model():
    """Lazy vLLM model loader — imports and instantiates on first call."""
    from vllm import LLM, SamplingParams  # type: ignore  # noqa: F401

    class _LocalModel:
        def __init__(self, model_name: str):
            self.model_name = model_name
            self.llm = LLM(
                model=model_name,
                download_dir="/data3/models/hub",
                dtype="bfloat16",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
            )

        def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
            extra = MODEL_SAMPLING.get(self.model_name, {})
            sp    = SamplingParams(temperature=temperature, max_tokens=512, **extra)
            return self.llm.chat(messages, sp)[0].outputs[0].text.strip()

    if _MODEL_NAME not in _registry:
        _registry[_MODEL_NAME] = _LocalModel(_MODEL_NAME)
    return _registry[_MODEL_NAME]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


# ─── System prompt variants ───────────────────────────────────────────────────
# Each variant is a string template. {tendency} is substituted for standard and
# tendency_free; minimal ignores it entirely.

_STANDARD = """\
You are an autonomous agent in a repeated group interaction.

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

_MINIMAL = """\
You are an agent in a repeated public goods game.
Each round you decide how much to contribute to a shared group fund.
Make your decision based on the information provided.
Output STRICT JSON only with the exact keys requested. No extra text."""

# Same structure as standard but tendency is qualitative, not numeric.
# Tests whether the number itself drives behavior vs. just the framing.
_TENDENCY_FREE = """\
You are an autonomous agent in a repeated group interaction.

You make decisions about contributing to a shared group fund.

You have a personal disposition toward {tendency_label}. This disposition
influences your decisions but does not rigidly determine them. It may shift
over time based on group experiences.

You may adjust your behavior based on:
- what others contributed
- how others evaluate behavior
- what seems to be expected in the group
- what outcomes your past actions produced.

Over time your experience may reinforce or weaken cooperative behavior,
shaping how strongly you follow or resist group expectations.

Output STRICT JSON only with the exact keys requested. No extra text."""


def _tendency_label(tendency: float) -> str:
    """Map numeric tendency to a qualitative label for the tendency_free prompt."""
    if tendency >= 0.75:
        return "cooperation — you strongly prioritize fairness and group wellbeing"
    elif tendency >= 0.5:
        return "moderate cooperation — you balance self-interest and group outcomes"
    elif tendency >= 0.25:
        return "mild self-interest — you lean toward keeping more for yourself"
    else:
        return "self-interest — you strongly prioritize your own material payoff"


def get_system_prompt(agent: "Agent", variant: str) -> str:
    if variant == "standard":
        return _STANDARD.format(tendency=agent.cooperation_tendency)
    elif variant == "minimal":
        return _MINIMAL
    elif variant == "tendency_free":
        return _TENDENCY_FREE.format(tendency_label=_tendency_label(agent.cooperation_tendency))
    else:
        raise ValueError(f"Unknown prompt variant: {variant!r}")


# ─── Data structures ─────────────────────────────────────────────────────────

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
    model_name: str
    prompt_variant: str
    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed_response: dict

    def to_dict(self) -> dict:
        return {
            "round": self.round_num, "call_type": self.call_type,
            "agent_id": self.agent_id, "target_id": self.target_id,
            "model_name": self.model_name, "prompt_variant": self.prompt_variant,
            "system_prompt": self.system_prompt, "user_prompt": self.user_prompt,
            "raw_response": self.raw_response, "parsed_response": self.parsed_response,
        }


@dataclass
class SimResults:
    condition: str
    seed: int
    prompt_variant: str
    agents: List[Agent]
    network: nx.DiGraph
    round_logs: List[dict]
    conversation_logs: List[ConversationEntry]

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "robustness_params": {"prompt_variant": self.prompt_variant},
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
            "robustness_params": {"prompt_variant": self.prompt_variant},
            "total_calls": len(self.conversation_logs),
            "conversations": [e.to_dict() for e in self.conversation_logs],
        }


# ─── Conditions ──────────────────────────────────────────────────────────────

def make_config(condition: str) -> SimConfig:
    cfg = SimConfig()
    if condition == "PURE_BASELINE":
        cfg.discussion_on = False
        cfg.selection_on  = False
        cfg.evaluation_on = False
        cfg.perception_on = False
    elif condition == "FULL":
        pass  # all mechanisms on by default
    else:
        raise ValueError(f"prompt_variants only supports PURE_BASELINE and FULL, got: {condition!r}")
    return cfg


# ─── Init ─────────────────────────────────────────────────────────────────────

def init_agents(config: SimConfig) -> List[Agent]:
    return [
        Agent(id=i, cooperation_tendency=round(random.uniform(0.0, 1.0), 2))
        for i in range(config.N)
    ]


def init_network(agents: List[Agent]) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(a.id for a in agents)
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i != j:
                G.add_edge(agents[i].id, agents[j].id, weight=1.0)
    return G


# ─── Network helpers ─────────────────────────────────────────────────────────

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
        candidates = [j for j in unassigned if j != seed_id]
        pw         = np.array([
            network[seed_id][j]["weight"] if network.has_edge(seed_id, j) else 1e-4
            for j in candidates
        ])
        pw         /= pw.sum()
        n_pick      = min(config.GROUP_SIZE - 1, len(candidates))
        chosen      = [int(x) for x in np.random.choice(candidates, size=n_pick, replace=False, p=pw)]
        group       = [seed_id] + chosen
        for g in group:
            unassigned.remove(g)
        groups.append(group)
    if unassigned:
        groups.append(unassigned)
    return [g for g in groups if len(g) >= 2]


def get_excluded_agents(agents: List[Agent], network: nx.DiGraph, config: SimConfig) -> set:
    excluded = set()
    all_ids  = [a.id for a in agents]
    for a in agents:
        others = [j for j in all_ids if j != a.id]
        if not others:
            continue
        avg_in = np.mean([
            network[j][a.id]["weight"] if network.has_edge(j, a.id) else 0.0
            for j in others
        ])
        if avg_in < config.participation_threshold:
            excluded.add(a.id)
    return excluded


def update_network(
    network: nx.DiGraph, evaluations: Dict[Tuple[int, int], float], config: SimConfig,
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
    agents: List[Agent], evaluations: Dict[Tuple[int, int], float], config: SimConfig,
) -> None:
    incoming: Dict[int, List[float]] = {}
    for (i, j), signal in evaluations.items():
        incoming.setdefault(j, []).append(signal)
    for agent in agents:
        signals = incoming.get(agent.id)
        if not signals:
            continue
        mean_sig = sum(signals) / len(signals)
        rate     = config.tendency_learning_rate_neg if mean_sig < 0 else config.tendency_learning_rate_pos
        agent.cooperation_tendency = round(
            max(0.0, min(1.0, agent.cooperation_tendency + rate * mean_sig)), 4
        )


def update_network_from_perception(
    network: nx.DiGraph, perceptions: Dict[int, dict], config: SimConfig,
) -> nx.DiGraph:
    for agent_id, perc in perceptions.items():
        def _ids(raw):
            out = []
            for x in (raw if isinstance(raw, list) else []):
                try:
                    out.append(int(x))
                except (ValueError, TypeError):
                    pass
            return out
        for j in _ids(perc.get("preferred_partners", [])):
            if network.has_edge(agent_id, j):
                network[agent_id][j]["weight"] = min(
                    2.0, network[agent_id][j]["weight"] + config.perception_update_rate_pos
                )
            else:
                network.add_edge(agent_id, j, weight=config.perception_update_rate_pos)
        for j in _ids(perc.get("agents_to_avoid", [])):
            if network.has_edge(agent_id, j):
                new_w = max(0.0, network[agent_id][j]["weight"] - config.perception_update_rate_neg)
                if new_w < config.tie_remove_threshold:
                    network.remove_edge(agent_id, j)
                else:
                    network[agent_id][j]["weight"] = new_w
    return network


def update_tendencies_from_norm(
    agents: List[Agent], perceptions: Dict[int, dict], config: SimConfig,
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
                agent.cooperation_tendency + config.norm_internalization_rate * (target - agent.cooperation_tendency)
            )), 4,
        )


# ─── Prompts ─────────────────────────────────────────────────────────────────

def build_discussion_prompt(
    t: int, agent: Agent, transcript: List[dict], config: SimConfig,
) -> str:
    transcript_str = (
        "\n".join(f"  Agent {m['agent_id']}: {m['message']}" for m in transcript)
        if transcript else "  (No messages yet — you go first.)"
    )
    memory_str = json.dumps(agent.memory[-config.history_window:], default=str) \
        if agent.memory else "  (No prior rounds yet.)"
    return (
        f"Round: {t}\n"
        f"Task: Group discussion before partner selection and contribution.\n\n"
        f"Your recent observations and reflections:\n{memory_str}\n\n"
        f"Discussion so far:\n{transcript_str}\n\n"
        f"Share what you think the group should do this round and why. "
        f"Refer to what others said if relevant. Keep it to 1–3 sentences.\n\n"
        f'Output JSON with EXACT keys:\n{{\n  "message": "<your statement>"\n}}'
    )


def build_decision_prompt(
    t: int, agent: Agent, group_ids: List[int], config: SimConfig,
    transcript: List[dict] = None,
) -> str:
    memory_str  = json.dumps(agent.memory[-config.history_window:], default=str) \
        if agent.memory else "  (No prior rounds yet.)"
    summary_str = json.dumps(agent.memory_summary, default=str) \
        if agent.memory_summary else "  (No earlier rounds.)"
    transcript_str = (
        "\n".join(f"  Agent {m['agent_id']}: {m['message']}" for m in transcript)
        if transcript else "  (No discussion this round.)"
    )
    if agent.memory:
        nr = agent.memory[-1].get("norm_reflection", {})
        perception_str = (
            f"  Injunctive norm: {nr.get('injunctive_norm', 'unknown')}\n"
            f"  Descriptive norm: {nr.get('descriptive_norm', 'unknown')}\n"
            f"  What you expect from others: {nr.get('expectation_of_others', 'unknown')}\n"
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
        f'{{\n  "contribution": <int 0-{config.ENDOWMENT}>\n}}'
    )


def build_evaluation_prompt(
    t: int, agent: Agent, group_ids: List[int],
    contributions: Dict[int, int], config: SimConfig,
) -> str:
    other_ids     = [j for j in group_ids if j != agent.id]
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
        f'Output JSON with EXACT keys:\n{{\n  "evaluations": {{{ id_list }}}\n}}'
    )


def build_perception_prompt(
    t: int, agent: Agent, group_ids: List[int],
    contributions: Dict[int, int], payoffs: Dict[int, float], config: SimConfig,
) -> str:
    other_ids     = [j for j in group_ids if j != agent.id]
    contrib_lines = "\n".join(
        f"  Agent {j}: contributed {contributions.get(j, '?')} / {config.ENDOWMENT}"
        for j in other_ids
    )
    avg         = sum(contributions.get(j, 0) for j in group_ids) / len(group_ids)
    memory_str  = json.dumps(agent.memory[-config.history_window:], default=str) \
        if agent.memory else "  (No prior rounds yet.)"
    summary_str = json.dumps(agent.memory_summary, default=str) \
        if agent.memory_summary else "  (No earlier rounds.)"
    return (
        f"Round {t} — Post-round reflection.\n\n"
        f"Your group: {group_ids}\n"
        f"Your contribution: {contributions.get(agent.id, '?')} / {config.ENDOWMENT}\n"
        f"Partner contributions:\n{contrib_lines}\n"
        f"Group average: {avg:.2f} / {config.ENDOWMENT}\n"
        f"Your payoff this round: {payoffs.get(agent.id, 0.0):.2f}\n\n"
        f"Earlier rounds summary:\n{summary_str}\n\n"
        f"Recent rounds (detailed):\n{memory_str}\n\n"
        f"Output JSON with EXACT keys:\n"
        f'{{\n'
        f'  "injunctive_norm": <what contribution level do you think members OUGHT to make? (0–{config.ENDOWMENT})>,\n'
        f'  "descriptive_norm": <what do members TYPICALLY contribute? (0–{config.ENDOWMENT})>,\n'
        f'  "expectation_of_others": "<what do you expect from your partners next round?>",\n'
        f'  "others_expectation_of_me": "<what do others expect from you?>",\n'
        f'  "preferred_partners": [<list of agent IDs you prefer to work with>],\n'
        f'  "agents_to_avoid": [<list of agent IDs you prefer to avoid>]\n'
        f'}}'
    )


# ─── LLM calls ───────────────────────────────────────────────────────────────

def _llm_call(
    agent: Agent,
    call_type: str,
    round_num: int,
    user_prompt: str,
    default_response: dict,
    temperature: float,
    conv_log: List[ConversationEntry],
    prompt_variant: str,
    target_id: Optional[int] = None,
) -> dict:
    sys_prompt   = get_system_prompt(agent, prompt_variant)
    raw_response = ""
    parsed       = dict(default_response)

    try:
        if _BACKEND == "openai":
            resp         = _client.chat.completions.create(
                model=_MODEL_NAME,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            raw_response = resp.choices[0].message.content
            parsed       = json.loads(raw_response)
        else:
            raw_response = _get_local_model().generate(sys_prompt, user_prompt, temperature)
            parsed       = _extract_json(raw_response)
    except Exception as e:
        logger.warning(f"LLM call type={call_type} agent={agent.id} round={round_num}: {e}")

    conv_log.append(ConversationEntry(
        round_num=round_num, call_type=call_type,
        agent_id=agent.id, target_id=target_id,
        model_name=_MODEL_NAME, prompt_variant=prompt_variant,
        system_prompt=sys_prompt, user_prompt=user_prompt,
        raw_response=raw_response, parsed_response=parsed,
    ))
    return parsed


def LLM_discuss(agent, round_num, transcript, config, conv_log, prompt_variant):
    return _llm_call(
        agent=agent, call_type="discussion", round_num=round_num,
        user_prompt=build_discussion_prompt(round_num, agent, transcript, config),
        default_response={"message": ""}, temperature=0.8,
        conv_log=conv_log, prompt_variant=prompt_variant,
    )


def LLM_decide(agent, round_num, group_ids, config, conv_log, prompt_variant, transcript=None):
    default = {"contribution": config.ENDOWMENT // 2}
    result  = _llm_call(
        agent=agent, call_type="decision", round_num=round_num,
        user_prompt=build_decision_prompt(round_num, agent, group_ids, config, transcript),
        default_response=default, temperature=0.7,
        conv_log=conv_log, prompt_variant=prompt_variant,
    )
    c = int(result.get("contribution", config.ENDOWMENT // 2))
    result["contribution"] = max(0, min(config.ENDOWMENT, c))
    return result


def LLM_evaluate(agent, round_num, group_ids, contributions, config, conv_log, prompt_variant):
    other_ids = [j for j in group_ids if j != agent.id]
    default   = {"evaluations": {str(j): 0.0 for j in other_ids}}
    result    = _llm_call(
        agent=agent, call_type="evaluate", round_num=round_num,
        user_prompt=build_evaluation_prompt(round_num, agent, group_ids, contributions, config),
        default_response=default, temperature=0.5,
        conv_log=conv_log, prompt_variant=prompt_variant,
    )
    evals_raw = result.get("evaluations", {})
    return {
        int(k): max(-1.0, min(1.0, float(v)))
        for k, v in evals_raw.items()
        if str(k).lstrip("-").isdigit()
    }


def LLM_perceive(agent, round_num, group_ids, contributions, payoffs, config, conv_log, prompt_variant):
    default = {
        "injunctive_norm": None, "descriptive_norm": None,
        "expectation_of_others": "", "others_expectation_of_me": "",
        "preferred_partners": [], "agents_to_avoid": [],
    }
    return _llm_call(
        agent=agent, call_type="perception", round_num=round_num,
        user_prompt=build_perception_prompt(
            round_num, agent, group_ids, contributions, payoffs, config
        ),
        default_response=default, temperature=0.6,
        conv_log=conv_log, prompt_variant=prompt_variant,
    )


# ─── Simulation ──────────────────────────────────────────────────────────────

def run_sim(condition: str, seed: int, prompt_variant: str = "standard") -> SimResults:
    random.seed(seed)
    np.random.seed(seed)

    config      = make_config(condition)
    agents      = init_agents(config)
    agent_by_id = {a.id: a for a in agents}
    network     = init_network(agents)
    round_logs: List[dict]             = []
    conv_log:   List[ConversationEntry] = []

    for t in range(1, config.ROUNDS + 1):
        logger.info(f"[{condition}|{prompt_variant}|seed={seed}] Round {t}/{config.ROUNDS}")

        # 1. Discussion (FULL only)
        transcript: List[dict] = []
        if config.discussion_on:
            agent_order = list(range(config.N))
            for _ in range(config.discussion_turns):
                random.shuffle(agent_order)
                for aid in agent_order:
                    result  = LLM_discuss(agent_by_id[aid], t, transcript, config, conv_log, prompt_variant)
                    message = str(result.get("message", "")).strip()
                    if message:
                        transcript.append({"agent_id": aid, "message": message})

        # 2. Group formation
        excluded_ids: set = set()
        if config.selection_on:
            excluded_ids = get_excluded_agents(agents, network, config)
            eligible     = [a for a in agents if a.id not in excluded_ids]
            groups       = form_groups(eligible, network, config)
        else:
            ids    = [a.id for a in agents]
            random.shuffle(ids)
            groups = [ids[i: i + config.GROUP_SIZE] for i in range(0, len(ids), config.GROUP_SIZE)]
            groups = [g for g in groups if len(g) >= 2]

        # 3. Decisions
        contributions: Dict[int, int] = {}
        for group_ids in groups:
            for i in group_ids:
                result = LLM_decide(
                    agent_by_id[i], t, group_ids, config, conv_log, prompt_variant, transcript
                )
                contributions[i] = result["contribution"]

        # 4. Payoffs
        payoffs: Dict[int, float] = {}
        for group_ids in groups:
            total = sum(contributions.get(i, 0) for i in group_ids)
            share = (config.MULTIPLIER * total) / len(group_ids)
            for i in group_ids:
                gain = float(config.ENDOWMENT - contributions.get(i, 0)) + share
                agent_by_id[i].material_payoff += gain
                payoffs[i] = gain

        # 5. Evaluation (FULL only)
        evaluations: Dict[Tuple[int, int], float] = {}
        if config.evaluation_on:
            for group_ids in groups:
                for i in group_ids:
                    ratings = LLM_evaluate(
                        agent_by_id[i], t, group_ids, contributions, config, conv_log, prompt_variant
                    )
                    for j, signal in ratings.items():
                        evaluations[(i, j)] = signal
            network = update_network(network, evaluations, config)
            update_tendencies(agents, evaluations, config)

        # 6. Perception (FULL only)
        perceptions: Dict[int, dict] = {}
        if config.perception_on:
            for group_ids in groups:
                for i in group_ids:
                    perceptions[i] = LLM_perceive(
                        agent_by_id[i], t, group_ids, contributions, payoffs,
                        config, conv_log, prompt_variant
                    )
            network = update_network_from_perception(network, perceptions, config)
            update_tendencies_from_norm(agents, perceptions, config)

        # 7. Memory update
        for a in agents:
            partners = []
            for g in groups:
                if a.id in g:
                    partners = [x for x in g if x != a.id]
                    break
            partner_contribs = {str(j): contributions.get(j) for j in partners}
            avg = (
                sum(contributions.get(x, 0) for x in partners + [a.id]) / (len(partners) + 1)
                if (partners or a.id in contributions) else None
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

        round_logs.append({
            "round": t,
            "condition": condition,
            "robustness_params": {"prompt_variant": prompt_variant},
            "discussion_transcript": transcript if config.discussion_on else None,
            "excluded_agents": sorted(excluded_ids),
            "groups": groups,
            "contributions": {str(k): v for k, v in contributions.items()},
            "payoffs": {str(k): round(v, 4) for k, v in payoffs.items()},
            "evaluations": {
                f"{i}->{j}": round(s, 4) for (i, j), s in evaluations.items()
            },
            "network_weights": [
                {"u": u, "v": v, "weight": round(d["weight"], 4)}
                for u, v, d in network.edges(data=True)
            ],
            "agent_states": [
                {"id": a.id, "cooperation_tendency": a.cooperation_tendency,
                 "material_payoff": round(a.material_payoff, 3)}
                for a in agents
            ],
            "perceptions": {str(k): v for k, v in perceptions.items()} if config.perception_on else None,
        })

    return SimResults(
        condition=condition, seed=seed, prompt_variant=prompt_variant,
        agents=agents, network=network,
        round_logs=round_logs, conversation_logs=conv_log,
    )


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run prompt variant robustness tests."
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["openai", "local"],
        default="openai",
        help="LLM backend to use (default: openai).",
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(AVAILABLE_MODELS.keys()),
        default="qwen",
        help="Local model key — only used when --backend local (default: qwen).",
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=7,
        metavar="GPU",
        help="CUDA device index — only used when --backend local (default: 7).",
    )
    parser.add_argument(
        "--variants", "-v",
        nargs="+",
        choices=PROMPT_VARIANTS,
        default=PROMPT_VARIANTS,
        metavar="VARIANT",
        help=f"Prompt variant(s) to run (default: all). Choices: {PROMPT_VARIANTS}.",
    )
    parser.add_argument(
        "--conditions", "-c",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
        metavar="COND",
        help="Condition(s) to run (default: all).",
    )
    parser.add_argument(
        "--seeds", "-s",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help="Explicit seed list. Overrides --n.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=len(ALL_SEEDS),
        metavar="N",
        help=f"Number of seeds from the default list (default: {len(ALL_SEEDS)}).",
    )
    args = parser.parse_args()

    seeds         = args.seeds if args.seeds is not None else ALL_SEEDS[:args.n]
    variants      = args.variants
    conditions    = args.conditions
    backend_label = "openai" if args.backend == "openai" else args.model

    _init_backend(args.backend, args.model, args.device)

    logger.info(f"Backend  : {args.backend}" + (f" ({args.model}, GPU {args.device})" if args.backend == "local" else ""))
    logger.info(f"Variants : {variants}")
    logger.info(f"Conds    : {conditions}")
    logger.info(f"Seeds    : {seeds}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for variant in variants:
        for seed in seeds:
            for cond in conditions:
                logger.info(f"\n{'='*50}")
                logger.info(f"variant={variant} | condition={cond} | seed={seed}")
                logger.info(f"{'='*50}")

                results = run_sim(condition=cond, seed=seed, prompt_variant=variant)

                out_dir = os.path.join(RESULTS_DIR, backend_label, variant, f"seed{seed}")
                os.makedirs(out_dir, exist_ok=True)

                log_path  = os.path.join(out_dir, f"log_{run_timestamp}_{cond}_seed{seed}.json")
                conv_path = os.path.join(out_dir, f"conversations_{run_timestamp}_{cond}_seed{seed}.json")

                with open(log_path, "w") as f:
                    json.dump(results.to_dict(), f, indent=2, default=str)
                with open(conv_path, "w") as f:
                    json.dump(results.conversations_to_dict(), f, indent=2, default=str)

                logger.info(f"Saved → {log_path}")
