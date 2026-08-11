"""
parameter_sweep.py
------------------
Robustness check: sensitivity of the FULL condition to key SimConfig
parameters, using a one-factor-at-a-time (OFAT) design.

Three parameters are swept independently, each at three levels (low /
default / high). All other parameters are held at their defaults. The
goal is to confirm that cooperation emergence in FULL is not driven by a
specific, cherry-picked parameter configuration.

Parameters swept:
  network_update_rate_neg  : [0.4, 0.8*, 1.2]   (* = default)
    Controls how aggressively negative evaluations lower edge weights.
    This is the most influential parameter in the ostracism mechanism.

  tie_remove_threshold     : [0.05, 0.15*, 0.30]
    Minimum weight below which a directed edge is deleted. Governs how
    easily a reputation link is severed.

  norm_internalization_rate: [0.1, 0.2*, 0.4]
    Rate at which agents update their cooperation tendency toward the
    injunctive norm they perceive. Higher = faster norm adoption.

Condition: FULL (all mechanisms active — the only condition where all
           three parameters are meaningfully engaged simultaneously)
Seeds    : 43–47 (5 seeds)
Backend  : "openai" (GPT-4o-mini) or "local" (vLLM, default: Qwen)

Output: results/robustness/parameter_sweep/{backend}/{param}/{value}/seed{s}/
"""

from __future__ import annotations

import argparse
import json
import logging
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

SEEDS = list(range(43, 48))

# OFAT sweep: parameter name → list of values to test (default marked in docstring)
PARAM_SWEEPS: Dict[str, List[float]] = {
    "network_update_rate_neg":   [0.4, 1.2],    # default 0.8 already covered by main sim
    "tie_remove_threshold":      [0.05, 0.30],   # default 0.15 already covered by main sim
    "norm_internalization_rate": [0.1, 0.4],     # default 0.2 already covered by main sim
}

RATE_PAIR_SWEEPS: List[Dict[str, float]] = [
    {"network_update_rate_pos": 0.5, "network_update_rate_neg": 0.5},
    {"network_update_rate_pos": 0.4, "network_update_rate_neg": 0.6},
    {"network_update_rate_pos": 0.2, "network_update_rate_neg": 0.8},
]

CONDITION = "FULL"

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../results/robustness/parameter_sweep",
)

# ─── Backend globals (populated by _init_backend) ────────────────────────────

BACKEND:     str  = ""
_client           = None
_MODEL_NAME: str  = ""

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
CACHE_DIR = "/data3/models/hub"

_registry: Dict[str, "_LocalModel"] = {}


class _LocalModel:
    def __init__(self, model_name: str):
        from vllm import LLM, SamplingParams  # type: ignore
        self.model_name = model_name
        self._SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_name, download_dir=CACHE_DIR,
            dtype="bfloat16", tensor_parallel_size=1, gpu_memory_utilization=0.9,
        )

    def generate(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        extra = MODEL_SAMPLING.get(self.model_name, {})
        sp    = self._SamplingParams(temperature=temperature, max_tokens=512, **extra)
        out   = self.llm.chat(messages, sp)
        return out[0].outputs[0].text.strip()


def _get_local_model() -> _LocalModel:
    if _MODEL_NAME not in _registry:
        _registry[_MODEL_NAME] = _LocalModel(_MODEL_NAME)
    return _registry[_MODEL_NAME]


def _init_backend(backend: str, local_model_key: str = "qwen", cuda_device: str = "7") -> None:
    global BACKEND, _client, _MODEL_NAME
    BACKEND = backend
    if backend == "openai":
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
        _client     = OpenAI()
        _MODEL_NAME = "gpt-4o-mini"
    elif backend == "local":
        if local_model_key not in AVAILABLE_MODELS:
            raise ValueError(f"Unknown local model {local_model_key!r}. Choose from: {list(AVAILABLE_MODELS)}")
        HF_TOKEN = os.getenv("HF_TOKEN", "")
        os.environ.setdefault("HF_TOKEN", HF_TOKEN)
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        _MODEL_NAME = AVAILABLE_MODELS[local_model_key]
    else:
        raise ValueError(f"Unknown backend {backend!r}. Use 'openai' or 'local'.")


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
    system_prompt: str
    user_prompt: str
    raw_response: str
    parsed_response: dict

    def to_dict(self) -> dict:
        return {
            "round": self.round_num, "call_type": self.call_type,
            "agent_id": self.agent_id, "target_id": self.target_id,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt, "user_prompt": self.user_prompt,
            "raw_response": self.raw_response, "parsed_response": self.parsed_response,
        }


@dataclass
class SimResults:
    condition: str
    seed: int
    param_overrides: Dict[str, float]
    agents: List[Agent]
    network: nx.DiGraph
    round_logs: List[dict]
    conversation_logs: List[ConversationEntry]

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "seed": self.seed,
            "robustness_params": self.param_overrides,
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
            "robustness_params": self.param_overrides,
            "total_calls": len(self.conversation_logs),
            "conversations": [e.to_dict() for e in self.conversation_logs],
        }


# ─── Config factory ──────────────────────────────────────────────────────────

def make_config(param_overrides: Dict[str, float] = None) -> SimConfig:
    """Return a FULL SimConfig with zero or more parameters overridden."""
    cfg = SimConfig()
    for name, value in (param_overrides or {}).items():
        if not hasattr(cfg, name):
            raise ValueError(f"SimConfig has no attribute: {name!r}")
        setattr(cfg, name, value)
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


# ─── Network / group helpers ─────────────────────────────────────────────────

def form_groups(agents: List[Agent], network: nx.DiGraph, config: SimConfig) -> List[List[int]]:
    unassigned = list(set(a.id for a in agents))
    groups: List[List[int]] = []
    while len(unassigned) >= config.GROUP_SIZE:
        sw   = np.array([
            np.mean([
                network[j][aid]["weight"] if network.has_edge(j, aid) else 1e-4
                for j in unassigned if j != aid
            ])
            for aid in unassigned
        ])
        sp      = sw / sw.sum()
        seed_id = int(np.random.choice(unassigned, p=sp))
        cands   = [j for j in unassigned if j != seed_id]
        pw      = np.array([
            network[seed_id][j]["weight"] if network.has_edge(seed_id, j) else 1e-4
            for j in cands
        ])
        pw     /= pw.sum()
        n_pick  = min(config.GROUP_SIZE - 1, len(cands))
        chosen  = [int(x) for x in np.random.choice(cands, size=n_pick, replace=False, p=pw)]
        group   = [seed_id] + chosen
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


def build_discussion_prompt(t, agent, transcript, config):
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


def build_decision_prompt(t, agent, group_ids, config, transcript=None):
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


def build_evaluation_prompt(t, agent, group_ids, contributions, config):
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


def build_perception_prompt(t, agent, group_ids, contributions, payoffs, config):
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

def _llm_call(agent, call_type, round_num, user_prompt, default_response, temperature, conv_log, target_id=None):
    sys_prompt   = get_system_prompt(agent)
    raw_response = ""
    parsed       = dict(default_response)

    try:
        if BACKEND == "openai":
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
        model_name=_MODEL_NAME,
        system_prompt=sys_prompt, user_prompt=user_prompt,
        raw_response=raw_response, parsed_response=parsed,
    ))
    return parsed


def LLM_discuss(agent, round_num, transcript, config, conv_log):
    return _llm_call(
        agent=agent, call_type="discussion", round_num=round_num,
        user_prompt=build_discussion_prompt(round_num, agent, transcript, config),
        default_response={"message": ""}, temperature=0.8, conv_log=conv_log,
    )


def LLM_decide(agent, round_num, group_ids, config, conv_log, transcript=None):
    default = {"contribution": config.ENDOWMENT // 2}
    result  = _llm_call(
        agent=agent, call_type="decision", round_num=round_num,
        user_prompt=build_decision_prompt(round_num, agent, group_ids, config, transcript),
        default_response=default, temperature=0.7, conv_log=conv_log,
    )
    c = int(result.get("contribution", config.ENDOWMENT // 2))
    result["contribution"] = max(0, min(config.ENDOWMENT, c))
    return result


def LLM_evaluate(agent, round_num, group_ids, contributions, config, conv_log):
    other_ids = [j for j in group_ids if j != agent.id]
    default   = {"evaluations": {str(j): 0.0 for j in other_ids}}
    result    = _llm_call(
        agent=agent, call_type="evaluate", round_num=round_num,
        user_prompt=build_evaluation_prompt(round_num, agent, group_ids, contributions, config),
        default_response=default, temperature=0.5, conv_log=conv_log,
    )
    evals_raw = result.get("evaluations", {})
    return {
        int(k): max(-1.0, min(1.0, float(v)))
        for k, v in evals_raw.items()
        if str(k).lstrip("-").isdigit()
    }


def LLM_perceive(agent, round_num, group_ids, contributions, payoffs, config, conv_log):
    default = {
        "injunctive_norm": None, "descriptive_norm": None,
        "expectation_of_others": "", "others_expectation_of_me": "",
        "preferred_partners": [], "agents_to_avoid": [],
    }
    return _llm_call(
        agent=agent, call_type="perception", round_num=round_num,
        user_prompt=build_perception_prompt(round_num, agent, group_ids, contributions, payoffs, config),
        default_response=default, temperature=0.6, conv_log=conv_log,
    )


# ─── Simulation ──────────────────────────────────────────────────────────────

def run_sim(seed: int, param_overrides: Dict[str, float] = None) -> SimResults:
    random.seed(seed)
    np.random.seed(seed)

    param_overrides = param_overrides or {}
    config      = make_config(param_overrides)
    agents      = init_agents(config)
    agent_by_id = {a.id: a for a in agents}
    network     = init_network(agents)
    round_logs: List[dict]             = []
    conv_log:   List[ConversationEntry] = []

    sweep_label = ", ".join(f"{k}={v}" for k, v in param_overrides.items()) or "default"

    for t in range(1, config.ROUNDS + 1):
        logger.info(f"[FULL|{sweep_label}|seed={seed}] Round {t}/{config.ROUNDS}")

        # 1. Discussion
        transcript: List[dict] = []
        agent_order = list(range(config.N))
        for _ in range(config.discussion_turns):
            random.shuffle(agent_order)
            for aid in agent_order:
                result  = LLM_discuss(agent_by_id[aid], t, transcript, config, conv_log)
                message = str(result.get("message", "")).strip()
                if message:
                    transcript.append({"agent_id": aid, "message": message})

        # 2. Group formation
        excluded_ids = get_excluded_agents(agents, network, config)
        eligible     = [a for a in agents if a.id not in excluded_ids]
        groups       = form_groups(eligible, network, config)

        # 3. Decisions
        contributions: Dict[int, int] = {}
        for group_ids in groups:
            for i in group_ids:
                result = LLM_decide(agent_by_id[i], t, group_ids, config, conv_log, transcript)
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

        # 5. Evaluation
        evaluations: Dict[Tuple[int, int], float] = {}
        for group_ids in groups:
            for i in group_ids:
                ratings = LLM_evaluate(agent_by_id[i], t, group_ids, contributions, config, conv_log)
                for j, signal in ratings.items():
                    evaluations[(i, j)] = signal
        network = update_network(network, evaluations, config)
        update_tendencies(agents, evaluations, config)

        # 6. Perception
        perceptions: Dict[int, dict] = {}
        for group_ids in groups:
            for i in group_ids:
                perceptions[i] = LLM_perceive(
                    agent_by_id[i], t, group_ids, contributions, payoffs, config, conv_log
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
            "condition": CONDITION,
            "robustness_params": param_overrides,
            "discussion_transcript": transcript,
            "excluded_agents": sorted(excluded_ids),
            "groups": groups,
            "contributions": {str(k): v for k, v in contributions.items()},
            "payoffs": {str(k): round(v, 4) for k, v in payoffs.items()},
            "evaluations": {f"{i}->{j}": round(s, 4) for (i, j), s in evaluations.items()},
            "network_weights": [
                {"u": u, "v": v, "weight": round(d["weight"], 4)}
                for u, v, d in network.edges(data=True)
            ],
            "agent_states": [
                {"id": a.id, "cooperation_tendency": a.cooperation_tendency,
                 "material_payoff": round(a.material_payoff, 3)}
                for a in agents
            ],
            "perceptions": {str(k): v for k, v in perceptions.items()},
        })

    return SimResults(
        condition=CONDITION, seed=seed,
        param_overrides=param_overrides,
        agents=agents, network=network,
        round_logs=round_logs, conversation_logs=conv_log,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OFAT parameter sweep for the FULL condition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend", "-b", choices=["openai", "local"], default="openai",
        help="LLM backend to use.",
    )
    parser.add_argument(
        "--model", "-m", choices=list(AVAILABLE_MODELS), default="qwen",
        help="Local model key (only used when --backend=local).",
    )
    parser.add_argument(
        "--seeds", "-s", nargs="+", type=int, default=SEEDS, metavar="SEED",
        help="Seeds to run.",
    )
    parser.add_argument(
        "--param", "-p", choices=list(PARAM_SWEEPS), default=None, metavar="PARAM",
        help="Restrict sweep to a single parameter (default: all).",
    )
    parser.add_argument(
        "--value", "-v", type=float, default=None, metavar="VALUE",
        help="Restrict sweep to a single value for --param (default: all values).",
    )
    parser.add_argument(
        "--cuda-device", default="7", metavar="N",
        help="CUDA device index passed to CUDA_VISIBLE_DEVICES (local backend only).",
    )
    parser.add_argument(
        "--pairs", action="store_true",
        help="Run the (network_update_rate_pos, network_update_rate_neg) pair sweep instead of OFAT.",
    )
    return parser.parse_args()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _parse_args()
    _init_backend(args.backend, args.model, args.cuda_device)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_label = "openai" if BACKEND == "openai" else args.model

    def _val_label(v: float) -> str:
        return str(v).replace(".", "p")

    def _save(results: SimResults, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        log_path  = os.path.join(out_dir, f"log_{run_timestamp}_FULL_seed{results.seed}.json")
        conv_path = os.path.join(out_dir, f"conversations_{run_timestamp}_FULL_seed{results.seed}.json")
        with open(log_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2, default=str)
        with open(conv_path, "w") as f:
            json.dump(results.conversations_to_dict(), f, indent=2, default=str)
        logger.info(f"Saved → {log_path}")

    if args.pairs:
        for overrides in RATE_PAIR_SWEEPS:
            pos = overrides["network_update_rate_pos"]
            neg = overrides["network_update_rate_neg"]
            pair_label = f"pos{_val_label(pos)}_neg{_val_label(neg)}"
            for seed in args.seeds:
                logger.info(f"\n{'='*50}")
                logger.info(f"rate pair pos={pos} neg={neg} | seed={seed}")
                logger.info(f"{'='*50}")
                results = run_sim(seed=seed, param_overrides=overrides)
                _save(results, os.path.join(RESULTS_DIR, backend_label, "rate_pairs", pair_label, f"seed{seed}"))
    else:
        sweep = {
            k: v for k, v in PARAM_SWEEPS.items()
            if args.param is None or k == args.param
        }
        if args.param and args.param not in PARAM_SWEEPS:
            raise SystemExit(f"Unknown param {args.param!r}. Choices: {list(PARAM_SWEEPS)}")

        for param_name, values in sweep.items():
            if args.value is not None:
                if args.value not in values:
                    raise SystemExit(
                        f"Value {args.value} not in sweep for {param_name!r}. Options: {values}"
                    )
                values = [args.value]
            for param_value in values:
                for seed in args.seeds:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"param={param_name} value={param_value} | seed={seed}")
                    logger.info(f"{'='*50}")
                    results = run_sim(seed=seed, param_overrides={param_name: param_value})
                    _save(results, os.path.join(
                        RESULTS_DIR, backend_label, param_name, _val_label(param_value), f"seed{seed}"
                    ))
