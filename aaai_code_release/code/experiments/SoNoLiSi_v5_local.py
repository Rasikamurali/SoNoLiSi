# ============================================================
# SoNoLiSi: Social Norm Learning and Selection
# Group-scoped multi-agent LLM simulation of norm emergence in a
# repeated public-goods game (v5, local)
#
# ENVIRONMENT. N LLM-driven agents repeatedly choose a contribution
# c in [0, ENDOWMENT] to a shared group fund. Each round, agents are
# partitioned into groups of GROUP_SIZE; a group's fund (the sum of
# its members' contributions, scaled by a multiplier) is split evenly
# among the group, so a member's payoff is (ENDOWMENT - c) + share.
# No agent is given an explicit belief-state or norm variable to
# reason over directly: normative/empirical expectations (NE/EE) and
# partner reputations are elicited from, and act back on, free-text
# LLM behavior only. Group formation happens first each round, and every
# subsequent step (discussion, evaluation, perception) is scoped to
# that round's group: agents only see and respond to their own
# groupmates, not the full population.
#
# MECHANISMS. Four independently toggleable channels (see
# make_config() below) let the experimental conditions isolate each
# one's causal contribution to norm emergence:
#   - Discussion (social learning): before deciding, each group holds
#     its own short natural-language exchange, grounded in that
#     group's discussion transcript only.
#   - Selection (social inclusion/exclusion): groups are formed each
#     round by sampling from a directed, weighted reputation network
#     rather than uniformly at random, so agents with a poor
#     reputation are less likely to be picked as partners and can
#     fall below a participation threshold that excludes them from
#     that round's groups entirely.
#   - Evaluation: after contributing, each agent privately rates its
#     groupmates in [-1, 1]; these ratings update the directed edges
#     of the reputation network (negative ratings degrade trust
#     faster than positive ratings rebuild it), and are only active
#     alongside selection, since they exist to drive it.
#   - Perception: each agent also produces a post-round self-report
#     of its normative expectation / NE ("what people should
#     contribute"), its empirical expectation / EE ("what people do
#     contribute"), and preferred/avoided groupmates. This report
#     (a) is folded into the agent's own memory, giving it continuity
#     of belief across rounds, (b) further updates the reputation
#     network via stated partner preferences (restricted to actual
#     groupmates), and (c) pulls the agent's cooperation_tendency
#     toward its own inferred normative expectation.
#
# CONDITIONS (make_config): the main arm ablates discussion (social
# learning, SL) and selection (social selection, SS) on top of a base
# expectation-elicitation channel (E) -- perception, on in every
# condition except the empty-set baseline, which elicits no
# expectations at all (evaluation tracks selection throughout, since
# it only serves to drive it):
#   FULL            E+SL+SS         discussion=on,  selection=on   (evaluation on,  perception on)
#   NO_DISCUSSION   E+SS            discussion=off, selection=on   (evaluation on,  perception on)
#   NO_SELECTION    E+SL            discussion=on,  selection=off  (evaluation off, perception on;  groups uniform-random)
#   BASELINE        E               discussion=off, selection=off  (evaluation off, perception on;  groups uniform-random)
#   PURE_BASELINE   ∅ (empty set)   discussion=off, selection=off  (evaluation off, perception off; groups uniform-random)
# A second, no-expectations arm additionally disables perception on
# top of the FULL/discussion-only/selection-only settings, isolating
# discussion and selection from expectation formation entirely:
#   FULL_NO_EXPECT              discussion=on,  selection=on
#   DISCUSSION_ONLY_NO_EXPECT   discussion=on,  selection=off
#   SELECTION_ONLY_NO_EXPECT    discussion=off, selection=on
#
# ROUND ORDER: (1) group formation, (2) discussion (per group), (3)
# contribution decisions, (4) payoffs, (5) evaluation (within group),
# (6) perception (within group), (7) memory update.
#
# MODELS. Runs either proprietary models via the OpenAI API (e.g.
# gpt-4o-mini, gpt-5-mini) or open-weight models served locally with
# vLLM -- Llama-3.1-8B/70B, Llama-2-13B, Mistral-7B/Nemo-12B, and
# Qwen2.5-7B/14B/72B (see AVAILABLE_MODELS) -- selected per --model
# value and mixable within one invocation. For the perception prompt's
# normative-expectation (NE) field, API models may report null when
# the group's expectations are unclear; open-weight models are
# instead prompted to always give a best-guess estimate, since they
# defaulted to null too readily otherwise.
#
# ROBUSTNESS DESIGNS. --n-agents/--group-size vary population and
# group size directly. --mcpr instead holds the marginal per-capita
# return constant as group size varies: the fund multiplier is
# computed per group as MCPR x actual group size rather than fixed,
# so the incentive to cooperate does not confound group-size effects.
# ============================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "..", "results")

# ============================================================
# Open-source (vLLM) model configuration
# ============================================================

# Both must be supplied by the environment -- no embedded defaults. CACHE_DIR
# is where vLLM downloads/caches model weights (can be large); point it at
# whatever local model cache this machine uses.
HF_TOKEN  = os.getenv("HF_TOKEN")
CACHE_DIR = os.getenv("HF_MODEL_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub"))
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)

# Model keys that select the local vLLM backend. Anything passed to
# --model that is NOT one of these keys is treated as an OpenAI model name.
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

# Per-model sampling params (vLLM backend only)
MODEL_SAMPLING = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct":    {"top_p": 0.9,  "repetition_penalty": 1.05},
    "meta-llama/Llama-2-13b-chat-hf":           {"top_p": 0.9,  "repetition_penalty": 1.05},
    "meta-llama/Meta-Llama-3.1-70B-Instruct":   {"top_p": 0.9,  "repetition_penalty": 1.05},
    "mistralai/Mistral-7B-Instruct-v0.3":       {"top_p": 0.92, "repetition_penalty": 1.08},
    "mistralai/Mistral-Nemo-Instruct-2407":     {"top_p": 0.92, "repetition_penalty": 1.05},
    "Qwen/Qwen2.5-7B-Instruct":                 {"top_p": 0.9,  "repetition_penalty": 1.1},
    "Qwen/Qwen2.5-14B-Instruct":                {"top_p": 0.9,  "repetition_penalty": 1.05},
    "Qwen/Qwen2.5-72B-Instruct":                {"top_p": 0.9,  "repetition_penalty": 1.05},
}

# Models that require tensor_parallel_size > 1 (too large for a single GPU)
MODEL_TENSOR_PARALLEL = {
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 2,
    "Qwen/Qwen2.5-72B-Instruct":              2,
}

# Cap context length for large models — their defaults (131k+) require far more
# KV cache than is available. Our simulation prompts are ≤ 4k tokens.
MODEL_MAX_LEN = {
    "meta-llama/Meta-Llama-3.1-70B-Instruct": 4096,
    "Qwen/Qwen2.5-72B-Instruct":              4096,
}

# Assign model keys to agent IDs for the local backend. Empty = all agents
# use whichever model the outer run loop is currently on (see main()).
MODEL_ASSIGNMENTS: Dict[int, str] = {}
DEFAULT_MODEL = "mistral"

# The model requested for the run currently executing -- either an
# AVAILABLE_MODELS key (local backend) or an OpenAI model name (API
# backend). Set by main() before each model's seeds/conditions loop;
# also settable directly by callers that use run_sim()/this module as a
# library rather than via the CLI.
CURRENT_MODEL: str = "gpt-4o-mini"

# OpenAI models that only accept temperature=1 (the API default); omit the
# param for these.
FIXED_TEMPERATURE_MODELS: set = {"gpt-5-mini", "gpt-5-mini-2025-08-01"}

# Accumulates OpenAI token usage across a run for cost reporting (API
# backend only -- vLLM calls don't populate this).
_usage: Dict[str, int] = defaultdict(int)  # prompt_tokens, cached_tokens, completion_tokens


def _is_local_model(model: str) -> bool:
    return model in AVAILABLE_MODELS


# ============================================================
# OpenAI (API) backend -- lazy import so a GPU-only node without the
# `openai` / `python-dotenv` packages (or without OPENAI_API_KEY set) can
# still run local models through this same file.
# ============================================================

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from dotenv import load_dotenv
            from openai import OpenAI
        except ImportError as e:
            raise EnvironmentError(
                "Requested an OpenAI model but the `openai` / `python-dotenv` "
                f"packages are not installed: {e}"
            )
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
        _openai_client = OpenAI()
    return _openai_client


# ============================================================
# Open-source (vLLM) backend -- lazy import so an API-only node without
# `vllm` installed (and without a GPU) can still run OpenAI models.
# ============================================================

class LocalModel:
    """Wraps a vLLM engine for a single model."""

    def __init__(self, model_name: str):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise EnvironmentError(
                f"Requested a local model ({model_name}) but the `vllm` package "
                f"is not installed: {e}"
            )
        self._SamplingParams = SamplingParams
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
        sampling_params = self._SamplingParams(
            temperature=temperature,
            max_tokens=512,
            **extra,
        )
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()


# Lazy model registry — each unique model loaded once
_model_registry: Dict[str, LocalModel] = {}


def _get_model(agent_id: int) -> LocalModel:
    key  = MODEL_ASSIGNMENTS.get(agent_id, DEFAULT_MODEL)
    name = AVAILABLE_MODELS.get(key, AVAILABLE_MODELS[DEFAULT_MODEL])
    if name not in _model_registry:
        _model_registry[name] = LocalModel(name)
    return _model_registry[name]


def _extract_json(text: str) -> dict:
    """Extract the first valid JSON object from raw model output."""
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


@dataclass
class SimConfig:
    N: int = 12
    ROUNDS: int = 20
    GROUP_SIZE: int = 4
    ENDOWMENT: int = 10
    MULTIPLIER: float = 1.6
    # If set (via --mcpr), the group-fund multiplier is computed per group as
    # MCPR x actual group size instead of the fixed MULTIPLIER above, holding
    # the per-token marginal return constant across group sizes (the
    # group-size-variation robustness sweep). None (default) preserves the
    # exact fixed-MULTIPLIER behavior/wording of every run to date.
    MCPR: Optional[float] = None

    # Condition toggles
    discussion_on: bool = True
    selection_on: bool = True     # if False → random group mixing

    # Discussion
    discussion_turns: int = 1
    history_window: int = 3

    # Network / selection
    network_update_rate_pos: float = 0.10
    network_update_rate_neg: float = 0.8
    tie_remove_threshold: float = 0.15
    participation_threshold: float = 0.3

    # Tendency learning
    tendency_learning_rate_pos: float = 0.015
    tendency_learning_rate_neg: float = 0.03

    # Perception-based network update
    perception_update_rate_pos: float = 0.2
    perception_update_rate_neg: float = 0.45

    # Norm internalization
    norm_internalization_rate: float = 0.2

    # Condition toggles
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

def make_config(condition: str, n_agents: int = 12, group_size: int = 4,
                mcpr: Optional[float] = None) -> SimConfig:
    cfg = SimConfig()
    cfg.N          = n_agents
    cfg.GROUP_SIZE = group_size
    cfg.MCPR       = mcpr
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
    # No-expectations ablation arm: perception/norm-reflection is disabled,
    # isolating discussion and/or selection from expectation formation.
    elif condition == "FULL_NO_EXPECT":
        cfg.discussion_on = True
        cfg.selection_on = True
        cfg.perception_on = False
    elif condition == "DISCUSSION_ONLY_NO_EXPECT":
        cfg.discussion_on = True
        cfg.selection_on = False
        cfg.evaluation_on = False
        cfg.perception_on = False
    elif condition == "SELECTION_ONLY_NO_EXPECT":
        cfg.discussion_on = False
        cfg.selection_on = True
        cfg.perception_on = False
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return cfg


NO_EXPECT_CONDITIONS = {"FULL_NO_EXPECT", "DISCUSSION_ONLY_NO_EXPECT", "SELECTION_ONLY_NO_EXPECT"}
STANDARD_CONDITIONS  = ["FULL", "NO_DISCUSSION", "NO_SELECTION", "BASELINE", "PURE_BASELINE"]
ALL_CONDITIONS       = STANDARD_CONDITIONS + sorted(NO_EXPECT_CONDITIONS)


def group_multiplier(config: SimConfig, group_size: int) -> float:
    """MCPR x actual group size if config.MCPR is set (group-size-variation
    experiment), else the fixed MULTIPLIER (every run to date)."""
    if config.MCPR is not None:
        return config.MCPR * group_size
    return config.MULTIPLIER


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


# ============================================================
# Network / Group Formation
# ============================================================

def form_groups(agents: List[Agent], network: nx.DiGraph, config: SimConfig) -> List[List[int]]:
    """
    Probabilistic group formation from directed edge weights.

      - Seed: sampled proportional to each agent's average INCOMING edge weight.
      - Partners: sampled proportional to the seed's OUTGOING edge weights.
    """
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
        chosen = [int(x) for x in np.random.choice(candidates, size=n_pick, replace=False, p=partner_probs)]

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
        rate = config.tendency_learning_rate_neg if mean_signal < 0 else config.tendency_learning_rate_pos
        new_tendency = agent.cooperation_tendency + rate * mean_signal
        agent.cooperation_tendency = round(max(0.0, min(1.0, new_tendency)), 4)


def update_network_from_perception(
    network: nx.DiGraph,
    perceptions: Dict[int, dict],
    group_membership: Dict[int, List[int]],
    config: SimConfig,
) -> nx.DiGraph:
    """
    Update edge weights from stated partner preferences, restricted to agents
    who were actually in the same group this round. Hallucinated IDs are ignored.
    """
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
            max(0.0, min(1.0, agent.cooperation_tendency + config.norm_internalization_rate * (target - agent.cooperation_tendency))),
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
                f"  Agent {a.id} excluded (avg incoming={avg_incoming:.3f} < {config.participation_threshold})"
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
    group_ids: List[int],
    transcript: List[dict],
    config: SimConfig,
) -> str:
    """
    Discussion prompt scoped to the agent's assigned group.
    The agent only sees and responds to their group members' messages.
    """
    transcript_str = (
        "\n".join(f"  Agent {m['agent_id']}: {m['message']}" for m in transcript)
        if transcript else "  (No messages yet — you go first.)"
    )
    memory_str = json.dumps(agent.memory[-config.history_window:], default=str) \
        if agent.memory else "  (No prior rounds yet.)"

    return (
        f"Round: {t}\n"
        f"Task: Group discussion before contribution. You are in a group with agents: {group_ids}.\n\n"
        f"Your recent observations and reflections:\n{memory_str}\n\n"
        f"Discussion so far (your group only):\n{transcript_str}\n\n"
        f"Share what you think your group should do this round and why. "
        f"Refer to what others said if relevant. Keep it to 1–3 sentences.\n\n"
        f"Output JSON with EXACT keys:\n{{\n  \"message\": \"<your statement>\"\n}}"
    )


def build_decision_prompt(
    t: int,
    agent: Agent,
    group_ids: List[int],
    config: SimConfig,
    transcript: List[dict] = None,
) -> str:
    memory_str = json.dumps(agent.memory[-config.history_window:], default=str) \
        if agent.memory else "  (No prior rounds yet.)"
    summary_str = json.dumps(agent.memory_summary, default=str) \
        if agent.memory_summary else "  (No earlier rounds.)"

    if transcript:
        transcript_str = "\n".join(f"  Agent {m['agent_id']}: {m['message']}" for m in transcript)
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

    # Fund-multiplier wording differs only when --mcpr is in use (see
    # SimConfig.MCPR / group_multiplier()); the fixed-MULTIPLIER wording
    # below is byte-identical to every run to date.
    if config.MCPR is not None:
        multiplier = group_multiplier(config, len(group_ids))
        fund_line = (
            f"  MCPR (marginal per-capita return) = {config.MCPR} per token contributed\n"
            f"  Multiplier = MCPR × group_size = {config.MCPR} × {len(group_ids)} = {multiplier:.2f}\n"
            f"  Group fund = sum(all contributions) × {multiplier:.2f}\n"
        )
    else:
        fund_line = f"  Group fund = sum(all contributions) × {config.MULTIPLIER}\n"

    return (
        f"Round: {t}\n"
        f"Task: Choose your contribution to the group fund.\n\n"
        f"Rules:\n"
        f"  Endowment = {config.ENDOWMENT} tokens\n"
        f"  Choose contribution c in [0, {config.ENDOWMENT}]\n"
        f"{fund_line}"
        f"  Each of the {len(group_ids)} members receives an equal share\n"
        f"  Your payoff = ({config.ENDOWMENT} - c) + fund / {len(group_ids)}\n\n"
        f"Your group this round (agent ids): {group_ids}\n\n"
        f"Discussion this round (your group only):\n{transcript_str}\n\n"
        f"Your norm reflection from last round:\n{perception_str}\n\n"
        f"Earlier rounds summary:\n{summary_str}\n\n"
        f"Recent rounds (detailed):\n{memory_str}\n\n"
        f"Output JSON with EXACT keys:\n{{\n  \"contribution\": <int 0-{config.ENDOWMENT}>\n}}"
    )


def build_evaluation_prompt(
    t: int, agent: Agent, group_ids: List[int], contributions: Dict[int, int], config: SimConfig
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
        f"Output JSON with EXACT keys:\n{{\n  \"evaluations\": {{{ id_list }}}\n}}"
    )


def build_perception_prompt(
    t: int, agent: Agent, group_ids: List[int],
    contributions: Dict[int, int], payoffs: Dict[int, float], config: SimConfig
) -> str:
    other_ids = [j for j in group_ids if j != agent.id]
    contrib_lines = "\n".join(
        f"  Agent {j}: contributed {contributions.get(j, '?')} / {config.ENDOWMENT}"
        for j in other_ids
    )
    avg = sum(contributions.get(j, 0) for j in group_ids) / len(group_ids)
    memory_str = json.dumps(agent.memory[-config.history_window:], default=str) \
        if agent.memory else "  (No prior rounds yet.)"
    summary_str = json.dumps(agent.memory_summary, default=str) \
        if agent.memory_summary else "  (No earlier rounds.)"
    all_ids = [j for j in group_ids if j != agent.id]

    # Injunctive-norm wording differs by backend (see module docstring):
    # the API backend allows an explicit "unclear -> null" escape hatch;
    # the local/open-source backend forces a best-guess estimate, since
    # those model families were tuned away from over-using null.
    if _is_local_model(CURRENT_MODEL):
        note_block = ""
        injunctive_desc = (
            "your best estimate of the contribution level this group thinks people "
            "SHOULD contribute, regardless of what they actually did — if unsure, "
            "estimate based on observed behavior"
        )
    else:
        note_block = (
            "Note: If you have limited experience with this group, the injunctive norm may be "
            "uncertain. Only report a confident value if the group's shared expectations are "
            "reasonably clear to you.\n\n"
        )
        injunctive_desc = (
            "the contribution level agents in this group think people SHOULD contribute, "
            "regardless of what they actually did — or null if unclear"
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
        f"For partner lists, choose only from agent IDs in your group: {all_ids}.\n\n"
        f"{note_block}"
        f"Output JSON with EXACT keys:\n"
        f'{{\n'
        f'  "injunctive_norm": <float 0-{config.ENDOWMENT}, {injunctive_desc}>,\n'
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
    model_name_used = CURRENT_MODEL

    try:
        if _is_local_model(CURRENT_MODEL):
            lm = _get_model(agent.id)
            model_name_used = lm.model_name
            raw_response = lm.generate(system_prompt, user_prompt, temperature=temperature)
            parsed = _extract_json(raw_response) or dict(default_response)
        else:
            client = _get_openai_client()
            call_kwargs: dict = dict(
                model=CURRENT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            if CURRENT_MODEL not in FIXED_TEMPERATURE_MODELS:
                call_kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**call_kwargs)
            raw_response = resp.choices[0].message.content
            parsed = json.loads(raw_response)
            u = resp.usage
            _usage["prompt_tokens"]      += u.prompt_tokens
            _usage["completion_tokens"]  += u.completion_tokens
            _usage["cached_tokens"]      += getattr(
                getattr(u, "prompt_tokens_details", None), "cached_tokens", 0
            ) or 0
    except Exception as e:
        logger.warning(
            f"LLM call type={call_type} agent={agent.id} round={round_num} failed: {e}."
        )

    conv_log.append(ConversationEntry(
        round_num=round_num,
        call_type=call_type,
        agent_id=agent.id,
        target_id=target_id,
        model_name=model_name_used,
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


def LLM_decide(
    agent: Agent,
    round_num: int,
    group_ids: List[int],
    config: SimConfig,
    conv_log: List[ConversationEntry],
    transcript: List[dict] = None,
) -> dict:
    result = _llm_call(agent, "decision", round_num,
                       build_decision_prompt(round_num, agent, group_ids, config, transcript),
                       {"contribution": config.ENDOWMENT // 2}, 0.7, conv_log)
    c = int(result.get("contribution", config.ENDOWMENT // 2))
    result["contribution"] = max(0, min(config.ENDOWMENT, c))
    return result


def LLM_evaluate(agent, round_num, group_ids, contributions, config, conv_log) -> Dict[int, float]:
    other_ids = [j for j in group_ids if j != agent.id]
    valid_ids = set(other_ids)
    result = _llm_call(agent, "evaluate", round_num,
                       build_evaluation_prompt(round_num, agent, group_ids, contributions, config),
                       {"evaluations": {str(j): 0.0 for j in other_ids}}, 0.5, conv_log)
    evals_raw = result.get("evaluations", {})
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
# Main Simulation
# ============================================================

def run_sim(condition: str, seed: int,
            n_agents: int = 12, group_size: int = 4,
            mcpr: Optional[float] = None) -> SimResults:
    random.seed(seed)
    np.random.seed(seed)

    config = make_config(condition, n_agents=n_agents, group_size=group_size, mcpr=mcpr)
    agents = init_agents(config)
    agent_by_id: Dict[int, Agent] = {a.id: a for a in agents}
    network = init_network(agents)
    round_logs: List[dict] = []
    conv_log: List[ConversationEntry] = []

    for t in range(1, config.ROUNDS + 1):
        logger.info(f"[{condition}|seed={seed}] Round {t}/{config.ROUNDS}")

        # ----------------------------------------------------------
        # 1. GROUP FORMATION
        # Happens first so that discussion and evaluation are
        # scoped to the assigned group.
        # ----------------------------------------------------------
        excluded_ids: set = set()

        if config.selection_on:
            excluded_ids = get_excluded_agents(agents, network, config)
            eligible = [a for a in agents if a.id not in excluded_ids]
            groups = form_groups(eligible, network, config)
        else:
            ids = [a.id for a in agents]
            random.shuffle(ids)
            groups = [ids[i: i + config.GROUP_SIZE] for i in range(0, len(ids), config.GROUP_SIZE)]
            groups = [g for g in groups if len(g) >= 2]

        # ----------------------------------------------------------
        # 2. DISCUSSION — per group, independent transcripts
        # Each group runs its own discussion; agents only hear and
        # respond to their assigned group members.
        # ----------------------------------------------------------
        group_transcripts: Dict[int, List[dict]] = {}  # group_index -> transcript

        if config.discussion_on:
            for g_idx, group_ids in enumerate(groups):
                transcript: List[dict] = []
                agent_order = list(group_ids)
                for _ in range(config.discussion_turns):
                    random.shuffle(agent_order)
                    for aid in agent_order:
                        result = LLM_discuss(agent_by_id[aid], t, group_ids, transcript, config, conv_log)
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
                result = LLM_decide(agent_by_id[i], t, group_ids, config, conv_log, transcript)
                contributions[i] = result["contribution"]

        # ----------------------------------------------------------
        # 4. PAYOFFS
        # ----------------------------------------------------------
        payoffs: Dict[int, float] = {}

        for group_ids in groups:
            multiplier = group_multiplier(config, len(group_ids))
            total = sum(contributions.get(i, 0) for i in group_ids)
            share = (multiplier * total) / len(group_ids)
            for i in group_ids:
                gain = float(config.ENDOWMENT - contributions.get(i, 0)) + share
                agent_by_id[i].material_payoff += gain
                payoffs[i] = gain

        # ----------------------------------------------------------
        # 5. EVALUATION — within group, skipped when evaluation_on is False
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

        # Build group membership map: agent_id -> list of other members in their group
        group_membership: Dict[int, List[int]] = {}
        for g in groups:
            for aid in g:
                group_membership[aid] = [x for x in g if x != aid]

        # ----------------------------------------------------------
        # 6. PERCEPTION — within group
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
        # Excluded agents get an explicit exclusion record — no fake
        # group data. Active agents get full round observations.
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
                partners = group_membership.get(a.id, [])
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
        # LOG ROUND
        # All group transcripts merged into a single list for logging,
        # tagged with group index so they can be reconstructed.
        # ----------------------------------------------------------
        merged_transcript = [
            {"group": g_idx, **entry}
            for g_idx, transcript in group_transcripts.items()
            for entry in transcript
        ] if config.discussion_on else None

        network_snapshot = [
            {"u": u, "v": v, "weight": round(d["weight"], 4)}
            for u, v, d in network.edges(data=True)
        ]

        round_logs.append({
            "round": t,
            "condition": condition,
            "mcpr": config.MCPR,
            "group_multipliers": {
                str(g_idx): round(group_multiplier(config, len(group_ids)), 4)
                for g_idx, group_ids in enumerate(groups)
            },
            "discussion_transcript": merged_transcript,
            "excluded_agents": sorted(excluded_ids),
            "groups": groups,
            "contributions": {str(k): v for k, v in contributions.items()},
            "payoffs": {str(k): round(v, 4) for k, v in payoffs.items()},
            "evaluations": {f"{i}->{j}": round(s, 4) for (i, j), s in evaluations.items()},
            "network_weights": network_snapshot,
            "agent_states": [
                {"id": a.id, "cooperation_tendency": a.cooperation_tendency,
                 "material_payoff": round(a.material_payoff, 3)}
                for a in agents
            ],
            "perceptions": {str(k): v for k, v in perceptions.items()} if config.perception_on else None,
        })

    return SimResults(condition, seed, agents, network, round_logs, conv_log)


# ============================================================
# Main
# ============================================================

def main(argv: Optional[List[str]] = None) -> None:
    global CURRENT_MODEL, DEFAULT_MODEL

    ALL_SEEDS     = list(range(43, 53))   # seeds 43–52 (10 seeds)
    OS_MODEL_KEYS = list(AVAILABLE_MODELS.keys())

    parser = argparse.ArgumentParser(
        description="Run SoNoLiSi v5 local simulation (OpenAI API and/or open-source vLLM models)."
    )
    parser.add_argument(
        "--model", "-m",
        nargs="+",
        default=["gpt-4o-mini"],
        metavar="MODEL",
        help="Model(s) to run, in any mix. A value matching one of the open-source "
             f"keys ({OS_MODEL_KEYS}) uses the local vLLM backend; any other value "
             "is treated as an OpenAI model name (e.g. gpt-4o-mini, gpt-5-mini). "
             "Default: gpt-4o-mini.",
    )
    parser.add_argument(
        "--conditions", "-c",
        nargs="+",
        choices=ALL_CONDITIONS,
        default=STANDARD_CONDITIONS,
        metavar="COND",
        help="Condition(s) to run. Choices: the 5 standard conditions "
             f"({STANDARD_CONDITIONS}) plus the 3 no-expectations ablation "
             f"conditions ({sorted(NO_EXPECT_CONDITIONS)}). Default: the 5 "
             "standard conditions.",
    )
    parser.add_argument(
        "--seeds", "-s",
        nargs="+",
        type=int,
        default=None,
        metavar="SEED",
        help="Explicit seed list (e.g. --seeds 43 44 45). Overrides --n.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        metavar="N",
        help="Number of seeds to run from the default list (or from --start-seed, "
             "if given). Default: 10.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="First seed of a sequential --n-length range (e.g. --start-seed 53 "
             "--n 40 runs seeds 53-92). Overrides the default seed list; ignored "
             "if --seeds is given.",
    )
    parser.add_argument(
        "--device", "-d",
        type=int,
        default=None,
        metavar="GPU",
        help="CUDA device index for local (vLLM) models; sets CUDA_VISIBLE_DEVICES "
             "(expanded across GPUs for models needing tensor-parallelism). Unused "
             "for OpenAI-only runs. Default: none (leaves CUDA_VISIBLE_DEVICES as-is).",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=12,
        metavar="N",
        help="Total number of agents in the simulation. Default: 12.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        metavar="G",
        help="Number of agents per group. Default: 4.",
    )
    parser.add_argument(
        "--mcpr",
        type=float,
        default=None,
        metavar="MCPR",
        help="Marginal per-capita return. If given, the group-fund "
             "multiplier is computed per group as MCPR x actual group size "
             "instead of the fixed default multiplier (1.6), holding the "
             "per-token return constant across group sizes, and results are "
             "written to a separate local_groupsizevary/ subtree. Default: "
             "unset (fixed multiplier, identical to every run to date).",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default=None,
        metavar="DIR",
        help="Extra path component under results/{model}/local/ (e.g. "
             "'additional_runs' to keep an extension batch of seeds separate "
             "from the original run). Default: none (writes directly to local/).",
    )
    args = parser.parse_args(argv)

    models_to_run = args.model
    local_models_requested = [m for m in models_to_run if _is_local_model(m)]

    if local_models_requested and args.device is not None:
        max_tp = max(MODEL_TENSOR_PARALLEL.get(AVAILABLE_MODELS[m], 1) for m in local_models_requested)
        gpu_ids = ",".join(str(args.device + i) for i in range(max_tp))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        logger.info(f"CUDA_VISIBLE_DEVICES: {gpu_ids}  (tp_size={max_tp})")
    elif not local_models_requested and args.device is not None:
        # Preserves the old API-backend file's behavior: harmless/unused for
        # API calls, but set (and logged) if explicitly requested.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    if args.seeds is not None:
        seeds = args.seeds
    elif args.start_seed is not None:
        seeds = list(range(args.start_seed, args.start_seed + args.n))
    else:
        seeds = ALL_SEEDS[:args.n]
    conditions = args.conditions
    n_agents   = args.n_agents
    group_size = args.group_size
    mcpr       = args.mcpr

    logger.info(f"Models     : {models_to_run}")
    logger.info(f"Conditions : {conditions}")
    logger.info(f"Seeds      : {seeds}")
    logger.info(f"Agents/Grp : N={n_agents}, G={group_size}")
    if mcpr is not None:
        logger.info(f"MCPR       : {mcpr}  (multiplier = {mcpr} × {group_size} = {mcpr * group_size:.2f})")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model in models_to_run:
        CURRENT_MODEL = model
        is_local = _is_local_model(model)
        if is_local:
            DEFAULT_MODEL = model
            _model_registry.clear()
            model_dir_name = model  # short key, e.g. "mistral"
        else:
            model_dir_name = model.replace("/", "_").replace(" ", "_")  # e.g. "gpt-4o-mini"

        logger.info(f"\n{'=' * 60}\nStarting model: {model}  (backend={'vLLM' if is_local else 'OpenAI API'})\n{'=' * 60}")

        for seed in seeds:
            for cond in conditions:
                logger.info(f"\n{'=' * 50}\nRunning: {cond} | seed={seed} | model={model}\n{'=' * 50}")
                _usage.clear()

                results = run_sim(condition=cond, seed=seed,
                                  n_agents=n_agents, group_size=group_size, mcpr=mcpr)

                if not is_local:
                    pt, ct, ot = _usage["prompt_tokens"], _usage["cached_tokens"], _usage["completion_tokens"]
                    logger.info(f"  Tokens — input: {pt:,}  (cached: {ct:,})  output: {ot:,}")

                # Directory + filename tagging: "local_groupsizevary" (--mcpr
                # given) takes precedence over "local"/"local_noexpect", since
                # MCPR sweeps are never combined with a no-expectations
                # condition in this release's runs.
                is_noexpect = cond in NO_EXPECT_CONDITIONS
                if mcpr is not None:
                    variant_dir = "local_groupsizevary"
                    variant_tag = f"N{n_agents}_G{group_size}_MCPR{mcpr}"
                    fname_tag   = "_groupsizevary"
                else:
                    variant_dir = "local_noexpect" if is_noexpect else "local"
                    variant_tag = f"N{n_agents}_G{group_size}" if (n_agents != 12 or group_size != 4) else ""
                    fname_tag   = "_noexpect" if is_noexpect else ""

                cond_dir = os.path.join(RESULTS_DIR, model_dir_name, variant_dir)
                if variant_tag:
                    cond_dir = os.path.join(cond_dir, variant_tag)
                if args.out_subdir:
                    cond_dir = os.path.join(cond_dir, args.out_subdir)
                seed_dir = os.path.join(cond_dir, f"seed{seed}")
                os.makedirs(seed_dir, exist_ok=True)

                if is_local:
                    fname_log  = f"log_v5os_local{fname_tag}_{model}_{run_timestamp}_{cond}_seed{seed}.json"
                    fname_conv = f"conversations_v5os_local{fname_tag}_{model}_{run_timestamp}_{cond}_seed{seed}.json"
                else:
                    fname_log  = f"log_v5local{fname_tag}_{run_timestamp}_{cond}_seed{seed}.json"
                    fname_conv = f"conversations_v5local{fname_tag}_{run_timestamp}_{cond}_seed{seed}.json"

                logs_path = os.path.join(seed_dir, fname_log)
                with open(logs_path, "w") as f:
                    json.dump(results.to_dict(), f, indent=2, default=str)
                logger.info(f"Saved logs → {logs_path}")

                conv_path = os.path.join(seed_dir, fname_conv)
                with open(conv_path, "w") as f:
                    json.dump(results.conversations_to_dict(), f, indent=2, default=str)
                logger.info(f"Saved conversations → {conv_path}")

    logger.info(f"\n{'=' * 50}\nRUN COMPLETE\n  Models    : {models_to_run}\n"
                f"  Seeds     : {seeds}\n  Conditions: {conditions}\n{'=' * 50}")


if __name__ == "__main__":
    main()
