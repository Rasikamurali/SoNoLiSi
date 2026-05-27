# ============================================================
# LLM-AGENT NORM EMERGENCE — v5-fixed OPEN-SOURCE BACKEND
#
# Identical simulation logic to SoNoLiSi_v5_fixed.py (fixed binary
# roles: cooperator | violator, directed network, sticky reputation)
# but uses local HuggingFace models instead of the OpenAI API.
# ============================================================

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
from vllm import LLM, SamplingParams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Model configuration
# ============================================================

HF_TOKEN    = os.getenv("HF_TOKEN", "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")
CACHE_DIR   = "/data3/models/hub"
RESULTS_DIR = "../results"

os.environ.setdefault("HF_TOKEN", HF_TOKEN)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Available models — add / remove as needed
AVAILABLE_MODELS = {
    "llama":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen":    "Qwen/Qwen2.5-7B-Instruct",
}

# Per-model sampling params
MODEL_SAMPLING = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"top_p": 0.9,  "repetition_penalty": 1.05},
    "mistralai/Mistral-7B-Instruct-v0.3":     {"top_p": 0.92, "repetition_penalty": 1.08},
    "Qwen/Qwen2.5-7B-Instruct":               {"top_p": 0.9,  "repetition_penalty": 1.1},
}

# Assign model keys to agent IDs. Empty = all agents use DEFAULT_MODEL.
# Example: MODEL_ASSIGNMENTS = {0: "mistral", 1: "qwen"}
MODEL_ASSIGNMENTS: Dict[int, str] = {}
DEFAULT_MODEL = "qwen"


# ============================================================
# Local model loader (vLLM)
# ============================================================

class LocalModel:
    """Wraps a vLLM engine for a single model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Loading model via vLLM: {model_name}")
        self.llm = LLM(
            model=model_name,
            download_dir=CACHE_DIR,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
        )
        logger.info(f"Model ready: {model_name}")

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        extra = MODEL_SAMPLING.get(self.model_name, {})
        sampling_params = SamplingParams(
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
    role: str = "cooperator"          # "cooperator" | "violator" — fixed at init
    memory: List[dict] = field(default_factory=list)
    material_payoff: float = 0.0


@dataclass
class SimConfig:
    N: int = 10
    ROUNDS: int = 20
    GROUP_SIZE: int = 4
    ENDOWMENT: int = 10
    MULTIPLIER: float = 1.6
    n_violators: int = 3

    discussion_on: bool = True
    selection_on: bool = True

    discussion_turns: int = 1
    history_window: int = 3

    network_update_rate_pos: float = 0.15
    network_update_rate_neg: float = 0.6
    tie_remove_threshold: float = 0.15

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
            "violator_ids": [a.id for a in self.agents if a.role == "violator"],
            "final_agents": [
                {"id": a.id, "role": a.role, "material_payoff": a.material_payoff}
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
        cfg.discussion_on = True;  cfg.selection_on = True
    elif condition == "NO_DISCUSSION":
        cfg.discussion_on = False; cfg.selection_on = True
    elif condition == "NO_SELECTION":
        cfg.discussion_on = True;  cfg.selection_on = False
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return cfg


# ============================================================
# Initialization
# ============================================================

def init_agents(config: SimConfig) -> List[Agent]:
    agents = [Agent(id=i, role="cooperator") for i in range(config.N)]
    for vid in random.sample(range(config.N), config.n_violators):
        agents[vid].role = "violator"
    for a in agents:
        logger.info(f"  Agent {a.id}: role={a.role}")
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
        chosen = list(np.random.choice(candidates, size=n_pick, replace=False, p=partner_probs))

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


# ============================================================
# Prompts
# ============================================================

SYSTEM_PROMPT_COOPERATOR = """You are an autonomous agent in a repeated group interaction.
You make decisions about contributing to a shared group fund.

Your goal is to act fairly and promote the wellbeing of the group.
You believe in cooperation and reciprocity: you contribute generously and expect others to do the same.
When others free-ride, you may adjust your behavior, but you generally uphold cooperative norms.

When deciding how much to contribute, you consider:
1. What is fair given what others are doing
2. What you believe the group expects
3. The long-term benefits of establishing cooperative norms
4. Past experiences with your group members

Output STRICT JSON only with the exact keys requested. No extra text."""

SYSTEM_PROMPT_VIOLATOR = """You are an autonomous agent in a repeated group interaction.
You make decisions about contributing to a shared group fund.

Your primary goal is to maximize your own material payoff.
You tend to contribute as little as possible while benefiting from others' contributions.
You may signal cooperation in discussions to avoid scrutiny, but your decisions prioritize self-interest.

When deciding how much to contribute, you consider:
1. How little you can contribute without being detected or penalized
2. What others are contributing (to avoid standing out)
3. The risk that low contributions will damage your reputation
4. Your own accumulated payoff relative to others

Output STRICT JSON only with the exact keys requested. No extra text."""


def get_system_prompt(agent: Agent) -> str:
    return SYSTEM_PROMPT_VIOLATOR if agent.role == "violator" else SYSTEM_PROMPT_COOPERATOR


def build_discussion_prompt(t: int, agent: Agent, transcript: List[dict], config: SimConfig) -> str:
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
        f"Keep it to 1–3 sentences.\n\n"
        f"Output JSON with EXACT keys:\n{{\n  \"message\": \"<your statement>\"\n}}"
    )


def build_decision_prompt(
    t: int, agent: Agent, group_ids: List[int], transcript: List[dict], config: SimConfig
) -> str:
    disc_str = (
        "\n".join(f"  Agent {m['agent_id']}: {m['message']}" for m in transcript)
        if transcript else "  (No discussion this round.)"
    )
    memory_str = json.dumps(agent.memory[-config.history_window:], default=str) \
        if agent.memory else "  (No prior rounds yet.)"
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
        f"Your recent observations and reflections:\n{memory_str}\n\n"
        f"Group discussion:\n{disc_str}\n\n"
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
    return (
        f"Round {t} — Post-round reflection.\n\n"
        f"This round:\n"
        f"  Your group: {group_ids}\n"
        f"  Your contribution: {contributions.get(agent.id, '?')} / {config.ENDOWMENT}\n"
        f"  Partner contributions:\n{contrib_lines}\n"
        f"  Group average: {avg:.2f} / {config.ENDOWMENT}\n"
        f"  Your payoff this round: {round(payoffs.get(agent.id, 0.0), 2)}\n\n"
        f"Your history (last {config.history_window} rounds):\n{memory_str}\n\n"
        f"Answer each question in exactly 1 sentence of 25–40 words.\n\n"
        f"Output JSON with EXACT keys:\n"
        f'{{\n'
        f'  "inferred_norm": "<What contribution level does this group treat as the norm?>",\n'
        f'  "common_behavior": "<What is the typical behavior among group members so far?>",\n'
        f'  "expectation_of_others": "<What do you expect others will contribute next round?>",\n'
        f'  "others_expectation_of_me": "<What do you think others expect you to contribute?>",\n'
        f'  "preferred_partners": "<Which agents would you choose to play with again and why?>",\n'
        f'  "agents_to_avoid": "<Which agents would you prefer to avoid next round, and why?>"\n'
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
    raw_response = ""
    parsed = dict(default_response)

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


def LLM_discuss(agent, round_num, transcript, config, conv_log):
    return _llm_call(agent, "discussion", round_num,
                     build_discussion_prompt(round_num, agent, transcript, config),
                     {"message": ""}, 0.8, conv_log)


def LLM_decide(agent, round_num, group_ids, transcript, config, conv_log):
    result = _llm_call(agent, "decision", round_num,
                       build_decision_prompt(round_num, agent, group_ids, transcript, config),
                       {"contribution": config.ENDOWMENT // 2}, 0.7, conv_log)
    c = int(result.get("contribution", config.ENDOWMENT // 2))
    result["contribution"] = max(0, min(config.ENDOWMENT, c))
    return result


def LLM_evaluate(agent, round_num, group_ids, contributions, config, conv_log) -> Dict[int, float]:
    other_ids = [j for j in group_ids if j != agent.id]
    result = _llm_call(agent, "evaluate", round_num,
                       build_evaluation_prompt(round_num, agent, group_ids, contributions, config),
                       {"evaluations": {str(j): 0.0 for j in other_ids}}, 0.5, conv_log)
    evals_raw = result.get("evaluations", {})
    return {
        int(k): max(-1.0, min(1.0, float(v)))
        for k, v in evals_raw.items()
        if str(k).lstrip("-").isdigit()
    }


def LLM_perceive(agent, round_num, group_ids, contributions, payoffs, config, conv_log):
    default = {k: "" for k in ["inferred_norm", "common_behavior", "expectation_of_others",
                                 "others_expectation_of_me", "preferred_partners", "agents_to_avoid"]}
    return _llm_call(agent, "perception", round_num,
                     build_perception_prompt(round_num, agent, group_ids, contributions, payoffs, config),
                     default, 0.6, conv_log)


# ============================================================
# Main Simulation
# ============================================================

def run_sim(condition: str, seed: int) -> SimResults:
    random.seed(seed)
    np.random.seed(seed)

    config = make_config(condition)
    agents = init_agents(config)
    agent_by_id: Dict[int, Agent] = {a.id: a for a in agents}
    network = init_network(agents)
    round_logs: List[dict] = []
    conv_log: List[ConversationEntry] = []

    for t in range(1, config.ROUNDS + 1):
        logger.info(f"[{condition}|seed={seed}] Round {t}/{config.ROUNDS}")

        # 1. DISCUSSION
        transcript: List[dict] = []
        if config.discussion_on:
            agent_order = list(range(config.N))
            for _ in range(config.discussion_turns):
                random.shuffle(agent_order)
                for aid in agent_order:
                    result = LLM_discuss(agent_by_id[aid], t, transcript, config, conv_log)
                    message = str(result.get("message", "")).strip()
                    if message:
                        transcript.append({"agent_id": aid, "message": message})

        # 2. GROUP FORMATION
        if config.selection_on:
            groups = form_groups(agents, network, config)
        else:
            ids = [a.id for a in agents]
            random.shuffle(ids)
            groups = [ids[i: i + config.GROUP_SIZE] for i in range(0, len(ids), config.GROUP_SIZE)]
            groups = [g for g in groups if len(g) >= 2]

        # 3. CONTRIBUTIONS
        contributions: Dict[int, int] = {}
        for group_ids in groups:
            for i in group_ids:
                result = LLM_decide(agent_by_id[i], t, group_ids, transcript, config, conv_log)
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

        # 5. EVALUATION + NETWORK UPDATE
        evaluations: Dict[Tuple[int, int], float] = {}
        for group_ids in groups:
            for i in group_ids:
                ratings = LLM_evaluate(agent_by_id[i], t, group_ids, contributions, config, conv_log)
                for j, signal in ratings.items():
                    evaluations[(i, j)] = signal

        network = update_network(network, evaluations, config)

        # 6. PERCEPTION
        perceptions: Dict[int, dict] = {}
        if config.perception_on:
            for group_ids in groups:
                for i in group_ids:
                    perceptions[i] = LLM_perceive(
                        agent_by_id[i], t, group_ids, contributions, payoffs, config, conv_log
                    )

        # 7. MEMORY UPDATE
        for a in agents:
            partners = next((
                [x for x in g if x != a.id] for g in groups if a.id in g
            ), [])
            partner_contribs = {str(j): contributions.get(j) for j in partners}
            all_ids = partners + [a.id]
            avg = (
                sum(contributions.get(x, 0) for x in all_ids) / len(all_ids)
                if any(x in contributions for x in all_ids) else None
            )
            a.memory.append({
                "round": t,
                "partners": partners,
                "my_contribution": contributions.get(a.id),
                "partner_contributions": partner_contribs,
                "group_avg_contribution": round(avg, 2) if avg is not None else None,
                "my_payoff": round(payoffs.get(a.id, 0.0), 3),
                "norm_reflection": perceptions.get(a.id, {}),
            })
            if len(a.memory) > config.history_window:
                a.memory = a.memory[-config.history_window:]

        # LOG
        round_logs.append({
            "round": t,
            "condition": condition,
            "discussion_transcript": transcript if config.discussion_on else None,
            "groups": groups,
            "contributions": {str(k): v for k, v in contributions.items()},
            "payoffs": {str(k): round(v, 4) for k, v in payoffs.items()},
            "evaluations": {f"{i}->{j}": round(s, 4) for (i, j), s in evaluations.items()},
            "network_weights": [
                {"u": u, "v": v, "weight": round(d["weight"], 4)}
                for u, v, d in network.edges(data=True)
            ],
            "agent_states": [
                {"id": a.id, "role": a.role, "material_payoff": round(a.material_payoff, 3)}
                for a in agents
            ],
            "perceptions": {str(k): v for k, v in perceptions.items()} if config.perception_on else None,
        })

    return SimResults(condition, seed, agents, network, round_logs, conv_log)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    conditions = ["FULL", "NO_DISCUSSION", "NO_SELECTION"]
    seeds = [42, 43, 45]

    for seed in seeds:
        for cond in conditions:
            logger.info(f"\n{'=' * 50}\nRunning: {cond} | seed={seed}\n{'=' * 50}")
            results = run_sim(condition=cond, seed=seed)

            logs_path = os.path.join(RESULTS_DIR, f"log_v5fixedos_{DEFAULT_MODEL}_{run_timestamp}_{cond}_seed{seed}.json")
            with open(logs_path, "w") as f:
                json.dump(results.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved logs → {logs_path}")

            conv_path = os.path.join(RESULTS_DIR, f"conversations_v5fixedos_{run_timestamp}_{cond}_seed{seed}.json")
            with open(conv_path, "w") as f:
                json.dump(results.conversations_to_dict(), f, indent=2, default=str)
            logger.info(f"Saved conversations → {conv_path}")
