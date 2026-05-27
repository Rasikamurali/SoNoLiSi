"""
Playing around to see what happens with norms and multi-LLM teams. 


"""

import numpy as np 
import pandas as pd 
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
from huggingface_hub import login
from datasets import load_dataset
from tqdm import tqdm
import regex as re
import json
import random
from collections import Counter

# Set up device and model
hf_token = "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"
login(token=hf_token)

set_seed(1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
cache_dir = "/data3/models/hub"
data_dir = "/data/rasimura/data/rasimura/datasets"
# model1_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model2_name = "mistralai/Mistral-7B-Instruct-v0.3"
# model3_name = "Qwen/Qwen2.5-7B-Instruct"

model_names = {
    "Agent 1": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Agent 2": "mistralai/Mistral-7B-Instruct-v0.3",
    "Agent 3": "Qwen/Qwen2.5-7B-Instruct"
}

# ---------------- Per-model generation settings ----------------
MODEL_GEN_KWARGS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "max_new_tokens": 220,
        "do_sample": True,          # enable sampling so temperature/top_p matter
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "max_new_tokens": 220,
        "do_sample": True,
        "temperature": 0.4,
        "top_p": 0.92,
        "repetition_penalty": 1.08,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "max_new_tokens": 220,
        "do_sample": True,
        "temperature": 0.35,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
    },
}


class AIAgent:
    """Initializer-only: each agent loads its own model + tokenizer from model_names."""
    def __init__(
        self,
        name: str,
        model_name: str,
        *,
        device: str = device,
        cache_dir: str = cache_dir,
        dtype = torch.bfloat16,
        attn_impl: str = "flash_attention_2",
    ):
        self.name = name
        self.model_name = model_name
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation=attn_impl,  # set to "eager" if FA2 not available
            cache_dir=cache_dir,
        ).eval().to(device)

# Build agents: one model per agent per your mapping
agents = [AIAgent(name, model_names[name]) for name in model_names]

def get_model_kwargs(agent_model_name: str, overrides: dict | None = None):
    base = MODEL_GEN_KWARGS.get(agent_model_name, {}).copy()
    if overrides:
        base.update(overrides)
    return base


def _render_chat(tokenizer, chat):
    """Render chat messages safely using the model's chat template."""
    try:
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    except Exception:
        # fallback formatting
        parts = []
        for m in chat:
            parts.append(f"{m['role'].upper()}: {m['content']}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)


def get_responses(model, tokenizer, prompts, model_kwargs=None, model_name=None):
    """
    Generate model responses for a list of prompts.
    Each prompt can be either:
      - a string (plain text prompt), or
      - a list of chat messages (dicts with 'role' and 'content').

    model_name (str): Optional; used to look up per-model settings in MODEL_GEN_KWARGS.
    """
    # 1️⃣ Combine model-specific kwargs if available
    final_kwargs = {}
    if "MODEL_GEN_KWARGS" in globals():
        name = model_name or getattr(model.config, "_name_or_path", None)
        if name in MODEL_GEN_KWARGS:
            final_kwargs.update(MODEL_GEN_KWARGS[name])
    if model_kwargs:
        final_kwargs.update(model_kwargs)
    if not final_kwargs:
        final_kwargs = {"max_new_tokens": 200, "do_sample": False, "repetition_penalty": 1.05}

    # 2️⃣ Format prompts
    if prompts and isinstance(prompts[0], list):
        formatted_prompts = [_render_chat(tokenizer, chat) for chat in prompts]
    else:
        formatted_prompts = prompts

    tokenizer.padding_side = "left"
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # 3️⃣ Generate
    input_lengths = inputs.attention_mask.sum(dim=1)
    outputs = model.generate(**inputs, **model_kwargs, pad_token_id=tokenizer.eos_token_id)

    # 4️⃣ Decode only new tokens
   
    responses = []
    for i in range(outputs.size(0)):
        start = input_lengths[i].item()
        responses.append(outputs[i, start:])
    return tokenizer.batch_decode(responses, skip_special_tokens=True)

# --- Prompt builders ---

def make_initial_prompts(agents):
    """
    Returns:
      prompts: list[chat] one per agent
      index_map: list[tuple] mapping back (entry_idx, agent_idx). Here we use entry_idx=0 for a single society.
    """
    prompts, index_map = [], []
    for agent_idx, _ in enumerate(agents):
        chat = [
            {
                "role": "system",
                "content": (
                    f"You are an agent and your agent id is: {agent_idx}. You are a college student and are about to attend a party."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"You are agent {agent_idx}, a college student. Please introduce yourself to the other agents "
                    "by stating your agent id. Do not give yourself a new name. Please state your feelings towards being at this party."
                ),
            },
        ]
        prompts.append(chat)
        index_map.append((0, agent_idx))  # single entry index = 0
    return prompts, index_map

def build_conversation_context(entry, upto_round, max_chars=None):
    """
    Assemble the full conversation up to `upto_round` (inclusive).
    Rounds are joined in chronological order: Initial -> Round 2 -> ... -> Round N.
    Set `max_chars` to cap very long histories (keeps the tail).
    """
    parts = []

    # Initial round
    init = entry.get("Initial round", {})
    if init:
        parts.append("=== Initial round ===")
        for agent_idx in sorted(init.keys()):
            parts.append(f"agent {agent_idx}: {init[agent_idx].strip()}")

    # Subsequent rounds (2..upto_round)
    for r in range(2, upto_round + 1):
        rr = entry.get(f"Round {r}", {})
        if rr:
            parts.append(f"=== Round {r} ===")
            for agent_idx in sorted(rr.keys()):
                parts.append(f"agent {agent_idx}: {rr[agent_idx].strip()}")

    context = " | ".join(parts)

    if max_chars is not None and len(context) > max_chars:
        # Keep the tail (most recent info) when truncating
        context = context[-max_chars:]
        context = context[context.find(" | ") + 3:] if " | " in context else context

    return context


def make_reflection_prompts(entries, round_num, agents, *, max_context_chars=None):
    """
    entries: list of dicts with keys "Initial round", "Round 2", ...
    Each entries[k][round_key] maps agent_idx -> response string.
    round_num: the round you're about to run (e.g., 2, 3, ...)
    """
    prompts, index_map = [], []

    for entry_idx, entry in enumerate(entries):
        # Build the full conversation up to the previous round
        upto_prev = (round_num - 1) if round_num > 1 else 1  # 1 => only "Initial round"
        context = build_conversation_context(entry, upto_prev, max_chars=max_context_chars)
        print(context)

        for agent_idx, _ in enumerate(agents):
            chat = [
    {
        "role": "system",
        "content": (
            f"You are agent {agent_idx}. You are a college student and are about to attend a party."
        ),
    },
    {
        "role": "user",
        "content": (
            "You are going for a party with some other college students"
            f"Here is the conversation you've had so far:\n\n{context}\n\n"
            "Acknowledge the other agents."
        ),
    },
]

            prompts.append(chat)
            index_map.append((entry_idx, agent_idx))

    return prompts, index_map


# --- Recording utility ---


import json

def to_json_output(entries):
    """
    Converts your `entries` (as built in the previous flow) into:
    {
      "initial response": {"agent 1": "...", "agent 2": "...", ...},
      "reflection rounds": {
        "round 2": {"agent 1": "...", ...},
        "round 3": {"agent 1": "...", ...},
        ...
      }
    }
    """
    # assuming one society; extend to loop if you keep multiple
    entry = entries[0] if entries else {}

    out = {
        "initial response": {},
        "reflection rounds": {}
    }

    # Initial
    for agent_idx, text in entry.get("Initial round", {}).items():
        out["initial response"][f"agent {agent_idx+1}"] = text.strip()

    # Rounds
    for key, val in entry.items():
        if key.startswith("Round "):
            try:
                round_num = int(key.split()[1])
            except Exception:
                continue
            out["reflection rounds"][f"round {round_num}"] = {
                f"agent {aid+1}": resp.strip() for aid, resp in val.items()
            }

    return out

# After you've run initial + reflection rounds and populated `entries`:


def record_outputs(entries, index_map, outputs, round_key):
    """
    Writes model outputs back into entries[entry_idx][round_key][agent_idx] = text
    """
    for (entry_idx, agent_idx), text in zip(index_map, outputs):
        if round_key not in entries[entry_idx]:
            entries[entry_idx][round_key] = {}
        entries[entry_idx][round_key][agent_idx] = text.strip()


# --- Main flow ---


from collections import defaultdict

entries = [ {} ]  # multiple societies? add more dicts here

# --- Initial round ---
initial_prompts, init_index_map = make_initial_prompts(agents)

# Group initial prompts by agent index
init_prompts_by_agent = defaultdict(list)
init_indices_by_agent = defaultdict(list)
for prompt, (entry_idx, agent_idx) in zip(initial_prompts, init_index_map):
    init_prompts_by_agent[agent_idx].append(prompt)
    init_indices_by_agent[agent_idx].append((entry_idx, agent_idx))

# Generate per agent with the right model/tokenizer
for agent_idx, agent in enumerate(agents):
    bucket = init_prompts_by_agent[agent_idx]
    if not bucket:
        continue
    kwargs = get_model_kwargs(agent.model_name)
    outs = get_responses(
    agent.model,
    agent.tokenizer,
    bucket,
    model_kwargs=kwargs,
    model_name=agent.model_name,   # ✅ add this line
)

    record_outputs(entries, init_indices_by_agent[agent_idx], outs, "Initial round")


# --- Reflection rounds ---
num_reflection_rounds = 5
for r in range(2, 2 + num_reflection_rounds):
    reflection_prompts, refl_index_map = make_reflection_prompts(entries, r, agents)

    refl_prompts_by_agent = defaultdict(list)
    refl_indices_by_agent = defaultdict(list)
    for prompt, (entry_idx, agent_idx) in zip(reflection_prompts, refl_index_map):
        refl_prompts_by_agent[agent_idx].append(prompt)
        refl_indices_by_agent[agent_idx].append((entry_idx, agent_idx))

    for agent_idx, agent in enumerate(agents):
        bucket = refl_prompts_by_agent[agent_idx]
        if not bucket:
            continue
        kwargs = get_model_kwargs(agent.model_name)
        outs = get_responses(agent.model, agent.tokenizer, bucket, model_kwargs=kwargs)
        record_outputs(entries, refl_indices_by_agent[agent_idx], outs, f"Round {r}")

# --- Final JSON ---
final_json = to_json_output(entries)



output_filename = '/data3/rasimura/social-norm-evo/results/collegeparty1.json'
with open(output_filename, 'w') as f:
    json.dump(final_json, f, indent=4)
