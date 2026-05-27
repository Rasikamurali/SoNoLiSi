"""
SoNoLiSi – Collaborative Writing Variant
=========================================
Agents co-write a news article about climate change on a live Google Doc.
The social norm being studied: "Always reference a point made by a previous
contributor before adding your own paragraph."

Setup (Google Service Account):
  1. Go to https://console.cloud.google.com → create/select a project
  2. Enable "Google Docs API" and "Google Drive API"
  3. IAM & Admin → Service Accounts → Create service account → Download JSON key
  4. Save the JSON key as  code/credentials.json
  5. Add to code/.env:
       GOOGLE_CREDENTIALS_PATH=credentials.json
       SHARE_WITH_EMAIL=rasika2murali1301@gmail.com   ← your personal Gmail
"""

from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

load_dotenv()

random.seed(1)
np.random.seed(1)

openai_client = OpenAI()
MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

N_AGENTS           = 8
N_SEEDED           = 3
ROUNDS             = 6
SPEAKERS_PER_ROUND = 3
TEMPERATURE        = 1.0

GROUP_GOAL    = "Collaboratively write a news article about climate change."
ARTICLE_TOPIC = "The urgency of addressing climate change: causes, impacts, and solutions"
NORM          = (
    "Always reference a specific point made by a previous contributor "
    "before adding your own paragraph."
)

GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/drive",
]

# Personal Gmail linked to the Google Cloud project — the doc is shared here.
SHARE_WITH_EMAIL = os.getenv("SHARE_WITH_EMAIL", "")

# If you pre-create a Google Doc manually and share it with the service account,
# paste its ID here (or in .env as GOOGLE_DOC_ID) to skip doc creation entirely.
# Get the ID from the URL: docs.google.com/document/d/<ID>/edit
GOOGLE_DOC_ID = os.getenv("GOOGLE_DOC_ID", "")

# ---------------------------------------------------------------------------
# Google Docs helpers
# ---------------------------------------------------------------------------

def setup_google_docs():
    """
    Authenticate via service account JSON.
    Path from GOOGLE_CREDENTIALS_PATH env var (default: credentials.json).
    Returns (docs_service, drive_service).
    """
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    creds = service_account.Credentials.from_service_account_file(
        creds_path, scopes=GOOGLE_SCOPES
    )
    docs_service  = build("docs",  "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)
    return docs_service, drive_service


def create_article_doc(docs_service, drive_service) -> tuple[str, str]:
    """
    Use a pre-existing Google Doc (GOOGLE_DOC_ID) if provided, otherwise
    try to create one via the Drive API.
    Returns (doc_id, view_url).
    """
    if GOOGLE_DOC_ID:
        # Use the doc the user already created and shared with the service account.
        # Just write the heading into it (the doc must be empty or this prepends).
        doc_id = GOOGLE_DOC_ID
        print(f"  Using pre-existing doc: {doc_id}")
        try:
            # Clear any existing content first, then insert heading
            doc = docs_service.documents().get(documentId=doc_id).execute()
            end_index = doc.get("body", {}).get("content", [{}])[-1].get("endIndex", 2)
            requests = []
            if end_index > 2:
                # Delete existing content (leave the mandatory trailing newline at index 1)
                requests.append({
                    "deleteContentRange": {
                        "range": {"startIndex": 1, "endIndex": end_index - 1}
                    }
                })
            requests.append({
                "insertText": {
                    "location": {"index": 1},
                    "text": f"{ARTICLE_TOPIC}\n\n",
                }
            })
            docs_service.documents().batchUpdate(
                documentId=doc_id, body={"requests": requests}
            ).execute()
            print("  Heading written OK")
        except Exception as e:
            print(f"  Warning: could not write heading ({e}). Continuing anyway.")
    else:
        # Fallback: try to create a new doc via Drive API.
        # Requires Drive storage quota to be available.
        print("  [1/3] Creating document via Drive API...")
        try:
            doc = drive_service.files().create(
                body={
                    "name": "Climate Change Article – Agent Simulation",
                    "mimeType": "application/vnd.google-apps.document",
                },
                fields="id",
            ).execute()
            doc_id = doc["id"]
            print(f"        OK — doc id: {doc_id}")
        except Exception as e:
            print(f"        FAILED: {e}")
            print("\n  Drive quota full or API not enabled.")
            print("  Fix: create a doc manually, share it with the service account,")
            print("  and set GOOGLE_DOC_ID=<doc-id> in .env")
            raise

        print("  [2/3] Inserting heading...")
        docs_service.documents().batchUpdate(
            documentId=doc_id,
            body={
                "requests": [
                    {
                        "insertText": {
                            "location": {"index": 1},
                            "text": f"{ARTICLE_TOPIC}\n\n",
                        }
                    }
                ]
            },
        ).execute()
        print("        OK")

        print(f"  [3/3] Sharing with {SHARE_WITH_EMAIL}...")
        if SHARE_WITH_EMAIL:
            try:
                drive_service.permissions().create(
                    fileId=doc_id,
                    body={"type": "user", "role": "writer",
                          "emailAddress": SHARE_WITH_EMAIL},
                    sendNotificationEmail=False,
                ).execute()
                print("        OK")
            except Exception as e:
                print(f"        Warning: could not share ({e}).")

    view_url = f"https://docs.google.com/document/d/{doc_id}/edit"
    return doc_id, view_url


def read_doc_content(docs_service, doc_id: str) -> str:
    """Return the plain text body of the Google Doc."""
    doc = docs_service.documents().get(documentId=doc_id).execute()
    body_content = doc.get("body", {}).get("content", [])
    text_parts = []
    for element in body_content:
        paragraph = element.get("paragraph")
        if paragraph:
            for run in paragraph.get("elements", []):
                text_run = run.get("textRun")
                if text_run:
                    text_parts.append(text_run.get("content", ""))
    return "".join(text_parts)


def append_to_doc(docs_service, doc_id: str, agent_id: int, text: str) -> None:
    """Append a labelled paragraph to the end of the document."""
    doc = docs_service.documents().get(documentId=doc_id).execute()
    end_index = doc.get("body", {}).get("content", [{}])[-1].get("endIndex", 2) - 1

    docs_service.documents().batchUpdate(
        documentId=doc_id,
        body={
            "requests": [
                {
                    "insertText": {
                        "location": {"index": end_index},
                        "text": f"[Agent {agent_id}]\n{text}\n\n",
                    }
                }
            ]
        },
    ).execute()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

@dataclass
class Agent:
    agent_id: int
    knows_norm: bool
    compliance_count: int = 0
    speak_count: int = 0
    inferred_norm: str = "NONE"
    oughtness: float = 0.0

    def compliance_probability(self) -> float:
        if self.knows_norm:
            return 0.9
        return 0.1 + 0.7 * self.oughtness

    def speak(self, docs_service, doc_id: str) -> bool:
        """Generate a paragraph and append it to the Google Doc. Returns compliance."""
        self.speak_count += 1
        current_text = read_doc_content(docs_service, doc_id)
        will_comply = np.random.rand() < self.compliance_probability()

        system_prompt = (
            f"You are Agent {self.agent_id}, participating in a collaborative writing task.\n"
            f"Goal: {GROUP_GOAL}\n"
            f"Article topic: {ARTICLE_TOPIC}\n\n"
        )
        if will_comply and self.knows_norm:
            system_prompt += f"Important norm: {NORM}\n\n"

        if current_text.strip():
            snippet = current_text.strip()[-600:]
            user_prompt = (
                "Here is what has been written so far (last section):\n\n"
                f"---\n{snippet}\n---\n\n"
                "Write the next paragraph for the article "
                "(100–150 words, no headers, plain prose)."
            )
        else:
            user_prompt = (
                "The article is just starting. Write an engaging opening paragraph "
                "about climate change (100–150 words, no headers, plain prose)."
            )

        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        paragraph = response.choices[0].message.content.strip()
        append_to_doc(docs_service, doc_id, self.agent_id, paragraph)

        compliant = self._check_compliance(paragraph, current_text)
        if compliant:
            self.compliance_count += 1
        return compliant

    def _check_compliance(self, paragraph: str, prior_text: str) -> bool:
        """Ask the LLM whether the paragraph references a previous contributor."""
        if not prior_text.strip():
            return True  # First contribution — norm cannot apply yet

        judge_prompt = (
            f"Social norm: {NORM}\n\n"
            f"Previous article content (excerpt):\n{prior_text.strip()[-400:]}\n\n"
            f"New paragraph:\n{paragraph}\n\n"
            "Did the new paragraph follow the norm by explicitly referencing "
            "or building on a point from the previous content? "
            "Answer YES or NO only."
        )
        resp = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        return resp.choices[0].message.content.strip().upper().startswith("YES")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def compliance_rate(agent: Agent) -> float:
    if agent.speak_count == 0:
        return 0.0
    return agent.compliance_count / agent.speak_count


def softmax_select(agents: list[Agent]) -> list[Agent]:
    weights = np.array([1.0 + compliance_rate(a) for a in agents])
    scaled  = weights / TEMPERATURE
    probs   = np.exp(scaled - np.max(scaled))
    probs   = probs / probs.sum()
    idx     = np.random.choice(len(agents), SPEAKERS_PER_ROUND, replace=False, p=probs)
    return [agents[i] for i in idx]


# ---------------------------------------------------------------------------
# Perception
# ---------------------------------------------------------------------------

def elicit_perception(agent: Agent, round_idx: int,
                      prevalence: float, selection_advantage: float,
                      doc_snippet: str) -> None:
    prompt = f"""
You are observing a collaborative news article writing session about climate change.

Round {round_idx} statistics:
- Fraction of contributors who referenced a previous writer's point: {prevalence:.2f}
- Selection advantage of those contributors: {selection_advantage:.2f}

Recent article excerpt:
---
{doc_snippet[-400:] if doc_snippet else "(nothing written yet)"}
---

1. What do you believe the group's social norm is for how contributors should write?
2. On a scale from 0 to 1, how strongly do you believe members OUGHT to follow it?

Respond in this exact format:

Norm: <short description>
Oughtness: <number between 0 and 1>
"""
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = response.choices[0].message.content
    norm_match  = re.search(r"Norm:\s*(.*)", text)
    ought_match = re.search(r"Oughtness:\s*([0-9]*\.?[0-9]+)", text)
    agent.inferred_norm = norm_match.group(1).strip() if norm_match else "NONE"
    raw_ought = float(ought_match.group(1)) if ought_match else 0.0
    agent.oughtness = max(0.0, min(1.0, raw_ought))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class NormSimulation:

    def __init__(self):
        seeded = set(random.sample(range(N_AGENTS), N_SEEDED))
        self.agents = [Agent(i, i in seeded) for i in range(N_AGENTS)]

    def run(self) -> None:
        print("Setting up Google Docs...")
        docs_service, drive_service = setup_google_docs()
        doc_id, view_url = create_article_doc(docs_service, drive_service)
        print(f"\nGoogle Doc created — open it to watch agents write in real time:")
        print(f"  {view_url}\n")
        print("Seeded agents (know the norm):", [a.agent_id for a in self.agents if a.knows_norm])

        for r in range(1, ROUNDS + 1):
            print(f"\n===== ROUND {r} =====")
            speakers = softmax_select(self.agents)
            round_flags = []

            for agent in speakers:
                print(f"  Agent {agent.agent_id} writing...", end=" ", flush=True)
                compliant = agent.speak(docs_service, doc_id)
                round_flags.append(compliant)
                print("compliant" if compliant else "non-compliant")

            prevalence = float(np.mean(round_flags))
            pop_share  = float(np.mean([compliance_rate(a) > 0.5 for a in self.agents]))
            spk_share  = float(np.mean([compliance_rate(a) > 0.5 for a in speakers]))
            selection_advantage = spk_share - pop_share

            doc_snippet = read_doc_content(docs_service, doc_id)
            for agent in self.agents:
                elicit_perception(agent, r, prevalence, selection_advantage, doc_snippet)
                print(f"  Agent {agent.agent_id} — norm: {agent.inferred_norm!r}  oughtness: {agent.oughtness:.2f}")
                if not agent.knows_norm and agent.oughtness > 0.8:
                    if random.random() < agent.oughtness:
                        agent.knows_norm = True
                        print(f"    *** Agent {agent.agent_id} adopted the norm! ***")

        print("\n" + "=" * 50)
        print("SIMULATION COMPLETE")
        print("Final norm adopters:", [a.agent_id for a in self.agents if a.knows_norm])
        print(f"Full article: {view_url}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sim = NormSimulation()
    sim.run()
