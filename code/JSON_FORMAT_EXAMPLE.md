# Simulation Data JSON Format

The `save_simulation_data()` function creates a comprehensive JSON file tracking all aspects of the simulation.

## JSON Structure

```json
{
  "metadata": {
    "timestamp": "2026-02-20T10:30:45.123456",
    "simulation_completed": true
  },

  "configuration": {
    "n_agents": 10,
    "n_seeded": 3,
    "seeded_agents": [0, 3, 7],
    "norm": "Always acknowledge others' contributions before making your own",
    "group_goal": "Edit a community standards policy draft",
    "n_rounds": 10,
    "selection_temperature": 1.0
  },

  "initial_policy": {
    "sections": {
      "Purpose": "This policy establishes community standards...",
      "Participation": "All community members have the right...",
      "Decision Process": "To be determined through group discussion."
    }
  },

  "rounds": [
    {
      "round_index": 0,
      "speakers": [2, 5, 8],

      "messages": [
        {
          "agent_id": 2,
          "text": "I propose amending the Purpose section to emphasize collaboration..."
        },
        {
          "agent_id": 5,
          "text": "I support Agent 2's proposal and suggest we also add..."
        },
        {
          "agent_id": 8,
          "text": "I'd like to add a new section on Conflict Resolution..."
        }
      ],

      "contributions": [
        {
          "agent_id": 2,
          "type": "amendment",
          "section": "Purpose",
          "content": "emphasize collaboration and mutual respect",
          "impact_score": 3.0,
          "endorsed_by": [5],
          "references": []
        },
        {
          "agent_id": 5,
          "type": "endorsement",
          "section": null,
          "content": "support Agent 2's proposal",
          "impact_score": 1.0,
          "endorsed_by": [],
          "references": [0]
        },
        {
          "agent_id": 8,
          "type": "addition",
          "section": "new",
          "content": "Conflict Resolution section with mediation process",
          "impact_score": 3.0,
          "endorsed_by": [],
          "references": []
        }
      ],

      "agent_perceptions": {
        "0": {
          "norm_scores": {
            "2": 1.0,
            "5": 1.0,
            "8": 0.0
          },
          "contribution_scores": {
            "2": 1.5,
            "5": 1.0,
            "8": 1.5
          }
        },
        "1": {
          "norm_scores": {...},
          "contribution_scores": {...}
        },
        ...
      },

      "leading_contributor": {
        "agent_id": 2,
        "cumulative_impact": 3.0
      },

      "policy_state": {
        "sections": {
          "Purpose": "This policy establishes community standards for collaboration and mutual respect...",
          "Participation": "All community members have the right...",
          "Decision Process": "To be determined through group discussion.",
          "Addition_0": "Conflict Resolution section with mediation process"
        },
        "version_count": 2
      }
    },

    // ... more rounds ...
  ],

  "final_results": {
    "policy": {
      "sections": {
        "Purpose": "Final version...",
        "Participation": "Final version...",
        "Decision Process": "Final version...",
        "Conflict Resolution": "Final version..."
      },
      "total_changes": 15,
      "version_history": [
        {
          "round": 0,
          "agent_id": 2,
          "description": "Amended section 'Purpose'"
        },
        {
          "round": 0,
          "agent_id": 8,
          "description": "Added/expanded section 'Addition_0'"
        },
        ...
      ]
    },

    "contributions": {
      "total": 28,
      "by_agent": {
        "0": {
          "count": 3,
          "total_impact": 7.5,
          "by_type": {
            "amendment": 2,
            "endorsement": 1
          }
        },
        "2": {
          "count": 5,
          "total_impact": 12.3,
          "by_type": {
            "amendment": 3,
            "addition": 2
          }
        },
        ...
      }
    },

    "primary_contributors": [
      {
        "agent_id": 2,
        "total_impact_score": 12.3
      },
      {
        "agent_id": 5,
        "total_impact_score": 9.8
      },
      {
        "agent_id": 8,
        "total_impact_score": 8.5
      }
    ],

    "speaker_statistics": {
      "unique_speakers": 8,
      "total_agents": 10,
      "diversity_percentage": 80.0,
      "speaking_opportunities": {
        "0": 2,
        "1": 1,
        "2": 4,
        "3": 3,
        "5": 4,
        "7": 2,
        "8": 3,
        "9": 1
      },
      "max_opportunities": 4,
      "min_opportunities": 1
    }
  }
}
```

## Key Features

### Per-Round Tracking
Each round captures:
- **Speakers**: Which agents were selected
- **Messages**: Complete conversation text
- **Contributions**: Parsed contribution type, section, content, and impact
- **Agent Perceptions**: Each agent's view of norm adherence and contribution quality
- **Leading Contributor**: Current top contributor by cumulative impact
- **Policy State**: Snapshot of policy sections at that point

### Agent Perceptions
For each agent, tracks their beliefs about all other agents:
- **norm_scores**: How well each agent follows the social norm
- **contribution_scores**: How valuable each agent's contributions are

### Final Results
Comprehensive summary including:
- Complete final policy
- Version history of all changes
- Contribution statistics by agent and type
- Primary contributors (stochastically selected)
- Speaker diversity metrics

## Usage

```python
from SoNoLiSi import NormPropagationSim

# Run simulation
sim = NormPropagationSim(
    n_agents=10,
    n_seeded=3,
    norm="Acknowledge others before proposing",
    group_goal="Edit community policy",
    n_rounds=10,
    seed=42
)
sim.run()

# Save to JSON
sim.save_simulation_data("my_simulation.json")

# Or with auto-generated timestamped filename
sim.save_simulation_data()  # Creates: simulation_data_20260220_103045.json
```

## Analysis Examples

```python
import json

# Load saved data
with open("simulation_data.json") as f:
    data = json.load(f)

# Analyze norm adherence over time
for round_data in data["rounds"]:
    round_idx = round_data["round_index"]

    # Get agent 0's perception of norm-following
    agent_0_perceptions = round_data["agent_perceptions"]["0"]
    norm_scores = agent_0_perceptions["norm_scores"]

    print(f"Round {round_idx}: Agent 0 sees norm scores: {norm_scores}")

# Find most influential contributor
final_contribs = data["final_results"]["contributions"]["by_agent"]
for agent_id, stats in final_contribs.items():
    print(f"Agent {agent_id}: {stats['total_impact']} total impact, {stats['count']} contributions")

# Track policy evolution
for vh in data["final_results"]["policy"]["version_history"]:
    print(f"Round {vh['round']}: Agent {vh['agent_id']} - {vh['description']}")
```
