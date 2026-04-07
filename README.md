---
title: CiteGuardian Environment Server
emoji: �
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# CiteGuardian

A reinforcement learning environment that simulates a professional peer-review workflow. The agent ingests a research paper and performs a multi-step logical audit to find structural omissions, citation orphans, and factual contradictions.

## Task Levels

Three difficulty levels are randomly selected on each `reset()`:

| Level | Name | Goal |
|-------|------|------|
| A | Structural Integrity | Find a missing mandatory section (e.g. Results) |
| B | Citation Synchronization | Find citation markers cited in text but absent from References, or vice-versa |
| C | Factual Contradiction | Find numeric mismatches between sections (e.g. 100 subjects → 85 subjects) |

## Reward Structure

| Event | Reward |
|-------|--------|
| First visit to a required section | +0.02 |
| Correct `FLAG_ERROR` on a seeded mistake | +0.30 |
| False positive flag | -0.10 |
| Every step taken | -0.01 |
| Completion bonus (scales with recall) | +0.10 → +0.40 |
| Perfect run (100% recall, 0 false positives) | clamped to **1.0** |

## Quick Start

```python
import asyncio
from citeGuardian import CiteguardianAction, CiteguardianEnv

async def main():
    env = await CiteguardianEnv.from_docker_image("openenv-citeguardian")
    result = await env.reset()
    print(result.observation.message)        # task description
    print(result.observation.task_level)     # 'A', 'B', or 'C'

    # Navigate to a section
    result = await env.step(CiteguardianAction(
        action_type="GO_TO", section_name="Methods"
    ))

    # Scan citations in current section
    result = await env.step(CiteguardianAction(action_type="SCAN_CITATIONS"))
    print(result.observation.tool_result)    # list of [N] markers

    # Compare two values
    result = await env.step(CiteguardianAction(
        action_type="COMPARE_VALUES", val1="100", val2="85"
    ))

    # Flag an error
    result = await env.step(CiteguardianAction(
        action_type="FLAG_ERROR",
        error_type="LOGICAL_INCONSISTENCY",
        text_snippet="85 subjects"
    ))

    # Submit
    result = await env.step(CiteguardianAction(action_type="SUBMIT"))
    print(result.reward)

    await env.close()

asyncio.run(main())
```

## Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `GO_TO` | `section_name` | Navigate to a paper section |
| `SCAN_CITATIONS` | — | List all `[N]` markers in current section |
| `COMPARE_VALUES` | `val1`, `val2` | Check if two values conflict |
| `FLAG_ERROR` | `error_type`, `text_snippet` | Flag a confirmed error |
| `SUBMIT` | — | End the audit |

**Error types for `FLAG_ERROR`:**
- `STRUCTURAL_ERROR` — mandatory section is missing
- `ORPHAN_CITATION` — citation marker has no matching reference entry, or vice-versa
- `LOGICAL_INCONSISTENCY` — numeric/factual contradiction between sections

## Observation Space

```python
CiteguardianObservation(
    current_view   # str  — up to ~1000 chars of the current section
    metadata       # dict — available_sections, visited_sections, citation_markers, etc.
    audit_log      # list — history of actions taken this episode
    tool_result    # any  — output of SCAN_CITATIONS or COMPARE_VALUES
    message        # str  — environment feedback
    task_level     # str  — 'A', 'B', or 'C'
    reward         # float — cumulative reward so far
    done           # bool
)
```

## Setup

### Build Docker image

```bash
docker build -t openenv-citeguardian .
```

### Run inference

```bash
# Copy and fill in your credentials
cp .env.example .env   # set HF_TOKEN, API_BASE_URL, MODEL_NAME, LOCAL_IMAGE_NAME

uv run python inference.py
```

### Pre-validate before submission

```bash
python prevalidation.py https://your-space.hf.space .
```

### Deploy to Hugging Face Spaces

```bash
openenv push
# or to a specific repo:
openenv push --repo-id my-org/citeguardian
```

## Project Structure

```
citeGuardian/
├── __init__.py                          # Package exports
├── client.py                            # CiteguardianEnv async client
├── models.py                            # Action & Observation models
├── inference.py                         # LLM agent loop
├── prevalidation.py                     # Pre-submission validator
├── openenv.yaml                         # OpenEnv manifest
├── pyproject.toml                       # Project metadata & dependencies
├── Dockerfile                           # Container image
└── server/
    ├── app.py                           # FastAPI application
    └── citeGuardian_environment.py      # Core RL environment logic
```
