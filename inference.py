"""
Inference Script — CiteGuardian
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    Docker image name (if using from_docker_image())

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from citeGuardian import CiteguardianAction, CiteguardianEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("CITEGUARDIAN_TASK", "audit")
BENCHMARK = os.getenv("CITEGUARDIAN_BENCHMARK", "citeGuardian")

MAX_STEPS = 30          # enough headroom for a full audit
TEMPERATURE = 0.2       # low temp for precise reasoning
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.5   # final reward >= 0.5 counts as success

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert academic peer-reviewer auditing a research paper.
    You have access to the following tools. Respond with EXACTLY one JSON object per turn.

    TOOLS:
    {"action_type": "GO_TO", "section_name": "<name>"}
        Navigate to a section. Use metadata.available_sections to see what exists.

    {"action_type": "SCAN_CITATIONS"}
        Returns all citation markers (e.g. [1], [2]) found in the current section.

    {"action_type": "COMPARE_VALUES", "val1": "<v1>", "val2": "<v2>"}
        Checks if two values conflict. Use for numeric mismatches between sections.

    {"action_type": "FLAG_ERROR", "error_type": "<type>", "text_snippet": "<snippet>"}
        Flag a confirmed error. Types:
          STRUCTURAL_ERROR      — a mandatory section (Abstract/Introduction/Methods/Results/Discussion/References) is missing
          ORPHAN_CITATION       — a [N] marker cited in text has no entry in References, OR a References entry is never cited
          LOGICAL_INCONSISTENCY — a numeric/factual value in one section contradicts another

    {"action_type": "SUBMIT"}
        End the audit. Call this only after you have flagged all errors you are confident about.

    MANDATORY SECTIONS: Abstract, Introduction, Methods, Results, Discussion, References

    STRATEGY BY TASK LEVEL (shown in observation as task_level):

    Task A — Structural Integrity:
      1. Use GO_TO on every mandatory section name.
      2. If a section is NOT in metadata.available_sections, it is missing.
      3. FLAG_ERROR with error_type=STRUCTURAL_ERROR and text_snippet=<missing section name>.
      4. SUBMIT immediately after flagging.

    Task B — Citation Synchronization:
      1. GO_TO each body section (Abstract, Introduction, Methods, Results, Discussion).
      2. SCAN_CITATIONS in each to collect all [N] markers used in the text.
      3. GO_TO References and SCAN_CITATIONS to collect all [N] markers defined there.
      4. Any [N] cited in text but absent from References → FLAG_ERROR ORPHAN_CITATION, snippet="[N]".
      5. Any [N] in References but never cited in text → FLAG_ERROR ORPHAN_CITATION, snippet="[N]".
      6. SUBMIT after all orphans are flagged.

    Task C — Factual Contradiction:
      1. GO_TO Methods, note any numeric claims (subject counts, sample sizes).
      2. GO_TO Results, note the same numeric claims.
      3. COMPARE_VALUES with the two numbers.
      4. If conflict_detected=true → FLAG_ERROR LOGICAL_INCONSISTENCY, snippet must contain the number from Results (e.g. "85").
      5. SUBMIT after flagging.

    CRITICAL RULES:
    - False positives cost -0.10 each. Only flag when certain.
    - Do NOT flag the same error twice.
    - Do NOT flag a LOGICAL_INCONSISTENCY on Task A or B.
    - The text_snippet for FLAG_ERROR must contain the key value/marker that identifies the error.
    - Respond with ONLY a valid JSON object — no markdown, no explanation.
""").strip()


def _obs_to_user_prompt(step: int, obs) -> str:
    """Convert an observation into a concise user-facing prompt for the LLM."""
    o = obs.observation if hasattr(obs, "observation") else obs
    recent_log = o.audit_log[-5:] if o.audit_log else []
    available = o.metadata.get("available_sections", [])
    visited = o.metadata.get("visited_sections", [])
    missing = [s for s in ["Abstract","Introduction","Methods","Results","Discussion","References"] if s not in available]

    return textwrap.dedent(f"""
        Step: {step}
        Task level: {o.task_level}  ← use the strategy for this task level
        Current section: {o.metadata.get('current_section', '?')}
        Available sections: {available}
        MISSING mandatory sections: {missing if missing else 'none'}
        Visited sections: {visited}
        Citation markers in current view: {o.metadata.get('citation_markers_in_view', [])}
        All citation markers across paper: {o.metadata.get('all_paper_citations', [])}
        Flags raised so far: {o.metadata.get('flags_raised', 0)}
        Last tool result: {json.dumps(o.tool_result)}
        Environment message: {o.message}
        Current view:
        {o.current_view[:800]}
        Recent audit log:
        {json.dumps(recent_log, indent=2)}
        What is your next action?
    """).strip()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_model_action(
    client: OpenAI,
    step: int,
    obs,
    messages: List[dict],
) -> CiteguardianAction:
    """Ask the LLM for the next action and parse it into a CiteguardianAction."""
    user_content = _obs_to_user_prompt(step, obs)
    messages.append({"role": "user", "content": user_content})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        messages.append({"role": "assistant", "content": raw})

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return CiteguardianAction(**data)

    except Exception as exc:
        print(f"[DEBUG] Model/parse error at step {step}: {exc}", flush=True)
        # Safe fallback: navigate to first unvisited section or submit
        return CiteguardianAction(action_type="SUBMIT")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def main() -> None:
    client_api = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await CiteguardianEnv.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # Persistent conversation history for the LLM
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation if hasattr(result, "observation") else result

        for step in range(1, MAX_STEPS + 1):
            done = getattr(result, "done", False) or getattr(obs, "done", False)
            if done:
                break

            action = get_model_action(client_api, step, result, messages)
            action_str = action.model_dump_json(exclude_none=True)

            try:
                result = await env.step(action)
                obs = result.observation if hasattr(result, "observation") else result
                reward = float(getattr(result, "reward", None) or getattr(obs, "reward", 0.0))
                done = getattr(result, "done", False) or getattr(obs, "done", False)
                error_msg = None
            except Exception as step_exc:
                print(f"[DEBUG] Step {step} error: {step_exc}", flush=True)
                reward = 0.0
                done = True
                error_msg = str(step_exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Final score: use the observation's reward field from the last step
        # (cumulative env reward, already in [0,1] after SUBMIT clamp)
        final_obs = obs if "obs" in dir() else None
        obs_reward = float(getattr(final_obs, "reward", 0.0)) if final_obs else 0.0
        score = max(obs_reward, rewards[-1] if rewards else 0.0)
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
