"""
Inference Script — CiteGuardian
===================================
MANDATORY env vars:
    API_BASE_URL        LLM endpoint
    MODEL_NAME          Model identifier
    HF_TOKEN            Hugging Face / API key
    LOCAL_IMAGE_NAME    Docker image name  (used when ENV_URL is not set)
    ENV_URL             Direct server URL  (takes priority over Docker image)

STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from citeGuardian import CiteguardianAction, CiteguardianEnv
import asyncio
import json
import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL") or os.getenv(
    "SERVER_URL")   # direct URL takes priority
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")  # validator injects API_KEY first
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")  # validator injects this
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("CITEGUARDIAN_TASK", "audit")
BENCHMARK = os.getenv("CITEGUARDIAN_BENCHMARK", "citeGuardian")

MAX_STEPS = 30
TEMPERATURE = 0.2
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.5

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
          STRUCTURAL_ERROR      — a mandatory section is missing
          ORPHAN_CITATION       — a [N] marker cited in text has no References entry, or vice-versa
          LOGICAL_INCONSISTENCY — numeric/factual contradiction between sections

    {"action_type": "SUBMIT"}
        End the audit after flagging all errors you are confident about.

    MANDATORY SECTIONS: Abstract, Introduction, Methods, Results, Discussion, References

    STRATEGY BY TASK LEVEL (shown in observation as task_level):

    Task A — Structural Integrity:
      1. Check metadata.available_sections against the mandatory list.
      2. FLAG_ERROR STRUCTURAL_ERROR with text_snippet=<missing section name>.
      3. SUBMIT.

    Task B — Citation Synchronization:
      1. GO_TO each body section and SCAN_CITATIONS to collect all [N] markers in text.
      2. GO_TO References and SCAN_CITATIONS to collect defined [N] markers.
      3. FLAG_ERROR ORPHAN_CITATION for each [N] in text but not in References (snippet="[N]").
      4. FLAG_ERROR ORPHAN_CITATION for each [N] in References but never cited (snippet="[N]").
      5. SUBMIT.

    Task C — Factual Contradiction:
      1. GO_TO Methods, note numeric claims.
      2. GO_TO Results, note the same numeric claims.
      3. COMPARE_VALUES with the two numbers.
      4. If conflict_detected=true → FLAG_ERROR LOGICAL_INCONSISTENCY, snippet=the Results number.
      5. SUBMIT.

    CRITICAL RULES:
    - False positives cost -0.10. Only flag when certain.
    - Never flag the same error twice.
    - Respond with ONLY a valid JSON object — no markdown, no explanation.
""").strip()


def _obs_to_user_prompt(step: int, result) -> str:
    o = result.observation if hasattr(result, "observation") else result
    recent_log = o.audit_log[-5:] if getattr(o, "audit_log", None) else []
    available = o.metadata.get("available_sections", []) if getattr(
        o, "metadata", None) else []
    visited = o.metadata.get("visited_sections", []) if getattr(
        o, "metadata", None) else []
    missing = [s for s in ["Abstract", "Introduction", "Methods", "Results", "Discussion", "References"]
               if s not in available]
    return textwrap.dedent(f"""
        Step: {step}
        Task level: {getattr(o, 'task_level', '?')}  ← use the strategy for this task level
        Current section: {o.metadata.get('current_section', '?') if getattr(o, 'metadata', None) else '?'}
        Available sections: {available}
        MISSING mandatory sections: {missing or 'none'}
        Visited sections: {visited}
        Citation markers in current view: {o.metadata.get('citation_markers_in_view', []) if getattr(o, 'metadata', None) else []}
        All citation markers across paper: {o.metadata.get('all_paper_citations', []) if getattr(o, 'metadata', None) else []}
        Flags raised so far: {o.metadata.get('flags_raised', 0) if getattr(o, 'metadata', None) else 0}
        Last tool result: {json.dumps(getattr(o, 'tool_result', None))}
        Environment message: {getattr(o, 'message', '')}
        Current view:
        {getattr(o, 'current_view', '')[:800]}
        Recent audit log:
        {json.dumps(recent_log, indent=2)}
        What is your next action?
    """).strip()


# ---------------------------------------------------------------------------
# Logging
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
# LLM
# ---------------------------------------------------------------------------

def get_model_action(client: OpenAI, step: int, result, messages: List[ChatCompletionMessageParam]) -> CiteguardianAction:
    messages.append(
        {"role": "user", "content": _obs_to_user_prompt(step, result)})
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
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return CiteguardianAction(**json.loads(raw))
    except Exception as exc:
        print(f"[DEBUG] Model/parse error at step {step}: {exc}", flush=True)
        return CiteguardianAction(action_type="SUBMIT")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None

    # Emit [START] immediately so the validator always sees it
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    # Debug: confirm we're using the validator's API endpoint
    print(f"[DEBUG] Using API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Using MODEL_NAME: {MODEL_NAME}", flush=True)

    try:
        # Connect — prefer a direct URL, fall back to Docker image
        if ENV_URL:
            try:
                env = CiteguardianEnv(base_url=ENV_URL)
                await env.connect()
            except Exception as conn_exc:
                print(f"[DEBUG] Failed to connect to {ENV_URL}: {conn_exc}", flush=True)
                raise
        elif IMAGE_NAME:
            try:
                env = await CiteguardianEnv.from_docker_image(IMAGE_NAME)
            except Exception as docker_exc:
                print(f"[DEBUG] Failed to start Docker image {IMAGE_NAME}: {docker_exc}", flush=True)
                raise
        else:
            raise ValueError(
                "Neither ENV_URL nor LOCAL_IMAGE_NAME is set. "
                "Set ENV_URL to connect to a running server, or LOCAL_IMAGE_NAME to start a Docker container."
            )

        client_api = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT}]

        result = await env.reset()
        last_obs_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            done = getattr(result, "done", False)
            if not done:
                obs = result.observation if hasattr(
                    result, "observation") else result
                done = getattr(obs, "done", False)
            if done:
                break

            action = get_model_action(client_api, step, result, messages)
            action_str = action.model_dump_json(exclude_none=True)

            try:
                result = await env.step(action)
                obs = result.observation if hasattr(
                    result, "observation") else result
                reward = float(getattr(result, "reward", None)
                               or getattr(obs, "reward", 0.0))
                done = getattr(result, "done", False) or getattr(
                    obs, "done", False)
                last_obs_reward = float(getattr(obs, "reward", reward))
                error_msg = None
            except Exception as step_exc:
                print(f"[DEBUG] Step {step} error: {step_exc}", flush=True)
                reward = 0.0
                done = True
                error_msg = str(step_exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str,
                     reward=reward, done=done, error=error_msg)

            if done:
                break

        score = min(max(last_obs_reward, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except ValueError as ve:
        # Config error — missing env vars
        print(f"[DEBUG] Configuration error: {ve}", flush=True)
    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        import traceback
        traceback.print_exc()

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken,
                score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
