"""
Inference Script — CiteGuardian
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from citeGuardian import CiteguardianAction, CiteguardianEnv

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("CITEGUARDIAN_TASK", "audit")
BENCHMARK = os.getenv("CITEGUARDIAN_BENCHMARK", "citeGuardian")
ENV_URL = os.getenv("ENV_URL") or os.getenv("SERVER_URL")

MAX_STEPS = 30
TEMPERATURE = 0.2
MAX_TOKENS = 256
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert academic peer-reviewer auditing a research paper.
    You have access to the following tools. Respond with EXACTLY one JSON object per turn.

    TOOLS:
    {"action_type": "GO_TO", "section_name": "<name>"}
    {"action_type": "SCAN_CITATIONS"}
    {"action_type": "COMPARE_VALUES", "val1": "<v1>", "val2": "<v2>"}
    {"action_type": "FLAG_ERROR", "error_type": "<STRUCTURAL_ERROR|ORPHAN_CITATION|LOGICAL_INCONSISTENCY>", "text_snippet": "<snippet>"}
    {"action_type": "SUBMIT"}

    MANDATORY SECTIONS: Abstract, Introduction, Methods, Results, Discussion, References

    Task A: Check available_sections. If any mandatory section missing → FLAG_ERROR STRUCTURAL_ERROR snippet=<section name> → SUBMIT.
    Task B: SCAN_CITATIONS in each section. Cross-check with References. FLAG_ERROR ORPHAN_CITATION for each mismatch → SUBMIT.
    Task C: GO_TO Methods and Results. COMPARE_VALUES on numeric claims. If conflict → FLAG_ERROR LOGICAL_INCONSISTENCY → SUBMIT.

    RULES: Only flag when certain. Never flag same error twice. Reply ONLY with valid JSON.
""").strip()


def _obs_to_user_prompt(step: int, result) -> str:
    o = result.observation if hasattr(result, "observation") else result
    available = o.metadata.get("available_sections", []) if getattr(o, "metadata", None) else []
    missing = [s for s in ["Abstract", "Introduction", "Methods", "Results", "Discussion", "References"]
               if s not in available]
    return textwrap.dedent(f"""
        Step: {step} | Task: {getattr(o, 'task_level', '?')}
        Current section: {o.metadata.get('current_section', '?') if getattr(o, 'metadata', None) else '?'}
        Available: {available} | Missing: {missing or 'none'}
        Visited: {o.metadata.get('visited_sections', []) if getattr(o, 'metadata', None) else []}
        Citations in view: {o.metadata.get('citation_markers_in_view', []) if getattr(o, 'metadata', None) else []}
        All citations: {o.metadata.get('all_paper_citations', []) if getattr(o, 'metadata', None) else []}
        Flags raised: {o.metadata.get('flags_raised', 0) if getattr(o, 'metadata', None) else 0}
        Tool result: {json.dumps(getattr(o, 'tool_result', None))}
        Message: {getattr(o, 'message', '')}
        View: {getattr(o, 'current_view', '')[:600]}
        Log: {json.dumps(o.audit_log[-3:] if getattr(o, 'audit_log', None) else [])}
        Next action?
    """).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def get_model_action(client: OpenAI, step: int, result, messages: List[ChatCompletionMessageParam]) -> CiteguardianAction:
    messages.append({"role": "user", "content": _obs_to_user_prompt(step, result)})
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


async def main() -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if ENV_URL:
            env = CiteguardianEnv(base_url=ENV_URL)
            await env.connect()
        else:
            env = await CiteguardianEnv.from_docker_image(IMAGE_NAME)

        client_api = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": SYSTEM_PROMPT}]

        result = await env.reset()
        last_obs_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            done = getattr(result, "done", False)
            if not done:
                obs = result.observation if hasattr(result, "observation") else result
                done = getattr(obs, "done", False)
            if done:
                break

            action = get_model_action(client_api, step, result, messages)
            action_str = action.model_dump_json(exclude_none=True)

            try:
                result = await env.step(action)
                obs = result.observation if hasattr(result, "observation") else result
                reward = float(getattr(result, "reward", None) or getattr(obs, "reward", 0.0))
                done = getattr(result, "done", False) or getattr(obs, "done", False)
                last_obs_reward = float(getattr(obs, "reward", reward))
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

        score = min(max(last_obs_reward, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

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
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
