"""Baseline inference script for Cyberlytics AI."""

import asyncio
import os
import sys
from typing import List, Optional, Tuple

from openai import OpenAI

from cyberlytics_env import CyberlyticsAction, CyberlyticsEnv

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("CYBERLYTICS_TASK", "phishing_detection")
BENCHMARK = os.getenv("CYBERLYTICS_BENCHMARK", "cyberlytics_env")
MAX_STEPS = int(os.getenv("CYBERLYTICS_MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("CYBERLYTICS_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("CYBERLYTICS_MAX_TOKENS", "150"))
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("OPENENV_BASE_URL")


def _debug(message: str) -> None:
    print(f"[DEBUG] {message}", file=sys.stderr, flush=True)


def _format_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _format_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_text = error if error else "null"
    done_text = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_text} error={error_text}",
        flush=True,
    )


def _format_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{value:.2f}" for value in rewards)
    success_text = "true" if success else "false"
    print(
        f"[END] success={success_text} steps={steps} score={score:.2f} rewards={rewards_text}",
        flush=True,
    )


def _clip_score(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _decide_action(client: OpenAI, observation: dict) -> Tuple[str, Optional[str]]:
    system_prompt = (
        "You are a SOC analyst AI. Choose exactly one action command from the allowed list. "
        "Allowed: analyze_log:<log_id>, flag_phishing_email, block_ip:<ip>, "
        "isolate_machine:<id>, stop_process:<pid>, ignore_alert, request_more_info, "
        "explain_reasoning:<text>."
    )
    user_prompt = (
        "Observation:\n"
        f"alerts={observation.get('alerts')}\n"
        f"logs={observation.get('logs')}\n"
        f"system_status={observation.get('system_status')}\n"
        f"threat_level={observation.get('threat_level')}\n"
        f"task={observation.get('task')}\n"
        "Pick the next action."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty_model_response")
        return text.splitlines()[0], None
    except Exception as exc:
        # Fallback action keeps episode moving when model/network fails.
        _debug(f"Model call failed, using fallback action: {exc}")
        return "request_more_info", str(exc)


async def _connect_env(base_url: str):
    if not base_url:
        raise RuntimeError("Missing ENV_BASE_URL or OPENENV_BASE_URL")

    _debug(f"ENV_BASE_URL={base_url}")

    try:
        # Prefer from_url per OpenEnv guidance; fallback keeps compatibility
        # with environments exposing from_base_url.
        if hasattr(CyberlyticsEnv, "from_url"):
            async_client = await CyberlyticsEnv.from_url(base_url)
        else:
            async_client = await CyberlyticsEnv.from_base_url(base_url)
        _debug("Environment connected successfully")
        return async_client.sync()
    except Exception as exc:
        _debug(f"Environment connection failed: {exc}")
        raise


def main() -> int:
    rewards: List[float] = []
    env = None
    _format_start(TASK_NAME)

    if not API_KEY:
        _debug("Missing HF_TOKEN or API_KEY")
        _format_end(False, 0, 0.0, rewards)
        return 1

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    try:
        env = asyncio.run(_connect_env(ENV_BASE_URL))
        result = env.reset(task_name=TASK_NAME)
        done = False

        for step in range(1, MAX_STEPS + 1):
            action_text, model_error = _decide_action(client, result.observation.model_dump())

            try:
                step_result = env.step(CyberlyticsAction(command=action_text))
                reward = float(step_result.reward or 0.0)
                rewards.append(reward)
                done = bool(step_result.done)

                step_error = step_result.observation.last_action_error
                if model_error and not step_error:
                    step_error = f"model_error:{model_error}"

                _format_step(step, action_text, reward, done, step_error)
                result = step_result
                if done:
                    break
            except Exception as exc:
                _debug(f"Step {step} failed: {exc}")
                rewards.append(0.0)
                _format_step(step, action_text, 0.0, False, f"step_error:{exc}")
                break

        score = sum(rewards) / max(len(rewards), 1)
        success = done and score > 0.2
        _format_end(success, len(rewards), _clip_score(score), rewards)
        return 0
    except Exception as exc:
        _debug(f"Fatal inference error: {exc}")
        _format_end(False, len(rewards), 0.0, rewards)
        return 1
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                _debug(f"Environment close failed: {close_exc}")


if __name__ == "__main__":
    raise SystemExit(main())
