"""Baseline inference script for Cyberlytics AI."""

import asyncio
import os
import sys
from typing import List

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
IMAGE_NAME = os.getenv("IMAGE_NAME", "cyberlytics-env:latest")


def _format_start(task_name: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}")


def _format_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_text = error if error else "null"
    done_text = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_text} error={error_text}"
    )


def _format_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_text = ",".join(f"{value:.2f}" for value in rewards)
    success_text = "true" if success else "false"
    print(
        f"[END] success={success_text} steps={steps} score={score:.2f} rewards={rewards_text}"
    )


def _decide_action(client: OpenAI, observation: dict) -> str:
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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = response.choices[0].message.content.strip()
    return text.splitlines()[0]


def main() -> int:
    if not API_KEY:
        print("Missing HF_TOKEN or API_KEY", file=sys.stderr)
        return 1

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    async_client = asyncio.run(CyberlyticsEnv.from_docker_image(IMAGE_NAME))
    env = async_client.sync()
    rewards: List[float] = []

    try:
        _format_start(TASK_NAME)
        result = env.reset(task_name=TASK_NAME)
        done = False

        for step in range(1, MAX_STEPS + 1):
            action_text = _decide_action(client, result.observation.model_dump())
            step_result = env.step(CyberlyticsAction(command=action_text))
            reward = float(step_result.reward or 0.0)
            rewards.append(reward)
            done = bool(step_result.done)
            _format_step(step, action_text, reward, done, step_result.observation.last_action_error)
            result = step_result
            if done:
                break

        score = sum(rewards) / max(len(rewards), 1)
        success = done and score > 0.2
        _format_end(success, len(rewards), max(0.0, min(score, 1.0)), rewards)
    except Exception as exc:
        _format_end(False, len(rewards), 0.0, rewards)
        raise exc
    finally:
        env.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
