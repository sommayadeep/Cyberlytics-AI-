"""Phase 2 inference script for Cyberlytics AI.

Emits strict START/STEP/END logs and handles validator/runtime failures
gracefully without unhandled crashes.
"""

import asyncio
import os
import sys
from typing import Any, List, Optional, Tuple

import requests
from openai import OpenAI

from cyberlytics_env.tasks import TASKS

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = os.getenv("CYBERLYTICS_BENCHMARK", "cyberlytics_env")
MAX_STEPS = int(os.getenv("CYBERLYTICS_MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("CYBERLYTICS_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("CYBERLYTICS_MAX_TOKENS", "150"))
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("OPENENV_BASE_URL")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
TASK_ORDER = ["phishing_detection", "malware_infection", "apt_mitre"]
SPACE_URL = os.getenv("SPACE_URL") or os.getenv("HF_SPACE_URL")


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
    # Keep score strictly between 0 and 1 for validator/display constraints.
    return max(0.01, min(value, 0.99))


def _decide_action(client: Optional[OpenAI], observation: dict[str, Any]) -> Tuple[str, Optional[str]]:
    if client is None:
        return "request_more_info", "missing_api_key"

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
        return "request_more_info", f"model_error:{exc}"


async def _connect_env() -> str:
    url_candidates = [
        ENV_BASE_URL,
        SPACE_URL,
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]

    for candidate in url_candidates:
        if not candidate:
            continue
        _debug(f"Trying env URL: {candidate}")
        try:
            response = requests.get(f"{candidate.rstrip('/')}/health", timeout=10)
            if response.status_code == 200:
                _debug("Environment reachable via URL")
                return candidate
        except Exception as exc:
            _debug(f"URL connection failed: {exc}")

    if LOCAL_IMAGE_NAME:
        _debug(
            "LOCAL_IMAGE_NAME is set but URL is required for Phase 2. "
            "Provide ENV_BASE_URL or OPENENV_BASE_URL."
        )

    raise RuntimeError(
        "Could not connect to environment via URL. Missing ENV_BASE_URL/OPENENV_BASE_URL"
    )


def _http_reset(base_url: str, task_name: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/reset",
        json={"task_name": task_name},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _http_step(base_url: str, action: str) -> dict[str, Any]:
    response = requests.post(
        f"{base_url.rstrip('/')}/step",
        json={"action": {"command": action}},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _build_client() -> Optional[OpenAI]:
    if not API_KEY:
        _debug("Missing HF_TOKEN or API_KEY; using fallback actions")
        return None
    try:
        return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as exc:
        _debug(f"OpenAI client init failed: {exc}")
        return None


def _run_task(client: Optional[OpenAI], task_name: str) -> int:
    rewards: List[float] = []
    env_base_url: Optional[str] = None
    done = False
    _format_start(task_name)

    try:
        env_base_url = asyncio.run(_connect_env())
        result = _http_reset(env_base_url, task_name)

        for step in range(1, MAX_STEPS + 1):
            observation = result.get("observation", {})
            action_text, model_error = _decide_action(client, observation)
            try:
                step_result = _http_step(env_base_url, action_text)
                reward = float(step_result.get("reward") or 0.0)
                rewards.append(reward)
                done = bool(step_result.get("done"))

                step_error = (step_result.get("observation") or {}).get("last_action_error")
                if model_error and not step_error:
                    step_error = model_error

                _format_step(step, action_text, reward, done, step_error)
                result = step_result
                if done:
                    break
            except Exception as exc:
                _debug(f"Step {step} failed for task={task_name}: {exc}")
                rewards.append(0.0)
                _format_step(step, action_text, 0.0, False, f"step_error:{exc}")
                break

        raw_score = sum(rewards) / max(len(rewards), 1)
        score = _clip_score(raw_score)
        success = done and score > 0.2
        _format_end(success, len(rewards), score, rewards)
        return 0
    except Exception as exc:
        _debug(f"Fatal task error for task={task_name}: {exc}")
        _format_end(False, len(rewards), _clip_score(0.0), rewards)
        return 1


def main() -> int:
    client = _build_client()
    exit_code = 0

    # Ensure all required tasks are evaluated in sequence.
    for task_name in TASK_ORDER:
        if task_name not in TASKS:
            _debug(f"Task not found in TASKS: {task_name}")
            _format_start(task_name)
            _format_step(1, "request_more_info", 0.0, True, "task_not_found")
            _format_end(False, 1, _clip_score(0.0), [0.0])
            exit_code = 1
            continue

        task_code = _run_task(client, task_name)
        if task_code != 0:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
