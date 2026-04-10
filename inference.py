"""
Phase 2 inference script for Cyberlytics AI (FIXED VERSION)

- Robust env connection (HF Space compatible)
- Supports /health OR root /
- Never crashes silently
- Strict START / STEP / END logging
"""

import asyncio
import os
import sys
from typing import Any, List, Optional, Tuple

import requests
from openai import OpenAI

from cyberlytics_env.tasks import TASKS

# ================= ENV CONFIG =================
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = os.getenv("CYBERLYTICS_BENCHMARK", "cyberlytics_env")
MAX_STEPS = int(os.getenv("CYBERLYTICS_MAX_STEPS", "8"))
TEMPERATURE = float(os.getenv("CYBERLYTICS_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("CYBERLYTICS_MAX_TOKENS", "150"))

ENV_BASE_URL = os.getenv("ENV_BASE_URL") or os.getenv("OPENENV_BASE_URL")

TASK_ORDER = ["phishing_detection", "malware_infection", "apt_mitre"]


# ================= LOGGING =================
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
    return max(0.01, min(value, 0.99))


# ================= MODEL =================
def _build_client() -> Optional[OpenAI]:
    if not API_KEY:
        _debug("No API key → using fallback actions")
        return None
    try:
        return OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as exc:
        _debug(f"Client init failed: {exc}")
        return None


def _decide_action(client: Optional[OpenAI], observation: dict[str, Any]) -> Tuple[str, Optional[str]]:
    if client is None:
        return "request_more_info", "no_model"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": "You are a SOC analyst. Return ONE action."},
                {"role": "user", "content": str(observation)},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        return text.splitlines()[0] if text else "request_more_info", None
    except Exception as exc:
        _debug(f"Model error: {exc}")
        return "request_more_info", f"model_error:{exc}"


# ================= ENV CONNECTION =================
async def _connect_env() -> str:
    import os
    import requests

    base = os.getenv("ENV_BASE_URL") or os.getenv("OPENENV_BASE_URL")

    if not base:
        raise RuntimeError("ENV_BASE_URL is required for Phase 2")

    base = base.rstrip("/")
    print(f"[DEBUG] Using ENV_BASE_URL: {base}", file=sys.stderr, flush=True)

    try:
        r = requests.get(base, timeout=10)
        if r.status_code == 200:
            print("[DEBUG] Environment reachable", file=sys.stderr, flush=True)
            return base
    except Exception as e:
        raise RuntimeError(f"Cannot connect to ENV_BASE_URL: {e}")

    raise RuntimeError("Environment not reachable")


# ================= HTTP CALLS =================
def _http_reset(base_url: str, task_name: str) -> dict[str, Any]:
    r = requests.post(
        f"{base_url}/reset",
        json={"task_name": task_name},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def _http_step(base_url: str, action: str) -> dict[str, Any]:
    r = requests.post(
        f"{base_url}/step",
        json={"action": {"command": action}},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


# ================= TASK RUN =================
def _run_task(client: Optional[OpenAI], task_name: str) -> int:
    rewards: List[float] = []
    _format_start(task_name)

    try:
        base_url = asyncio.run(_connect_env())
        result = _http_reset(base_url, task_name)

        for step in range(1, MAX_STEPS + 1):
            obs = result.get("observation", {})
            action, err = _decide_action(client, obs)

            try:
                res = _http_step(base_url, action)
                reward = float(res.get("reward", 0.0))
                done = bool(res.get("done"))

                rewards.append(reward)
                _format_step(step, action, reward, done, err)

                result = res
                if done:
                    break

            except Exception as exc:
                _debug(f"Step error: {exc}")
                rewards.append(0.0)
                _format_step(step, action, 0.0, False, f"step_error:{exc}")
                break

        score = _clip_score(sum(rewards) / max(len(rewards), 1))
        success = score > 0.2
        _format_end(success, len(rewards), score, rewards)
        return 0

    except Exception as exc:
        _debug(f"Fatal error: {exc}")
        _format_end(False, 0, 0.01, [])
        return 1


# ================= MAIN =================
def main() -> int:
    client = _build_client()
    exit_code = 0

    for task in TASK_ORDER:
        if task not in TASKS:
            _format_start(task)
            _format_step(1, "request_more_info", 0.0, True, "task_missing")
            _format_end(False, 1, 0.01, [0.0])
            exit_code = 1
            continue

        if _run_task(client, task) != 0:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())