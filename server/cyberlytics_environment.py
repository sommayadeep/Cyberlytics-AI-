"""Cyberlytics AI environment implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from cyberlytics_env.models import CyberlyticsAction, CyberlyticsObservation, CyberlyticsState
from cyberlytics_env.tasks import TASKS, TaskSpec


@dataclass
class EpisodeTracker:
    required_actions: List[str] = field(default_factory=list)
    taken_actions: List[str] = field(default_factory=list)
    penalty_actions: List[str] = field(default_factory=list)
    step_count: int = 0
    cumulative_reward: float = 0.0


class CyberlyticsEnvironment(Environment[CyberlyticsAction, CyberlyticsObservation, CyberlyticsState]):
    """OpenEnv-compatible environment for SOC incident response."""

    def __init__(self):
        self._state = CyberlyticsState(episode_id=str(uuid4()), step_count=0)
        self._task: Optional[TaskSpec] = None
        self._tracker = EpisodeTracker()
        self._last_action_error: Optional[str] = None

    def _select_task(self, task_name: Optional[str]) -> TaskSpec:
        if task_name and task_name in TASKS:
            return TASKS[task_name]
        return TASKS["phishing_detection"]

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> CyberlyticsObservation:
        task_name = kwargs.get("task_name") or kwargs.get("task") or "phishing_detection"
        self._task = self._select_task(task_name)
        self._tracker = EpisodeTracker(required_actions=list(self._task.required_actions))
        self._last_action_error = None
        self._state = CyberlyticsState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=self._task.name,
            step_limit=12,
        )
        return self._make_observation(done=False, reward=0.0)

    def step(self, action: CyberlyticsAction) -> CyberlyticsObservation:
        if self._task is None:
            self._task = TASKS["phishing_detection"]
        self._state.step_count += 1
        self._tracker.step_count += 1

        reward = 0.0
        done = False
        command = action.command.strip()

        if not command:
            self._last_action_error = "empty_action"
            reward -= 0.1
        else:
            reward += self._apply_action(command)

        reward -= 0.01 * self._tracker.step_count

        if self._is_success():
            reward += 0.4
            done = True
        elif self._tracker.step_count >= self._state.step_limit:
            reward -= 0.2
            done = True

        self._tracker.cumulative_reward += reward
        return self._make_observation(done=done, reward=reward)

    def _apply_action(self, command: str) -> float:
        reward = 0.0
        self._last_action_error = None

        if command in self._tracker.taken_actions:
            reward -= 0.1
        self._tracker.taken_actions.append(command)

        if command.startswith("explain_reasoning:"):
            reasoning = command.split(":", 1)[1].strip()
            if len(reasoning.split()) >= 6:
                reward += 0.05
            return reward

        if command in self._task.forbidden_actions:
            reward -= 0.5
            self._tracker.penalty_actions.append(command)
            return reward

        if command in self._tracker.required_actions:
            reward += 0.3
        elif command.startswith("analyze_log"):
            reward += 0.1
        elif command.startswith("request_more_info"):
            reward += 0.05
        elif command.startswith("ignore_alert"):
            reward -= 0.2
        else:
            reward -= 0.1

        return reward

    def _is_success(self) -> bool:
        return all(req in self._tracker.taken_actions for req in self._tracker.required_actions)

    def _make_observation(self, done: bool, reward: float) -> CyberlyticsObservation:
        assert self._task is not None
        return CyberlyticsObservation(
            alerts=list(self._task.alerts),
            logs=list(self._task.logs),
            system_status=dict(self._task.system_status),
            threat_level=self._task.threat_level,
            task={"name": self._task.name, "goal": self._task.goal},
            last_action_error=self._last_action_error,
            done=done,
            reward=reward,
            metadata={"cumulative_reward": self._tracker.cumulative_reward},
        )

    @property
    def state(self) -> CyberlyticsState:
        return self._state
