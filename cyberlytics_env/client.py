"""Client for the Cyberlytics AI environment."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CyberlyticsAction, CyberlyticsObservation, CyberlyticsState


class CyberlyticsEnv(
    EnvClient[CyberlyticsAction, CyberlyticsObservation, CyberlyticsState]
):
    """HTTP client for Cyberlytics AI."""

    def _step_payload(self, action: CyberlyticsAction) -> Dict[str, Any]:
        return {"command": action.command}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CyberlyticsObservation]:
        obs_data = payload.get("observation", {})
        observation = CyberlyticsObservation(
            alerts=obs_data.get("alerts", []),
            logs=obs_data.get("logs", []),
            system_status=obs_data.get("system_status", {}),
            threat_level=obs_data.get("threat_level", "low"),
            task=obs_data.get("task", {}),
            last_action_error=obs_data.get("last_action_error"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done"))

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return CyberlyticsState(**payload)
