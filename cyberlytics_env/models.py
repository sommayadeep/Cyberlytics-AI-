"""Data models for the Cyberlytics AI environment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class CyberlyticsAction(Action):
    """Action payload for the Cyberlytics AI environment."""

    command: str = Field(..., description="Action string like 'block_ip:203.0.113.9'")


class CyberlyticsObservation(Observation):
    """Observation for the Cyberlytics AI environment."""

    alerts: List[Dict[str, Any]] = Field(default_factory=list)
    logs: List[Dict[str, Any]] = Field(default_factory=list)
    system_status: Dict[str, Any] = Field(default_factory=dict)
    threat_level: str = Field(default="low")
    task: Dict[str, Any] = Field(default_factory=dict)
    last_action_error: Optional[str] = Field(default=None)


class CyberlyticsState(State):
    """State metadata for the Cyberlytics AI environment."""

    task_name: str = Field(default="")
    step_limit: int = Field(default=12)
