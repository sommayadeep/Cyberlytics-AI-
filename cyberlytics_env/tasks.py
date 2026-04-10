"""Task definitions and graders for Cyberlytics AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TaskSpec:
    name: str
    goal: str
    alerts: List[Dict[str, Any]]
    logs: List[Dict[str, Any]]
    threat_level: str
    system_status: Dict[str, Any]
    required_actions: List[str]
    forbidden_actions: List[str] = field(default_factory=list)


PHISHING_TASK = TaskSpec(
    name="phishing_detection",
    goal="Identify phishing email and block sender",
    alerts=[{"id": "a1", "severity": "medium", "description": "User reported suspicious email"}],
    logs=[
        {"id": "l1", "source": "email", "message": "From: payroll@company-secure.com"},
        {"id": "l2", "source": "email", "message": "Link: http://company-payroll-reset.com"},
    ],
    threat_level="medium",
    system_status={"infected_hosts": [], "blocked_ips": [], "blocked_senders": []},
    required_actions=["flag_phishing_email", "block_ip:203.0.113.9"],
    forbidden_actions=["ignore_alert"],
)

MALWARE_TASK = TaskSpec(
    name="malware_infection",
    goal="Detect malware, isolate machine, stop process",
    alerts=[{"id": "a1", "severity": "high", "description": "Suspicious process detected"}],
    logs=[
        {"id": "l1", "source": "edr", "message": "Process: svchost32.exe PID 4432"},
        {"id": "l2", "source": "edr", "message": "Outbound beacon to 198.51.100.21"},
    ],
    threat_level="high",
    system_status={"infected_hosts": ["host-12"], "blocked_ips": []},
    required_actions=["isolate_machine:host-12", "stop_process:4432"],
    forbidden_actions=["ignore_alert"],
)

APT_TASK = TaskSpec(
    name="apt_mitre",
    goal="Correlate signals and mitigate APT intrusion",
    alerts=[
        {"id": "a1", "severity": "high", "description": "Impossible travel login"},
        {"id": "a2", "severity": "high", "description": "Data exfiltration spike"},
    ],
    logs=[
        {"id": "l1", "source": "auth", "message": "Login from 203.0.113.7 (new geo)"},
        {"id": "l2", "source": "netflow", "message": "Large outbound transfer to 198.51.100.99"},
        {"id": "l3", "source": "edr", "message": "Lateral movement: host-5 -> host-7"},
    ],
    threat_level="high",
    system_status={"infected_hosts": ["host-5", "host-7"], "blocked_ips": []},
    required_actions=[
        "analyze_log:l3",
        "isolate_machine:host-7",
        "block_ip:198.51.100.99",
    ],
    forbidden_actions=["ignore_alert"],
)

TASKS = {
    PHISHING_TASK.name: PHISHING_TASK,
    MALWARE_TASK.name: MALWARE_TASK,
    APT_TASK.name: APT_TASK,
}
