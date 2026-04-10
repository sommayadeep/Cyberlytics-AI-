---
---
title: Cyberlytics AI
emoji: 🔥
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
short_description: OpenEnv RL environment for cybersecurity incident response.
app_port: 8000
author: sommayadeep
tags:
  - openenv
  - cybersecurity
---

# Cyberlytics AI: Autonomous Incident Response Training Environment

Cyberlytics AI is an OpenEnv-based reinforcement learning environment designed to simulate real-world cybersecurity incident response workflows. The environment models a Security Operations Center (SOC) where an AI agent interacts with logs, alerts, and system states to detect and mitigate cyber threats. Agents must perform sequential decision-making across tasks such as phishing detection, malware containment, and advanced persistent threat (APT) mitigation.

The environment provides structured observations, realistic action spaces, and a dense reward function that incentivizes accurate, timely, and efficient responses while penalizing incorrect or harmful actions.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the server
uv run --project . server
```

In another terminal, run the baseline inference script:

```bash
API_BASE_URL=http://localhost:8000 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
HF_TOKEN=your_token_here \
python inference.py
```

## Environment Details

### Action Space
Actions are modeled as a single structured command string with optional parameters:

- `analyze_log:<log_id>`
- `flag_phishing_email`
- `block_ip:<ip>`
- `isolate_machine:<id>`
- `stop_process:<pid>`
- `ignore_alert`
- `request_more_info`
- `explain_reasoning:<free_text>`

### Observation Space

```json
{
  "alerts": [{"id": "a1", "severity": "high", "description": "Suspicious login"}],
  "logs": [{"id": "l1", "source": "auth", "message": "Failed SSH login"}],
  "system_status": {"infected_hosts": ["host-1"], "blocked_ips": ["203.0.113.9"]},
  "threat_level": "low|medium|high",
  "task": {"name": "phishing_detection", "goal": "Identify phishing and block sender"}
}
```

### Reward Design

Rewards provide dense shaping across the episode:

- Correctly detect threat: +0.3
- Correct mitigation: +0.4
- Fast response (per-step bonus): +0.2 total
- False positive: -0.2
- Ignore real attack: -0.5
- Loop/useless actions: -0.1
- Time pressure: reward -= 0.01 per step
- Explainability bonus: +0.05 for useful reasoning text

### Tasks

1. **Easy: Phishing Detection**
   - Identify phishing via email/log signals and block sender.
2. **Medium: Malware Infection**
   - Detect malicious process, isolate machine, stop process.
3. **Hard: Multi-step APT**
   - Correlate login anomaly, lateral movement, and data exfiltration, then mitigate.

Each task has a deterministic grader with a score in [0, 1].

## Project Structure

```
cyberlytics_env/
├── __init__.py
├── client.py
├── models.py
├── tasks.py
└── server/
    ├── app.py
    └── cyberlytics_environment.py
```

## Notes

- This environment is designed to be deterministic for reproducibility.
- The baseline inference script emits structured stdout logs required by validators.
- Ensure environment variables `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are set before running `inference.py`.
