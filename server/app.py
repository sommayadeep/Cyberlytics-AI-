"""FastAPI application for the Cyberlytics AI environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install dependencies with 'uv sync'."
    ) from exc

try:
    from cyberlytics_env.models import CyberlyticsAction, CyberlyticsObservation
    from cyberlytics_env.tasks import TASKS
    from server.cyberlytics_environment import CyberlyticsEnvironment
except ModuleNotFoundError:
    from models import CyberlyticsAction, CyberlyticsObservation
    from tasks import TASKS
    from server.cyberlytics_environment import CyberlyticsEnvironment



app = create_app(
    CyberlyticsEnvironment,
    CyberlyticsAction,
    CyberlyticsObservation,
    env_name="cyberlytics_env",
    max_concurrent_envs=1,
)

import gradio as gr


def _init_env():
    return CyberlyticsEnvironment()


def _reset_env(env, task_name):
    if env is None:
        env = _init_env()
    obs = env.reset(task_name=task_name)
    return env, obs.model_dump()


def _get_state(env):
    if env is None:
        env = _init_env()
    return env.state.model_dump()


def _step_env(env, action_text):
    if env is None:
        env = _init_env()
    action = CyberlyticsAction(command=action_text or "")
    obs = env.step(action)
    return env, obs.model_dump()


def _build_ui():
    task_choices = sorted(TASKS.keys())
    example_actions = [
        "analyze_log:l1",
        "flag_phishing_email",
        "block_ip:203.0.113.9",
        "isolate_machine:host-5",
        "stop_process:1337",
        "request_more_info",
        "ignore_alert",
        "explain_reasoning:Investigated logs and correlated anomalies",
    ]

    with gr.Blocks() as demo:
        gr.Markdown(
            "# Cyberlytics AI OpenEnv Demo\n"
            "Use the OpenEnv API at /reset, /step, /state. UI is available at /ui."
        )
        env_state = gr.State(_init_env())
        task_picker = gr.Dropdown(choices=task_choices, value=task_choices[0], label="Task")

        with gr.Row():
            reset_btn = gr.Button("Reset Environment")
            reset_out = gr.JSON()
            reset_btn.click(fn=_reset_env, inputs=[env_state, task_picker], outputs=[env_state, reset_out])

            state_btn = gr.Button("Get State")
            state_out = gr.JSON()
            state_btn.click(fn=_get_state, inputs=[env_state], outputs=state_out)

        gr.Markdown("## Step Environment")
        action_pick = gr.Dropdown(choices=example_actions, label="Example Actions")
        action_in = gr.Textbox(label="Action (string)")
        step_btn = gr.Button("Step")
        step_out = gr.JSON()
        action_pick.change(lambda value: value, inputs=action_pick, outputs=action_in)
        step_btn.click(fn=_step_env, inputs=[env_state, action_in], outputs=[env_state, step_out])

    return demo


app = gr.mount_gradio_app(app, _build_ui(), path="/ui")


# Add root endpoint for Hugging Face health check
from fastapi import Request
@app.get("/")
async def root(request: Request):
    return {"status": "Cyberlytics AI OpenEnv is running"}


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
