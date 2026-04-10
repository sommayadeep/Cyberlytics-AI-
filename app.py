import gradio as gr

from cyberlytics_env.models import CyberlyticsAction
from cyberlytics_env.tasks import TASKS
from server.cyberlytics_environment import CyberlyticsEnvironment


def check_status():
    return {"status": "Cyberlytics AI OpenEnv is running"}


def init_env():
    return CyberlyticsEnvironment()


def reset_env(env, task_name):
    if env is None:
        env = init_env()
    obs = env.reset(task_name=task_name)
    return env, obs.model_dump()


def get_state(env):
    if env is None:
        env = init_env()
    return env.state.model_dump()


def step_env(env, action_text):
    if env is None:
        env = init_env()
    action = CyberlyticsAction(command=action_text or "")
    obs = env.step(action)
    return env, obs.model_dump()

EXAMPLE_ACTIONS = [
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
        "Interact with your deployed RL environment. Use Reset, then Step with an action string."
    )
    status_btn = gr.Button("Check Server Status")
    status_out = gr.JSON()
    status_btn.click(fn=check_status, outputs=status_out)

    gr.Markdown("## Environment API Calls")
    env_state = gr.State(init_env())
    task_choices = sorted(TASKS.keys())
    task_picker = gr.Dropdown(choices=task_choices, value=task_choices[0], label="Task")
    with gr.Row():
        reset_btn = gr.Button("Reset Environment")
        reset_out = gr.JSON()
        reset_btn.click(fn=reset_env, inputs=[env_state, task_picker], outputs=[env_state, reset_out])

        state_btn = gr.Button("Get State")
        state_out = gr.JSON()
        state_btn.click(fn=get_state, inputs=[env_state], outputs=state_out)

    gr.Markdown("## Step Environment")
    gr.Markdown(
        "Action format examples: `block_ip:203.0.113.9`, `isolate_machine:host-5`, "
        "`explain_reasoning:<text>`"
    )
    action_pick = gr.Dropdown(choices=EXAMPLE_ACTIONS, label="Example Actions")
    action_in = gr.Textbox(label="Action (string)")
    step_btn = gr.Button("Step")
    step_out = gr.JSON()
    action_pick.change(lambda value: value, inputs=action_pick, outputs=action_in)
    step_btn.click(fn=step_env, inputs=[env_state, action_in], outputs=[env_state, step_out])

demo.launch()
