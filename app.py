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

with gr.Blocks() as demo:
    gr.Markdown("# Cyberlytics AI OpenEnv Demo\nInteract with your deployed RL environment.")
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
    action_in = gr.Textbox(label="Action (string)")
    step_btn = gr.Button("Step")
    step_out = gr.JSON()
    step_btn.click(fn=step_env, inputs=[env_state, action_in], outputs=[env_state, step_out])

demo.launch()
