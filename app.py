import gradio as gr
import requests
import os

API_URL = os.getenv("API_URL") or "https://sommayadeep-cyberlytics-ai.hf.space"

# Helper to call the root endpoint
def check_status():
    try:
        r = requests.get(f"{API_URL}/")
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"Status code: {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Helper to call /reset, /step, /state endpoints
def call_env_api(endpoint, payload=None):
    try:
        url = f"{API_URL}/{endpoint}".rstrip("/")
        if payload:
            r = requests.post(url, json=payload)
        else:
            r = requests.post(url)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks() as demo:
    gr.Markdown("# Cyberlytics AI OpenEnv Demo\nInteract with your deployed RL environment.")
    status_btn = gr.Button("Check Server Status")
    status_out = gr.JSON()
    status_btn.click(fn=check_status, outputs=status_out)

    gr.Markdown("## Environment API Calls")
    with gr.Row():
        reset_btn = gr.Button("Reset Environment")
        reset_out = gr.JSON()
        reset_btn.click(fn=lambda: call_env_api("reset"), outputs=reset_out)

        state_btn = gr.Button("Get State")
        state_out = gr.JSON()
        state_btn.click(fn=lambda: call_env_api("state"), outputs=state_out)

    gr.Markdown("## Step Environment")
    action_in = gr.Textbox(label="Action (as JSON or string)")
    step_btn = gr.Button("Step")
    step_out = gr.JSON()
    def step_env(action):
        try:
            import json
            try:
                action_obj = json.loads(action)
            except Exception:
                action_obj = {"action": action}
            return call_env_api("step", action_obj)
        except Exception as e:
            return {"error": str(e)}
    step_btn.click(fn=step_env, inputs=action_in, outputs=step_out)

demo.launch()
