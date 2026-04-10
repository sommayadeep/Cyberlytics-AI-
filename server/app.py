"""FastAPI application for the Cyberlytics AI environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required. Install dependencies with 'uv sync'."
    ) from exc

try:
    from cyberlytics_env.models import CyberlyticsAction, CyberlyticsObservation
    from server.cyberlytics_environment import CyberlyticsEnvironment
except ModuleNotFoundError:
    from models import CyberlyticsAction, CyberlyticsObservation
    from server.cyberlytics_environment import CyberlyticsEnvironment



app = create_app(
    CyberlyticsEnvironment,
    CyberlyticsAction,
    CyberlyticsObservation,
    env_name="cyberlytics_env",
    max_concurrent_envs=1,
)

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
