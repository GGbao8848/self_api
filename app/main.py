from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()
setup_logging(settings.log_level)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Image preprocessing API with async tasks, auth, artifact uploads, and "
        "system probes"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origin_list or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "env": settings.app_env,
        "docs": "/docs",
    }


app.include_router(api_router, prefix=settings.api_v1_prefix)

app.mount(
    "/static",
    StaticFiles(directory=str(settings.project_root / "app" / "static")),
    name="static",
)


@app.get("/train-ui")
def train_ui() -> FileResponse:
    return FileResponse(settings.project_root / "app" / "static" / "train-ui" / "index.html")


@app.get("/pipeline-ui")
def pipeline_ui() -> FileResponse:
    return FileResponse(
        settings.project_root / "app" / "static" / "pipeline-ui" / "index.html",
    )
