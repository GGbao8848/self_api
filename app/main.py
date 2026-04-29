from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

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
        "agent_ui": "/agent-ui",
    }


app.include_router(api_router, prefix=settings.api_v1_prefix)

_agent_ui_dist = settings.project_root / "ai-agent-chat" / "dist"


@app.get("/agent-ui")
def agent_ui_index() -> FileResponse:
    index_path = _agent_ui_dist / "index.html"
    if not index_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="agent UI is not built yet; expected dist/index.html under ai-agent-chat",
        )
    return FileResponse(index_path)


@app.get("/agent-ui/{asset_path:path}")
def agent_ui_assets(asset_path: str) -> FileResponse:
    requested_path = (_agent_ui_dist / asset_path).resolve()
    dist_root = _agent_ui_dist.resolve()
    if not str(requested_path).startswith(str(dist_root)):
        raise HTTPException(status_code=404, detail="invalid agent UI asset path")

    if requested_path.is_file():
        return FileResponse(requested_path)

    index_path = _agent_ui_dist / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    raise HTTPException(
        status_code=404,
        detail="agent UI is not built yet; expected dist/index.html under ai-agent-chat",
    )
