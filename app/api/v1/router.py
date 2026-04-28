from fastapi import APIRouter

from app.api.v1.endpoints import agent, artifacts, auth, files, preprocess, system, tasks

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(auth.router)
api_router.include_router(agent.router)
api_router.include_router(files.router)
api_router.include_router(artifacts.router)
api_router.include_router(tasks.router)
api_router.include_router(preprocess.router)
