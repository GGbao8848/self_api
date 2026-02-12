from fastapi import APIRouter

from app.api.v1.endpoints import preprocess, system

api_router = APIRouter()
api_router.include_router(system.router)
api_router.include_router(preprocess.router)
