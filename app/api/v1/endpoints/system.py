from fastapi import APIRouter

router = APIRouter(tags=["system"])


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
