from urllib.parse import urljoin

from fastapi import Request

from app.core.config import get_settings


def build_route_url(request: Request, route_name: str, **path_params: str) -> str:
    settings = get_settings()
    public_base_url = settings.normalized_public_base_url
    if public_base_url:
        route_path = str(request.app.url_path_for(route_name, **path_params))
        return urljoin(f"{public_base_url}/", route_path.lstrip("/"))
    return str(request.url_for(route_name, **path_params))
