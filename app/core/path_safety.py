from pathlib import Path

from app.core.config import get_settings


def _path_access_fix_suggestion(path: Path, allowed_roots: list[Path]) -> str:
    roots = ", ".join(str(root) for root in allowed_roots) if allowed_roots else "(none)"
    parent_hint = str(path.parent if path.parent != path else path)
    return (
        f"{path} is outside allowed data roots (allowed: {roots}). "
        "How to fix: "
        f"1) move or bind-mount the data under one of the allowed roots such as {roots}; "
        f"2) or add {parent_hint} to SELF_API_FILE_ACCESS_ROOTS in .env and restart self_api."
    )


def get_allowed_roots() -> list[Path]:
    return get_settings().resolved_file_access_roots


def is_path_allowed(path: Path) -> bool:
    allowed_roots = get_allowed_roots()
    if not allowed_roots:
        return True

    resolved = path.resolve(strict=False)
    return any(resolved == root or root in resolved.parents for root in allowed_roots)


def resolve_safe_path(
    value: str | Path,
    *,
    field_name: str,
    must_exist: bool = False,
    expect_directory: bool | None = None,
    expect_file: bool | None = None,
) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    allowed_roots = get_allowed_roots()
    if not is_path_allowed(path):
        raise ValueError(f"{field_name} " + _path_access_fix_suggestion(path, allowed_roots))

    if must_exist and not path.exists():
        raise ValueError(f"{field_name} does not exist: {path}")
    if expect_directory is True and path.exists() and not path.is_dir():
        raise ValueError(f"{field_name} is not a directory: {path}")
    if expect_file is True and path.exists() and not path.is_file():
        raise ValueError(f"{field_name} is not a file: {path}")
    return path
