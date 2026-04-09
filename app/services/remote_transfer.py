"""跨机器 SFTP 传输服务：基于 paramiko 将本地文件/目录上传到远程 SFTP 服务器。"""

import re
from pathlib import Path
from urllib.parse import urlparse

import paramiko

from app.core.path_safety import resolve_safe_path
from app.schemas.preprocess import (
    RemoteTransferRequest,
    RemoteTransferResponse,
)
from app.services.task_manager import ensure_current_task_active


def _parse_target(target: str) -> tuple[str, int, str]:
    """解析 target 字符串，返回 (host, port, remote_path)。

    支持格式：
    - sftp://host/path
    - sftp://user@host/path
    - sftp://host:port/path
    - user@host:path (scp 风格)
    """
    target = target.strip()
    if not target:
        raise ValueError("target is empty")

    # sftp:// 或 ssh://
    if target.startswith("sftp://") or target.startswith("ssh://"):
        parsed = urlparse(target)
        if not parsed.hostname:
            raise ValueError(f"invalid target: missing host in {target}")
        host = parsed.hostname
        port = parsed.port or 22
        path = parsed.path or "/"
        path = path if path.startswith("/") else f"/{path}"
        return host, port, path

    # user@host:path (scp 风格)
    scp_match = re.match(r"^([^@]+)@([^:]+):(.+)$", target)
    if scp_match:
        return scp_match.group(2), 22, scp_match.group(3)

    # host:path (无 user)
    colon_match = re.match(r"^([^@:]+):(.+)$", target)
    if colon_match:
        return colon_match.group(1), 22, colon_match.group(2)

    raise ValueError(f"invalid target format: {target}")


def _extract_username_from_target(target: str) -> str | None:
    """从 target 中提取用户名（若有）。"""
    target = target.strip()
    if target.startswith("sftp://") or target.startswith("ssh://"):
        parsed = urlparse(target)
        if parsed.username:
            return parsed.username
    if "@" in target and ":" in target:
        m = re.match(r"^([^@]+)@[^:]+:", target)
        if m:
            return m.group(1)
    return None


def run_remote_transfer(request: RemoteTransferRequest) -> RemoteTransferResponse:
    """将本地 source_path 上传到远程 SFTP target。"""
    source_path = resolve_safe_path(
        request.source_path,
        field_name="source_path",
        must_exist=True,
    )

    host, port, remote_path = _parse_target(request.target)
    username = request.username or _extract_username_from_target(request.target)
    if not username:
        raise ValueError("username is required: set username in request or use user@host:path in target")

    # 认证：password 或 private_key
    if request.password and request.private_key_path:
        raise ValueError("use either password or private_key_path, not both")
    if not request.password and not request.private_key_path:
        raise ValueError("either password or private_key_path is required")

    # 兼容 RSA 和 Ed25519
    pkey = None
    if request.private_key_path:
        key_path = Path(request.private_key_path).expanduser().resolve()
        if not key_path.exists():
            raise ValueError(f"private_key_path does not exist: {key_path}")
        try:
            pkey = paramiko.Ed25519Key.from_private_key_file(str(key_path))
        except paramiko.ssh_exception.SSHException:
            try:
                pkey = paramiko.RSAKey.from_private_key_file(str(key_path))
            except paramiko.ssh_exception.SSHException as e:
                raise ValueError(f"failed to load private key: {e}") from e

    port = request.port if request.port != 22 else port

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=host,
            port=port,
            username=username,
            password=request.password,
            pkey=pkey,
            timeout=30,
        )
    except paramiko.AuthenticationException as e:
        raise ValueError(f"SSH authentication failed: {e}") from e
    except paramiko.SSHException as e:
        raise ValueError(f"SSH connection failed: {e}") from e

    transferred_files = 0
    total_bytes = 0
    transferred_type: str = "file"

    try:
        sftp = client.open_sftp()

        def _mkdir_p(sftp_client: paramiko.SFTPClient, path: str) -> None:
            """在远端递归创建目录（类似 mkdir -p）。"""
            parts = path.split("/")
            current = ""
            for p in parts:
                if not p:
                    continue
                current = f"{current}/{p}" if current else f"/{p}"
                try:
                    sftp_client.stat(current)
                except FileNotFoundError:
                    sftp_client.mkdir(current)

        # 统一语义：target 始终视为“目录路径”。
        # source_path 无论是文件还是目录，都复制到 target 下并保留源名。
        if source_path.is_file():
            ensure_current_task_active()
            target_dir = remote_path.rstrip("/")
            if target_dir in ("", "/"):
                remote_file = f"/{source_path.name}"
            else:
                remote_file = f"{target_dir}/{source_path.name}"

            try:
                stat = sftp.stat(remote_file)
                if stat and not request.overwrite:
                    raise ValueError(f"remote file exists and overwrite=false: {remote_file}")
            except FileNotFoundError:
                pass

            # 文件上传前确保 target 目录存在，避免 [Errno 2] No such file。
            remote_parent = str(Path(remote_file).parent).replace("\\", "/")
            if remote_parent and remote_parent not in (".", "/"):
                _mkdir_p(sftp, remote_parent)

            sftp.put(str(source_path), remote_file)
            transferred_files = 1
            total_bytes = source_path.stat().st_size
            transferred_type = "file"
            final_remote = remote_file
        else:
            # 目录：递归上传
            base_name = source_path.name
            remote_dir = remote_path.rstrip("/")
            if remote_dir in ("", "/"):
                remote_dir = f"/{base_name}"
            else:
                remote_dir = f"{remote_dir}/{base_name}"

            _mkdir_p(sftp, remote_dir)

            def _upload_dir(local: Path, remote: str) -> None:
                nonlocal transferred_files, total_bytes
                for item in sorted(local.iterdir()):
                    ensure_current_task_active()
                    remote_item = f"{remote}/{item.name}"
                    if item.is_file():
                        if not request.overwrite:
                            try:
                                sftp.stat(remote_item)
                                continue  # skip existing
                            except FileNotFoundError:
                                pass
                        sftp.put(str(item), remote_item)
                        transferred_files += 1
                        total_bytes += item.stat().st_size
                    else:
                        _mkdir_p(sftp, remote_item)
                        _upload_dir(item, remote_item)

            _upload_dir(source_path, remote_dir)
            transferred_type = "directory"
            final_remote = remote_dir

        return RemoteTransferResponse(
            source_path=str(source_path),
            target=request.target,
            target_host=host,
            target_port=port,
            target_path=final_remote,
            transferred_type=transferred_type,
            transferred_files=transferred_files,
            total_bytes=total_bytes,
        )
    finally:
        client.close()
