"""跨机器远程解压服务：通过 SSH 在远端执行 unzip。"""

import re
import shlex
from pathlib import Path
from urllib.parse import urlparse

import paramiko

from app.schemas.preprocess import RemoteUnzipRequest, RemoteUnzipResponse


def _parse_remote_path(target: str) -> tuple[str, int, str]:
    target = target.strip()
    if not target:
        raise ValueError("path is empty")

    if target.startswith("sftp://") or target.startswith("ssh://"):
        parsed = urlparse(target)
        if not parsed.hostname:
            raise ValueError(f"invalid path: missing host in {target}")
        host = parsed.hostname
        port = parsed.port or 22
        path = parsed.path or "/"
        path = path if path.startswith("/") else f"/{path}"
        return host, port, path

    scp_match = re.match(r"^([^@]+)@([^:]+):(.+)$", target)
    if scp_match:
        return scp_match.group(2), 22, scp_match.group(3)

    raise ValueError(
        f"invalid remote path format: {target}. expected sftp://host/path or user@host:path"
    )


def _extract_username_from_target(target: str) -> str | None:
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


def run_remote_unzip(request: RemoteUnzipRequest) -> RemoteUnzipResponse:
    archive_host, archive_port, archive_remote_path = _parse_remote_path(request.archive_path)

    if request.output_dir:
        output_host, output_port, output_remote_path = _parse_remote_path(request.output_dir)
        if output_host != archive_host or output_port != archive_port:
            raise ValueError("archive_path and output_dir must point to the same remote host/port")
    else:
        output_host, output_port = archive_host, archive_port
        archive_path_obj = Path(archive_remote_path)
        output_remote_path = str(archive_path_obj.parent / archive_path_obj.stem).replace("\\", "/")

    username = request.username or _extract_username_from_target(request.archive_path)
    if not username:
        raise ValueError("username is required: set username in request or use user@host:path")

    if request.password and request.private_key_path:
        raise ValueError("use either password or private_key_path, not both")
    if not request.password and not request.private_key_path:
        raise ValueError("either password or private_key_path is required")

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

    port = request.port if request.port != 22 else archive_port

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=archive_host,
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

    overwrite_flag = "-o" if request.overwrite else "-n"
    command = (
        f"mkdir -p {shlex.quote(output_remote_path)} && "
        f"unzip {overwrite_flag} {shlex.quote(archive_remote_path)} -d {shlex.quote(output_remote_path)}"
    )
    try:
        _, stdout, stderr = client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        err_text = stderr.read().decode("utf-8", errors="replace").strip()
        if exit_code != 0:
            raise ValueError(
                f"remote unzip failed (exit={exit_code}) on {archive_host}:{port}: {err_text or 'unknown error'}"
            )
        return RemoteUnzipResponse(
            archive_path=archive_remote_path,
            output_dir=output_remote_path,
            target_host=output_host,
            target_port=output_port,
            command=command,
        )
    finally:
        client.close()
