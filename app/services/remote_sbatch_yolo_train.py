"""Remote YOLO training via SSH + sbatch."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from urllib.parse import urlparse

import paramiko

from app.schemas.preprocess import (
    RemoteSbatchYoloTrainRequest,
    RemoteSbatchYoloTrainResponse,
)
from app.services.yolo_model_resolver import build_remote_yolo_model_resolver_shell


def _parse_remote_path(target: str) -> tuple[str, int, str]:
    text = target.strip()
    if not text:
        raise ValueError("path is empty")

    if text.startswith("sftp://") or text.startswith("ssh://"):
        parsed = urlparse(text)
        if not parsed.hostname:
            raise ValueError(f"invalid path: missing host in {target}")
        host = parsed.hostname
        port = parsed.port or 22
        path = parsed.path or "/"
        path = path if path.startswith("/") else f"/{path}"
        return host, port, path

    scp_match = re.match(r"^([^@]+)@([^:]+):(.+)$", text)
    if scp_match:
        return scp_match.group(2), 22, scp_match.group(3)

    host_match = re.match(r"^([^@:]+):(.+)$", text)
    if host_match:
        return host_match.group(1), 22, host_match.group(2)

    raise ValueError(
        f"invalid remote path format: {target}. expected absolute path, sftp://host/path, or user@host:path"
    )


def _extract_username(target: str) -> str | None:
    text = target.strip()
    if text.startswith("sftp://") or text.startswith("ssh://"):
        parsed = urlparse(text)
        return parsed.username
    match = re.match(r"^([^@]+)@[^:]+:", text)
    if match:
        return match.group(1)
    return None


def _resolve_remote_location(
    value: str,
    *,
    field_name: str,
    fallback_host: str | None,
    fallback_port: int,
) -> tuple[str, int, str]:
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is empty")
    if text.startswith("/"):
        if not fallback_host:
            raise ValueError(f"{field_name} must include host information or request.host must be set")
        return fallback_host, fallback_port, text
    return _parse_remote_path(text)


def _load_private_key(private_key_path: str) -> paramiko.PKey:
    key_path = Path(private_key_path).expanduser().resolve()
    if not key_path.exists():
        raise ValueError(f"private_key_path does not exist: {key_path}")
    try:
        return paramiko.Ed25519Key.from_private_key_file(str(key_path))
    except paramiko.ssh_exception.SSHException:
        try:
            return paramiko.RSAKey.from_private_key_file(str(key_path))
        except paramiko.ssh_exception.SSHException as exc:
            raise ValueError(f"failed to load private key: {exc}") from exc


def run_remote_sbatch_yolo_train(
    request: RemoteSbatchYoloTrainRequest,
) -> RemoteSbatchYoloTrainResponse:
    yaml_host, yaml_port, yaml_remote_path = _resolve_remote_location(
        request.yaml_path,
        field_name="yaml_path",
        fallback_host=request.host,
        fallback_port=request.port,
    )
    root_host, root_port, root_remote_path = _resolve_remote_location(
        request.project_root_dir,
        field_name="project_root_dir",
        fallback_host=request.host or yaml_host,
        fallback_port=request.port if request.host else yaml_port,
    )
    if (yaml_host, yaml_port) != (root_host, root_port):
        raise ValueError("yaml_path and project_root_dir must point to the same remote host/port")

    username = request.username or _extract_username(request.yaml_path) or _extract_username(request.project_root_dir)
    if not username:
        raise ValueError("username is required: set username in request or use user@host:path")
    if request.password and request.private_key_path:
        raise ValueError("use either password or private_key_path, not both")
    if not request.password and not request.private_key_path:
        raise ValueError("either password or private_key_path is required")

    pkey = _load_private_key(request.private_key_path) if request.private_key_path else None

    stdout_path = request.stdout_path or str(Path(root_remote_path) / "logs" / "slurm-%j.out").replace("\\", "/")
    stderr_path = request.stderr_path or str(Path(root_remote_path) / "logs" / "slurm-%j.err").replace("\\", "/")
    log_dir = str(Path(stdout_path).parent).replace("\\", "/")
    if str(Path(stderr_path).parent).replace("\\", "/") != log_dir:
        raise ValueError("stdout_path and stderr_path must use the same parent directory")

    train_tokens: list[str] = [
        "conda",
        "run",
        "-n",
        request.yolo_train_env,
        "--no-capture-output",
        "yolo",
        "train",
        f"data={yaml_remote_path}",
        f"epochs={request.epochs}",
        f"imgsz={request.imgsz}",
        f"project={request.project}",
        f"name={request.name}",
    ]
    if request.batch is not None:
        train_tokens.append(f"batch={request.batch}")
    if request.workers is not None:
        train_tokens.append(f"workers={request.workers}")
    if request.cache is not None:
        train_tokens.append(f"cache={request.cache}")
    if request.device:
        train_tokens.append(f"device={request.device}")

    model_resolver_shell = build_remote_yolo_model_resolver_shell(request.model)
    train_command = " ".join(shlex.quote(token) for token in train_tokens)
    wrap_command = f'{model_resolver_shell}; {train_command} "model=$resolved_model"'
    sbatch_tokens: list[str] = [
        "sbatch",
        "--parsable",
        "--job-name",
        request.job_name,
        "--chdir",
        root_remote_path,
        "--output",
        stdout_path,
        "--error",
        stderr_path,
    ]
    if request.partition:
        sbatch_tokens.extend(["--partition", request.partition])
    if request.nodelist:
        sbatch_tokens.extend(["--nodelist", request.nodelist])
    if request.exclude:
        sbatch_tokens.extend(["--exclude", request.exclude])
    sbatch_tokens.extend(["--wrap", wrap_command])

    command = f"mkdir -p {shlex.quote(log_dir)} && {shlex.join(sbatch_tokens)}"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=yaml_host,
            port=yaml_port,
            username=username,
            password=request.password,
            pkey=pkey,
            timeout=30,
        )
    except paramiko.AuthenticationException as exc:
        raise ValueError(f"SSH authentication failed: {exc}") from exc
    except paramiko.SSHException as exc:
        raise ValueError(f"SSH connection failed: {exc}") from exc

    try:
        _, stdout, stderr = client.exec_command(command)
        exit_code = stdout.channel.recv_exit_status()
        stdout_text = stdout.read().decode("utf-8", errors="replace").strip()
        stderr_text = stderr.read().decode("utf-8", errors="replace").strip()
        if exit_code != 0:
            raise ValueError(
                f"remote sbatch submit failed (exit={exit_code}) on {yaml_host}:{yaml_port}: {stderr_text or stdout_text or 'unknown error'}"
            )
        first_line = stdout_text.splitlines()[0] if stdout_text else ""
        job_id = first_line.split(";", 1)[0].strip()
        if not job_id:
            raise ValueError(f"sbatch returned no job id: {stdout_text or stderr_text}")
        return RemoteSbatchYoloTrainResponse(
            yaml_path=yaml_remote_path,
            project_root_dir=root_remote_path,
            target_host=yaml_host,
            target_port=yaml_port,
            project=request.project,
            name=request.name,
            command=command,
            job_id=job_id,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout=stdout_text,
            stderr=stderr_text,
        )
    finally:
        client.close()
