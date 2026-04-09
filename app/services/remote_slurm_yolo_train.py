"""远程 SLURM YOLO 训练：自动获取 token 并提交 job/submit 任务。"""

import json
import os
import re
import shlex
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from app.schemas.preprocess import (
    RemoteSlurmYoloTrainRequest,
    RemoteSlurmYoloTrainResponse,
)


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


def _parse_path_only(target: str) -> str:
    text = target.strip()
    if not text:
        raise ValueError("path is empty")
    if text.startswith("/"):
        return text
    _, _, remote_path = _parse_remote_path(text)
    return remote_path


def _project_and_name_from_yaml(yaml_remote_path: str) -> tuple[str, str]:
    normalized = str(Path(yaml_remote_path)).replace("\\", "/")
    marker = "/dataset/"
    if marker not in normalized:
        raise ValueError(
            f"yaml_path must contain {marker!r} segment (got {yaml_remote_path!r})"
        )
    prefix = normalized.split(marker, 1)[0]
    project = str(Path(prefix) / "runs" / "train").replace("\\", "/")
    name = Path(normalized).stem
    return project, name


def _post_json(
    *,
    url: str,
    payload: dict,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    req = Request(url=url, data=body, headers=request_headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON response from {url}: {raw[:300]}") from e


def _fetch_slurm_token(slurm_user: str) -> str:
    token_api = os.getenv("SELF_API_SLURM_TOKEN_API", "http://172.31.1.9:2591/api/token").strip()
    lifespan = int(os.getenv("SELF_API_SLURM_TOKEN_LIFESPAN", "3600"))
    payload = {"username": slurm_user, "lifespan": lifespan}
    try:
        resp = _post_json(url=token_api, payload=payload, timeout=20)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ValueError(f"failed to fetch slurm token: HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise ValueError(f"failed to fetch slurm token: {e}") from e

    token = str(resp.get("token", "")).strip()
    if not token:
        raise ValueError(f"token API returned no token: {resp}")
    return token


def _build_submit_payload(
    *,
    request: RemoteSlurmYoloTrainRequest,
    yaml_remote_path: str,
    root_remote_path: str,
) -> dict:
    auto_project, auto_name = _project_and_name_from_yaml(yaml_remote_path)
    project = (request.project or auto_project).strip()
    name = (request.name or auto_name).strip()
    cache_value = request.cache if request.cache is not None else True
    if cache_value is False:
        raise ValueError("cache=False 不允许，按 subyolo 规则必须启用 cache")

    base_args: list[str] = [
        "train",
        f"model={shlex.quote(request.model)}",
        f"data={shlex.quote(yaml_remote_path)}",
        f"epochs={request.epochs}",
        f"imgsz={request.imgsz}",
        f"project={shlex.quote(project)}",
        f"name={shlex.quote(name)}",
        f"cache={cache_value}",
    ]
    if request.workers is not None:
        base_args.append(f"workers={request.workers}")
    if request.batch is not None:
        base_args.append(f"batch={request.batch}")
    if request.device:
        base_args.append(f"device={shlex.quote(request.device)}")

    # 复用 yolotrain.sh 的核心规则：
    # - 自动探测 GPU 列表并设置 CUDA_VISIBLE_DEVICES
    # - 若未显式提供 device，则按 GPU 数自动扩增 batch
    final_args_expr = " ".join(base_args)
    shell_script = (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "source ~/.bashrc >/dev/null 2>&1 || true\n"
        "module load micromamba >/dev/null 2>&1 || true\n"
        f"micromamba activate {shlex.quote(os.getenv('SELF_API_YOLO_CONDA_ENV', '/mnt/usrhome/lsl/ndata/conda/envs/yolo'))}\n"
        "gpu_list=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, || true)\n"
        "if [ -z \"${gpu_list}\" ]; then gpu_list=\"0\"; fi\n"
        "export CUDA_VISIBLE_DEVICES=\"$gpu_list\"\n"
        "num_gpus=$(echo \"$gpu_list\" | awk -F',' '{print NF}')\n"
        "if [ -z \"$num_gpus\" ] || [ \"$num_gpus\" -le 0 ]; then num_gpus=1; fi\n"
        f"YOLO_ARGS='{final_args_expr}'\n"
        "if [[ \"$YOLO_ARGS\" != *\"device=\"* ]]; then\n"
        "  if [[ \"$YOLO_ARGS\" =~ batch=([0-9]+) ]]; then\n"
        "    base_batch=${BASH_REMATCH[1]}\n"
        "  else\n"
        "    base_batch=8\n"
        "    YOLO_ARGS=\"$YOLO_ARGS batch=$base_batch\"\n"
        "  fi\n"
        "  batch=$((base_batch * num_gpus))\n"
        "  YOLO_ARGS=\"${YOLO_ARGS} device=${gpu_list}\"\n"
        "  YOLO_ARGS=$(echo \"$YOLO_ARGS\" | sed -E \"s/batch=[0-9]+/batch=${batch}/\")\n"
        "fi\n"
        f"cd {shlex.quote(root_remote_path)}\n"
        "echo \"Final command: yolo ${YOLO_ARGS}\"\n"
        "eval yolo ${YOLO_ARGS}\n"
    )
    partition = (request.partition or os.getenv("SELF_API_SLURM_PARTITION", "gpu")).strip() or "gpu"
    stdout_template = os.getenv("SELF_API_SLURM_STDOUT_TEMPLATE", "/tmp/slurm-%j.out").strip()
    stderr_template = os.getenv("SELF_API_SLURM_STDERR_TEMPLATE", "/tmp/slurm-%j.err").strip()
    # Slurm REST 任务环境建议显式传 PATH，避免远端 batch 环境过于精简。
    env_list = [
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    ]
    job: dict = {
        "name": "self_api_train",
        "partition": partition,
        "tasks": 1,
        "current_working_directory": root_remote_path,
        "environment": env_list,
        "standard_output": stdout_template,
        "standard_error": stderr_template,
    }
    if request.nodelist:
        job["required_nodes"] = request.nodelist.strip()
    if request.exclude:
        job["excluded_nodes"] = request.exclude.strip()

    return {
        "script": shell_script,
        "job": job,
    }


def run_remote_slurm_yolo_train(
    request: RemoteSlurmYoloTrainRequest,
) -> RemoteSlurmYoloTrainResponse:
    yaml_remote_path = _parse_path_only(request.yaml_path)
    root_remote_path = _parse_path_only(request.project_root_dir)
    slurm_user = (request.username or "").strip()
    if not slurm_user:
        raise ValueError("username is required: used as slurm token user")

    token = _fetch_slurm_token(slurm_user)
    slurm_submit_url = os.getenv(
        "SELF_API_SLURM_SUBMIT_API",
        "http://172.31.1.9:6820/slurm/v0.0.42/job/submit",
    ).strip()
    payload = _build_submit_payload(
        request=request,
        yaml_remote_path=yaml_remote_path,
        root_remote_path=root_remote_path,
    )
    command = (
        f"POST {slurm_submit_url} "
        f"(Authorization: Bearer <auto-token-for-{slurm_user}>)"
    )
    try:
        submit_resp = _post_json(
            url=slurm_submit_url,
            payload=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        return RemoteSlurmYoloTrainResponse(
            status="failed",
            yaml_path=yaml_remote_path,
            project_root_dir=root_remote_path,
            target_host="172.31.1.9",
            target_port=6820,
            command=command,
            exit_code=1,
            stdout="",
            stderr=f"submit failed: HTTP {e.code}: {detail}",
        )
    except URLError as e:
        return RemoteSlurmYoloTrainResponse(
            status="failed",
            yaml_path=yaml_remote_path,
            project_root_dir=root_remote_path,
            target_host="172.31.1.9",
            target_port=6820,
            command=command,
            exit_code=1,
            stdout="",
            stderr=f"submit failed: {e}",
        )

    errors = submit_resp.get("errors") or []
    exit_code = 0 if not errors else 1
    return RemoteSlurmYoloTrainResponse(
        status="ok" if exit_code == 0 else "failed",
        yaml_path=yaml_remote_path,
        project_root_dir=root_remote_path,
        target_host="172.31.1.9",
        target_port=6820,
        command=command,
        exit_code=exit_code,
        stdout=json.dumps(
            {
                "job_id": submit_resp.get("job_id"),
                "step_id": submit_resp.get("step_id"),
                "payload": payload,
                "submit_response": submit_resp,
            },
            ensure_ascii=False,
        ),
        stderr=json.dumps(errors, ensure_ascii=False),
    )
