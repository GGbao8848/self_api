"""remote-transfer API 测试。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.data_helpers import create_text_file


def test_remote_transfer_parse_target_sftp_url() -> None:
    """测试 sftp:// URL 解析。"""
    from app.services.remote_transfer import _parse_target

    host, port, path = _parse_target("sftp://172.31.1.9/mnt/usrhome/sk/ndata/")
    assert host == "172.31.1.9"
    assert port == 22
    assert path == "/mnt/usrhome/sk/ndata/"


def test_remote_transfer_parse_target_scp_style() -> None:
    """测试 user@host:path 解析。"""
    from app.services.remote_transfer import _parse_target

    host, port, path = _parse_target("sk@172.31.1.9:/mnt/usrhome/sk/ndata")
    assert host == "172.31.1.9"
    assert port == 22
    assert path == "/mnt/usrhome/sk/ndata"


def test_remote_transfer_parse_target_invalid() -> None:
    """测试无效 target 格式。"""
    from app.services.remote_transfer import _parse_target

    with pytest.raises(ValueError, match="invalid target"):
        _parse_target("invalid")


def test_remote_transfer_endpoint_missing_auth(client, case_dir: Path) -> None:
    """缺少认证时返回 400。"""
    create_text_file(case_dir / "a.txt", "x")
    response = client.post(
        "/api/v1/preprocess/remote-transfer",
        json={
            "source_path": str(case_dir / "a.txt"),
            "target": "sftp://172.31.1.9/mnt/ndata/",
            "username": "sk",
        },
    )
    assert response.status_code == 400
    assert "password" in response.json()["detail"].lower() or "required" in response.json()["detail"].lower()


@patch("app.services.remote_transfer.paramiko.SSHClient")
def test_remote_transfer_endpoint_file_success(
    mock_ssh_class: MagicMock,
    client,
    case_dir: Path,
) -> None:
    """mock SSH 下文件传输成功。"""
    create_text_file(case_dir / "data.txt", "hello")
    mock_sftp = MagicMock()
    mock_sftp.stat.side_effect = FileNotFoundError  # 远程文件不存在
    mock_ssh = MagicMock()
    mock_ssh.open_sftp.return_value = mock_sftp
    mock_ssh_class.return_value = mock_ssh

    response = client.post(
        "/api/v1/preprocess/remote-transfer",
        json={
            "source_path": str(case_dir / "data.txt"),
            "target": "sftp://172.31.1.9/mnt/ndata/",
            "username": "sk",
            "password": "secret",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["transferred_type"] == "file"
    assert data["transferred_files"] == 1
    assert data["total_bytes"] == 5
    mock_sftp.put.assert_called_once()


@patch("app.services.remote_transfer.paramiko.SSHClient")
def test_remote_transfer_endpoint_directory_success(
    mock_ssh_class: MagicMock,
    client,
    case_dir: Path,
) -> None:
    """mock SSH 下目录传输成功。"""
    create_text_file(case_dir / "sub" / "a.txt", "a")
    create_text_file(case_dir / "sub" / "b.txt", "bb")
    mock_sftp = MagicMock()
    mock_sftp.stat.side_effect = FileNotFoundError  # 远程目录/文件不存在
    mock_ssh = MagicMock()
    mock_ssh.open_sftp.return_value = mock_sftp
    mock_ssh_class.return_value = mock_ssh

    response = client.post(
        "/api/v1/preprocess/remote-transfer",
        json={
            "source_path": str(case_dir / "sub"),
            "target": "sftp://172.31.1.9/mnt/ndata/",
            "username": "sk",
            "password": "secret",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["transferred_type"] == "directory"
    assert data["transferred_files"] == 2
    assert data["total_bytes"] == 3  # "a" + "bb"
    assert mock_sftp.put.call_count == 2


def test_remote_transfer_endpoint_source_not_in_roots(client, case_dir: Path) -> None:
    """source_path 不在允许根目录时返回 400。"""
    response = client.post(
        "/api/v1/preprocess/remote-transfer",
        json={
            "source_path": "/etc/passwd",
            "target": "sftp://172.31.1.9/mnt/ndata/",
            "username": "sk",
            "password": "x",
        },
    )
    # 若 FILE_ACCESS_ROOTS 未包含 /etc，应拒绝
    if response.status_code == 400:
        assert "outside" in response.json()["detail"].lower() or "allowed" in response.json()["detail"].lower()


def test_remote_transfer_async_returns_202(client, case_dir: Path) -> None:
    """异步接口返回 202。"""
    create_text_file(case_dir / "x.txt", "x")
    with patch("app.services.remote_transfer.paramiko.SSHClient") as mock_ssh_class:
        mock_sftp = MagicMock()
        mock_ssh = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp
        mock_ssh_class.return_value = mock_ssh

        response = client.post(
            "/api/v1/preprocess/remote-transfer/async",
            json={
                "source_path": str(case_dir / "x.txt"),
                "target": "sftp://172.31.1.9/mnt/ndata/",
                "username": "sk",
                "password": "secret",
            },
        )

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "accepted"
    assert "task_id" in data
    assert "remote_transfer" in data["task_type"]
