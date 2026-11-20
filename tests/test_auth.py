"""鉴权层单测：JWT 签发与 Bearer 校验（不依赖任何外部服务）。

覆盖四个边界：登录成功 / 错误密码 401 / 缺 token 401 / 无效 token 401。
账号密码取自 settings 而非硬编码，改演示账号不会让测试误报。
这四条路径都不会走到业务服务（登录只比对配置、401 在依赖层被拦下），
故不注入 FakeService，直接用 TestClient(app)。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.conf.settings import settings
from app.main import app

# 所有 REST 接口的统一前缀（与 app/main.py 挂载路由时一致）
PREFIX = "/api/v1"


def _login(client, password):
    """登录辅助：用户名固定取配置，只变密码。"""
    return client.post(f"{PREFIX}/auth/login",
                       json={"username": settings.AUTH_USERNAME, "password": password})


def test_login_success():
    """正确账号密码：200 且返回非空 access_token、token_type=bearer。"""
    with TestClient(app) as client:
        resp = _login(client, settings.AUTH_PASSWORD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"


def test_login_wrong_password():
    """错误密码：401，不签发 token（统一错误处理器把 ApiError 转成规范 JSON）。"""
    with TestClient(app) as client:
        resp = _login(client, "wrong-password")
    assert resp.status_code == 401


def test_protected_route_requires_token():
    """不带 token：GET /kb/sources 被 require_auth 拦截，返回 401。"""
    with TestClient(app) as client:
        resp = client.get(f"{PREFIX}/kb/sources")
    assert resp.status_code == 401


def test_invalid_token_rejected():
    """无效 token：即便带了 Bearer 头，签名校验失败仍返回 401。"""
    with TestClient(app) as client:
        resp = client.get(f"{PREFIX}/kb/sources",
                          headers={"Authorization": "Bearer not-a-valid-jwt"})
    assert resp.status_code == 401