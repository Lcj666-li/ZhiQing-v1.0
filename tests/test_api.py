"""API 层单测：TestClient + dependency_overrides 注入 FakeService，不启动真实服务。

只测 HTTP/WebSocket 的输入输出契约：健康检查、Trace-Id、知识域列表、
会话生命周期、首页、非流式问答三分支（寒暄/转流式/参数校验）、
WS 流式协议（start->token*->end）。业务逻辑由 test_pipeline / test_faq_retriever 覆盖。
业务接口受 JWT 保护：fixture 先登录拿 token；WS 握手无法带自定义头，token 走 ?token=。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.api.routes import get_service, reset_service
from app.conf.settings import settings
from app.main import app
from tests.fakes import FakeService

# 所有 REST 接口的统一前缀（与 app/main.py 挂载路由时一致）
PREFIX = "/api/v1"


@pytest.fixture
def client():
    """每个测试独享的 TestClient：注入 FakeService、登录拿 JWT、结束后清理覆盖。

    reset_service() 防止上一个测试的服务单例残留污染本测试。
    token 同时存到 client.auth_token，供 WebSocket 以 ?token= 使用。
    """
    reset_service()
    service = FakeService()
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as c:
        token = c.post(f"{PREFIX}/auth/login",
                       json={"username": settings.AUTH_USERNAME,
                             "password": settings.AUTH_PASSWORD}).json()["access_token"]
        c.headers.update({"Authorization": f"Bearer {token}"})
        c.auth_token = token
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    """健康检查接口可用：返回 200 且 status=healthy（运维探活依赖它，无需鉴权）。"""
    resp = client.get(f"{PREFIX}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_trace_id_header(client):
    """链路追踪中间件生效：任意响应都应带 X-Trace-Id 头（排障时用它串日志）。"""
    resp = client.get(f"{PREFIX}/health")
    assert resp.headers.get("X-Trace-Id")


def test_sources(client):
    """知识域列表接口：返回 200 且与配置完全一致（前端下拉框数据源）。

    断言"接口返回 == settings.VALID_SOURCES"而不是硬编码某个域名：
    知识域随业务配置变化（当前为 7 个域），测契约不测魔法值。
    """
    resp = client.get(f"{PREFIX}/kb/sources")
    assert resp.status_code == 200
    assert resp.json()["sources"] == settings.VALID_SOURCES


def test_create_session_and_history(client):
    """会话生命周期：创建返回 36 位 UUID -> 新会话历史为空 -> 清空操作成功。"""
    session_id = client.post(f"{PREFIX}/sessions").json()["session_id"]
    assert len(session_id) == 36

    resp = client.get(f"{PREFIX}/sessions/{session_id}/messages")
    assert resp.status_code == 200
    assert resp.json()["history"] == []

    assert client.delete(f"{PREFIX}/sessions/{session_id}/messages").json()["status"] == "success"


def test_index_page(client):
    """根路径返回前端首页：200 且内容是 HTML（静态页面挂载正确）。"""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


def test_ask_greeting(client):
    """寒暄分支：'你好'命中路由层正则话术，直接返回固定欢迎语，
    is_streaming=False 表示无需走流式接口（不消耗任何模型调用）。"""
    resp = client.post(f"{PREFIX}/qa/ask", json={"query": "你好"})
    body = resp.json()
    assert body["is_streaming"] is False
    assert "知擎" in body["answer"]


def test_ask_need_stream(client):
    """转流式分支：FAQ 未命中且需要 RAG 时，非流式接口不硬答，
    而是返回 is_streaming=True 提示客户端改用 /qa/stream。
    FakeService 默认 faq_answer=None => need_rag=True，走"请使用流式接口"。"""
    body = client.post(f"{PREFIX}/qa/ask", json={"query": "任意问题"}).json()
    assert body["is_streaming"] is True


def test_ask_validation_error_on_empty_query(client):
    """参数校验分支：空 query 被 Pydantic 请求模型拦截返回 422，不进入业务逻辑。"""
    resp = client.post(f"{PREFIX}/qa/ask", json={"query": ""})
    assert resp.status_code == 422


def test_websocket_stream(client):
    """WebSocket 流式协议：事件序列必须是 start 开头、end 结尾，
    中间 token 事件拼接出的完整文本与 FakeService 预设答案一致。
    这验证了路由层对 (token, done) 二元组的转发顺序正确。
    握手无法带 Authorization 头，token 走 ?token= 查询参数。"""
    session_id = client.post(f"{PREFIX}/sessions").json()["session_id"]
    events, text = [], ""
    with client.websocket_connect(f"{PREFIX}/qa/stream?token={client.auth_token}") as ws:
        ws.send_text(json.dumps({"query": "知擎怎么用", "session_id": session_id, "subject": "it"}))
        # 限制接收条数，防止协议异常时死循环
        for _ in range(20):
            msg = ws.receive_json()
            events.append(msg["type"])
            if msg["type"] == "token":
                text += msg["token"]
            if msg["type"] in {"end", "error"}:
                break

    assert events[0] == "start"
    assert events[-1] == "end"
    assert text == "这是流式回答"


def test_websocket_greeting(client):
    """WebSocket 寒暄短路：'您好'不走 RAG，协议仍完整——
    start -> 单条 token（固定话术）-> end，共三条消息。"""
    with client.websocket_connect(f"{PREFIX}/qa/stream?token={client.auth_token}") as ws:
        ws.send_text(json.dumps({"query": "您好"}))
        first = ws.receive_json()
        token = ws.receive_json()
        end = ws.receive_json()
    assert first["type"] == "start"
    assert token["type"] == "token" and "知擎" in token["token"]
    assert end["type"] == "end"