"""接口出入参模型（Pydantic）：API 层的数据契约。

路由用这些模型声明请求体与响应体，入参校验、出参序列化与接口文档都以此为准，
在边界处挡住脏数据，也让前端联调有唯一可信来源。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """登录请求（POST /auth/login 的请求体）。"""

    username: str = Field(..., description="用户名", min_length=1)
    password: str = Field(..., description="密码", min_length=1)


class TokenResponse(BaseModel):
    """登录响应：签发好的 JWT，前端保存后放到 Authorization: Bearer <token>。"""

    access_token: str
    token_type: str = "bearer"


class AskRequest(BaseModel):
    """一次问答请求（POST /qa/ask 的请求体）。"""

    # min_length=1 拦截空问题
    query: str = Field(..., description="用户问题", min_length=1)
    # 知识域过滤：只在 RAG 通道生效，用于把检索范围限定到某个知识域（如 it、hr）
    subject: Optional[str] = Field(None, description="知识域过滤，如 it、hr")
    # 不传则由服务端新建一个会话 ID；传了则复用，实现多轮上下文记忆
    session_id: Optional[str] = Field(None, description="会话 ID，用于多轮对话")


class AskResponse(BaseModel):
    """非流式问答响应（POST /qa/ask 的返回体）。"""

    answer: str  # 回答正文；若需要走 RAG，这里会是一句"请改用流式接口"的提示
    # 告诉前端本次是否走了流式判定：True 表示该问题需要检索，应改用 WebSocket /qa/stream
    is_streaming: bool
    session_id: str  # 回传会话 ID，前端保存后用于后续多轮提问
    processing_time: float  # 本次处理耗时（秒），便于前端展示与性能观测
    # 本次回答引用的片段（FAQ 命中时是标准问句，RAG 时是文档片段）
    # default_factory=list：每次实例化都新建独立列表，避免多个响应共享同一个可变默认值
    sources: list[dict[str, Any]] = Field(default_factory=list)


class SessionCreated(BaseModel):
    """创建会话的响应（POST /sessions）：只回一个新分配的会话 ID。"""

    session_id: str


class SourceList(BaseModel):
    """可用知识域列表的响应（GET /kb/sources）：给前端下拉框填充选项。"""

    sources: list[str]


class HistoryResponse(BaseModel):
    """会话历史响应（GET /sessions/{id}/messages）：回传该会话最近若干轮问答。"""

    session_id: str
    # 每条历史是一轮问答的字典（含问题、回答、时间等），结构由服务层 recent_turns 决定
    history: list[dict[str, Any]]


class ActionResult(BaseModel):
    """通用操作结果（如清空会话历史）：status 表示成败，message 是给人看的说明。"""

    status: str
    message: str = ""
