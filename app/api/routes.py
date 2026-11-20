"""REST / WebSocket 路由。

路由只做"参数校验 + 调用服务 + 组织响应"，业务逻辑都在 services 层。

架构位置：请求进入后的第一站业务代码（前面只有中间件）。整体流向：
    浏览器 → FastAPI 路由(本文件) → QaService → FAQ 通道(问句向量命中) / RAG 通道(检索+大模型)。
本文件提供三类接口：
    · 鉴权：POST /auth/login 签发 JWT，业务接口用 Bearer token 保护（见 auth.py）；
    · REST：健康检查、知识域列表、会话创建/历史/清空、非流式问答 /qa/ask；
    · WebSocket：流式问答 /qa/stream，边生成边推送，体验类似"打字机"（token 走 ?token=）。

分层原则：路由层保持"薄"，只负责协议与校验；真正的双通道分流、降级、
会话落库都在 QaService 里，这样业务可单独测试、复用（CLI 也能直接调服务）。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from app.api.auth import create_token, require_auth, verify_token
from app.api.errors import ApiError
from app.api.schemas import (ActionResult, AskRequest, AskResponse, HistoryResponse,
                             LoginRequest, SessionCreated, SourceList, TokenResponse)
from app.conf.logging import logger
from app.conf.settings import settings
from app.rag.types import RagTrace
from app.services.qa_service import QaService

router = APIRouter()

# 进程内单例（首次请求时装配，避免 import 阶段就加载模型）
_service: QaService | None = None


def get_service() -> QaService:
    """FastAPI 依赖：获取问答服务。测试里可用 app.dependency_overrides 替换。

    懒加载单例：第一次调用才 build()（连库、加载模型，较重），之后复用同一实例。
    """
    global _service
    if _service is None:
        _service = QaService.build()
    return _service


def reset_service() -> None:
    """测试辅助：清空服务单例，防止脏单例串到下一个用例。"""
    global _service
    _service = None


# 寒暄没有检索价值，走 FAQ/RAG 会白耗一次向量检索甚至大模型调用，
# 故用正则直接回固定话术：零延迟、零成本。每条是 (正则, 回复) 二元组。
GREETING_RULES = [
    (r"^(你好|您好|hi|hello)", f"你好！我是{settings.APP_NAME}助手，可以帮你查制度、查资料、答疑，直接问我就行。"),
    (r"^(你是谁|您是谁|你叫什么|who are you)", f"我是{settings.APP_NAME}，企业知识库问答助手，回答会尽量给出处。"),
    (r"^(在吗|在不在|有人吗)", "我在，随时可以提问。"),
    (r"^(谢谢|多谢|感谢)", "不客气，还有其他问题随时问我。"),
]


def match_greeting(query: str) -> Optional[str]:
    """命中寒暄返回对应固定话术，否则返回 None（交给后续 FAQ/RAG）。"""
    # (query or "") 兜住 None，strip 避免" 你好 "这类漏判
    text = (query or "").strip()
    for pattern, reply in GREETING_RULES:
        if re.match(pattern, text, re.IGNORECASE):
            return reply
    return None


# ---------------------------------------------------------------------- #
# 基础接口
# ---------------------------------------------------------------------- #
@router.get("/health", summary="健康检查")
async def health() -> dict:
    """健康检查：返回服务存活状态与应用名，供负载均衡/容器探针轮询（无需鉴权）。"""
    return {"status": "healthy", "app": settings.APP_NAME}


# ---------------------------------------------------------------------- #
# 鉴权
# ---------------------------------------------------------------------- #
@router.post("/auth/login", response_model=TokenResponse, summary="登录换取 JWT")
async def login(payload: LoginRequest) -> TokenResponse:
    """校验账号密码（config.ini [auth] 段），成功签发 JWT。

    单账号设计：知识库是内部工具，不做用户表/注册，账号密码放配置里，
    生产环境用环境变量 AUTH_PASSWORD / JWT_SECRET 覆盖。
    """
    if payload.username != settings.AUTH_USERNAME or payload.password != settings.AUTH_PASSWORD:
        raise ApiError("用户名或密码错误", code=401, status_code=401)
    return TokenResponse(access_token=create_token(payload.username))


@router.get("/kb/sources", response_model=SourceList, summary="可用的知识域列表")
async def list_sources(_user: str = Depends(require_auth)) -> SourceList:
    """返回可选知识域（学科）列表，前端据此渲染筛选下拉框。"""
    return SourceList(sources=settings.VALID_SOURCES)


# ---------------------------------------------------------------------- #
# 会话
# ---------------------------------------------------------------------- #
@router.post("/sessions", response_model=SessionCreated, summary="创建会话")
async def create_session(_user: str = Depends(require_auth)) -> SessionCreated:
    """创建一个新会话，返回随机 UUID 作为 session_id，后续提问带上它串联上下文。"""
    return SessionCreated(session_id=str(uuid.uuid4()))


@router.get("/sessions/{session_id}/messages", response_model=HistoryResponse, summary="查询会话历史")
async def get_history(session_id: str, service: QaService = Depends(get_service),
                      _user: str = Depends(require_auth)) -> HistoryResponse:
    """查询某会话最近若干轮问答历史。"""
    try:
        return HistoryResponse(session_id=session_id, history=service.recent_turns(session_id))
    except Exception as exc:  # pragma: no cover
        # 把底层异常包装成统一的 ApiError；from exc 保留原始异常链，日志里能看到根因
        raise ApiError(f"获取会话历史失败：{exc}", code=500, status_code=500) from exc


@router.delete("/sessions/{session_id}/messages", response_model=ActionResult, summary="清空会话历史")
async def clear_history(session_id: str, service: QaService = Depends(get_service),
                        _user: str = Depends(require_auth)) -> ActionResult:
    """清空某会话的历史（服务层做软删除，保留数据可追溯）。"""
    if service.clear_session(session_id):
        return ActionResult(status="success", message="会话历史已清空")
    # 返回 False 视为失败，抛业务异常交给统一错误处理器
    raise ApiError("清空会话历史失败", code=500, status_code=500)


# ---------------------------------------------------------------------- #
# 问答（非流式，只做快速判定）
# ---------------------------------------------------------------------- #
@router.post("/qa/ask", response_model=AskResponse, summary="非流式问答")
async def ask(payload: AskRequest, service: QaService = Depends(get_service),
              _user: str = Depends(require_auth)) -> AskResponse:
    """非流式问答：只做 FAQ 快速判定，不在这里跑完整 RAG。

    设计取舍：RAG 检索+大模型生成较慢且天然是流式的，若在同步 REST 接口里等它跑完，
    请求会长时间挂起。所以本接口只尝试"秒回"的 FAQ：
        · 命中 FAQ → 直接返回标准答案；
        · 需要 RAG → 返回提示，让前端改用 WebSocket /qa/stream 获取流式回答。
    """
    started = time.time()
    # 没传会话就新建一个，保证响应里总有 session_id 供前端后续多轮使用
    session_id = payload.session_id or str(uuid.uuid4())

    # 寒暄最优先：命中直接秒回，不消耗检索/模型资源
    greeting = match_greeting(payload.query)
    if greeting:
        return AskResponse(answer=greeting, is_streaming=False, session_id=session_id,
                           processing_time=time.time() - started)

    # 只调 FAQ 通道，不走 service.stream_answer，故不会触发 RAG
    trace = RagTrace()
    try:
        answer, need_rag = service.faq.match(payload.query, trace=trace)
    except Exception as exc:
        # FAQ 出错不阻断：记日志并按"需要 RAG"处理，把问题引导到流式接口
        logger.error("FAQ 查询异常：%s", exc)
        answer, need_rag = None, True

    # 需要检索：不同步等待 RAG，is_streaming=True 提示前端改用流式接口
    if need_rag:
        return AskResponse(answer="该问题需要检索知识库，请使用流式接口 /qa/stream 获取回答",
                           is_streaming=True, session_id=session_id,
                           processing_time=time.time() - started)

    # FAQ 命中：返回标准答案与引用来源（answer or "" 兜住 None）
    return AskResponse(answer=answer or "", is_streaming=False, session_id=session_id,
                       processing_time=time.time() - started,
                       sources=trace.as_dict()["sources"])


# ---------------------------------------------------------------------- #
# 问答（WebSocket 流式）
# ---------------------------------------------------------------------- #
@router.websocket("/qa/stream")
async def qa_stream(websocket: WebSocket, service: QaService = Depends(get_service)) -> None:
    """WebSocket 流式问答接口。

    协议时序：
        客户端发送 JSON：{query, subject, session_id}
        服务端推送：start → token*（0 到多个片段）→ end（正常结束）或 error（出错）
    一条连接可复用多轮：while True 循环里每收到一条消息就完成一次"start→token*→end"。

    鉴权：WebSocket 握手无法自定义 Authorization 头，token 走 URL 查询参数
    ?token=<JWT>，校验失败直接以 4401 关闭连接（不进入问答循环）。
    """
    # 先验 token 再 accept：无效连接不占用任何服务资源
    token = websocket.query_params.get("token", "")
    try:
        verify_token(token)
    except ValueError:
        await websocket.close(code=4401)
        return

    await websocket.accept()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                # 非法 JSON：回一条 error 帧并 continue，不断开连接，允许客户端重发
                await websocket.send_json({"type": "error", "error": "请求必须是 JSON 字符串"})
                continue

            query = (payload.get("query") or "").strip()
            # 向后兼容老客户端的 source_filter 字段
            subject = payload.get("subject") or payload.get("source_filter")
            # 没带会话就新建，保证多轮上下文可用
            session_id = payload.get("session_id") or str(uuid.uuid4())
            started = time.time()

            # 先发 start 帧：告诉前端"开始回答了"，并回传 session_id 供其保存
            await websocket.send_json({"type": "start", "session_id": session_id, "app": settings.APP_NAME})

            # 寒暄同样要走完整协议：前面已发 start，这里补一个 token + 一个 end，前端逻辑才统一
            greeting = match_greeting(query)
            if greeting:
                await websocket.send_json({"type": "token", "token": greeting, "session_id": session_id})
                await websocket.send_json({"type": "end", "session_id": session_id, "is_complete": True,
                                           "processing_time": time.time() - started, "sources": []})
                continue

            # 空问题直接回 error 帧；注意校验放在 start 之后，保证协议时序一致
            if not query:
                await websocket.send_json({"type": "error", "error": "query 不能为空"})
                continue

            # trace 收集本次引用来源，end 帧里通过 **trace.as_dict() 一并回传
            trace = RagTrace()
            for token, is_complete in service.stream_answer(query, session_id=session_id, subject=subject,
                                                            trace=trace):
                # FAQ 命中时 token 与 is_complete 同帧产出：必须先发 token 再发 end，
                # 否则前端收到 end 即停止接收，这条答案就丢了。两个 if 顺序不能颠倒。
                if token:
                    await websocket.send_json({"type": "token", "token": token, "session_id": session_id})
                if is_complete:
                    await websocket.send_json({"type": "end", "session_id": session_id, "is_complete": True,
                                               "processing_time": time.time() - started,
                                               **trace.as_dict()})
                    break
                # 10ms 节流：给前端渲染留喘息，也让出事件循环给其它连接（仅在未结束时执行）
                await asyncio.sleep(0.01)

    except WebSocketDisconnect as exc:
        # 客户端正常断开（关页面/切走）：这是预期情况，记 info 即可，不当作错误
        logger.info("WebSocket 断开：code=%s", getattr(exc, "code", "-"))
    except Exception as exc:  # pragma: no cover
        # 其它未预期异常：记 error + 堆栈，并尽量给前端回一条 error 帧告知失败
        logger.error("WebSocket 异常：%s", exc, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            # 连接可能已损坏，连 error 帧都发不出去；此时静默忽略，避免二次异常掩盖原始错误
            pass
