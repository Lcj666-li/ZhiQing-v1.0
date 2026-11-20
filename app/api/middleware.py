"""请求级中间件：为每个请求生成 trace_id，打通日志与前端排查。

trace_id 写进 ContextVar：每个异步任务持有独立的值，并发请求互不串扰，
任意深度的代码都能用 trace_id_var.get() 取到当前请求的 ID，无需逐层传参。
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.conf.logging import logger, trace_id_var

# 约定的链路 ID 头名：请求进来先读它（上游/前端可透传），响应出去再写回它，形成闭环
TRACE_HEADER = "X-Trace-Id"


class TraceIdMiddleware(BaseHTTPMiddleware):
    """从请求头取 trace_id（没有就生成），写入上下文并回写到响应头。"""

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        # 请求头优先，缺失则生成 16 位短 ID
        trace_id = request.headers.get(TRACE_HEADER) or uuid.uuid4().hex[:16]
        # set() 返回旧值句柄，稍后 reset 还原，保证上下文干净
        token = trace_id_var.set(trace_id)
        started = time.time()
        try:
            response = await call_next(request)
        finally:
            # 即使下游抛异常也要记录耗时日志，便于排查慢请求/失败请求
            elapsed_ms = (time.time() - started) * 1000
            logger.info("%s %s -> %.1fms", request.method, request.url.path, elapsed_ms)
        # 回写响应头：前端/网关可用同一个 trace_id 去日志里检索整条链路
        response.headers[TRACE_HEADER] = trace_id
        # 还原 contextvar，防止 ID 泄漏到复用的执行上下文里
        trace_id_var.reset(token)
        return response
