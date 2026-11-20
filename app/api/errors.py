"""统一错误处理：所有异常都转成 {code, message, trace_id} 结构，方便前端与排障。

架构位置：API 层的"异常出口"。在 app/main.py 里通过 register_error_handlers(app) 挂载。
请求链路中任意一层抛出异常，都会被这里的处理器接住，转成规范 JSON 返回，
避免把原始堆栈/敏感信息直接暴露给前端；同时借助 contextvar 里的 trace_id 关联日志。

设计意图：区分"可预期的业务异常(ApiError)"与"未预期异常(Exception)"——
前者用 warning 记录、返回业务自定义状态码；后者用 error + 堆栈记录、统一回 500 与友好话术。
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.conf.logging import logger, trace_id_var


class ApiError(Exception):
    """业务异常（可预期的错误）。

    由路由/服务层主动 raise，用来表达"参数不对、资源不存在、操作失败"等已知情况，
    与程序 bug 导致的未预期异常区分开，便于分级记录与返回不同的状态码。
    """

    def __init__(self, message: str, code: int = 400, status_code: int = 400) -> None:
        """构造业务异常。

        message：给用户看的错误说明；code：业务错误码（响应体内）；
        status_code：HTTP 状态码。两者分开，让业务语义码与 HTTP 传输码解耦。
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        # 调用父类构造，让 str(exc)、日志与堆栈里也能显示 message
        super().__init__(message)


def _payload(code: int, message: str) -> dict:
    """拼装统一错误响应体，并带上当前请求的 trace_id 方便按链路检索日志。"""
    # trace_id_var.get() 取的是中间件写入 contextvar 的值；无请求上下文时为默认 "-"
    return {"code": code, "message": message, "trace_id": trace_id_var.get()}


def register_error_handlers(app: FastAPI) -> None:
    """把异常处理器挂到 FastAPI 实例上。"""

    @app.exception_handler(ApiError)
    async def _handle_api_error(_: Request, exc: ApiError) -> JSONResponse:  # noqa: ANN001
        # 可预期错误用 warning 级别即可，不需要打印堆栈（不是程序 bug）
        logger.warning("业务异常：%s", exc.message)
        return JSONResponse(status_code=exc.status_code, content=_payload(exc.code, exc.message))

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:  # noqa: ANN001
        # 兜底：任何未被上面的处理器捕获的异常都到这里；exc_info=True 打印完整堆栈便于定位 bug
        logger.error("未预期异常：%s", exc, exc_info=True)
        # 对外只回友好话术，不泄漏内部异常细节（堆栈已写进日志，可用 trace_id 关联）
        return JSONResponse(status_code=500, content=_payload(500, "服务内部错误，请稍后重试"))
