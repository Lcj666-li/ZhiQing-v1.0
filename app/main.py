"""知擎应用入口：装配 FastAPI（中间件 / 路由 / 静态页面 / 异常处理）。

采用应用工厂（create_app 返回 app），装配集中、便于测试时另建干净实例；
模块底部的 app = create_app() 是 uvicorn 实际加载的对象。

启动方式：
    python run.py                       # 推荐
    uvicorn app.main:app --port 8080    # 也可以
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.errors import register_error_handlers
from app.api.middleware import TraceIdMiddleware
from app.api.routes import router
from app.conf.logging import logger
from app.conf.settings import settings


def create_app() -> FastAPI:
    """应用工厂：创建并配置好一个可直接运行的 FastAPI 实例。"""
    app = FastAPI(
        title=f"{settings.APP_NAME} 知识问答平台",
        description="FAQ 精确问答 + RAG 检索增强，支持多轮会话与流式输出",
        version="1.0.0",
        # 把 Swagger 文档与 OpenAPI 描述也放到统一前缀下，避免与业务/静态路径冲突
        docs_url=f"{settings.API_PREFIX}/docs",
        openapi_url=f"{settings.API_PREFIX}/openapi.json",
    )

    # 跨域：内网工具默认放开；生产应把 "*" 换成具体域名白名单
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # trace_id 中间件：为每个请求生成/透传链路 ID，写进 contextvar 供全程日志使用
    app.add_middleware(TraceIdMiddleware)

    # 统一错误处理：所有异常都转成 {code, message, trace_id}，前端只需按一种结构解析
    register_error_handlers(app)
    # 业务路由统一加前缀（如 /api/v1），便于版本管理与网关转发
    app.include_router(router, prefix=settings.API_PREFIX)

    # 静态资源 + 首页
    web_dir = settings.WEB_DIR
    os.makedirs(web_dir, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=web_dir), name="assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        """返回单页应用首页。"""
        return FileResponse(os.path.join(web_dir, "index.html"))

    @app.get("/health", include_in_schema=False)
    async def health_alias() -> dict:
        """给探针用的短路径。"""
        # 与 /api/v1/health 等价的裸路径，方便 K8s/负载均衡器做存活探测
        return {"status": "healthy"}

    logger.info("%s 应用装配完成，接口前缀 %s", settings.APP_NAME, settings.API_PREFIX)
    return app


# 模块级单例：uvicorn 通过 "app.main:app" 加载的就是这个对象（import 时即完成装配）
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    # 允许直接 python app/main.py 启动（等价于 run.py，但不支持 --reload 等参数）
    import uvicorn

    uvicorn.run("app.main:app", host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8080)),
                reload=False)
