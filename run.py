"""知擎启动入口：python run.py [--port 8080] [--host 0.0.0.0] [--reload]

用字符串导入路径 "app.main:app" 交给 uvicorn 而非直接传 app 对象，
是为了让 --reload 热重载能工作：uvicorn 需要在子进程里重新 import 应用。
"""

from __future__ import annotations

import argparse
import os

import uvicorn


def main() -> None:
    """解析命令行参数并启动 uvicorn 服务。

    参数取值优先级：命令行显式传入 > 环境变量(HOST/PORT) > 代码内默认值。
    这样同一份代码既能本地 python run.py 直接跑，也能在容器里用环境变量注入配置。
    """
    parser = argparse.ArgumentParser(description="知擎知识问答平台")
    # 0.0.0.0 接受所有网卡连接（局域网/容器可访问）；只本机调试可改 127.0.0.1
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8080)))
    # 生产环境不要开启 --reload
    parser.add_argument("--reload", action="store_true", help="开发模式：代码改动自动重载")
    args = parser.parse_args()

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
