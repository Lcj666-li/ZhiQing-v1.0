"""命令行问答：复用与 Web 相同的服务层，终端里逐 token 打印。

用法：
    python -m scripts.cli
    python -m scripts.cli --session demo-1 --subject it

运行前需准备好 MySQL/Milvus/bge-m3/大模型等依赖（见 docker-compose.yml 与 README）。
"""

from __future__ import annotations

import argparse
import sys
import uuid

from app.conf.logging import logger
from app.conf.settings import settings
from app.services.qa_service import QaService


def main() -> int:
    """CLI 主循环。返回进程退出码：0 正常退出，1 异常退出。"""
    parser = argparse.ArgumentParser(description="知擎命令行问答")
    parser.add_argument("--session", default=None, help="会话 ID（用于多轮）")
    parser.add_argument("--subject", default=None, help="知识域过滤，如 it")
    args = parser.parse_args()

    session_id = args.session or str(uuid.uuid4())
    print(f"{settings.APP_NAME} 已启动 | 会话 {session_id} | 输入 exit 退出")
    service = QaService.build()

    while True:
        try:
            query = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            return 0
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("再见")
            return 0

        print("知擎：", end="", flush=True)
        for token, done in service.stream_answer(query, session_id=session_id, subject=args.subject):
            if done:
                print()
                break
            print(token, end="", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        logger.error("CLI 异常退出：%s", exc, exc_info=True)
        sys.exit(1)
