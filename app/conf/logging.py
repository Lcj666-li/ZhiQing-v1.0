"""知擎日志设施：控制台 + 文件双输出，自动带上链路 trace_id。

架构位置：横切所有层的基础设施。中间件把 trace_id 写进 ContextVar，本模块的过滤器在每条日志
产出时读取它并注入，于是无论日志来自路由、服务还是基础设施，都带着同一个链路 ID，排障时
用 trace_id 一搜就能串起一次请求的完整轨迹（配合 errors.py 回给前端的 trace_id）。

用法：
    from app.conf.logging import logger
    logger.info("...")

需要按模块命名时：
    from app.conf.logging import get_logger
    log = get_logger(__name__)
"""

from __future__ import annotations

import logging
import os
from contextvars import ContextVar

from app.conf.settings import settings

# 贯穿一次请求的链路 ID（由 API 中间件写入）；default="-" 供无请求上下文时（如启动阶段）占位
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

# %(trace_id)s 是自定义字段，靠下面的 _TraceFilter 注入后才存在
_FORMAT = "%(asctime)s | %(levelname)-7s | %(trace_id)s | %(name)s:%(lineno)d | %(message)s"


class _TraceFilter(logging.Filter):
    """把当前上下文的 trace_id 注入日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        # 每条日志实时取当前上下文的 trace_id，并发请求互不干扰
        record.trace_id = trace_id_var.get()
        return True


def build_logger(name: str = "zhiqing", level: int = logging.DEBUG,
                 file_level: int = logging.DEBUG, console_level: int = logging.INFO,
                 log_file: str | None = None) -> logging.Logger:
    """创建（或复用）一个带控制台与文件双 handler 的 logger。

    name：logger 名（同名复用同一对象）；level：logger 自身最低级别；
    file_level/console_level：文件/控制台各自的级别；log_file：默认取 settings.LOG_FILE。
    """
    log_file = log_file or settings.LOG_FILE
    # 先确保日志目录存在，否则 FileHandler 打开文件会失败
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    log = logging.getLogger(name)
    log.setLevel(level)
    # 关闭 propagate：不再上交给 root logger，避免与根 handler 重复打印同一条
    log.propagate = False

    # 只有第一次（还没有 handler）才装配，防止多次调用叠加 handler 导致日志重复
    if not log.handlers:
        formatter = logging.Formatter(_FORMAT)
        trace_filter = _TraceFilter()

        console = logging.StreamHandler()
        console.setLevel(console_level)
        console.setFormatter(formatter)
        console.addFilter(trace_filter)

        # 文件落盘更详尽（DEBUG 级），便于事后排查
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(trace_filter)

        log.addHandler(console)
        log.addHandler(file_handler)

    return log


def get_logger(name: str, **kwargs) -> logging.Logger:
    """按模块名取 logger（同名复用，不会重复挂 handler），通常传 __name__。"""
    return build_logger(name, **kwargs)


# 全项目默认 logger
logger = build_logger("zhiqing")
