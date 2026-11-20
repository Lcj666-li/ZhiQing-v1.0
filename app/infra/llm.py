"""大模型访问层：把 DashScope（OpenAI 兼容协议）封装成一个可注入的调用对象。

上层只依赖"prompt -> token 流"抽象（ChatModel 实例可直接调用，逐 token 产出）；
换 OpenAI 兼容服务商（DeepSeek / Moonshot / vLLM）只改 base_url / api_key / model 配置。
"""

from __future__ import annotations

import time
from typing import Iterator

from openai import OpenAI

from app.conf.logging import logger
from app.conf.settings import settings

# 全局 system 提示：给模型设定"人设与底线"（专业严谨、不确定要说明），
# 每次调用都会作为第一条 message 注入，统一约束所有通道的回答风格。
SYSTEM_PROMPT = "你是知擎助手，回答专业、严谨、简洁，不确定的信息要明确说明。"


class ChatModel:
    """OpenAI 兼容聊天模型（流式）。

    一个实例持有一个 openai.OpenAI 客户端，可被反复调用；
    通过 __call__ 支持流式（逐 token），与 RAG 管道的 llm 抽象一致。
    """

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 base_url: str | None = None, timeout: int | None = None) -> None:
        """构造客户端。四个参数都支持"显式传入 > 配置默认"，便于测试时注入假模型。"""
        self.model = model or settings.LLM_MODEL
        self.timeout = timeout or settings.LLM_TIMEOUT
        self._client = OpenAI(
            api_key=api_key or settings.DASHSCOPE_API_KEY,
            base_url=base_url or settings.DASHSCOPE_BASE_URL,
            timeout=self.timeout,
        )
    # ------------------------------------------------------------------ #
    @staticmethod
    def _messages(prompt: str) -> list[dict]:
        """把单条 prompt 组装成 OpenAI 要求的 messages 列表：system 人设 + user 问题。"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

    def __call__(self, prompt: str) -> Iterator[str]:
        """流式调用：逐 token yield。

        新增文本在 choices[0].delta.content；空 chunk / 空 delta 跳过；记录首 token 耗时。
        """
        started = time.time()
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(prompt),
            timeout=self.timeout,
            stream=True,
        )
        first_token_at = None
        for chunk in completion:
            # 有些 chunk 只带角色/结束标记，没有 choices，直接跳过
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content
            if piece:
                if first_token_at is None:
                    first_token_at = time.time() - started
                    logger.debug("首 token 耗时 %.2fs", first_token_at)
                yield piece
