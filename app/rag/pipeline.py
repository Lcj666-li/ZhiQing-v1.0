"""RAG 主流程：意图识别 -> 知识域路由 -> bge-m3 向量检索 -> 上下文拼装 -> 流式生成。

【架构位置】rag 层的核心编排者，是"专业咨询"通道的大脑。
    上游：services 层调用 stream_answer(query, subject, history, trace)；
    下游依赖（均通过构造函数注入，便于替换/测试）：
        · intent —— 意图分类器（关键词规则 + TF-IDF/LR），判"通用知识 / 专业咨询"；
        · index  —— MilvusVectorIndex，bge-m3 稠密向量检索（知识库规模小，直接向量召回）；
        · llm    —— 大模型调用对象，负责最终生成；
        · prompts —— 提示词库。

【数据流向（一次专业问答）】
    query -> intent.predict 分类 ->（专业）route_domain 定位知识域
          -> index.search 向量召回 -> 拼成带 [1][2] 编号的 context
          -> 填入 rag_prompt（含 history/phone）-> _stream 逐 token 吐给前端
          -> 同时把引用来源写进 trace.sources，接口层据此返回 sources[]。

llm 参数约定为 callable(prompt) -> Iterator[str] | str | None，三种形态都能适配。
"""

from __future__ import annotations

import time
from typing import Iterable, Iterator

from app.conf.logging import logger
from app.conf.settings import settings
from app.rag.prompts import PromptLibrary
from app.rag.intent import is_domain_query, route_domain
from app.rag.types import RagTrace, SourceRef


class RagPipeline:
    """检索增强生成管道（依赖注入：index/llm/intent 从外部传入，单测可注入假对象）。"""

    def __init__(self, index, llm, intent, prompts: PromptLibrary | None = None) -> None:
        self.index = index
        self.llm = llm
        self.intent = intent
        self.prompts = prompts or PromptLibrary()
        # 主提示词模板只取一次缓存复用，避免每次问答都重新构造
        self.rag_prompt = self.prompts.rag_prompt()
        logger.info("RAG 管道初始化完成")

    # ------------------------------------------------------------------ #
    # 模型调用：兼容"返回字符串"和"逐 token 生成器"两种实现
    # ------------------------------------------------------------------ #
    def _iter_llm(self, prompt: str) -> Iterator[str]:
        """把 llm 的返回形态统一成"逐 token 迭代器"。

        llm(prompt) 可能返回：None（失败）、str（非流式整段）、Iterator[str]（流式）。
        """
        result = self.llm(prompt)
        if result is None:
            return
        if isinstance(result, str):
            if result:
                yield result
            return
        for token in result:
            if token:
                yield token

    def _stream(self, prompt: str, retries: int = 3) -> Iterator[str]:
        """流式：逐 token 输出；已吐出内容后不再重试，避免前端收到重复答案。

        【流式重试的关键约束】token 是边生成边发给前端的。一旦已经吐出过内容
        （produced=True），哪怕中途异常也绝不重试——否则前端会先收到半截答案、
        又收到一份从头开始的答案，造成重复/拼接错乱。只在"一个字都没吐"时才允许重试。
        """
        for attempt in range(1, retries + 1):
            produced = False
            try:
                for token in self._iter_llm(prompt):
                    produced = True
                    yield token
                if produced:
                    return
                logger.warning("大模型返回空内容（第 %s/%s 次）", attempt, retries)
            except Exception as exc:
                logger.error("大模型流式调用失败（第 %s/%s 次）：%s", attempt, retries, exc)
                if produced:
                    return
            if attempt < retries:
                time.sleep(1)

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, subject: str | None = None) -> list:
        """bge-m3 单向量检索：按知识域过滤，召回 top-k 分块。

        知识库规模小（几十份文档、几百个分块），直接向量召回即可，
        不需要混合检索/重排这类大规模场景的组件。
        检索异常只记日志并返回空列表：宁可走兜底话术，也不让异常打断问答。
        """
        started = time.time()
        try:
            chunks = self.index.search(query, subject_filter=subject)
        except Exception as exc:
            logger.error("向量检索失败：%s", exc)
            return []
        selected = chunks[:settings.RETRIEVAL_K]
        logger.info("检索完成：域=%s，入选 %s 个分块，耗时 %.2fs",
                    subject or "不限", len(selected), time.time() - started)
        return selected

    # ------------------------------------------------------------------ #
    # 端到端
    # ------------------------------------------------------------------ #
    @staticmethod
    def _format_history(history: Iterable | None) -> str:
        """把多轮历史拼成"用户：...\n助手：..."纯文本，填进提示词的 {history}。

        兼容两种行格式：dict（{question, answer}）或序列（[q, a]）；空历史返回 ""。
        """
        if not history:
            return ""
        lines = []
        for row in history:
            if isinstance(row, dict):
                q, a = row.get("question", ""), row.get("answer", "")
            else:
                q, a = (row[0], row[1]) if len(row) >= 2 else ("", "")
            if q or a:
                lines.append(f"用户：{q}\n助手：{a}")
        return "\n\n".join(lines)

    def stream_answer(self, query: str, subject: str | None = None,
                      history: Iterable | None = None,
                      trace: RagTrace | None = None) -> Iterator[str]:
        """从 query 到答案的完整流程（流式）。

        trace 用于回传"本次回答引用了哪些片段"，接口层据此返回 sources[]。
        任何异常都被兜住并 yield 一句"联系人工客服"的话术，保证用户永远有响应。
        """
        started = time.time()
        history_text = self._format_history(history)
        phone = settings.CUSTOMER_SERVICE_PHONE

        try:
            category = self.intent.predict(query)
            # 规则兜底：分类器准确率有限，两道规则纠偏防止企业问题被凭空作答：
            # 1) subject 非空：用户显式选了知识域，意图明确，强制"专业咨询"；
            # 2) 判为通用但命中领域关键词（vpn/报销/考勤…）：强制改回"专业咨询"。
            if subject:
                category = "专业咨询"
            elif category == "通用知识" and is_domain_query(query):
                logger.info("规则兜底：命中知识域关键词，强制走检索（分类器判为通用知识）")
                category = "专业咨询"
            logger.info("意图识别：%s（query=%s）", category, query)
            if trace is not None:
                trace.intent = category

            if category == "通用知识":
                # 通用知识：不检索，context 留空，直接让大模型用自身知识作答
                prompt = self.rag_prompt.format(context="", question=query, history=history_text, phone=phone)
            else:
                # 知识域路由：用户显式选的域优先，否则按关键词规则路由（可能为 None=不限域）
                domain = subject or route_domain(query)
                chunks = self.retrieve(query, subject=domain)
                # 给每个分块加 [1][2]… 编号：与 trace.sources 的 index 同源（同一个 i+1），
                # 保证"答案正文里引用的 [2]"与"前端来源卡片的第 2 条"严格对应。
                context = "\n\n".join(f"[{i + 1}] {doc.page_content}" for i, doc in enumerate(chunks))
                logger.info("上下文拼装完成：%s 个分块，%s 字符", len(chunks), len(context))
                if trace is not None:
                    trace.sources = [SourceRef.from_document(i + 1, doc) for i, doc in enumerate(chunks)]
                prompt = self.rag_prompt.format(context=context, question=query,
                                                history=history_text, phone=phone)

            produced = False
            for token in self._stream(prompt):
                produced = True
                yield token
            # 一个字都没吐出来（模型彻底失败）=> 给人工客服兜底话术，而不是空白响应
            if not produced:
                yield f"抱歉，服务暂时不可用，请稍后重试或联系人工客服 {phone}。"
            logger.info("回答生成完成，耗时 %.2fs", time.time() - started)
        except Exception as exc:
            logger.error("问答流程异常：%s", exc, exc_info=True)
            yield f"抱歉，处理您的问题时出现异常，请稍后重试或联系人工客服 {phone}。"