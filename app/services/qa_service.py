"""问答编排层：把 FAQ 与 RAG 两条链路组装成一个对外的流式问答服务。

架构位置：请求链路的中枢。路由层把问题交给这里的 QaService，由它决定走哪条通道：
    浏览器 → FastAPI 路由 → QaService.stream_answer
        ├─ FAQ 通道(FaqRetriever)：标准问句向量命中，秒回 MySQL 标准答案；
        └─ RAG 通道(RagPipeline)：知识域路由 + bge-m3 向量检索 + 大模型流式生成。
编排层的三大职责：① 双通道分流（FAQ 优先，未命中转 RAG）；
② 容错降级（FAQ 抛异常不阻断，自动转 RAG）；③ 会话落库时机（FAQ 立即写、RAG 收完再写）。

调用方（Web / CLI）只需要：
    service = QaService.build()
    for token, done in service.stream_answer(query, session_id=sid, subject="it"):
        ...
"""

from __future__ import annotations

import time
import uuid
from typing import Iterator

from app.conf.logging import logger
from app.conf.settings import settings
from app.infra.llm import ChatModel
from app.rag.types import RagTrace
from app.infra.mysql import MySqlRepo
from app.infra.vector_index import MilvusVectorIndex
from app.rag.intent import IntentClassifier
from app.rag.pipeline import RagPipeline
from app.rag.prompts import PromptLibrary
from app.services.faq import FaqRetriever


class QaService:
    """FAQ + RAG 双通道问答服务。

    通过构造函数接收三个协作者（faq 检索器、pipeline 检索生成链、repo 数据库），
    而不是在内部自己 new——依赖注入：测试时可传入假对象，无需真连外部服务。
    """

    def __init__(self, faq: FaqRetriever, pipeline: RagPipeline, repo: MySqlRepo) -> None:
        self.faq = faq
        self.pipeline = pipeline
        self.repo = repo
        self.settings = settings

    # ------------------------------------------------------------------ #
    # 装配
    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls) -> "QaService":
        """生产环境装配：MySQL + Milvus（知识库/FAQ 两个集合）+ bge-m3 + 大模型。

        连库、加载嵌入模型较重，由路由层懒加载只调一次（见 routes.get_service）。
        FAQ 集合复用知识库索引已加载的 bge-m3（embedding_model 注入），避免 2.2G 模型加载两遍。
        """
        started = time.time()
        repo = MySqlRepo()              # MySQL：会话历史 + FAQ 标准答案
        kb_index = MilvusVectorIndex()  # Milvus：知识库分块向量（RAG 用）
        faq_index = MilvusVectorIndex(
            milvus_collection_name=settings.MILVUS_FAQ_COLLECTION_NAME,
            embedding_model=kb_index.embedding_model,
        )                               # Milvus：80 条 FAQ 标准问句向量
        chat = ChatModel()              # 大模型客户端（OpenAI 兼容接口）
        pipeline = RagPipeline(
            index=kb_index,
            llm=chat,
            intent=IntentClassifier(),
            prompts=PromptLibrary(),
        )
        faq = FaqRetriever(index=faq_index, repo=repo)
        service = cls(faq=faq, pipeline=pipeline, repo=repo)
        service.ensure_schema()
        logger.info("问答服务装配完成，耗时 %.1fs", time.time() - started)
        return service

    def ensure_schema(self) -> None:
        """确保会话表存在（首次部署/空库时自动建表，免去手动迁移）。"""
        self.repo.ensure_chat_table()

    # ------------------------------------------------------------------ #
    # 会话
    # ------------------------------------------------------------------ #
    @staticmethod
    def new_session_id() -> str:
        """生成一个新的会话 ID（UUID4，全局唯一，无需查库）。"""
        return str(uuid.uuid4())

    def recent_turns(self, session_id: str) -> list[dict]:
        """取某会话最近若干轮历史；无 session_id 时返回空列表（首轮提问没有上下文）。"""
        return self.repo.recent_turns(session_id) if session_id else []

    def clear_session(self, session_id: str) -> bool:
        """清空会话历史：软删除（标记而非物理删除），保留数据可追溯，返回是否成功。"""
        return self.repo.soft_delete_turns(session_id)

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def stream_answer(self, query: str, session_id: str | None = None,
                      subject: str | None = None,
                      trace=None) -> Iterator[tuple[str, bool]]:
        """流式问答：产出 (token, 是否结束)；trace 会带上本次引用的片段。

        每次产出二元组 (token, is_complete)：token 是要显示的文字片段，is_complete 标记收尾。
        完整流程：
            1) 取历史上下文（供 RAG 多轮参考）；
            2) FAQ 优先匹配；若 FAQ 抛异常则降级（answer=None, need_rag=True）转 RAG；
            3) FAQ 命中：立即写库 → 一次性 yield (答案, True) → return（短路，不再走 RAG）；
            4) FAQ 判定无需 RAG 又无答案：回兜底话术；
            5) 需要 RAG：逐块 yield (片段, False)，收完后拼成整段写库，最后 yield ("", True)。
        """
        started = time.time()
        history = self.recent_turns(session_id) if session_id else []

        try:
            answer, need_rag = self.faq.match(query, trace=trace)
        except Exception as exc:
            # 容错降级：FAQ 依赖 Milvus/MySQL，若它们抖动抛异常，不能让整次问答失败；
            # 吞掉异常、记日志，强制 need_rag=True，把问题交给 RAG 通道兜底。
            logger.error("FAQ 模块异常，降级到 RAG：%s", exc)
            answer, need_rag = None, True

        if answer:
            # 会话写入时机（FAQ）：答案此刻已完整拿到，立即落库这一轮问答
            if session_id:
                self.repo.insert_turn(session_id, query, answer)
            logger.info("FAQ 命中，耗时 %.2fs", time.time() - started)
            # FAQ 是"一次性"的：token=完整答案且 is_complete=True 同时给出。
            # 路由层必须"先发 token 再发 end"，否则前端会丢答案。
            yield answer, True
            return

        logger.info("FAQ 未命中，转入 RAG 链路")

        # FAQ 判定"无需 RAG"（如非法/空 query）：没有可检索的内容，直接回兜底话术，不空跑 RAG
        if not need_rag:
            yield (f"暂未检索到与「{query}」匹配的内容，建议换个问法，或联系人工客服 "
                   f"{settings.CUSTOMER_SERVICE_PHONE}。"), True
            return

        # collected 累积所有片段，用于流结束后拼回完整答案（落库需要整段，而非碎片）
        collected = []
        for token in self.pipeline.stream_answer(query, subject=subject, history=history, trace=trace):
            collected.append(token)
            yield token, False

        # 会话写入时机（RAG）：与 FAQ"立即写"不同，必须等整段收完再写，
        # 因为生成过程中答案还不完整；"".join 拼回全文，strip 去首尾空白。
        full_answer = "".join(collected).strip()
        # 双重守卫：有会话且答案非空才落库，避免把空回答写进历史污染后续上下文
        if session_id and full_answer:
            self.repo.insert_turn(session_id, query, full_answer)

        logger.info("RAG 回答完成，耗时 %.2fs", time.time() - started)
        # 收尾：空 token + is_complete=True，路由层因 token 为空只发 end 帧
        yield "", True