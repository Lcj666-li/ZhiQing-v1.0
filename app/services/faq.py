"""FAQ 快速通道：80 条标准问句向量化后做相似度命中，命中即秒回标准答案。

架构位置：双通道中的"FAQ 快速通道"。QaService 先调本模块 match()，
命中就直接返回 MySQL 里的标准答案，未命中才转 RAG。

【实现方式】入库时（scripts/init_faq_store.py）把 80 条标准问句用 bge-m3
    向量化写入独立的 Milvus 集合 zhiqing_faq；在线时用同一个 embedding 模型
    编码用户问题，取 Top1 相似度：
        · 相似度 >= faq_min_score（默认 0.85，bge-m3 已归一化，内积=余弦）：
          认为用户问的就是这条标准问句（换个说法也能命中），
          用命中的标准问句去 MySQL 取答案；
        · 低于阈值：不硬套近似答案，转 RAG 通道。
    阈值取 0.85 是因为"同一问句的不同问法"余弦通常在 0.9 以上，
    而"相关但不同"的问题多在 0.6~0.8，0.85 能干净地分开两者。

【容错】向量检索/取答案的任何异常由上层 QaService 捕获并降级转 RAG，
    FAQ 通道故障不会让整次问答失败。
"""

from __future__ import annotations

from app.conf.logging import logger
from app.conf.settings import settings
from app.rag.types import SNIPPET_LEN, TITLE_LEN, SourceRef


class FaqRetriever:
    """FAQ 向量检索器（依赖注入：index=FAQ 向量集合，repo=MySQL 答案源）。"""

    def __init__(self, index, repo, min_score: float | None = None) -> None:
        self.index = index
        self.repo = repo
        self.min_score = settings.FAQ_MIN_SCORE if min_score is None else min_score

    def match(self, query: str, trace=None) -> tuple[str | None, bool]:
        """返回 (答案, 是否需要转 RAG)；答案是 None 表示没命中。

        非法/空 query 返回 (None, False)：没有可检索的内容，上层直接回兜底话术。
        """
        if not isinstance(query, str) or not query.strip():
            logger.warning("非法 query：%r", query)
            return None, False

        # Top1 向量命中：k=1 只取最相似的一条标准问句
        docs, scores = self.index.search(query, k=1, min_score=self.min_score, with_scores=True)
        if not docs:
            return None, True

        matched = docs[0].page_content
        logger.info("FAQ 候选：%s（相似度 %.3f，阈值 %.2f）", matched, scores[0], self.min_score)

        # 用命中的"标准问句"作键去 MySQL 取答案（不同问法共享同一条标准答案）
        answer = self.repo.answer_of(matched)
        if answer:
            if trace is not None:
                # 记录引用来源供前端展示"出处"：FAQ 命中统一标 source_type=faq，
                # 与 RAG 的 doc 来源共用 SourceRef 结构，前端只渲染一种卡片。
                trace.sources = [SourceRef(
                    index=1,
                    title=matched[:TITLE_LEN],
                    subject="FAQ",
                    chunk_id=f"faq:{matched[:32]}",
                    snippet=answer[:SNIPPET_LEN],
                    source_type="faq",
                )]
            return answer, False

        logger.error("问题已匹配但库里没有答案：%s", matched)
        return None, True