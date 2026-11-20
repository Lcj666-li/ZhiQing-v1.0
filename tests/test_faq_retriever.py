"""FAQ 向量通道单测：阈值命中 / 低于阈值转 RAG / 非法输入 / 缺答案降级。

四种结局：命中返回标准答案并写 source_type=faq 的来源引用；
低于阈值 (None, True) 转 RAG；空/纯空白 query (None, False) 且不发起检索；
问句命中但缺答案 (None, True) 降级转 RAG。
FakeIndex 的 min_score 过滤与真实索引一致，AnswerRepo 为最小假仓储，
阈值分支无需 Milvus/MySQL 即可精确覆盖。
"""

from __future__ import annotations

from app.rag.types import RagTrace
from app.services.faq import FaqRetriever
from tests.fakes import FakeDoc, FakeIndex


class AnswerRepo:
    """最小 FAQ 假仓储：标准问句 -> 标准答案 的映射，只提供 answer_of。"""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def answer_of(self, question: str):
        return self.mapping.get(question)


QUESTION = "VPN 连不上怎么办"
ANSWER = "请先确认域账号与密码，再重新连接。"


def test_faq_hit_returns_answer_and_source():
    """命中：相似度 0.95 >= 阈值 0.85，返回标准答案、need_rag=False，
    且 trace.sources[0] 标 source_type=faq / subject=FAQ（前端据此渲染来源卡片）。"""
    index = FakeIndex(docs=[FakeDoc(QUESTION)], scores=[0.95])
    retriever = FaqRetriever(index=index, repo=AnswerRepo({QUESTION: ANSWER}))
    trace = RagTrace()
    answer, need_rag = retriever.match(QUESTION, trace=trace)
    assert answer == ANSWER
    assert need_rag is False
    assert trace.sources and trace.sources[0].source_type == "faq"
    assert trace.sources[0].subject == "FAQ"


def test_faq_below_threshold_falls_to_rag():
    """低于阈值：相似度 0.5 < 0.85，被 min_score 过滤掉，(None, True) 转 RAG。"""
    index = FakeIndex(docs=[FakeDoc(QUESTION)], scores=[0.5])
    retriever = FaqRetriever(index=index, repo=AnswerRepo({QUESTION: ANSWER}))
    answer, need_rag = retriever.match(QUESTION)
    assert answer is None
    assert need_rag is True


def test_faq_invalid_query():
    """非法输入：空串/纯空白 query 返回 (None, False)，且不发起任何检索。"""
    index = FakeIndex(docs=[FakeDoc(QUESTION)], scores=[0.95])
    retriever = FaqRetriever(index=index, repo=AnswerRepo({QUESTION: ANSWER}))
    assert retriever.match("") == (None, False)
    assert retriever.match("   ") == (None, False)
    assert index.calls == []


def test_faq_missing_answer_degrades():
    """缺答案：问句命中（0.95）但库里没有这条问句的答案，(None, True) 降级转 RAG。"""
    index = FakeIndex(docs=[FakeDoc(QUESTION)], scores=[0.95])
    retriever = FaqRetriever(index=index, repo=AnswerRepo({}))  # 空映射 => answer_of 返回 None
    answer, need_rag = retriever.match(QUESTION)
    assert answer is None
    assert need_rag is True