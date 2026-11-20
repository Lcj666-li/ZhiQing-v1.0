"""检索来源（citations）：把"答案引用了哪些片段"结构化，供接口返回与前端展示。

【架构位置】rag 层的数据结构定义。RagPipeline 拼装上下文时用
    SourceRef.from_document(i+1, doc) 生成来源卡片，写进 RagTrace.sources，
    接口层再把 trace.as_dict() 序列化进响应的 sources[] 字段。

设计要点：
1. index 与提示词里的 [1][2] 编号严格一致：context 编号和 SourceRef.index
   都取自同一个 enumerate 的 i+1，同源生成、天然对齐；
2. 只回传轻量字段（标题/知识域/片段摘要/原文文件名），不把整段原文塞给前端；
3. doc 与 faq 两种来源共用一套结构，前端只渲染一种卡片；
4. source 字段记录原文文件名，配合 chunk_id 可回溯到入库前的原始文档。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 来源卡片的长度上限：片段摘要 160 字、标题 40 字。
SNIPPET_LEN = 160
TITLE_LEN = 40


def first_line_title(content: str) -> str:
    """从片段内容里取一个人类可读的标题（md 取首行标题，其它取首行文本）。"""
    for line in (content or "").splitlines():
        text = line.strip().lstrip("#").strip()
        if text:
            return text[:TITLE_LEN]
    return "未命名片段"


def _clean_snippet(content: str, limit: int = SNIPPET_LEN) -> str:
    """把片段压成一行、去掉 markdown 符号，便于前端展示。"""
    text = " ".join(line.strip().lstrip("#").strip() for line in (content or "").splitlines())
    text = " ".join(text.split())
    return text[:limit]


@dataclass
class SourceRef:
    """一条来源引用（对应前端一张"来源卡片"）。

    字段：
        index：引用编号，与答案正文里的 [index] 一致；
        title：片段标题（首行文本，≤40 字）；
        subject：知识域；
        chunk_id：分块 id（doc_i_chunk_j），便于回溯定位原文；
        snippet：片段摘要（≤160 字，单行、去 markdown）；
        source_type：doc（文档检索）/ faq（标准问答）；
        source：原文文件名，答案引用可回溯到入库前的原始文档。
    """

    index: int
    title: str
    subject: str = ""
    chunk_id: str = ""
    snippet: str = ""
    source_type: str = "doc"  # doc / faq
    source: str = ""

    def as_dict(self) -> dict:
        """转成可 JSON 序列化的字典（type 字段名对外简化）。"""
        return {
            "index": self.index,
            "title": self.title,
            "subject": self.subject,
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
            "type": self.source_type,
            "source": self.source,
        }

    @classmethod
    def from_document(cls, index: int, doc) -> "SourceRef":
        """从 LangChain Document 工厂式构造一条来源引用。

        index 由 pipeline 传入，与上下文编号共用同一个 i+1，保证编号一致。
        """
        content = getattr(doc, "page_content", "") or ""
        meta = getattr(doc, "metadata", {}) or {}
        return cls(
            index=index,
            title=first_line_title(content),
            subject=str(meta.get("subject", "")),
            chunk_id=str(meta.get("chunk_id", "")),
            snippet=_clean_snippet(content),
            source_type="doc",
            source=str(meta.get("source", "")),
        )


@dataclass
class RagTrace:
    """一次问答的检索轨迹（只供接口层读取，不参与业务判断）。

    记录"这次回答用了什么意图、引用了哪些来源"，
    既方便前端展示来源，也便于调试/可观测。
    """

    sources: list[SourceRef] = field(default_factory=list)
    intent: str = ""

    def as_dict(self) -> dict:
        return {
            "sources": [s.as_dict() for s in self.sources],
            "intent": self.intent,
        }