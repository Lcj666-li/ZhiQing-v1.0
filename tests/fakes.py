"""测试替身：用最小假对象替代真实依赖，保证单测不需要 MySQL/Milvus/模型。

被测代码均通过构造函数注入接收依赖，测试时换成这里的假对象（接口签名一致、行为可控）。

【替身对照表（假对象 -> 真实依赖 -> 替代要点）】
    FakeDoc     -> langchain Document        只保留 page_content/metadata 两属性
    FakeIndex   -> MilvusVectorIndex         不连向量库，返回固定分块并记录调用参数
    FakeIntent  -> IntentClassifier          不加载分类模型，直接返回预设类别
    FakeLLM     -> ChatModel（infra/llm.py） 不调大模型，三种行为模式可切换
    FakeFaq     -> FaqRetriever              不查 Milvus/MySQL，返回预设命中结果
    FakeRepo    -> MySqlRepo（会话部分）      用内存字典存对话轮次
    FakeService -> QaService                 给 API 层测试用的整体替身
"""

from __future__ import annotations


class FakeDoc:
    """最小化文档对象，替代 langchain 的 Document（只需两属性）。"""

    def __init__(self, content: str, metadata: dict | None = None) -> None:
        self.page_content = content
        self.metadata = metadata or {}


class FakeIndex:
    """假向量库：记录检索参数，返回固定分块（可选带分数），替代 MilvusVectorIndex。

    calls 记录每次 search 的 (query, subject_filter)，供断言检索行为与过滤参数；
    scores 与 docs 一一对应，min_score 过滤逻辑与真实索引一致，FAQ 阈值分支也能用替身测。
    """

    def __init__(self, docs: list[FakeDoc] | None = None,
                 scores: list[float] | None = None) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.docs = docs if docs is not None else [FakeDoc("VPN 故障请先确认域账号与密码。")]
        self.scores = scores if scores is not None else [0.9] * len(self.docs)
        self.upserted: list = []

    def search(self, query: str, k=None, subject_filter: str | None = None,
               min_score=None, with_scores: bool = False):
        self.calls.append((query, subject_filter))
        pairs = list(zip(self.docs, self.scores))
        if min_score is not None:
            pairs = [(d, s) for d, s in pairs if s >= min_score]
        if k is not None:
            pairs = pairs[:k]
        docs = [d for d, _ in pairs]
        scores = [s for _, s in pairs]
        return (docs, scores) if with_scores else docs

    def upsert_chunks(self, chunks):
        self.upserted.append(chunks)


class FakeIntent:
    """假意图识别器：恒返回预设类别，精确控制管道走哪条分支。"""

    def __init__(self, category: str = "专业咨询") -> None:
        self.category = category

    def predict(self, query: str) -> str:
        return self.category


class FakeLLM:
    """替代 ChatModel（callable(prompt)），可切换三种行为：
    stream 逐字符生成 / string 整段返回 / raise 抛异常模拟模型故障。

    prompts 记录收到的每条 prompt——不测模型输出（不可控），测喂给模型的输入。
    """

    def __init__(self, mode: str = "stream", text: str = "这是知擎的回答。") -> None:
        self.mode = mode
        self.text = text
        self.prompts: list[str] = []

    def __call__(self, prompt: str):
        self.prompts.append(prompt)
        if self.mode == "raise":
            raise RuntimeError("模拟模型故障")
        if self.mode == "string":
            return self.text

        def gen():
            for ch in self.text:
                yield ch

        return gen()


class FakeFaq:
    """假 FAQ 检索器：预设 answer/need_rag 模拟三种结局。

    命中（answer 非空）、未命中转 RAG（None, True）、非法输入（None, False）。
    命中且传了 trace 时填充一条来源引用，模拟真实检索器回传 sources 的行为。
    """

    def __init__(self, answer: str | None = None, need_rag: bool = True) -> None:
        self.answer = answer
        self.need_rag = need_rag

    def match(self, query: str, trace=None):
        if self.answer and trace is not None:
            from app.rag.types import SourceRef
            trace.sources = [SourceRef(index=1, title="FAQ 标准问句", subject="FAQ",
                                       chunk_id="faq:demo", snippet=self.answer[:80], source_type="faq")]
        return self.answer, self.need_rag


class FakeRepo:
    """假会话仓储（内存字典），提供与 MySqlRepo 同名的四个会话方法。

    soft_delete_turns 在替身里是直接 pop，真实实现是软删除——
    对上层行为（历史清空）而言等价，测试只关心可观察结果。
    """

    def __init__(self) -> None:
        self.turns: dict[str, list[dict]] = {}

    def recent_turns(self, session_id: str, limit: int = 5):
        return self.turns.get(session_id, [])[-limit:]

    def insert_turn(self, session_id: str, question: str, answer: str) -> None:
        self.turns.setdefault(session_id, []).append({"question": question, "answer": answer})

    def soft_delete_turns(self, session_id: str) -> bool:
        self.turns.pop(session_id, None)
        return True

    def ensure_chat_table(self) -> None:
        return None


class FakeService:
    """假问答服务：替代 QaService，给 API 层单测用。

    stream_answer 产出 (文本片段, 是否结束) 二元组，与真实服务约定一致：
    query 含 "FAQ" 时答案与结束标记同帧产出，专门验证路由层"先发 token 再发 end"；
    其它 query 分三片流式输出后发 ("", True) 哨兵。两条路径都写入 FakeRepo 供历史断言。
    """

    def __init__(self, faq_answer: str | None = None) -> None:
        self.faq = FakeFaq(answer=faq_answer, need_rag=faq_answer is None)
        self.repo = FakeRepo()

    def recent_turns(self, session_id: str):
        return self.repo.recent_turns(session_id)

    def clear_session(self, session_id: str) -> bool:
        return self.repo.soft_delete_turns(session_id)

    def stream_answer(self, query: str, session_id: str | None = None, subject: str | None = None,
                      trace=None):
        if "FAQ" in query:
            self.repo.insert_turn(session_id or "s", query, "FAQ 固定答案")
            yield "FAQ 固定答案", True
            return
        for piece in ("这是", "流式", "回答"):
            yield piece, False
        self.repo.insert_turn(session_id or "s", query, "这是流式回答")
        yield "", True