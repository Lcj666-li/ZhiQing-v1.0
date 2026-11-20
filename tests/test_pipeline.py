"""RAG 管道单测：分支、上下文/历史拼装、规则路由、流式与异常兜底。

覆盖：通用知识分支跳过检索；专业咨询分支检索内容与域过滤参数进入 prompt；
多轮历史拼进 prompt；分类器误判时关键词规则强制纠偏路由；
模型故障重试后输出兜底话术；历史格式化的边界行为。
意图/索引/LLM 全用替身，只有 PromptLibrary 用真实实现（纯逻辑，值得真测）；
通过 FakeLLM.prompts 断言模型实际收到的输入——不测模型输出（不可控），测输入。
"""

from __future__ import annotations

from app.rag.pipeline import RagPipeline
from app.rag.prompts import PromptLibrary
from tests.fakes import FakeIndex, FakeIntent, FakeLLM


def build(llm, *, category="专业咨询", index=None):
    """装配全替身管道。默认"专业咨询"覆盖最常见路径，各测试按需覆写单个变量。"""
    return RagPipeline(
        index=index or FakeIndex(),
        llm=llm,
        intent=FakeIntent(category),
        prompts=PromptLibrary(),
    )


def test_general_question_skips_retrieval():
    """通用知识分支：意图判为"通用知识"且未命中领域词时不查向量库。

    index.calls 为空证明一次检索都没发起。
    """
    index = FakeIndex()
    llm = FakeLLM("stream", "通用知识回答")
    pipeline = build(llm, category="通用知识", index=index)
    out = "".join(pipeline.stream_answer("5 乘 9 等于多少"))
    assert out == "通用知识回答"
    assert index.calls == []


def test_professional_question_uses_context(fake_index):
    """专业咨询分支：检索被调用、学科过滤传对、检索内容进了 prompt。

    calls[0][1] == "it" 验证 subject 过滤透传到索引层；
    检索分块内容出现在 llm.prompts[0]，验证"检索增强"落实到了模型输入上。
    """
    llm = FakeLLM("string", "专业回答")
    pipeline = build(llm, index=fake_index)
    out = "".join(pipeline.stream_answer("VPN 连不上怎么办", subject="it"))
    assert out == "专业回答"
    assert fake_index.calls and fake_index.calls[0][1] == "it"
    assert "VPN 报错 691 表示账号或密码错误" in llm.prompts[0]


def test_history_is_injected(fake_index):
    """多轮对话：上一轮的问题与答案都必须出现在 prompt 里，模型才能理解指代。"""
    llm = FakeLLM("string", "带历史的回答")
    pipeline = build(llm, index=fake_index)
    history = [{"question": "上一轮问题", "answer": "上一轮答案"}]
    list(pipeline.stream_answer("这一轮问题", subject="it", history=history))
    prompt = llm.prompts[0]
    assert "上一轮问题" in prompt and "上一轮答案" in prompt


def test_rule_fallback_routes_domain(fake_index):
    """规则兜底路由：分类器判为"通用知识"，但 query 命中领域关键词（vpn），
    管道强制改回"专业咨询"并路由到 it 域（两道纠偏规则之一）。
    """
    llm = FakeLLM("string", "路由回答")
    pipeline = build(llm, category="通用知识", index=fake_index)
    out = "".join(pipeline.stream_answer("VPN 连不上怎么办"))
    assert out == "路由回答"
    assert fake_index.calls and fake_index.calls[0][1] == "it"


def test_llm_failure_falls_back(fake_index):
    """模型故障兜底：FakeLLM("raise") 每次都抛异常，管道重试 3 次后
    应输出含"客服"或"重试"的安抚话术，而不是把异常抛给用户。

    重试间隔 sleep(1)，本用例约需 2 秒，是全套测试里最慢的一个。
    """
    pipeline = build(FakeLLM("raise"), index=fake_index)
    out = "".join(pipeline.stream_answer("知擎怎么用", subject="it"))
    assert "客服" in out or "重试" in out


def test_empty_history_formats_to_empty_string():
    """历史格式化静态方法的边界行为：None/空列表 -> 空串（prompt 里
    不出现多余内容）；(问,答) 元组 -> "用户：q\\n助手：a" 的标准格式。
    同时覆盖 dict 与元组两种历史行格式（对应 MySQL 行与内存替身）。
    """
    assert RagPipeline._format_history(None) == ""
    assert RagPipeline._format_history([]) == ""
    assert RagPipeline._format_history([("q", "a")]) == "用户：q\n助手：a"