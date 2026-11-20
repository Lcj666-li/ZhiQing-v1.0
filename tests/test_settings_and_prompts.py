"""配置与提示词的基础单测（不依赖任何外部服务）。

【覆盖场景总览】
    两类"低成本高价值"的冒烟测试：
    1. settings 配置合理性：应用名/关键目录存在/检索参数的大小关系/
       知识域列表形态。配置错误往往在运行很久后才暴露（比如目录写错
       导致入库为空），这类测试把问题拦在最早期；
    2. 提示词模板可渲染：rag_prompt 的变量声明与模板正文一致、
       format 后变量值真的出现在文本里。模板里少写/多写一个 {变量}
       都会让线上问答直接抛 KeyError，必须有用例守着。

【为什么参数要断言"大小关系"而不是具体数值？】
    具体数值（如分块 500 字）允许按业务调整，但"chunk_size > overlap >= 0"、
    "0 < min_score < faq_min_score <= 1"是不变量（分块与双通道阈值的
    结构约束）。测不变量而非魔法数字，测试才不会因正常调参而误报。
"""

from __future__ import annotations

import os

from app.conf.settings import settings
from app.rag.prompts import PromptLibrary


def test_settings_paths_exist():
    """基础配置冒烟：应用名正确、项目根/模型/数据三个关键目录真实存在、
    API 前缀符合 /api/ 规范（目录缺失说明部署环境没准备好）。"""
    assert settings.APP_NAME == "知擎"
    assert os.path.isdir(settings.PROJECT_ROOT)
    assert os.path.isdir(settings.MODELS_DIR)
    assert os.path.isdir(settings.DATA_DIR)
    assert settings.API_PREFIX.startswith("/api/")


def test_settings_retrieval_and_sources():
    """检索参数不变量 + 知识域形态：
    chunk_size > overlap >= 0（重叠不能超过块大小）、retrieval_k > 0、
    0 < min_score < faq_min_score <= 1（FAQ 阈值比 RAG 更严，二者都是余弦相似度）；
    VALID_SOURCES 是非空 list 且每项为非空字符串（路由层直接序列化给前端下拉框）。
    不硬编码具体域名——知识域随业务配置变化（当前 7 个域），测形态不测魔法值。"""
    assert settings.CHUNK_SIZE > settings.CHUNK_OVERLAP >= 0
    assert settings.RETRIEVAL_K > 0
    assert 0 < settings.MIN_SCORE < settings.FAQ_MIN_SCORE <= 1
    assert isinstance(settings.VALID_SOURCES, list)
    assert settings.VALID_SOURCES
    assert all(isinstance(s, str) and s.strip() for s in settings.VALID_SOURCES)


def test_rag_prompt_renders_all_variables():
    """主作答模板：声明的输入变量恰好是 {context, question, history, phone}
    四个（不多不少），且 format 后四个值都真实出现在渲染文本里——
    保证管道拼装上下文/历史/客服电话时不会遇到缺变量或漏渲染。"""
    prompt = PromptLibrary.rag_prompt()
    assert set(prompt.input_variables) == {"context", "question", "history", "phone"}
    rendered = prompt.format(context="上下文A", question="问题B", history="历史C", phone="10086")
    for token in ("上下文A", "问题B", "历史C", "10086"):
        assert token in rendered