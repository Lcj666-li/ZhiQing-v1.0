"""pytest 公共配置：把项目根目录加入 sys.path，并提供假文档/假向量库 fixture。

tests 为离线单测：不需要 MySQL/Milvus/大模型。其余测试替身见 tests/fakes.py。
"""

from __future__ import annotations

import os
import sys

import pytest

# 插入项目根目录，保证任意目录下运行 pytest 都能 `import app...`
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.fakes import FakeDoc, FakeIndex  # noqa: E402


@pytest.fixture
def fake_docs() -> list[FakeDoc]:
    """两条带元数据的假分块文档，模拟检索命中后返回给 LLM 的上下文片段。"""
    return [
        FakeDoc("VPN 报错 691 表示账号或密码错误，请确认域账号后重连。",
                {"subject": "it", "chunk_id": "doc_0_chunk_0", "source": "网络与VPN故障排查手册.md"}),
        FakeDoc("员工无线网使用 ZQ-Office，域账号认证；访客用 ZQ-Guest。",
                {"subject": "it", "chunk_id": "doc_0_chunk_1", "source": "网络与VPN故障排查手册.md"}),
    ]


@pytest.fixture
def fake_index(fake_docs) -> FakeIndex:
    """预装好假文档的假向量库。"""
    return FakeIndex(fake_docs)