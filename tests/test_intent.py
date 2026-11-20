"""意图路由单测：关键词规则的域路由（纯逻辑，不加载模型）。

覆盖典型问句路由到正确域（it/finance/admin），以及业务无关问句
一个域都不命中（None 且 is_domain_query=False）。
分类器本体依赖训练产物，规则层是在线兜底、无依赖即可稳定测试，故重点测它。
"""

from __future__ import annotations

import pytest

from app.rag.intent import is_domain_query, route_domain


@pytest.mark.parametrize("query,domain", [
    ("VPN 连不上怎么办", "it"),
    ("差旅报销的发票要求", "finance"),
    ("预订会议室", "admin"),
])
def test_route_domain_hits(query, domain):
    """典型问句路由到正确域，且 is_domain_query 判为 True（需要查库）。

    三个例子分别覆盖：英文缩写大小写（VPN->vpn）、多关键词计票
    （报销/发票/差旅三词都落在 finance）、易歧义词（"会议室"整体
    只在 admin 词表，避免被 oa 的"会议"类词抢走）。
    """
    assert route_domain(query) == domain
    assert is_domain_query(query) is True


def test_route_domain_miss():
    """与业务无关的问句：一个领域关键词都不命中，route 返回 None、
    is_domain_query=False（这类问题走"通用知识"分支，让大模型直接作答不查库）。"""
    query = "帮我写一首关于春天的诗"
    assert route_domain(query) is None
    assert is_domain_query(query) is False