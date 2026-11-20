"""入库单测：单层分块与元数据完整性（用 tmp_path，不碰真实知识库）。

覆盖正常目录（.md/.txt 切块，chunk_id/subject/source/timestamp 齐全，
subject 从目录名推断）与空目录（返回空列表而非报错）。
"""

from __future__ import annotations

from app.rag.ingest import build_chunks


def test_build_chunks_metadata(tmp_path):
    """正常入库：it_data 目录写一份 .md 与一份 .txt，切块后每块元数据齐全。

    断言 subject=="it"（目录名 it_data 去 _data）、source 是原文文件名、
    chunk_id 形如 doc_i_chunk_j——这些是答案引用回溯原文的依据。
    """
    kb_dir = tmp_path / "it_data"
    kb_dir.mkdir()
    (kb_dir / "网络手册.md").write_text("# VPN 排查\n\n报错 691 表示账号或密码错误。\n", encoding="utf-8")
    (kb_dir / "无线说明.txt").write_text("员工无线网使用 ZQ-Office，域账号认证。访客使用 ZQ-Guest。\n", encoding="utf-8")

    chunks = build_chunks(str(kb_dir))
    assert chunks
    for chunk in chunks:
        meta = chunk.metadata
        assert meta["subject"] == "it"
        assert meta["source"] in {"网络手册.md", "无线说明.txt"}
        assert meta["chunk_id"].startswith("doc_") and "_chunk_" in meta["chunk_id"]
        assert meta["timestamp"]


def test_build_chunks_empty_dir(tmp_path):
    """空目录：没有任何受支持文件，返回空列表而非抛异常。"""
    empty_dir = tmp_path / "oa_data"
    empty_dir.mkdir()
    assert build_chunks(str(empty_dir)) == []