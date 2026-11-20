"""FAQ 命令行自测：不启动 Web，直接验证标准问句的向量命中与阈值判定。

用法：
    python -m scripts.faq_cli
    python -m scripts.faq_cli --threshold 0.9

只装配 MySQL + FAQ 向量集合，不加载知识库集合和大模型，
适合调整 FAQ 语料或阈值后快速验证命中效果（通道逻辑见 app/services/faq.py）。
需要 MySQL 与 Milvus 在运行，且已执行 scripts/init_faq_store.py 完成导入。
"""

from __future__ import annotations

import argparse

from app.conf.settings import settings
from app.infra.mysql import MySqlRepo
from app.infra.vector_index import MilvusVectorIndex
from app.services.faq import FaqRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="FAQ 检索自测")
    parser.add_argument("--threshold", type=float, default=settings.FAQ_MIN_SCORE,
                        help="命中阈值（余弦相似度，0~1）")
    args = parser.parse_args()

    repo = MySqlRepo()
    faq_index = MilvusVectorIndex(milvus_collection_name=settings.MILVUS_FAQ_COLLECTION_NAME)
    faq = FaqRetriever(index=faq_index, repo=repo, min_score=args.threshold)
    print(f"{settings.APP_NAME} FAQ 自测 | 阈值 {args.threshold} | 输入 exit 退出")

    try:
        while True:
            try:
                query = input("\n问题：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n退出")
                return 0
            if not query or query.lower() in {"exit", "quit"}:
                return 0

            answer, need_rag = faq.match(query)
            if answer:
                print(f"[命中 FAQ]\n{answer}")
            else:
                print("[未命中] 需要转 RAG" if need_rag else "[非法输入]")
    finally:
        repo.close()


if __name__ == "__main__":
    raise SystemExit(main())