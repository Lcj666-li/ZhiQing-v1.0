"""初始化 FAQ 知识库：MySQL 建表导入 CSV + 标准问句向量化入 Milvus（幂等）。

用法：
    python -m scripts.init_faq_store
    python -m scripts.init_faq_store --csv data/faq/faq_seed.csv
    python -m scripts.init_faq_store --reset      # 换整套 FAQ 语料时用

MySQL 是答案的权威数据源（CSV 列：学科名称/问题/答案）；
Milvus 集合 zhiqing_faq 只存标准问句向量，在线命中后回 MySQL 取答案。

【幂等性（两处）】
    · MySQL：question 唯一索引 + INSERT IGNORE，重复执行不产生重复行；
    · Milvus：主键取问句内容 md5 + upsert，重复入库是覆盖而非新增。

【--reset】MySQL 表与向量集合一起重置，保证"向量里的问句"与"表里的答案"始终一一对应。
"""

from __future__ import annotations

import argparse
import sys

from langchain_core.documents import Document

from app.conf.logging import logger
from app.conf.settings import settings
from app.infra.mysql import MySqlRepo
from app.infra.vector_index import MilvusVectorIndex


def main() -> int:
    parser = argparse.ArgumentParser(description="初始化 FAQ 知识库（MySQL + Milvus 向量）")
    parser.add_argument("--csv", default=settings.FAQ_CSV, help="FAQ CSV 路径（列：知识域/问题/答案）")
    parser.add_argument("--reset", action="store_true", help="先清空 FAQ 表与向量集合再导入")
    args = parser.parse_args()

    repo = MySqlRepo()
    try:
        faq_index = MilvusVectorIndex(milvus_collection_name=settings.MILVUS_FAQ_COLLECTION_NAME)

        if args.reset:
            repo.ensure_faq_table()
            repo.execute(f"TRUNCATE TABLE `{settings.FAQ_TABLE}`")
            repo.commit()
            logger.info("已清空 FAQ 表：%s", settings.FAQ_TABLE)
            faq_index.recreate_collection()
            logger.info("已重建 FAQ 向量集合：%s", settings.MILVUS_FAQ_COLLECTION_NAME)

        inserted, skipped = repo.import_faq_csv(args.csv)
        total = repo.count_faq()
        logger.info("MySQL 导入完成：新增 %s，跳过 %s，当前共 %s 条", inserted, skipped, total)

        questions = repo.all_questions()
        docs = [Document(page_content=q, metadata={"subject": "FAQ", "source": "faq_seed.csv"})
                for q in questions]
        faq_index.upsert_chunks(docs)

        print(f"OK 新增 {inserted} 条，跳过 {skipped} 条，库中共 {total} 条；"
              f"FAQ 向量集合 {settings.MILVUS_FAQ_COLLECTION_NAME} 已写入 {len(docs)} 条问句")
        return 0
    except Exception as exc:
        logger.error("FAQ 初始化失败：%s", exc, exc_info=True)
        print(f"FAILED: {exc}")
        return 1
    finally:
        repo.close()


if __name__ == "__main__":
    sys.exit(main())