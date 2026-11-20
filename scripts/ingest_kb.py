"""知识库入库：把 data/kb/{知识域}_data 下的文档切块并写入 Milvus。

用法：
    python -m scripts.ingest_kb
    python -m scripts.ingest_kb --subject it        # 只处理某个知识域
    python -m scripts.ingest_kb --dir data/kb       # 指定知识库根目录
    python -m scripts.ingest_kb --recreate          # 换知识库时删集合重建

【目录约定】子目录命名必须是 {域名}_data（如 it_data），
    ingest.py 靠目录名反推 subject 元数据，检索时用它做知识域过滤。

【幂等性】分块主键取内容 md5，upsert 语义保证重复入库是覆盖而非重复插入，可安全重跑。
"""

from __future__ import annotations

import argparse
import os
import sys

from app.conf.logging import logger
from app.conf.settings import settings
from app.infra.vector_index import MilvusVectorIndex
from app.rag.ingest import build_chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库入库")
    parser.add_argument("--dir", default=settings.KB_DIR, help="知识库根目录，默认 data/kb")
    parser.add_argument("--subject", default=None, help="只处理指定知识域，如 it")
    parser.add_argument("--recreate", action="store_true", help="先删除并重建集合（换知识库时用）")
    args = parser.parse_args()

    subjects = [args.subject] if args.subject else settings.VALID_SOURCES
    index = MilvusVectorIndex()
    if args.recreate:
        index.recreate_collection()
    total = 0

    # 单域失败不影响其它域（隔离故障域）
    for subject in subjects:
        directory = os.path.join(args.dir, f"{subject}_data")
        if not os.path.isdir(directory):
            logger.warning("目录不存在，跳过：%s", directory)
            continue

        logger.info("开始处理目录：%s", directory)
        try:
            chunks = build_chunks(
                directory,
                chunk_size=settings.CHUNK_SIZE,
                overlap_size=settings.CHUNK_OVERLAP,
            )
            if not chunks:
                logger.warning("目录无有效文档：%s", directory)
                continue
            # 分批向量化再写 Milvus，防止一次性编码全部分块把显存/内存打爆
            index.upsert_chunks(chunks)
            total += len(chunks)
            logger.info("知识域 %s 入库完成：%s 个分块", subject, len(chunks))
        except Exception as exc:
            logger.error("处理目录失败 %s：%s", directory, exc, exc_info=True)

    logger.info("全部入库完成，共写入 %s 个分块", total)
    print(f"OK 共写入 {total} 个分块，集合：{settings.MILVUS_DATABASE_NAME}.{settings.MILVUS_COLLECTION_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())