"""文档向量的存储与检索：bge-m3 单向量检索（Milvus）。

入库方向：ingest.build_chunks 产出分块 -> upsert_chunks 向量化并写入 Milvus；
检索方向：pipeline.retrieve -> search(query) -> 返回排好序的分块作为上下文。

【核心设计：单层分块 + 稠密单向量直接召回】
    知识库规模小（几十份制度文档、几百个分块），不需要混合检索/重排：
    bge-m3 稠密向量 + IVF_FLAT 内积检索一次召回 top-k，
    再用相关性下限过滤域外问题，链路短、延迟低、可解释。
    FAQ 标准问句复用同一个类，只是换一个更小的集合。
"""
import datetime
import hashlib
import os

import numpy as np
import torch
from langchain_core.documents import Document
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import DataType, MilvusClient

from app.conf.logging import logger
from app.conf.settings import settings


class MilvusVectorIndex:
    """Milvus 单向量索引：建表、向量化入库、按知识域过滤的 top-k 检索。

    同一个类服务两个集合：知识库分块集合（zhiqing_kb）与 FAQ 问句集合（zhiqing_faq）。
    embedding_model 可从外部注入：两个集合共用一份 bge-m3，避免重复加载占显存。
    """

    def __init__(self,
                 milvus_host=settings.MILVUS_HOST,
                 milvus_port=settings.MILVUS_PORT,
                 milvus_database_name=settings.MILVUS_DATABASE_NAME,
                 milvus_collection_name=settings.MILVUS_COLLECTION_NAME,
                 dim=1024,
                 embedding_model=None):
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.milvus_database_name = milvus_database_name
        self.milvus_collection_name = milvus_collection_name
        self.dim = dim

        self.milvus_client = MilvusClient(uri=f'http://{self.milvus_host}:{self.milvus_port}',
                                          db_name=self.milvus_database_name)
        self._create_or_load_collection()

        if embedding_model is not None:
            self.embedding_model = embedding_model
        else:
            self.device = self._resolve_device()
            self.use_fp16 = self._resolve_fp16()
            embedding_model_path = os.path.join(settings.MODELS_DIR, settings.EMBEDDING_MODEL)
            # return_sparse=False：只算稠密向量，不算稀疏词频向量（单向量检索用不到）
            self.embedding_model = BGEM3EmbeddingFunction(
                model_name_or_path=embedding_model_path,
                use_fp16=self.use_fp16,
                device=self.device,
                return_sparse=False,
            )
            logger.info('embedding 模型加载完成：device=%s, fp16=%s', self.device, self.use_fp16)
            if self.device == 'cuda':
                logger.info('GPU：%s，显存 %.1f GB',
                            torch.cuda.get_device_name(0),
                            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)

    @staticmethod
    def _resolve_device() -> str:
        """解析推理设备：auto / cuda / cpu；配置要求 cuda 但不可用时回退 cpu。"""
        pref = str(getattr(settings, 'EMBEDDING_DEVICE', 'auto')).lower()
        if pref == 'cpu':
            return 'cpu'
        if pref == 'cuda':
            if torch.cuda.is_available():
                return 'cuda'
            logger.warning('配置要求 cuda，但当前 torch 不支持 CUDA，已回退 cpu')
            return 'cpu'
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _resolve_fp16(self) -> bool:
        """解析半精度开关：auto（有 GPU 就开）/ true / false。

        fp16 每个参数占 2 字节（fp32 占 4 字节），6GB 显存的卡上开 fp16
        可省一半显存，对检索精度影响很小；CPU 上不开（半精度支持差、反而更慢）。
        """
        pref = str(getattr(settings, 'EMBEDDING_FP16', 'auto')).lower()
        if pref in ('1', 'true', 'yes', 'on'):
            return True
        if pref in ('0', 'false', 'no', 'off'):
            return False
        return self.device == 'cuda'

    def _create_or_load_collection(self):
        """创建或加载 Milvus 集合（建表 + 建索引 + 载入内存）。

        6 字段 schema：每一行存"一个分块"：
            1. id           分块内容 md5，作主键 => 幂等 upsert 的依据；
            2. content      分块原文 => 命中后直接喂给大模型做上下文；
            3. dense_vector 稠密语义向量（1024 维）=> 语义召回；
            4. subject      知识域 => 支持按域过滤（expr）；
            5. source       来源文件名 => 答案引用溯源；
            6. timestamp    入库时间 => 可追溯。
        """
        if not self.milvus_client.has_collection(self.milvus_collection_name):
            logger.info('表%s不存在，执行建表逻辑', self.milvus_collection_name)
            try:
                schema = self.milvus_client.create_schema(auto_id=False, enable_dynamic_field=True)
                schema.add_field(field_name='id', datatype=DataType.VARCHAR, max_length=128, is_primary=True)
                schema.add_field(field_name='content', datatype=DataType.VARCHAR, max_length=8192)
                schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=self.dim)
                schema.add_field(field_name='subject', datatype=DataType.VARCHAR, max_length=32)
                schema.add_field(field_name='source', datatype=DataType.VARCHAR, max_length=256)
                schema.add_field(field_name='timestamp', datatype=DataType.VARCHAR, max_length=64)

                index_params = self.milvus_client.prepare_index_params()
                # IVF_FLAT + IP：聚成 128 个簇，查询只探最近的若干簇（nprobe），
                # 簇内保留原始向量不压缩，适合中小规模知识库。
                # bge-m3 向量已 L2 归一化，内积 = 余弦相似度。
                index_params.add_index(
                    field_name='dense_vector',
                    index_name='dense_vector_index',
                    metric_type='IP',
                    index_type='IVF_FLAT',
                    params={'nlist': 128},
                )
                self.milvus_client.create_collection(
                    collection_name=self.milvus_collection_name,
                    schema=schema,
                    index_params=index_params,
                )
                logger.info('执行建表：%s成功', self.milvus_collection_name)
            except Exception as e:
                logger.error('执行建表失败：%s, 原因：%s', self.milvus_collection_name, e)

        try:
            self.milvus_client.load_collection(self.milvus_collection_name)
            logger.info('集合:%s加载完毕！', self.milvus_collection_name)
        except Exception as e:
            logger.error('执行加载集合%s失败, 原因%s', self.milvus_collection_name, e)

    def recreate_collection(self):
        """删除并重建集合（换知识库/换 embedding 维度时使用）。"""
        if self.milvus_client.has_collection(self.milvus_collection_name):
            self.milvus_client.drop_collection(self.milvus_collection_name)
            logger.info('已删除集合：%s', self.milvus_collection_name)
        self._create_or_load_collection()

    def upsert_chunks(self, documents):
        """分批向量化 + 分批写入，避免一次性 batch 把显存/内存打爆。

        分块主键取内容 md5，同一批文档重复入库是幂等的（upsert 覆盖）。
        """
        batch_size = max(1, int(getattr(settings, 'EMBED_BATCH_SIZE', 16)))
        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            rows = self._embed_rows(batch)
            if not rows:
                continue
            logger.info('写入第 %s-%s 个分块 -> %s', start + 1, start + len(batch),
                        self.milvus_collection_name)
            self.milvus_client.upsert(collection_name=self.milvus_collection_name, data=rows)
            total += len(rows)

        if total:
            try:
                self.milvus_client.flush(collection_name=self.milvus_collection_name)
                stats = self.milvus_client.get_collection_stats(self.milvus_collection_name)
                logger.info('写入并 flush 完成：本次 %s 条，集合数据量 %s', total, stats.get('row_count'))
            except Exception as flush_err:
                logger.warning('flush 失败（upsert 已生效，可忽略）: %s', flush_err)

    def _embed_rows(self, documents):
        """对一批分块做向量化，并组装成 Milvus 行。"""
        texts = [document.page_content for document in documents]
        embeddings = self.embedding_model(texts)
        rows = []
        for i, document in enumerate(documents):
            content = document.page_content
            rows.append({
                # 主键 = 内容 md5：内容相同 => 主键相同 => upsert 覆盖而非新增（幂等）
                'id': hashlib.md5(content.encode('utf-8')).hexdigest(),
                'content': content,
                # Milvus 字段要求 float32 ndarray（GPU/fp16 推理输出可能是 float16/list）
                'dense_vector': np.asarray(embeddings['dense'][i], dtype=np.float32),
                'subject': document.metadata.get('subject', 'unknown'),
                'source': document.metadata.get('source', ''),
                'timestamp': document.metadata.get('timestamp', datetime.datetime.now().isoformat()),
            })
        return rows

    def search(self, query, k=None, subject_filter=None, min_score=None, with_scores=False):
        """单向量检索：query 向量化 -> Milvus top-k 召回 -> 相关性下限过滤。

        参数：
            query：用户问题；k：召回条数（默认 RETRIEVAL_K）；
            subject_filter：按知识域过滤（None 不过滤）；
            min_score：相关性下限（余弦相似度），低于此分的分块丢弃（默认 MIN_SCORE）；
            with_scores：True 时返回 (docs, scores)，便于上层展示/调试。
        """
        k = settings.RETRIEVAL_K if k is None else k
        if min_score is None:
            min_score = settings.MIN_SCORE

        embeddings = self.embedding_model([str(query)])
        dense_vector = np.asarray(embeddings['dense'][0], dtype=np.float32)
        filter_expr = f'subject == "{subject_filter}"' if subject_filter else ""

        results = self.milvus_client.search(
            collection_name=self.milvus_collection_name,
            data=[dense_vector],
            anns_field='dense_vector',
            search_params={'metric_type': 'IP', 'params': {'nprobe': 10}},
            limit=k,
            filter=filter_expr,
            output_fields=['content', 'subject', 'source', 'timestamp'],
        )[0]

        docs, scores = [], []
        for hit in results:
            score = float(hit['distance'])
            # 相关性下限：域外无关问题的相似度低，在此被滤掉，
            # 上层拿到空上下文就走"信息不足/人工客服"兜底，不挂无关来源
            if score < min_score:
                continue
            entity = hit['entity']
            docs.append(Document(
                page_content=entity['content'],
                metadata={
                    'chunk_id': hit['id'],
                    'subject': entity.get('subject', ''),
                    'source': entity.get('source', ''),
                    'timestamp': entity.get('timestamp', ''),
                },
            ))
            scores.append(score)
        if min_score:
            logger.info('相关性下限 %.2f：%s 个候选中保留 %s 个', min_score, len(results), len(docs))
        return (docs, scores) if with_scores else docs


if __name__ == '__main__':
    store = MilvusVectorIndex()
    results = store.search('VPN 连不上怎么办')
    print(results)