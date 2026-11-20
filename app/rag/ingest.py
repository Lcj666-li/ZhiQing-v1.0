"""文档入库：读取 -> 单层分块 -> 组装元数据（向量化在 infra.vector_index）。

【架构位置】rag 层的"入库（ingestion）"入口，是知识库构建的第一步。
    数据流：原始文件（Markdown/PDF/DOCX/txt）
        -> load_documents 加载成 List[Document] 并打上知识域/来源/时间戳元数据
        -> build_chunks 单层切分（约 500 字、重叠 60 字）
        -> 交给 infra.vector_index.upsert_chunks 做 bge-m3 向量化并写入 Milvus。

【分块策略】知识库规模小（几十份制度文档），采用单层分块即可：
    · chunk_size=500：一个分块聚焦一个小节，检索精度与上下文完整性兼顾；
    · overlap=60：相邻块共享一段文本，避免句子被切断导致语义丢失；
    · .md 用 MarkdownTextSplitter（尊重标题/列表结构），
      其它文本用 ChineseTextSplitter（按中文标点断句，避免半句切断）。

【可回溯】每个分块带 chunk_id（doc_i_chunk_j）、source（原文文件名）、
    subject（知识域）、timestamp，答案引用的来源卡片据此回溯到原始文档。
"""

import os
from datetime import datetime

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter

from app.rag.loaders import PdfLoader, DocxLoader
from app.rag.splitters.chinese import ChineseTextSplitter
from app.conf.logging import logger
from app.conf.settings import settings

# 扩展名 -> 加载器。.md 用 TextLoader 读纯文本，标题结构由 MarkdownTextSplitter 负责。
# 注意 python-docx 只支持新版 .docx，老版二进制 .doc 无法解析（历史遗留映射）。
document_loaders = {
    '.txt': TextLoader,
    '.md': TextLoader,
    '.doc': DocxLoader,
    '.docx': DocxLoader,
    '.pdf': PdfLoader,
}


def load_documents(directory_path):
    """递归加载目录下所有受支持的文件，返回 List[Document]（每篇带元数据）。

    知识域从目录名推断（约定 {域名}_data，去掉 _data 后缀即域，如 it_data -> it）。
    单个文件加载失败只记日志、跳过，不影响其它文件（入库尽量"能进多少进多少"）。
    """
    documents = []
    supported_extensions = document_loaders.keys()
    subject = os.path.basename(directory_path).replace("_data", "")

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            _, extension = os.path.splitext(file_name)
            file_path = os.path.join(root, file_name)
            if extension not in supported_extensions:
                logger.warning(f'处理文件:{file_path}失败，不支持的类型')
                continue
            try:
                loader_class = document_loaders[extension]
                if extension in ('.txt', '.md'):
                    loader = loader_class(file_path=file_path, encoding='utf-8')
                else:
                    loader = loader_class(file_path)
                loaded_documents = loader.load()
                for loaded_document in loaded_documents:
                    loaded_document.metadata['timestamp'] = datetime.now().isoformat()
                    loaded_document.metadata['subject'] = subject
                    loaded_document.metadata['extension'] = extension
                    # source 统一记文件名：来源引用据此回溯到入库前的原始文档
                    loaded_document.metadata['source'] = file_name
                documents.extend(loaded_documents)
            except Exception as e:
                logger.error(f'处理文件:{file_path}失败，原因:{e}')

    return documents


def build_chunks(
        directory_path: str,
        chunk_size=settings.CHUNK_SIZE,
        overlap_size=settings.CHUNK_OVERLAP,
):
    """把知识域目录下的文档切成单层分块，返回带完整元数据的分块列表。

    id 命名：doc_{文档序号}_chunk_{块序号}，全局唯一且可追溯归属。
    每个分块的 metadata：chunk_id / subject / source / timestamp，
    与 Milvus 6 字段 schema 一一对应（主键由内容 md5 生成，在 vector_index 里做）。
    """
    documents = load_documents(directory_path)

    markdown_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    text_splitter = ChineseTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)

    chunks = []
    for i, document in enumerate(documents):
        splitter = markdown_splitter if document.metadata['extension'] == '.md' else text_splitter
        for j, chunk in enumerate(splitter.split_documents([document])):
            chunk.metadata['chunk_id'] = f"doc_{i}_chunk_{j}"
            chunks.append(chunk)

    logger.info("目录 %s 切分完成：%s 篇文档 -> %s 个分块", directory_path, len(documents), len(chunks))
    return chunks


if __name__ == '__main__':
    result = build_chunks(os.path.join(settings.DATA_DIR, 'kb', 'it_data'))
    print(len(result))