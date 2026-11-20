"""文档加载器包：统一导出，ingest.py 只需 from app.rag.loaders import ...。"""

from app.rag.loaders.docx import DocxLoader
from app.rag.loaders.pdf import PdfLoader

__all__ = ["DocxLoader", "PdfLoader"]