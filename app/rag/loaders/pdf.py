"""PDF 加载器：PyMuPDF(fitz) 逐页提取文字层。

知识库均为文字版 PDF（内部有文字层），page.get_text("text") 直接提取，无需 OCR。
注意：import 名是 fitz，但安装包叫 PyMuPDF；pip 上另有一个无关的旧包也叫 "fitz"，不要装错。
"""

from typing import Iterator

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class PdfLoader(BaseLoader):
    """PDF 文档加载器：逐页提取文字层，整份 PDF 汇总为一个 Document。

    页与页之间用换行分隔，切分职责交给切分器，加载器不做切分。
    """

    def __init__(self, file_path: str) -> None:
        """记录待加载 PDF 的路径（构造时不做 IO，解析推迟到 lazy_load）。"""
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载入口：把整份 PDF 解析成一个 Document 后 yield。"""
        yield Document(page_content=self.pdf2text(), metadata={"source": self.file_path})

    def pdf2text(self) -> str:
        """逐页提取文字层文本，按页拼接返回全文字符串。"""
        pages = []
        with fitz.open(self.file_path) as doc:
            for page in doc:
                pages.append(page.get_text("text"))
        return "\n".join(pages)


if __name__ == '__main__':
    import sys
    loader = PdfLoader(file_path=sys.argv[1])
    print(loader.load())