"""Word 加载器：python-docx 按"文档原始顺序"提取段落与表格文字。

doc.paragraphs 与 doc.tables 是两个独立列表，分别遍历会丢失段落与表格的交错顺序
（"说明段->表格->结论段"会被拆成先所有段落后所有表格），切块后语义就乱了。
因此 iter_block_items 直接遍历文档 body 的底层 XML 子节点（CT_P=段落 / CT_Tbl=表格），
按 XML 出现顺序 yield，还原作者写作时的阅读顺序。
"""

from typing import Iterator

from docx.table import _Cell, Table
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.text.paragraph import Paragraph
# Document 工厂函数与文档类型同名不同物，起别名区分：
# Docu1 打开文件返回文档对象；Docu2 仅用于 isinstance 判断。
from docx import Document as Docu1
from docx.document import Document as Docu2
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class DocxLoader(BaseLoader):
    """Word 文档加载器：按原始顺序遍历段落与表格，整份文档汇总为一个 Document。"""

    def __init__(self, filepath: str) -> None:
        """记录待加载 Word 文件的路径（构造时不做 IO）。

        注意：python-docx 只支持新版 .docx（Office Open XML），
        老版二进制 .doc 实际无法解析，ingest.py 里的 .doc 映射属于历史遗留约定。
        """
        self.filepath = filepath

    def lazy_load(self) -> Iterator[Document]:
        """惰性加载入口：解析整份 Word 后 yield 一个 Document。"""
        yield Document(page_content=self.doc2text(self.filepath), metadata={"source": self.filepath})

    def doc2text(self, filepath) -> str:
        """核心解析函数：按文档原始顺序提取所有块的文字，返回全文字符串。"""
        doc = Docu1(filepath)
        resp = ""

        def iter_block_items(parent):
            """按 XML 顺序产出段落/表格对象（保持原文顺序的关键）。"""
            if isinstance(parent, Docu2):
                parent_elm = parent.element.body
            elif isinstance(parent, _Cell):
                # 单元格取 _tc 元素：同一套逻辑也能递归处理"表中表"
                parent_elm = parent._tc
            else:
                raise ValueError("DocxLoader parse fail")
            for child in parent_elm.iterchildren():
                if isinstance(child, CT_P):
                    # CT_P 是底层 XML 节点，包装成 Paragraph 才有 .text 等高层 API
                    yield Paragraph(child, parent)
                elif isinstance(child, CT_Tbl):
                    yield Table(child, parent)

        for block in iter_block_items(doc):
            if isinstance(block, Paragraph):
                resp += block.text.strip() + "\n"
            elif isinstance(block, Table):
                # 行 -> 单元格 -> 段落 三层遍历。拍平成一维文本会丢失行列结构，
                # 但保留了内容可被检索到
                for row in block.rows:
                    for cell in row.cells:
                        for paragraph in cell.paragraphs:
                            resp += paragraph.text.strip() + "\n"
        return resp


if __name__ == '__main__':
    import sys
    loader = DocxLoader(filepath=sys.argv[1])
    print(loader.load())