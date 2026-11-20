"""自研中文递归文本切分器：ingest.py 用它按 CHUNK_SIZE / CHUNK_OVERLAP 切出检索用分块。

LangChain 默认分隔符为英文设计（段 -> 换行 -> 空格 -> 单字符），中文句内没有空格，
用默认分隔符会退化到按单字硬切、拦腰截断句子。本类重写两处：
1. 分隔符层级换成中文标点：段 -> 换行 -> 句号/叹号/问号 -> 分号 -> 逗号；
2. 标点保留在前一块末尾（_split_text_with_regex_from_end），每个 chunk 都是带完整句号的自然句。

切分策略为"递归 + 合并"：优先用大单位（段落）切，超长块降级用小单位（句子、逗号）继续切，
尽量在语义边界下刀；再用父类 _merge_splits 把相邻碎块拼回接近 chunk_size，
并按 chunk_overlap 保留块间重叠，避免边界信息丢失。
"""
import re
from typing import List, Optional, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
        text: str, separator: str, keep_separator: bool
) -> List[str]:
    """按分隔符切分文本，并把分隔符保留在前一片段的末尾。

    re.split 加捕获组让分隔符留在结果里（正文/分隔符交替排列），
    zip 把偶数下标正文与奇数下标分隔符按位配对拼接，标点即归属前句；
    原文不以分隔符结尾时，最后一个正文片段没有配对，补 _splits[-1:] 防止丢字。

    separator 可为正则（如 "。|！|？"），空串表示按单字符切；
    keep_separator=False 时直接丢弃分隔符。
    """
    if separator:
        if keep_separator:
            # 捕获组让分隔符也出现在 split 结果里（正文/分隔符交替排列）
            _splits = re.split(f"({separator})", text)
            # 偶数列=正文、奇数列=分隔符，zip 按位配对拼接 => 标点归前句
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
            # 奇数长度说明末尾有个未配对的正文片段，补回来防止丢字
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
        else:
            splits = re.split(separator, text)
    else:
        # 空分隔符：终极兜底，按单字符切
        splits = list(text)
    return [s for s in splits if s != ""]


class ChineseTextSplitter(RecursiveCharacterTextSplitter):
    """中文递归切分器：中文标点分隔符层级 + 标点后置保留 + 递归切分再合并。

    只重写 __init__（换分隔符）和 _split_text（切分+合并调度），其余能力复用父类。
    """

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """separators：分隔符层级列表，顺序=优先级（越靠前单位越大），None 用中文默认层级。

        keep_separator=True 标点保留在前块末尾；is_separator_regex=True 分隔符按正则解释。
        其余参数（chunk_size / chunk_overlap / length_function 等）透传父类。
        """
        # 注意：separators 不传给父类，由本类自己管理（父类默认是英文分隔符）
        super().__init__(keep_separator=keep_separator, **kwargs)
        # 中文分隔符层级（从大到小，递归时逐级降级）：
        #   1. "\n\n"          段落：最理想的语义边界，优先按段切
        #   2. "\n"            换行：段内再按行切
        #   3. "。|！|？"       中文句末标点：按完整句子切
        #   4. r"\.\s|\!\s|\?\s" 英文句末标点+后随空白：要求标点后有空白，
        #                      避免把小数点 "3.14"、缩写 "U.S." 误判为句尾
        #   5. r"；|;\s"        分号：句子仍超长时按分句切
        #   6. r"，|,\s"        逗号：最后的"软边界"，按子句切
        #   （若全部失效，_split_text 里还有按单字符切的终极兜底）
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            r"\.\s|\!\s|\?\s",
            r"；|;\s",
            r"，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """递归切分主逻辑（重写父类同名方法），返回最终 chunk 列表。

        从粗到细选第一个在文本中真实出现的分隔符切分；短片段进缓冲区等待合并，
        超长片段先结算缓冲区再用更细的分隔符递归切；最后合并缓冲区剩余碎块、清洗空白。
        """
        final_chunks = []
        # 默认取最细的分隔符，再从头扫描找第一个出现的更粗分隔符
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                # 空串分隔符 = 按单字符切，终极兜底，选中后不再降级
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                # 记录更细的分隔符，留给超长片段递归降级
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # _good_splits：待合并的短片段缓冲区
        _good_splits = []
        # 标点已保留在片段末尾，合并时片段之间不需再插分隔符，故用空串
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    # 先结算缓冲区，保持原文顺序；_merge_splits 按 chunk_size 装满一块并保留重叠
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    # 没有更细的分隔符：原样收下超长块（如整段无标点的长串）
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        # 结算末尾剩余的短片段
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        # 压缩连续换行、去首尾空白、丢弃纯空白块
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip()!=""]


# 本地演示：python -m app.rag.splitters.chinese
if __name__ == "__main__":
    text_splitter = ChineseTextSplitter(
        keep_separator=True,
        is_separator_regex=True,
        chunk_size=150,
        chunk_overlap=10
    )
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
        ]
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
