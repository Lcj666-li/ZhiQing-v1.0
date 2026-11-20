"""意图识别：关键词规则为主 + 轻量文本分类（TF-IDF + 逻辑回归），不做大模型微调。

【架构位置】rag 层，是问答链路的"第一道分流闸"。
    query -> IntentClassifier.predict -> "通用知识"/"专业咨询"
        · 通用知识：不查知识库，直接让大模型作答；
        · 专业咨询：进入 RAG 检索流程（route_domain 定位知识域 -> 向量检索 -> 生成）。

【两层设计】
    1. 关键词规则（DOMAIN_KEYWORDS）：每个知识域一组特征词。
       is_domain_query 判断"是否需要查库"（规则兜底，纠正分类器误判）；
       route_domain 按命中词数最多的域路由（用户未显式选域时使用）。
    2. 轻量分类器：TF-IDF（字符 1-2 gram）+ LogisticRegression 二分类，
       用 210 条标注数据（通用/专业各 105）离线训练（scripts/train_intent.py），
       产物 models/intent_clf.joblib，在线推理毫秒级、无需 GPU。
       模型文件不存在时自动退化为纯关键词规则，服务照样能跑。

【为什么不用 BERT/大模型微调】知识库只有 7 个域、几十份文档，
    问题分布集中，字符级 TF-IDF + 线性分类器已足够区分"通用/专业"，
    训练秒级完成、推理无显存开销，维护成本远低于深度模型。
"""

from __future__ import annotations

import os

import joblib

from app.conf.logging import logger
from app.conf.settings import settings

# 每个知识域的关键词表：命中即认为问题落在该域。
# 注意词表之间刻意避免歧义包含（如"会议"只出现在 admin 的"会议室"里，
# 防止"预订会议室"被路由到 oa）。
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "it": ("vpn", "账号", "密码", "域", "权限", "共享盘", "打印机", "邮箱", "投屏",
           "网络", "wifi", "无线", "证书", "u盘", "工单", "蓝屏", "软件", "安装", "电脑"),
    "oa": ("考勤", "打卡", "请假", "年假", "调休", "加班", "审批", "流程",
           "出差", "外勤", "公告", "通知"),
    "hr": ("入职", "离职", "转正", "福利", "培训", "体检", "社保", "公积金",
           "工资", "薪酬", "绩效", "试用期"),
    "finance": ("报销", "发票", "差旅", "预算", "付款", "借款", "费用",
                "结算", "供应商", "采购"),
    "legal": ("合同", "用印", "法务", "合规", "审查", "盖章", "知识产权",
              "商标", "专利"),
    "security": ("钓鱼", "泄露", "外发", "安全", "加密", "病毒", "信息安全",
                 "脱敏", "中毒", "可疑"),
    "admin": ("会议室", "资产", "办公用品", "车辆", "用车", "工位", "门禁", "访客", "食堂"),
}


def is_domain_query(query: str) -> bool:
    """粗粒度判断：问题是否落在企业知识库的业务范围内（任一域命中即为真）。

    先 lower() 再匹配，VPN/vpn、WiFi/wifi 等大小写写法都能命中。
    用子串包含而非分词，简单快速，适合做高召回的兜底规则。
    """
    text = (query or "").lower()
    return any(keyword in text
               for keywords in DOMAIN_KEYWORDS.values()
               for keyword in keywords)


def route_domain(query: str) -> str | None:
    """按关键词命中数把问题路由到某个知识域；一个都不命中返回 None。

    计票规则：统计每个域的特征词在 query 中出现的个数，取最高票的域。
    命中数相同时按 DOMAIN_KEYWORDS 的声明顺序取先者（dict 保序），结果稳定可复现。
    """
    text = (query or "").lower()
    best_domain, best_hits = None, 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits > best_hits:
            best_domain, best_hits = domain, hits
    return best_domain


class IntentClassifier:
    """轻量意图分类器：加载离线训练好的 TF-IDF + 逻辑回归 Pipeline。

    model_file 可注入（测试时传假模型路径）；默认取 settings.INTENT_MODEL_FILE。
    模型文件缺失时不报错，predict 退化为关键词规则判断。
    """

    def __init__(self, model_file: str | None = None) -> None:
        self.model_file = model_file or settings.INTENT_MODEL_FILE
        self.model = None
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            logger.info("意图分类器加载成功：%s", self.model_file)
        else:
            logger.warning("意图模型 %s 不存在，退化为关键词规则（可执行 scripts/train_intent.py 训练）",
                           self.model_file)

    def predict(self, query: str) -> str:
        """输入一句 query，输出"通用知识"或"专业咨询"。"""
        if self.model is None:
            # 规则兜底：命中领域词就查库，否则交给大模型直接作答
            return "专业咨询" if is_domain_query(query) else "通用知识"
        return str(self.model.predict([query])[0])


if __name__ == "__main__":  # 冒烟测试：python -m app.rag.intent
    clf = IntentClassifier()
    for q in ["VPN 连不上怎么办", "差旅报销的发票要求", "帮我写一首关于春天的诗", "预订会议室"]:
        print(f"{q} -> {clf.predict(q)} / 域={route_domain(q)}")