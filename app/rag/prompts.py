"""提示词库：进入大模型的作答模板集中在这里，方便统一调优。

【架构位置】rag 层的"提示词单一来源"。pipeline 通过 PromptLibrary 取模板，
    再 .format(...) 填入变量后发给 LLM。集中管理的好处：调优只改这一处。

模板变量：{context}/{question}/{history}/{phone}，
渲染后的最终提示词要保证"能答就答、答不了给人工兜底"。

【重要约束】template 三引号内是"提示词正文"，属于字符串字面量，
    所有讲解性注释一律写在字符串外部。
"""

from __future__ import annotations

from langchain_core.prompts import PromptTemplate


class PromptLibrary:
    """知擎的提示词集合（全部为静态方法，无状态）。"""

    @staticmethod
    def rag_prompt() -> PromptTemplate:
        """结合检索上下文 + 多轮历史生成答案的主模板。

        模板设计意图：
            · 规则 1 要求"优先用参考资料 + 标注来源序号"，与 pipeline 的 [1][2] 编号呼应；
            · 规则 2 允许通用知识补充但要注明，平衡"有用"与"不幻觉"；
            · 规则 3 限定历史只用于理解指代，防止照抄旧结论；
            · 规则 5 给出"信息不足 -> 人工客服电话"的兜底出口（{phone} 注入）。
        """
        return PromptTemplate(
            template="""你是「知擎」知识问答助手，服务对象是企业内部员工。
请遵循以下规则回答：

1. 优先使用【参考资料】中的内容作答，并在关键结论后用（来源：文档片段序号）标注依据；
2. 参考资料不足时，可以补充你的通用知识，但要注明"以下为通用知识补充"；
3. 【对话历史】只用于理解上下文与指代，不要照抄历史中的旧结论；
4. 回答要分点、口语化，避免堆砌套话；不确定就直说，不要编造；
5. 完全无法回答时，回复：信息不足，请联系人工客服（{phone}）。

【参考资料】
{context}

【对话历史】
{history}

【用户问题】
{question}

【回答】""",
            input_variables=["context", "question", "history", "phone"],
        )


if __name__ == "__main__":  # 本地预览渲染效果：python -m app.rag.prompts
    prompt = PromptLibrary.rag_prompt().format(
        context="示例上下文", question="知擎支持哪些文档格式？", history="", phone="10086"
    )
    print(prompt)