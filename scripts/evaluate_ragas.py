"""RAGAS 离线评测：对 (question, answer, contexts, ground_truth) 样本打分。

用法：
    python -m scripts.evaluate_ragas
    python -m scripts.evaluate_ragas --small      # 小样本先跑通流程、省 token

指标：faithfulness / answer_relevancy 衡量生成质量，
context_precision / context_recall 衡量检索质量，四者组合可定位问题出在哪一段。
每个指标都要调用裁判 LLM 打分，会产生 token 费用。

评测集为 JSON 列表，每条含 question / answer / context / ground_truth 四个字段
（answer 与 context 由被评测的 RAG 系统预先生成）。
"""

from __future__ import annotations

import argparse
import json
import os

from app.conf.settings import settings

METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def load_dataset(small: bool = False) -> list[dict]:
    """读取 data/eval/ 下的评测数据集，small=True 时读小样本文件。"""
    name = "rag_evaluate_data_small.json" if small else "rag_evaluate_data.json"
    path = os.path.join(settings.DATA_DIR, "eval", name)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS 评测")
    parser.add_argument("--small", action="store_true", help="使用小样本")
    args = parser.parse_args()

    # 评测依赖为可选安装，缺失时给出提示而不是启动即崩溃
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:  # pragma: no cover
        print(f"缺少评测依赖：{exc}\n请先 pip install ragas datasets langchain-openai")
        return 1

    # RAGAS 要求列表型字段 contexts，本地文件的 context 需映射
    rows = load_dataset(args.small)
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["context"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    # temperature=0：同一条样本的打分尽量稳定可复现
    judge = ChatOpenAI(model=settings.LLM_MODEL, base_url=settings.DASHSCOPE_BASE_URL,
                       api_key=settings.DASHSCOPE_API_KEY, temperature=0)

    print(f"开始评测：{len(rows)} 条样本，指标 {METRICS}")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
    )
    print(result)

    out_path = os.path.join(settings.DATA_DIR, "eval", "ragas_results.csv")
    result.to_pandas().to_csv(out_path, index=False)
    print(f"结果已保存：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
