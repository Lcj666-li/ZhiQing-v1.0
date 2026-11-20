"""离线训练意图分类器：TF-IDF（字符 1-2 gram）+ 逻辑回归，二分类"通用知识/专业咨询"。

用法（项目根目录执行一次即可）：
    python -m scripts.train_intent

读取 data/intent/intent_dataset.jsonl，8:2 分层切分（random_state=42 可复现），
输出验证集准确率与分类报告，整条 Pipeline 保存到 models/intent_clf.joblib
（向量器+分类器一起保存，保证训练/推理特征一致）。
"""

from __future__ import annotations

import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.conf.logging import logger
from app.conf.settings import settings


def main() -> None:
    data_file = settings.INTENT_DATASET
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"数据集文件 {data_file} 不存在")

    with open(data_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    texts = [item["query"] for item in data]
    labels = [item["label"] for item in data]
    logger.info("加载标注数据 %s 条", len(texts))

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    model.fit(train_texts, train_labels)

    preds = model.predict(val_texts)
    acc = accuracy_score(val_labels, preds)
    # labels 显式指定顺序，避免 sklearn 按 Unicode 排序导致 target_names 与行错位
    report = classification_report(val_labels, preds,
                                   labels=["通用知识", "专业咨询"],
                                   target_names=["通用知识", "专业咨询"])
    logger.info("验证集准确率：%.4f", acc)
    logger.info("分类报告：\n%s", report)
    print(f"验证集准确率：{acc:.4f}")
    print(report)

    os.makedirs(os.path.dirname(settings.INTENT_MODEL_FILE), exist_ok=True)
    joblib.dump(model, settings.INTENT_MODEL_FILE)
    logger.info("模型已保存：%s", settings.INTENT_MODEL_FILE)
    print(f"模型已保存：{settings.INTENT_MODEL_FILE}")


if __name__ == "__main__":
    main()