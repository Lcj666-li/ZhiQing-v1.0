"""知擎（ZhiQing）全局配置。

配置来源优先级：环境变量（含 .env） > config.ini > 代码内兜底默认值。
本地开发写 .env / config.ini 即可；生产/容器用环境变量注入，密钥不进代码库。
所有路径类配置都以项目根目录为基准，避免受启动目录影响。
"""

from __future__ import annotations

import ast
import configparser
import os
from pathlib import Path

try:  # 本地开发时从 .env 读取连接信息；生产用容器环境变量注入
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    # python-dotenv 为可选依赖，缺失时跳过 .env 加载
    load_dotenv = None


class Settings:
    """项目配置对象（单例，模块底部导出 settings）。"""

    APP_NAME = "知擎"

    def __init__(self, config_file: str | None = None) -> None:
        """初始化配置。config_file 可指定自定义 ini 路径（便于测试），默认用项目根的 config.ini。"""
        # 所有路径基于项目根拼接，与启动时的工作目录无关
        self.PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
        self.LOG_DIR = os.path.join(self.PROJECT_ROOT, "logs")
        self.MODELS_DIR = os.path.join(self.PROJECT_ROOT, "models")
        self.DATA_DIR = os.path.join(self.PROJECT_ROOT, "data")
        self.KB_DIR = os.path.join(self.DATA_DIR, "kb")
        self.FAQ_CSV = os.path.join(self.DATA_DIR, "faq", "faq_seed.csv")
        self.INTENT_DATASET = os.path.join(self.DATA_DIR, "intent", "intent_dataset.jsonl")
        # 意图分类器训练产物（TF-IDF + 逻辑回归，scripts/train_intent.py 生成）
        self.INTENT_MODEL_FILE = os.path.join(self.MODELS_DIR, "intent_clf.joblib")
        self.WEB_DIR = os.path.join(self.PROJECT_ROOT, "web")

        if load_dotenv is not None:
            env_file = os.path.join(self.PROJECT_ROOT, ".env")
            if os.path.exists(env_file):
                # override=False：真实环境变量优先于 .env 文件，.env 只作本地开发的兜底填充
                load_dotenv(env_file, override=False)

        self._parser = configparser.ConfigParser()
        config_file = config_file or os.path.join(self.PROJECT_ROOT, "config.ini")
        self._parser.read(config_file, encoding="utf-8")

        self._load_mysql()
        self._load_vector_db()
        self._load_llm()
        self._load_retrieval()
        self._load_app()
        self._load_auth()

    # ------------------------------------------------------------------ #
    # 分块加载，便于阅读与单测
    # ------------------------------------------------------------------ #
    def _env(self, key: str, section: str, option: str, fallback: str) -> str:
        """三级优先级取值：环境变量 > config.ini > 代码兜底默认值。

        注意：`or` 把空字符串也当假，"显式设为空的环境变量"会继续回退到 ini/默认值。
        """
        return os.getenv(key) or self._parser.get(section, option, fallback=fallback)

    def _load_mysql(self) -> None:
        """加载 MySQL 连接参数（会话历史、FAQ 数据的持久化存储）。"""
        self.MYSQL_HOST = self._env("MYSQL_HOST", "mysql", "host", "localhost")
        self.MYSQL_PORT = int(self._env("MYSQL_PORT", "mysql", "port", "3306"))
        self.MYSQL_USER = self._env("MYSQL_USER", "mysql", "user", "zhiqing")
        self.MYSQL_PASSWORD = self._env("MYSQL_PASSWORD", "mysql", "password", "123456")
        self.MYSQL_DATABASE = self._env("MYSQL_DATABASE", "mysql", "database", "zhiqing_kb")
        # 表名集中管理，避免散落在 SQL 字符串里
        self.FAQ_TABLE = "faq_entry"
        self.CHAT_TABLE = "chat_turn"

    def _load_vector_db(self) -> None:
        """加载 Milvus 向量库与嵌入模型配置（RAG 检索通道的核心）。"""
        self.MILVUS_HOST = self._env("MILVUS_HOST", "milvus", "host", "localhost")
        self.MILVUS_PORT = int(self._env("MILVUS_PORT", "milvus", "port", "19530"))
        self.MILVUS_DATABASE_NAME = self._env("MILVUS_DATABASE_NAME", "milvus", "database_name", "zhiqing")
        self.MILVUS_COLLECTION_NAME = self._env("MILVUS_COLLECTION_NAME", "milvus", "collection_name", "zhiqing_kb")
        self.MILVUS_FAQ_COLLECTION_NAME = self._env("MILVUS_FAQ_COLLECTION_NAME", "milvus", "faq_collection_name", "zhiqing_faq")
        self.EMBEDDING_MODEL = self._parser.get("milvus", "embedding_model", fallback="bge-m3")
        # 推理设备与精度：auto = 有 GPU 就 cuda + fp16（6GB 显存建议保持 auto）
        self.EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE") or self._parser.get("milvus", "device", fallback="auto")
        self.EMBEDDING_FP16 = os.getenv("EMBEDDING_FP16") or self._parser.get("milvus", "fp16", fallback="auto")
        self.EMBED_BATCH_SIZE = int(self._parser.get("milvus", "embed_batch", fallback="16"))

    def _load_llm(self) -> None:
        """加载大模型（DashScope/通义）配置：模型名、兼容 OpenAI 的 base_url、密钥、超时。"""
        self.LLM_MODEL = self._env("LLM_MODEL", "llm", "model", "qwen-plus")
        self.DASHSCOPE_BASE_URL = self._env(
            "DASHSCOPE_BASE_URL", "llm", "base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # 密钥只从环境变量取（不走 _env/ini），避免把敏感信息写进配置文件或代码库
        self.DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
        self.LLM_TIMEOUT = int(self._parser.get("llm", "timeout", fallback="60"))
        # 没配密钥不直接崩溃，只打印告警：此时 RAG 生成会退化为兜底话术，FAQ 通道仍可正常用
        if not self.DASHSCOPE_API_KEY:
            print("[知擎] 警告：未配置 DASHSCOPE_API_KEY，问答链路会退化为兜底话术")

    def _load_retrieval(self) -> None:
        """加载 RAG 检索/分块参数。

        知识库规模小（几十份制度文档），采用单层分块：chunk_size 控制块大小，
        overlap 是相邻块的重叠字数，避免句子被切断丢语义。
        retrieval_k 是向量召回条数；min_score 是相关性下限（余弦相似度），
        低于它的分块丢弃，域外问题因此被滤成空上下文。
        """
        self.CHUNK_SIZE = self._parser.getint("retrieval", "chunk_size", fallback=500)
        self.CHUNK_OVERLAP = self._parser.getint("retrieval", "chunk_overlap", fallback=60)
        self.RETRIEVAL_K = self._parser.getint("retrieval", "retrieval_k", fallback=5)
        self.MIN_SCORE = float(self._parser.get("retrieval", "min_score", fallback="0.35"))
        self.FAQ_MIN_SCORE = float(self._parser.get("retrieval", "faq_min_score", fallback="0.85"))

    def _load_app(self) -> None:
        """加载应用级配置：知识域白名单、客服电话、API 前缀、日志文件路径。"""
        raw_sources = self._parser.get(
            "app", "valid_sources", fallback='["it", "oa", "security", "hr", "finance", "legal", "admin"]'
        )
        # ini 里存的是形如 '["it","oa"]' 的字符串，用 ast.literal_eval 安全地解析成真正的列表。
        # 不用 eval：literal_eval 只允许字面量，不会执行任意代码，避免配置注入风险。
        self.VALID_SOURCES = ast.literal_eval(raw_sources)
        self.CUSTOMER_SERVICE_PHONE = self._parser.get("app", "customer_service_phone", fallback="10086")
        self.API_PREFIX = self._parser.get("app", "api_prefix", fallback="/api/v1")
        self.LOG_FILE = self._parser.get("logger", "log_file", fallback="logs/app.log")
        # 若配置的是相对路径，则拼到项目根下，保证日志落盘位置稳定（同样不受启动目录影响）
        if not os.path.isabs(self.LOG_FILE):
            self.LOG_FILE = os.path.join(self.PROJECT_ROOT, self.LOG_FILE)

    def _load_auth(self) -> None:
        """加载 JWT 鉴权配置：演示账号、签名密钥、token 有效期。

        密钥与口令支持环境变量注入（AUTH_PASSWORD / JWT_SECRET），
        生产部署不要沿用 config.ini 里的演示默认值。
        """
        self.AUTH_USERNAME = self._env("AUTH_USERNAME", "auth", "username", "admin")
        self.AUTH_PASSWORD = self._env("AUTH_PASSWORD", "auth", "password", "zhiqing@2026")
        self.JWT_SECRET = self._env("JWT_SECRET", "auth", "secret_key", "zhiqing-dev-secret-change-in-prod")
        self.JWT_EXPIRE_MINUTES = int(self._parser.get("auth", "token_expire_minutes", fallback="1440"))


# 全项目共用的配置单例
settings = Settings()

if __name__ == "__main__":  # 本地自检：python -m app.conf.settings
    print(f"APP={settings.APP_NAME} ROOT={settings.PROJECT_ROOT}")
    print(f"MySQL={settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}")
    print(f"Milvus={settings.MILVUS_HOST}:{settings.MILVUS_PORT}/{settings.MILVUS_DATABASE_NAME}"
          f".{settings.MILVUS_COLLECTION_NAME}")
    print(f"LLM={settings.LLM_MODEL} API_PREFIX={settings.API_PREFIX}")
