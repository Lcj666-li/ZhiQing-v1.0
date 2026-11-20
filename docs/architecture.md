# 知擎 v1.0 技术设计文档

## 1. 系统概述

### 1.1 简介

知擎是面向企业内部知识库的智能问答服务，基于 FastAPI，采用「FAQ 精确问答 + RAG 检索增强」双通道。
知识库规模小（7 个知识域、30 份制度文档、80 条标准 FAQ），检索用 **bge-m3 单向量直接召回**，
链路短、延迟低、可解释；支持多轮会话、WebSocket 流式输出与 JWT 鉴权。

### 1.2 技术栈

| 关注点 | 选型 |
| --- | --- |
| Web 框架 | FastAPI + Uvicorn（ASGI 异步），Starlette 提供 WebSocket |
| 向量检索 | Milvus（pymilvus）+ bge-m3 稠密向量（milvus-model / FlagEmbedding，1024 维） |
| 大模型生成 | OpenAI 兼容协议（openai SDK），接 DashScope / DeepSeek 等 |
| 意图识别 | 关键词规则 + TF-IDF / 逻辑回归（scikit-learn，joblib 产物） |
| 关系存储 | MySQL（PyMySQL）：FAQ 标准答案 + 会话历史 |
| 文档编排 | LangChain（core / community / text-splitters）：加载 / 分块 / 提示词 |
| 鉴权 | PyJWT（HS256） |
| 校验 / 序列化 | Pydantic v2 |
| 离线评测 | RAGAS + datasets |

### 1.3 分层架构

```
API 层 (app/api)        路由 + JWT 鉴权 + 出入参校验 + 统一错误 + trace_id 中间件（保持"薄"）
服务层 (app/services)   QaService 双通道编排、FaqRetriever
RAG 层 (app/rag)        意图、管道、入库、提示词、引用类型、加载器 / 切分器
基础设施层 (app/infra)  MySqlRepo、MilvusVectorIndex、ChatModel
配置 (app/conf)         settings 单例、带 trace_id 的日志
```

分层原则：路由只做协议与校验，业务编排集中在服务层，外部依赖收敛在 infra 层；
各层通过**构造函数注入**接收依赖，便于替换与单元测试。

## 2. 整体架构

```
┌──────────┐   HTTP/WS   ┌────────────────────┐
│  浏览器   │◄──────────►│ FastAPI（中间件+路由）│
└──────────┘             └─────────┬──────────┘
                                   │ JWT 校验后
                          ┌────────▼─────────┐
                          │     QaService     │
                          └───┬───────────┬───┘
                  FAQ 快速通道 │           │ RAG 通道
                  ┌───────────▼──┐   ┌────▼──────────────────┐
                  │ FaqRetriever │   │ RagPipeline           │
                  │ 问句向量 Top1 │   │ 意图→路由→检索→生成    │
                  └───┬──────┬───┘   └───┬─────────┬─────────┘
                      │      │           │         │
                 ┌────▼─┐ ┌──▼─────┐ ┌───▼────┐ ┌──▼─────┐
                 │Milvus│ │ MySQL  │ │ Milvus │ │  LLM   │
                 │_faq  │ │faq_entry│ │_kb     │ │(OpenAI)│
                 └──────┘ └────────┘ └────────┘ └────────┘
```

请求流向：浏览器 → CORS / trace_id 中间件 → 路由（JWT）→ QaService → FAQ / RAG 通道 → MySQL / Milvus / LLM。

## 3. 双通道问答流程

### 3.1 QaService.stream_answer（产出 `(token, is_complete)` 二元组）

1. **FAQ 通道优先**：`FaqRetriever.match(query, trace)`
   - 非法 / 空 query → `(None, False)` → 回兜底话术；
   - 问句向量 Top1 相似度 `>= faq_min_score(0.85)` 且 MySQL 有答案 → 命中：落库 + `yield(answer, True)`；
   - 未命中 / 缺答案 / 异常 → `need_rag=True`，转 RAG（FAQ 故障不阻断问答）。
2. **RAG 通道**：`RagPipeline.stream_answer(query, subject, history, trace)`
   - 逐 token `yield(token, False)`，结束 `yield("", True)`；
   - 完整答案落库（有 session 且非空时）。
3. **寒暄**在路由层正则拦截（你好 / 谢谢 / 在吗…），不进服务，零检索零模型成本。

### 3.2 RAG 管道（RagPipeline）

`意图识别 → 知识域路由 → 向量检索 → 上下文拼装 → 流式生成`

- `intent.predict`：通用知识 / 专业咨询；**两道纠偏**——subject 非空、或判为通用但命中领域词，都强制专业咨询；
- 通用知识：context 留空，直接让模型用自身知识作答；
- 专业咨询：`domain = subject or route_domain(query)` → `index.search` 召回 top-k →
  context 拼成 `[1][2]…` 编号 → `trace.sources` 用同一 `i+1` 编号（正文引用与来源卡片严格对应）；
- `_stream` 重试 3 次、间隔 1s，**已吐字则绝不重试**（避免前端收到重复答案）；彻底失败 → 人工客服兜底话术。

## 4. 意图识别（app/rag/intent.py）

- **关键词规则** `DOMAIN_KEYWORDS`（7 域，词表刻意避免歧义包含）：
  - `is_domain_query`：任一域命中即为真（要不要查库，高召回兜底）；
  - `route_domain`：按命中词数最多的域路由，平票按声明顺序取先者（结果稳定可复现）。
- **轻量分类器**：`TfidfVectorizer(analyzer="char_wb", ngram_range=(1,2), sublinear_tf=True)` +
  `LogisticRegression(max_iter=1000)`；210 条标注数据（通用/专业各 105），8:2 分层切分（`random_state=42`），
  `scripts/train_intent.py` 训练 → `models/intent_clf.joblib`（向量器+分类器整条 Pipeline 一起保存）。
- **模型缺失自动退化为纯关键词规则**，服务照样能跑。
- **不做大模型微调**：小库问题分布集中，字符级 TF-IDF + 线性分类器已足够，训练秒级、推理毫秒级无显存开销。

## 5. 检索与入库

### 5.1 入库（app/rag/ingest.py）

- 加载器：`.md/.txt → TextLoader`，`.pdf → PdfLoader`（PyMuPDF/fitz 逐页取文本），
  `.docx → DocxLoader`（python-docx，按文档顺序遍历段落与表格）；
- 元数据：`subject`（目录名 `{域}_data` 去 `_data`）、`source`（文件名）、`timestamp`、`extension`；
- **单层分块**：`chunk_size=500 / overlap=60`；`.md` 用 MarkdownTextSplitter（尊重标题/列表），
  其它用自研 ChineseTextSplitter（中文标点递归切分 + 合并，标点保留在前块末尾）；
- `chunk_id = doc_{文档序号}_chunk_{块序号}`；单文件加载失败只记日志跳过（能进多少进多少）。

### 5.2 向量索引（app/infra/vector_index.py）

- `MilvusVectorIndex`：6 字段 schema（见 §7），`id = 内容 md5` → 幂等 upsert；
- bge-m3：`BGEM3EmbeddingFunction(return_sparse=False)` 只算稠密向量；`device/fp16 = auto`（有 GPU 用 cuda+fp16）；
- 索引：`IVF_FLAT`，`metric_type=IP`（向量已 L2 归一化，内积 = 余弦相似度），`nlist=128`，查询 `nprobe=10`；
- `search`：query 向量化 → top-k 召回 → `min_score(0.35)` 过滤域外问题 → 返回 `Document`（`metadata.chunk_id = hit.id`）；
- **两个集合共用一份 embedding**（构造函数注入 `embedding_model`，避免重复加载占显存）：
  `zhiqing_kb`（文档分块）+ `zhiqing_faq`（80 条标准问句）。

## 6. 数据库设计（MySQL，utf8mb4）

### 6.1 FAQ 表 `faq_entry`

```sql
CREATE TABLE IF NOT EXISTS faq_entry (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    subject     VARCHAR(64)  NOT NULL DEFAULT '',   -- 学科/来源
    question    VARCHAR(512) NOT NULL,              -- 标准问句
    answer      MEDIUMTEXT   NOT NULL,              -- 标准答案
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_question (question(191))          -- 前 191 字符唯一索引（utf8mb4 安全长度）
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

- `INSERT IGNORE` + 唯一索引 → 重复导入幂等；CSV 列：`学科名称 / 问题 / 答案`；
- `answer_of(question)`：向量命中标准问句后，按问句精确回查答案。

### 6.2 会话表 `chat_turn`

```sql
CREATE TABLE IF NOT EXISTS chat_turn (
    id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    session_id  VARCHAR(64) NOT NULL,
    question    TEXT        NOT NULL,
    answer      MEDIUMTEXT  NOT NULL,
    created_at  DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    is_deleted  TINYINT(1)  NOT NULL DEFAULT 0,     -- 软删除标记
    KEY idx_session (session_id, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

- **软删除**：清空会话只置 `is_deleted=1`，数据资产保留可审计；
- `recent_turns`：按 `id DESC LIMIT N` 取最新 N 轮再反转成旧→新
  （自增 id 严格单调，避免同秒 `created_at` 排序不稳）；只取 `is_deleted=0`。

## 7. Milvus 集合设计

知识库集合 `zhiqing_kb` 与 FAQ 集合 `zhiqing_faq` 共用同一 schema：

```python
schema.add_field("id",           DataType.VARCHAR, max_length=128, is_primary=True)  # 内容 md5
schema.add_field("content",      DataType.VARCHAR, max_length=8192)                  # 分块原文
schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)                    # bge-m3 稠密向量
schema.add_field("subject",      DataType.VARCHAR, max_length=32)                    # 知识域（过滤）
schema.add_field("source",       DataType.VARCHAR, max_length=256)                   # 来源文件名
schema.add_field("timestamp",    DataType.VARCHAR, max_length=64)                    # 入库时间
# 索引：dense_vector / IVF_FLAT / IP / nlist=128
```

## 8. 接口与流式协议

- REST + WebSocket 路由详见 [docs/api.md](api.md)；
- WS 协议：`start → token* → end`（正常）或 `error`（出错）；一条连接可复用多轮；
- **关键时序**：FAQ 命中时 `(answer, True)` 同帧产出，必须**先发 token 再发 end**，否则前端收到 end 即停收、答案丢失；
- WS 鉴权：`?token=`，accept 前用 `verify_token` 校验，失败 `close(4401)`。

## 9. 安全设计

- **JWT 单账号鉴权**（HS256，含 `sub`+`exp`）：REST 走 Bearer、WS 走 `?token=`，缺失/无效 401、WS 4401；
  内部工具不做用户表/注册，账号密码在 `config.ini [auth]`；
- **密钥不进代码库**：`DASHSCOPE_API_KEY / JWT_SECRET / AUTH_PASSWORD` 支持环境变量注入，生产覆盖默认值；
- **注入防护**：SQL 全部参数化（`%s` 占位，pymysql 转义）；Pydantic 边界校验；错误响应不泄漏堆栈；
- **数据隔离**：会话按 `session_id` 隔离；软删除保留数据资产。

## 10. 降级与可靠性

| 故障 | 降级行为 |
| --- | --- |
| FAQ 检索/取答案异常 | 转 RAG，不阻断问答 |
| 向量检索异常 | 空上下文继续生成 |
| 大模型异常 | 重试 3 次（已吐字不重试）→ 人工客服兜底话术 |
| 意图模型文件缺失 | 退化为纯关键词规则 |
| MySQL 读历史失败 | 空历史，退化为单轮问答 |

`trace_id` 由中间件写入 contextvar，全链路日志可据此串联排障。

## 11. 性能与评测

- **延迟设计**：单向量召回（无 rerank）+ FAQ 向量秒回 + GPU fp16，检索段开销低；端到端延迟主要由大模型生成决定；
- **评测**：`scripts/evaluate_ragas.py` 用 RAGAS 四指标（faithfulness / answer_relevancy / context_precision / context_recall）
  对 `data/eval/rag_evaluate_data.json`（30 条，基于真实文档事实构造）打分，裁判模型 `temperature=0`；`--small` 用 5 条先跑通；
- **实测延迟与分数需在 Milvus/MySQL 就绪 + 有效模型 key 下实跑复测**，本文档不内置实测值；
- GPU 沙箱：`device=auto / fp16=auto`，6GB 显存保持 fp16 省显存（详见 README §6）。

## 12. 测试策略

- **33 个 pytest 离线单测**，全部用测试替身（`tests/fakes.py`），不依赖 MySQL / Milvus / 模型权重；
- 覆盖分布：鉴权 4 / API 协议 10 / RAG 管道 6 / 意图路由 4 / FAQ 通道 4 / 入库 2 / 配置与提示词 3；
- API 测试用 `TestClient` + `app.dependency_overrides[get_service]` 注入 `FakeService`，走完整路由/中间件/序列化链路；
- 核心手法：**不测模型输出（不可控），测喂给模型的输入**（`FakeLLM.prompts` 记录实际 prompt）。

## 13. 部署设计

- 启动：`python run.py` 或 `uvicorn app.main:app --port 8080`；
- 容器：`Dockerfile`（python:3.10-slim）+ `docker-compose.yml`（image `zhiqing-system:1.0`）；
- 依赖服务：MySQL + Milvus（外部提供）；`config.ini` 随镜像 COPY，检索/知识域等参数由 ini 提供；
- 配置优先级：**环境变量 / .env > config.ini > 代码默认值**。

## 14. 扩展性设计

- **依赖注入**：index / llm / intent / repo 均可替换，便于测试与演进；
- 检索层：库规模增大后再引入混合检索 / 重排，或接入 ES/BM25 第三路召回；
- 鉴权层：多用户 / 细粒度权限需引入用户表与 RBAC；
- 水平扩容：WebSocket 需粘性会话或共享会话状态与连接路由。