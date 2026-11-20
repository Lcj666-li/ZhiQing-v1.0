# 知擎 · ZhiQing v1.0

> 面向企业内部知识库的智能问答服务：**FAQ 精确问答 + RAG 检索增强**双通道，
> 支持多轮会话、WebSocket 流式输出、知识域过滤与 JWT 鉴权。
> 知识库规模小（几十份制度文档），采用 **bge-m3 单向量直接召回**，链路短、延迟低、单机可跑通。

- 语言 / 框架：Python 3.10 · FastAPI · WebSocket · LangChain
- 存储：MySQL（FAQ 标准答案与会话历史）· Milvus（向量库）
- 模型：bge-m3（稠密向量，1024 维）· TF-IDF + 逻辑回归（意图识别）· Qwen/DeepSeek（OpenAI 兼容协议生成）
- 鉴权：JWT（HS256），REST 走 Bearer、WebSocket 走 `?token=`

---

## 1. 它解决什么问题

企业知识问答的两个现实矛盾：

1. **高频问题答案固定，却每次都调大模型** → 贵、慢、还可能改写错。
2. **低频问题说法千变万化，纯关键词检索召回不全** → 答不上来。

知擎的做法是**先精确、后生成**：FAQ 通道把用户问题向量化，与 80 条标准问句算相似度，
命中阈值就直接回 MySQL 里的标准答案；判不出来再进 RAG 通道，
用「意图识别 → 知识域路由 → 向量检索 → 流式生成」拿到有出处的回答。两条通道共用一套会话与流式协议。

**业务场景**：企业内部 IT 服务台 / 员工自助问答。

- 知识域（7 个）：`it`（账号/网络/设备）、`oa`（考勤/请假/审批）、`security`（信息安全/合规）、
  `hr`（入离职/福利/绩效）、`finance`（报销/发票/预算）、`legal`（合同/用印/知产）、`admin`（会议室/资产/门禁）
- 数据构成：**80 条标准 FAQ**（向量命中秒回）+ **30 份制度/手册文档**（Markdown 19 / PDF 6 / DOCX 5，向量检索后生成）
- 典型问题："VPN 连不上提示 691"、"年假怎么算"、"差旅报销发票要求"、"预订会议室"

```
浏览器 ──HTTP/WS──► FastAPI（app/api，JWT 鉴权）
                      │
                QaService（app/services/qa_service.py）
                      │
        ┌─────────────┴──────────────┐
        │                            │
   FAQ 快速通道                   RAG 通道
   ├─ bge-m3 问句向量 Top1        ├─ IntentClassifier：关键词规则 + TF-IDF/LR
   ├─ 相似度 >= 0.85 命中         ├─ route_domain：7 知识域路由
   └─ MySQL faq_entry 取标准答案  ├─ MilvusVectorIndex：bge-m3 单向量召回 + min_score 过滤
                                 └─ PromptLibrary + ChatModel：流式生成
```

---

## 2. 目录结构

```
ZhiQing-v1.0/
├── app/
│   ├── main.py                 # FastAPI 装配（中间件 / 路由 / 静态页 / 异常处理）
│   ├── api/
│   │   ├── routes.py           # REST + WebSocket 路由
│   │   ├── auth.py             # JWT 签发 / 校验 / Bearer 依赖
│   │   ├── schemas.py          # 出入参模型（Pydantic）
│   │   ├── middleware.py       # trace_id 链路追踪
│   │   └── errors.py           # 统一错误响应
│   ├── conf/
│   │   ├── settings.py         # 配置单例（.env + config.ini）
│   │   └── logging.py          # 带 trace_id 的日志
│   ├── infra/                  # 外部依赖适配层
│   │   ├── mysql.py            # 仓储：FAQ 表 / 会话表
│   │   ├── vector_index.py     # Milvus 集合管理 + bge-m3 单向量检索
│   │   └── llm.py              # OpenAI 兼容流式客户端
│   ├── rag/
│   │   ├── ingest.py           # 文档加载 + 单层分块
│   │   ├── pipeline.py         # RAG 主流程（意图→路由→检索→生成）
│   │   ├── intent.py           # 关键词规则 + TF-IDF/LR 意图识别
│   │   ├── prompts.py          # 提示词库
│   │   ├── types.py            # SourceRef / RagTrace（引用溯源）
│   │   ├── loaders/            # pdf / docx
│   │   └── splitters/          # 中文递归切分
│   └── services/
│       ├── qa_service.py       # FAQ + RAG 编排
│       └── faq.py              # FAQ 向量检索器
├── scripts/                    # train_intent / init_faq_store / ingest_kb / cli / faq_cli / evaluate_ragas
├── tests/                      # 33 个 pytest 离线单测
├── web/                        # 前端页面（原生 HTML + JS，含登录）
├── data/                       # faq(80) / kb(7域30份) / intent(210) / eval(30)
├── models/                     # bge-m3 权重 + intent_clf.joblib（本地放置，不入库，见 3.2）
├── docs/                       # 架构 / 接口文档
├── config.ini / .env           # 配置
├── run.py                      # 启动入口
└── requirements.txt
```

---

## 3. 快速开始

### 3.1 依赖服务

| 服务 | 用途 | 默认 |
| --- | --- | --- |
| MySQL | `zhiqing_kb.faq_entry`、`chat_turn` | localhost:3306 |
| Milvus | 向量库 `zhiqing.zhiqing_kb` + `zhiqing.zhiqing_faq` | localhost:19530 |

Milvus 首次使用需要建库：

```python
from pymilvus import MilvusClient
MilvusClient(uri='http://127.0.0.1:19530').create_database('zhiqing')
```

### 3.2 安装与初始化

```bash
pip install -r requirements.txt        # GPU 版 torch 安装见 requirements.txt 内注释
cp .env.example .env                   # 填 MySQL / Milvus / DASHSCOPE_API_KEY / JWT

# 下载 bge-m3 模型到 models/bge-m3（权重不入库，两个来源任选）
# HuggingFace: huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
# ModelScope（国内更快）: modelscope download --model BAAI/bge-m3 --local_dir models/bge-m3

python -m scripts.train_intent         # 训练意图分类器 -> models/intent_clf.joblib
python -m scripts.init_faq_store       # 建表 + 导入 80 条 FAQ + 问句向量化（幂等）
python -m scripts.ingest_kb            # 30 份文档单层分块 + 写入 Milvus
python run.py                          # 启动服务，默认 8080
```

浏览器打开 <http://127.0.0.1:8080/>，**先用演示账号登录**（`admin / zhiqing@2026`，见 `config.ini [auth]`，
生产务必修改），再提问。接口文档 <http://127.0.0.1:8080/api/v1/docs>。

### 3.3 其它入口

| 命令 | 作用 |
| --- | --- |
| `python run.py --reload` | 开发模式热重载 |
| `python -m scripts.cli` | 命令行问答（与 Web 同一套服务层） |
| `python -m scripts.faq_cli` | 只测 FAQ 通道与阈值 |
| `python -m scripts.evaluate_ragas --small` | RAGAS 评测（5 条小样本先跑通，需 key） |
| `python -m scripts.evaluate_ragas` | RAGAS 全量评测（30 条，需 key） |
| `pytest` | 33 个离线单测 |

---

## 4. 接口一览

完整说明见 [docs/api.md](docs/api.md)。

| 方法 | 路径 | 说明 | 鉴权 |
| --- | --- | --- | --- |
| POST | `/api/v1/auth/login` | 登录换取 JWT | 无 |
| GET | `/api/v1/health` | 健康检查 | 无 |
| GET | `/api/v1/kb/sources` | 可用知识域 | Bearer |
| POST | `/api/v1/sessions` | 创建会话 | Bearer |
| GET | `/api/v1/sessions/{id}/messages` | 查会话历史 | Bearer |
| DELETE | `/api/v1/sessions/{id}/messages` | 清空（软删除） | Bearer |
| POST | `/api/v1/qa/ask` | 非流式（快速判定是否走 RAG） | Bearer |
| WS | `/api/v1/qa/stream?token=<JWT>` | 流式问答（`start` → `token*` → `end`） | `?token=` |

- 除 `/health`、`/auth/login` 外均需 JWT：REST 带 `Authorization: Bearer <token>`；
  WebSocket 握手无法自定义头，token 走 `?token=`，校验失败以 **4401** 关闭连接。
- 所有响应头带 `X-Trace-Id`，出错时返回 `{code, message, trace_id}`，可直接用于日志检索。

### 4.1 回答可追溯（sources / 引用回链）

每条回答都带结构化来源，答案正文里的 `（来源：[n]）` 与 `sources[n].index` 一一对应：

```json
{ "type": "end", "sources": [
  { "index": 1, "title": "网络与VPN故障排查手册.md", "subject": "it",
    "chunk_id": "doc_0_chunk_0", "snippet": "…", "type": "doc",
    "source": "网络与VPN故障排查手册.md" } ] }
```

- 后端：`app/rag/types.py` 定义 `SourceRef / RagTrace`，管道检索完成后把分块整理成来源；
- 接口：`/qa/ask` 返回 `sources`，WebSocket 的 `end` 事件带 `sources / intent`；
- FAQ 命中时来源标 `type=faq`、`subject=FAQ`，与文档来源共用一种卡片结构；
- 前端：答案下方渲染"参考来源"卡片（编号 + 标题 + 知识域 + 片段），可直接核对原文。

---

## 5. 关键设计

**1）双通道分流，成本优先**
FAQ 向量命中即秒回标准答案，高频问题不产生大模型调用；RAG 只在 FAQ 判不出来时介入。

**2）单层分块**
知识库规模小（30 份文档、几百分块），500 字块 + 60 字重叠即可兼顾检索精度与语义完整；
`.md` 用 MarkdownTextSplitter 尊重标题结构，其它文本用自研中文递归切分器按标点断句。
分块主键取内容 md5，重复入库幂等（upsert 覆盖）。

**3）bge-m3 单向量检索**
稠密向量（1024 维、L2 归一化）+ Milvus `IVF_FLAT` 内积检索（内积 = 余弦相似度），
一次召回 top-k，再用相关性下限 `min_score=0.35` 过滤域外问题（域外被滤成空上下文 → 走"信息不足"兜底）。
小库无需混合检索/交叉编码重排这类大规模重武器，链路短、可解释。

**4）意图路由：规则为主 + 轻量分类，不做大模型微调**
7 域关键词规则判"要不要查库 / 路由到哪个域"（`is_domain_query` / `route_domain`）；
TF-IDF（字符 1-2 gram）+ 逻辑回归（210 条标注数据离线训练）判"通用知识 / 专业咨询"，
在线推理毫秒级、无显存开销。模型文件缺失时自动退化为纯关键词规则，服务照样能跑。
两道纠偏：用户显式选域、或判为通用但命中领域词，都强制走检索。

**5）多轮会话**
历史按 `session_id` 存 MySQL，取最近 5 轮拼进提示词；清空走软删除，保留数据资产。

**6）JWT 鉴权**
单账号设计（内部工具，不做用户表/注册），`/auth/login` 签发 HS256 token，
REST 走 Bearer、WebSocket 走 `?token=`，过期/无效返回 401 / 4401。密钥与口令支持环境变量注入。

**7）全链路降级**
FAQ 异常 → 转 RAG；检索异常 → 空上下文继续生成；模型异常 → 重试 3 次后给人工兜底话术。
**流式重试关键约束**：一旦已经吐出过内容就绝不重试，避免前端收到重复答案。

---

## 6. 性能与评测

### 6.1 评测方法（RAGAS）

`scripts/evaluate_ragas.py` 用 RAGAS 四指标对 `data/eval/rag_evaluate_data.json`（**30 条**，
基于真实文档事实构造 question/context/answer/ground_truth）打分，裁判模型 `temperature=0` 尽量可复现：

| 指标 | 归因 | 测什么 |
| --- | --- | --- |
| faithfulness | 生成 | 答案论断是否都有上下文依据（反幻觉） |
| answer_relevancy | 生成 | 答案是否切题、有无答非所问 |
| context_precision | 检索 | 有用片段是否排在前面（信噪比） |
| context_recall | 检索 | 标准答案信息是否被检索片段覆盖（漏检） |

`--small` 先用 5 条小样本跑通流程、省 token，确认无误再跑全量。

### 6.2 延迟设计

单向量召回（无 rerank）+ FAQ 向量秒回 + GPU fp16，使检索段开销很低；
**端到端延迟主要由大模型生成决定**。`config.ini [milvus]` 可调：

```ini
device = auto      # auto（有 GPU 就用 cuda）/ cuda / cpu
fp16 = auto        # auto（有 GPU 就开）/ true / false，6GB 显存建议保持 auto 省显存
embed_batch = 16   # 每次向量化/写入的分块数，显存小可调小
```

### 6.3 GPU 沙箱（可选，不污染原环境）

```powershell
# 1) 建独立 venv 沙箱（继承现有环境依赖，不动原环境）
python -m venv --system-site-packages .venv-gpu
# 2) 装 CUDA 版 torch
.venv-gpu\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
# 3) 用沙箱启动
.venv-gpu\Scripts\python.exe run.py
```

> **说明**：RAGAS 分数与端到端延迟依赖有效的 `DASHSCOPE_API_KEY` 与就绪的 MySQL/Milvus，
> 可用 `scripts/evaluate_ragas.py` 实跑复现；未配置 key 时问答走兜底话术。

---

## 7. 测试

```bash
pytest                 # 33 个离线用例
```

单测全部使用测试替身（`tests/fakes.py`），不需要 MySQL、Milvus 与模型权重，随时可跑。覆盖分布：

| 文件 | 用例数 | 覆盖 |
| --- | --- | --- |
| test_auth.py | 4 | JWT 登录 / 错误密码 / 缺 token / 无效 token |
| test_api.py | 10 | 健康检查 / trace_id / 知识域 / 会话 / 首页 / 问答三分支 / WS 协议 |
| test_pipeline.py | 6 | 通用跳过检索 / 专业上下文 / 历史注入 / 规则路由 / 故障兜底 / 历史格式化 |
| test_intent.py | 4 | 关键词域路由（it/finance/admin）/ 无关问句不命中 |
| test_faq_retriever.py | 4 | 阈值命中 / 低于阈值转 RAG / 非法输入 / 缺答案降级 |
| test_ingest.py | 2 | 分块元数据完整性 / 空目录 |
| test_settings_and_prompts.py | 3 | 配置冒烟 / 检索不变量 / 提示词渲染 |

---

## 8. 已知限制

- 检索链路强依赖 Milvus：不可用时服务无法启动（可改为懒加载 + 只读降级）。
- 单实例部署，WebSocket 未做粘性会话；水平扩容需引入共享会话状态与连接路由。
- 单账号 JWT（内部工具够用）；多用户/细粒度权限需引入用户表与 RBAC。
- 意图分类器为轻量线性模型，复杂/跨域问题可能误判，靠关键词规则兜底纠偏。

---

## 9. 扩展方向

- 检索层：库规模增大后再引入混合检索/重排；接入 ES/BM25 做第三路召回。
- 生成层：答案引用片段高亮；语义级答案缓存降低重复生成。
- 工程层：请求限流、异步任务化入库、Prometheus 指标、Docker Compose 一键起 MySQL/Milvus。