<div align="center">

# 知擎 ZhiQing

**企业内部知识库问答：FAQ 精确命中 + RAG 检索生成，双通道分流**

`FastAPI` · `Milvus + bge-m3` · `MySQL` · `WebSocket 流式` · `JWT`

高频问题不调大模型、直接回标准答案；低频问题走检索增强，每条回答都带可核对的出处

[Get Started](#get-started) · [How It Works](#how-it-works) · [API Reference](#api-reference) · [接口文档](docs/api.md)

![Python](https://img.shields.io/badge/python-3.10-blue)
![Framework](https://img.shields.io/badge/fastapi-0.115-009688)
![Retrieval](https://img.shields.io/badge/vector-Milvus%20%C2%B7%20bge--m3-orange)
![Tests](https://img.shields.io/badge/unit%20tests-33-brightgreen)
![Stream](https://img.shields.io/badge/stream-WebSocket-lightgrey)

</div>

---

面向企业 IT 服务台 / 员工自助问答的知识库服务。把 80 条高频标准问答和 30 份制度文档
放进同一套服务里：**能精确命中的绝不生成，必须生成的绝不编造出处**。

## Contents

- [What it does](#what-it-does)
- [Get Started](#get-started)
- [Why ZhiQing?](#why-zhiqing)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Tests](#tests)
- [Project Layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Data and Disclaimer](#data-and-disclaimer)

## What it does

| 能力 | 说明 | 你可以这样问 |
| --- | --- | --- |
| 🎯 FAQ 秒回 | 问句向量化后与 80 条标准问句算余弦，≥ 0.85 直接取 MySQL 里的标准答案 | VPN 连不上提示 691 |
| 📚 RAG 生成 | 意图识别 → 知识域路由 → 向量召回 → 流式生成，答案带引用编号 | 差旅报销发票有什么要求 |
| 🗂 7 个知识域 | it / oa / security / hr / finance / legal / admin，前端可显式选域收窄检索 | 年假怎么算 |
| 💬 多轮会话 | 按 `session_id` 存 MySQL，取最近 5 轮拼进提示词；清空走软删除 | 那加班呢？ |
| 🌊 流式输出 | WebSocket 逐 token 上屏，`end` 事件带来源与耗时 | — |
| 🔗 引用溯源 | 正文里的 `（来源：[n]）` 与 `sources[n].index` 一一对应，可点开核对原文 | — |
| 🚪 域外兜底 | 召回分块全被 `min_score` 滤掉 → 空上下文 → 明确答"信息不足"并给客服电话 | 今天股市怎么样 |

**数据构成**（都在仓库里，可直接复现）：

- `data/faq/faq_seed.csv`：**80 条**标准问答，覆盖 it 34 / oa 20 / hr 14 / security 12
- `data/kb/`：**30 份**制度文档（Markdown 19 / PDF 6 / DOCX 5），分布在 7 个域目录
- `data/intent/intent_dataset.jsonl`：**210 条**意图标注（通用知识 105 / 专业咨询 105）
- `data/eval/`：**30 条** RAGAS 评测集（另附 5 条小样本先跑通链路）

## Get Started

### 1. 准备依赖服务

| 服务 | 用途 | 默认地址 |
| --- | --- | --- |
| MySQL | `zhiqing_kb.faq_entry`、`chat_turn` | localhost:3306 |
| Milvus | 向量库 `zhiqing.zhiqing_kb` + `zhiqing.zhiqing_faq` | localhost:19530 |

Milvus 只自动建**集合**，不自动建**库**，首次需手动建一次：

```python
from pymilvus import MilvusClient
MilvusClient(uri='http://127.0.0.1:19530').create_database('zhiqing')
```

### 2. 装依赖与模型权重

```bash
pip install -r requirements.txt        # GPU 版 torch 安装见文件内注释
cp .env.example .env                   # 填 MySQL / Milvus / DASHSCOPE_API_KEY / JWT 密钥

# bge-m3 权重（约 2.2GB）不入库，两个来源任选：
# HuggingFace: huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
# ModelScope（国内更快）: modelscope download --model BAAI/bge-m3 --local_dir models/bge-m3
```

### 3. 初始化数据并启动

```bash
python -m scripts.train_intent         # 训练意图分类器 -> models/intent_clf.joblib
python -m scripts.init_faq_store       # 建表 + 导入 80 条 FAQ + 问句向量化（幂等，可重复执行）
python -m scripts.ingest_kb            # 30 份文档单层分块 + 写入 Milvus
python run.py                          # 启动服务，默认 0.0.0.0:8080
```

浏览器打开 <http://127.0.0.1:8080/>，**先登录再提问**（演示账号 `admin / zhiqing@2026`，
来自 `config.ini [auth]`，生产务必改掉）。接口文档在 <http://127.0.0.1:8080/api/v1/docs>。

### 4. 其它入口

| 命令 | 作用 |
| --- | --- |
| `python run.py --reload` | 开发模式热重载 |
| `python -m scripts.cli` | 命令行问答（与 Web 走同一套服务层） |
| `python -m scripts.faq_cli` | 只测 FAQ 通道与阈值 |
| `python -m scripts.evaluate_ragas --small` | RAGAS 评测（5 条小样本，需 key） |
| `python -m scripts.evaluate_ragas` | RAGAS 全量评测（30 条，需 key） |
| `pytest` | 33 个离线单测（不需要数据库与模型权重） |

## Why ZhiQing?

不是"把文档切一切喂给大模型"，是先承认两个现实矛盾，再按它们做取舍：

| 常见做法的问题 | 这里的做法 |
| --- | --- |
| 答案固定的高频问题每次都调大模型 → 贵、慢、还会把标准答案改写错 | FAQ 通道先跑：命中 0.85 阈值直接回 MySQL 标准答案，**零模型调用** |
| 低频问题说法千变万化，纯关键词检索召回不全 | 问句与分块都走向量召回，"年假怎么算"和"入职满一年有几天假"能对上 |
| 小库也堆混合检索 + 交叉编码重排，链路长、延迟高 | 几十份文档规模用 **bge-m3 单向量 + `min_score` 过滤**，一次召回，链路可解释 |
| 判不出领域就硬答，域外问题容易编 | 相关性下限把域外问题滤成**空上下文**，兜底话术明确说"信息不足" |
| 意图识别一上来就微调大模型，在线推理贵 | 关键词规则判"要不要查库 / 路由哪个域"，TF-IDF + 逻辑回归判"通用 / 专业"，毫秒级、无显存开销 |
| 大模型或检索一挂整个接口 500 | 逐层降级：FAQ 异常转 RAG、检索异常空上下文继续生成、模型异常重试后给人工话术 |
| 流式失败重连导致同一答案上屏两遍 | **已吐出过内容就绝不重试**（重试仅在零 token 时触发） |

## How It Works

```
浏览器 ──HTTP / WebSocket──► FastAPI（app/api，JWT 鉴权 · trace_id 链路追踪）
                                │
                        QaService（app/services/qa_service.py）
                                │
                  ┌─────────────┴──────────────┐
                  │                            │
             FAQ 快速通道                   RAG 通道
             ├─ bge-m3 问句向量 Top1        ├─ IntentClassifier：关键词规则 + TF-IDF/LR
             ├─ 相似度 ≥ 0.85 即命中        ├─ route_domain：7 知识域路由
             └─ MySQL faq_entry 取标准答案  ├─ MilvusVectorIndex：bge-m3 单向量召回
                                          │     + min_score=0.35 过滤
                                          └─ PromptLibrary + ChatModel：流式生成
                  │                            │
                  └───────────┬────────────────┘
                              v
              统一会话与流式协议（start → token* → end）
              end 事件带 sources / intent，前端渲染"参考来源"卡片
```

三条主要约束：

1. **先精确、后生成**：FAQ 判不出来才进 RAG，两条通道共用一套会话与流式协议。
2. **答案必须可追溯**：`SourceRef` 在检索完成时就把分块整理成来源列表，正文编号与
   `sources[n].index` 同源生成，不靠事后解析文本。
3. **任何一环失败都不崩**：降级路径见上面的表，兜底答案同样带 `sources` 结构。

## Key Features

- 🎯 **双通道分流**：高频走 FAQ 向量命中，低频走 RAG 生成，成本与召回各取所长
- 🔍 **bge-m3 单向量检索**：1024 维稠密向量（L2 归一化）+ Milvus `IVF_FLAT` 内积 = 余弦相似度
- ✂️ **单层分块**：500 字块 + 60 字重叠；`.md` 用 MarkdownTextSplitter 尊重标题结构，
  其它文本用自研中文递归切分器按标点断句；分块主键取内容 md5，重复入库幂等
- 🧭 **意图路由两道纠偏**：用户显式选域、或判为通用但命中领域词，都强制走检索
- 🔐 **JWT 鉴权**：REST 走 `Authorization: Bearer`，WebSocket 握手无法自定义头 → token 走
  `?token=`，校验失败以 **4401** 关闭连接
- 🧾 **全链路追踪**：所有响应头带 `X-Trace-Id`，出错返回 `{code, message, trace_id}`，可直接查日志
- 🌡 **设备自适应**：`device` / `fp16` 支持 `auto`，有 GPU 就用 CUDA + fp16，6GB 显存也能跑
- 📉 **无模型也能服务**：`models/intent_clf.joblib` 缺失时自动退化为纯关键词规则

## API Reference

完整请求/响应示例见 [docs/api.md](docs/api.md)。所有路径挂在 `/api/v1` 前缀下。

| 方法 | 路径 | 说明 | 鉴权 |
| --- | --- | --- | --- |
| POST | `/auth/login` | 登录换取 JWT | 无 |
| GET | `/health` | 健康检查 | 无 |
| GET | `/kb/sources` | 可用知识域 | Bearer |
| POST | `/sessions` | 创建会话（返回 UUID） | Bearer |
| GET | `/sessions/{id}/messages` | 查会话历史 | Bearer |
| DELETE | `/sessions/{id}/messages` | 清空历史（软删除） | Bearer |
| POST | `/qa/ask` | 非流式：只判 FAQ 是否命中，未命中即转 RAG | Bearer |
| WS | `/qa/stream?token=<JWT>` | 流式问答 | `?token=` |

WebSocket 事件：客户端发 `{query, subject, session_id}`，服务端依次回
`start` → `token`* → `end`（带 `sources` / `intent` / `processing_time`），异常回 `error`。

每条回答的来源结构（FAQ 命中时 `type=faq`、`subject=FAQ`，与文档来源共用一种卡片）：

```json
{ "type": "end", "sources": [
  { "index": 1, "title": "网络与VPN故障排查手册.md", "subject": "it",
    "chunk_id": "doc_0_chunk_0", "snippet": "…", "type": "doc",
    "source": "网络与VPN故障排查手册.md" } ] }
```

## Configuration

配置分两层：`.env`（密钥与连接串，优先级高）与 `config.ini`（分节参数）。常用项：

| 位置 | 键 | 默认 | 说明 |
| --- | --- | --- | --- |
| `.env` | `DASHSCOPE_API_KEY` | 空 | **只从环境变量读**，不写进配置文件；缺失时只告警不崩溃 |
| `config.ini [llm]` | `model` / `base_url` | `deepseek-chat` / OpenAI 兼容端点 | 生成模型走 OpenAI 兼容协议，换服务商只改这两项 |
| `[retrieval]` | `faq_min_score` | `0.85` | FAQ 命中阈值，调低会误命中，调高则更多问题走 RAG |
| `[retrieval]` | `min_score` | `0.35` | 分块相关性下限，域外问题靠它被滤成空上下文 |
| `[retrieval]` | `chunk_size` / `chunk_overlap` / `retrieval_k` | `500` / `60` / `5` | 单层分块与召回条数 |
| `[milvus]` | `device` / `fp16` / `embed_batch` | `auto` / `auto` / `16` | 显存小可把 `embed_batch` 调小 |
| `[app]` | `valid_sources` | 7 个域 | 知识域白名单 |
| `[auth]` | `username` / `password` / `secret_key` / `token_expire_minutes` | — | 支持环境变量覆盖，生产务必改默认口令 |

## Evaluation

`scripts/evaluate_ragas.py` 用 RAGAS 四指标对 `data/eval/rag_evaluate_data.json`（**30 条**
基于真实文档事实构造的 question / context / answer / ground_truth）打分，裁判模型
`temperature=0` 尽量可复现：

| 指标 | 归因 | 测什么 |
| --- | --- | --- |
| faithfulness | 生成 | 答案论断是否都有上下文依据（反幻觉） |
| answer_relevancy | 生成 | 答案是否切题、有无答非所问 |
| context_precision | 检索 | 有用片段是否排在前面（信噪比） |
| context_recall | 检索 | 标准答案信息是否被检索片段覆盖（漏检） |

`--small` 先用 5 条小样本跑通流程、省 token，确认无误再跑全量。

> 本 README **不预填分数**：跑分需要有效的 `DASHSCOPE_API_KEY` 与就绪的 MySQL / Milvus，
> 请按上面的命令实跑复现后再引用数字。FAQ 阈值、`min_score`、`retrieval_k` 的改动都会
> 反映在 `context_precision` / `context_recall` 上，评测脚本就是用来验收这些调整的。

## Tests

```bash
pytest                 # 33 个离线用例，不需要数据库与模型权重
```

单测全部使用测试替身（`tests/fakes.py` 的 FakeIndex / FakeLLM / FakeRepo），
**不需要 MySQL、Milvus 与模型权重**，随时可跑。覆盖分布：

| 文件 | 用例数 | 覆盖 |
| --- | --- | --- |
| test_api.py | 10 | 健康检查 / trace_id / 知识域 / 会话 / 首页 / 问答三分支 / WS 协议 |
| test_pipeline.py | 6 | 通用跳过检索 / 专业上下文 / 历史注入 / 规则路由 / 故障兜底 / 历史格式化 |
| test_auth.py | 4 | JWT 登录 / 错误密码 / 缺 token / 无效 token |
| test_faq_retriever.py | 4 | 阈值命中 / 低于阈值转 RAG / 非法输入 / 缺答案降级 |
| test_intent.py | 4 | 关键词域路由（it / finance / admin）/ 无关问句不命中 |
| test_settings_and_prompts.py | 3 | 配置冒烟 / 检索不变量 / 提示词渲染 |
| test_ingest.py | 2 | 分块元数据完整性 / 空目录 |

## Project Layout

```
├─ app/
│  ├─ main.py              FastAPI 装配（中间件 / 路由 / 静态页 / 异常处理）
│  ├─ api/                 routes · auth(JWT) · schemas · middleware(trace_id) · errors
│  ├─ conf/                settings(.env + config.ini) · logging
│  ├─ infra/               mysql(FAQ/会话仓储) · vector_index(Milvus+bge-m3) · llm(流式客户端)
│  ├─ rag/                 ingest · pipeline · intent · prompts · types(SourceRef/RagTrace)
│  │                       loaders(pdf,docx) · splitters(中文递归切分)
│  └─ services/            qa_service(FAQ+RAG 编排) · faq(FAQ 向量检索器)
├─ scripts/                train_intent · init_faq_store · ingest_kb · cli · faq_cli · evaluate_ragas
├─ tests/                  33 个离线单测（fakes.py 提供测试替身）
├─ web/                    前端页面（原生 HTML + JS，含登录与来源卡片）
├─ data/                   faq(80) · kb(7 域 30 份) · intent(210) · eval(30)
├─ models/                 bge-m3 权重 + intent_clf.joblib（本地放置，已 gitignore）
├─ docs/                   api.md · architecture.md · requirements.md · 技术细节.md
├─ config.ini / .env.example
├─ run.py                  启动入口（默认 8080）
├─ Dockerfile / docker-compose.yml
└─ requirements.txt
```

分块/检索/降级的完整设计取舍、已知限制与扩展方向写在
[docs/技术细节.md](docs/技术细节.md)，整体架构见 [docs/architecture.md](docs/architecture.md)。

## Troubleshooting

**启动就报连不上 Milvus / 集合不存在**
Milvus 的 `zhiqing` **库**要手动建一次（见 Get Started 第 1 步）；集合会自建。检索层强依赖
Milvus，不可用时服务无法正常提供问答。

**`No module named 'docx'` / `'pymilvus'`**
`pip install -r requirements.txt` 未装全，或装进了别的解释器。

**回答全是"服务暂时不可用，请联系人工客服"**
没配 `DASHSCOPE_API_KEY`（启动时会打印告警）。此时 **FAQ 通道仍然可用**，
可用 `python -m scripts.faq_cli` 验证命中链路。

**明明该命中 FAQ 却走了 RAG**
用 `python -m scripts.faq_cli` 看实际相似度。问句写法与标准问句差异大时相似度会低于 0.85；
先补 FAQ 语料，再考虑调阈值。

**RAG 答得比原文含糊 / 引用错位**
先确认 `python -m scripts.ingest_kb` 跑过且没有 `目录不存在，跳过`；再检查
`config.ini [app] valid_sources` 与前端选的域是否一致。

**问"今天股市怎么样"却硬答了**
`min_score=0.35` 会把域外分块全滤掉、走"信息不足"兜底。若仍生成，多半是把
`[retrieval] min_score` 调低了，或 `[llm]` 被判为通用知识——这是轻量线性模型的已知局限，
靠关键词规则纠偏。

**6GB 显存跑 bge-m3 吃力**
保持 `device = auto`、`fp16 = auto`，把 `embed_batch` 从 16 调小到 4～8。

**WebSocket 连上就被关（4401）**
token 缺失或过期。握手阶段无法自定义请求头，只能带 `?token=<JWT>`；重新登录取新 token。

## Data and Disclaimer

- `data/` 下的 FAQ、制度文档、标注与评测集均为**演示用整理数据**，不对应任何真实企业内容；
  制度文档由公开科普资料改写整理。
- 服务默认只监听 `0.0.0.0:8080`，**未实现 HTTPS、多用户与细粒度权限、限流与审计日志**，
  定位是单机可跑通的内部工具原型；对外部署请自行加反向代理、TLS 与访问控制。
- 默认口令 `zhiqing@2026` 与 JWT 密钥是开发占位值，**生产必须替换**；
  密钥支持环境变量注入，`DASHSCOPE_API_KEY` 只从环境变量读取、不落配置文件。
- 模型权重（`models/bge-m3`，约 2.2GB）已在 `.gitignore` 中排除，需自行下载。
- License：本仓库**尚未指定开源协议**，如需对外授权请先补 `LICENSE`。

---

如果这个项目对你有参考价值，欢迎给个 ⭐；提 issue 前请先跑 `pytest` 并说明卡在 Get Started 的哪一步。
