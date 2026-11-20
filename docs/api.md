# 知擎 API 文档（v1.0）

- Base URL：`http://127.0.0.1:8080`
- 接口前缀：`/api/v1`
- 交互式文档：`/api/v1/docs`
- 所有响应头带 `X-Trace-Id`（客户端也可主动传入，便于串联日志）
- 鉴权：除 `/health`、`/auth/login` 外均需 JWT。REST 带 `Authorization: Bearer <token>`；
  WebSocket 握手无法自定义头，token 走 `?token=<token>`。

## 统一错误格式

业务异常（ApiError）与未预期异常统一返回：

```json
{ "code": 400, "message": "错误说明", "trace_id": "9f2c1a7b3d4e5f60" }
```

| HTTP | 场景 | 响应体 |
| --- | --- | --- |
| 400 | 参数不合法（业务异常） | `{code, message, trace_id}` |
| 401 | 登录失败（账号/密码错） | `{code:401, message, trace_id}` |
| 401 | 缺少/无效 Bearer token | `{ "detail": "..." }`（HTTPException 默认格式，带 `WWW-Authenticate: Bearer`） |
| 422 | Pydantic 校验失败（如 query 为空） | FastAPI 默认校验错误体 |
| 500 | 服务内部错误（详情看日志） | `{code:500, message, trace_id}` |

---

## 1. 登录换取 JWT

`POST /api/v1/auth/login`（无需鉴权）

请求：

```json
{ "username": "admin", "password": "zhiqing@2026" }
```

响应：

```json
{ "access_token": "<JWT>", "token_type": "bearer" }
```

- 账号密码来自 `config.ini [auth]`（生产用环境变量 `AUTH_USERNAME/AUTH_PASSWORD` 覆盖）；
- token 为 HS256 JWT，含 `sub`（用户名）与 `exp`（过期时间，默认 1440 分钟）；
- 之后 REST 请求带 `Authorization: Bearer <access_token>`，WebSocket 用 `?token=<access_token>`。

## 2. 健康检查

`GET /api/v1/health`（无需鉴权）

```json
{ "status": "healthy", "app": "知擎" }
```

> 探针可用更短的 `GET /health`。

## 3. 知识域列表

`GET /api/v1/kb/sources`（Bearer）

```json
{ "sources": ["it", "oa", "security", "hr", "finance", "legal", "admin"] }
```

## 4. 会话

### 4.1 创建会话

`POST /api/v1/sessions`（Bearer）

```json
{ "session_id": "0f3a6c1e-8b2d-4f5a-9c1d-7e2b3a4c5d6e" }
```

### 4.2 查询历史

`GET /api/v1/sessions/{session_id}/messages`（Bearer）

```json
{ "session_id": "0f3a...",
  "history": [ { "question": "VPN 连不上怎么办", "answer": "报错 691 表示账号或密码错误……" } ] }
```

### 4.3 清空历史（软删除）

`DELETE /api/v1/sessions/{session_id}/messages`（Bearer）

```json
{ "status": "success", "message": "会话历史已清空" }
```

## 5. 非流式问答

`POST /api/v1/qa/ask`（Bearer）

请求：

```json
{ "query": "年假怎么计算", "subject": "hr", "session_id": "可选" }
```

响应（FAQ 命中，带引用来源）：

```json
{ "answer": "入职满 1 年 5 天、满 3 年 7 天、满 5 年 10 天……", "is_streaming": false,
  "session_id": "0f3a...", "processing_time": 0.012,
  "sources": [
    { "index": 1, "title": "年假怎么计算？", "subject": "FAQ",
      "chunk_id": "faq:年假怎么计算？", "snippet": "入职满 1 年 5 天……",
      "type": "faq", "source": "" }
  ] }
```

`sources` 字段说明：

| 字段 | 含义 |
| --- | --- |
| `index` | 与答案里 `（来源：[n]）` 的编号一一对应 |
| `title` | 人类可读标题（md 取首行标题，PDF/DOCX 取首行文本） |
| `subject` | 知识域（it/oa/security/hr/finance/legal/admin）；FAQ 命中时为 `FAQ` |
| `chunk_id` | 分块 ID `doc_i_chunk_j`（可回查原文）；FAQ 命中时为 `faq:<问句前缀>` |
| `snippet` | 前 160 字的片段摘要，前端直接展示 |
| `type` | `doc`（文档检索）/ `faq`（标准问答） |
| `source` | 原文文件名（FAQ 命中时为空） |

响应（需要 RAG）：

```json
{ "answer": "该问题需要检索知识库，请使用流式接口 /qa/stream 获取回答",
  "is_streaming": true, "session_id": "0f3a...", "processing_time": 0.008 }
```

> 该接口只做"要不要走 RAG"的快速判定，真正的生成走 WebSocket。

## 6. 流式问答（WebSocket）

连接：`ws://127.0.0.1:8080/api/v1/qa/stream?token=<JWT>`

> token 无效/过期：服务端在 accept 前以 **4401** 关闭连接，不进入问答循环。

客户端发送：

```json
{ "query": "VPN 连不上提示 691 怎么办", "subject": "it", "session_id": "0f3a..." }
```

服务端事件序列：

```json
{ "type": "start", "session_id": "0f3a...", "app": "知擎" }
{ "type": "token", "token": "报错 691", "session_id": "0f3a..." }
{ "type": "token", "token": "表示账号或密码错误", "session_id": "0f3a..." }
{ "type": "end", "session_id": "0f3a...", "is_complete": true, "processing_time": 0.72,
  "intent": "专业咨询",
  "sources": [
    { "index": 1, "title": "网络与VPN故障排查手册.md", "subject": "it",
      "chunk_id": "doc_0_chunk_0", "snippet": "报错 691 表示账号或密码错误……",
      "type": "doc", "source": "网络与VPN故障排查手册.md" }
  ] }
```

> `end` 事件里的 `sources` 与答案正文中的 `（来源：[1]）` 严格对应；`intent` 为本次意图分类结果。

异常：

```json
{ "type": "error", "error": "错误说明" }
```

约定：

- 同一个连接可连续发多轮 query；
- `subject` 为可选知识域过滤（兼容旧字段 `source_filter`）；
- 寒暄（你好/谢谢等）也走完整协议：start → 单条 token → end；
- FAQ 命中时答案与结束标志同帧产出，服务端**先发 token 再发 end**（顺序不可颠倒，否则前端丢答案）；
- 结束后服务端不断开连接，可直接发下一条。

## 7. 前端联调提示

- 先 `POST /api/v1/auth/login` 拿 token 存起来；REST 带 Bearer 头、WS 带 `?token=`；
- 再 `POST /api/v1/sessions` 拿 `session_id`，之后每次提问都带上，形成多轮上下文；
- 收到 `is_streaming: true` 再建 WebSocket，避免为 FAQ 命中建立无意义连接；
- 逐 token 渲染，`end` 到达后再做落库/统计；遇到 401 清 token 重新登录。