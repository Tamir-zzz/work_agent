# Memory+Planning Agent & 智舱指令 Agent

一个用于「大模型应用 / Agent 开发」求职面试的演示项目，包含两大子系统：

- **记忆 + 自主规划 Agent**：分级记忆（短期+长期）、冲突去重、LRU 遗忘、规划-执行-反思闭环、ReAct 工具调用、流式输出。
- **智舱指令 Agent（Cockpit）**：语音/文字 → 意图理解 → 结构化指令 → 安全门分级 → 记忆 → 执行，演示如何把一个通用 Agent 做成"车载域控"里的受控指令执行器。

核心卖点：不是套 LangChain 全家桶，而是**自研可讲解的编排 + 可落地的工程取舍**，每块都经得起追问。

## ✨ 亮点速览

| 能力 | 说明 | 面试可讲深度 |
|---|---|---|
| 分级记忆 | 短期（内存有界）+ 长期（真实 Embedding 存 SQLite） | 高 |
| **按记忆类型选择去重策略** | 事实/经验：相似度合并（累积）；**偏好/目的地：同一主题 latest-wins 覆盖** | 高（亮点） |
| LRU 遗忘 | 按最近访问时间裁剪，记忆规模可控 | 高 |
| 自主规划闭环 | Planner 拆任务树 → Executor 执行 → Reflector 抽经验沉淀长期记忆 | 高 |
| 规划的记忆续接 | 含糊后续目标（"再规划一下"）也能召回前文续上主题 | 高 |
| ReAct + 工具调用 | 解析 tool_calls → 执行 → 回填 ToolMessage，含流式工具循环 | 高 |
| 安全门分级（智舱） | INFO / COMFORT / CONTROLLED / FORBIDDEN 四级裁决 + 危险词兜底 | 高（差异化） |
| 受控指令二次确认 | 需确认的指令**精确暂存、确认后重放**，不重新让模型猜 | 高（差异化） |
| 流式输出 | LLM stream_turn + Streamlit write_stream（底层 SSE） | 中高 |

## 🏗 技术栈

- **LLM / Embedding**：OpenAI 兼容接口（OpenAI / 通义 / DeepSeek / ollama 等均可），含 Function Calling 与 `/embeddings`；Embedding 不可用时自动降级为哈希向量（离线可跑）。
- **编排**：自研轻量 ReAct 主循环 + 规划-执行-反思，不依赖 LangChain，便于面试讲解。
- **记忆**：真实 Embedding（向量持久化 SQLite）+ 结构化元数据 + 分级/去重/遗忘。
- **智舱指令层**：Pydantic 结构化命令 + 动作 `Action` 白名单 + 安全门 + 模拟 `VehicleState`。
- **UI 后端**：Streamlit（单服务，SSE 流式）；此外保留 FastAPI + 原生 ChatUI 入口。

## 📁 目录结构

```
agent/
├── app.py                # 命令行入口（记忆+规划 Agent）
├── app_ui.py             # Streamlit：记忆规划 UI（8501）
├── app_cockpit.py        # Streamlit：智舱 UI（8502）
├── requirements.txt
├── .env.example          # 模型 / Embedding 端点配置样例
├── config/settings.py    # 全局配置
├── core/
│   ├── llm.py            # LLM 客户端（OpenAI 兼容，含流式 + tool_calls）
│   ├── models.py         # Pydantic 数据模型
│   ├── embedding.py      # 真实 Embedding + 哈希降级
│   └── agent.py          # ReAct 主循环（记忆/工具/规划/流式）
├── memory/
│   ├── base.py           # 记忆抽象接口
│   ├── short_term.py     # 短期记忆（进程内存、有界）
│   ├── long_term.py      # 长期记忆（向量 + SQLite，线程安全）
│   └── manager.py        # 分级记忆管理器 [面试亮点]
├── planner/
│   ├── planner.py        # 目标 → 任务树（含记忆召回续接）
│   ├── executor.py       # 规划-执行-反思-修订闭环 [面试亮点]
│   └── reflector.py      # 反思：抽取 lesson/fact 沉淀长期记忆
├── tools/
│   ├── base.py           # 工具接口 + 注册中心
│   └── builtin.py        # 内置工具示例
├── cockpit/              # 智舱指令层
│   ├── schemas.py        # VehicleCommand + SafetyLevel + SafetyVerdict（Action 白名单）
│   ├── safety.py         # SafetyGate 安全门（分级 + 危险词）
│   ├── state.py          # 模拟 VehicleState（温度/车窗/车速…）
│   ├── commands.py       # CommandExecutor：指令落地到状态
│   └── controller.py     # 意图+记忆+安全门+执行编排，含 pending 二次确认
├── api/
│   ├── main.py           # FastAPI：/chat /memories /plan /forget + ChatUI
│   └── schemas.py
├── static/index.html     # 原生 ChatUI（记忆库 + 任务进度可视化）
└── scripts/demo.py       # 记忆"二次受益"演示
```

## 🚀 环境要求

- Python **3.10+**（代码使用 `X | None`、`list[float]` 等新语法）
- 在 `.env`（复制自 `.env.example`）配置 LLM / Embedding 端点。**未配置时 Embedding 自动降级为哈希向量**，仍可运行。

```bash
cp .env.example .env      # 填入 API Key 与端点
pip install -r requirements.txt
```

## ▶️ 快速开始

### 方式一：命令行（记忆 + 规划 Agent）

```bash
python app.py "你好"
python app.py "帮我规划如何学习 Agent 开发" --plan
```

### 方式二：Stremlit UI（推荐演示）

```bash
# 记忆 + 规划 Agent
streamlit run app_ui.py --server.port 8501

# 智舱指令 Agent
streamlit run app_cockpit.py --server.port 8502
```

浏览器打开对应端口体验。智舱页支持：语音外的文字指令、车况面板、安全门裁决可视化、记忆库、受控指令二次确认。

### 方式三：FastAPI + 原生 ChatUI（可选）

```bash
uvicorn api.main:app --reload
# 打开 http://localhost:8000
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"你好"}'
```

### 方式四：记忆"二次受益"演示（面试话术素材）

```bash
python scripts/demo.py
```

## 🔐 智舱指令层（Cockpit）设计

`raw_input → 意图理解(function calling) → VehicleCommand(结构化 + Action 白名单) → SafetyGate → 执行/确认/拦截/问答`

**安全门四级分流**（[safety.py](cockpit/safety.py)）：

| 级别 | 例子 | 行为 |
|---|---|---|
| 🔵 INFO | "现在车速多少？" | 直接问答 |
| 🟢 COMFORT | "空调 24 度 / 音量 15" | 直接执行 |
| 🟡 CONTROLLED | "自动泊车 / 定速巡航" | **二次确认后执行** |
| 🔴 FORBIDDEN | "关掉 ESP / 超速" | 拦截 |

**关键工程点**：
1. **动作白名单 `Action` enum**：模型只能从固定动作集合中选，杜绝"自由文本动作名漂移 + 绕过安全判定"（曾出现的真实缺陷：`把ESP关掉`被模型误解析成舒适级的 `disable`）。
2. **危险语义优先于结构化判定**：哪怕模型把危险输入错解析成无害动作，危险词规则（esp/气囊/超速…）仍会拦截。
3. **二次确认为"精确暂存 + 重放"**：需确认的受控指令原样存入 pending 队列，用户确认后从队列逐条执行，**不重新丢给模型猜测**，保证参数永远精确。
4. **按记忆类型选去重策略**：偏好/目的地用"同一主题 latest-wins"，避免 `24℃；24℃` 这类拼接重复。

**智舱 + 记忆**（[controller.py](cockpit/controller.py)）：
- 召回长期记忆注入意图上下文；
- 显式"记住 XX"沉淀长期偏好/常用目的地；
- 执行偏好型动作后自动学习偏好；
- 导航"回家 / 去公司"自动解析记忆中的目的地。

## 🧠 记忆链路（面试解说主线）

```
用户输入 -> 1. 召回(长期语义检索 + 短期最近) -> 注入上下文
        -> 2. Agent ReAct 循环(思考-工具-回答)
        -> 3. 沉淀(对话 -> 短期)
        -> 4. 重要信息 -> 升级长期(真实 Embedding + 冲突去重 + LRU 遗忘)
```

1. **何时写**：对话默认进短期；被判定为"目标/事实/教训/偏好/目的地"或用户强调的重要信息升级长期。
2. **怎么写**：长期记忆用真实 Embedding（语义检索）+ 结构化元数据双写。
3. **怎么检**：`MemoryManager.recall` 合并长期命中 + 短期最近上下文后注入 System Prompt。
4. **冲突去重**：写入长期前用 Embedding 找最相似记忆，相似度 ≥ `memory_conflict_threshold`(0.85) 则合并；**偏好/目的地按主题 latest-wins**。
5. **遗忘（LRU）**：超 `memory_capacity`(200) 后按"最近最少访问"删除最久未用。
6. **反思闭环**：规划执行后 `planner/reflector.py` 抽取经验/事实写入长期记忆，供下次决策召回。
7. **线程安全**：SQLite 连接设 `check_same_thread=False` + 锁，兼容 Streamlit 多线程 rerun。

## 🧭 配置项（.env）

| 变量 | 说明 | 默认 |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI 兼容 LLM 端点 | gpt-4o-mini |
| `EMBEDDING_MODEL` | 同一端点下 /embeddings 模型，用于长期记忆 | text-embedding-3-small |
| `MEMORY_DB_PATH` | 长期记忆向量存储路径 | ./data/memory.db |
| `MEMORY_CAPACITY` | 长期记忆容量，超量触发 LRU 遗忘 | 200 |

## 🧭 本地模型（ollama）可选配置

用本地 qwen + embedding 时：

```bash
ollama pull qwen2.5:7b
# 长期记忆若用真实向量（可选，推荐）
ollama pull nomic-embed-text
```

然后 `.env`：

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
EMBEDDING_MODEL=nomic-embed-text
```

## 🗺 迭代/面试准备建议

- 已实现：ReAct 工具调用闭环、反思模块、记忆冲突去重 / LRU 遗忘、真实 Embedding、ChatUI、流式输出、智舱安全门、受控指令二次确认、目的地与偏好记忆。
- 可补充（按性价比）：评测脚本（带记忆 vs 不带记忆量化提升）、重要度评估（TCEN）、规划中途修订、并行执行任务 DAG、pytest + CI。

## ⚠️ 说明

- 代码使用 Python 3.10+ 新语法；Embedding 未配置端点时自动降级为哈希向量（"能跑"模式）。
- LLM 与 Embedding 均走 OpenAI 兼容协议，可平替各种国产/开源网关。
- 智舱 `VehicleState` 为模拟实现；接入真实车身/CAN 时仅需替换该实现。