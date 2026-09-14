# Memory+Planning Agent & 智舱指令 Agent

基于大模型的 Agent 演示项目，重点展示「如何从零造一个 Agent，再逐步做工程化优化」。包含两个核心子系统：

- **记忆 + 自主规划 Agent**：分级记忆（短期 + 长期）、冲突去重、LRU 遗忘、规划-执行-反思闭环、ReAct 工具调用、流式输出。
- **智舱指令 Agent**：文字/语音输入 → 意图理解 → 结构化指令 → 安全门分级 → 记忆 → 执行，把通用 Agent 做成车辆里的受控指令执行器，核心是"安全"。

编排层采用**自研 → 框架**的演进路径，这是本项目最大的看点和面试叙事线：

1. **先自研轻量 ReAct**：手写 while 主循环 + `tool_calls` 解析 + 记忆/规划/安全门全部自研业务逻辑，几百行可控、可讲，先把 Agent 原理讲透。
2. **后用 LangGraph 做工程化优化**：把最外层编排换成图式状态机（`StateGraph` 条件边、`bind_tools` 标准工具调用、智舱 `interrupt()` + Checkpointer 实现受控指令二次确认），业务逻辑全部复用、不重写。
3. **双引擎共存、UI 一键 A/B**：自研 ReAct 原样保留可切换，侧边栏单选切换、共享同一记忆池与安全门，便于对比两种实现。

一切 LLM / Embedding 走 OpenAI 兼容协议，可平替本地 ollama 或任意国产/开源网关。

## 功能

**记忆**
- 短期记忆（进程内存、有界）
- 长期记忆（真实 Embedding 向量，持久化 SQLite）
- 冲突去重：事实按相似度合并；偏好/目的地按同一主题取最新值覆盖
- LRU 遗忘：超容量后按最近访问时间删除最久未用条目

**规划**
- 目标 → 任务树的拆解与执行
- 规划中召回记忆，使「再规划一下」这类含糊请求能续接前文
- 执行后反思模块抽取经验/教训沉淀到长期记忆

**编排**
- 自研 ReAct 主循环（手写 while，含规划-执行-反思闭环）
- LangGraph 图式状态机（recall→agent→act→remember），双引擎 UI 一键切换
- 智舱安全门四级：信息查询 / 直接执行 / 需二次确认 / 拦截（含危险词兜底）
- 智舱受控指令二次确认：自研为待确认队列精确重放；LangGraph 用原生 `interrupt()` + Checkpointer 实现，同一安全门逻辑复用
- 受控指令待确认精确暂存、确认后重放，避免重新让模型猜测
- 模拟车辆状态（温度、风速、音量、车速等）

**交互**
- Streamlit UI（记忆面 + 智舱面）、FastAPI 接口、命令行、记忆演示脚本

## 技术栈

- Python 3.10+
- LLM / Embedding：OpenAI 兼容客户端（openai SDK）
- 编排：自研 ReAct 或 LangGraph 状态机（UI 可切换），FastAPI + SSE
- 记忆：SQLite 持久化向量 + 结构化元数据
- UI：Streamlit（另保留 FastAPI + 原生 HTML ChatUI）

## 目录结构

```
work_agent/
├── app.py                # 命令行入口（记忆 + 规划 Agent）
├── app_ui.py             # Streamlit：记忆规划 UI
├── app_cockpit.py        # Streamlit：智舱 UI
├── requirements.txt
├── .env.example          # 配置样例
├── config/settings.py    # 全局配置
├── core/
│   ├── llm.py            # LLM 客户端（含流式 + tool_calls）
│   ├── models.py         # 数据模型
│   ├── embedding.py      # 真实 Embedding + 降级
│   └── agent.py          # ReAct 主循环
├── memory/
│   ├── base.py           # 记忆抽象接口
│   ├── short_term.py     # 短期记忆
│   ├── long_term.py      # 长期记忆（向量 + SQLite）
│   └── manager.py        # 分级记忆管理器
├── planner/
│   ├── planner.py        # 目标 → 任务树
│   ├── executor.py       # 规划-执行-反思闭环
│   └── reflector.py      # 反思：沉淀长期记忆
├── tools/
│   ├── base.py           # 工具接口 + 注册中心
│   └── builtin.py        # 内置工具示例
├── cockpit/
│   ├── schemas.py        # 指令模型 + 安全等级
│   ├── safety.py         # 安全门
│   ├── state.py          # 模拟车辆状态
│   ├── commands.py       # 指令执行器
│   └── controller.py     # 意图 + 记忆 + 安全门 + 执行编排
├── langgraph_flow/       # 优化版：LangGraph 图式编排
│   ├── llm.py            # LangChain ChatOpenAI（复用 .env 配置）
│   ├── agent.py          # 记忆+规划 Agent 图式 ReAct
│   └── cockpit.py        # 智舱图式编排 + interrupt 二次确认
├── api/main.py           # FastAPI 接口 + ChatUI
├── static/index.html     # 原生 ChatUI
└── scripts/demo.py       # 记忆演示脚本
```

## 安装与启动

```bash
cp .env.example .env      # 填入 LLM / Embedding 端点与 Key
pip install -r requirements.txt
```

**命令行（记忆 + 规划 Agent）**

```bash
python app.py "你好"
python app.py "帮我规划如何学习 Agent 开发" --plan
```

**Streamlit UI（推荐演示）**

```bash
streamlit run app_ui.py --server.port 8501        # 记忆 + 规划
streamlit run app_cockpit.py --server.port 8502   # 智舱
```

**FastAPI + 原生 ChatUI（可选）**

```bash
uvicorn api.main:app --reload
# 打开 http://localhost:8000
curl -X POST localhost:8000/chat -H 'Content-Type: application/json' -d '{"message":"你好"}'
```

**记忆演示脚本**

```bash
python scripts/demo.py
```

## 配置（.env）

| 变量 | 说明 | 默认 |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI 兼容 LLM 端点 | gpt-4o-mini |
| `EMBEDDING_MODEL` | 同一端点下 /embeddings 模型，用于长期记忆 | text-embedding-3-small |
| `MEMORY_DB_PATH` | 长期记忆存储路径 | ./data/memory.db |
| `MEMORY_CAPACITY` | 长期记忆容量，超量触发 LRU 遗忘 | 200 |

未配置 Embedding 端点时，长期记忆自动降级为哈希向量，仍可运行。

### 本地 ollama 示例

```bash
ollama pull qwen2.5:7b        # LLM
ollama pull nomic-embed-text  # Embedding（可选）
```

```dotenv
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
EMBEDDING_MODEL=nomic-embed-text
```

## 编排演进：自研 → LangGraph（面试可讲）

最外层编排先自研手写 ReAct（便于搞懂原理），后迁到 LangGraph 图式状态机做工程化；**自研 ReAct 代码原样保留、可切换**。迁移最小化：只装 `langgraph` + `langchain-openai` 两个包，业务组件（记忆、工具、安全门、指令模型、规划）全部复用不重写。

设计要点：
- 两个引擎暴露**对齐签名**（Agent：`run / run_stream`；Cockpit：`handle / confirm_pending / has_pending`），UI 只换对象即可 A/B。
- LLM 走 langchain `ChatOpenAI` + `bind_tools`（标准 LangChain 工具调用链），配置仍读 `.env`。
- 规划拆解/反思仍走自研 `Planner/Reflector`（与 LangGraph agent 节点的 `ChatOpenAI` 并存），保持"规划-执行-反思"叙事与最小改动。
- 智舱受控指令二次确认由自研待确认队列改成 LangGraph 原生 `interrupt()` + `MemorySaver` Checkpointer 恢复。

两个引擎接口对齐，UI 里单选切换、共享同一记忆池与安全门，方便对比：

| 维度 | 自研 ReAct | LangGraph 优化 |
|---|---|---|
| 主循环 | 手写 `while` + 状态拼接 | 图式状态机（`StateGraph` + 条件边） |
| 工具调用回填 | 手写解析 tool_calls | `bind_tools` + langchain 消息契约 |
| 智舱二次确认 | 待确认队列 + 手动重放 | 原生 `interrupt()` + `MemorySaver` Checkpointer 恢复 |
| 一次输入多条 | for 循环串行处理 | 图节点内循环，同上 |
| 状态可续跑/断点 | 无 | checkpointer 支持中断-恢复、时间旅行 |

业务逻辑（记忆 recall/写入、工具注册、安全门分级、命令执行、模拟车况）全部复用，未改动自研核心，体现了"先自研搞懂原理、再引入框架做工程化"的思路。

## 说明与边界

- 代码使用 Python 3.10+ 新语法。
- 智舱的 `VehicleState` 为**模拟实现**；接入真实车身/CAN 时仅需替换该实现。
- Embedding 不可用时会降级为哈希向量（"能跑"模式）。
- LangGraph 侧进程内 `MemorySaver` 支持中断-恢复，但**跨进程重启不持久**；投产可换 SQLite/redis Checkpointer。
- 编排引擎默认为自研 ReAct；LangGraph 为可选优化路径，未配置 LLM 端点时 LangGraph 路径会直接报错（自研降级仍可"能跑"）。
- 各模块均为独立可替换组件，方便按需扩展。