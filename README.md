# Memory+Planning Agent & 智舱指令 Agent

基于大模型的 Agent 演示项目，包含两个子系统：

- **记忆 + 自主规划 Agent**：分级记忆（短期 + 长期）、冲突去重、LRU 遗忘、规划-执行-反思闭环、ReAct 工具调用、流式输出。
- **智舱指令 Agent**：文字输入 → 意图理解 → 结构化指令 → 安全门分级 → 记忆 → 执行，演示如何把通用 Agent 做成车载环境里的受控指令执行器，核心是"安全"。

采用自研轻量编排，不依赖 LangChain。LLM 与 Embedding 均走 OpenAI 兼容协议，可平替各类国产/开源网关或本地 ollama。

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

**智舱指令层**
- 安全门四级：信息查询 / 直接执行 / 需二次确认 / 拦截（含危险词兜底）
- 受控指令二次确认：待确认指令精确暂存，确认后重放，避免重新让模型猜测
- 模拟车辆状态（温度、风速、音量、车速等）

**交互**
- Streamlit UI（记忆面 + 智舱面）、FastAPI 接口、命令行、记忆演示脚本

## 技术栈

- Python 3.10+
- LLM / Embedding：OpenAI 兼容客户端（openai SDK）
- 编排：自研 ReAct + 规划执行，FastAPI + SSE
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

## 说明与边界

- 代码使用 Python 3.10+ 新语法。
- 智舱的 `VehicleState` 为**模拟实现**；接入真实车身/CAN 时仅需替换该实现。
- Embedding 不可用时会降级为哈希向量（"能跑"模式）。
- 各模块均为独立可替换组件，方便按需扩展。