# 引入 LangGraph 优化编排（自研 ReAct → LangGraph）

## Context / 背景

当前《记忆+规划 Agent》与《智舱 Cockpit Agent》的最外层编排是**自研 ReAct 手写循环**（`core/agent.py` 的 for 循环、`cockpit/controller.py` 的 handle 流程）。为形成「最开始怎么样 / 后面优化了什么」的面试叙事，并展示对 LangGraph 主流编排框架的掌握，计划**把最外层编排迁到 LagGraph 图式状态机**，业务逻辑（记忆/工具/规划/安全门/指令模型）全部复用，**自研 ReAct 代码保留可切换**。

已确认决策：
1. 两套 Agent（记忆+规划、智舱）都用 LangGraph 重写编排。
2. LLM 用 langchain 的 `ChatOpenAI` + `bind_tools`（标准 LangChain 样式）。
3. 演示方式：在 `app_ui` / `app_cockpit` 侧边栏加「编排引擎」单选，同会话 A/B 切换。
4. 业务组件只复用不重写；两引擎暴露一致签名，UI 换对象即可切换。

## 现状关键接口（复用点）

- `core/llm.py` `OpenAICompatibleLLM.chat_message/stream_turn`（自研 LLM，保留；LangGraph 路径改用 ChatOpenAI）。
- `memory/manager.py` `MemoryManager.recall/format_context/remember/remember_important/memory_list`；短期 `short_term.recent_context(k)`；long_term 已线程安全。
- `tools/base.py` `ToolRegistry.specs / call(name,args)->ToolResult / tools()`；`tools/builtin.py build_default_registry(memory)`。
- `planner/` `Planner(llm)`、`Reflector(llm,memory)`、`PlanExecutor`；`core/models.py` 的 `PlanSession/TaskNode/TaskStatus`。规划拆解/反思仍走原实现。
- `cockpit/` `CockpitController.handle(text, auto_confirm)->CockpitOutcome`、`_pending`、`confirm_pending()`、`has_pending`；`SafetyGate.classify(cmd)`；`CommandExecutor.execute(cmd)`；`schemas.VehicleCommand`；`state.VehicleState`。
- `config/settings.py` `settings.llm_base_url/api_key/model/temperature/max_tokens` 直接喂 ChatOpenAI。
- `app_ui.py`（现仅 `st.toggle("规划模式")`）、`app_cockpit.py`（`controller.has_pending` 面板 + `do_confirm` 按钮），均用 st.session_state + rerun。

## 依赖

装进 conda 环境 **ag**（`/opt/conda/envs/ag/bin/pip install`）最小集，并同步 `requirements.txt`：

- `langchain-openai`（含 langchain-core）
- `langgraph`（含 langgraph-checkpoint / MemorySaver，供智舱 `interrupt()`）

不装 `langgraph-sdk`、不装 `langgraph-checkpoint-sqlite`（智舱 `interrupt()` 用进程内 `MemorySaver()` + thread_id 即可，跨重启不持久，演示足够）。

## 文件改动清单

### 新增
1. `langgraph_flow/__init__.py` — 空包导出。
2. `langgraph_flow/llm.py`
   - `get_chat_model()`：用 `settings.*` 构造 `ChatOpenAI`。
   - `act_node(state, registry)`：解析 `last.tool_calls` → `registry.call(name, args)` → 产出 `ToolMessage`（工具 spec 直接收 `registry.specs`，无需 `@tool`）。
3. `langgraph_flow/agent.py` — `LangGraphAgent`
   - `AgentState(messages: Annotated[list, add_messages], user_input)`。
   - 节点：`recall`(memory.format_context + 最近短记上下文 + HumanMessage) → `agent`(model.ainvoke，model 为 `ChatOpenAI.bind_tools(registry.specs)`，只内聚在本节点) → `act` → `remember`。
   - 条件边：`agent` 有 tool_calls → `act`，否则 → `remember` → END。
   - `run_stream`：`app.astream_events(v2)` 过滤 agent 节点 `on_chat_model_stream` 逐 chunk `yield`，工具调用前 `yield "(调用工具: xxx)"` 轨迹。`run` 走 `ainvoke`。
   - 规划复用：`_plan_stream` 中逐任务执行的 `agent.run_stream` 参数化，可由引擎选择 LangGraph 版。
4. `langgraph_flow/cockpit.py` — `LangGraphCockpit`
   - 节点：`intent`(bind_tools 产出 VehicleCommand) → `safety`(SafetyGate.classify) → 条件边：info/comfort→`execute`，forbidden→`blocked`，controlled→`ask_confirm`。
   - `ask_confirm`：`interrupt({...})` 挂起，配合 `MemorySaver` checkpointer；UI 确认时用 `Command(resume={"confirmed":True})` 恢复到 `execute`。
   - `execute` 复用 `CommandExecutor.execute` + 原 `controller._log_action/_learn_preference/_resolve_destination`（组合复用，不重写）。
   - 对外 `handle(text, auto_confirm=False) -> CockpitOutcome`、`confirm() -> list[CockpitOutcome]`、属性 `has_pending`（对齐自研版签名）。

### 修改
5. `requirements.txt` — 追加 `langchain-openai`、`langgraph`。
6. `app_ui.py` — 侧边栏加 `st.radio("编排引擎", ["自研ReAct","LangGraph"])`；`_plan_stream` 增加 `agent` 参数；主区按引擎选对象后调用。
7. `app_cockpit.py` — 侧边栏加引擎 radio；按引擎构造 `LangGraphCockpit` 或 `CockpitController`；受控确认按钮对 LangGraph 走 `confirm()`。

### 不改
`core/agent.py`、`core/llm.py`、`cockpit/controller.py` 等——自研 ReAct 原样保留，双引擎共存。

## 接口对齐（UI 只换对象）

- Agent：`async run(user_input, plan=None) -> str`；`async run_stream(user_input, plan=None) -> AsyncIterator[str]`。
- Cockpit：`async handle(text, auto_confirm=False) -> CockpitOutcome`；`async confirm() -> list[CockpitOutcome]`；`has_pending`。

## 验证方式

端点 `http://localhost:11434/v1`（ollama，qwen2.5:7b，api_key=`ollama`），复用 `.env`。

1. 安装依赖后用 `app_ui` 跑 A/B：同一 prompt（如「建个 SQLite 表」强制工具调用）在自研/LangGraph 两引擎切换，比对流式透明度、工具轨迹、记忆沉淀。
2. 规划模式 + LangGraph：「帮我规划…」→ 拆解→逐任务→反思沉淀；「再规划一下」续接（记忆召回）。
3. 智舱 A/B：舒适「温度调到24」直接执行；受控「自动停个车 / 急速降温」挂起 interrupt→面板确认→执行、拒绝则不执行；危险「关闭esp」拦截；「导航去家」解析记忆地址、执行后沉淀偏好。
4. 可选 `scripts/test_langgraph.py`：纯脚本驱动两图 + `Command(resume=...)` 一次性回归。

## 范围与风险

- 进程内 MemorySaver 的 interrupt 跨重启不持久（刻意最小化；面试可说会投生产可换 sqlite checkpoint）。
- 规划拆解/反思仍走自研 BaseLLM（与 LangGraph agent 节点的 ChatOpenAI 并存），保持"规划-执行-反思"叙事与最小改动。
- 若 ollama 不可用，ChatOpenAI 走降级不可行时会报错——验证前需确认端点连通。