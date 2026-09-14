"""LangGraph 版"记忆+规划 Agent"：用图式状态机替换自研 ReAct 循环。

与 core/agent.py 的 Agent 暴露一致接口(run/run_stream)，业务逻辑(记忆/工具)全部复用：
- recall  : MemoryManager.format_context + 短期近期上下文 -> 构建 System/User 消息
- agent   : ChatOpenAI(bind_tools) 推理，可能产出 tool_calls
- act     : ToolRegistry.call 逐个执行工具并回填 ToolMessage
- remember: 对话沉淀进短期记忆

识别它的标志: 工具调用由图上 `act` 与 `remember` 的条件边驱动，流式用 astream_events。
"""
from __future__ import annotations

from functools import partial
from typing import Annotated, Any, AsyncIterator, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core.agent import _AGENT_SYSTEM
from core.models import PlanContext
from langgraph_flow.llm import get_chat_model
from memory.manager import MemoryManager
from tools.base import ToolRegistry
from tools.builtin import build_default_registry


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    answer: str


def _to_msg(m: dict) -> AnyMessage:
    role = m.get("role", "assistant")
    content = m.get("content", "")
    if role == "user":
        return HumanMessage(content=content)
    return AIMessage(content=content)


async def _recall(state: AgentState, memory: MemoryManager) -> dict:
    ctx = await memory.format_context(state["user_input"], top_k=5)
    recent = memory.short_term.recent_context(k=5)
    messages: list[AnyMessage] = [
        SystemMessage(content=_AGENT_SYSTEM.format(memory_context=ctx or "(无)"))
    ]
    messages += [_to_msg(m) for m in recent]
    messages.append(HumanMessage(content=state["user_input"]))
    return {"messages": messages}


async def _agent(state: AgentState, model) -> dict:
    resp = await model.ainvoke(state["messages"])
    return {"messages": [resp]}


async def _act(state: AgentState, tools: ToolRegistry) -> dict:
    last = state["messages"][-1]
    out = []
    for c in getattr(last, "tool_calls", None) or []:
        r = await tools.call(c["name"], c["args"])
        out.append(ToolMessage(content=r.content, tool_call_id=c["id"]))
    return {"messages": out}


async def _remember(state: AgentState, memory: MemoryManager) -> dict:
    answer = state["messages"][-1].content or ""
    await memory.remember(f"用户: {state['user_input']}\n助手: {answer}")
    return {"answer": answer or state["user_input"]}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    return "act" if getattr(last, "tool_calls", None) else "remember"


class LangGraphAgent:
    """图式编排的记忆+规划 Agent。接口与 core.agent.Agent 对齐，便于 UI 单开关切换。"""

    def __init__(self, memory: MemoryManager | None = None, tools: ToolRegistry | None = None) -> None:
        self.memory = memory or MemoryManager()
        self.tools = tools or build_default_registry(self.memory)

        model = get_chat_model()
        if len(self.tools.tools()):
            model = model.bind_tools(self.tools.specs)
        self._model = model

        g = StateGraph(AgentState)
        g.add_node("recall", partial(_recall, memory=self.memory))
        g.add_node("agent", partial(_agent, model=model))
        g.add_node("act", partial(_act, tools=self.tools))
        g.add_node("remember", partial(_remember, memory=self.memory))
        g.add_edge(START, "recall")
        g.add_edge("recall", "agent")
        g.add_conditional_edges("agent", _should_continue, {"act": "act", "remember": "remember"})
        g.add_edge("act", "agent")
        g.add_edge("remember", END)
        self._app = g.compile()

    def _goal(self, user_input: str, plan: PlanContext | None) -> str:
        if plan and plan.current_task:
            return f"[当前子任务] {plan.current_task.title}\n{plan.current_task.description}"
        return user_input

    async def run(self, user_input: str, plan: PlanContext | None = None) -> str:
        state = await self._app.ainvoke({"messages": [], "user_input": self._goal(user_input, plan)})
        return state.get("answer") or ""

    async def run_stream(self, user_input: str, plan: PlanContext | None = None) -> AsyncIterator[str]:
        async for ev in self._app.astream_events(
            {"messages": [], "user_input": self._goal(user_input, plan)}, version="v2"
        ):
            kind = ev["event"]
            if kind == "on_chat_model_stream":
                if ev.get("metadata", {}).get("langgraph_node") != "agent":
                    continue
                chunk = ev["data"].get("chunk")
                if chunk is None:
                    continue
                tok = chunk.content
                if isinstance(tok, str):
                    if tok:
                        yield tok
                elif isinstance(tok, list):
                    for b in tok:
                        if isinstance(b, dict):
                            t = b.get("text") or b.get("content") or ""
                            if t:
                                yield t
            elif kind == "on_tool_start":
                yield f"\n(调用工具: {ev.get('name', '')})\n"