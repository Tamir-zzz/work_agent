"""Streamlit ChatUI：纯 Streamlit 单服务。

Streamlit 底层通过 SSE 向浏览器推送增量数据；st.write_stream 把 async 生成器产出的
text token 逐段渲染，实现"边生成边显示"的流式效果，避免整段等待的卡顿感。

运行：streamlit run app_ui.py   # 浏览器自动打开
"""
from __future__ import annotations

import asyncio
from typing import Any

import streamlit as st

from core.agent import Agent
from core.models import PlanContext, TaskNode, TaskStatus
from planner.planner import Planner
from planner.reflector import Reflector
from tools.builtin import build_default_registry

# ---- 进程级单例(演示足够；真实多用户需按会话隔离) ----
_agent = Agent()
_agent.tools = build_default_registry(_agent.memory)
_planner = Planner(_agent.llm, memory=_agent.memory)
_reflector = Reflector(_agent.llm, _agent.memory)


def _run(coro):  # 在同步上下文执行异步协程(仅用于无跑动 loop 的脚本线程)
    return asyncio.run(coro)


# ---------------- 规划流式生成 ----------------
def _flatten(node: TaskNode) -> list[str]:
    lines = [f"- {node.title}"]
    for s in node.subtasks:
        lines += ["  " + x for x in _flatten(s)]
    return lines


async def _plan_stream(goal: str, holder: dict[str, Any]):
    """规划模式流式生成：拆解 -> 逐个执行(流式) -> 反思沉淀，全程逐段产出文本。"""
    # 记住本次目标，使后续含糊提问("再规划一下")仍能召回本主题续接
    await _agent.memory.remember(f"用户目标: {goal}")
    plan = await _planner.create_plan(goal)
    done_tasks: list[TaskNode] = []
    yield "计划已生成：\n"
    for line in _flatten(plan.root):
        yield f"{line}\n"
    yield "\n开始逐步执行：\n"

    while True:
        todos = _planner.next_todo(plan)
        if not todos:
            break
        task = todos[0]
        task.status = TaskStatus.IN_PROGRESS
        yield f"\n■ {task.title}\n"
        buf: list[str] = []
        task_ctx = PlanContext(current_task=task)
        async for tok in _agent.run_stream(
            f"{task.title} {task.description}".strip(), plan=task_ctx
        ):
            buf.append(tok)
            yield tok
        result = "".join(buf)
        _planner.mark_done(plan, task, result)
        done_tasks.append(task)

    holder["plan"] = plan.model_dump()

    reflections = await _reflector.reflect(goal, done_tasks)
    if reflections:
        yield "\n\n[反思已沉淀长期记忆]\n"
        for m in reflections:
            yield f"- {m.content}\n"


# ---------------- 侧边栏渲染 ----------------
def render_memory() -> None:
    st.subheader("记忆库")
    data = _run(_agent.memory.memory_list(limit=50))
    if not data["long_term"] and not data["short_term"]:
        st.caption("(空)")
        return
    if data["long_term"]:
        with st.expander(f"长期记忆 {len(data['long_term'])}", expanded=False):
            for m in data["long_term"]:
                tag = m["metadata"].get("topic") or m["metadata"].get("kind") or "long"
                st.markdown(f"**长期 · {tag}**")
                st.write(m["content"])
    if data["short_term"]:
        with st.expander(f"短期记忆 {len(data['short_term'])}", expanded=False):
            for m in data["short_term"]:
                st.write(m["content"])


def _render_task_tree(t: dict[str, Any], depth: int = 0) -> None:
    mark = {"done": "[done]", "in_progress": "[doing]", "blocked": "[blocked]", "pending": "[todo]"}.get(
        t.get("status"), "?"
    )
    st.markdown(
        "&nbsp;&nbsp;" * depth + f"- **{mark}** {t['title']}"
    )
    for s in t.get("subtasks") or []:
        _render_task_tree(s, depth + 1)


def render_plan() -> None:
    st.subheader("任务进度")
    plan = st.session_state.get("last_plan")
    if not plan or not plan.get("root"):
        st.caption("尚未执行规划")
        return
    st.write("目标：", plan.get("goal"))
    _render_task_tree(plan["root"])


# ---------------- 主聊天区 ----------------
def render_chat(plan_mode: bool) -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    prompt = st.chat_input("输入消息，例如：帮我规划学习 Agent 开发")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    holder: dict[str, Any] = {}
    with st.chat_message("assistant"):
        gen = (
            _plan_stream(prompt, holder)
            if plan_mode
            else _agent.run_stream(prompt)
        )
        try:
            out = st.write_stream(gen)
        except Exception as e:  # noqa: BLE001
            out = f"(流式输出异常: {e})"
            st.write(out)
        st.session_state["messages"].append({"role": "assistant", "content": out})
        if holder:
            st.session_state["last_plan"] = holder.get("plan")


def main() -> None:
    st.set_page_config(page_title="Memory+Planning Agent", layout="wide")

    with st.sidebar:
        st.title("控制台")
        plan_mode = st.toggle("规划模式", value=False)
        if st.button("清空记忆", type="primary", use_container_width=True):
            _run(_agent.memory.clear())
            st.rerun()
        st.divider()
        render_memory()
        st.divider()
        render_plan()

    st.title("Memory + Planning Agent")
    render_chat(plan_mode)


main()