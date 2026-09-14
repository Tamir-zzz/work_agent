"""座舱任务规划 Agent - Streamlit 演示页。

启动: streamlit run --server.port 8501 app_cockpit.py
展示: 意图理解 -> 结构化指令 -> 安全门分级(INFO/COMFORT/CONTROLLED/FORBIDDEN) -> 执行 -> 车况实时更新
"""
from __future__ import annotations

import asyncio
import json

import streamlit as st

from cockpit import CockpitController
from cockpit.state import VehicleState
from core.llm import get_llm
from langgraph_flow import LangGraphCockpit
from memory.manager import MemoryManager

st.set_page_config(page_title="座舱 Agent 演示", page_icon="🚗", layout="wide")

_shared_state = VehicleState()
_shared_mem = MemoryManager()


def get_controllers() -> dict:
    """进程级单例，保持车况/记忆跨会话连续；两个编排引擎共享同一车况与记忆池。"""
    if "controllers" not in st.session_state:
        st.session_state["controllers"] = {
            "react": CockpitController(get_llm(), state=_shared_state, memory=_shared_mem),
            "lg": LangGraphCockpit(state=_shared_state, memory=_shared_mem),
        }
    return st.session_state["controllers"]


controllers = get_controllers()
engine = st.sidebar.radio(
    "编排引擎", ["自研 ReAct", "LangGraph"],
    index=0,
    help="自研 ReAct 为手写 handle 流程；LangGraph 用图式状态机 + interrupt() 实现受控指令二次确认，安全门逻辑复用。",
)
controller = controllers["lg" if engine == "LangGraph" else "react"]

# ---------- 侧边栏：车况 + 控制 ----------
with st.sidebar:
    st.header("🚗 车况仪表")
    auto_confirm = st.toggle("自动确认受控指令(演示遮罩)", value=False,
                             help="开启后 controlled 指令自动执行，演示用；真实场景建议关闭")
    if st.button("重置车辆状态 + 清空记忆"):
        fresh = VehicleState()
        for ct in controllers.values():
            ct.state = fresh
        asyncio.run(_shared_mem.clear())
        st.session_state["history"] = []
        st.rerun()
    s = controller.state.snapshot()
    c1, c2 = st.columns(2)
    c1.metric("车厢温度", f"{s['cabin_temp']:.0f}℃", delta=f"设定{s['climate_set']:.0f}")
    c1.metric("风速", f"{s['fan_level']}/7", help="空调风量")
    c2.metric("车速", f"{s['speed']} km/h")
    c2.metric("音量", f"{s['volume']}")
    st.progress(s["range_km"] / 500, text=f"续航 {s['range_km']} km")
    with st.expander("详细状态", expanded=False):
        st.write(s)

    st.divider()
    st.subheader("🧠 记忆库")
    mems = asyncio.run(controller.memory.memory_list(limit=30))
    long = mems["long_term"]
    short = mems["short_term"]
    if long:
        st.markdown(f"**长期({len(long)})**")
        for m in long[:8]:
            tag = m["metadata"].get("kind", "long")
            st.markdown(f"`{tag}`　{m['content'][:60]}")
    else:
        st.markdown("_长期记忆为空，说「记住我空调要20度」试试_")
    if short:
        st.markdown(f"**短期({len(short)})**")
        for m in short[:5]:
            st.caption(f"· {m['content'][:50]}")

# ---------- 主区 ----------
st.title("🚗 座舱任务规划 Agent")
st.caption("意图理解 → 结构化指令 → 安全门分级(INFO / COMFORT / CONTROLLED / FORBIDDEN) → 执行")

if "history" not in st.session_state:
    st.session_state["history"] = []


def render_verdict_from(d) -> str:
    m = {
        "info": ("🔵", "info", "信息查询"),
        "controlled": ("🟡", "controlled", "受控·需确认"),
        "forbidden": ("🔴", "forbidden", "已拦截"),
    }
    return m.get(d.get("level"), ("🟢", "comfort", "舒适类"))


# 用户在确认面板点了"确认执行" -> 从暂存队列精确重放受控指令
if st.session_state.pop("do_confirm", False) and controller.has_pending:
    outs = asyncio.run(controller.confirm_pending())
    results = [o.results[0] for o in outs]
    text = "\n".join(r["answer"] for r in results)
    with st.chat_message("assistant"):
        for r in results:
            dom, act = r["command"]["domain"], r["command"]["action"]
            icon, _, label = render_verdict_from(r["verdict"])
            st.markdown(f"✅ **已确认执行**　`{dom}/{act}`")
        st.write(text)
    st.session_state["history"].append(
        {"role": "assistant", "content": text, "results": results}
    )
    st.rerun()


for item in st.session_state["history"]:
    with st.chat_message(item["role"]):
        if item["role"] == "assistant":
            for r in item.get("results", []):
                dom = r["command"]["domain"]
                act = r["command"]["action"]
                icon, _, label = render_verdict_from(r["verdict"])
                st.markdown(f"{icon} **{label}**　`{dom}/{act}`")
            if item.get("results"):
                with st.expander("安全门裁决 + 结构化命令", expanded=False):
                    st.code(json.dumps(item["results"], ensure_ascii=False, indent=2))
            st.write(item["content"])
        else:
            st.write(item["content"])

prompt = st.chat_input("说点什么，比如：把空调调到24度 / 帮我自动停个车")
if prompt:
    st.session_state["history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("意图理解并过安全门…"):
            outcome = asyncio.run(controller.handle(prompt, auto_confirm=auto_confirm))

        item = {"role": "assistant", "content": outcome.answer, "results": []}
        if outcome.remembered:
            st.markdown("🧠 **已沉淀长期记忆**")
        if outcome.recalled:
            with st.expander("🧠 本次召回的记忆", expanded=False):
                st.code(outcome.recalled)
        for r in outcome.results:
            dom = r["command"]["domain"]
            act = r["command"]["action"]
            icon, _, label = render_verdict_from(r["verdict"])
            item["results"].append(r)
            st.markdown(f"{icon} **{label}**　`{dom}/{act}`")
        if item["results"]:
            with st.expander("安全门裁决 + 结构化命令", expanded=False):
                st.code(json.dumps(item["results"], ensure_ascii=False, indent=2))
        st.write(outcome.answer)

    st.session_state["history"].append(item)
    st.rerun()

# 有待二次确认的受控指令 -> 固定面板展示, 一键精确重放
if controller.has_pending:
    st.info(f"🟡 有 {len(controller._pending)} 项受控指令待确认：")
    for cmd, _ in controller._pending:
        st.markdown(f"　`{cmd.domain}/{cmd.action}`　参数: {cmd.params}")
    if st.button("✅ 确认执行以上受控指令", type="primary"):
        st.session_state["do_confirm"] = True
        st.rerun()