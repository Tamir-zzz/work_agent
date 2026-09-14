"""LangGraph 版"智舱指令 Agent"：用图式状态机替换手写 handle 流程。

intent(情感理解, ChatOpenAI+bind_tools) -> safety(安全门分级, 条件边)
  -> execute(非受控即时执行) -> (有受控?) ask_confirm(interrupt 二次确认)
      -> execute_controlled(确认后精确重放) -> finish
  -> answer(纯问答)

与自研 CockpitController 暴露一致接口(handle/confirm_pending/has_pending)，
复用其 gate/executor/state/memory 等业务逻辑；受控确认用 LangGraph 原生 interrupt()+MemorySaver。
"""
from __future__ import annotations

import asyncio
import json
import threading

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from cockpit.controller import (
    CockpitController,
    CockpitOutcome,
    _COCKPIT_SYSTEM,
    _COMMAND_TOOL,
)
from cockpit.schemas import Domain, SafetyVerdict, VehicleCommand
from langgraph_flow.llm import get_chat_model
from memory.manager import MemoryManager


def _run(coro):
    """在同步图节点内执行异步协程(记忆读写)。

    LangGraph 的同步节点可能运行在已有 event loop 的线程(wrap 适配器)里，
    直接 asyncio.run 会抛 "cannot be called from a running event loop"；
    因此放入全新线程、各自跑一个干净 loop，返回结果。
    """
    box: dict = {}

    def _target() -> None:
        box["ret"] = asyncio.run(coro)

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    return box.get("ret")


def _rebuild(d: dict) -> VehicleCommand:
    try:
        domain = Domain(d.get("domain", "safety"))
    except ValueError:
        domain = Domain.SAFETY
    return VehicleCommand(
        domain=domain,
        action=d.get("action", ""),
        params=dict(d.get("params") or {}),
        target=d.get("target"),
        priority=d.get("priority", 0),
        raw_input=d.get("raw_input", ""),
    )


class CockpitState(dict):
    user_input: str
    recalled: str
    auto_confirm: bool
    pure_answer: str
    pairs: list
    results: list
    controlled_pairs: list
    confirmed: bool
    answer: str


class LangGraphCockpit:
    def __init__(
        self,
        memory: MemoryManager | None = None,
        gate=None,
        state=None,
        executor=None,
    ) -> None:
        # 复用自研控制器的业务组件与记忆辅助方法
        self._core = CockpitController(
            llm=get_chat_model(), gate=gate, state=state, executor=executor, memory=memory
        )
        self.memory = memory or self._core.memory
        self._model = get_chat_model().bind_tools([_COMMAND_TOOL])

        self._pending_cmds: list[VehicleCommand] = []
        self._resume_cfg: dict | None = None
        self._thread_id = 0

        g = StateGraph(CockpitState)
        g.add_node("intent", self._intent)
        g.add_node("answer", self._answer)
        g.add_node("safety", self._safety)
        g.add_node("execute", self._execute)
        g.add_node("ask_confirm", self._ask_confirm)
        g.add_node("execute_controlled", self._execute_controlled)
        g.add_node("finish", self._finish)
        g.add_edge(START, "intent")
        g.add_conditional_edges("intent", self._route_intent, {"answer": "answer", "safety": "safety"})
        g.add_edge("safety", "execute")
        g.add_conditional_edges(
            "execute", self._route_execute, {"ask_confirm": "ask_confirm", "finish": "finish"}
        )
        g.add_conditional_edges(
            "ask_confirm", self._route_confirm, {"execute_controlled": "execute_controlled", "finish": "finish"}
        )
        g.add_edge("answer", END)
        g.add_edge("finish", END)
        self._app = g.compile(checkpointer=MemorySaver())

    # ---------- 兼容自研控制器接口 ----------
    @property
    def state(self):
        return self._core.state

    @state.setter
    def state(self, v):
        self._core.state = v

    @property
    def has_pending(self) -> bool:
        return bool(self._pending_cmds)

    @property
    def _pending(self) -> list:
        return [(c, "") for c in self._pending_cmds]

    def _next_thread(self) -> str:
        self._thread_id += 1
        return f"cockpit-{self._thread_id}"

    # ---------- 对外主入口 ----------
    async def handle(self, text: str, auto_confirm: bool = False) -> CockpitOutcome:
        self._pending_cmds = []
        recalled = await self._core._recall_context(text) if self.memory else ""

        noted = await self._core._try_remember(text)
        if noted:
            return CockpitOutcome(raw_text=text, answer=noted, remembered=True, recalled=recalled)

        self._resume_cfg = {"configurable": {"thread_id": self._next_thread()}}
        inp = {
            "user_input": text,
            "recalled": recalled,
            "auto_confirm": bool(auto_confirm),
            "pairs": [],
            "results": [],
            "controlled_pairs": [],
        }
        raw = self._app.invoke(inp, self._resume_cfg)

        # 被中断 -> 有受控指令待二次确认
        if self._pending_cmds:
            first = self._pending_cmds[0]
            verdict = self._core.gate.classify(first)
            ans = f"[需确认] {verdict.summary}\n该操作为受控指令，请确认是否执行。"
            return CockpitOutcome(
                raw_text=text, command=first, verdict=verdict, answer=ans,
                recalled=recalled, confirmed=False,
                results=[
                    {"command": c.model_dump(mode="json"), "verdict": verdict.model_dump(mode="json"),
                     "executed": False, "answer": ans}
                    for c in self._pending_cmds
                ],
            )

        results = raw.get("results") or []
        pure = raw.get("pure_answer") or ""
        if pure and not results:
            return CockpitOutcome(raw_text=text, answer=pure, recalled=recalled)
        answer = raw.get("answer") or pure or ""
        first_cmd = _rebuild(results[0]["command"]) if results else None
        verdict = SafetyVerdict(**results[0]["verdict"]) if results and results[0].get("verdict") else None
        return CockpitOutcome(
            raw_text=text, command=first_cmd, verdict=verdict, answer=answer,
            executed=any(r.get("executed") for r in results), recalled=recalled, results=results,
        )

    async def confirm_pending(self) -> list[CockpitOutcome]:
        if not self._pending_cmds or not self._resume_cfg:
            return []
        out = self._app.invoke(input=Command(resume={"confirmed": True}), config=self._resume_cfg)
        results = [r for r in (out.get("results") or []) if r.get("executed")]
        executed_domains = [r["command"]["action"] for r in results]
        self._pending_cmds = [] if executed_domains else self._pending_cmds
        outcomes = []
        for r in results:
            cmd = _rebuild(r["command"])
            verdict = SafetyVerdict(**r["verdict"]) if r.get("verdict") else self._core.gate.classify(cmd)
            outcomes.append(CockpitOutcome(
                raw_text="", command=cmd, verdict=verdict, answer=r.get("answer", ""),
                executed=True, confirmed=True, results=[r],
            ))
        return outcomes

    # ---------- 图节点(同步, 供 invoke+interrupt 使用) ----------
    def _intent(self, state: CockpitState) -> dict:
        text = state["user_input"]
        recalled = state.get("recalled") or ""
        user_block = recalled + f"\n车辆状态: {json.dumps(self._core.state.snapshot(), ensure_ascii=False)}"
        messages = [
            {"role": "system", "content": _COCKPIT_SYSTEM},
            {"role": "user", "content": f"{user_block}\n用户: {text}"},
        ]
        resp = self._model.invoke(messages)
        calls = getattr(resp, "tool_calls", None) or []
        commands = []
        for tc in calls:
            fake = {"function": {"name": tc["name"], "arguments": json.dumps(tc.get("args") or {})}}
            commands.append(self._core._build_command(text, fake))
        if not commands:
            return {"pure_answer": resp.content or "(模型未给出可执行指令，请换个说法。)"}
        # 导航续接：别名 -> 记忆里的真实地址
        for c in commands:
            if c.action == "navigate_to":
                alias = c.params.get("destination") or c.target or ""
                if alias and self.memory:
                    resolved = _run(self._core._resolve_destination(alias))
                    if resolved:
                        c.params["destination"] = resolved
                        c.params["_alias"] = alias
        pairs = []
        for c in commands:
            v = self._core.gate.classify(c)
            pairs.append({
                "cmd": c.model_dump(mode="json"),
                "verdict": v.model_dump(mode="json"),
                "level": v.level.value,
            })
        return {"pairs": pairs}

    def _answer(self, state: CockpitState) -> dict:
        return {"answer": state.get("pure_answer") or ""}

    def _safety(self, state: CockpitState) -> dict:
        # 独立安全门节点：重判一次裁决(SSP：分级独立于意图理解)
        pairs = []
        for p in state.get("pairs") or []:
            cmd = _rebuild(p["cmd"])
            v = self._core.gate.classify(cmd)
            p["verdict"] = v.model_dump(mode="json")
            p["level"] = v.level.value
            pairs.append(p)
        return {"pairs": pairs}

    def _route_intent(self, state: CockpitState) -> str:
        return "safety" if state.get("pairs") else "answer"

    def _execute(self, state: CockpitState) -> dict:
        auto = state.get("auto_confirm")
        results = []
        controlled_pairs = []
        for p in state.get("pairs") or []:
            cmd = _rebuild(p["cmd"])
            level = p["level"]
            if level == "forbidden":
                results.append({
                    "command": p["cmd"], "verdict": p["verdict"], "executed": False,
                    "answer": f"已拦截：{p['verdict'].get('reason', '')}",
                    "level": level,
                })
            elif level == "controlled":
                if auto:
                    ans = self._core.executor.execute(cmd)
                    self._write_memory(cmd)
                    results.append({"command": p["cmd"], "verdict": p["verdict"], "executed": True, "answer": ans, "level": level})
                else:
                    controlled_pairs.append(p)
            else:  # info / comfort
                ans = self._core.executor.execute(cmd)
                if level == "info":
                    ans = ans or p["verdict"].get("summary", "")
                else:
                    self._write_memory(cmd)
                results.append({"command": p["cmd"], "verdict": p["verdict"], "executed": True, "answer": ans, "level": level})
        return {"results": results, "controlled_pairs": controlled_pairs}

    def _route_execute(self, state: CockpitState) -> str:
        return "ask_confirm" if state.get("controlled_pairs") else "finish"

    def _ask_confirm(self, state: CockpitState) -> dict:
        cps = state.get("controlled_pairs") or []
        self._pending_cmds = [_rebuild(p["cmd"]) for p in cps]
        decision = interrupt({"commands": [p["cmd"] for p in cps], "question": "受控指令，确认执行？"})
        confirmed = bool(decision.get("confirmed")) if isinstance(decision, dict) else bool(decision)
        return {"confirmed": confirmed}

    def _route_confirm(self, state: CockpitState) -> str:
        return "execute_controlled" if state.get("confirmed") else "finish"

    def _execute_controlled(self, state: CockpitState) -> dict:
        results = list(state.get("results") or [])
        for p in state.get("controlled_pairs") or []:
            cmd = _rebuild(p["cmd"])
            ans = self._core.executor.execute(cmd)
            self._write_memory(cmd)
            results.append({
                "command": p["cmd"], "verdict": p["verdict"], "executed": True,
                "answer": ans, "level": "controlled",
            })
        self._pending_cmds = []
        return {"results": results}

    def _finish(self, state: CockpitState) -> dict:
        results = state.get("results") or []
        ans = "\n".join(r.get("answer") for r in results if r.get("answer"))
        if not ans:
            ans = state.get("pure_answer") or "[完成]"
        return {"answer": ans}

    # ---------- 记忆写入(偏好学习 + 动作日志) ----------
    def _write_memory(self, cmd: VehicleCommand) -> None:
        if not self.memory:
            return
        _run(self._core._log_action(cmd))
        _run(self._core._learn_preference(cmd))