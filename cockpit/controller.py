"""座舱意图理解 + 执行控制器：把自然语言/语音 变成 结构化指令 -> 安全门 -> 执行。

复用核心 Agent 的 function calling 链路：LLM 产出 tool_call(emit_command)，
我们把 tool_call 参数构造成 VehicleCommand，交 SafetyGate 裁决后，按等级执行或阻断。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from cockpit.commands import CommandExecutor
from cockpit.safety import SafetyGate
from cockpit.schemas import Action, Domain, SafetyVerdict, VehicleCommand
from cockpit.state import VehicleState
from core.llm import BaseLLM
from memory.manager import MemoryManager

# 常用目的地捕获：家 / 公司(单位) 等别名 -> 地址
_HOME_RE = re.compile(r"(?:我|我们)?家(?:的地址)?(?:在|就是|是|=|住|位于)\s*(?P<addr>.+?)(?:[。！]|$)")
_OFFICE_RE = re.compile(r"(?:我|我们|我的)?(?:公司|单位|上班的地方)(?:的地址)?(?:在|就是|是|=|位于)\s*(?P<addr>.+?)(?:[。！]|$)")
_DEST_ALIAS_RE = re.compile(r"目的地\s*(?:是|为|叫|等于|=|→)?\s*`?(?P<alias>[^，,：:。]+)`?\s*(?:的地址)?(?:在|是|为|=|→)?\s*(?P<addr>.+?)(?:[。！]|$)")

_COCKPIT_SYSTEM = """你是车规级(车载)座舱助手。根据用户指令确定要执行的车控动作。

规则：
- 只有明确的车控意图才调用 emit_command 工具；纯问答/查询(如车速、续航)也可调用
  emit_command 且 domain=info。
- 参数要具体、合理；与车辆运动轨迹相关的指令 domain 用 cruise/park/nav。
- 不要虚构车辆没有的操作；简单一问一答可直接用自然语言回答，不必调用工具。
- 只要用户表达"去某个地方/回家/回公司/导航到XXX"，就必须调用 emit_command：
  domain=nav action=navigate_to，destination 填用户说的地点或别名(家/公司)。
- 若用户要求危险操作(超速、关闭安全系统等)，仍应如实输出对应命令，交由安全门拦截。"""

_COMMAND_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "emit_command",
        "description": "把用户的车控意图转成一条结构化车辆指令。",
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "enum": [d.value for d in Domain],
                    "description": "能力域",
                },
                "action": {
                    "type": "string",
                    "enum": [a.value for a in Action],
                    "description": "动作名(只能从白名单中选)",
                },
                "params": {"type": "object", "description": "动作关键参数, 如 温度/音量等"},
                "target": {"type": "string", "description": "分区, 如 driver/passenger"},
            },
            "required": ["domain", "action"],
        },
    },
}


@dataclass
class CockpitOutcome:
    """一次座舱交互的完整结果，供 UI/日志展示。"""

    raw_text: str = ""
    command: VehicleCommand | None = None
    verdict: SafetyVerdict | None = None
    answer: str = ""
    executed: bool = False
    confirmed: bool = False
    remembered: bool = False
    recalled: str = ""
    # 多条指令输入时逐条执行的结果明细
    results: list[dict] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.results is None:
            self.results = []


class CockpitController:
    def __init__(
        self,
        llm: BaseLLM,
        gate: SafetyGate | None = None,
        state: VehicleState | None = None,
        executor: CommandExecutor | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self.llm = llm
        self.state = state or VehicleState()
        self.executor = executor or CommandExecutor(self.state)
        self.gate = gate or SafetyGate()
        self.memory = memory  # 可空：未接记忆时照常工作
        # 待二次确认的受控指令(命令精确暂存，确认时重放而非重新让模型猜)
        self._pending: list[tuple[VehicleCommand, str]] = []

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    # ---------- 主入口 ----------
    async def handle(self, text: str, auto_confirm: bool = False) -> CockpitOutcome:
        self._pending = []  # 新输入会清掉上一轮的待确认指令(新意图取代旧意图)
        # 0) 记忆召回 -> 注入意图上下文
        recalled = await self._recall_context(text) if self.memory else ""

        # 1) 显式"记住XX" -> 直接沉淀长期记忆, 不走车控
        noted = await self._try_remember(text)
        if noted:
            return CockpitOutcome(raw_text=text, answer=noted, remembered=True, recalled=recalled)

        # 2) 意图 -> (可能多条)指令 -> 逐条过安全门 -> 逐条执行
        msg = self._run_llm(text, recalled)
        cmds = [
            self._build_command(text, tc)
            for tc in (msg.get("tool_calls") or [])
            if tc.get("function", {}).get("name") == "emit_command"
        ]
        if not cmds:
            answer = msg.get("content") or "(模型未给出可执行指令，请换个说法。)"
            return CockpitOutcome(raw_text=text, answer=answer, recalled=recalled)

        outcomes = [await self._handle_one(text, c, auto_confirm, recalled) for c in cmds]
        if len(outcomes) == 1:
            return outcomes[0]

        first = outcomes[0]
        return CockpitOutcome(
            raw_text=text,
            command=first.command,
            verdict=first.verdict,
            answer="\n".join(o.answer for o in outcomes if o.answer),
            executed=any(o.executed for o in outcomes),
            confirmed=any(o.confirmed for o in outcomes),
            recalled=recalled,
            results=[r for o in outcomes for r in o.results],
        )

    async def _handle_one(
        self, text: str, cmd: VehicleCommand, auto_confirm: bool, recalled: str
    ) -> CockpitOutcome:
        # 导航续接：把"导航去家/公司"解析成记忆里的真实地址
        if cmd.action == "navigate_to":
            alias = cmd.params.get("destination") or cmd.target or ""
            resolved = await self._resolve_destination(alias)
            if resolved:
                cmd.params["destination"] = resolved
                cmd.params["_alias"] = alias

        verdict = self.gate.classify(cmd)
        outcome = self._dispatch(text, cmd, verdict, auto_confirm=auto_confirm)
        outcome.recalled = recalled

        # 需二次确认且未确认的受控指令 -> 暂存，待用户确认后精确重放
        if verdict.level.value == "controlled" and not outcome.executed:
            self._pending.append((cmd, text))

        # 执行后写记忆：动作日志(短期) + 偏好学习(长期)
        if self.memory and outcome.executed:
            await self._log_action(cmd)
            await self._learn_preference(cmd)

        outcome.results.append({
            "command": cmd.model_dump(mode="json"),
            "verdict": verdict.model_dump(mode="json"),
            "executed": outcome.executed,
            "answer": outcome.answer,
        })
        return outcome

    async def confirm_pending(self) -> list[CockpitOutcome]:
        """用户确认后，从暂存队列精确重放执行所有受控指令(不再重新问模型)。"""
        outcomes: list[CockpitOutcome] = []
        for cmd, text in self._pending:
            verdict = self.gate.classify(cmd)
            outcome = self._dispatch(text, cmd, verdict, auto_confirm=True)
            outcome.recalled = ""
            if self.memory and outcome.executed:
                await self._log_action(cmd)
                await self._learn_preference(cmd)
            outcome.results.append({
                "command": cmd.model_dump(mode="json"),
                "verdict": verdict.model_dump(mode="json"),
                "executed": outcome.executed,
                "answer": outcome.answer,
            })
            outcomes.append(outcome)
        self._pending = []
        return outcomes

    # ---------- 记忆：召回 ----------
    async def _recall_context(self, text: str, top_k: int = 5) -> str:
        hits = await self.memory.recall(text, top_k=top_k) if self.memory else []
        if not hits:
            return ""
        lines = [f"- {m.content}" for m in hits]
        return "\n".join(["用户已知偏好/背景:", *lines])

    # ---------- 记忆：显式记住 ----------
    async def _try_remember(self, text: str) -> str | None:
        # 1) 常用目的地(家/公司/自定义别名) -> 结构化沉淀, 供导航解析
        dest = self._capture_destination(text)
        if dest:
            alias, addr = dest
            await self.memory.remember_important(
                f"导航目的地 {alias} → {addr}",
                {"kind": "destination", "alias": alias, "important": True, "source": "cockpit"},
            )
            return f"已记住目的地：{alias} → {addr}"

        # 2) 其他"记住 XX" -> 通用偏好
        m = re.search(r"(?:记住|记下|记得)(.+?)(?:$|。|！)", text)
        if not m:
            return None
        note = m.group(1).strip()
        if not note:
            return None
        await self.memory.remember_important(
            note,
            {"kind": "preference", "source": "cockpit", "important": True},
        )
        return f"已记住：{note}"

    def _capture_destination(self, text: str) -> tuple[str, str] | None:
        m = _DEST_ALIAS_RE.search(text)
        if m:
            alias, addr = m.group("alias").strip(), m.group("addr").strip()
            if alias and addr:
                return alias, addr
        m = _HOME_RE.search(text)
        if m and m.group("addr").strip():
            return "家", m.group("addr").strip()
        m = _OFFICE_RE.search(text)
        if m and m.group("addr").strip():
            return "公司", m.group("addr").strip()
        return None

    # ---------- 记忆：导航目的地解析(别名 -> 真实地址) ----------
    async def _resolve_destination(self, query: str) -> str | None:
        if not self.memory or not query:
            return None

        def norm(a: str) -> str:
            return re.sub(r"^(我|我们|我的|咱们)+", "", a).strip()

        q = norm(query)
        for e in await self.memory.long_term.all(limit=1000):
            if e.metadata.get("kind") != "destination":
                continue
            if norm(e.metadata.get("alias", "")) != q:
                continue
            addr = e.content.rsplit("→", 1)[-1].strip()
            if addr:
                return addr
        return None

    # ---------- 记忆：偏好学习(单值偏好用"同主题取最新值"upsert, 而非相似合并累加) ----------
    async def _learn_preference(self, cmd: VehicleCommand) -> None:
        if cmd.action == "set_temperature":
            content = f"用户空调偏好设定温度 {cmd.params.get('temperature')} 摄氏度"
            marker = "空调偏好"
            topic = "pref_climate_temp"
        elif cmd.action == "set_volume":
            content = f"用户多媒体音量偏好设为 {cmd.params.get('volume')}"
            marker = "音量偏好"
            topic = "pref_media_volume"
        else:
            return
        # 先清掉同主题旧偏好(kind=preference 且内容含 marker), 再写入单条最新值
        await self._purge_preference(marker)
        await self.memory.remember_important(
            content, {"kind": "preference", "topic": topic, "source": "cockpit", "important": True}
        )

    async def _purge_preference(self, marker: str) -> None:
        for e in await self.memory.long_term.all(limit=500):
            if e.metadata.get("kind") == "preference" and marker in e.content:
                await self.memory.long_term.delete(e.id)

    async def _log_action(self, cmd: VehicleCommand) -> None:
        await self.memory.remember(
            f"[座舱动作] {cmd.domain.value}/{cmd.action} → {json.dumps(cmd.params, ensure_ascii=False)}",
            {"level": "action_log"},
        )

    # ---------- 意图 -> 指令 ----------
    def _run_llm(self, text: str, recalled: str) -> dict[str, Any]:
        user_block = recalled + f"\n车辆状态: {json.dumps(self.state.snapshot(), ensure_ascii=False)}"
        messages = [
            {"role": "system", "content": _COCKPIT_SYSTEM},
            {"role": "user", "content": f"{user_block}\n用户: {text}"},
        ]
        return self.llm.chat_message(messages, tools=[_COMMAND_TOOL], temperature=0.1)

    def _build_command(self, text: str, tool_call: dict) -> VehicleCommand:
        try:
            args = json.loads(tool_call.get("function", {}).get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        domain_raw = str(args.get("domain", "")).lower()
        domain = None
        for d in Domain:
            if d.value == domain_raw:
                domain = d
                break
        # 动作必须命中白名单，否则归入 safety 域交由安全门拦截(防止逃逸)
        action = str(args.get("action", ""))
        if action not in {a.value for a in Action}:
            domain = Domain.SAFETY
            action = "invalid_action"
        if domain is None:
            domain = Domain.SAFETY  # 未识别域 -> 入 forbidden 兜底拦截
        return VehicleCommand(
            domain=domain,
            action=action,
            params=dict(args.get("params") or {}),
            target=args.get("target"),
            raw_input=text,
        )

    # ---------- 裁决 -> 执行/阻断 ----------
    def _dispatch(
        self,
        text: str,
        cmd: VehicleCommand,
        verdict: SafetyVerdict,
        auto_confirm: bool,
    ) -> CockpitOutcome:
        level = verdict.level.value
        if level == "info":
            return CockpitOutcome(
                raw_text=text, command=cmd, verdict=verdict,
                answer=self.executor.execute(cmd) or verdict.summary,
            )
        if level == "forbidden":
            return CockpitOutcome(
                raw_text=text, command=cmd, verdict=verdict,
                answer=f"已拦截：{verdict.reason}",
            )
        if level == "controlled":
            if auto_confirm:
                return CockpitOutcome(
                    raw_text=text, command=cmd, verdict=verdict,
                    answer=self.executor.execute(cmd), executed=True, confirmed=True,
                )
            return CockpitOutcome(
                raw_text=text, command=cmd, verdict=verdict,
                answer=f"[需确认] {verdict.summary}\n该操作为受控指令，请确认是否执行。",
            )
        # comfort
        return CockpitOutcome(
            raw_text=text, command=cmd, verdict=verdict,
            answer=self.executor.execute(cmd), executed=True,
        )