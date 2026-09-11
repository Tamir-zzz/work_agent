"""反思模块：在一轮规划执行后，从目标与任务结果中抽取"经验/事实"沉淀进长期记忆。

把"规划结果"转化成"下一次决策可复用的记忆"，是记忆+自主规划形成闭环的关键衔接点。
"""
from __future__ import annotations

import json
from typing import Any

from core.llm import BaseLLM
from core.models import MemoryEntry, TaskNode, TaskStatus
from memory.manager import MemoryManager

_REFLECT_SYSTEM = """你是反思器。基于用户目标及其子任务执行结果，提炼有价值的经验与事实。

对每一条输出：
- topic: 简短主题(2-6字左右)
- kind: 只能是 lesson(经验教训) 或 fact(事实/结论)
- content: 一句话，客观、可复用于后续决策

仅输出 JSON 数组，格式: [{"topic": "...", "kind": "lesson|fact", "content": "..."}]
若无可提炼内容，输出 []。不要输出其他文字。"""


class Reflector:
    def __init__(self, llm: BaseLLM, memory: MemoryManager) -> None:
        self.llm = llm
        self.memory = memory

    async def reflect(self, goal: str, tasks: list[TaskNode]) -> list[MemoryEntry]:
        """从目标与已执行任务中抽取反思，写入长期记忆，返回新沉淀的条目。"""
        if not tasks:
            return []

        executed = [
            t for t in tasks if t.status in (TaskStatus.DONE, TaskStatus.BLOCKED)
        ]
        if not executed:
            return []

        # 1. 把目标 + 任务结果整理成反思输入
        summary_lines = [f"目标: {goal}"]
        for t in executed:
            status = "完成" if t.status == TaskStatus.DONE else "受阻"
            summary_lines.append(f"- [{status}] {t.title}: {t.result or '无输出'}")
        summary = "\n".join(summary_lines)

        # 2. 让模型抽取经验/事实
        raw = self.llm.chat(
            messages=[
                {"role": "system", "content": _REFLECT_SYSTEM},
                {"role": "user", "content": summary},
            ],
            temperature=0.2,
        )
        items = self._parse(raw)

        # 3. 写入长期记忆(important 强制升级 long-term)
        entries: list[MemoryEntry] = []
        for it in items:
            topic = it.get("topic", "反思")
            kind = it.get("kind", "fact")
            content = it.get("content", "")
            if not content:
                continue
            entry = await self.memory.remember_important(
                f"[{kind}] {content}",
                {"important": True, "topic": topic, "kind": kind, "source": "reflection"},
            )
            entries.append(entry)
        return entries

    def _parse(self, raw: str) -> list[dict[str, Any]]:
        """容错解析模型输出：提取第一个 JSON 数组。失败返回空列表。"""
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()
            data = json.loads(text)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
            if isinstance(data, dict) and isinstance(data.get("reflections"), list):
                return data["reflections"]
        except json.JSONDecodeError:
            pass
        return []