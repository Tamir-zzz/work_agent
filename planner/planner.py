"""自主规划引擎：把长期目标拆解成语义化的任务树并调度执行(规划-执行-反思-修订循环)。

本文件是"自主规划"重点，面试可深挖：
- 从单条 goal 到多级任务树的生成策略
- 如何在执行过程中根据结果增删改任务节点(计划修订)
- 深度控制(防止无限下钻)
"""
from __future__ import annotations

import json

from config.settings import settings
from core.models import PlanSession, TaskNode, TaskStatus
from core.llm import BaseLLM

_PLAN_SYSTEM = (
    "你是一个任务规划器。用户会给出一个目标，请把它拆解为可执行的任务树(最多3层，"
    "每层2-5个任务)。每个任务包含 title 和 description。只输出 JSON，不要其他文字。\n"
    'JSON格式: {"title": "...", "subtasks": [{"title": "...", "description": "..."}]}'
)


class Planner:
    def __init__(
        self,
        llm: BaseLLM,
        max_depth: int | None = None,
        memory=None,
    ) -> None:
        self.llm = llm
        self.max_depth = max_depth or settings.max_plan_depth
        self.memory = memory  # 可选 MemoryManager，用于规划时召回相关历史记忆

    async def create_plan(self, goal: str) -> PlanSession:
        """根据目标生成初始计划，返回结构化 PlanSession。

        若配置了记忆，会先按 goal 召回相关历史记忆并注入上下文，使含糊的后续
        目标(如"再帮我规划一下")仍能基于此前对话续接，而不是丢失主题。
        """
        system_prompt = _PLAN_SYSTEM
        if self.memory is not None:
            ctx = await self.memory.format_context(goal, top_k=5)
            if ctx:
                system_prompt += (
                    "\n\n以下是用户相关的历史记忆，可辅助理解目标(可能为空)：\n" + ctx
                )
        raw = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"目标: {goal}"},
            ],
            temperature=0.3,
        )
        parsed = self._parse(raw)
        root = self._build_tree(parsed, depth=0)
        return PlanSession(goal=goal, root=root)

    def _parse(self, raw: str) -> dict:
        """容错解析模型输出：提取第一个 JSON 对象。"""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1].strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 兜底：至少返回一个顶层任务
            return {"title": raw.strip()[:80], "subtasks": []}

    def _build_tree(self, node: dict, depth: int) -> TaskNode:
        """把解析出的 dict 递归转成 TaskNode，控制深度限制。"""
        children = []
        if depth < self.max_depth:
            for subtask in node.get("subtasks", []) or []:
                children.append(self._build_tree(subtask, depth + 1))
        return TaskNode(
            title=node.get("title", "未命名任务"),
            description=node.get("description", ""),
            subtasks=children,
        )

    def next_todo(self, plan: PlanSession) -> list[TaskNode]:
        """返回当前最应该执行的任务(第一个 PENDING 叶子)。"""
        stack = [plan.root]
        while stack:
            node = stack.pop(0)
            if node.status == TaskStatus.DONE:
                continue
            if node.subtasks:
                stack = node.subtasks + stack
                continue
            return [node]
        return []

    def mark_done(self, plan: PlanSession, task: TaskNode, result: str) -> None:
        task.status = TaskStatus.DONE
        task.result = result

    def revise(self, plan: PlanSession, evidence: str) -> dict:
        """执行后可调用反思修订；骨架实现为占位，可直接返回空修订。"""
        return {"changed": False, "reason": evidence}