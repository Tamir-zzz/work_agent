"""规划执行器：驱动"规划->执行->反思->修订"闭环。

流程：
1. 从 planner 取下一个待办任务
2. 通过核心 Agent 执行该任务(实际是调用 agent.run 完成一个子任务)
3. 本轮任务执行完后，用 reflector 从结果抽取经验/事实沉淀进长期记忆
"""
from __future__ import annotations

from typing import Awaitable, Callable

from core.models import PlanSession, MemoryEntry, TaskNode, TaskStatus
from planner.planner import Planner
from planner.reflector import Reflector


class PlanExecutor:
    def __init__(
        self,
        planner: Planner,
        task_runner: Callable[[TaskNode], Awaitable[str]],
        reflector: Reflector | None = None,
    ) -> None:
        self.planner = planner
        self.task_runner = task_runner
        self.reflector = reflector

    async def execute(self, plan: PlanSession) -> tuple[list[TaskNode], list[MemoryEntry]]:
        """按顺序执行计划的全部待办叶子任务。

        返回 (已执行任务列表, 反思沉淀出的长期记忆条目；未配置 reflector 为空列表)。
        """
        done_tasks: list[TaskNode] = []
        while True:
            todos = self.planner.next_todo(plan)
            if not todos:
                break
            task = todos[0]
            task.status = TaskStatus.IN_PROGRESS
            try:
                result = await self.task_runner(task)
                self.planner.mark_done(plan, task, result)
            except Exception as e:  # noqa: BLE001
                task.status = TaskStatus.BLOCKED
                task.result = f"执行失败: {e}"
            done_tasks.append(task)

        # 反思：从已执行任务中抽取经验/事实写入长期记忆
        reflections: list[MemoryEntry] = []
        if self.reflector is not None and done_tasks:
            reflections = await self.reflector.reflect(plan.goal, done_tasks)

        return done_tasks, reflections