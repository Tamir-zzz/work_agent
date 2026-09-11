"""Pydantic 数据模型：贯穿 Agent 循环、记忆、规划的核心数据结构。"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------- 消息与上下文 ----------
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


# ---------- 工具 ----------
class ToolSpec(BaseModel):
    """工具声明，描述工具的输入参数，供模型 Function Calling 使用。"""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    content: str


# ---------- 记忆 ----------
class MemoryEntry(BaseModel):
    """一条记忆：统一结构，区分分级与来源。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    level: Literal["short_term", "long_term"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: str | None = None


class ReflectiveMemory(BaseModel):
    """反思产生的结构化记忆：目标、经验、教训。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    topic: str
    content: str
    kind: Literal["goal", "lesson", "fact"]
    created_at: str | None = None


# ---------- 规划 ----------
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class TaskNode(BaseModel):
    """任务树节点：规划的最小单元，可嵌套子任务。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    subtasks: list["TaskNode"] = Field(default_factory=list)
    result: str = ""
    order: int = 0


class PlanSession(BaseModel):
    """一次自主规划会话的完整记录，目标是任务树根节点。"""

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str
    root: TaskNode = Field(default_factory=TaskNode)
    created_at: str | None = None


# ---------- 规划上下文 ----------
class PlanContext(BaseModel):
    """PlanContext 是 Agent 完成任务上下文的内部数据结构，承载目标、当前状态、协作对象与执行策略。

    它作为招一次会话的核心交换对象，供主循环读取和逐步推进。
    """

    plan: PlanSession | None = None
    current_task: TaskNode | None = None
    depth: int = 0
    collected_evidence: list[ToolResult] = Field(default_factory=list)
    scratchpad: str = ""


# ---------- 规划上下文重导出（语义别名，明确主循环入口）
PlanningContext = PlanContext