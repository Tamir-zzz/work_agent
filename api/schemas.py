"""API 数据模型。"""
from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    # 可选：使用规划引擎处理
    with_planning: bool = False
    goal: str | None = None


class ChatResponse(BaseModel):
    answer: str