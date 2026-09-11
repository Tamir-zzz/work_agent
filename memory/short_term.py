"""短期记忆：基于内存的有界容器。会话生命周期内生效，容量受限时按先进先出(可配置)淘汰。

实现要点：
- 使用 collections.deque 记录按时间排序的条目，满足容量上限。
- 不使用向量，语义检索退化为基础条目返回(由 callsite 决定回填策略)。
"""
from __future__ import annotations

from collections import deque

from core.models import MemoryEntry, Role
from memory.base import BaseMemory


class ShortTermMemory(BaseMemory):
    def __init__(self, capacity: int = 50) -> None:
        self._capacity = capacity
        self._entries: deque[MemoryEntry] = deque(maxlen=capacity)

    async def add(self, content: str, metadata: dict | None = None) -> MemoryEntry:
        entry = MemoryEntry(level="short_term", content=content, metadata=metadata or {})
        self._entries.append(entry)
        return entry

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        # 短期记忆不做语义检索，返回最近的条目供上下文使用
        return list(self._entries)[-top_k:]

    async def get(self, entry_id: str) -> MemoryEntry | None:
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None

    async def delete(self, entry_id: str) -> bool:
        for i, e in enumerate(self._entries):
            if e.id == entry_id:
                del self._entries[i]
                return True
        return False

    async def save(self, entry: MemoryEntry) -> None:
        await self.delete(entry.id)
        self._entries.append(entry)

    async def clear(self) -> None:
        self._entries.clear()

    def recent_context(self, k: int = 5) -> list[dict]:
        """转成给 LLM 的对话消息上下文。Role 仅支持 user/assistant。"""
        msgs = []
        for e in list(self._entries)[-k:]:
            role = e.metadata.get("role", Role.ASSISTANT.value)
            if role not in (Role.USER.value, Role.ASSISTANT.value):
                role = Role.ASSISTANT.value
            msgs.append({"role": role, "content": e.content})
        return msgs