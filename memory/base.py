"""记忆抽象接口：所有记忆实现(短期/长期)统一遵循此协议。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import MemoryEntry


class BaseMemory(ABC):
    """记忆存储接口。

    设计为可插拔：短期记忆用无状态容器，长期记忆用向量+SQLite，
    上层 MemoryManager 统一调度，不感知具体实现。
    """

    @abstractmethod
    async def add(self, content: str, metadata: dict | None = None) -> MemoryEntry:
        """写入一条记忆并返回该条目。"""

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """语义检索最相关的记忆。

        注意：纯内存实现无法做语义检索，可退化为关键字/最近优先匹配，
        long_term 实现则使用向量检索。
        """

    @abstractmethod
    async def get(self, entry_id: str) -> MemoryEntry | None:
        """按 id 获取单条记忆。"""

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """删除一条记忆，返回是否成功。"""

    async def save(self, entry: MemoryEntry) -> None:
        """更新/持久化一条已存在的记忆(默认无操作)。"""

    async def clear(self) -> None:
        """清空该级别全部记忆(默认无操作)。"""