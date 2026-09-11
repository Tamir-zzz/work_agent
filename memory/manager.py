"""分级记忆管理器：统一调度短期/长期记忆 + 冲突去重 + 遗忘裁剪。

管控逻辑(本文件是记忆重点，面试可深挖)：
1. 写入分级：常规对话进短期；被判定为"值得长期记住"(目标/事实/教训)的进长期。
2. 冲突去重：写入长期前用真实 Embedding 找最相似记忆，超过阈值则合并/更新而非新增，
   避免同一主题的重复记忆堆叠。
3. 遗忘裁剪：长期写满后，按 LRU(最近最少访问)删除最久未用的记忆，控制记忆规模。
4. 检索分级：召回长期相关记忆 + 短期最近上下文，合并后排序返回。
"""
from __future__ import annotations

from pathlib import Path

from config.settings import settings
from core.models import MemoryEntry
from memory.base import BaseMemory
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory


class MemoryManager:
    # 合并冲突时的铺接提示：保留历史，追加新的表述
    _MERGE_MAX = 500

    def __init__(self, db_path: Path | None = None) -> None:
        db_path = db_path or settings.memory_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.short_term = ShortTermMemory(capacity=50)
        self.long_term = LongTermMemory(db_path)

    # ---------- 写入(含去重) ----------
    async def remember(self, content: str, metadata: dict | None = None) -> MemoryEntry:
        """写入短期记忆；若 metadata 标记重要(goal/fact/lesson)则升级长期。"""
        short = await self.short_term.add(content, metadata)
        if (metadata or {}).get("important"):
            await self.remember_important(content, metadata)
        return short

    async def remember_important(self, content: str, metadata: dict | None = None) -> MemoryEntry:
        """强制写入长期记忆，带冲突检测与遗忘裁剪。"""
        metadata = dict(metadata or {})
        # 1. 冲突检测：找最相似记忆，超过阈值则合并而非新增
        existing, sim = await self.long_term.best_match(content)
        if existing is not None and sim >= settings.memory_conflict_threshold:
            merged_meta = dict(existing.metadata)
            merged_meta.update(metadata)
            merged_meta["updated"] = str(True)  # 标记为被合并更新过
            return await self._upsert(existing, self._merge_text(existing.content, content), merged_meta)

        # 2. 新增
        entry = await self.long_term.add(content, metadata)

        # 3. 遗忘裁剪：写满后删除最久未访问的记忆
        cap = int(metadata.get("capacity") or settings.memory_capacity)
        if cap and await self.long_term.count() > cap:
            await self.long_term.prune(cap)
        return entry

    def _merge_text(self, old: str, new: str) -> str:
        """把新表述并入旧记忆，避免重复的同时保留语义增量。
        已完全包含新内容则原样返回(真去重)，避免"24℃；24℃"这类拼接重复。超长时截断。"""
        if not new:
            return old
        if new in old:
            return old
        if old in new:
            return new
        combined = f"{old}；{new}"
        return combined[: self._MERGE_MAX]

    async def _upsert(self, entry: MemoryEntry, content: str, metadata: dict) -> MemoryEntry:
        """覆盖更新一条已存在的长期记忆(先删后插，重建向量)。"""
        await self.long_term.delete(entry.id)
        return await self.long_term.add(content, metadata)

    # ---------- 检索 ----------
    async def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """合并长期语义检索结果 + 短期最近上下文。"""
        long_hits = await self.long_term.search(query, top_k=top_k)
        short_hits = await self.short_term.search(query, top_k=3)
        merged = list(long_hits)
        for s in short_hits:
            if all(s.id != l.id for l in merged):
                merged.append(s)
        return merged

    # ---------- 遗忘/浏览 ----------
    async def forget_least_recently_used(self, capacity: int | None = None) -> list[MemoryEntry]:
        """主动触发 LRU 遗忘，返回被遗忘的条目。"""
        return await self.long_term.prune(capacity)

    async def clear(self) -> None:
        """清空全部记忆(长期 + 短期)。用于一键重置演示。"""
        await self.long_term.clear()
        await self.short_term.clear()

    async def memory_list(self, limit: int = 200) -> dict:
        """给 ChatUI 用的记忆总览：长期(带来源) + 短期最近。"""
        long_terms = await self.long_term.all(limit=limit)
        short_terms = await self.short_term.search("", top_k=20)
        return {
            "long_term": [
                {"id": m.id, "content": m.content, "metadata": m.metadata}
                for m in long_terms
            ],
            "short_term": [
                {"id": m.id, "content": m.content, "metadata": m.metadata}
                for m in short_terms
            ],
        }

    # ---------- 工具 ----------
    async def format_context(self, query: str, top_k: int = 5) -> str:
        """把召回的记忆格式成送给 LLM 的上下文文本，方便注入 System Prompt。"""
        hits = await self.recall(query, top_k=top_k)
        if not hits:
            return ""
        lines = []
        for i, m in enumerate(hits, 1):
            lines.append(f"[{m.level} #{i}] {m.content}")
        return "\n".join(lines)