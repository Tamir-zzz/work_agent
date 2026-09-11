"""长期记忆：真实 Embedding(向量，存 SQLite) + 结构化元数据(SQLite) 混合存储。

设计要点(面试可深挖)：
- 向量检索与冲突检测共用同一套真实 Embedding，保证"语义"一致(而非两套指标打架)。
- 每条记忆记录 last_accessed_at，供 LRU 遗忘裁剪(近期访问过的记忆更倾向保留)。
- Embedding 接口不可用时降级为哈希向量，保证离线可用。

相比依赖外部向量库，这里把向量直接持久化在 SQLite，部署更轻、也便于 Interview 现场跑通。
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

from config.settings import settings
from core.embedding import Embedder, cosine_sim, get_embedder
from core.models import MemoryEntry
from memory.base import BaseMemory


class LongTermMemory(BaseMemory):
    def __init__(self, db_path: Path, embedder: Embedder | None = None) -> None:
        self._db_path = db_path
        self._embedder = embedder or get_embedder()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: 允许跨线程复用同一连接(Streamlit 每次 rerun 可能换线程)。
        # 并发安全由 self._lock 保证: 所有 sqlite 访问均在锁内同步完成, 不在锁内 await。
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata TEXT,            -- json 序列化
                    embedding TEXT,           -- json 序列化的向量
                    created_at INTEGER,       -- unix 秒
                    last_accessed_at INTEGER  -- unix 秒, 用于 LRU 遗忘
                )
                """
            )
            self._conn.commit()

    # ---------- 内部工具 ----------
    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def _encode(self, value: list[float]) -> str:
        return json.dumps(value)

    def _decode(self, value: str | None) -> list[float] | None:
        if not value:
            return None
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None

    # ---------- 存储操作 ----------
    async def add(self, content: str, metadata: dict | None = None) -> MemoryEntry:
        now = int(time.time())
        text = self._normalize_text(content)
        vec = self._embedder.embed(text)[0]
        entry = MemoryEntry(
            level="long_term",
            content=text,
            metadata=metadata or {},
            embedding=vec,
            created_at=str(now),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO entries (id, content, metadata, embedding, created_at, last_accessed_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (entry.id, text, json.dumps(entry.metadata, ensure_ascii=False),
                 self._encode(vec), now, now),
            )
            self._conn.commit()
        return entry

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        qvec = self._embedder.embed(query)[0]
        scored: list[tuple[float, MemoryEntry]] = []
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, metadata, embedding, created_at FROM entries"
            ).fetchall()
        for eid, content, meta, emb, ts in rows:
            entry = self._to_entry(eid, content, meta, ts)
            vec = self._decode(emb) or entry.embedding
            if not vec:
                continue
            sim = cosine_sim(qvec, vec)
            scored.append((sim, entry))
        scored.sort(key=lambda x: x[0], reverse=True)

        # 更新被命中的记忆访问时间(LRU 依据)
        hits = [e for _, e in scored[:top_k]]
        await self._touch([e.id for e in hits])
        return hits

    async def get(self, entry_id: str) -> MemoryEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, content, metadata, embedding, created_at FROM entries WHERE id = ?",
                (entry_id,),
            ).fetchone()
        if not row:
            return None
        entry = self._to_entry(*row)
        await self._touch([entry.id])
        return entry

    async def delete(self, entry_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            self._conn.commit()
        return cur.rowcount > 0

    async def save(self, entry: MemoryEntry) -> None:
        with self._lock:
            cur = self._conn.execute("SELECT 1 FROM entries WHERE id = ?", (entry.id,))
            exists = cur.fetchone() is not None
        if exists:
            await self.delete(entry.id)
        await self.add(entry.content, entry.metadata)

    async def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM entries")
            self._conn.commit()

    async def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    async def all(self, limit: int = 200) -> list[MemoryEntry]:
        """返回全部长期记忆(最新创建的在前)，供 UI/遗忘使用。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, metadata, embedding, created_at FROM entries"
                " ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._to_entry(r[0], r[1], r[2], r[4]) for r in rows]

    async def best_match(self, text: str) -> tuple[MemoryEntry | None, float]:
        """返回与给定文本最相似的长期记忆及其余弦相似度，用于冲突去重。"""
        qvec = self._embedder.embed(text)[0]
        best_entry, best_sim = None, 0.0
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, metadata, embedding, created_at FROM entries"
            ).fetchall()
        for row in rows:
            eid, content, meta, emb, ts = row
            entry = self._to_entry(eid, content, meta, ts)
            vec = self._decode(emb)
            if not vec:
                continue
            sim = cosine_sim(qvec, vec)
            if sim > best_sim:
                best_sim, best_entry = sim, entry
        return best_entry, best_sim

    # ---------- 遗忘策略(LRU) ----------
    async def prune(self, capacity: int | None = None) -> list[MemoryEntry]:
        """容量超出 capacity 时，删除最近最少访问的条目(保留后进入的)。
        返回被遗忘的条目。"""
        cap = capacity if capacity is not None else settings.memory_capacity
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, metadata, embedding, created_at FROM entries"
                " ORDER BY last_accessed_at DESC"
            ).fetchall()
        victims: list[MemoryEntry] = []
        for row in rows[cap:]:
            entry = self._to_entry(row[0], row[1], row[2], row[4])
            victims.append(entry)
            await self.delete(entry.id)
        return victims

    async def _touch(self, ids: list[str]) -> None:
        if not ids:
            return
        now = int(time.time())
        with self._lock:
            self._conn.executemany(
                "UPDATE entries SET last_accessed_at = ? WHERE id = ?",
                [(now, eid) for eid in ids],
            )
            self._conn.commit()

    def _to_entry(self, eid: str, content: str, meta: str, ts) -> MemoryEntry:
        try:
            m = json.loads(meta) if meta else {}
        except Exception:
            m = {}
        return MemoryEntry(
            id=eid,
            level="long_term",
            content=content,
            metadata=m,
            created_at=str(ts) if ts is not None else None,
        )