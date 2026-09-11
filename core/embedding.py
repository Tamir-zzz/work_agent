"""Embedding 客户端：调用 OpenAI 兼容 /embeddings 接口产出真实向量。

供长期记忆的语义检索与冲突去重使用。若接口不可用自动降级为哈希向量(保证能跑)。
"""
from __future__ import annotations

import math
from typing import Iterable

from config.settings import settings


def cosine_sim(a: Iterable[float], b: Iterable[float]) -> float:
    """两个向量的余弦相似度，用于冲突检测与检索评分。"""
    a = list(a)
    b = list(b)
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-8
    nb = math.sqrt(sum(x * x for x in b)) or 1e-8
    return dot / (na * nb)


class Embedder:
    """真实 Embedding。构建时惰性导入 openai，未安装则标记不可用。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.model = model or settings.embedding_model
        self._client = None
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key or settings.llm_api_key,
                base_url=base_url or settings.llm_base_url,
            )
        except Exception:  # pragma: no cover
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """把单条或批量文本转成向量。接口失败时降级为局部哈希向量。"""
        if isinstance(texts, str):
            texts = [texts]
        texts = [t.strip() or " " for t in texts]

        if self.available:
            try:
                out: list[list[float]] = []
                for i in range(0, len(texts), settings.embedding_batch_size):
                    batch = texts[i : i + settings.embedding_batch_size]
                    resp = self._client.embeddings.create(model=self.model, input=batch)
                    out.extend(d.embedding for d in resp.data)
                return out
            except Exception:
                pass  # 降级

        return [_hash_embed(t) for t in texts]


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    """无 embedding 接口时的降级向量(字符哈希)。仅供开发/离线演示，正式请用真实模型。"""
    vec = [0.0] * dim
    for ch in text:
        vec[hash(ch) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

_embedder = Embedder()


def get_embedder() -> Embedder:
    return _embedder