"""LLM 客户端：基于 OpenAI 兼容接口封装，支持工具调用与任意文本聊天。

注意：模块内 import 使用局部导入，避免在未安装依赖时让整个包 import 失败。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from config.settings import settings
from core.models import Message


class BaseLLM(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str:
        """发送一次完整的对话，返回 assistant 的文本内容。"""

    @abstractmethod
    def chat_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """发送一次对话，返回完整的 assistant 消息(dict)：
        {"role": "assistant", "content": ..., "tool_calls": [...]}
        供 ReAct 循环解析 function calling。
        """

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式对话，逐 token 产出文本。"""

    @abstractmethod
    def stream_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> tuple[AsyncIterator[str], dict[str, Any]]:
        """流式生成一轮 assistant 回合。

        返回 (gen, box)：
        - gen: async generator，逐 token 产出文本(content)，供 UI 边生成边显示
        - box: dict，生成结束后写入 {"content": 完整文本, "tool_calls": [...] | None}
          供 ReAct 循环判断是否需要执行工具。
        """


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self) -> None:
        from openai import AsyncOpenAI, OpenAI

        self._client = OpenAI(
            api_key=settings.llm_api_key, base_url=settings.llm_base_url
        )
        self._async_client = AsyncOpenAI(
            api_key=settings.llm_api_key, base_url=settings.llm_base_url
        )

    def _base_kwargs(self) -> dict[str, Any]:
        return {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> str:
        params = {**self._base_kwargs(), **kwargs}
        if tools:
            params["tools"] = tools
        resp = self._client.chat.completions.create(messages=messages, **params)
        return resp.choices[0].message.content or ""

    def chat_message(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        params = {**self._base_kwargs(), **kwargs}
        if tools:
            params["tools"] = tools
        resp = self._client.chat.completions.create(messages=messages, **params)
        msg = resp.choices[0].message
        result: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if getattr(msg, "tool_calls", None):
            calls = []
            for tc in msg.tool_calls:
                calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            result["tool_calls"] = calls
        return result

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> AsyncIterator[str]:
        params = {**self._base_kwargs(), **kwargs}
        stream = await self._async_client.chat.completions.create(
            messages=messages, stream=True, **params
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def stream_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> tuple[AsyncIterator[str], dict[str, Any]]:
        """流式生成一轮 assistant 回合：逐 token 产出文本，并把完整内容与
        工具调用写进 box，供 ReAct 循环继续执行工具与回填。"""
        box: dict[str, Any] = {"content": "", "tool_calls": None}
        params = {**self._base_kwargs(), **kwargs}
        if tools:
            params["tools"] = tools

        async def gen() -> AsyncIterator[str]:
            stream = await self._async_client.chat.completions.create(
                messages=messages, stream=True, **params
            )
            content_parts: list[str] = []
            tool_map: dict[int, dict[str, Any]] = {}
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = getattr(chunk.choices[0], "delta", None)
                if delta is None:
                    continue
                content = getattr(delta, "content", None)
                if content:
                    content_parts.append(content)
                    yield content
                tcalls = getattr(delta, "tool_calls", None)
                if not tcalls:
                    continue
                for tc in tcalls:
                    entry = tool_map.setdefault(
                        tc.index,
                        {"id": "", "function": {"name": "", "arguments": ""}},
                    )
                    if tc.id:
                        entry["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is None:
                        continue
                    if fn.name:
                        entry["function"]["name"] += fn.name
                    if fn.arguments:
                        entry["function"]["arguments"] += fn.arguments

            box["content"] = "".join(content_parts)
            if tool_map:
                box["tool_calls"] = [
                    {
                        "id": v["id"] or str(i),
                        "type": "function",
                        "function": v["function"],
                    }
                    for i, v in sorted(tool_map.items())
                ]

        return gen(), box


def get_llm() -> BaseLLM:
    return OpenAICompatibleLLM()