"""LangChain/LangGraph 侧的 LLM 复用封装。

与 core/llm.py 不同，这里用 langchain 官方的 ChatOpenAI(而非自研 OpenAICompatibleLLM)，
走标准 bind_tools -> invoke/astream_events 的工具调用链路。配置仍来自 .env(settings)。
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from config.settings import settings


def get_chat_model(**kwargs) -> ChatOpenAI:
    """从 .env 构造一个 langchain ChatOpenAI 实例(OpenAI 兼容，可用于 ollama 等网关)。"""
    return ChatOpenAI(
        model=kwargs.pop("model", settings.llm_model),
        base_url=kwargs.pop("base_url", settings.llm_base_url),
        api_key=kwargs.pop("api_key", settings.llm_api_key),
        temperature=kwargs.pop("temperature", settings.llm_temperature),
        max_tokens=kwargs.pop("max_tokens", settings.llm_max_tokens),
        **kwargs,
    )