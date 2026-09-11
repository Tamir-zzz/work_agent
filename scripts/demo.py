"""演示脚本：演示记忆召回 + 记忆沉淀，展示"一次会话，二次受益"。"""
from __future__ import annotations

import asyncio

from core.agent import Agent
from memory.manager import MemoryManager
from tools.builtin import build_default_registry


async def demo() -> None:
    agent = Agent(tools=build_default_registry(MemoryManager()))
    print(">>> 第一次：告诉 agent 一个长期目标")
    r1 = await agent.run("我关注流浪狗领养，想做一个领养信息管理的小工具")
    print(f"Agent: {r1}\n")

    print(">>> 沉淀重要记忆")
    await agent.memory.remember_important(
        "用户关注流浪狗领养，正在做领养信息管理工具", {"important": True, "topic": "goal"}
    )
    print(">>> 第二次：跨会话提问(类似面试官追问，检验记忆是否生效)")
    r2 = await agent.run("我上次说要做什么来着？")
    print(f"Agent: {r2}")

    print("\n>>> 当前长期记忆条目：")
    for m in await agent.memory.long_term.search("领养", top_k=5):
        print(f" - [{m.metadata.get('topic', 'general')}] {m.content}")


if __name__ == "__main__":
    asyncio.run(demo())