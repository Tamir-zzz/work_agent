"""项目入口：命令行试用 Agent。用法：python app.py "你的问题" [--plan]"""
from __future__ import annotations

import argparse
import asyncio


async def _run(message: str, with_planning: bool) -> None:
    from core.agent import Agent
    from planner.executor import PlanExecutor
    from planner.planner import Planner
    from planner.reflector import Reflector
    from tools.builtin import build_default_registry

    agent = Agent()
    agent.tools = build_default_registry(agent.memory)

    if with_planning:
        planner = Planner(agent.llm)
        reflector = Reflector(agent.llm, agent.memory)
        executor = PlanExecutor(planner, agent.run_task, reflector=reflector)
        plan = await planner.create_plan(message)
        tasks, reflections = await executor.execute(plan)
        print("=== 计划执行结果 ===")
        for t in tasks:
            print(f"[{t.status.value}] {t.title}\n{t.result}")
        if reflections:
            print("\n=== 反思沉淀到长期记忆 ===")
            for m in reflections:
                print(f"- {m.content}")
        return

    answer = await agent.run(message)
    print(f"Agent: {answer}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory+Planning Agent 命令行试用")
    parser.add_argument("message", nargs="*", default=["你好，介绍一下你自己"])
    parser.add_argument("--plan", action="store_true", help="使用规划引擎执行")
    args = parser.parse_args()
    asyncio.run(_run(" ".join(args.message), args.plan))


if __name__ == "__main__":
    main()