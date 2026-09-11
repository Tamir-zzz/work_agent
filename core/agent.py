"""核心 Agent：把记忆、工具、规划组装成一个可对话、可自主完成任务的 Agent。

工作流程(单轮回合)：
1. 记忆召回：根据用户输入从 MemoryManager 取出相关记忆作为上下文
2. ReAct 循环：模型思考 -> (可选)调用工具 -> 观察结果 -> 继续或给出最终回答
3. 记忆写入：把这次对话沉淀进短期记忆；重要信息标记为长期

重点：本文件把三大亮点能力(记忆、工具、规划)串成主循环，是 Agent 的组装点。
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from config.settings import settings
from core.llm import BaseLLM, get_llm
from core.models import PlanContext
from memory.manager import MemoryManager
from tools.base import ToolRegistry

_AGENT_SYSTEM = """你是一个具备记忆与自主规划能力的智能体，你可以调用工具完成任务，也可以直接基于记忆回答。

决策规则：
- 若需要外部信息或执行动作，应使用工具。
- 若已有足够信息，直接给出简洁、准确的最终答复。
- 每步先说清楚你在做什么，再给出结果。

以下是从记忆中检索到的相关信息(可能为空)：
{memory_context}"""


class Agent:
    def __init__(
        self,
        llm: BaseLLM | None = None,
        memory: MemoryManager | None = None,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.llm = llm or get_llm()
        self.memory = memory or MemoryManager()
        self.tools = tools or ToolRegistry()

    async def run(self, user_input: str, plan: PlanContext | None = None) -> str:
        """处理一次用户输入，返回最终回答。

        骨架聚焦"记忆 + 自主思考"。若传入带 current_task 的 PlanContext，
        则以当前任务为目标执行；否则以用户输入为直接目标。
        """
        # 1. 记忆召回注入上下文
        memory_ctx = await self.memory.format_context(user_input, top_k=5)
        goal = user_input
        if plan and plan.current_task:
            goal = (
                f"[当前子任务] {plan.current_task.title}\n"
                f"{plan.current_task.description}"
            )

        system_prompt = _AGENT_SYSTEM.format(memory_context=memory_ctx or "(无)")
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages += self.memory.short_term.recent_context(k=5)
        messages.append({"role": "user", "content": goal})

        # 2. ReAct 循环：模型可多轮思考-调用工具，直到给出无工具调用的最终回答
        answer = ""
        for _ in range(settings.max_agent_steps):
            msg = self.llm.chat_message(
                messages,
                tools=self.tools.specs if len(self.tools.tools()) else None,
            )
            messages.append(msg)

            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                # 无工具调用 -> 模型给出最终回答，结束循环
                answer = msg.get("content") or ""
                break

            # 有工具调用 -> 逐个执行并把结果(ToolMessage)回填给模型
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await self.tools.call(name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "content": result.content,
                    }
                )

        # 3. 对话沉淀进短期记忆(重要信息由上层/MemoryManager 判定是否升级长期)
        await self.memory.remember(f"用户: {user_input}\n助手: {answer}")
        return answer

    async def run_stream(
        self, user_input: str, plan: PlanContext | None = None
    ) -> AsyncIterator[str]:
        """流式处理一次用户输入：逐 token 产出答案，便于 UI 边生成边显示。

        同样支持 ReAct 工具调用循环：先在后台执行工具并回填，随后把最终
        回答逐 token 产出。相比 run() 的一次性返回，体感显著更快。
        """
        # 1. 记忆召回注入上下文
        memory_ctx = await self.memory.format_context(user_input, top_k=5)
        goal = user_input
        if plan and plan.current_task:
            goal = (
                f"[当前子任务] {plan.current_task.title}\n"
                f"{plan.current_task.description}"
            )

        system_prompt = _AGENT_SYSTEM.format(memory_context=memory_ctx or "(无)")
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages += self.memory.short_term.recent_context(k=5)
        messages.append({"role": "user", "content": goal})

        tools = self.tools.specs if len(self.tools.tools()) else None
        answer = ""
        for _ in range(settings.max_agent_steps):
            gen, box = self.llm.stream_turn(messages, tools=tools)
            async for tok in gen:
                yield tok

            content = box.get("content") or ""
            tool_calls = box.get("tool_calls")
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if tool_calls:
                # 有工具调用 -> 执行并回填 ToolMessage，继续下一轮生成
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = await self.tools.call(fn.get("name", ""), args)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "content": result.content,
                        }
                    )
                continue

            answer = content
            break

        # 2. 沉淀进短期记忆
        await self.memory.remember(f"用户: {user_input}\n助手: {answer}")

    async def run_task(self, task_title: str, task_desc: str = "") -> str:
        """执行携带 TaskNode 意图的单任务，供 PlanExecutor 复用。"""
        return await self.run(f"{task_title} {task_desc}".strip())