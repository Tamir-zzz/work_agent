"""FastAPI 后端：提供对话接口(普通/规划)、记忆总览、任务进度，以及 ChatUI 静态页面。

启动: uvicorn api.main:app --reload   # 打开 http://localhost:8000 即可见 ChatUI
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from api.schemas import ChatRequest, ChatResponse
from core.agent import Agent
from core.models import PlanContext, PlanSession
from planner.executor import PlanExecutor
from planner.planner import Planner
from planner.reflector import Reflector
from tools.builtin import build_default_registry

# 组装全局 Agent(演示时单例，真实场景按会话隔离)
_agent = Agent()
_agent.tools = build_default_registry(_agent.memory)

_planner = Planner(_agent.llm, memory=_agent.memory)
_reflector = Reflector(_agent.llm, _agent.memory)
_executor = PlanExecutor(_planner, _agent.run_task, reflector=_reflector)

# 最近一次规划会话，供 ChatUI 展示任务进度
_last_plan: PlanSession | None = None

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Memory+Planning Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    global _last_plan
    goal = req.goal or req.message
    if req.with_planning:
        # 规划模式：建计划 -> 逐个执行叶子任务 -> 反思沉淀长期记忆
        plan_session: PlanSession = await _planner.create_plan(goal)
        _last_plan = plan_session  # 记录，供 GET /plan 展示进度
        tasks, reflections = await _executor.execute(plan_session)
        answers = "\n".join(
            f"■ {t.title}\n{t.result}" for t in tasks if t.result
        )
        if reflections:
            answers += "\n\n[反思沉淀记忆]\n" + "\n".join(
                f"- {m.content}" for m in reflections
            )
        return ChatResponse(answer=answers or "规划执行完成，暂无输出。")
    answer = await _agent.run(req.message)
    return ChatResponse(answer=answer)


@app.get("/chat/stream")
async def stream(q: str = ""):
    """SSE 流式对话(就绪的骨架事件流)。"""
    async def gen():
        for chunk in ["记忆召回...", "正在思考...", "完成"]:
            yield {"event": "message", "data": chunk}
            await asyncio.sleep(0.2)

    return EventSourceResponse(gen())


@app.get("/memories")
async def memories(limit: int = 200):
    """记忆总览：长期 + 短期，供 ChatUI 可视化记忆库。"""
    return await _agent.memory.memory_list(limit=limit)


@app.get("/plan")
async def plan():
    """最近一次规划的任务树及状态，供 ChatUI 展示任务进度。"""
    return _last_plan.model_dump() if _last_plan else {"session_id": None, "goal": None, "root": None}


@app.post("/forget")
async def forget(capacity: int | None = None):
    """手动触发 LRU 遗忘，返回被遗忘的记忆。"""
    victims = await _agent.memory.forget_least_recently_used(capacity)
    return {"forgotten": [v.content for v in victims], "count": len(victims)}


# 静态 ChatUI(放在所有 API 路由之后，避免根挂载遮蔽接口)
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(_STATIC_DIR / "index.html")