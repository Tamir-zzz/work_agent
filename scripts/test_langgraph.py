"""LangGraph 编排回归脚本：一次性把两张图 + 智舱 interrupt 二次确认全部跑通。

用法：python scripts/test_langgraph.py   （需先确保 ollama / LLM 端点连通）
覆盖：
- LangGraphAgent    : recall->agent->remember 走通，回答非空
- LangGraphCockpit  : 舒适指令直接执行
- LangGraphCockpit  : 受控指令 interrupt() 挂起 -> confirm_pending()(Command resume) 恢复执行
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph_flow import LangGraphAgent, LangGraphCockpit


async def test_agent() -> None:
    a = LangGraphAgent()
    print("[Agent] nodes:", sorted(a._app.get_graph().nodes.keys()))
    ans = await asyncio.wait_for(a.run("用一句话介绍你自己"), timeout=90)
    print("[Agent] answer:", (ans or "(空)")[:120])
    assert ans.strip(), "LangGraphAgent 回答为空"


async def test_cockpit() -> None:
    c = LangGraphCockpit()
    print("[Cockpit] nodes:", sorted(c._app.get_graph().nodes.keys()))

    # 舒适指令 -> 立即执行
    o1 = await asyncio.wait_for(c.handle("把空调调到24度", auto_confirm=True), timeout=90)
    print("[Cockpit] comfort:", (o1.answer or "(空)")[:90])
    assert o1.results and all(r.get("executed") for r in o1.results)

    # 受控指令 -> interrupt 挂起 -> confirm_pending() 重放
    o2 = await asyncio.wait_for(c.handle("帮我自动停个车", auto_confirm=False), timeout=90)
    print("[Cockpit] controlled pending:", c.has_pending)
    assert c.has_pending, "受控指令未被 interrupt 挂起"

    outs = await asyncio.wait_for(c.confirm_pending(), timeout=90)
    print("[Cockpit] confirmed outcomes:", len(outs),
          "executed:", all(x.executed for x in outs))
    assert outs and all(x.executed for x in outs), "确认后未执行受控指令"
    assert not c.has_pending, "确认后仍有待确认指令"


async def main() -> None:
    await test_agent()
    print()
    await test_cockpit()
    print("\n✅ LangGraph 回归全部通过")


if __name__ == "__main__":
    asyncio.run(main())