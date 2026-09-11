"""内置工具：工具调用层的示例，演示 Agent 如何借助外部能力完成任务。

包含 format_adoption、search_web、list_memories 三个工具，演示"功能调用->工具响应->结果回填"链路。
实际项目可按需扩充。
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from core.models import ToolResult
from tools.base import BaseTool


class FormatAdoptionTool(BaseTool):
    """演示工具1：将收养数据格式化为项目可用的标准结构。暂作占位实现。"""

    name = "format_adoption"
    description = "将用户提供的狗/猫领养原始信息格式化为标准数据对象。"
    parameters = {
        "type": "object",
        "properties": {
            "species": {"type": "string", "description": "动物物种，如 dog/cat"},
            "name": {"type": "string", "description": "名字"},
            "age": {"type": "string", "description": "年龄"},
        },
        "required": ["species", "name"],
    }

    async def run(self, **kwargs) -> str:
        return json.dumps(
            {"status": "ok", "normalized": kwargs}, ensure_ascii=False
        )


class SearchWebTool(BaseTool):
    """演示工具2：简易 web 搜索(占位，真实场景可接入搜索 API)。"""

    name = "search_web"
    description = "在网页上搜索给定关键词，返回前几条结果标题。"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    async def run(self, **kwargs) -> str:
        query = kwargs.get("query", "")
        url = (
            "https://www.baidu.com/s?wd="
            + urllib.parse.quote(query)
        )
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                # 粗略截取部分文本，实际项目使用更可靠的解析
                return f"已请求搜索[{query}], 返回 HTML 长度 {len(html)}"
        except Exception as e:  # noqa: BLE001
            return f"搜索[{query}] 失败: {e}"


class ListMemoriesTool(BaseTool):
    """演示工具3：列出记忆中保存的条目，展示"记忆->工具"如何打通。"""

    name = "list_memories"
    description = "列出当前已保存的重要记忆条目(目标/事实/教训)。"
    parameters = {
        "type": "object",
        "properties": {"top": {"type": "integer", "description": "返回条数"}},
        "required": [],
    }

    def __init__(self, memory_manager) -> None:
        self.memory_manager = memory_manager

    async def run(self, **kwargs) -> str:
        top = int(kwargs.get("top", 10))
        entries = await self.memory_manager.long_term.search("", top_k=top)
        if not entries:
            # 空查询可能返回不了，改为最近读取
            entries = await self.memory_manager.recall("", top_k=top)
        if not entries:
            return "暂无记忆"
        return "\n".join(f"- [{m.content}]" for m in entries[:top])


def build_default_registry(memory_manager) -> "ToolRegistry":
    from tools.base import ToolRegistry

    registry = ToolRegistry()
    registry.register(FormatAdoptionTool())
    registry.register(SearchWebTool())
    registry.register(ListMemoriesTool(memory_manager))
    return registry