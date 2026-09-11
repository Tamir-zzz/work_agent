"""工具接口与注册中心：Agent 可通过 Function Calling 调用上层能力。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models import ToolResult, ToolSpec


class BaseTool(ABC):
    """一个可执行工具的抽象。包含声明(给模型看)与执行逻辑。"""

    name: str
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    @abstractmethod
    async def run(self, **kwargs) -> str:
        """执行工具，返回文本结果。抛异常时由注册中心捕获转成失败结果。"""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    @property
    def specs(self) -> list[dict[str, Any]]:
        """转成 OpenAI 的 tools 参数格式。"""
        return [tool.spec.model_dump() for tool in self._tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(tool_name=name, success=False, content=f"未知工具: {name}")
        try:
            content = await tool.run(**arguments)
            return ToolResult(tool_name=name, success=True, content=content)
        except Exception as e:  # noqa: BLE001 - 工具执行应屏蔽异常细节
            return ToolResult(tool_name=name, success=False, content=f"工具执行失败: {e}")