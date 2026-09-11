"""核心 Agent 包。

注意：不在这里急切导入 Agent，因为 Agent 会反向依赖 memory/tools，
急切导入会导致 `memory.base -> core.models -> core.__init__` 发生循环导入。
改为惰性访问，仅在使用时加载。
"""


def __getattr__(name: str):
    # 惰性导出 Agent, 供 `from core import Agent` 使用
    if name == "Agent":
        from .agent import Agent

        return Agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Agent"]