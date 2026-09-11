"""cockpit: 座舱任务规划 Agent 的指令层。

核心数据流:
    raw_input -> (意图理解, 待接) -> VehicleCommand
                                  -> SafetyGate.classify() -> SafetyVerdict
                                    - info       : 直接问答
                                    - comfort    : 直接执行
                                    - controlled : 二次确认后执行
                                    - forbidden  : 拦截
"""

from cockpit.schemas import Action, Domain, SafetyLevel, SafetyVerdict, VehicleCommand
from cockpit.safety import SafetyGate
from cockpit.state import VehicleState
from cockpit.commands import CommandExecutor, build_executor
from cockpit.controller import CockpitController, CockpitOutcome

__all__ = [
    "Action",
    "Domain",
    "SafetyLevel",
    "SafetyVerdict",
    "VehicleCommand",
    "SafetyGate",
    "VehicleState",
    "CommandExecutor",
    "build_executor",
    "CockpitController",
    "CockpitOutcome",
]