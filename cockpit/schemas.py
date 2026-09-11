"""座舱指令统一数据模型：把 Agent 的自由语义命令收敛为结构化、可校验、可安全分类的指令。

设计要点(座舱面试可深挖)：
- 车控与信息查询分离(Domain/ACTION)，外部执行层只认强类型 JSON，不信任自由文本。
- 所有指令经过 SafetyGate 分类后才允许下发，安全是独立于 LLM 的一道规则闸。
- VehicleCommand 是"产物"，SafetyVerdict 是"裁决结果"，两者是安全链路的核心结构。
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Domain(str, Enum):
    """车内能力域。新增域只需在此登记 + 在 commands.py 注册执行器。"""

    CLIMATE = "climate"      # 空调/温控
    WINDOW = "window"        # 车窗/天窗
    SEAT = "seat"            # 座椅
    MEDIA = "media"          # 多媒体
    NAV = "nav"              # 导航
    PARK = "park"            # 泊车
    CRUISE = "cruise"        # 巡航/辅助驾驶
    LIGHT = "light"          # 灯光
    VEHICLE = "vehicle"      # 整车状态/驱动模式
    INFO = "info"            # 信息查询(不产生车控)
    SAFETY = "safety"        # 安全相关(多数被 gate 判为 forbidden)


class Action(str, Enum):
    """动作白名单：模型只能从中选，杜绝自由文本逃逸安全判定。"""

    # climate
    SET_TEMPERATURE = "set_temperature"
    SET_FAN_SPEED = "set_fan_speed"
    TURN_ON = "turn_on"
    TURN_OFF = "turn_off"
    RAPID_COOL = "rapid_cool"
    # window
    OPEN = "open"
    CLOSE = "close"
    OPEN_ALL = "open_all"
    CLOSE_ALL = "close_all"
    # seat
    SET_HEAT = "set_heat"
    # media
    SET_VOLUME = "set_volume"
    PLAY = "play"
    PAUSE = "pause"
    # nav / park / cruise / light / vehicle
    NAVIGATE_TO = "navigate_to"
    AUTO_PARK = "auto_park"
    ENGAGE = "engage"
    DISENGAGE = "disengage"
    SET_CRUISE_SPEED = "set_cruise_speed"
    SET_INTERIOR_LIGHT = "set_interior_light"
    SET_AMBIENT = "set_ambient"
    SET_CHARGE_LIMIT = "set_charge_limit"
    DRIVE_MODE = "drive_mode"
    # read / query (info)
    GET_STATUS = "get_status"
    GET_SPEED = "get_speed"
    GET_RANGE = "get_range"
    GET_CLIMATE = "get_climate"
    QUERY_ALL = "query_all"


class SafetyLevel(str, Enum):
    """安全分级：安全门对每条指令给出的等级判定。"""

    INFO = "info"              # 仅信息查询，不执行任何车控
    COMFORT = "comfort"        # 舒适类控制，可立即执行
    CONTROLLED = "controlled"  # 受控指令，需用户二次确认后执行
    FORBIDDEN = "forbidden"    # 危险/越权，一律拦截


class VehicleCommand(BaseModel):
    """一条结构化车内指令。Agent 必须产成此结构，SafetyGate 基于它裁决。"""

    domain: Domain
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    target: str | None = None        # 温区/车窗/座位/灯光分区等
    priority: int = 0                # 0=默认, 大=越高优先
    raw_input: str = ""              # 触发该指令的原始语音/文字(用于安全词拦截与审计)


class SafetyVerdict(BaseModel):
    """安全门裁决结果。allowed=False 时命令不得下发执行。"""

    command: VehicleCommand
    level: SafetyLevel
    allowed: bool = False
    requires_confirmation: bool = False
    reason: str = ""

    @property
    def summary(self) -> str:
        """给人看的一句话裁决摘要(面试/演示话术)。"""
        suffix = ["无需确认" if not self.requires_confirmation else "需用户二次确认"]
        status = (
            "允许执行"
            if self.allowed
            else ("拦截" if self.level == SafetyLevel.FORBIDDEN else "暂缓/需确认")
        )
        return f"[{self.level.value}] {self.command.domain.value}/{self.command.action} → {status}；{suffix[0]}。{self.reason}"