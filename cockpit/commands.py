"""座舱指令执行器：把通过安全门裁决后的结构指令落地到 VehicleState，返回人话反馈。

动作名与 schemas.Action 白名单对齐，避免自由文本导致执行器对不上。
所有只读动作(get_*)统一走 _handle_read，不随域改变。
"""
from __future__ import annotations

from cockpit.schemas import VehicleCommand
from cockpit.state import VehicleState

_READ_ACTIONS = {"get_status", "get_speed", "get_range", "get_climate", "query_all", "get_eta", "get_sensors"}


class CommandExecutor:
    """根据 (domain, action) 分发到 handler，更新模拟车状态并返回人话反馈。"""

    def __init__(self, state: VehicleState | None = None) -> None:
        self.state = state or VehicleState()
        # 动作分发表：通过安全门后按动作执行，避免依赖模型可能标错的域标签。
        self._action_map = {
            # climate
            "set_temperature": self._on_climate, "set_fan_speed": self._on_climate,
            "turn_on": self._on_climate, "turn_off": self._on_climate, "rapid_cool": self._on_climate,
            # window
            "open": self._on_window, "close": self._on_window,
            "open_all": self._on_window, "close_all": self._on_window,
            # seat
            "set_heat": self._on_seat,
            # media
            "set_volume": self._on_media, "play": self._on_media, "pause": self._on_media,
            # vehicle
            "drive_mode": self._on_vehicle, "set_charge_limit": self._on_vehicle,
            # cruise
            "engage": self._on_cruise, "disengage": self._on_cruise, "set_cruise_speed": self._on_cruise,
            # park / nav / light
            "auto_park": self._on_park, "navigate_to": self._on_nav,
            "set_interior_light": self._on_light, "set_ambient": self._on_light,
        }

    def execute(self, cmd: VehicleCommand) -> str:
        if cmd.action in _READ_ACTIONS:
            return self._handle_read(cmd.action)
        handler = self._action_map.get(cmd.action)
        if handler:
            return handler(cmd)
        fallback = getattr(self, f"_on_{cmd.domain.value}", None)
        return (
            fallback(cmd)
            if fallback is not None
            else f"未实现域 {cmd.domain.value} 的执行器，指令未执行。"
        )

    # ---------- 只读(统一处理, 不随域) ----------
    def _handle_read(self, action: str) -> str:
        s = self.state.snapshot()
        if action == "get_speed":
            return f"当前车速 {s['speed']} km/h。"
        if action == "get_range":
            return f"续航约 {s['range_km']} 公里。"
        if action == "get_climate":
            return f"车厢 {s['cabin_temp']:.1f}℃，设定 {s['climate_set']:.1f}℃，空调{'开' if s['ac_on'] else '关'}，风量 {s['fan_level']}。"
        if action in ("get_eta",):
            return "预计还需 18 分钟到达目的地。"
        if action in ("get_sensors",):
            return "车身传感器正常，周围无障碍物。"
        return f"车辆状态快照: {s}"

    # ---------- 空调 ----------
    def _on_climate(self, cmd: VehicleCommand) -> str:
        s = self.state
        a = cmd.action
        if a == "set_temperature":
            v = float(cmd.params.get("temperature", cmd.target or 24))
            s.climate_set = v
            s.cabin_temp = v
            s.ac_on = True
            return f"空调温度设为 {v:.1f}℃(已自动开制冷)。"
        if a == "set_fan_speed":
            s.fan_level = max(0, min(7, int(cmd.params.get("fan_level", cmd.params.get("level", 3)))))
            s.ac_on = True
            return f"空调风量调到 {s.fan_level} 档。"
        if a == "turn_on":
            s.ac_on = True
            return "空调已开启。"
        if a == "turn_off":
            s.ac_on = False
            return "空调已关闭。"
        if a == "rapid_cool":
            s.climate_set = 18.0
            s.fan_level = 7
            s.ac_on = True
            return "已进入急速降温(温度 18℃/满风量)。"
        return "空调指令未识别。"

    # ---------- 车窗 ----------
    def _on_window(self, cmd: VehicleCommand) -> str:
        s = self.state
        zone = cmd.target or "driver"
        a = cmd.action
        if a == "open":
            s.windows[zone] = 100
            return f"{zone} 车窗已打开。"
        if a == "close":
            s.windows[zone] = 0
            return f"{zone} 车窗已关闭。"
        if a == "open_all":
            for k in s.windows:
                s.windows[k] = 100
            return "全车车窗已打开。"
        if a == "close_all":
            for k in s.windows:
                s.windows[k] = 0
            return "全车车窗已关闭。"
        return "车窗指令未识别。"

    # ---------- 座椅 ----------
    def _on_seat(self, cmd: VehicleCommand) -> str:
        s = self.state
        zone = cmd.target or "driver"
        if cmd.action == "set_heat":
            v = max(0, min(3, int(cmd.params.get("level", 1))))
            s.seat_heat[zone] = v
            return f"{zone} 座椅加热调到 {v} 档。"
        return "座椅指令未识别。"

    # ---------- 多媒体 ----------
    def _on_media(self, cmd: VehicleCommand) -> str:
        s = self.state
        a = cmd.action
        if a == "set_volume":
            s.volume = max(0, min(100, int(cmd.params.get("volume", 12))))
            return f"音量调到 {s.volume}。"
        if a == "play":
            s.media_playing = True
            return "开始播放。"
        if a == "pause":
            s.media_playing = False
            return "已暂停。"
        return "媒体指令未识别。"

    # ---------- 车辆整车 ----------
    def _on_vehicle(self, cmd: VehicleCommand) -> str:
        s = self.state
        a = cmd.action
        if a == "drive_mode":
            s.drive_mode = cmd.params.get("mode", "comfort")
            return f"驱动模式切换为 {s.drive_mode}。"
        if a == "set_charge_limit":
            return f"充电上限设置为 {cmd.params.get('percent', 80)}%。"
        return "整车指令未识别。"

    # ---------- 巡航 ----------
    def _on_cruise(self, cmd: VehicleCommand) -> str:
        s = self.state
        a = cmd.action
        if a == "engage":
            s.cruise_on = True
            return "巡航已开启。"
        if a == "disengage":
            s.cruise_on = False
            return "巡航已退出。"
        if a == "set_cruise_speed":
            sp = int(cmd.params.get("speed", 100))
            s.speed = sp
            s.cruise_on = True
            return f"巡航速度设定为 {sp} km/h。"
        return "巡航指令未识别。"

    # ---------- 泊车 ----------
    def _on_park(self, cmd: VehicleCommand) -> str:
        if cmd.action == "auto_park":
            return "自动泊车已接管，正在规划入位轨迹……"
        return "泊车指令未识别。"

    # ---------- 导航 ----------
    def _on_nav(self, cmd: VehicleCommand) -> str:
        if cmd.action == "navigate_to":
            dest = cmd.params.get("destination") or cmd.target or "目的地"
            alias = cmd.params.get("_alias")
            note = f"(即「{alias}」)" if alias and alias != dest else ""
            return f"已规划前往「{dest}」的导航路线{note}。"
        return "导航指令未识别。"

    # ---------- 灯光 ----------
    def _on_light(self, cmd: VehicleCommand) -> str:
        s = self.state
        a = cmd.action
        if a == "set_interior_light":
            s.headlight = bool(cmd.params.get("on", True))
            return f"近光灯已{'开启' if s.headlight else '关闭'}。"
        if a == "set_ambient":
            s.ambient_light = cmd.params.get("color", "off")
            return f"氛围灯设为 {s.ambient_light}。"
        return "灯光指令未识别。"


def build_executor(state: VehicleState | None = None) -> CommandExecutor:
    return CommandExecutor(state)