"""模拟车量状态(VehicleState)：演示用内存快照，指令执行器读写它。

真实项目里替换为读取车身状态(空调温度/车速/门窗/座椅传感器)，接口一致即可。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VehicleState:
    cabin_temp: float = 26.0          # 车厢当前温度 °C
    climate_set: float = 26.0         # 空调设定温度 °C
    ac_on: bool = False
    fan_level: int = 0                # 0-7
    windows: dict[str, int] = field(default_factory=lambda: {"driver": 0, "passenger": 0, "left_rear": 0, "right_rear": 0})
    seat_heat: dict[str, int] = field(default_factory=lambda: {"driver": 0, "passenger": 0})
    volume: int = 12                  # 媒体音量 0-100
    media_playing: bool = False
    headlight: bool = False           # 近光灯
    ambient_light: str = "off"
    drive_mode: str = "comfort"
    cruise_on: bool = False
    gear: str = "P"
    speed: int = 0                    # km/h
    range_km: int = 480

    def snapshot(self) -> dict[str, Any]:
        return {
            "cabin_temp": self.cabin_temp,
            "climate_set": self.climate_set,
            "ac_on": self.ac_on,
            "fan_level": self.fan_level,
            "windows": dict(self.windows),
            "seat_heat": dict(self.seat_heat),
            "volume": self.volume,
            "media_playing": self.media_playing,
            "headlight": self.headlight,
            "ambient_light": self.ambient_light,
            "drive_mode": self.drive_mode,
            "cruise_on": self.cruise_on,
            "gear": self.gear,
            "speed": self.speed,
            "range_km": self.range_km,
        }