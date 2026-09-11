"""安全门(Safety Gate)：对结构化指令做分级裁决，是"座舱 Agent 与通用 chatbot 拉开差距"的关键。

分级规则(可配置)：
- INFO      不执行车控，直接问答。
- COMFORT   舒适类控制，可立即执行。
- CONTROLLED 受控指令(改动车辆轨迹/需专注)，一律要求二次确认。
- FORBIDDEN 危险/越权指令与危险语义，直接拦截。

除按 域+动作 查表外，还叠加一层"危险词拦截"(针对 raw_input)，防止模型产出的
结构化指令动作名逃逸判定时，语义上仍是危险请求。
"""
from __future__ import annotations

import re
from typing import Iterable

from cockpit.schemas import Domain, SafetyLevel, SafetyVerdict, VehicleCommand

# ---- 受控指令：需要二次确认 ----
_CONTROLLED: set[tuple[Domain, str]] = {
    (Domain.NAV, "navigate_to"),
    (Domain.NAV, "set_favorite"),
    (Domain.NAV, "cancel"),
    (Domain.PARK, "auto_park"),
    (Domain.CRUISE, "engage"),
    (Domain.CRUISE, "set_cruise_speed"),
    (Domain.CRUISE, "disengage"),
    (Domain.VEHICLE, "set_charge_limit"),
    (Domain.VEHICLE, "drive_mode"),   # 切换运动/越野等
    (Domain.WINDOW, "open_all"),       # 全车开窗(雨天/高速风险)
    (Domain.CLIMATE, "rapid_cool"),    # 急速降温
}

# ---- 危险/越权指令：直接拦截 ----
_FORBIDDEN: set[tuple[Domain, str]] = {
    (Domain.SAFETY, "disable_esp"),
    (Domain.SAFETY, "disable_abs"),
    (Domain.SAFETY, "disable_airbag"),
    (Domain.SAFETY, "disable_lane_keep"),
    (Domain.SAFETY, "disable_blind_spot"),
    (Domain.SAFETY, "disable_emergency_brake"),
    (Domain.SAFETY, "override_speed_limit"),
    (Domain.VEHICLE, "unlock_while_driving"),
    (Domain.VEHICLE, "dismiss_warning"),
}

# ---- 信息查询：默认 info(以 get_ 前缀兜底) ----
_HINT_RE = re.compile(r"^(get|query|read|status)_", re.IGNORECASE)

# ---- 危险语义检索(对 raw_input, 防止动作名逃逸) ----
_FORBIDDEN_WORDS: tuple[str, ...] = (
    "超速", "不要刹", "松开刹车", "别刹车", "闯(红灯|红绿灯)",
    "安全气囊", "气囊", "关闭esp", "关掉esp", "esp", "关闭abs", "abs",
    "取消限速", "解除限速", "屏蔽告警",
    "关闭(车道保持|盲区|刹车辅助|自动刹车)",
)


class SafetyGate:
    def __init__(
        self,
        controlled: set[tuple[Domain, str]] | None = None,
        forbidden: set[tuple[Domain, str]] | None = None,
        forbid_words: Iterable[str] = (),
    ) -> None:
        self._controlled = controlled if controlled is not None else _CONTROLLED
        self._forbidden = forbidden if forbidden is not None else _FORBIDDEN
        self._forbid_words = list(forbid_words) or list(_FORBIDDEN_WORDS)

    def classify(self, cmd: VehicleCommand) -> SafetyVerdict:
        # 0) safety 域一律拦截(动作是否白名单都不允许下发)
        if cmd.domain == Domain.SAFETY:
            return SafetyVerdict(
                command=cmd,
                level=SafetyLevel.FORBIDDEN,
                allowed=False,
                reason="安全域指令，一律拦截。",
            )

        # 1) 危险语义优先(对原始输入), 优先级最高
        hit = self._scan_words(cmd.raw_input)
        if hit:
            return SafetyVerdict(
                command=cmd,
                level=SafetyLevel.FORBIDDEN,
                allowed=False,
                reason=f"检测到危险语义: 「{hit}」",
            )

        # 2) 域+动作查表
        key = (cmd.domain, cmd.action)
        if key in self._forbidden:
            return SafetyVerdict(
                command=cmd,
                level=SafetyLevel.FORBIDDEN,
                allowed=False,
                reason="该指令被判定为危险/越权操作。",
            )
        if key in self._controlled:
            return SafetyVerdict(
                command=cmd,
                level=SafetyLevel.CONTROLLED,
                allowed=True,
                requires_confirmation=True,
                reason="受控指令，须用户二次确认后方可执行。",
            )

        # 3) 信息查询
        if cmd.domain == Domain.INFO or _HINT_RE.match(cmd.action):
            return SafetyVerdict(
                command=cmd,
                level=SafetyLevel.INFO,
                allowed=True,
                reason="信息查询，无需执行车控。",
            )

        # 4) 其余默认舒适类
        return SafetyVerdict(
            command=cmd,
            level=SafetyLevel.COMFORT,
            allowed=True,
            reason="舒适类控制，可直接执行。",
        )

    def _scan_words(self, text: str) -> str | None:
        if not text:
            return None
        for pat in self._forbid_words:
            if re.search(pat, text):
                return pat
        return None