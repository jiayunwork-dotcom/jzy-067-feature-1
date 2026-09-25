"""土壤参数与入参校验。

校验必须在任何迭代发生之前完成：

- ``Ks``（饱和导水率）必须为正的有限数；
- ``psi``（湿润锋基质吸力，取正数值）必须为正的有限数；
- ``delta_theta``（锋前后体积含水量差）必须落在 (0, 1]；
- 时间不得为负；
- 降雨强度 ``i`` 若给出，必须为正的有限数。

布尔值在 Python 里也是 ``int``，这里明确拒绝，避免 ``True`` 被当成 1 混过去。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .errors import ValidationError


def _number(value: Any, field: str) -> float:
    """取出一个有限实数，拒绝布尔、None 与无法转成浮点的值。"""
    if isinstance(value, bool):
        raise ValidationError("invalid_parameter", f"参数 {field} 必须是数值，收到布尔值")
    if value is None:
        raise ValidationError("missing_parameter", f"缺少必填参数 {field}")
    if not isinstance(value, (int, float)):
        raise ValidationError(
            "invalid_parameter", f"参数 {field} 必须是数值，收到 {type(value).__name__}"
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise ValidationError("invalid_parameter", f"参数 {field} 必须是有限数值")
    return fv


def validate_soil_params(
    Ks: Any, psi: Any, delta_theta: Any
) -> tuple[float, float, float]:
    """校验三元土壤参数，返回干净的 float 三元组。"""
    Ks_f = _number(Ks, "Ks")
    psi_f = _number(psi, "psi")
    dtheta_f = _number(delta_theta, "delta_theta")

    if Ks_f <= 0.0:
        raise ValidationError("invalid_parameter", "饱和导水率 Ks 必须为正数")
    if psi_f <= 0.0:
        raise ValidationError("invalid_parameter", "湿润锋基质吸力 psi 必须为正数")
    if not (0.0 < dtheta_f <= 1.0):
        raise ValidationError(
            "invalid_parameter", "含水量差 delta_theta 必须落在开区间 (0, 1] 内"
        )
    return Ks_f, psi_f, dtheta_f


def validate_time(t: Any, field: str = "t") -> float:
    """校验时刻，时间不得为负。"""
    tf = _number(t, field)
    if tf < 0.0:
        raise ValidationError("invalid_parameter", f"时间 {field} 不能为负")
    return tf


def validate_rainfall(i: Any) -> float:
    """校验降雨强度（供积水判定用），必须为正的有限数。"""
    iv = _number(i, "i")
    if iv <= 0.0:
        raise ValidationError("invalid_parameter", "降雨强度 i 必须为正数")
    return iv


def parse_soil_params(
    data: Mapping[str, Any], *, rainfall: bool = False
) -> dict[str, Any]:
    """从请求映射里取出并校验土壤参数（及可选降雨强度）。"""
    if not isinstance(data, Mapping):
        raise ValidationError("invalid_body", "请求体必须是 JSON 对象")
    Ks, psi, delta_theta = validate_soil_params(
        data.get("Ks"), data.get("psi"), data.get("delta_theta")
    )
    parsed: dict[str, Any] = {
        "Ks": Ks,
        "psi": psi,
        "delta_theta": delta_theta,
    }
    if rainfall:
        parsed["i"] = validate_rainfall(data.get("i"))
    return parsed
