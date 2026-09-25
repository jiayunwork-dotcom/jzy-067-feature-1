"""标定（反演）HTTP 路由：只做入参与序列化，数值过程委托 model.calibration。

路由（只回 JSON）：

- ``POST /calibrate``                 提交标定作业（202 + job_id）
- ``GET  /calibrate/<job_id>``        标定状态/完整结果
- ``POST /calibrate/<job_id>/cancel`` 取消；取消后绝不交半成品参数
"""

from __future__ import annotations

import math
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from .calibration import validate_observations
from .errors import ServiceError, ValidationError
from .profiles import _safe_name
from .validation import validate_rainfall

cal_api = Blueprint("cal_api", __name__)


def _json_body() -> dict[str, Any]:
    if not request.is_json:
        raise ServiceError("invalid_body", "Content-Type 必须是 application/json",
                           status_code=415)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ServiceError("invalid_body", "请求体必须是 JSON 对象")
    return data


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("invalid_parameter", f"{field} 必须是数值")
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValidationError("invalid_parameter", f"{field} 必须为正有限数")
    return v


def _parse_fix(data: dict[str, Any]) -> dict[str, float]:
    raw = data.get("fix")
    if raw is None:
        return {}
    if not isinstance(raw, dict) or len(raw) != 1:
        raise ValidationError(
            "invalid_parameter", "fix 必须是恰含一个键的对象：{\"Ks\": v} 或 {\"A\": v}"
        )
    name = next(iter(raw))
    if name not in ("Ks", "A"):
        raise ValidationError("invalid_parameter", f"fix 只能固定 Ks 或 A，收到 {name!r}")
    return {name: _positive_number(raw[name], f"fix.{name}")}


def _parse_initial(data: dict[str, Any]) -> dict[str, float] | None:
    initial_raw = data.get("initial")
    profile_name = data.get("initial_profile")
    if initial_raw is not None and profile_name is not None:
        raise ValidationError(
            "invalid_parameter", "initial 与 initial_profile 只能给一个"
        )
    if profile_name is not None:
        store = current_app.extensions["ga_profiles"]
        p = store.get(str(profile_name))  # 不存在时 404
        return {"Ks": p.Ks, "A": p.psi * p.delta_theta}
    if initial_raw is None:
        return None
    if not isinstance(initial_raw, dict) or not initial_raw:
        raise ValidationError("invalid_parameter", "initial 必须是非空对象")
    unknown = set(initial_raw) - {"Ks", "A"}
    if unknown:
        raise ValidationError(
            "invalid_parameter", f"initial 只接受 Ks 或 A，收到 {sorted(unknown)}"
        )
    return {k: _positive_number(v, f"initial.{k}") for k, v in initial_raw.items()}


def _parse_split(data: dict[str, Any], save_as: str | None) -> dict[str, float]:
    has_dtheta = data.get("delta_theta") is not None
    has_psi = data.get("psi") is not None
    if has_dtheta and has_psi:
        raise ValidationError(
            "invalid_parameter", "delta_theta 与 psi 只能给一个（用于把 A 拆回档案参数）"
        )
    if save_as is not None and not (has_dtheta or has_psi):
        raise ValidationError(
            "missing_parameter",
            "save_as 建档需要同时给 delta_theta 或 psi 之一，"
            "把标定出的 A = psi·delta_theta 拆回档案参数",
        )
    if has_dtheta:
        v = _positive_number(data["delta_theta"], "delta_theta")
        if v > 1.0:
            raise ValidationError("invalid_parameter",
                                  "delta_theta 必须落在 (0, 1]")
        return {"delta_theta": v}
    if has_psi:
        return {"psi": _positive_number(data["psi"], "psi")}
    return {}


def _parse_options(data: dict[str, Any]) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if "max_iter" in data:
        v = data["max_iter"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValidationError("invalid_parameter", "max_iter 必须是正整数")
        if v > 10000:
            raise ValidationError("invalid_parameter", "max_iter 过大（上限 10000）")
        opts["max_iter"] = int(v)
    for key in ("gtol", "xtol", "ftol"):
        if key in data:
            opts[key] = _positive_number(data[key], key)
    return opts


@cal_api.post("/calibrate")
def create_calibration() -> Any:
    data = _json_body()

    already_ponded = bool(data.get("already_ponded", False))
    i = None
    if data.get("i") is not None:
        if already_ponded:
            raise ValidationError(
                "invalid_parameter",
                "already_ponded 与降雨强度 i 矛盾：要么自始积水，要么降雨分段",
            )
        i = validate_rainfall(data["i"])

    fixed = _parse_fix(data)
    n_free = 2 - len(fixed)
    observations = validate_observations(data.get("observations"), n_free=n_free)

    initial = _parse_initial(data)
    if i is not None:
        if fixed.get("Ks", 0.0) >= i:
            raise ValidationError(
                "invalid_parameter",
                "固定的 Ks 不小于降雨强度 i：永不积水，观测对参数无信号",
            )
        if initial and initial.get("Ks", 0.0) >= i:
            raise ValidationError(
                "invalid_parameter", "初值 Ks 必须小于降雨强度 i（无信号死区）"
            )

    save_as = data.get("save_as")
    if save_as is not None:
        save_as = _safe_name(str(save_as))
    split = _parse_split(data, save_as)
    options = _parse_options(data)

    obs_echo = [{"t": ob.t, "F": ob.F} for ob in observations]
    spec: dict[str, Any] = {
        "observations": observations,  # 每份作业自己的观测对象列表
        "i": i,
        "fixed": fixed,
        "initial": initial,
        "split": split,
        "save_as": save_as,
        "options": options,
        "request_echo": {
            "observations": obs_echo,
            "i": i,
            "fix": fixed or None,
            "initial": initial,
            "save_as": save_as,
            "options": options,
        },
    }
    mgr = current_app.extensions["ga_caljobs"]
    job_id = mgr.submit(spec)
    body = mgr.status(job_id)
    body["links"] = {
        "status": f"/calibrate/{job_id}",
        "cancel": f"/calibrate/{job_id}/cancel",
    }
    return jsonify(body), 202


@cal_api.get("/calibrate/<job_id>")
def calibration_status(job_id: str) -> Any:
    mgr = current_app.extensions["ga_caljobs"]
    return jsonify(mgr.status(job_id))


@cal_api.post("/calibrate/<job_id>/cancel")
def calibration_cancel(job_id: str) -> Any:
    mgr = current_app.extensions["ga_caljobs"]
    return jsonify(mgr.cancel(job_id)), 202
