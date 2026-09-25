"""标定（反演）HTTP 路由：入参解析与序列化，数值过程委托 calibration 各模块。

路由（纯 JSON，无网页界面）：

- ``POST /calibrate``                       同步标定：观测 -> 参数 + 可信度
- ``POST /calibrations``                    提交标定后台作业，202 回 job_id
- ``GET  /calibrations/<job_id>``           作业状态/标定结果
- ``POST /calibrations/<job_id>/cancel``    取消作业（绝不交半成品参数）

标定结果可凭 ``save_profile`` 落成命名工况，与手工建档共用同一套
:class:`~model.profiles.ProfileStore` 存取；已有工况也能作为 ``initial``
初值起点（``initial_profile``）。
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, current_app, jsonify, request

from .calibration import (
    DEFAULT_CAL_MAX_ITER,
    validate_observations,
    validate_positive_param,
)
from .errors import ServiceError, ValidationError

cal_api = Blueprint("cal_api", __name__)


def _json_body() -> dict[str, Any]:
    if not request.is_json:
        raise ServiceError("invalid_body", "Content-Type 必须是 application/json",
                           status_code=415)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ServiceError("invalid_body", "请求体必须是 JSON 对象")
    return data


def _parse_fixed(data: dict[str, Any]) -> dict[str, float] | None:
    if "fixed" not in data or data["fixed"] is None:
        return None
    raw = data["fixed"]
    if not isinstance(raw, dict):
        raise ValidationError("invalid_parameter", "fixed 必须是对象")
    out: dict[str, float] = {}
    if "Ks" in raw and raw["Ks"] is not None:
        out["Ks"] = validate_positive_param(raw["Ks"], "fixed.Ks")
    # 兼容两种写法：A（psi·delta_theta 组合量）或直接给 psi+delta_theta
    if "A" in raw and raw["A"] is not None:
        out["A"] = validate_positive_param(raw["A"], "fixed.A")
    if "psi" in raw and raw["psi"] is not None:
        psi = validate_positive_param(raw["psi"], "fixed.psi")
        dtheta = float(raw.get("delta_theta", 1.0))
        if not (0.0 < dtheta <= 1.0):
            raise ValidationError("invalid_parameter",
                                  "fixed.delta_theta 必须落在 (0, 1]")
        if "A" in out:
            raise ValidationError("invalid_parameter", "fixed.A 与 fixed.psi 不能同时给")
        out["A"] = psi * dtheta
    if not out:
        raise ValidationError("invalid_parameter",
                              "fixed 至少要含 Ks、A 或 psi(+delta_theta) 之一")
    if len(out) == 2 and set(out) == {"Ks", "A"}:
        raise ValidationError("invalid_parameter", "两个参数都被固定则无需标定")
    return out


def _parse_initial(data: dict[str, Any]) -> dict[str, float] | None:
    raw: dict[str, Any] | None = None
    if isinstance(data.get("initial"), dict):
        raw = data["initial"]
    if data.get("initial_profile"):
        store = current_app.extensions["ga_profiles"]
        p = store.get(str(data["initial_profile"]))
        from_profile = {"Ks": p.Ks, "A": p.psi * p.delta_theta}
        raw = {**from_profile, **(raw or {})}
    if raw is None:
        return None
    out: dict[str, float] = {}
    if raw.get("Ks") is not None:
        out["Ks"] = validate_positive_param(raw["Ks"], "initial.Ks")
    if raw.get("A") is not None:
        out["A"] = validate_positive_param(raw["A"], "initial.A")
    if raw.get("psi") is not None:
        psi = validate_positive_param(raw["psi"], "initial.psi")
        dtheta = float(raw.get("delta_theta", 1.0))
        if not (0.0 < dtheta <= 1.0):
            raise ValidationError("invalid_parameter",
                                  "initial.delta_theta 必须落在 (0, 1]")
        out["A"] = psi * dtheta
    if not out:
        return None
    return out


def _parse_calibration(data: dict[str, Any]) -> dict[str, Any]:
    """把请求解析成标定作业规格；一切入箱校验发生在这里、迭代之前。"""
    obs = validate_observations(data.get("observations"))

    i = None
    if data.get("i") is not None:
        i = validate_positive_param(data["i"], "i")

    spec: dict[str, Any] = {
        "observations": [{"t": o.t, "F": o.F} for o in obs],
        "n_observations": len(obs),
        "i": i,
        "mode": "rainfall" if i is not None else "ponded",
        "fixed": _parse_fixed(data),
        "initial": _parse_initial(data),
    }

    if "max_iter" in data and data["max_iter"] is not None:
        v = data["max_iter"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValidationError("invalid_parameter", "max_iter 必须是正整数")
        spec["max_iter"] = int(v)
    else:
        spec["max_iter"] = DEFAULT_CAL_MAX_ITER

    # 结果落工况档案：只有分别辨识出 Ks、A 时才允许（组合不可辨无法建档）。
    # A 拆成 psi/delta_theta 必须补一个 delta_theta（或 psi）。
    save = data.get("save_profile")
    if save is not None:
        if not isinstance(save, dict):
            raise ValidationError("invalid_parameter", "save_profile 必须是对象")
        name = save.get("name")
        if not isinstance(name, str) or not name:
            raise ValidationError("invalid_parameter",
                                  "save_profile.name 必填，用于命名工况")
        split = save.get("delta_theta", save.get("psi"))
        if split is None:
            raise ValidationError(
                "missing_parameter",
                "把标定结果存成工况需补 delta_theta 或 psi 之一，"
                "以便把可辨识的 A=psi·delta_theta 拆开",
            )
        spec["save_profile"] = {
            "name": str(name),
            "delta_theta": (float(save["delta_theta"])
                            if save.get("delta_theta") is not None else None),
            "psi": float(save["psi"]) if save.get("psi") is not None else None,
            "description": str(save.get("description", "")),
        }
    return spec


def save_profile_if_requested(spec: dict[str, Any], result: dict[str, Any]) -> None:
    """标定成功且要求建档时，把结果写进与手工工况同一套仓库。

    同步路由与后台作业完成回调共用本函数（后者在 app context 内调用）。
    """
    save = spec.get("save_profile")
    if not save:
        result["saved_profile"] = None
        return
    fitted = result["fitted"]
    if fitted.get("Ks") is None or fitted.get("A") is None:
        raise ServiceError(
            "unidentifiable_for_profile",
            "本次标定只能定出 Ks·A 组合、单参不可辨，无法落成完整工况；"
            "请固定其一后重标，或提供含自由入渗段的观测打破简并",
            status_code=409,
            details={"identifiability": result["identifiability"]["status"]},
        )
    Ks, A = fitted["Ks"], fitted["A"]
    if save["delta_theta"] is not None:
        dtheta = save["delta_theta"]
        if not (0.0 < dtheta <= 1.0):
            raise ValidationError("invalid_parameter",
                                  "delta_theta 必须落在 (0, 1]")
        psi = A / dtheta
    else:
        psi = save["psi"]
        if psi <= 0.0:
            raise ValidationError("invalid_parameter", "psi 必须为正")
        dtheta = A / psi
        if not (0.0 < dtheta <= 1.0):
            raise ValidationError(
                "invalid_parameter",
                f"由 A={A:g} 与 psi={psi:g} 推出 delta_theta={dtheta:g}，"
                "不在 (0, 1] 内，请改用 delta_theta 拆分",
            )
    store = current_app.extensions["ga_profiles"]
    profile = store.create(save["name"], Ks, psi, dtheta,
                           description=save["description"], overwrite=True)
    result["saved_profile"] = profile.to_dict()


# --------------------------------------------------------------------------- #
# 同步标定
# --------------------------------------------------------------------------- #

@cal_api.post("/calibrate")
def calibrate() -> Any:
    data = _json_body()
    spec = _parse_calibration(data)
    mgr = current_app.extensions["ga_calibration_jobs"]
    # 同步路径：数值层异常（ConvergenceError/CalibrationError/ValidationError）
    # 由全局 errorhandler 转成统一错误结构
    result = mgr.run_inline(spec)
    save_profile_if_requested(spec, result)
    return jsonify(result)


# --------------------------------------------------------------------------- #
# 后台标定作业
# --------------------------------------------------------------------------- #

@cal_api.post("/calibrations")
def create_calibration() -> Any:
    data = _json_body()
    spec = _parse_calibration(data)
    mgr = current_app.extensions["ga_calibration_jobs"]
    job_id = mgr.submit(spec)
    body = mgr.status(job_id)
    body["links"] = {
        "status": f"/calibrations/{job_id}",
        "cancel": f"/calibrations/{job_id}/cancel",
    }
    return jsonify(body), 202


@cal_api.get("/calibrations/<job_id>")
def calibration_status(job_id: str) -> Any:
    mgr = current_app.extensions["ga_calibration_jobs"]
    return jsonify(mgr.status(job_id))


@cal_api.post("/calibrations/<job_id>/cancel")
def calibration_cancel(job_id: str) -> Any:
    mgr = current_app.extensions["ga_calibration_jobs"]
    return jsonify(mgr.cancel(job_id)), 202
