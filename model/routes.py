"""HTTP 路由：只做入参与序列化，数值过程全部委托给 model 各模块。

路由（无版本前缀、只回 JSON，没有网页界面）：

- ``GET  /health``                       健康检查 + 预置工况自检
- ``POST /infiltrate``                   单点：累积入渗量、入渗率、残差
- ``POST /ponding``                      积水时刻与「是否会积水」判定
- ``POST /series``                       长历时点列（后台可取消作业）
- ``GET  /series/<job_id>``              作业状态/结果
- ``POST /series/<job_id>/cancel``       取消作业
- ``GET  /profiles``                     工况列表
- ``PUT  /profiles/<name>``              建档/覆盖
- ``GET  /profiles/<name>``              取名工况
- ``DELETE /profiles/<name>``            删除工况
"""

from __future__ import annotations

import math
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from .errors import ServiceError
from .infiltration import analyze_ponding, state_at_time
from .jobs import JobManager
from .profiles import LOAM, SoilProfile, seed_defaults
from .solver import solve_cumulative
from .validation import (
    validate_rainfall,
    validate_soil_params,
    validate_time,
)

api = Blueprint("api", __name__)


# --------------------------------------------------------------------------- #
# 请求解析辅助
# --------------------------------------------------------------------------- #

def _json_body() -> dict[str, Any]:
    if not request.is_json:
        raise ServiceError("invalid_body", "Content-Type 必须是 application/json",
                           status_code=415)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ServiceError("invalid_body", "请求体必须是 JSON 对象")
    return data


def _solver_options(data: dict[str, Any]) -> dict[str, float | int | None]:
    opts: dict[str, float | int | None] = {}
    for key in ("abs_tol", "rel_tol"):
        if key in data:
            v = data[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ServiceError("invalid_parameter", f"{key} 必须是数值")
            if not (math.isfinite(float(v)) and float(v) > 0.0):
                raise ServiceError("invalid_parameter", f"{key} 必须为正")
            opts[key] = float(v)
    if "max_iter" in data:
        v = data["max_iter"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ServiceError("invalid_parameter", "max_iter 必须是正整数")
        opts["max_iter"] = int(v)
    return opts


def _soil_inputs(data: dict[str, Any]) -> dict[str, Any]:
    """从 body 直接取土壤参数；若给了 profile 名则以档案为准。"""
    if "profile" in data and data["profile"] is not None:
        p = current_app.extensions["ga_profiles"].get(str(data["profile"]))
        return {"Ks": p.Ks, "psi": p.psi, "delta_theta": p.delta_theta,
                "profile_name": p.name}
    Ks, psi, dtheta = validate_soil_params(
        data.get("Ks"), data.get("psi"), data.get("delta_theta")
    )
    return {"Ks": Ks, "psi": psi, "delta_theta": dtheta, "profile_name": None}


def _ponding_dict(info: Any) -> dict[str, Any]:
    return {
        "will_pond": info.will_pond,
        "i": None if (isinstance(info.i, float) and math.isnan(info.i)) else info.i,
        "tp": info.tp,
        "Fp": info.Fp,
        "equivalent_time": info.equivalent_time,
        "explanation": info.explanation,
    }


def _state_body(st: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "t": st.t,
        "cumulative_infiltration_F": st.F,
        "infiltration_rate_f": st.f,
        "infiltration_capacity": st.f_capacity,
        "phase": st.phase,
        "mode": st.mode,
        "implicit_equation": {
            "residual": st.residual,
            "tolerance": st.tolerance,
            "iterations": st.iterations,
            "residual_within_tolerance": (
                None if st.residual is None else abs(st.residual) <= st.tolerance
            ),
        },
        "ponding": _ponding_dict(st.ponding),
        "rainfall_i": st.i,
    }
    if st.extra:
        body["elapsed_ponded_time"] = st.extra["elapsed_ponded_time"]
    return body


# --------------------------------------------------------------------------- #
# 健康检查
# --------------------------------------------------------------------------- #

@api.get("/health")
def health() -> Any:
    store = current_app.extensions["ga_profiles"]
    seed_defaults(store)
    loam_ok = False
    one_hour: dict[str, Any] | None = None
    try:
        p = store.get(LOAM.name)
        sol = solve_cumulative(p.Ks, p.psi, p.delta_theta, 1.0)
        loam_ok = (
            abs(sol.residual) <= sol.tolerance
            and sol.F > p.Ks * 1.0  # 吸力项作用：F 必须明显大于 Ks*t
        )
        one_hour = {
            "F": sol.F,
            "Ks_t": p.Ks,
            "residual": sol.residual,
            "tolerance": sol.tolerance,
        }
    except ServiceError:
        pass
    return jsonify({
        "status": "ok" if loam_ok else "degraded",
        "profiles_dir": str(store.dir),
        "loam_self_check": loam_ok,
        "loam_one_hour": one_hour,
    })


# --------------------------------------------------------------------------- #
# 单点入渗核算
# --------------------------------------------------------------------------- #

@api.post("/infiltrate")
def infiltrate() -> Any:
    data = _json_body()
    soil = _soil_inputs(data)
    t = validate_time(data.get("t"))
    opts = _solver_options(data)

    already_ponded = bool(data.get("already_ponded", False))
    i = validate_rainfall(data["i"]) if data.get("i") is not None else None

    st = state_at_time(
        soil["Ks"], soil["psi"], soil["delta_theta"], t,
        i=i, already_ponded=already_ponded, **opts,
    )
    body = _state_body(st)
    if soil["profile_name"]:
        body["profile"] = soil["profile_name"]
    body["Ks"] = soil["Ks"]
    body["psi"] = soil["psi"]
    body["delta_theta"] = soil["delta_theta"]
    return jsonify(body)


# --------------------------------------------------------------------------- #
# 积水时刻判定
# --------------------------------------------------------------------------- #

@api.post("/ponding")
def ponding() -> Any:
    data = _json_body()
    soil = _soil_inputs(data)
    i = validate_rainfall(data.get("i"))
    info = analyze_ponding(soil["Ks"], soil["psi"], soil["delta_theta"], i)
    return jsonify({
        "Ks": soil["Ks"],
        "psi": soil["psi"],
        "delta_theta": soil["delta_theta"],
        "rainfall_i": i,
        "will_pond": info.will_pond,
        "ponding_time_tp": info.tp,
        "F_at_ponding": info.Fp,
        "equivalent_ponded_time": info.equivalent_time,
        "explanation": info.explanation,
    })


# --------------------------------------------------------------------------- #
# 长历时点列（后台作业）
# --------------------------------------------------------------------------- #

def _parse_series_spec(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    soil = _soil_inputs(data)
    duration = validate_time(data.get("duration"), "duration")
    already_ponded = bool(data.get("already_ponded", False))

    n_points = data.get("n_points", 101)
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 2:
        raise ServiceError("invalid_parameter", "n_points 必须是 >= 2 的整数")
    if n_points > 100_000:
        raise ServiceError("invalid_parameter", "n_points 过大（上限 100000）")

    i = None
    if not already_ponded:
        if data.get("i") is None:
            raise ServiceError(
                "missing_parameter",
                "未声明 already_ponded 时必须给降雨强度 i（自由入渗+积水判定）",
            )
        i = validate_rainfall(data["i"])

    opts = _solver_options(data)
    params: dict[str, Any] = {
        "Ks": soil["Ks"], "psi": soil["psi"], "delta_theta": soil["delta_theta"],
        "i": i,
    }
    params.update(opts)
    series = {
        "duration": duration,
        "n_points": n_points,
        "already_ponded": already_ponded,
        "profile_name": soil["profile_name"],
    }
    return params, series


@api.post("/series")
def create_series() -> Any:
    data = _json_body()
    params, series = _parse_series_spec(data)
    mgr: JobManager = current_app.extensions["ga_jobs"]
    job_id = mgr.submit(params, series)
    body = mgr.status(job_id, include_points=False)
    body["links"] = {
        "status": f"/series/{job_id}",
        "cancel": f"/series/{job_id}/cancel",
    }
    return jsonify(body), 202


@api.get("/series/<job_id>")
def series_status(job_id: str) -> Any:
    mgr: JobManager = current_app.extensions["ga_jobs"]
    include = request.args.get("include_points", "1") not in ("0", "false", "False")
    return jsonify(mgr.status(job_id, include_points=include))


@api.post("/series/<job_id>/cancel")
def series_cancel(job_id: str) -> Any:
    mgr: JobManager = current_app.extensions["ga_jobs"]
    return jsonify(mgr.cancel(job_id)), 202


# --------------------------------------------------------------------------- #
# 工况建档
# --------------------------------------------------------------------------- #

@api.get("/profiles")
def list_profiles() -> Any:
    store = current_app.extensions["ga_profiles"]
    seed_defaults(store)
    names = store.list_names()
    return jsonify({"profiles": [store.get(n).to_dict() for n in names]})


@api.put("/profiles/<name>")
def put_profile(name: str) -> Any:
    data = _json_body()
    store = current_app.extensions["ga_profiles"]
    Ks, psi, dtheta = validate_soil_params(
        data.get("Ks"), data.get("psi"), data.get("delta_theta")
    )
    profile = store.create(
        name, Ks, psi, dtheta,
        description=str(data.get("description", "")),
        overwrite=True,
    )
    return jsonify(profile.to_dict())


@api.get("/profiles/<name>")
def get_profile(name: str) -> Any:
    store = current_app.extensions["ga_profiles"]
    profile: SoilProfile = store.get(name)
    return jsonify(profile.to_dict())


@api.delete("/profiles/<name>")
def delete_profile(name: str) -> Any:
    store = current_app.extensions["ga_profiles"]
    store.delete(name)
    return jsonify({"deleted": name})
