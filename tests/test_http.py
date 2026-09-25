"""HTTP 路由端到端测试。"""

from __future__ import annotations

import time

from model.hydrograph import HydrographCancelled

LOAM = {"Ks": 1.09, "psi": 11.01, "delta_theta": 0.434}


def _poll(client, job_id, states, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/series/{job_id}")
        body = r.get_json()
        if body["state"] in states:
            return body
        time.sleep(0.01)
    raise AssertionError(f"作业 {job_id} 未到达 {states}")


# --------------------------------------------------------------------------- #
# 健康检查与预置工况
# --------------------------------------------------------------------------- #

def test_health_loam_self_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert body["loam_self_check"] is True
    assert body["loam_one_hour"]["F"] > 1.5 * body["loam_one_hour"]["Ks_t"]


def test_profiles_seeded_loam(client):
    r = client.get("/profiles")
    assert r.status_code == 200
    names = [p["name"] for p in r.get_json()["profiles"]]
    assert "loam" in names


# --------------------------------------------------------------------------- #
# 单点入渗
# --------------------------------------------------------------------------- #

def test_infiltrate_happy_path(client):
    r = client.post("/infiltrate", json={**LOAM, "t": 1.0})
    assert r.status_code == 200
    body = r.get_json()
    F = body["cumulative_infiltration_F"]
    assert F > LOAM["Ks"] * 1.0  # 吸力项：F > Ks*t
    imp = body["implicit_equation"]
    assert abs(imp["residual"]) <= imp["tolerance"]
    assert imp["residual_within_tolerance"] is True
    assert body["infiltration_rate_f"] >= LOAM["Ks"]
    assert body["phase"] == "ponded"


def test_infiltrate_zero_time(client):
    r = client.post("/infiltrate", json={**LOAM, "t": 0.0})
    body = r.get_json()
    assert body["cumulative_infiltration_F"] == 0.0
    assert body["infiltration_rate_f"] == 1.0e6  # 约定初值封顶


def test_infiltrate_invalid_params_blocked_before_iteration(client):
    cases = [
        {**LOAM, "Ks": -1.0, "t": 1.0},
        {**LOAM, "psi": 0.0, "t": 1.0},
        {**LOAM, "delta_theta": 0.0, "t": 1.0},
        {**LOAM, "delta_theta": 1.5, "t": 1.0},
        {**LOAM, "t": -0.5},
        {**LOAM, "Ks": True, "t": 1.0},
    ]
    for payload in cases:
        r = client.post("/infiltrate", json=payload)
        assert r.status_code == 400, payload
        body = r.get_json()
        assert body["error"]["code"] == "invalid_parameter"
        assert body["error"]["reason"]
        assert "cumulative_infiltration_F" not in body


def test_infiltrate_non_convergence_returns_422_not_guess(client):
    r = client.post("/infiltrate", json={**LOAM, "t": 1.0, "max_iter": 1})
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"]["code"] == "not_converged"
    assert "cumulative_infiltration_F" not in body  # 没解出来就不给 F


def test_infiltrate_with_rainfall_phases(client):
    # 积水前
    r = client.post("/infiltrate", json={**LOAM, "t": 0.05, "i": 5.0})
    body = r.get_json()
    assert body["phase"] == "free"
    assert body["cumulative_infiltration_F"] == 5.0 * 0.05
    assert body["implicit_equation"]["residual"] is None

    # 积水后
    r = client.post("/infiltrate", json={**LOAM, "t": 3.0, "i": 5.0})
    body = r.get_json()
    assert body["phase"] == "ponded"
    assert abs(body["implicit_equation"]["residual"]) <= \
        body["implicit_equation"]["tolerance"]
    assert body["infiltration_rate_f"] < 5.0  # 能力已低于供水


def test_infiltrate_via_profile_name(client):
    r = client.post("/infiltrate", json={"profile": "loam", "t": 1.0})
    assert r.status_code == 200
    assert r.get_json()["profile"] == "loam"
    r = client.post("/infiltrate", json={"profile": "nope", "t": 1.0})
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# 积水判定
# --------------------------------------------------------------------------- #

def test_ponding_never_when_i_le_Ks(client):
    for i in (0.5, 1.09):
        r = client.post("/ponding", json={**LOAM, "i": i})
        body = r.get_json()
        assert body["will_pond"] is False
        assert body["ponding_time_tp"] is None
        assert "不会积水" in body["explanation"] or "永远" in body["explanation"]


def test_ponding_time_when_i_gt_Ks(client):
    r = client.post("/ponding", json={**LOAM, "i": 5.0})
    body = r.get_json()
    assert body["will_pond"] is True
    tp = body["ponding_time_tp"]
    Fp = body["F_at_ponding"]
    assert tp > 0.0
    assert Fp == tp * 5.0
    # 积水时刻能力等于供水
    A = LOAM["psi"] * LOAM["delta_theta"]
    assert LOAM["Ks"] * (1 + A / Fp) == 5.0


def test_ponding_invalid_i(client):
    r = client.post("/ponding", json={**LOAM, "i": -1.0})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_parameter"


# --------------------------------------------------------------------------- #
# 点列作业
# --------------------------------------------------------------------------- #

def test_series_job_lifecycle(client):
    r = client.post("/series", json={**LOAM, "duration": 3.0, "n_points": 101,
                                     "i": 5.0})
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    body = _poll(client, job_id, {"completed"})
    assert body["complete"] is True
    assert len(body["points"]) == 101
    assert body["phase_switches"]
    assert body["points"][0]["F"] == 0.0
    # 点列里 F 单调不减
    Fs = [p["F"] for p in body["points"]]
    assert all(b >= a for a, b in zip(Fs, Fs[1:]))


def test_series_job_cancel_never_partial(client, app):
    def fake_runner(*args, **kwargs):
        raise HydrographCancelled([{"t": 0.0, "F": 0.0}], 0)

    app.extensions["ga_jobs"].set_runner(fake_runner)
    r = client.post("/series", json={**LOAM, "duration": 3.0, "n_points": 101,
                                     "i": 5.0})
    job_id = r.get_json()["job_id"]
    body = _poll(client, job_id, {"cancelled"})
    assert body["complete"] is False
    assert body["points"] == []  # 半截点列绝不交出


def test_series_job_failure_422ish_state(client):
    r = client.post("/series", json={**LOAM, "duration": 3.0, "n_points": 10,
                                     "i": 5.0, "max_iter": 1})
    job_id = r.get_json()["job_id"]
    body = _poll(client, job_id, {"failed"})
    assert body["error"]["code"] == "not_converged"
    assert body["complete"] is False


def test_series_unknown_job_404(client):
    assert client.get("/series/deadbeef").status_code == 404
    assert client.post("/series/deadbeef/cancel").status_code == 404


def test_series_missing_i_rejected(client):
    r = client.post("/series", json={**LOAM, "duration": 3.0, "n_points": 10})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "missing_parameter"


# --------------------------------------------------------------------------- #
# 工况建档 HTTP
# --------------------------------------------------------------------------- #

def test_profile_crud_over_http(client):
    r = client.put("/profiles/sand", json={"Ks": 8.25, "psi": 4.95,
                                           "delta_theta": 0.417,
                                           "description": "砂土"})
    assert r.status_code == 200
    r = client.get("/profiles/sand")
    assert r.get_json()["Ks"] == 8.25

    r = client.post("/infiltrate", json={"profile": "sand", "t": 1.0})
    assert r.status_code == 200
    assert r.get_json()["cumulative_infiltration_F"] > 8.25

    r = client.delete("/profiles/sand")
    assert r.status_code == 200
    assert client.get("/profiles/sand").status_code == 404


def test_profile_invalid_params_400(client):
    r = client.put("/profiles/bad", json={"Ks": -1.0, "psi": 10.0,
                                          "delta_theta": 0.4})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_parameter"
