"""标定 HTTP 路由端到端测试。

覆盖：同步标定的三种可辨识性处置、后台作业生命周期与取消、入箱校验、
标定结果落工况档案/已有工况当初值、错误结构统一。
"""

from __future__ import annotations

import time

from model.calibration import CalibrationCancelled
from model.infiltration import analyze_ponding, state_at_time

TRUE = {"Ks": 1.09, "psi": 11.01, "delta_theta": 0.434}
A_TRUE = TRUE["psi"] * TRUE["delta_theta"]
I = 5.0


def _rain_obs(ts=None, i=I):
    ts = ts or [0.02 * k for k in range(1, 41)]
    return [{"t": t,
             "F": state_at_time(TRUE["Ks"], TRUE["psi"], TRUE["delta_theta"],
                                t, i=i).F} for t in ts]


def _ponded_obs(ts=None):
    ts = ts or [0.02 * k for k in range(1, 16)]
    return [{"t": t,
             "F": state_at_time(TRUE["Ks"], TRUE["psi"], TRUE["delta_theta"],
                                t, already_ponded=True).F} for t in ts]


def _poll(client, job_id, states, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/calibrations/{job_id}").get_json()
        if body["state"] in states:
            return body
        time.sleep(0.01)
    raise AssertionError(f"标定作业 {job_id} 未到达 {states}")


# --------------------------------------------------------------------------- #
# 同步标定：可辨识数据高精度还原
# --------------------------------------------------------------------------- #

def test_calibrate_rainfall_recovers_parameters(client):
    r = client.post("/calibrate", json={"observations": _rain_obs(), "i": I})
    assert r.status_code == 200
    body = r.get_json()
    assert body["converged"] is True
    assert body["identifiability"]["status"] == "individual"
    assert body["fitted"]["Ks"] == pytest_rel(TRUE["Ks"], 1e-8)
    assert body["fitted"]["A"] == pytest_rel(A_TRUE, 1e-8)
    # 可复核量齐全
    assert body["sse"] >= 0.0
    assert body["iterations"] >= 1
    assert len(body["residuals"]) == 40
    assert body["mode"] == "rainfall"


def pytest_rel(target, rel):
    import pytest
    return pytest.approx(target, rel=rel)


def test_calibrate_ponded_only_reports_combination_only(client):
    r = client.post("/calibrate", json={"observations": _ponded_obs()})
    assert r.status_code == 200
    body = r.get_json()
    assert body["identifiability"]["status"] == "combination_only"
    # 绝不虚报精确单参
    assert body["fitted"]["Ks"] is None
    assert body["fitted"]["A"] is None
    assert body["ridge_point"] is not None
    assert body["identifiability"]["identifiable_combination"]["value"] > 0
    assert body["identifiability"]["note"]


def test_calibrate_with_fixed_Ks_recovers_A(client):
    r = client.post("/calibrate", json={
        "observations": _ponded_obs(),
        "fixed": {"Ks": TRUE["Ks"]},
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["identifiability"]["status"] == "individual"
    assert body["fitted"]["A"] == pytest_rel(A_TRUE, 1e-9)
    assert body["fixed"] == {"Ks": TRUE["Ks"]}


def test_calibrate_fixed_via_psi_delta_theta(client):
    r = client.post("/calibrate", json={
        "observations": _ponded_obs(),
        "fixed": {"psi": TRUE["psi"], "delta_theta": TRUE["delta_theta"]},
    })
    body = r.get_json()
    assert body["identifiability"]["status"] == "individual"
    assert body["fitted"]["Ks"] == pytest_rel(TRUE["Ks"], 1e-9)


# --------------------------------------------------------------------------- #
# 入箱校验：400，迭代之前
# --------------------------------------------------------------------------- #

def test_calibrate_too_few_points_400(client):
    r = client.post("/calibrate", json={
        "observations": [{"t": 0.0, "F": 0.0}, {"t": 1.0, "F": 1.0}]})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "too_few_observations"


def test_calibrate_times_not_increasing_400(client):
    r = client.post("/calibrate", json={"observations": [
        {"t": 1.0, "F": 1.0}, {"t": 1.0, "F": 2.0}, {"t": 2.0, "F": 3.0}]})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "times_not_increasing"


def test_calibrate_negative_F_400(client):
    r = client.post("/calibrate", json={"observations": [
        {"t": 0.0, "F": 0.0}, {"t": 1.0, "F": -0.2}, {"t": 2.0, "F": 3.0}]})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "negative_infiltration"


def test_calibrate_bad_body_4xx(client):
    assert client.post("/calibrate", data="not json").status_code == 415
    r = client.post("/calibrate", json={"observations": "nope"})
    assert r.status_code == 400
    r = client.post("/calibrate", json={"observations": [
        {"t": 0, "F": 0}, {"t": 1, "F": 1}, {"t": 2, "F": 1}], "i": -1})
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# 不收敛/无约束：422，绝不吐半成品参数
# --------------------------------------------------------------------------- #

def test_calibrate_non_convergence_422_withholds_parameters(client):
    r = client.post("/calibrate", json={
        "observations": _rain_obs(), "i": I, "max_iter": 1})
    assert r.status_code == 422
    body = r.get_json()
    assert body["error"]["code"] == "not_converged"
    assert "fitted" not in body
    assert body["error"]["details"]["best_sse_reached"] > 0


def test_calibrate_free_only_data_422(client):
    obs = [{"t": 0.001 * k, "F": I * 0.001 * k} for k in range(1, 8)]
    r = client.post("/calibrate", json={"observations": obs, "i": I})
    assert r.status_code == 422
    assert r.get_json()["error"]["code"] == "uninformative_observations"


# --------------------------------------------------------------------------- #
# 结果落工况档案，与手工工况同一套存取
# --------------------------------------------------------------------------- #

def test_calibration_result_saved_as_named_profile(client):
    r = client.post("/calibrate", json={
        "observations": _rain_obs(), "i": I,
        "save_profile": {"name": "field_A3", "delta_theta": TRUE["delta_theta"],
                         "description": "田间双环反演"},
    })
    assert r.status_code == 200
    saved = r.get_json()["saved_profile"]
    assert saved["name"] == "field_A3"
    assert abs(saved["Ks"] - TRUE["Ks"]) < 1e-7
    # psi 由 A/delta_theta 还原
    assert abs(saved["psi"] - TRUE["psi"]) < 1e-6
    # 走同一套工况存取
    got = client.get("/profiles/field_A3").get_json()
    assert abs(got["Ks"] - TRUE["Ks"]) < 1e-7
    # 该工况还能进正问题
    fwd = client.post("/infiltrate", json={"profile": "field_A3", "t": 1.0})
    assert fwd.status_code == 200


def test_combination_only_cannot_be_saved_as_profile(client):
    r = client.post("/calibrate", json={
        "observations": _ponded_obs(),
        "save_profile": {"name": "nogo", "delta_theta": 0.4},
    })
    assert r.status_code == 409
    assert r.get_json()["error"]["code"] == "unidentifiable_for_profile"
    assert client.get("/profiles/nogo").status_code == 404


def test_save_profile_requires_split_info(client):
    r = client.post("/calibrate", json={
        "observations": _rain_obs(), "i": I,
        "save_profile": {"name": "x"},
    })
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "missing_parameter"


def test_existing_profile_used_as_initial(client):
    # loam 与真值差别不小，作为初值仍应收敛到观测对应的真值
    r = client.post("/calibrate", json={
        "observations": _rain_obs(), "i": I,
        "initial_profile": "loam",
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["fitted"]["Ks"] == pytest_rel(TRUE["Ks"], 1e-7)
    assert body["fitted"]["A"] == pytest_rel(A_TRUE, 1e-7)


# --------------------------------------------------------------------------- #
# 后台作业
# --------------------------------------------------------------------------- #

def test_calibration_job_lifecycle_over_http(client):
    r = client.post("/calibrations", json={"observations": _rain_obs(), "i": I})
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    body = _poll(client, job_id, {"completed"})
    assert body["complete"] is True
    assert body["result"]["identifiability"]["status"] == "individual"
    assert body["progress"]["iterations"] >= 1


def test_calibration_job_failure_over_http(client):
    r = client.post("/calibrations", json={
        "observations": _rain_obs(), "i": I, "max_iter": 1})
    job_id = r.get_json()["job_id"]
    body = _poll(client, job_id, {"failed"})
    assert body["complete"] is False
    assert body["error"]["code"] == "not_converged"
    assert "result" not in body


def test_calibration_job_cancel_never_partial(client, app):
    def fake_runner(obs, **kwargs):
        raise CalibrationCancelled(9, [])

    app.extensions["ga_calibration_jobs"].set_runner(fake_runner)
    r = client.post("/calibrations", json={"observations": _ponded_obs()})
    job_id = r.get_json()["job_id"]
    body = _poll(client, job_id, {"cancelled"})
    assert body["complete"] is False
    assert "result" not in body
    assert body["iterations_before_cancel"] == 9


def test_calibration_job_unknown_404(client):
    assert client.get("/calibrations/nope").status_code == 404
    assert client.post("/calibrations/nope/cancel").status_code == 404


def test_two_concurrent_calibrations_do_not_mix(client):
    r1 = client.post("/calibrations", json={"observations": _ponded_obs()})
    r2 = client.post("/calibrations", json={
        "observations": _rain_obs(), "i": I})
    id1, id2 = r1.get_json()["job_id"], r2.get_json()["job_id"]
    b1 = _poll(client, id1, {"completed"})
    b2 = _poll(client, id2, {"completed"})
    assert b1["result"]["identifiability"]["status"] == "combination_only"
    assert b2["result"]["identifiability"]["status"] == "individual"
    assert len(b1["result"]["residuals"]) == 15
    assert len(b2["result"]["residuals"]) == 40


def test_background_job_saves_profile_on_completion(client):
    r = client.post("/calibrations", json={
        "observations": _rain_obs(), "i": I,
        "save_profile": {"name": "field_bg", "delta_theta": TRUE["delta_theta"]},
    })
    job_id = r.get_json()["job_id"]
    body = _poll(client, job_id, {"completed"})
    saved = body["result"]["saved_profile"]
    assert saved is not None and saved["name"] == "field_bg"
    # 无需再触发 GET 惰性建档：完成时已落盘
    assert client.get("/profiles/field_bg").status_code == 200


def test_background_job_unidentifiable_carries_save_error_not_raise(client):
    r = client.post("/calibrations", json={
        "observations": _ponded_obs(),
        "save_profile": {"name": "field_bg_bad", "delta_theta": 0.4},
    })
    job_id = r.get_json()["job_id"]
    body = _poll(client, job_id, {"completed"})
    # 标定本身完成（组合可辨），但建档被拒，错误挂在 save_error
    assert body["result"]["identifiability"]["status"] == "combination_only"
    assert body["save_error"]["code"] == "unidentifiable_for_profile"
    assert client.get("/profiles/field_bg_bad").status_code == 404
