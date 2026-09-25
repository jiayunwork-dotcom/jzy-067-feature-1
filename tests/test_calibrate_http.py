"""标定 HTTP 路由与后台作业测试。

钉牢：提交/查询/取消语义、失败不交参数、建档与取回、初值复用、
并发隔离、观测校验 400、病态输入落 failed 而非假成功。
"""

from __future__ import annotations

import math
import time

import pytest

from model.calibration import CalibrationCancelled
from model.infiltration import analyze_ponding, state_at_time
from model.solver import solve_cumulative

LOAM = {"Ks": 1.09, "psi": 11.01, "delta_theta": 0.434}
A_LOAM = LOAM["psi"] * LOAM["delta_theta"]


def lin(a, b, n):
    return [a + k * (b - a) / (n - 1) for k in range(n)]


def rain_obs(Ks, A, i, ts):
    return [{"t": t, "F": state_at_time(Ks, A, 1.0, t, i=i).F} for t in ts]


def ponded_obs(Ks, A, ts):
    return [{"t": t, "F": solve_cumulative(Ks, A, 1.0, t).F} for t in ts]


def loam_rain_payload(**extra):
    i = 3.0 * LOAM["Ks"]
    tp = analyze_ponding(LOAM["Ks"], A_LOAM, 1.0, i).tp
    ts = lin(0.02, 10.0 * tp, 25)
    payload = {"observations": rain_obs(LOAM["Ks"], A_LOAM, i, ts), "i": i}
    payload.update(extra)
    return payload


def wait_job(client, job_id, states=("completed", "failed", "cancelled"),
             timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/calibrate/{job_id}")
        body = r.get_json()
        if body["state"] in states:
            return body
        time.sleep(0.01)
    raise AssertionError(f"标定作业 {job_id} 未在 {timeout}s 内到达 {states}")


# --------------------------------------------------------------------------- #
# 端到端：可辨识数据 → 高精度还原
# --------------------------------------------------------------------------- #

def test_calibrate_end_to_end_rainfall_roundtrip(client):
    r = client.post("/calibrate", json=loam_rain_payload())
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    body = wait_job(client, job_id)
    assert body["state"] == "completed"
    assert body["complete"] is True

    result = body["result"]
    assert result["converged"] is True
    assert result["model_mode"] == "rainfall"
    assert result["identifiability"]["status"] == "both_identifiable"
    assert abs(result["parameters"]["Ks"]["value"] / LOAM["Ks"] - 1.0) < 1e-8
    assert abs(result["parameters"]["A"]["value"] / A_LOAM - 1.0) < 1e-8
    assert result["sum_squared_residuals"] < 1e-18
    assert result["iterations"] >= 1
    assert len(result["residuals"]) == 25
    # 逐点残差平方和与回报的目标函数一致
    ssr = sum(p["residual"] ** 2 for p in result["residuals"])
    assert abs(ssr - result["sum_squared_residuals"]) < 1e-24


def test_calibrate_ponded_degenerate_flags_combination_only(client):
    Ks_t, A_t = 0.2, 31.63 * 0.476
    ts = lin(0.05, 0.5, 20)
    r = client.post("/calibrate", json={"observations": ponded_obs(Ks_t, A_t, ts)})
    assert r.status_code == 202
    body = wait_job(client, r.get_json()["job_id"])
    assert body["state"] == "completed"
    result = body["result"]
    idf = result["identifiability"]
    assert idf["status"] == "combination_only"
    assert result["parameters"]["Ks"]["identifiable"] is False
    assert result["parameters"]["A"]["identifiable"] is False
    combo = idf["identifiable_combination"]
    w = combo["weights"]
    true_combo = math.exp(w["ln_Ks"] * math.log(Ks_t) + w["ln_A"] * math.log(A_t))
    assert abs(combo["value"] / true_combo - 1.0) < 1e-8
    assert idf["resolution_hint"]


# --------------------------------------------------------------------------- #
# 建档与初值复用
# --------------------------------------------------------------------------- #

def test_save_as_profile_and_fetch_back(client):
    payload = loam_rain_payload(save_as="field-plot-3", delta_theta=LOAM["delta_theta"])
    r = client.post("/calibrate", json=payload)
    assert r.status_code == 202
    body = wait_job(client, r.get_json()["job_id"])
    assert body["state"] == "completed"
    save = body["result"]["profile_save"]
    assert save["requested"] is True
    assert save["saved"] is True

    # 与手工建档走同一套存取：GET /profiles/<name>
    r = client.get("/profiles/field-plot-3")
    assert r.status_code == 200
    p = r.get_json()
    assert abs(p["Ks"] / LOAM["Ks"] - 1.0) < 1e-8
    assert abs(p["delta_theta"] - LOAM["delta_theta"]) < 1e-12
    assert abs(p["psi"] / LOAM["psi"] - 1.0) < 1e-8

    # 建档工况可进正向核算
    r = client.post("/infiltrate", json={"profile": "field-plot-3", "t": 1.0})
    assert r.status_code == 200


def test_combination_only_result_is_not_saved(client):
    Ks_t, A_t = 0.2, 31.63 * 0.476
    ts = lin(0.05, 0.5, 20)
    r = client.post("/calibrate", json={
        "observations": ponded_obs(Ks_t, A_t, ts),
        "save_as": "should-not-stick",
        "delta_theta": 0.476,
    })
    assert r.status_code == 202
    body = wait_job(client, r.get_json()["job_id"])
    assert body["state"] == "completed"
    assert body["result"]["identifiability"]["status"] == "combination_only"
    save = body["result"]["profile_save"]
    assert save["requested"] is True
    assert save["saved"] is False  # 组合可辨的结果不许固化成工况
    r = client.get("/profiles/should-not-stick")
    assert r.status_code == 404


def test_save_as_requires_split_info(client):
    payload = loam_rain_payload(save_as="no-split")
    r = client.post("/calibrate", json=payload)
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "missing_parameter"


def test_initial_profile_used_as_starting_point(client):
    # 先把真参数建一份档，再故意从别的初值起标——用档案当初值
    client.put("/profiles/near-loam", json={
        "Ks": 2.0, "psi": 20.0, "delta_theta": 0.5, "description": "离真值不远",
    })
    payload = loam_rain_payload(initial_profile="near-loam")
    r = client.post("/calibrate", json=payload)
    assert r.status_code == 202
    body = wait_job(client, r.get_json()["job_id"])
    assert body["state"] == "completed"
    assert abs(body["result"]["parameters"]["Ks"]["value"] / LOAM["Ks"] - 1.0) < 1e-8
    # 首个起点必须来自档案（Ks=2.0, A=20*0.5=10）
    first_attempt = body["result"]["attempts"][0]
    assert first_attempt["start"]["Ks"] == pytest.approx(2.0)
    assert first_attempt["start"]["A"] == pytest.approx(10.0)


def test_initial_and_initial_profile_conflict(client):
    payload = loam_rain_payload(initial={"Ks": 1.0}, initial_profile="loam")
    r = client.post("/calibrate", json=payload)
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# 失败语义：病态输入落 failed，绝不交参数
# --------------------------------------------------------------------------- #

def test_pathological_max_iter_one_fails_honestly(client):
    payload = loam_rain_payload(max_iter=1)
    r = client.post("/calibrate", json=payload)
    assert r.status_code == 202
    body = wait_job(client, r.get_json()["job_id"])
    assert body["state"] == "failed"
    assert body["complete"] is False
    assert body["result"] is None  # 半成品参数绝不外吐
    assert body["error"]["code"] == "calibration_not_converged"
    assert body["error"]["details"]["max_iter"] == 1


def test_garbage_observations_fail_or_honest(client):
    ts = lin(0.05, 6.0, 20)
    garbage = [3.0 + 2.0 * ((-1) ** k) + 0.1 * k for k in range(20)]
    obs = [{"t": t, "F": f} for t, f in zip(ts, garbage)]
    r = client.post("/calibrate", json={"observations": obs})
    assert r.status_code == 202
    body = wait_job(client, r.get_json()["job_id"])
    if body["state"] == "failed":
        assert body["result"] is None
        assert body["error"]["code"] in (
            "calibration_not_converged", "degenerate_solution")
    else:
        # 允许收敛，但残差必须诚实地大
        assert body["state"] == "completed"
        ssr = body["result"]["sum_squared_residuals"]
        assert ssr > 1.0


# --------------------------------------------------------------------------- #
# 取消语义
# --------------------------------------------------------------------------- #

def test_cancelled_calibration_never_returns_params(client, app):
    # 确定性替身：模拟迭代到一半被取消
    def fake_runner(*args, **kwargs):
        raise CalibrationCancelled(7)

    mgr = app.extensions["ga_caljobs"]
    payload = loam_rain_payload()
    job_id = mgr.submit(
        {
            "observations": [],
            "i": None,
            "fixed": {},
            "initial": None,
            "split": {},
            "save_as": None,
            "options": {},
            "request_echo": {},
        },
        runner=fake_runner,
    )
    deadline = time.time() + 5.0
    while time.time() < deadline:
        snap = mgr.status(job_id)
        if snap["state"] == "cancelled":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("作业未进入 cancelled")
    assert snap["complete"] is False
    assert snap["result"] is None
    assert snap["iterations_before_cancel"] == 7


def test_cancel_real_job_midway(client):
    # 大量观测 + 多起点：提交后立刻取消，终态要么 cancelled 要么 completed
    Ks_t, A_t = 1.09, A_LOAM
    ts = lin(0.02, 8.0, 400)
    obs = ponded_obs(Ks_t, A_t, ts)
    r = client.post("/calibrate", json={"observations": obs})
    assert r.status_code == 202
    job_id = r.get_json()["job_id"]
    r = client.post(f"/calibrate/{job_id}/cancel")
    assert r.status_code == 202
    body = wait_job(client, job_id, states=("cancelled", "completed"))
    if body["state"] == "cancelled":
        assert body["complete"] is False
        assert body["result"] is None
    else:
        assert body["complete"] is True
        assert body["result"]["converged"] is True


def test_cancel_unknown_job_404(client):
    r = client.post("/calibrate/nope/cancel")
    assert r.status_code == 404
    r = client.get("/calibrate/nope")
    assert r.status_code == 404


# --------------------------------------------------------------------------- #
# 并发隔离：多份标定各自的观测与结果互不串账
# --------------------------------------------------------------------------- #

def test_concurrent_calibrations_keep_separate_accounts(client):
    cases = [
        (1.09, A_LOAM, 3.0 * 1.09),
        (0.5, 8.0, 3.0 * 0.5),
        (2.5, 3.0, 3.0 * 2.5),
    ]
    job_ids = []
    for Ks, A, i in cases:
        tp = analyze_ponding(Ks, A, 1.0, i).tp
        ts = lin(0.02, 10.0 * tp, 25)
        r = client.post("/calibrate", json={
            "observations": rain_obs(Ks, A, i, ts), "i": i,
        })
        assert r.status_code == 202
        job_ids.append(r.get_json()["job_id"])

    bodies = [wait_job(client, jid) for jid in job_ids]
    for body, (Ks, A, i) in zip(bodies, cases):
        assert body["state"] == "completed"
        result = body["result"]
        assert abs(result["parameters"]["Ks"]["value"] / Ks - 1.0) < 1e-6
        assert abs(result["parameters"]["A"]["value"] / A - 1.0) < 1e-6
        assert result["rainfall_i"] == i
        # 观测回显必须是本作业自己的数据
        assert len(body["request"]["observations"]) == 25


# --------------------------------------------------------------------------- #
# 观测与参数校验：400 挡在迭代之前
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "obs,frag",
    [
        ([{"t": 0.1, "F": 0.2}], "太少"),
        ([{"t": 0.0, "F": 0.0}, {"t": 0.2, "F": 0.1}, {"t": 0.2, "F": 0.3}], "递增"),
        ([{"t": 0.0, "F": 0.0}, {"t": 0.2, "F": -0.1}, {"t": 0.3, "F": 0.2}], "负"),
        ([{"t": 0.0, "F": 0.0}, {"t": 0.2, "F": 0.0}, {"t": 0.3, "F": 0.0}], "信号"),
    ],
)
def test_bad_observations_400(client, obs, frag):
    r = client.post("/calibrate", json={"observations": obs})
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_observation"


def test_parallel_array_observations_accepted(client):
    Ks_t, A_t = 8.25, 4.95 * 0.417
    ts = lin(0.05, 6.0, 10)
    Fs = [solve_cumulative(Ks_t, A_t, 1.0, t).F for t in ts]
    r = client.post("/calibrate", json={"observations": {"t": ts, "F": Fs}})
    assert r.status_code == 202
    body = wait_job(client, r.get_json()["job_id"])
    assert body["state"] == "completed"
    assert abs(body["result"]["parameters"]["Ks"]["value"] / Ks_t - 1.0) < 1e-7


def test_missing_observations_400(client):
    r = client.post("/calibrate", json={"i": 3.0})
    assert r.status_code == 400


def test_conflicting_mode_flags_400(client):
    payload = loam_rain_payload(already_ponded=True)
    r = client.post("/calibrate", json=payload)
    assert r.status_code == 400


def test_initial_Ks_in_dead_zone_400(client):
    payload = loam_rain_payload(initial={"Ks": 100.0, "A": 5.0})
    r = client.post("/calibrate", json=payload)
    assert r.status_code == 400


def test_fix_allows_identifiable_single_param(client):
    Ks_t, A_t = 0.2, 31.63 * 0.476
    ts = lin(0.05, 0.5, 20)
    r = client.post("/calibrate", json={
        "observations": ponded_obs(Ks_t, A_t, ts),
        "fix": {"Ks": Ks_t},
    })
    assert r.status_code == 202
    body = wait_job(client, r.get_json()["job_id"])
    assert body["state"] == "completed"
    result = body["result"]
    assert result["identifiability"]["status"] == "identifiable_via_constraint"
    assert abs(result["parameters"]["A"]["value"] / A_t - 1.0) < 1e-9
    assert result["parameters"]["Ks"]["role"] == "fixed"
