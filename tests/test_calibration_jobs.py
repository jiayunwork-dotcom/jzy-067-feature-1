"""标定后台作业生命周期：完成、失败、取消语义、并发隔离。"""

from __future__ import annotations

import time

import pytest

from model.calibration import CalibrationCancelled
from model.calibration_jobs import CalibrationJobManager
from model.errors import ServiceError
from model.infiltration import state_at_time

TRUE = dict(Ks=1.09, psi=11.01, delta_theta=0.434)
A_TRUE = TRUE["psi"] * TRUE["delta_theta"]
I = 5.0


def _rain_spec(ts=None, i=I):
    ts = ts or [0.02 * k for k in range(1, 41)]
    obs = [{"t": t, "F": state_at_time(TRUE["Ks"], TRUE["psi"], TRUE["delta_theta"],
                                       t, i=i).F} for t in ts]
    return {"observations": obs, "i": i, "max_iter": 200}


def _ponded_spec(ts=None):
    ts = ts or [0.02 * k for k in range(1, 16)]
    obs = [{"t": t, "F": state_at_time(TRUE["Ks"], TRUE["psi"], TRUE["delta_theta"],
                                       t, already_ponded=True).F} for t in ts]
    return {"observations": obs, "i": None, "max_iter": 200}


@pytest.fixture()
def mgr():
    m = CalibrationJobManager(workers=4)
    yield m
    m.shutdown()


def _wait(mgr, jid, states, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = mgr.status(jid)["state"]
        if st in states:
            return mgr.status(jid)
        time.sleep(0.005)
    raise AssertionError(f"作业未在 {timeout}s 内到达 {states}")


def test_calibration_job_completes_with_auditable_result(mgr):
    jid = mgr.submit(_rain_spec())
    snap = _wait(mgr, jid, {"completed"})
    assert snap["complete"] is True
    res = snap["result"]
    assert res["identifiability"]["status"] == "individual"
    assert res["fitted"]["Ks"] == pytest.approx(TRUE["Ks"], rel=1e-8)
    assert res["fitted"]["A"] == pytest.approx(A_TRUE, rel=1e-8)
    assert res["iterations"] >= 1
    assert res["sse"] >= 0.0
    assert len(res["residuals"]) == 40


def test_calibration_job_failure_state_carries_error_not_parameters(mgr):
    spec = _rain_spec()
    spec["max_iter"] = 1
    jid = mgr.submit(spec)
    snap = _wait(mgr, jid, {"failed"})
    assert snap["complete"] is False
    assert snap["error"]["code"] == "not_converged"
    assert "result" not in snap  # 失败绝不吐半成品参数


def test_uninformative_data_jobs_fails_cleanly(mgr):
    spec = {
        "observations": [{"t": 0.001 * k, "F": I * 0.001 * k} for k in range(1, 8)],
        "i": I,
    }
    jid = mgr.submit(spec)
    snap = _wait(mgr, jid, {"failed"})
    assert snap["error"]["code"] == "uninformative_observations"
    assert "result" not in snap


def test_cancelled_job_never_returns_half_parameters(mgr):
    def fake_runner(obs, **kwargs):
        raise CalibrationCancelled(7, [{"iteration": 7, "sse": 0.123}])

    jid = mgr.submit(_ponded_spec(), runner=fake_runner)
    snap = _wait(mgr, jid, {"cancelled"})
    assert snap["state"] == "cancelled"
    assert snap["complete"] is False
    assert "result" not in snap
    assert snap["iterations_before_cancel"] == 7


def test_concurrent_calibrations_keep_isolated_accounts(mgr):
    schemes = [
        _ponded_spec([0.02 * k for k in range(1, 12)]),
        _rain_spec([0.02 * k for k in range(1, 30)]),
        _ponded_spec([0.05 * k for k in range(1, 20)]),
    ]
    ids = [mgr.submit(s) for s in schemes]
    for jid in ids:
        _wait(mgr, jid, {"completed", "failed"})

    snaps = [mgr.status(jid) for jid in ids]
    # 每份作业的观测数、结果互不串账
    assert [s["spec"]["n_observations"] for s in snaps] == [11, 29, 19]
    for s in snaps:
        assert s["state"] == "completed"
        assert len(s["result"]["residuals"]) == s["spec"]["n_observations"]
    # 两个积水作业给出组合不可辨，降雨作业给出 individual
    assert snaps[0]["result"]["identifiability"]["status"] == "combination_only"
    assert snaps[1]["result"]["identifiability"]["status"] == "individual"
    assert snaps[2]["result"]["identifiability"]["status"] == "combination_only"


def test_cancel_unknown_calibration_job_404(mgr):
    with pytest.raises(ServiceError) as exc:
        mgr.cancel("nope")
    assert exc.value.status_code == 404
    with pytest.raises(ServiceError):
        mgr.status("nope")
