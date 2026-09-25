"""后台作业生命周期测试：完成、失败、取消（绝不交半截点列）、并发隔离。"""

from __future__ import annotations

import time

from model.errors import ConvergenceError
from model.hydrograph import HydrographCancelled
from model.jobs import JobManager

LOAM_PARAMS = dict(Ks=1.09, psi=11.01, delta_theta=0.434, i=5.0, max_iter=50)
LOAM_SERIES = dict(duration=3.0, n_points=301, already_ponded=False)


def _wait_state(mgr, job_id, states, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = mgr.status(job_id, include_points=False)["state"]
        if st in states:
            return st
        time.sleep(0.01)
    raise AssertionError(f"作业未在 {timeout}s 内到达 {states}")


def test_job_completes_with_full_points(job_manager):
    jid = job_manager.submit(dict(LOAM_PARAMS), dict(LOAM_SERIES))
    _wait_state(job_manager, jid, {"completed"})
    snap = job_manager.status(jid)
    assert snap["complete"] is True
    assert len(snap["points"]) == 301
    assert snap["ponding"]["will_pond"] is True


def test_job_failure_reports_convergence_error(job_manager):
    bad = dict(LOAM_PARAMS)
    bad["max_iter"] = 1  # 一步不可能收敛
    jid = job_manager.submit(bad, dict(LOAM_SERIES))
    _wait_state(job_manager, jid, {"failed"})
    snap = job_manager.status(jid)
    assert snap["complete"] is False
    assert snap["error"]["code"] == "not_converged"
    assert "points" not in snap  # 失败时不吐任何点列


def test_cancelled_job_never_returns_partial_points(job_manager):
    # 确定性替身：模拟推进到第 50 个点被取消
    def fake_runner(*args, **kwargs):
        partial = [{"t": k, "F": float(k)} for k in range(50)]
        raise HydrographCancelled(partial, 49)

    jid = job_manager.submit(dict(LOAM_PARAMS), dict(LOAM_SERIES),
                             runner=fake_runner)
    _wait_state(job_manager, jid, {"cancelled"})
    snap = job_manager.status(jid)
    assert snap["state"] == "cancelled"
    assert snap["complete"] is False
    assert snap["points"] == []  # 半截点列绝不外吐
    assert snap["points_computed_before_cancel"] == 50  # 只给诊断计数


def test_real_job_cancel_midway_keeps_private_progress(job_manager):
    # 真实长点列 + 运行中取消：要么 running 要么 cancelled，终态不交出未完成点列
    params = dict(LOAM_PARAMS)
    params.pop("i")
    series = dict(duration=500.0, n_points=200_000, already_ponded=True)
    jid = job_manager.submit(params, series)
    job_manager.cancel(jid)
    _wait_state(job_manager, jid, {"cancelled", "completed"}, timeout=15.0)
    snap = job_manager.status(jid)
    if snap["state"] == "cancelled":
        assert snap["complete"] is False
        assert snap["points"] == []
    else:
        assert snap["complete"] is True


def test_concurrent_jobs_keep_separate_accounts(job_manager):
    schemes = [
        (dict(Ks=1.09, psi=11.01, delta_theta=0.434, i=5.0),
         dict(duration=3.0, n_points=200, already_ponded=False)),
        (dict(Ks=0.3, psi=30.0, delta_theta=0.5, i=2.0),
         dict(duration=8.0, n_points=200, already_ponded=False)),
        (dict(Ks=2.5, psi=5.0, delta_theta=0.2, i=10.0),
         dict(duration=1.0, n_points=200, already_ponded=False)),
    ]
    ids = [job_manager.submit(p, s) for p, s in schemes]
    for jid in ids:
        _wait_state(job_manager, jid, {"completed"})

    from model.infiltration import state_at_time

    for jid, (p, s) in zip(ids, schemes):
        snap = job_manager.status(jid)
        assert snap["params"]["Ks"] == p["Ks"]
        assert snap["params"]["i"] == p["i"]
        assert len(snap["points"]) == s["n_points"]
        # 抽点与独立账下核算对比
        for pt in snap["points"][::40]:
            ref = state_at_time(p["Ks"], p["psi"], p["delta_theta"], pt["t"], i=p["i"])
            assert pt["F"] == ref.F or abs(pt["F"] - ref.F) < 1e-9


def test_cancel_unknown_job_404():
    import pytest
    from model.errors import ServiceError

    mgr = JobManager(workers=1)
    try:
        with pytest.raises(ServiceError) as exc:
            mgr.cancel("nope")
        assert exc.value.status_code == 404
    finally:
        mgr.shutdown()
