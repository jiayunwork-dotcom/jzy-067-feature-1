"""历时点列分段推进与可取消语义。"""

from __future__ import annotations

import threading

import pytest

from model.hydrograph import HydrographCancelled, run_hydrograph

LOAM = dict(Ks=1.09, psi=11.01, delta_theta=0.434)


def _Fs(points):
    return [p["F"] for p in points]


def _rates(points):
    return [p["f"] for p in points]


def test_ponded_from_zero_series_monotonic():
    hg = run_hydrograph(**LOAM, duration=10.0, n_points=51, already_ponded=True)
    assert hg.complete is True
    assert len(hg.points) == 51
    Fs = _Fs(hg.points)
    assert Fs[0] == 0.0
    for a, b in zip(Fs, Fs[1:]):
        assert b >= a
    rates = _rates(hg.points)
    for a, b in zip(rates, rates[1:]):
        assert b <= a
    # 每个积水点残差都达标（t=0 残差为 0）
    for p in hg.points:
        if p["t"] > 0.0:
            assert abs(p["residual"]) <= p["tolerance"]
        assert p["phase"] == "ponded"


def test_rainfall_series_free_then_ponded_switch():
    i = 5.0
    hg = run_hydrograph(**LOAM, duration=3.0, n_points=301, i=i)
    assert hg.ponding["will_pond"] is True
    tp = hg.ponding["tp"]

    phases = {p["phase"] for p in hg.points}
    assert "free" in phases and "ponded" in phases

    # F 跨过积水时刻连续：找开关前后的两点
    switch = hg.phase_switches[0]
    assert switch["from"] == "free" and switch["to"] == "ponded"
    assert switch["t"] >= tp

    pre = [p for p in hg.points if p["t"] < tp]
    post = [p for p in hg.points if p["t"] > tp]
    assert pre and post
    # 自由段严格 F=i t
    for p in pre:
        assert p["F"] == pytest.approx(i * p["t"])
        assert p["residual"] is None
    # 积水段残差达标，且入渗率 <= i
    for p in post:
        assert abs(p["residual"]) <= p["tolerance"]
        assert p["f"] <= i * (1 + 1e-9)

    # 全局 F 单调不减（自由段到积水段也不许跳）
    Fs = _Fs(hg.points)
    for a, b in zip(Fs, Fs[1:]):
        assert b >= a - 1e-12


def test_rainfall_below_Ks_series_all_free():
    hg = run_hydrograph(**LOAM, duration=5.0, n_points=50, i=0.5)
    assert hg.ponding["will_pond"] is False
    assert all(p["phase"] == "free" for p in hg.points)
    for p in hg.points:
        assert p["residual"] is None
        assert p["f"] == 0.5


# --------------------------------------------------------------------------- #
# 取消语义
# --------------------------------------------------------------------------- #

def test_cancel_before_start_raises_with_incomplete_marker():
    # 取消信号一开始就置位：第一个点之前必须抛出，不带完整点列
    with pytest.raises(HydrographCancelled) as exc:
        run_hydrograph(**LOAM, duration=10.0, n_points=1001,
                       already_ponded=True,
                       should_cancel=lambda: True)
    assert exc.value.partial_points == []
    assert exc.value.last_index == -1


def test_cancel_midway_via_event():
    event = threading.Event()
    calls = {"n": 0}

    def cancel_after_few():
        calls["n"] += 1
        return calls["n"] > 10  # 前 10 个点之后取消

    with pytest.raises(HydrographCancelled) as exc:
        run_hydrograph(**LOAM, duration=10.0, n_points=500,
                       already_ponded=True,
                       should_cancel=cancel_after_few)
    partial = exc.value.partial_points
    assert 0 < len(partial) < 500
    # 被取消时抛出的结构不被任何正常路径包装成完整结果


# --------------------------------------------------------------------------- #
# 多方案并发隔离：各自临时量互不串账
# --------------------------------------------------------------------------- #

def test_concurrent_hydrographs_do_not_cross_accounts():
    from model.infiltration import state_at_time

    # 前两份自始积水，后两份给定降雨（一份积水、一份永远自由）
    schemes = [
        dict(Ks=1.09, psi=11.01, delta_theta=0.434, duration=3.0,
             i=None, already_ponded=True),
        dict(Ks=0.3, psi=30.0, delta_theta=0.5, duration=8.0,
             i=None, already_ponded=True),
        dict(Ks=2.5, psi=5.0, delta_theta=0.2, duration=1.0,
             i=10.0, already_ponded=False),
        dict(Ks=0.8, psi=20.0, delta_theta=0.45, duration=6.0,
             i=0.4, already_ponded=False),  # i <= Ks，不积水
    ]
    results: dict[int, object] = {}
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(schemes))

    def worker(idx, sc):
        try:
            barrier.wait()
            results[idx] = run_hydrograph(
                sc["Ks"], sc["psi"], sc["delta_theta"], sc["duration"],
                n_points=200, i=sc["i"], already_ponded=sc["already_ponded"],
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k, sc))
               for k, sc in enumerate(schemes)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors

    for idx, sc in enumerate(schemes):
        hg = results[idx]
        assert hg.n_points == 200
        # 用独立的逐点核算对账，确保这份账没被别份方案的临时量污染
        for p in hg.points[::13]:  # 抽样核对即可
            ref = state_at_time(
                sc["Ks"], sc["psi"], sc["delta_theta"], p["t"],
                i=sc["i"], already_ponded=sc["already_ponded"],
            )
            assert p["F"] == pytest.approx(ref.F, rel=1e-9)
            assert p["phase"] == ref.phase

    # 第四份 i <= Ks，全程 free 且 F=i t
    for p in results[3].points:
        assert p["phase"] == "free"
        assert p["F"] == pytest.approx(0.4 * p["t"])
    # 前两份自始积水
    assert all(p["phase"] == "ponded" for p in results[0].points)
    assert all(p["phase"] == "ponded" for p in results[1].points)
    # 第三份存在 free->ponded 切换
    assert results[2].phase_switches
