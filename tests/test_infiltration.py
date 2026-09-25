"""入渗率、积水时刻与参数趋势的因果测试。"""

from __future__ import annotations

import math

import pytest

from model.infiltration import (
    F_FLOOR,
    INITIAL_RATE_CAP,
    analyze_ponding,
    infiltration_rate,
    state_at_time,
)
from model.solver import solve_cumulative, suction_head

LOAM = dict(Ks=1.09, psi=11.01, delta_theta=0.434)


# --------------------------------------------------------------------------- #
# t=0 初值与入渗率
# --------------------------------------------------------------------------- #

def test_rate_infinite_capped_to_convention():
    A = suction_head(LOAM["psi"], LOAM["delta_theta"])
    assert infiltration_rate(0.0, LOAM["Ks"], A) == INITIAL_RATE_CAP
    assert infiltration_rate(-1e9, LOAM["Ks"], A) == INITIAL_RATE_CAP
    # 略高于下限仍可能被封顶，但必须是有限数
    assert math.isfinite(infiltration_rate(F_FLOOR / 2, LOAM["Ks"], A))


def test_state_zero_ponded_from_zero():
    st = state_at_time(**LOAM, t=0.0, already_ponded=True)
    assert st.F == 0.0
    assert st.f == INITIAL_RATE_CAP
    assert st.phase == "ponded"
    assert st.residual == 0.0


# --------------------------------------------------------------------------- #
# 入渗率一路走低并趋近 Ks
# --------------------------------------------------------------------------- #

def test_rate_decreases_and_approaches_Ks():
    ts = [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 5000.0, 1e5]
    A = suction_head(LOAM["psi"], LOAM["delta_theta"])
    rates = []
    Fs = []
    for t in ts:
        sol = solve_cumulative(**LOAM, t=t)
        Fs.append(sol.F)
        rates.append(infiltration_rate(sol.F, LOAM["Ks"], A))

    # 一路非增
    for r0, r1 in zip(rates, rates[1:]):
        assert r1 <= r0 + 1e-12
    # 长历时从上方趋近 Ks
    assert rates[-1] == pytest.approx(LOAM["Ks"], rel=1e-4)
    assert all(r >= LOAM["Ks"] for r in rates)


# --------------------------------------------------------------------------- #
# 积水判定
# --------------------------------------------------------------------------- #

def test_i_below_Ks_never_ponds():
    for i in (1.0, 1.09, 0.001):
        info = analyze_ponding(LOAM["Ks"], LOAM["psi"], LOAM["delta_theta"], i)
        assert info.will_pond is False
        assert info.tp is None and info.Fp is None


def test_i_equals_Ks_never_ponds():
    info = analyze_ponding(LOAM["Ks"], LOAM["psi"], LOAM["delta_theta"], LOAM["Ks"])
    assert info.will_pond is False


def test_i_above_Ks_ponds_at_expected_time():
    i = 5.0
    info = analyze_ponding(LOAM["Ks"], LOAM["psi"], LOAM["delta_theta"], i)
    A = suction_head(LOAM["psi"], LOAM["delta_theta"])
    assert info.will_pond is True
    assert info.Fp == pytest.approx(LOAM["Ks"] * A / (i - LOAM["Ks"]))
    assert info.tp == pytest.approx(info.Fp / i)
    # 积水时刻能力恰等于供水强度
    assert LOAM["Ks"] * (1.0 + A / info.Fp) == pytest.approx(i)
    # i 越接近 Ks，tp 越大
    info2 = analyze_ponding(LOAM["Ks"], LOAM["psi"], LOAM["delta_theta"], 2.0)
    assert info2.tp > info.tp


def test_state_free_then_ponded_phases():
    i = 5.0
    info = analyze_ponding(LOAM["Ks"], LOAM["psi"], LOAM["delta_theta"], i)
    tp = info.tp

    pre = state_at_time(**LOAM, t=tp * 0.5, i=i)
    assert pre.phase == "free"
    assert pre.F == pytest.approx(i * tp * 0.5)
    assert pre.f == i  # 供多少渗多少
    assert pre.f_capacity >= i  # 能力仍高于供水
    assert pre.residual is None  # 自由段不走隐式

    at = state_at_time(**LOAM, t=tp, i=i)
    assert at.phase == "ponding"
    assert at.F == pytest.approx(info.Fp)

    post = state_at_time(**LOAM, t=tp * 3.0, i=i)
    assert post.phase == "ponded"
    # 积水后入渗量比自由段外推 F=i t 小（能力限制）
    assert post.F < i * tp * 3.0
    # 积水后残差必须达标
    assert abs(post.residual) <= post.tolerance
    # F 在积水时刻连续
    bridge = state_at_time(**LOAM, t=tp * (1.0 + 1e-13), i=i)
    assert bridge.F == pytest.approx(info.Fp, rel=1e-9)


# --------------------------------------------------------------------------- #
# 参数趋势（把因果关系钉牢）
# --------------------------------------------------------------------------- #

def test_higher_psi_more_infiltration_early():
    # 单独抬高吸力 psi，早期渗进去的更多
    t = 0.2
    low = solve_cumulative(LOAM["Ks"], psi=5.0, delta_theta=LOAM["delta_theta"], t=t)
    high = solve_cumulative(LOAM["Ks"], psi=30.0, delta_theta=LOAM["delta_theta"], t=t)
    assert high.F > low.F


def test_higher_delta_theta_increases_infiltration():
    # 标准 Green-Ampt：A=psi*delta_theta，f=Ks(1+A/F)。
    # 固定时刻 t，由隐式式可得 F 随 A 单调增（早期 F~sqrt(2KsAt)）。
    # 故单独加大 delta_theta，同时刻渗进去的水更多。
    small = solve_cumulative(LOAM["Ks"], 11.01, 0.2, t=1.0)
    large = solve_cumulative(LOAM["Ks"], 11.01, 0.6, t=1.0)
    assert large.F > small.F


def test_higher_delta_theta_reaches_same_F_sooner():
    # 显函数 t(F)=[F-A ln(1+F/A)]/Ks 对 A 的导数恒负：
    # dt/dA = -(ln(1+F/A) - F/(A+F))/Ks < 0
    # （因 x-ln(1+x) > 0 for x>0）。所以渗到同样的 F，delta_theta 越大
    # 花的时间越短——这与"更大的含水量差让入渗能力更高"一致。
    F_target = 2.0
    t_small = ponded_time_for(F_target, LOAM["Ks"], 11.01, 0.2)
    t_large = ponded_time_for(F_target, LOAM["Ks"], 11.01, 0.6)
    assert t_large < t_small
    # 解析导数直接核对符号
    A = 11.01 * 0.4
    dtdA = -(math.log1p(F_target / A) - F_target / (A + F_target)) / LOAM["Ks"]
    assert dtdA < 0.0


def ponded_time_for(F, Ks, psi, delta_theta):
    A = suction_head(psi, delta_theta)
    return (F - A * math.log1p(F / A)) / Ks


def test_higher_Ks_more_infiltration_same_time():
    t = 1.0
    low = solve_cumulative(0.5, LOAM["psi"], LOAM["delta_theta"], t=t)
    high = solve_cumulative(3.0, LOAM["psi"], LOAM["delta_theta"], t=t)
    assert high.F > low.F
