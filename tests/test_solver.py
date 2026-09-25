"""隐式求解器的正确性测试。

重点钉死两件事：
1. **对数项符号**：真解必须 F > Ks*t（t>0）。若把
   F - A ln(1+F/A) 的减号写成加号，解会掉到 Ks*t 以下；
2. **残差达标**：任何返回的 F 都必须满足 |g(F)| <= tolerance。
另外覆盖 t=0 边界、长/短历时、牛顿不收敛必须报错而不是吐初值。
"""

from __future__ import annotations

import math

import pytest

from model.errors import ConvergenceError
from model.solver import (
    DEFAULT_ABS_TOL,
    ponded_time,
    residual_at,
    solve_cumulative,
    suction_head,
)

LOAM = dict(Ks=1.09, psi=11.01, delta_theta=0.434)


# --------------------------------------------------------------------------- #
# t=0 边界
# --------------------------------------------------------------------------- #

def test_zero_time_gives_zero_F_and_finite_cap():
    sol = solve_cumulative(**LOAM, t=0.0)
    assert sol.F == 0.0
    assert sol.residual == 0.0
    assert sol.converged is True
    assert sol.iterations == 0


def test_F_floor_and_rate_cap_values():
    # 入渗率 f=Ks(1+A/F) 在 F->0 发散，服务侧封顶（在 infiltration 测试里验）
    from model.infiltration import F_FLOOR, INITIAL_RATE_CAP

    assert F_FLOOR > 0.0
    assert math.isfinite(INITIAL_RATE_CAP)
    assert INITIAL_RATE_CAP > 0.0


# --------------------------------------------------------------------------- #
# 符号陷阱：F 必须严格大于 Ks*t
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("t", [1e-6, 1e-3, 0.1, 1.0, 10.0, 100.0, 1e4])
def test_sign_trap_F_greater_than_Ks_t(t):
    sol = solve_cumulative(**LOAM, t=t)
    rhs = LOAM["Ks"] * t
    assert sol.F > rhs, (
        f"t={t}: F={sol.F} 没有大于 Ks*t={rhs}——对数项符号是否写反？"
    )
    # 吸力项的贡献差值恒正
    A = suction_head(LOAM["psi"], LOAM["delta_theta"])
    assert sol.F - rhs == pytest.approx(A * math.log1p(sol.F / A), rel=1e-9)


def test_loam_one_hour_visibly_above_Ks_t():
    # 预置壤土：一小时累积入渗因吸力项明显大于 Ks*t=1.09 cm
    sol = solve_cumulative(**LOAM, t=1.0)
    assert sol.F > 1.5 * LOAM["Ks"]
    assert abs(sol.residual) <= sol.tolerance


# --------------------------------------------------------------------------- #
# 残差质量
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("t", [0.0, 1e-9, 1e-4, 0.5, 3.0, 25.0, 500.0, 1e5])
def test_residual_within_tolerance(t):
    sol = solve_cumulative(**LOAM, t=t)
    tol = max(DEFAULT_ABS_TOL, 1e-12 * LOAM["Ks"] * t)
    assert abs(sol.residual) <= tol
    # 直接重算一遍，防止返回的 residual 是自己伪造的
    A = suction_head(LOAM["psi"], LOAM["delta_theta"])
    direct = residual_at(sol.F, A, LOAM["Ks"] * t)
    assert abs(direct) <= tol


def test_tighter_custom_tolerance():
    sol = solve_cumulative(**LOAM, t=2.0, abs_tol=1e-13, rel_tol=1e-14)
    assert abs(sol.residual) <= 1e-13


def test_round_trip_time_inversion():
    # F -> t（显式）-> F（隐式）应回到原值
    sol = solve_cumulative(**LOAM, t=7.5)
    A = suction_head(LOAM["psi"], LOAM["delta_theta"])
    t_back = ponded_time(sol.F, LOAM["Ks"], A)
    assert t_back == pytest.approx(7.5, rel=1e-10)


# --------------------------------------------------------------------------- #
# 单调性：F 只增不减
# --------------------------------------------------------------------------- #

def test_F_monotonic_in_time():
    ts = [0.0, 0.01, 0.05, 0.2, 1.0, 4.0, 20.0, 100.0, 1000.0]
    Fs = [solve_cumulative(**LOAM, t=t).F for t in ts]
    for a, b in zip(Fs, Fs[1:]):
        assert b >= a


# --------------------------------------------------------------------------- #
# 不收敛必须报错，绝不返回没解出来的初值
# --------------------------------------------------------------------------- #

def test_non_convergence_raises_never_returns_guess():
    with pytest.raises(ConvergenceError) as exc:
        solve_cumulative(**LOAM, t=1.0, max_iter=1)
    details = exc.value.details
    assert details["iterations"] == 1
    assert abs(details["residual"]) > details["tolerance"]
    # 错误结构中可附带中间 F，但服务层不会把它当答案（在路由测试里验）


def test_hint_does_not_break_convergence():
    # 分段推进用前一步 F 作提示，解不受提示值影响
    a = solve_cumulative(**LOAM, t=5.0)
    b = solve_cumulative(**LOAM, t=5.0, F_hint=0.3)
    assert a.F == pytest.approx(b.F, rel=1e-11)
