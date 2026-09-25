"""入渗率与积水时刻判定。

本模块在 :mod:`model.solver` 之上解释物理过程，本身不实现隐式迭代：

1. 瞬时入渗率（积水式）``f = Ks (1 + A/F)``，F→0 时形式上发散，
   统一封顶到约定初值 :data:`INITIAL_RATE_CAP`；
2. 降雨强度 ``i <= Ks`` 时入渗能力永远降不到供水强度，**不会积水**；
3. ``i > Ks`` 时用 Mein–Larson 处理：
   自由入渗段 F = i t；能力降到供水强度时 Fp = Ks A/(i-Ks)，
   故积水时刻 ``tp = Fp/i``；积水后以等效积水历时
   ``te = [Fp - A ln(1+Fp/A)] / Ks`` 为偏移接积水隐式：
   ``g(F) = Ks (te + t - tp)``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .solver import (
    DEFAULT_ABS_TOL,
    DEFAULT_MAX_ITER,
    DEFAULT_REL_TOL,
    PondedSolution,
    ponded_time,
    residual_at,
    solve_cumulative,
    suction_head,
)

# 累积入渗量数值下限（长度）：F 小于该值即视作零
F_FLOOR = 1e-10
# 入渗率约定初值（与 Ks 同单位）：形式上的 f(0)=+inf 封顶到此
INITIAL_RATE_CAP = 1.0e6


@dataclass(frozen=True)
class PondingInfo:
    """积水判定结果。

    Attributes:
        will_pond: 该降雨强度下最终是否会积水。
        i: 降雨强度。
        tp: 积水开始时刻（钟表时间）；不会积水时为 None。
        Fp: 积水时刻的累积入渗量；不会积水时为 None。
        equivalent_time: 积水时刻对应的等效积水历时；不会积水时为 None。
        explanation: 判定说明。
    """

    will_pond: bool
    i: float
    tp: float | None = None
    Fp: float | None = None
    equivalent_time: float | None = None
    explanation: str = ""


@dataclass(frozen=True)
class InfiltrationState:
    """某一时刻的入渗核算结果。"""

    t: float
    F: float
    f: float
    f_capacity: float
    phase: str  # "free" | "ponding" | "ponded"
    mode: str   # "ponded_from_zero" | "rainfall"
    residual: float | None
    tolerance: float | None
    iterations: int | None
    i: float | None
    ponding: PondingInfo
    extra: dict[str, Any] = field(default_factory=dict)


def infiltration_rate(F: float, Ks: float, A: float) -> float:
    """瞬时入渗率 f = Ks(1 + A/F)，F 趋零时封顶到约定初值。"""
    if F <= F_FLOOR:
        return INITIAL_RATE_CAP
    raw = Ks * (1.0 + A / F)
    return min(raw, INITIAL_RATE_CAP)


def analyze_ponding(Ks: float, psi: float, delta_theta: float, i: float) -> PondingInfo:
    """给定恒定降雨强度，判定是否积水并给出积水时刻。"""
    A = suction_head(psi, delta_theta)
    if i <= Ks:
        return PondingInfo(
            will_pond=False,
            i=i,
            explanation=(
                f"降雨强度 i={i:g} 不超过饱和导水率 Ks={Ks:g}，"
                "入渗能力始终高于供水强度，地表永远不会积水，全程自由入渗。"
            ),
        )
    Fp = Ks * A / (i - Ks)
    tp = Fp / i  # 自由段 F = i t，钟表时间
    te = ponded_time(Fp, Ks, A)  # 等效积水历时
    return PondingInfo(
        will_pond=True,
        i=i,
        tp=tp,
        Fp=Fp,
        equivalent_time=te,
        explanation=(
            f"降雨强度 i={i:g} 大于饱和导水率 Ks={Ks:g}。"
            f"0~{tp:g} 为自由入渗（F=i·t），t={tp:g} 时入渗能力降到供水强度，"
            f"地表开始积水（Fp={Fp:g}），此后转入积水隐式推进。"
        ),
    )


def solve_ponded_state(
    Ks: float,
    psi: float,
    delta_theta: float,
    t: float,
    *,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    F_hint: float | None = None,
) -> PondedSolution:
    """从时刻零就积水时，求 t 时刻的隐式解。"""
    return solve_cumulative(
        Ks, psi, delta_theta, t,
        abs_tol=abs_tol, rel_tol=rel_tol, max_iter=max_iter, F_hint=F_hint,
    )


def state_at_time(
    Ks: float,
    psi: float,
    delta_theta: float,
    t: float,
    *,
    i: float | None = None,
    already_ponded: bool = False,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    F_hint: float | None = None,
) -> InfiltrationState:
    """核算任意时刻的入渗状态。

    Args:
        already_ponded: 调用方直接声明「地表已经积水」，从时刻零走积水式。
        i: 降雨强度。给出且未声明积水时，先自由入渗、后积水分段。
        F_hint: 上一步的 F，仅传给积水段的牛顿迭代做初值提示。
    """
    A = suction_head(psi, delta_theta)

    if already_ponded or i is None:
        sol = solve_ponded_state(
            Ks, psi, delta_theta, t,
            abs_tol=abs_tol, rel_tol=rel_tol, max_iter=max_iter, F_hint=F_hint,
        )
        f = infiltration_rate(sol.F, Ks, A)
        return InfiltrationState(
            t=t, F=sol.F, f=f, f_capacity=f,
            phase="ponded", mode="ponded_from_zero",
            residual=sol.residual, tolerance=sol.tolerance,
            iterations=sol.iterations, i=None,
            ponding=PondingInfo(will_pond=True, i=float("nan"),
                                explanation="调用方声明地表自始积水，t=0 起走积水式。"),
        )

    info = analyze_ponding(Ks, psi, delta_theta, i)

    if not info.will_pond:
        F = i * t
        return InfiltrationState(
            t=t, F=F, f=i, f_capacity=infiltration_rate(F, Ks, A),
            phase="free", mode="rainfall",
            residual=None, tolerance=None, iterations=None, i=i, ponding=info,
        )

    assert info.tp is not None and info.Fp is not None and info.equivalent_time is not None
    tp, Fp, te = info.tp, info.Fp, info.equivalent_time

    if t < tp:
        F = i * t
        return InfiltrationState(
            t=t, F=F, f=i, f_capacity=infiltration_rate(F, Ks, A),
            phase="free", mode="rainfall",
            residual=None, tolerance=None, iterations=None, i=i, ponding=info,
        )

    if t <= tp + 1e-12 * max(1.0, tp):
        # 恰在积水时刻：F=Fp，隐式残差以等效历时 te 计
        sol_residual = residual_at(Fp, A, Ks * te)
        return InfiltrationState(
            t=t, F=Fp, f=i, f_capacity=i,
            phase="ponding", mode="rainfall",
            residual=sol_residual, tolerance=max(abs_tol, rel_tol * Ks * te),
            iterations=0, i=i, ponding=info,
        )

    # 积水段：g(F) = Ks (te + t - tp)
    elapsed = te + (t - tp)
    sol = solve_cumulative(
        Ks, psi, delta_theta, elapsed,
        abs_tol=abs_tol, rel_tol=rel_tol, max_iter=max_iter,
        F_hint=F_hint if F_hint is not None else Fp,
    )
    f = infiltration_rate(sol.F, Ks, A)
    return InfiltrationState(
        t=t, F=sol.F, f=f, f_capacity=f,
        phase="ponded", mode="rainfall",
        residual=sol.residual, tolerance=sol.tolerance,
        iterations=sol.iterations, i=i, ponding=info,
        extra={"elapsed_ponded_time": elapsed},
    )
