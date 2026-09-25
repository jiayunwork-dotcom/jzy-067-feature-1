"""Green-Ampt 隐式累积入渗量求解器（本文件只负责数值求解，不碰积水分段）。

积水情形下累积入渗量 F 与时间 t 的关系：

    F - psi * dtheta * ln(1 + F / (psi * dtheta)) = Ks * t

令 A = psi * dtheta（湿润锋吸力当量，量纲为长度），右端 rhs = Ks * t：

    g(F) = F - A * ln(1 + F/A) - rhs = 0

**符号是这里最隐蔽的坑**：对数项必须是减号。F - A ln(...) 严格小于 F，
所以真解恒有 F > Ks*t（t>0）；若写成加号，解会掉到 Ks*t 以下，
早期入渗行为整体失真。测试专门用 F > Ks*t 这一因果把符号钉死。

数值方法：牛顿迭代，g'(F) = F / (F + A)，即

    F_{n+1} = F_n - g(F_n) * (F_n + A) / F_n

并配符号括号兜底：每一步若牛顿点跳出括号就退回二分点，按残差符号收缩
括号，保证即使牛顿发散也一定能在有限步内落到阈值内。收敛不了时抛
``ConvergenceError``，绝不返回没解出来的迭代值。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import ConvergenceError

# 残差默认阈值：|g(F)| <= max(abs_tol, rel_tol * rhs)
DEFAULT_ABS_TOL = 1e-10
DEFAULT_REL_TOL = 1e-12
DEFAULT_MAX_ITER = 50


@dataclass(frozen=True)
class PondedSolution:
    """隐式方程求解结果。

    Attributes:
        F: 累积入渗量（长度）。
        t: 对应的积水历时（时间）。
        residual: 隐式方程两侧残差 g(F) = F - A ln(1+F/A) - Ks t。
        tolerance: 本次采用的残差阈值。
        iterations: 实际迭代次数。
        converged: 是否把残差压进了阈值。
    """

    F: float
    t: float
    residual: float
    tolerance: float
    iterations: int
    converged: bool


def suction_head(psi: float, delta_theta: float) -> float:
    """A = psi * delta_theta，湿润锋吸力当量。"""
    return psi * delta_theta


def residual_at(F: float, A: float, rhs: float) -> float:
    """隐式方程残差。对数项为减号，勿反。"""
    return F - A * math.log1p(F / A) - rhs


def ponded_time(F: float, Ks: float, A: float) -> float:
    """时间作为入渗量的显函数：t = [F - A ln(1+F/A)] / Ks。"""
    return (F - A * math.log1p(F / A)) / Ks


def _initial_bracket(A: float, rhs: float, F_hint: float | None) -> tuple[float, float]:
    """构造一个夹住根的括号 [lo, hi]，保证 g(lo) <= 0 <= g(hi)。"""
    candidates: list[float] = []
    if F_hint is not None and F_hint > 0.0:
        # 分段推进时上一步的 F 对新的 rhs（更大）落在根的左侧
        candidates.append(F_hint)
    # 短历时展开 F ~ sqrt(2 A rhs)，该近似在根的上方（残差为正）
    candidates.append(math.sqrt(2.0 * A * rhs))
    # 长历时近似 rhs + A ln(1+rhs/A)，该近似在根的下方（残差为负）
    candidates.append(rhs + A * math.log1p(rhs / A))

    lo = math.inf
    hi = -math.inf
    found_lo = found_hi = False
    for c in candidates:
        if not math.isfinite(c) or c <= 0.0:
            continue
        g = residual_at(c, A, rhs)
        if g <= 0.0:
            found_lo = True
            lo = min(lo, c)
        if g >= 0.0:
            found_hi = True
            hi = max(hi, c)
        if found_lo and found_hi and lo > 0.0:
            break

    if not found_lo:
        lo, found_lo = _expand(A, rhs, min(c for c in candidates if c > 0.0), -1.0)
    if not found_hi:
        hi, found_hi = _expand(A, rhs, max(candidates), 1.0)
    if not (found_lo and found_hi):
        raise ConvergenceError(
            "bracket_failed",
            "无法构造隐式方程的有根括号",
            details={"A": A, "rhs": rhs},
        )
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _expand(A: float, rhs: float, seed: float, direction: float) -> tuple[float, bool]:
    """沿指定方向（-1 往小、+1 往大）几何扩张直到残差变号。"""
    x = seed
    for _ in range(200):
        g = residual_at(x, A, rhs)
        if direction < 0.0 and g <= 0.0:
            return x, True
        if direction > 0.0 and g >= 0.0:
            return x, True
        if direction < 0.0:
            x *= 0.5
            if x <= 0.0:
                return 0.0, residual_at(0.0, A, rhs) <= 0.0
        else:
            x *= 2.0
            if not math.isfinite(x):
                return x, False
    return x, False


def solve_cumulative(
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
    """对 F 求解 F - A ln(1+F/A) = Ks t。

    Args:
        Ks: 饱和导水率（长度/时间），须为正。
        psi: 湿润锋基质吸力（长度），取正数，须为正。
        delta_theta: 锋前后含水量差，(0, 1]。
        t: 积水历时，非负。
        abs_tol / rel_tol: 残差阈值，收敛条件
            ``|g(F)| <= max(abs_tol, rel_tol*rhs)``。
        max_iter: 最大迭代步数，超出即抛 ``ConvergenceError``。
        F_hint: 已知偏小的 F 初值（分段推进时上一步的解），用于收紧括号。

    Returns:
        PondedSolution。t=0 时 F=0、残差 0，直接返回。
    """
    A = suction_head(psi, delta_theta)
    rhs = Ks * t

    if t <= 0.0:
        return PondedSolution(F=0.0, t=float(t), residual=0.0,
                              tolerance=abs_tol, iterations=0, converged=True)

    lo, hi = _initial_bracket(A, rhs, F_hint)
    tol = max(abs_tol, rel_tol * rhs)

    F = hi  # 从根上方起步，牛顿迭代单调下降
    last_residual = residual_at(F, A, rhs)
    iterations = 0
    for n in range(1, max_iter + 1):
        iterations = n
        g = residual_at(F, A, rhs)
        last_residual = g
        if abs(g) <= tol:
            return PondedSolution(F=F, t=float(t), residual=g, tolerance=tol,
                                  iterations=iterations, converged=True)
        # 牛顿步：g'(F) = F/(F+A)
        F_newton = F - g * (F + A) / F
        # 跳出括号或出现非有限值则退回二分点
        if not (math.isfinite(F_newton) and lo < F_newton < hi):
            F_newton = 0.5 * (lo + hi)
        F = F_newton

        g = residual_at(F, A, rhs)
        if g <= 0.0:
            lo = F
        else:
            hi = F

    # 迭代耗尽仍未达标：宁可报错，也不把没解出来的 F 交出去
    raise ConvergenceError(
        "not_converged",
        f"牛顿迭代在 {max_iter} 步内未将残差压到 {tol:g} 以下",
        details={
            "F": F,
            "residual": last_residual,
            "tolerance": tol,
            "iterations": iterations,
        },
    )
