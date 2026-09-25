"""Green-Ampt 入渗观测的参数反演（标定）。

正问题（:mod:`model.solver` / :mod:`model.infiltration`）由参数算入渗过程；
本模块反过来：吃进一串实测 ``(时刻 t, 累积入渗量 F)``，在参数空间里找一组
值，使各观测点模型值与实测值之差的平方和（SSE）最小，并老实交代这次标定
信不信得过。

被反演的核心是两个量：

- ``Ks``：饱和导水率；
- ``A = psi * delta_theta``：湿润锋吸力与含水量差的乘积项（长度量纲）。

正问题自始至终只通过乘积 ``A`` 接触 ``psi`` 与 ``delta_theta``，所以内部
适配时以 ``psi=A, delta_theta=1.0`` 调 :func:`state_at_time`，不新写任何
正向方程；需要落工况档案时再由调用方补一个 ``delta_theta`` 把 ``A`` 拆开。

可辨识性（本模块最重要的一节）
------------------------------

只给**积水段**观测时，模型曲线

    F - A ln(1 + F/A) = Ks t

对 (Ks, A) 的约束在一个方向上几乎没有曲率（短历时段 F≈√(2 A Ks t)，
只钉得住乘积 Ks·A）。沿这个弱方向同时挪动两个参数，SSE 几乎不动——
单个参数各自定不准，能定准的只是强方向上的那个组合。服务在收敛点对
（对数参数的）灵敏度矩阵做 SVD：最小奇异值相对过小即判
``combination_only``——**只回报可辨识组合与弱方向，绝不把一组看着精确、
其实躺在脊线上的 Ks/A 当答案**。

以下三种情形简并被打破，两个量可分别标出（``individual``）：

1. 观测带已知降雨强度 ``i``，且含有自由入渗段/跨过积水起始点——
   积水时刻 ``tp = Ks A / [i(i-Ks)]`` 直接参与约束；
2. 调用方固定其中一个参数（``fix_Ks`` 或 ``fix_A``），只剩一个自由量；
3. 数据本身横跨早、晚两种渐近形态，SVD 证实两个方向都有足够曲率。

迭代收敛铁律与正问题一致：只有把 SSE 压到迭代再也走不动（步长与目标
下降都落入判据）且停点在参数域内部，才产出结果；迭代耗尽、撞到参数
边界、数值跑飞或被取消，一律抛错，**绝不交半路参数**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .errors import ServiceError, ValidationError
from .infiltration import state_at_time

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

MIN_OBSERVATIONS = 3
# 对数参数的活动范围：参数落在 [e^-LOG_BOUND, e^LOG_BOUND]，
# 既挡住数值跑飞，也用来识别“最优值撞边界”这种非内部驻点。
LOG_BOUND = 25.0
_BOUND_MARGIN = 1e-3
# 有限差分步长（对数空间，相对步长）
_FD_STEP = 1e-7
# LM 收敛判据
_XTOL = 1e-11
_FTOL = 1e-13
_GTOL = 1e-13
DEFAULT_CAL_MAX_ITER = 200

# 可辨识性阈值：归一化灵敏度矩阵最小/最大奇异值比低于它即判单参不可辨。
# 取 1e-3：弱方向曲率不到强方向千分之一时，沿脊挪动的参数变化已无意义。
IDENTIFIABILITY_RATIO = 1e-3
_SENSITIVITY_FLOOR = 1e-8

# 拟合质量（仅供诚实汇报，不用来掩盖不收敛）
NRMSE_GOOD = 0.05
NRMSE_POOR = 0.20

CancelCheck = Callable[[], bool]
ProgressCb = Callable[[int, float, dict[str, float]], None]


class CalibrationError(ServiceError):
    """反演迭代未能确立可信的最小点。

    与正问题的 :class:`~model.errors.ConvergenceError` 同级：发生时绝不把
    最后一步的中间参数当标定结果吐出。
    """

    status_code = 422


class CalibrationCancelled(Exception):
    """标定作业被外部取消。带出的信息仅供诊断，绝不当结果。"""

    def __init__(self, iterations: int, trace: list[dict[str, Any]]) -> None:
        super().__init__("标定作业已取消")
        self.iterations = iterations
        self.trace = trace


class _StartOutOfDomain(Exception):
    """多起点之一落在参数域之外（如扰动后 Ks≥i），跳过该起点。"""


# --------------------------------------------------------------------------- #
# 观测校验（必须挡在一切迭代之前）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Observation:
    """一条实测点：钟表时刻 t 与该时刻的实测累积入渗量 F。"""

    t: float
    F: float


def _obs_number(value: Any, field_name: str, idx: int) -> float:
    if isinstance(value, bool):
        raise ValidationError(
            "invalid_parameter", f"第 {idx} 个观测点的 {field_name} 不能是布尔值"
        )
    if not isinstance(value, (int, float)):
        raise ValidationError(
            "invalid_parameter",
            f"第 {idx} 个观测点的 {field_name} 必须是数值，收到 {type(value).__name__}",
        )
    fv = float(value)
    if not math.isfinite(fv):
        raise ValidationError(
            "invalid_parameter", f"第 {idx} 个观测点的 {field_name} 必须有限"
        )
    return fv


def validate_observations(raw: Any) -> list[Observation]:
    """校验观测序列，在任何迭代发生之前挡下坏数据。

    硬性拦截（400）：不是数组、点数太少、时刻不严格递增、t 或 F 为负、
    出现非有限值。F 不单调（物理上不可能）不在此拦——交给 SSE 诚实暴露
    成一个很差的拟合，而不是替调用方改写数据。
    """
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValidationError("invalid_parameter", "observations 必须是点序列数组")
    if len(raw) < MIN_OBSERVATIONS:
        raise ValidationError(
            "too_few_observations",
            f"观测点至少需要 {MIN_OBSERVATIONS} 个（收到 {len(raw)} 个），"
            "点数太少无法支撑参数反演",
        )

    obs: list[Observation] = []
    prev_t: float | None = None
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValidationError(
                "invalid_parameter", f"第 {idx} 个观测点必须是 {{'t':..,'F':..}} 对象"
            )
        t = _obs_number(item.get("t"), "t", idx)
        F = _obs_number(item.get("F"), "F", idx)
        if t < 0.0:
            raise ValidationError(
                "invalid_parameter", f"第 {idx} 个观测点时刻 t={t} 为负"
            )
        if F < 0.0:
            raise ValidationError(
                "negative_infiltration",
                f"第 {idx} 个观测点累积入渗量 F={F} 为负，物理上不成立",
            )
        if prev_t is not None and t <= prev_t:
            raise ValidationError(
                "times_not_increasing",
                f"第 {idx} 个观测点时刻 {t:g} 不严格大于上一时刻 {prev_t:g}",
            )
        prev_t = t
        obs.append(Observation(t=t, F=F))

    if all(abs(o.F) < 1e-300 for o in obs):
        raise ValidationError(
            "invalid_parameter", "所有观测点入渗量均为 0，不含任何可拟合信息"
        )
    return obs


def validate_positive_param(value: Any, name: str) -> float:
    """校验固定值/初值类的正参数。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("invalid_parameter", f"{name} 必须是数值")
    fv = float(value)
    if not math.isfinite(fv) or fv <= 0.0:
        raise ValidationError("invalid_parameter", f"{name} 必须为正的有限数")
    return fv


# --------------------------------------------------------------------------- #
# 正问题适配：模型只认 (Ks, A)，复用既有正向解算
# --------------------------------------------------------------------------- #

def model_cumulative(Ks: float, A: float, t: float, i: float | None) -> float:
    """给定 (Ks, A) 求 t 时刻模型累积入渗量。

    以 psi=A、delta_theta=1.0 走既有 :func:`state_at_time`：正问题中二者
    只以乘积 A 出现，结果与拆分方式无关。
    """
    st = state_at_time(Ks, A, 1.0, t, i=i)
    return st.F


def _assemble(free_names: Sequence[str], x: Sequence[float],
              fixed: dict[str, float]) -> dict[str, float]:
    p = dict(fixed)
    for name, v in zip(free_names, x):
        p[name] = math.exp(v)
    return p


def _residuals(params: dict[str, float], obs: Sequence[Observation],
               i: float | None) -> list[float]:
    Ks, A = params["Ks"], params["A"]
    return [model_cumulative(Ks, A, o.t, i) - o.F for o in obs]


def _sse(r: Sequence[float]) -> float:
    return sum(v * v for v in r)


# --------------------------------------------------------------------------- #
# 初值启发式（多起点兜底用）
# --------------------------------------------------------------------------- #

def _invert_A_for_point(F: float, Ks: float, t: float) -> float | None:
    """已知 Ks，由单点积水关系反解 A：F - A ln(1+F/A) = Ks t。

    h(A)=F-A ln(1+F/A) 随 A 单调递减（A→0 时 h→F，A→∞ 时 h→0），
    故 0 < Ks t < F 时根唯一，二分求之。
    """
    rhs = Ks * t
    if rhs <= 0.0 or rhs >= F:
        return None
    lo, hi = 1e-12, 1e12
    if F - lo * math.log1p(F / lo) <= rhs or F - hi * math.log1p(F / hi) >= rhs:
        return None
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if F - mid * math.log1p(F / mid) > rhs:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1.0 + 1e-10:
            break
    return 0.5 * (lo + hi)


def _heuristic_initial(obs: Sequence[Observation], i: float | None,
                       fixed: dict[str, float]) -> dict[str, float]:
    """给一组落在数据量级附近的 (Ks, A) 初值。"""
    ts = [o.t for o in obs]
    Fs = [o.F for o in obs]

    # 末段弦斜率≈渐近入渗率（≥Ks）；首点用短历时展开 F²≈2 A Ks t
    late_slope = (Fs[-1] - Fs[-2]) / (ts[-1] - ts[-2]) if ts[-1] > ts[-2] else 0.0

    if i is None:
        Ks0 = fixed.get("Ks", max(late_slope * 0.8, 1e-6))
        A_est = [a for o in obs
                 if (a := _invert_A_for_point(o.F, Ks0, o.t)) is not None]
        A0 = fixed.get("A", A_est[len(A_est) // 2] if A_est
                       else Fs[-1] ** 2 / (2.0 * Ks0 * ts[-1]))
        return {"Ks": max(Ks0, 1e-9), "A": max(A0, 1e-9)}

    # 降雨情形：Ks 必须小于 i。从弦斜率跌破 i 的位置估积水时刻，
    # 再用 tp = Ks A/[i(i-Ks)] 反推 A。
    slopes = [(Fs[k + 1] - Fs[k]) / (ts[k + 1] - ts[k])
              for k in range(len(obs) - 1) if ts[k + 1] > ts[k]]
    kink_slope = next((s for s in slopes if s < 0.98 * i), None)
    Ks0 = fixed.get("Ks", min(0.5 * i, max(late_slope * 0.8, 1e-6)))
    if Ks0 >= i:
        Ks0 = 0.5 * i
    if "A" in fixed:
        return {"Ks": Ks0, "A": fixed["A"]}
    if kink_slope is not None and kink_slope > Ks0:
        kidx = next(k for k in range(len(slopes)) if slopes[k] < 0.98 * i)
        tp = 0.5 * (ts[kidx] + ts[kidx + 1])
        A0 = tp * i * (i - Ks0) / Ks0
    else:
        # 看不到明显转折：取一个积水后点按积水式反演，量级兜底
        A_from_pts = [a for o in obs
                      if (a := _invert_A_for_point(o.F, Ks0, o.t)) is not None]
        A0 = A_from_pts[len(A_from_pts) // 2] if A_from_pts else 1.0
    return {"Ks": max(Ks0, 1e-9), "A": max(A0, 1e-9)}


def _initial_starts(obs: Sequence[Observation], i: float | None,
                    fixed: dict[str, float],
                    explicit: dict[str, float] | None) -> list[dict[str, float]]:
    """汇总多起点：显式初值优先，启发式 + 两个受扰点兜底。"""
    starts: list[dict[str, float]] = []
    if explicit is not None:
        starts.append(dict(explicit))
    starts.append(_heuristic_initial(obs, i, fixed))
    h = starts[-1]
    starts.append({"Ks": h["Ks"] * 1.7, "A": h["A"] / 1.7})
    starts.append({"Ks": max(h["Ks"] / 1.7, 1e-8), "A": h["A"] * 1.7})

    uniq: list[dict[str, float]] = []
    cap = 0.9 * i if i is not None else math.inf
    for s in starts:
        s.update(fixed)
        s["Ks"] = min(s["Ks"], cap)
        if not all(math.isfinite(v) and v > 0.0 for v in s.values()):
            continue
        if any(abs(s["Ks"] / u["Ks"] - 1.0) < 1e-8 and
               abs(s["A"] / u["A"] - 1.0) < 1e-8 for u in uniq):
            continue
        uniq.append(s)
    if uniq:
        return uniq
    fallback_Ks = min(1.0, 0.5 * i) if i is not None else 1.0
    return [{"Ks": fallback_Ks, "A": 1.0, **fixed}]


# --------------------------------------------------------------------------- #
# Levenberg–Marquardt（对数参数空间）
# --------------------------------------------------------------------------- #

@dataclass
class _LMOutcome:
    converged: bool
    x: list[float]
    params: dict[str, float]
    residuals: list[float]
    sse: float
    iterations: int
    evaluations: int
    at_boundary: bool
    reason: str
    trace: list[dict[str, Any]] = field(default_factory=list)


def _solve_2x2(a: float, b: float, d: float, rhs1: float,
               rhs2: float) -> tuple[float, float]:
    det = a * d - b * b
    if det <= 0.0 or not math.isfinite(det):
        # 阻尼正规方程的右端本就是 -Jr（最速下降方向）：矩阵奇异时
        # 直接沿该方向走，由外层信赖域的 mu 控制步长。
        return rhs1, rhs2
    return ((d * rhs1 - b * rhs2) / det, (a * rhs2 - b * rhs1) / det)


def _levenberg_marquardt(
    start: dict[str, float],
    free_names: Sequence[str],
    fixed: dict[str, float],
    obs: Sequence[Observation],
    i: float | None,
    *,
    log_lo: Sequence[float],
    log_hi: Sequence[float],
    max_iter: int,
    should_cancel: CancelCheck | None,
    progress: ProgressCb | None,
    start_index: int,
) -> _LMOutcome:
    x0 = [math.log(start[name]) for name in free_names]
    # 初值本身可能贴着参数域（多起点扰动可能越过降雨模式的 Ks<i 上界）：
    # 越界的起点直接放弃，而不是夹到边界上，否则初值就贴着墙。
    if any(not (lo + _BOUND_MARGIN < v < hi - _BOUND_MARGIN)
           for v, lo, hi in zip(x0, log_lo, log_hi)):
        raise _StartOutOfDomain()
    x = list(x0)
    p = _assemble(free_names, x, fixed)
    r = _residuals(p, obs, i)
    evaluations = 1
    sse = _sse(r)
    if not math.isfinite(sse):
        raise CalibrationError("numerical_overflow", "初值点模型评估出现非有限残差")

    nf = len(free_names)
    mu = 1e-3
    nu = 2.0
    trace: list[dict[str, Any]] = []
    last_rel_step = math.inf
    last_rel_drop = math.inf

    def record(it: int, accepted: bool) -> None:
        trace.append({
            "start": start_index,
            "iteration": it,
            "accepted": accepted,
            "Ks": p["Ks"], "A": p["A"], "sse": sse, "mu": mu,
        })

    def clip(v: float, lo: float, hi: float) -> tuple[float, bool]:
        if not math.isfinite(v):
            return (hi if v > 0 or math.isnan(v) else lo), True
        if v > hi:
            return hi, True
        if v < lo:
            return lo, True
        return v, False

    for it in range(1, max_iter + 1):
        if should_cancel is not None and should_cancel():
            raise CalibrationCancelled(it, trace)

        # 前向差分雅可比（对数参数）；扰动点同样夹在参数域内
        J = [[0.0] * nf for _ in obs]
        for k in range(nf):
            xb = x[k]
            step = min(_FD_STEP, 0.5 * (log_hi[k] - log_lo[k]),
                       log_hi[k] - _BOUND_MARGIN - xb,
                       xb - log_lo[k] - _BOUND_MARGIN)
            x[k] = xb + step
            pk = _assemble(free_names, x, fixed)
            x[k] = xb
            rk = _residuals(pk, obs, i)
            evaluations += 1
            for j in range(len(obs)):
                J[j][k] = (rk[j] - r[j]) / step

        # JᵀJ 与 Jᵀr
        JJ = [[0.0] * nf for _ in range(nf)]
        Jr = [0.0] * nf
        for j, row in enumerate(J):
            for a in range(nf):
                Jr[a] += row[a] * r[j]
                for b in range(a, nf):
                    JJ[a][b] += row[a] * row[b]
        for a in range(nf):
            for b in range(a):
                JJ[a][b] = JJ[b][a]

        scale2 = [max(JJ[k][k], 1e-300) for k in range(nf)]
        grad_inf = max(abs(Jr[k]) / scale2[k] for k in range(nf))
        if grad_inf <= _GTOL:
            record(it, True)
            if progress:
                progress(it, sse, p)
            return _LMOutcome(True, x, p, r, sse, it, evaluations, False,
                              "梯度已平", trace)

        accepted = False
        rho = -1.0
        sse_new = math.inf
        clipped = False
        d_try: list[float] = []
        for _inner in range(30):
            damp = [JJ[k][k] * (1.0 + mu) if JJ[k][k] > 0.0 else mu
                    for k in range(nf)]
            if nf == 1:
                d = [-Jr[0] / damp[0]] if damp[0] > 0.0 else [0.0]
            else:
                d = list(_solve_2x2(damp[0], JJ[0][1], damp[1], -Jr[0], -Jr[1]))

            x_new = [0.0] * nf
            clip_flags = [False] * nf
            for k in range(nf):
                x_new[k], clip_flags[k] = clip(x[k] + d[k], log_lo[k], log_hi[k])
            clipped = any(clip_flags)
            p_new = _assemble(free_names, x_new, fixed)
            r_new = _residuals(p_new, obs, i)
            evaluations += 1
            sse_new = _sse(r_new)

            # 预测下降量（信赖域二次模型）：-rᵀJd + ½(Jd)ᵀ(Jd)
            pred = 0.0
            for j in range(len(obs)):
                Jd = sum(J[j][a] * d[a] for a in range(nf))
                pred += -r[j] * Jd + 0.5 * Jd * Jd
            if pred <= 0.0:
                pred = 1e-300

            rho = (sse - sse_new) / pred if math.isfinite(sse_new) else -1.0
            d_try = d
            if rho > 0.0:
                accepted = True
                break
            mu *= nu
            nu *= 2.0
            if mu > 1e16:
                break

        if accepted:
            rel_drop = (sse - sse_new) / max(sse, 1e-300)
            rel_step = max(abs(d_try[k]) / (_XTOL + abs(x[k]))
                           for k in range(nf)) if d_try else 0.0
            x = x_new
            p, r, sse = p_new, r_new, sse_new
            last_rel_step, last_rel_drop = rel_step, rel_drop
            mu = max(mu * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), 1e-12)
            nu = 2.0
            record(it, True)
            if progress:
                progress(it, sse, p)
            at_any_bound = clipped or any(
                abs(x[k] - log_lo[k]) <= _BOUND_MARGIN
                or abs(log_hi[k] - x[k]) <= _BOUND_MARGIN
                for k in range(nf)
            )
            if at_any_bound:
                # 唯一能下降的步子撞到参数域边界（如 A→0、Ks→i）：不是可信
                # 的内部最小点，按未收敛处理，绝不交出边界上的参数
                return _LMOutcome(False, x, p, r, sse, it, evaluations, True,
                                  "最优解撞到参数域边界，未形成内部驻点", trace)
            if rel_step <= _XTOL and abs(rel_drop) <= _FTOL:
                return _LMOutcome(True, x, p, r, sse, it, evaluations, False,
                                  "步长与目标下降均落入收敛判据", trace)
        else:
            record(it, False)
            if progress:
                progress(it, sse, p)
            # 所有阻尼档位都被拒绝：步长已被压到极小。若这一步在当前尺度
            # 上本来就走不动，则噪声底附近的驻点也算收敛；否则才算失败。
            stalled = all(abs(d_try[k]) <= _XTOL * (1.0 + abs(x[k]))
                          for k in range(nf)) if d_try else True
            if stalled:
                return _LMOutcome(True, x, p, r, sse, it, evaluations, False,
                                  "试探步在当前尺度上已走不动（噪声底驻点）", trace)
            if mu > 1e14:
                return _LMOutcome(False, x, p, r, sse, it, evaluations, False,
                                  "信赖域反复拒绝试探步，无法继续下降", trace)

    return _LMOutcome(False, x, p, r, sse, max_iter, evaluations, False,
                      f"迭代 {max_iter} 步内未压入收敛判据（末步相对步长 "
                      f"{last_rel_step:.2e}）", trace)


# --------------------------------------------------------------------------- #
# 可辨识性分析：归一化灵敏度矩阵的 SVD（参数维数 ≤ 2，闭式 2×2 特征分解）
# --------------------------------------------------------------------------- #

def _sensitivity_jacobian(params: dict[str, float], free_names: Sequence[str],
                          obs: Sequence[Observation], i: float | None,
                          scale: float) -> list[list[float]]:
    """对自由参数（取对数）的归一化前差雅可比：∂F_model/∂ln p / F_scale。"""
    r0 = _residuals(params, obs, i)
    nf = len(free_names)
    J = [[0.0] * nf for _ in obs]
    for k, name in enumerate(free_names):
        full = dict(params)
        full[name] = params[name] * math.exp(_FD_STEP)
        rk = _residuals(full, obs, i)
        for j in range(len(obs)):
            J[j][k] = ((rk[j] - r0[j]) / _FD_STEP) / scale
    return J


def _eigen_2x2(m00: float, m01: float, m11: float,
               lam: float) -> tuple[float, float]:
    """返回 (M - lam I) 的归一化零向量。"""
    if abs(m01) > 1e-300:
        vx, vy = lam - m11, m01
    elif abs(m00 - lam) >= abs(m11 - lam):
        vx, vy = 0.0, 1.0
    else:
        vx, vy = 1.0, 0.0
    nrm = math.hypot(vx, vy)
    return vx / nrm, vy / nrm


def _svd_analysis(J: Sequence[Sequence[float]]) -> dict[str, Any]:
    nf = len(J[0])
    JJ = [[0.0] * nf for _ in range(nf)]
    for row in J:
        for a in range(nf):
            for b in range(nf):
                JJ[a][b] += row[a] * row[b] / len(J)
    if nf == 1:
        s = math.sqrt(max(JJ[0][0], 0.0))
        return {"singular_values": [s], "condition_number": 1.0,
                "weak": None, "strong": (1.0,)}

    tr = JJ[0][0] + JJ[1][1]
    disc = math.sqrt(max(0.0, 0.25 * (JJ[0][0] - JJ[1][1]) ** 2 + JJ[0][1] ** 2))
    lam_max = 0.5 * tr + disc
    lam_min = max(0.5 * tr - disc, 0.0)
    s_max = math.sqrt(max(lam_max, 0.0))
    s_min = math.sqrt(lam_min)
    weak = _eigen_2x2(JJ[0][0], JJ[0][1], JJ[1][1], lam_min)
    strong = _eigen_2x2(JJ[0][0], JJ[0][1], JJ[1][1], lam_max)
    return {
        "singular_values": [s_max, s_min],
        "condition_number": (s_max / s_min) if s_min > 0.0 else math.inf,
        "weak": weak,
        "strong": strong,
    }


def _ponded_phase_coverage(params: dict[str, float],
                           obs: Sequence[Observation], i: float) -> dict[str, Any]:
    """统计在当前参数下，有多少观测点真正落在积水相（t > tp）。

    降雨模式打破简并靠的是积水时刻 tp 与积水后曲线同时受约束；若所有点
    都在自由段（模型只是把 tp 挤到最后一个点之后/之上就能零残差），
    观测对土壤参数实质无约束。
    """
    from .infiltration import analyze_ponding

    info = analyze_ponding(params["Ks"], params["A"], 1.0, i)
    if not info.will_pond or info.tp is None:
        return {"tp": None, "n_ponded": 0, "n_free": len(obs)}
    tol = 1e-9 * max(1.0, info.tp)
    n_ponded = sum(1 for o in obs if o.t > info.tp + tol)
    n_free = sum(1 for o in obs if o.t < info.tp - tol)
    return {"tp": info.tp, "n_ponded": n_ponded, "n_free": n_free}


def _identifiability(outcome: _LMOutcome, free_names: Sequence[str],
                     fixed: dict[str, float], obs: Sequence[Observation],
                     i: float | None, F_scale: float) -> dict[str, Any]:
    """在最小点做 SVD 灵敏度分析，裁定可辨识性。

    规矩（与正问题的分段规矩同级）：只给积水段观测时，(Ks, A) 在方程里
    以组合形式进方程、存在沿脊挪动的方向——无论当前数据窗口在数值上让
    弱方向显得多平或多斜，一律按 ``combination_only`` 处置，不给虚假的
    单参值；SVD 的奇异值/弱方向作为诊断佐证随结果回报。简并只可能被
    以下信息打破：固定一个参数，或给出降雨强度（积水时刻 tp 参与约束）。
    """
    J = _sensitivity_jacobian(outcome.params, free_names, obs, i, F_scale)
    ana = _svd_analysis(J)
    sv = ana["singular_values"]
    s_max = sv[0]
    s_min = sv[1] if len(sv) == 2 else s_max
    base = {
        "free_parameters": list(free_names),
        "fixed": dict(fixed),
        "singular_values": [s_max, s_min],
        "condition_number": (ana["condition_number"] if len(free_names) == 2 else 1.0),
    }

    # 降雨模式：先看相态覆盖。全部在自由段时土壤参数不进模型，
    # 模型只要把 tp 挤过最后一个点就能零残差，数据实质无约束。
    coverage = None
    if i is not None:
        coverage = _ponded_phase_coverage(outcome.params, obs, i)
        if coverage["n_ponded"] < 2:
            raise CalibrationError(
                "uninformative_observations",
                "观测几乎全部落在自由入渗段（最优积水时刻被挤到最后一个"
                "观测点边上，真正位于积水后的点不足 2 个）：自由段 F=i·t "
                "不依赖 Ks/A，请补入积水起始之后的观测。",
                details=coverage,
            )

    # 单自由参数：简并方向不存在。但数据对它也得真有灵敏度
    # （全部落在自由段 F=i·t 时，两个参数都不进模型）。
    if len(free_names) == 1:
        if s_max <= _SENSITIVITY_FLOOR:
            raise CalibrationError(
                "uninformative_observations",
                "所有观测点对自由参数都没有灵敏度（可能全部落在自由入渗段 "
                "F=i·t，且该段不依赖土壤参数），无法标定",
            )
        base["phase_coverage"] = coverage
        base.update({
            "status": "individual",
            "weak_direction": None,
            "identifiable_combination": None,
            "note": f"仅 {free_names[0]} 自由（其余已固定），不存在简并方向。",
        })
        return base

    if s_max <= _SENSITIVITY_FLOOR:
        raise CalibrationError(
            "uninformative_observations",
            "所有观测点对两个参数都没有灵敏度（例如全部落在自由入渗段 "
            "F=i·t），数据对参数毫无约束，无法标定",
        )

    ratio = s_min / s_max if s_max > 0.0 else 0.0
    w = ana["weak"]
    if w[0] < 0.0:
        w = (-w[0], -w[1])
    svec = ana["strong"]
    if svec[0] < 0.0:
        svec = (-svec[0], -svec[1])
    Ks_h, A_h = outcome.params["Ks"], outcome.params["A"]
    combo_value = math.exp(svec[0] * math.log(Ks_h) + svec[1] * math.log(A_h))
    hold_product = abs((w[0] - w[1]) / math.sqrt(2.0))

    if (i is not None and ratio >= IDENTIFIABILITY_RATIO
            and coverage is not None and coverage["n_ponded"] >= 2
            and coverage["n_free"] >= 1):
        # 简并被打破：积水前/积水后都有点，tp 与积水曲线共同约束两个量
        base["phase_coverage"] = coverage
        base.update({
            "status": "individual",
            "weak_direction": None,
            "identifiable_combination": None,
            "note": (
                "观测含已知降雨强度，且同时覆盖自由入渗段"
                f"（{coverage['n_free']} 点）与积水后段（{coverage['n_ponded']} 点），"
                "积水时刻 tp=Ks·A/[i(i−Ks)] 与积水后曲线共同约束；"
                f"SVD 最小/最大奇异值比 {ratio:.3g} ≥ {IDENTIFIABILITY_RATIO:g}，"
                "Ks 与 A 可分别辨识。"
            ),
        })
        return base

    # 积水段观测：组合可辨、单参不可辨。
    note = (
        "观测只覆盖积水段，Ks 与 A 以组合形式一起进入 "
        "F−A ln(1+F/A)=Ks·t：沿对数方向 (d ln Ks, d ln A)="
        f"({w[0]:.3f}, {w[1]:.3f}) 同时挪动两参数，残差平方和几乎不动，"
        "Ks 与 A 各自不可信，只有强方向上的组合可辨识。"
    )
    if ratio >= IDENTIFIABILITY_RATIO:
        note += (
            f" 本次数据窗口较宽（SVD 奇异值比 {ratio:.3g} 并不算小），"
            "数值上勉强能分，但积水段在模型结构上本就允许沿脊挪动，"
            "按规矩仍只交付组合，拒绝回报看似精确的单参值；"
            "要分别标定请固定其一，或提供含自由入渗段、带降雨强度的观测。"
        )
    if hold_product > 0.9:
        note += (
            f" 弱方向基本就是“保持 Ks·A={Ks_h * A_h:.6g} 不变”的方向"
            "（短历时 F≈√(2 A Ks t) 的典型简并）。"
        )
    if i is not None:
        if coverage is not None and coverage["n_free"] == 0:
            note += (
                " 虽给了降雨强度 i，但观测全部位于积水之后、没有自由段点，"
                "积水起始时刻 tp 未被观测夹住，简并未被有效打破。"
            )
        elif coverage is not None and coverage["n_ponded"] < 2:
            note += (
                " 虽给了降雨强度 i，但积水后的观测点不足，"
                "积水起始时刻 tp 未被有效约束，简并未被打破。"
            )
        else:
            note += (
                " 虽给了降雨强度 i，但 SVD 弱方向仍明显，"
                "简并未被有效打破。"
            )
    base["phase_coverage"] = coverage
    base.update({
        "status": "combination_only",
        "weak_direction": {"log_Ks": w[0], "log_A": w[1]},
        "identifiable_combination": {
            "form": "Ks^a * A^b",
            "log_coefficients": {"log_Ks": svec[0], "log_A": svec[1]},
            "a": svec[0], "b": svec[1],
            "value": combo_value,
        },
        "Ks_times_A": Ks_h * A_h,
        "hold_product_alignment": hold_product,
        "singular_ratio": ratio,
        "note": note,
    })
    return base


# --------------------------------------------------------------------------- #
# 结果结构
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CalibrationResult:
    converged: bool
    mode: str
    rainfall_i: float | None
    free_parameters: list[str]
    fixed: dict[str, float]
    fitted: dict[str, Any]
    ridge_point: dict[str, float] | None
    sse: float
    rmse: float
    nrmse: float
    quality: str
    iterations: int
    total_iterations: int
    function_evaluations: int
    n_starts: int
    convergence_reason: str
    identifiability: dict[str, Any]
    residuals: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    n_observations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "mode": self.mode,
            "rainfall_i": self.rainfall_i,
            "free_parameters": self.free_parameters,
            "fixed": self.fixed,
            "fitted": self.fitted,
            "ridge_point": self.ridge_point,
            "sse": self.sse,
            "rmse": self.rmse,
            "nrmse": self.nrmse,
            "fit_quality": self.quality,
            "iterations": self.iterations,
            "total_iterations": self.total_iterations,
            "function_evaluations": self.function_evaluations,
            "n_starts": self.n_starts,
            "convergence_reason": self.convergence_reason,
            "identifiability": self.identifiability,
            "residuals": self.residuals,
            "iteration_trace": self.trace,
            "n_observations": self.n_observations,
        }


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

def run_calibration(
    observations: Sequence[Observation] | Sequence[dict[str, Any]],
    *,
    i: float | None = None,
    fixed: dict[str, float] | None = None,
    initial: dict[str, float] | None = None,
    max_iter: int = DEFAULT_CAL_MAX_ITER,
    should_cancel: CancelCheck | None = None,
    on_progress: ProgressCb | None = None,
) -> CalibrationResult:
    """对观测序列做非线性最小二乘标定。

    Args:
        observations: 已校验的 :class:`Observation` 列表（或原始 dict 列表，
            此时先过 :func:`validate_observations`）。
        i: 已知恒定降雨强度；给 None 表示自始积水。
        fixed: 固定参数，形如 ``{"Ks": x}`` 或 ``{"A": x}``，用来打破简并。
        initial: 显式初值 ``{"Ks": .., "A": ..}``；None 则用启发式多起点。
        max_iter: 单个起点的最大 LM 迭代步数。
        should_cancel / on_progress: 后台作业的取消与进度钩子。

    Raises:
        ValidationError: 观测或参数入箱前不合法。
        CalibrationError: 所有起点都没能在内部驻点收敛 / 数据无约束。
        CalibrationCancelled: 被外部取消。
    """
    obs: Sequence[Observation] = (
        observations if (observations and isinstance(observations[0], Observation))
        else validate_observations(observations)
    )
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise ValidationError("invalid_parameter", "max_iter 必须是正整数")
    fixed_in: dict[str, float] = {}
    for name, val in (fixed or {}).items():
        if name not in ("Ks", "A"):
            raise ValidationError("invalid_parameter", f"未知固定参数 {name}")
        fixed_in[name] = validate_positive_param(val, f"fixed.{name}")
    if len(fixed_in) == 2:
        raise ValidationError("invalid_parameter",
                              "两个参数都被固定则无需标定")
    initial_in = None
    if initial is not None:
        initial_in = {
            "Ks": validate_positive_param(initial.get("Ks"), "initial.Ks"),
            "A": validate_positive_param(initial.get("A"), "initial.A"),
        }

    free_names = [n for n in ("Ks", "A") if n not in fixed_in]
    # 参数域（对数空间）：降雨模式 Ks>=i 时模型退化为全程 F=i·t，
    # 对土壤参数完全平坦（伪台地），故 Ks 必须钉在 i 以下；A 无上界。
    # 若 Ks 被固定在 >= i，则积水时刻不存在，下面用观测形态另行把关。
    log_lo: list[float] = []
    log_hi: list[float] = []
    for name in free_names:
        log_lo.append(-LOG_BOUND)
        if name == "Ks":
            log_hi.append(math.log(i) if i is not None else LOG_BOUND)
        else:
            log_hi.append(LOG_BOUND)
    if i is not None and fixed_in.get("Ks", 0.0) >= i:
        raise ValidationError(
            "invalid_parameter",
            f"固定的 Ks={fixed_in['Ks']:g} 不小于降雨强度 i={i:g}：此工况永不"
            "积水，土壤参数不进自由入渗段 F=i·t，无法据其标定",
        )
    starts = _initial_starts(obs, i, fixed_in, initial_in)

    outcomes: list[_LMOutcome] = []
    total_iter = 0
    total_evals = 0
    for si, st in enumerate(starts):
        if should_cancel is not None and should_cancel():
            raise CalibrationCancelled(total_iter, [])
        try:
            oc = _levenberg_marquardt(
                st, free_names, fixed_in, obs, i,
                log_lo=log_lo, log_hi=log_hi,
                max_iter=max_iter, should_cancel=should_cancel,
                progress=on_progress, start_index=si,
            )
        except _StartOutOfDomain:
            continue
        outcomes.append(oc)
        total_iter += oc.iterations
        total_evals += oc.evaluations

    if not outcomes:  # pragma: no cover - 启发式起点恒在域内，仅防御
        raise CalibrationError("not_converged", "所有初值均落在可行参数域之外")

    good = [o for o in outcomes if o.converged]
    if not good:
        worst = min(outcomes, key=lambda o: o.sse)
        raise CalibrationError(
            "not_converged",
            "所有起点的最小二乘迭代都未能压入收敛判据，拒绝返回中间参数",
            details={
                "reason": worst.reason,
                "iterations": total_iter,
                "best_sse_reached": worst.sse,
                "at_boundary": worst.at_boundary,
                "n_starts": len(starts),
            },
        )
    winner = min(good, key=lambda o: o.sse)

    F_max = max(o.F for o in obs)
    F_scale = max(F_max, 1e-300)
    ident = _identifiability(winner, free_names, fixed_in, obs, i, F_scale)

    rmse = math.sqrt(winner.sse / len(obs))
    nrmse = rmse / F_scale
    quality = ("good" if nrmse <= NRMSE_GOOD else
               "poor" if nrmse >= NRMSE_POOR else "fair")

    point_rows: list[dict[str, Any]] = []
    for o, rv in zip(obs, winner.residuals):
        Fm = o.F + rv
        point_rows.append({
            "t": o.t, "F_observed": o.F, "F_model": Fm,
            "residual": rv,
        })

    if ident["status"] == "individual":
        fitted: dict[str, Any] = {
            "Ks": winner.params["Ks"],
            "A": winner.params["A"],
            "psi": None,
            "delta_theta": None,
        }
        ridge = None
    else:
        # 单参不可辨：绝不把脊线上的点冒充标定值放在顶层答案里
        fitted = {"Ks": None, "A": None, "psi": None, "delta_theta": None}
        ridge = {"Ks": winner.params["Ks"], "A": winner.params["A"],
                 "note": "残差脊线上的任意一点，仅作诊断，Ks/A 单项不可信"}

    return CalibrationResult(
        converged=True,
        mode="ponded" if i is None else "rainfall",
        rainfall_i=i,
        free_parameters=list(free_names),
        fixed=dict(fixed_in),
        fitted=fitted,
        ridge_point=ridge,
        sse=winner.sse,
        rmse=rmse,
        nrmse=nrmse,
        quality=quality,
        iterations=winner.iterations,
        total_iterations=total_iter,
        function_evaluations=total_evals,
        n_starts=len(starts),
        convergence_reason=winner.reason,
        identifiability=ident,
        residuals=point_rows,
        trace=winner.trace,
        n_observations=len(obs),
    )
