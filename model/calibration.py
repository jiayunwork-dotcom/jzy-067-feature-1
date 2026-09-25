"""Green–Ampt 入渗反演：由实测 (t, F) 观测标定土壤参数（独立成块，不碰正向分段逻辑）。

被标定的两个核心量：

- ``Ks``   饱和导水率；
- ``A``    湿润锋吸力当量 ``psi * delta_theta``（乘积项本身，不拆）。

目标函数为各观测点模型值与实测值之差的平方和，用 Levenberg–Marquardt
（Nielsen 阻尼更新 + 多起点兜底）在对数参数空间迭代；正向求值反复调用
:func:`model.infiltration.state_at_time` / :func:`model.solver.solve_cumulative`
这套已有正问题解算，本模块不重写隐式求解。

**可辨识性是本模块的核心责任**。只有积水段观测时，Ks 与 A 沿
``(ln Ks, ln A)`` 的某个方向近似简并（短时极限下只剩乘积 Ks·A 可定），
单靠条件数说话：对数空间列归一化雅可比的相关系数 |rho| 对应
``kappa = sqrt((1+|rho|)/(1-|rho|))``，超过 :data:`KAPPA_IDENTIFIABLE_MAX`
即判 ``combination_only``——明确回报可辨识的组合，绝不把两个单参当成
都定准了交差。降雨模式下积水起始时刻与参数挂钩、简并被打破，同一套
判据会自然落到 ``both_identifiable``。判据只看数据，不看模式名。

铁律与正问题一致：迭代压不进收敛判据、或数值跑飞（顶到参数盒边界、
落入 Ks>=i 无信号死区），抛 :class:`model.errors.ConvergenceError`，
绝不把半路参数当结果。收敛判据中步长/目标函数两条必须搭配梯度佐证，
防止被大阻尼压出的假收敛。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .errors import ConvergenceError, ValidationError
from .infiltration import analyze_ponding, state_at_time
from .solver import solve_cumulative

# --------------------------------------------------------------------------- #
# 常量与判据约定
# --------------------------------------------------------------------------- #

# 对数参数盒：Ks、A 的搜索范围 [1e-12, 1e12]（单位自洽下的物理宽限）
_LN_LO = math.log(1e-12)
_LN_HI = math.log(1e12)

# LM 默认收敛判据
DEFAULT_CAL_MAX_ITER = 100
DEFAULT_GTOL = 1e-10     # 梯度判据：max|J^T r| <= gtol * max(1, SSR)
DEFAULT_XTOL = 1e-12     # 步长判据（需梯度佐证）
DEFAULT_FTOL = 1e-14     # 目标下降判据（需梯度佐证）
# 步长/目标判据的梯度佐证阈值（比正式梯度判据宽，防大阻尼假收敛）
_GRAD_LOOSE = 1e-4

# 可辨识性约定：列归一化对数雅可比的条件数上限。
# kappa > 50 即 |rho| > 0.9992——两列在任意现实噪声下都不可区分。
KAPPA_IDENTIFIABLE_MAX = 50.0

# 单观测 (t, F) 的字段名
OBS_T = "t"
OBS_F = "F"


class CalibrationCancelled(Exception):
    """标定迭代被外部取消。只带诊断计数，绝不带半成品参数。"""

    def __init__(self, iterations_done: int) -> None:
        super().__init__("标定作业已取消")
        self.iterations_done = iterations_done


# --------------------------------------------------------------------------- #
# 观测校验（挡在任何迭代之前）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Observation:
    """一对实测（时刻, 累积入渗量）。"""

    t: float
    F: float


def _obs_number(value: Any, field: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            "invalid_observation",
            f"第 {index} 个观测的 {field} 必须是数值",
        )
    v = float(value)
    if not math.isfinite(v):
        raise ValidationError(
            "invalid_observation", f"第 {index} 个观测的 {field} 必须是有限数值"
        )
    return v


def validate_observations(raw: Any, *, n_free: int = 2) -> list[Observation]:
    """把请求里的观测序列校验并规范化为 Observation 列表。

    接受两种形态：``[{"t":..,"F":..}, ...]`` 或 ``{"t":[...], "F":[...]}``。
    规则：点数至少 n_free+1；时刻非负且严格递增；入渗量非负有限；
    至少一个正入渗量（否则数据不含可标定信号）；t=0 处 F 必须为 0
    （模型定义 F(0)=0，非零初值是 malformed 观测）。
    """
    pairs: list[tuple[Any, Any]] = []
    if isinstance(raw, dict):
        ts, Fs = raw.get("t"), raw.get("F")
        if not isinstance(ts, list) or not isinstance(Fs, list) or len(ts) != len(Fs):
            raise ValidationError(
                "invalid_observation",
                "观测的平行数组形态必须是 {\"t\": [...], \"F\": [...]} 且等长",
            )
        pairs = list(zip(ts, Fs))
    elif isinstance(raw, list):
        for k, item in enumerate(raw):
            if not isinstance(item, dict) or "t" not in item or "F" not in item:
                raise ValidationError(
                    "invalid_observation",
                    f"第 {k} 个观测必须是含 t 与 F 的对象",
                )
            pairs.append((item["t"], item["F"]))
    else:
        raise ValidationError(
            "invalid_observation",
            "observations 必须是 [{\"t\",\"F\"}, ...] 或 {\"t\":[...],\"F\":[...]}",
        )

    min_points = n_free + 1
    if len(pairs) < min_points:
        raise ValidationError(
            "invalid_observation",
            f"观测点太少：{len(pairs)} 个，标定 {n_free} 个自由参数至少需要 "
            f"{min_points} 个",
        )

    obs: list[Observation] = []
    prev_t = -math.inf
    any_positive_F = False
    for k, (raw_t, raw_F) in enumerate(pairs):
        t = _obs_number(raw_t, "t", k)
        F = _obs_number(raw_F, "F", k)
        if t < 0.0:
            raise ValidationError(
                "invalid_observation", f"第 {k} 个观测时刻为负（t={t:g}）"
            )
        if t <= prev_t:
            raise ValidationError(
                "invalid_observation",
                f"观测时刻必须严格递增：第 {k} 个 t={t:g} 不晚于前一个 {prev_t:g}",
            )
        if F < 0.0:
            raise ValidationError(
                "invalid_observation", f"第 {k} 个观测入渗量为负（F={F:g}）"
            )
        if t == 0.0 and F != 0.0:
            raise ValidationError(
                "invalid_observation",
                f"第 {k} 个观测 t=0 但 F={F:g}≠0：累积入渗量零点必须为 0",
            )
        if F > 0.0:
            any_positive_F = True
        obs.append(Observation(t=t, F=F))
        prev_t = t

    if not any_positive_F:
        raise ValidationError(
            "invalid_observation", "所有观测入渗量都是 0，数据不含可标定信号"
        )
    return obs


# --------------------------------------------------------------------------- #
# 正向模型包装：自由参数向量 -> 各观测点的模型 F
# --------------------------------------------------------------------------- #

class _ForwardModel:
    """把 (i, 固定参数) 绑定后的正向求值器。

    自由参数是对数空间里的 [ln Ks]、[ln A] 或其子集（固定其一）。
    降雨模式下 Ks >= i 是无信号死区（模型退化为 F=i·t，与参数无关），
    求值直接判失败，让 LM 把该方向当作不可行步。
    """

    def __init__(
        self,
        obs: Sequence[Observation],
        i: float | None,
        fixed: dict[str, float],
    ) -> None:
        self._obs = list(obs)
        self._i = i
        self.fixed = dict(fixed)
        self.free_names: tuple[str, ...] = tuple(
            name for name in ("Ks", "A") if name not in self.fixed
        )
        self.evaluations = 0

    @property
    def mode(self) -> str:
        return "rainfall" if self._i is not None else "ponded_from_zero"

    def _params(self, x: Sequence[float]) -> tuple[float, float]:
        values = dict(zip(self.free_names, (math.exp(v) for v in x)))
        Ks = values.get("Ks", self.fixed.get("Ks"))
        A = values.get("A", self.fixed.get("A"))
        assert Ks is not None and A is not None
        return Ks, A

    def in_box(self, x: Sequence[float]) -> bool:
        return all(_LN_LO <= v <= _LN_HI for v in x)

    def try_residuals(self, x: Sequence[float]) -> tuple[list[float], float] | None:
        """求残差列与 SSR；越盒、死区、正向不收敛、非有限值一律返回 None。"""
        if not self.in_box(x):
            return None
        Ks, A = self._params(x)
        if self._i is not None and Ks >= self._i:
            return None  # 死区：Ks>=i 时永不积水，模型不含参数信号
        try:
            residuals = []
            for ob in self._obs:
                if self._i is None:
                    F_model = solve_cumulative(Ks, A, 1.0, ob.t).F
                else:
                    F_model = state_at_time(Ks, A, 1.0, ob.t, i=self._i).F
                residuals.append(F_model - ob.F)
        except (ConvergenceError, OverflowError, ValueError, ZeroDivisionError):
            return None
        self.evaluations += 1
        ssr = sum(r * r for r in residuals)
        if not math.isfinite(ssr):
            return None
        return residuals, ssr

    def _F_at(self, Ks: float, A: float, t: float) -> float:
        if self._i is None:
            return solve_cumulative(Ks, A, 1.0, t).F
        return state_at_time(Ks, A, 1.0, t, i=self._i).F

    def jacobian(self, x: Sequence[float]) -> list[tuple[float, ...]]:
        """对数空间雅可比（n_obs × n_free）。

        积水模式用解析列：dF/dlnKs = Ks·t·(F+A)/F，dF/dlnA = F − dF/dlnKs；
        降雨模式（积水点随参数移动）用中心差分；差分支点跨进死区/盒外时
        退回单侧差分，绝不让支点处的异常外泄中断迭代。
        """
        rows: list[tuple[float, ...]] = []
        if self._i is None:
            Ks, A = self._params(x)
            col: dict[str, int] = {name: k for k, name in enumerate(self.free_names)}
            for ob in self._obs:
                F = solve_cumulative(Ks, A, 1.0, ob.t).F
                j_Ks = Ks * ob.t * (F + A) / F if F > 0.0 else 0.0
                j_A = F - j_Ks
                row = [0.0, 0.0]
                if "Ks" in col:
                    row[col["Ks"]] = j_Ks
                if "A" in col:
                    row[col["A"]] = j_A
                rows.append(tuple(row[: len(self.free_names)]))
            return rows

        h = 1e-6

        def feasible(pt: Sequence[float]) -> bool:
            """盒内且不跨死区（不触发正向求解）。"""
            if not self.in_box(pt):
                return False
            if self._i is None:
                return True
            Ks_try, _ = self._params(pt)
            return Ks_try < self._i

        for ob in self._obs:
            Ks_c, A_c = self._params(x)
            F_c = self._F_at(Ks_c, A_c, ob.t)
            row = []
            for k in range(len(self.free_names)):
                xp = list(x)
                xm = list(x)
                xp[k] += h
                xm[k] -= h
                p_ok, m_ok = feasible(xp), feasible(xm)
                try:
                    if p_ok and m_ok:
                        Fp = self._F_at(*self._params(xp), ob.t)
                        Fm = self._F_at(*self._params(xm), ob.t)
                        row.append((Fp - Fm) / (2.0 * h))
                    elif p_ok or m_ok:
                        side, sign = (xp, 1.0) if p_ok else (xm, -1.0)
                        F_s = self._F_at(*self._params(side), ob.t)
                        row.append(sign * (F_s - F_c) / h)
                    else:
                        row.append(0.0)
                except (ConvergenceError, OverflowError, ValueError,
                        ZeroDivisionError):
                    row.append(0.0)
            rows.append(tuple(row))
        return rows


# --------------------------------------------------------------------------- #
# Levenberg–Marquardt（1~2 个自由参数，Nielsen 阻尼更新）
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _LMOutcome:
    x: tuple[float, ...]
    ssr: float
    iterations: int
    criterion: str  # "gradient" | "step" | "objective"
    jacobian: list[tuple[float, ...]]
    residuals: list[float]
    gradient_inf: float


def _solve_normal(H: list[list[float]], g: list[float], lam: float) -> list[float] | None:
    """解 (H·(1+λ) 对角阻尼) δ = −g，m ∈ {1, 2}；奇异或非有限返回 None。"""
    m = len(g)
    if m == 1:
        a = H[0][0] * (1.0 + lam)
        if a <= 0.0 or not math.isfinite(a):
            return None
        return [-g[0] / a]
    a11 = H[0][0] * (1.0 + lam)
    a22 = H[1][1] * (1.0 + lam)
    a12 = H[0][1]
    det = a11 * a22 - a12 * a12
    if det <= 0.0 or not math.isfinite(det):
        return None
    d1 = (-g[0] * a22 + a12 * g[1]) / det
    d2 = (-a11 * g[1] + a12 * g[0]) / det
    if not (math.isfinite(d1) and math.isfinite(d2)):
        return None
    return [d1, d2]


def _lm_run(
    model: _ForwardModel,
    x0: Sequence[float],
    *,
    max_iter: int,
    gtol: float,
    xtol: float,
    ftol: float,
    should_cancel: Callable[[], bool] | None,
    progress: Callable[[int, float], None] | None,
    iter_offset: int = 0,
) -> _LMOutcome:
    """从 x0 跑一次 LM。压不进判据抛 ConvergenceError，绝不交半路值。"""
    m = len(x0)
    x = [float(v) for v in x0]
    got = model.try_residuals(x)
    if got is None:
        raise ConvergenceError(
            "calibration_not_converged",
            "初始点无法求值（越出参数盒或落入 Ks≥i 无信号死区）",
            details={"start": [math.exp(v) for v in x0]},
        )
    residuals, ssr = got
    lam, nu = 1e-3, 2.0
    J: list[tuple[float, ...]] = []
    ginf = math.inf

    for it in range(1, max_iter + 1):
        if should_cancel is not None and should_cancel():
            raise CalibrationCancelled(iter_offset + it - 1)

        J = model.jacobian(x)
        g = [sum(row[k] * r for row, r in zip(J, residuals)) for k in range(m)]
        H = [
            [sum(row[a] * row[b] for row in J) for b in range(m)]
            for a in range(m)
        ]
        h_trace = sum(H[k][k] for k in range(m))
        if h_trace == 0.0:
            raise ConvergenceError(
                "no_parameter_signal",
                "雅可比恒为零：观测对参数没有任何敏感度"
                "（如降雨模式下观测全落在自由入渗段 F=i·t）",
            )
        ginf = max(abs(v) for v in g)
        if ginf <= gtol * max(1.0, ssr):
            return _LMOutcome(tuple(x), ssr, it, "gradient", J, residuals, ginf)
        grad_ok_loose = ginf <= _GRAD_LOOSE * max(1.0, ssr)

        accepted = False
        delta: list[float] = []
        new_res: list[float] = []
        new_ssr = math.inf
        for _ in range(50):
            delta = _solve_normal(H, g, lam) or []
            if delta:
                trial = [x[k] + delta[k] for k in range(m)]
                got = model.try_residuals(trial)
            else:
                got = None
            if got is not None:
                new_res, new_ssr = got
                if new_ssr < ssr:
                    # Nielsen 增益比更新阻尼
                    denom = 0.5 * sum(
                        delta[k] * (lam * H[k][k] * delta[k] - g[k]) for k in range(m)
                    )
                    rho = (ssr - new_ssr) / denom if denom > 0.0 else 1.0
                    lam = lam * max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                    nu = 2.0
                    accepted = True
                    break
            lam *= nu
            nu *= 2.0
            if lam > 1e13:
                break
        if not accepted:
            raise ConvergenceError(
                "calibration_not_converged",
                f"LM 第 {it} 步停滞：阻尼放到 1e13 仍找不到下降步"
                f"（SSR={ssr:.6g}，|g|={ginf:.3g}）",
                details={"ssr": ssr, "gradient_inf": ginf, "iterations": it},
            )

        step_norm = math.hypot(*delta)
        decrease = ssr - new_ssr
        x = [x[k] + delta[k] for k in range(m)]
        residuals, ssr = new_res, new_ssr
        if progress is not None:
            progress(iter_offset + it, ssr)
        # 步长/目标判据必须搭配梯度佐证，否则只是被大阻尼压住
        if grad_ok_loose and step_norm <= xtol * (math.hypot(*x) + xtol):
            return _LMOutcome(tuple(x), ssr, it, "step", J, residuals, ginf)
        if grad_ok_loose and decrease <= ftol * max(1.0, ssr):
            return _LMOutcome(tuple(x), ssr, it, "objective", J, residuals, ginf)

    raise ConvergenceError(
        "calibration_not_converged",
        f"LM 在 {max_iter} 步内未压进收敛判据（SSR={ssr:.6g}，|g|={ginf:.3g}）",
        details={
            "ssr": ssr,
            "gradient_inf": ginf,
            "iterations": max_iter,
            "max_iter": max_iter,
        },
    )


# --------------------------------------------------------------------------- #
# 可辨识性分析
# --------------------------------------------------------------------------- #

def _identifiability(
    model: _ForwardModel,
    J: list[tuple[float, ...]],
    ssr: float,
    x: Sequence[float],
) -> dict[str, Any]:
    """由解点处的对数雅可比判定可辨识性并给出不确定度。"""
    n = len(J)
    m = len(model.free_names)
    dof = max(n - m, 1)
    s2 = ssr / dof

    if m == 1:
        h = sum(row[0] * row[0] for row in J)
        se = math.sqrt(s2 / h) if h > 0.0 else math.inf
        name = model.free_names[0]
        return {
            "status": "identifiable_via_constraint",
            "condition_number": None,
            "explanation": (
                f"另一个参数已被调用方固定，唯一自由参数 {name} 由数据直接标定；"
                "不存在组合简并方向。"
            ),
            "parameters": {name: {"relative_std_error": se}},
        }

    n1 = math.sqrt(sum(row[0] * row[0] for row in J))
    n2 = math.sqrt(sum(row[1] * row[1] for row in J))
    if n1 == 0.0 or n2 == 0.0:
        dead = model.free_names[0] if n1 == 0.0 else model.free_names[1]
        return {
            "status": "combination_only",
            "condition_number": math.inf,
            "jacobian_correlation": None,
            "explanation": (
                f"参数 {dead} 对观测没有任何敏感度（雅可比列恒为零），"
                "它完全不可辨；另一个参数可由数据定出。"
            ),
            "resolution_hint": (
                "要钉死两个参数，请补充能覆盖参数敏感区的观测"
                "（如含自由入渗段并给降雨强度 i），或用 fix 固定其一。"
            ),
            "parameters": {
                model.free_names[0]: {"relative_std_error": math.inf if n1 == 0.0 else 0.0},
                model.free_names[1]: {"relative_std_error": math.inf if n2 == 0.0 else 0.0},
            },
        }
    rho = sum(row[0] * row[1] for row in J) / (n1 * n2)
    denom = 1.0 - abs(rho)
    kappa = math.sqrt((1.0 + abs(rho)) / denom) if denom > 1e-15 else math.inf

    H11 = n1 * n1
    H22 = n2 * n2
    H12 = rho * n1 * n2
    det = H11 * H22 - H12 * H12
    if det <= 0.0:
        # 两列在数值上完全平行：只有组合方向，单参标准误为无穷
        inv11 = inv22 = math.inf
        se1 = se2 = math.inf
    else:
        inv11 = H22 / det
        inv22 = H11 / det
        se1 = math.sqrt(s2 * inv11)
        se2 = math.sqrt(s2 * inv22)

    names = model.free_names  # ("Ks", "A")
    params = {
        names[0]: {"relative_std_error": se1},
        names[1]: {"relative_std_error": se2},
    }

    if kappa <= KAPPA_IDENTIFIABLE_MAX:
        return {
            "status": "both_identifiable",
            "condition_number": kappa,
            "jacobian_correlation": rho,
            "explanation": (
                f"列归一化对数雅可比条件数 κ={kappa:.3g} ≤ "
                f"{KAPPA_IDENTIFIABLE_MAX:g}，两个参数可被数据分别钉死。"
            ),
            "parameters": params,
        }

    # 组合可辨：归一化 2×2 雅可比的特征方向恒为 (±1,1)/√2
    signs = (1.0 if rho >= 0.0 else -1.0, 1.0)
    w1 = (signs[0] / math.sqrt(2.0), 1.0 / math.sqrt(2.0))
    w2 = (-w1[1], w1[0])
    combo_log_value = w1[0] * x[0] + w1[1] * x[1]
    # 组合方向与简并方向的 log 空间标准误
    if det <= 0.0:
        se_combo = 0.0
        se_degenerate = math.inf
    else:
        se_combo = math.sqrt(s2 * (w1[0] ** 2 * inv11 + w1[1] ** 2 * inv22
                                   + 2.0 * w1[0] * w1[1] * (-H12 / det)))
        se_degenerate = math.sqrt(s2 * (w2[0] ** 2 * inv11 + w2[1] ** 2 * inv22
                                        + 2.0 * w2[0] * w2[1] * (-H12 / det)))
    combo_kind = "product_Ks_A" if rho >= 0.0 else "ratio_Ks_over_A"
    return {
        "status": "combination_only",
        "condition_number": kappa,
        "jacobian_correlation": rho,
        "explanation": (
            f"列归一化对数雅可比条件数 κ={kappa:.3g} 超过 "
            f"{KAPPA_IDENTIFIABLE_MAX:g}：沿 (ln Ks, ln A) 的简并方向移动时"
            "目标函数几乎不动，两个参数各自定不准；能定准的只是它们的组合。"
            "下面给出的参数值落在解谷上、随初值漂移，不可当作分别标定的结果。"
        ),
        "resolution_hint": (
            "要分别钉死两个参数，请补充能打破简并的信息：用 fix 固定 Ks 或 A "
            "之一；或提供含自由入渗段的观测并给出降雨强度 i，"
            "让积水起始时刻参与约束。"
        ),
        "identifiable_combination": {
            "kind": combo_kind,
            "weights": {"ln_Ks": w1[0], "ln_A": w1[1]},
            "log_value": combo_log_value,
            "value": math.exp(combo_log_value),
            "relative_std_error": se_combo,
        },
        "degenerate_direction": {
            "weights": {"ln_Ks": w2[0], "ln_A": w2[1]},
            "relative_std_error": se_degenerate,
        },
        "parameters": params,
    }


# --------------------------------------------------------------------------- #
# 初值猜测与多起点兜底
# --------------------------------------------------------------------------- #

def heuristic_initial(obs: Sequence[Observation], i: float | None,
                      fixed: dict[str, float]) -> dict[str, float]:
    """由观测粗猜 (Ks, A)：末段斜率估 Ks，中点短时展开估乘积 Ks·A。"""
    first, mid, last = obs[0], obs[len(obs) // 2], obs[-1]
    span = max(last.t - obs[-2].t, 1e-300)
    slope = max((last.F - obs[-2].F) / span, 1e-12)
    Ks0 = fixed.get("Ks", min(0.7 * slope, 0.9 * i) if i is not None else 0.7 * slope)
    Ks0 = min(max(Ks0, 1e-9), 1e9)
    if "A" in fixed:
        A0 = fixed["A"]
    else:
        # 短时展开 F ≈ sqrt(2·A·Ks·t) → A ≈ F²/(2·Ks·t)
        anchor = mid if mid.t > 0.0 and mid.F > 0.0 else last
        A0 = anchor.F * anchor.F / (2.0 * Ks0 * max(anchor.t, 1e-300))
        A0 = min(max(A0, 1e-9), 1e9)
    return {"Ks": Ks0, "A": A0}


def _start_points(
    model: _ForwardModel,
    initial: dict[str, float] | None,
    obs: Sequence[Observation],
    i: float | None,
) -> list[tuple[float, ...]]:
    """首起点 + 兜底起点（对数空间），全部夹在参数盒内。"""
    heur = heuristic_initial(obs, i, model.fixed)
    base = dict(heur)
    if initial:
        base.update(initial)
    base.update(model.fixed)

    def clip(v: float) -> float:
        return min(max(v, 1e-9), 1e9)

    seeds: list[dict[str, float]] = [base]
    # 沿简并方向与正交方向拉伸的兜底起点
    for f_Ks, f_A in ((0.2, 5.0), (5.0, 0.2), (0.1, 0.1), (10.0, 10.0)):
        seeds.append({
            "Ks": clip(base["Ks"] * f_Ks),
            "A": clip(base["A"] * f_A),
        })

    points: list[tuple[float, ...]] = []
    seen: set[tuple[float, ...]] = set()
    for seed in seeds:
        seed.update(model.fixed)
        pt = tuple(
            math.log(min(max(seed[name], 1e-12), 1e12))
            for name in model.free_names
        )
        if pt not in seen:
            seen.add(pt)
            points.append(pt)
    return points


# --------------------------------------------------------------------------- #
# 标定主入口
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CalibrationResult:
    """一次收敛标定的完整可复核结果。"""

    Ks: float
    A: float
    ssr: float
    iterations: int
    total_iterations: int
    forward_evaluations: int
    criterion: str
    mode: str
    i: float | None
    fixed: dict[str, float]
    n_observations: int
    residuals: list[dict[str, float]]
    identifiability: dict[str, Any]
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def identifiable(self) -> bool:
        return self.identifiability["status"] != "combination_only"

    def to_dict(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        idf_params = self.identifiability.get("parameters", {})
        for name, value in (("Ks", self.Ks), ("A", self.A)):
            entry: dict[str, Any] = {"value": value}
            if name in self.fixed:
                entry["role"] = "fixed"
                entry["identifiable"] = True
            else:
                entry["role"] = "calibrated"
                entry["identifiable"] = self.identifiability["status"] != "combination_only"
                extra = idf_params.get(name)
                if extra:
                    entry["relative_std_error"] = extra["relative_std_error"]
            params[name] = entry
        return {
            "converged": True,
            "model_mode": self.mode,
            "rainfall_i": self.i,
            "parameters": params,
            "sum_squared_residuals": self.ssr,
            "n_observations": self.n_observations,
            "iterations": self.iterations,
            "total_iterations": self.total_iterations,
            "forward_evaluations": self.forward_evaluations,
            "convergence_criterion": self.criterion,
            "residuals": self.residuals,
            "identifiability": self.identifiability,
            "attempts": self.attempts,
        }


def calibrate(
    observations: Sequence[Observation],
    *,
    i: float | None = None,
    fixed: dict[str, float] | None = None,
    initial: dict[str, float] | None = None,
    max_iter: int = DEFAULT_CAL_MAX_ITER,
    gtol: float = DEFAULT_GTOL,
    xtol: float = DEFAULT_XTOL,
    ftol: float = DEFAULT_FTOL,
    should_cancel: Callable[[], bool] | None = None,
    progress: Callable[[int, int, float], None] | None = None,
) -> CalibrationResult:
    """标定主流程：多起点 LM，第一个收敛的起点胜出。

    Args:
        observations: 已校验的观测序列。
        i: 降雨强度；None 表示自始积水。
        fixed: 固定参数（{"Ks": v} 或 {"A": v}，至多一个）。
        initial: 初值 {"Ks":..,"A":..}（缺项由启发式补齐）。
        should_cancel: 返回 True 即刻取消（抛 CalibrationCancelled）。
        progress: 每个被接受的 LM 步回调 (attempt_index, iteration, ssr)。

    Raises:
        CalibrationCancelled: 被取消。
        ConvergenceError: 所有起点都压不进判据 / 无信号 / 数值跑飞。
    """
    fixed = dict(fixed or {})
    for name, value in fixed.items():
        if name not in ("Ks", "A"):
            raise ValidationError("invalid_parameter",
                                  f"fix 只能固定 Ks 或 A，收到 {name!r}")
        if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
            raise ValidationError("invalid_parameter", f"固定的 {name} 必须为正有限数")
    if i is not None and fixed.get("Ks", 0.0) >= i:
        raise ValidationError(
            "invalid_parameter",
            "固定的 Ks 不小于降雨强度 i：永不积水，观测对参数无信号",
        )
    if initial:
        for name, value in initial.items():
            if name not in ("Ks", "A"):
                raise ValidationError("invalid_parameter",
                                      f"initial 只接受 Ks 或 A，收到 {name!r}")
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0):
                raise ValidationError("invalid_parameter",
                                      f"初值 {name} 必须为正有限数")
        if i is not None and initial.get("Ks", 0.0) >= i:
            raise ValidationError(
                "invalid_parameter",
                "初值 Ks 不小于降雨强度 i：落入无信号死区，无法起步",
            )
    model = _ForwardModel(observations, i, fixed)
    starts = _start_points(model, initial, observations, i)

    attempts: list[dict[str, Any]] = []
    total_iters = 0
    best: _LMOutcome | None = None
    best_x: tuple[float, ...] | None = None
    failure_reasons: list[str] = []

    for attempt_idx, x0 in enumerate(starts):
        if should_cancel is not None and should_cancel():
            raise CalibrationCancelled(total_iters)
        start_desc = {
            name: math.exp(v) for name, v in zip(model.free_names, x0)
        }
        start_desc.update(fixed)

        def _progress(it: int, ssr: float, _idx: int = attempt_idx) -> None:
            if progress is not None:
                progress(_idx, it, ssr)

        try:
            outcome = _lm_run(
                model, x0,
                max_iter=max_iter, gtol=gtol, xtol=xtol, ftol=ftol,
                should_cancel=should_cancel, progress=_progress,
                iter_offset=total_iters,
            )
        except CalibrationCancelled:
            raise
        except ConvergenceError as exc:
            attempts.append({
                "attempt": attempt_idx,
                "start": start_desc,
                "status": "failed",
                "reason": exc.reason,
            })
            total_iters += int(exc.details.get("iterations", 0))
            failure_reasons.append(exc.reason)
            continue

        total_iters += outcome.iterations
        attempts.append({
            "attempt": attempt_idx,
            "start": start_desc,
            "status": "converged",
            "iterations": outcome.iterations,
            "ssr": outcome.ssr,
            "criterion": outcome.criterion,
        })
        best, best_x = outcome, outcome.x
        break

    if best is None or best_x is None:
        raise ConvergenceError(
            "calibration_not_converged",
            "所有起点都未能把目标函数压进收敛判据："
            + "；".join(failure_reasons[:3]),
            details={
                "attempts": attempts,
                "total_iterations": total_iters,
                "max_iter": max_iter,
            },
        )

    # 解点不得顶在参数盒边界（顶边界 = 无约束最小在盒外，解不可信）
    if any(abs(v - _LN_LO) < 1e-6 or abs(v - _LN_HI) < 1e-6 for v in best_x):
        raise ConvergenceError(
            "degenerate_solution",
            "标定迭代顶到参数搜索盒边界，最小值落在盒外，"
            "观测很可能不符合 Green–Ampt 模型形态，判为拟合失败",
            details={"ssr": best.ssr},
        )

    values = dict(zip(model.free_names, (math.exp(v) for v in best_x)))
    Ks = values.get("Ks", fixed.get("Ks"))
    A = values.get("A", fixed.get("A"))
    assert Ks is not None and A is not None

    # 降雨模式诚实性检查：最后一个观测必须真正深入积水段。
    # 拟合会把 tp 往观测窗外推来对 F=i·t 数据取得（近）零残差——此时任何
    # tp≥t_max 的参数组合都同样完美，参数完全不受约束，这不是标定成功。
    if i is not None:
        ponding = analyze_ponding(Ks, A, 1.0, i)
        t_max = model._obs[-1].t
        if ponding.tp is None or not ponding.tp <= 0.98 * t_max:
            raise ConvergenceError(
                "no_parameter_signal",
                "拟合出的积水时刻不早于最后观测时刻的 98%：数据基本全程"
                "自由入渗 F=i·t，积水隐式方程没有被观测真正约束到，"
                "参数不可标定（请补积水起始时刻之后更远处的观测）",
                details={
                    "ssr": best.ssr,
                    "tp": ponding.tp,
                    "t_max": t_max,
                    "required_last_observation_beyond": "tp*1.02",
                },
            )

    identifiability = _identifiability(model, best.jacobian, best.ssr, best_x)

    residuals = [
        {
            "t": ob.t,
            "F_observed": ob.F,
            "F_model": ob.F + r,
            "residual": r,
        }
        for ob, r in zip(model._obs, best.residuals)
    ]

    return CalibrationResult(
        Ks=Ks, A=A, ssr=best.ssr,
        iterations=best.iterations, total_iterations=total_iters,
        forward_evaluations=model.evaluations,
        criterion=best.criterion, mode=model.mode, i=i, fixed=fixed,
        n_observations=len(model._obs),
        residuals=residuals,
        identifiability=identifiability,
        attempts=attempts,
    )
