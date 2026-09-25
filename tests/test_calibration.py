"""反演标定数值层测试。

钉牢的因果链（需求逐条）：

1. 已知参数正向生成无噪声观测 → 反向标定高精度还原（可辨识前提下）；
2. 纯积水段、F≪A 的简并数据 → combination_only，绝不虚报单参；
3. 含自由入渗段、简并被打破 → 两个参数分别标出；
4. 固定一个参数 → 另一个直接标定；
5. 小扰动观测 → 参数连续漂移、SSR 单调抬升、仍收敛；
6. 乱跳/非单调观测 → 报错或诚实大残差，绝不假装标得准；
7. 迭代打转（max_iter 被压死）→ 落报错分支，不交半路参数；
8. 取消只抛取消异常，不带半成品；
9. 观测校验在迭代之前挡下坏数据。
"""

from __future__ import annotations

import math

import pytest

from model.calibration import (
    CalibrationCancelled,
    KAPPA_IDENTIFIABLE_MAX,
    calibrate,
    validate_observations,
)
from model.errors import ConvergenceError, ValidationError
from model.infiltration import analyze_ponding, state_at_time
from model.solver import solve_cumulative


def lin(a, b, n):
    return [a + k * (b - a) / (n - 1) for k in range(n)]


def make_obs(ts, Fs):
    return validate_observations([{"t": t, "F": f} for t, f in zip(ts, Fs)])


def ponded_F(Ks, A, t):
    return solve_cumulative(Ks, A, 1.0, t).F


def rain_F(Ks, A, i, t):
    return state_at_time(Ks, A, 1.0, t, i=i).F


# 典型真参数
LOAM = (1.09, 11.01 * 0.434)     # Ks, A
CLAY = (0.2, 31.63 * 0.476)
SAND = (8.25, 4.95 * 0.417)


# --------------------------------------------------------------------------- #
# 1. 无噪声正向生成 → 高精度还原
# --------------------------------------------------------------------------- #

def test_ponded_identifiable_roundtrip_high_precision():
    Ks_t, A_t = SAND
    ts = lin(0.05, 6.0, 20)
    obs = make_obs(ts, [ponded_F(Ks_t, A_t, t) for t in ts])
    res = calibrate(obs)
    assert res.ssr < 1e-18
    assert res.iterations >= 1
    assert res.criterion in ("gradient", "step", "objective")
    # 高精度还原（远高于野外任何噪声水平）
    assert rel(res.Ks, Ks_t) < 1e-8
    assert rel(res.A, A_t) < 1e-8
    assert res.identifiability["status"] == "both_identifiable"
    assert res.identifiability["condition_number"] <= KAPPA_IDENTIFIABLE_MAX
    # 逐点残差与目标函数对得上
    point_ssr = sum(r["residual"] ** 2 for r in res.residuals)
    assert abs(point_ssr - res.ssr) < 1e-24
    assert len(res.residuals) == len(obs)


def test_rainfall_breaks_degeneracy_recovers_both():
    Ks_t, A_t = LOAM
    i = 3.0 * Ks_t
    tp = analyze_ponding(Ks_t, A_t, 1.0, i).tp
    # 观测跨越积水起始点，自由段与积水后约 10 个 tp 都有数据
    ts = lin(0.02, 10.0 * tp, 25)
    assert any(t < tp for t in ts) and any(t > 3.0 * tp for t in ts)
    obs = make_obs(ts, [rain_F(Ks_t, A_t, i, t) for t in ts])
    res = calibrate(obs, i=i)
    assert res.mode == "rainfall"
    assert rel(res.Ks, Ks_t) < 1e-8
    assert rel(res.A, A_t) < 1e-8
    assert res.ssr < 1e-18
    assert res.identifiability["status"] == "both_identifiable"
    kappa = res.identifiability["condition_number"]
    assert kappa <= KAPPA_IDENTIFIABLE_MAX
    params = res.to_dict()["parameters"]
    assert params["Ks"]["identifiable"] is True
    assert params["A"]["identifiable"] is True


# --------------------------------------------------------------------------- #
# 2. 纯积水段简并 → 只交组合，单参明确标为不可辨
# --------------------------------------------------------------------------- #

def test_ponded_only_degenerate_combination_only():
    Ks_t, A_t = CLAY
    ts = lin(0.05, 0.5, 20)  # F(end)/A ≈ 0.12，短历时强简并
    Fend = ponded_F(Ks_t, A_t, ts[-1])
    assert Fend / A_t < 0.2
    obs = make_obs(ts, [ponded_F(Ks_t, A_t, t) for t in ts])
    res = calibrate(obs)

    assert res.identifiability["status"] == "combination_only"
    kappa = res.identifiability["condition_number"]
    assert kappa > KAPPA_IDENTIFIABLE_MAX

    # 对外结构必须把单参标成不可辨，哪怕数值点恰好落在真值上
    params = res.to_dict()["parameters"]
    assert params["Ks"]["identifiable"] is False
    assert params["A"]["identifiable"] is False
    assert params["Ks"]["role"] == "calibrated"

    # 可辨识的组合方向上，组合值被高精度还原
    combo = res.identifiability["identifiable_combination"]
    w = combo["weights"]
    true_combo = math.exp(w["ln_Ks"] * math.log(Ks_t) + w["ln_A"] * math.log(A_t))
    assert rel(combo["value"], true_combo) < 1e-8

    # 简并方向的标准误远大于组合方向（有残差尺度时才有量级可比性；
    # 无噪声数据下验证方向结构：权重正交且组合方向沿 (±1,1)/sqrt2）
    assert abs(abs(w["ln_Ks"]) - 1.0 / math.sqrt(2.0)) < 1e-12
    assert abs(w["ln_A"] - 1.0 / math.sqrt(2.0)) < 1e-12

    # 结果里必须给出打破简并的处置建议
    assert "resolution_hint" in res.identifiability


def test_combination_value_invariant_across_starting_points():
    """解谷上的落点可随初值漂移，但可辨识组合必须稳定。"""
    Ks_t, A_t = CLAY
    ts = lin(0.05, 0.5, 20)
    obs = make_obs(ts, [ponded_F(Ks_t, A_t, t) for t in ts])

    combos = []
    for init in (None, {"Ks": 0.05, "A": 60.0}, {"Ks": 1.0, "A": 3.0}):
        res = calibrate(obs, initial=init)
        assert res.identifiability["status"] == "combination_only"
        combos.append(res.identifiability["identifiable_combination"]["value"])
    assert rel(combos[0], combos[1]) < 1e-8
    assert rel(combos[0], combos[2]) < 1e-8


# --------------------------------------------------------------------------- #
# 3/4. 固定其一 → 另一个直接可辨
# --------------------------------------------------------------------------- #

def test_fix_Ks_recovers_A():
    Ks_t, A_t = CLAY
    ts = lin(0.05, 0.5, 20)
    obs = make_obs(ts, [ponded_F(Ks_t, A_t, t) for t in ts])
    res = calibrate(obs, fixed={"Ks": Ks_t})
    assert rel(res.A, A_t) < 1e-9
    assert res.Ks == Ks_t  # 固定值原样回传
    assert res.identifiability["status"] == "identifiable_via_constraint"
    params = res.to_dict()["parameters"]
    assert params["Ks"]["role"] == "fixed"
    assert params["A"]["role"] == "calibrated"
    assert params["A"]["identifiable"] is True


def test_fix_A_recovers_Ks_in_rainfall():
    Ks_t, A_t = LOAM
    i = 3.0 * Ks_t
    tp = analyze_ponding(Ks_t, A_t, 1.0, i).tp
    ts = lin(0.02, 10.0 * tp, 25)
    obs = make_obs(ts, [rain_F(Ks_t, A_t, i, t) for t in ts])
    res = calibrate(obs, i=i, fixed={"A": A_t})
    assert rel(res.Ks, Ks_t) < 1e-9
    assert res.A == A_t
    assert res.identifiability["status"] == "identifiable_via_constraint"


# --------------------------------------------------------------------------- #
# 5. 小扰动 → 连续、SSR 抬升、仍收敛
# --------------------------------------------------------------------------- #

def test_perturbed_observations_continuous_and_still_converges():
    Ks_t, A_t = LOAM
    i = 3.0 * Ks_t
    tp = analyze_ponding(Ks_t, A_t, 1.0, i).tp
    ts = lin(0.02, 10.0 * tp, 25)
    clean = [rain_F(Ks_t, A_t, i, t) for t in ts]

    rows = []
    for eps in (0.0, 1e-4, 1e-3, 1e-2):
        noisy = [f * (1.0 + eps * math.sin(13.0 * k + 1.0)) for k, f in enumerate(clean)]
        res = calibrate(make_obs(ts, noisy), i=i)
        rows.append((eps, res.Ks, res.A, res.ssr))

    # SSR 随扰动幅度严格抬升
    for a, b in zip(rows, rows[1:]):
        assert b[3] > a[3]
        assert b[0] > a[0]
    # 参数随扰动连续漂移（1% 扰动时偏离真值不到 10%）
    _, Ks_n, A_n, ssr_n = rows[-1]
    assert rel(Ks_n, Ks_t) < 0.1
    assert rel(A_n, A_t) < 0.1
    # 所有扰动档都收敛
    assert all(r[3] < math.inf for r in rows)


# --------------------------------------------------------------------------- #
# 6. 不合模型形态的观测：报错或诚实大残差
# --------------------------------------------------------------------------- #

def test_oscillating_nonmonotonic_data_is_rejected():
    ts = lin(0.05, 6.0, 20)
    garbage = [3.0 + 2.0 * ((-1) ** k) + 0.1 * k for k in range(20)]
    obs = make_obs(ts, garbage)
    with pytest.raises(ConvergenceError) as exc:
        calibrate(obs)
    # 必须落到明确的标定失败错误码，而不是吐一组精小残差
    assert exc.value.code in ("calibration_not_converged", "degenerate_solution")


def test_mild_model_misfit_converges_to_honest_large_residual():
    """形态合理但不是 GA 曲线：允许收敛，但残差必须诚实地大。"""
    ts = lin(0.05, 6.0, 25)
    misfit = [2.5 * t ** 0.7 for t in ts]
    res = calibrate(make_obs(ts, misfit))
    # 残差量级与失配相当（归一化 RMSE 必须显著大于零）
    rmse = math.sqrt(res.ssr / len(ts))
    f_scale = misfit[-1]
    assert rmse / f_scale > 1e-3
    # 模型点单调（正向模型保证），且逐点残差之和等于 SSR
    model_Fs = [r["F_model"] for r in res.residuals]
    assert all(b >= a for a, b in zip(model_Fs, model_Fs[1:]))
    assert abs(sum(r["residual"] ** 2 for r in res.residuals) - res.ssr) < 1e-20


# --------------------------------------------------------------------------- #
# 7. 病态输入：迭代被压死 → 报错，绝不交半路参数
# --------------------------------------------------------------------------- #

def test_iteration_limit_is_honest_failure():
    Ks_t, A_t = LOAM
    i = 3.0 * Ks_t
    tp = analyze_ponding(Ks_t, A_t, 1.0, i).tp
    ts = lin(0.02, 10.0 * tp, 25)
    obs = make_obs(ts, [rain_F(Ks_t, A_t, i, t) for t in ts])
    with pytest.raises(ConvergenceError) as exc:
        calibrate(obs, i=i, max_iter=1)
    assert exc.value.code == "calibration_not_converged"
    # 错误结构必须交代清楚：试了几步、上限是多少、各起点结局
    assert exc.value.details["max_iter"] == 1
    assert exc.value.details["total_iterations"] >= 1
    assert exc.value.details["attempts"]


def test_oscillating_data_stall_is_honest_failure():
    # 同一组打转数据在每个起点都压不进判据：错误详情里带各起点尝试
    ts = lin(0.05, 6.0, 20)
    garbage = [3.0 + 2.0 * ((-1) ** k) + 0.1 * k for k in range(20)]
    with pytest.raises(ConvergenceError) as exc:
        calibrate(make_obs(ts, garbage))
    assert exc.value.details.get("attempts"), "必须回报各起点尝试记录"
    assert all(a["status"] == "failed" for a in exc.value.details["attempts"])


def test_all_free_segment_observations_have_no_signal():
    Ks_t, A_t = LOAM
    i = 10.0 * Ks_t
    tp = analyze_ponding(Ks_t, A_t, 1.0, i).tp
    ts = lin(0.001, 0.5 * tp, 10)  # 全部在积水时刻之前：F=i·t 与参数无关
    obs = make_obs(ts, [rain_F(Ks_t, A_t, i, t) for t in ts])
    with pytest.raises(ConvergenceError) as exc:
        calibrate(obs, i=i)
    assert exc.value.code == "no_parameter_signal"


# --------------------------------------------------------------------------- #
# 8. 取消语义
# --------------------------------------------------------------------------- #

def test_cancel_raises_without_half_result():
    Ks_t, A_t = LOAM
    ts = lin(0.05, 6.0, 40)
    obs = make_obs(ts, [ponded_F(Ks_t, A_t, t) for t in ts])
    state = {"n": 0}

    def cancel_soon():
        state["n"] += 1
        return state["n"] > 2

    with pytest.raises(CalibrationCancelled) as exc:
        calibrate(obs, should_cancel=cancel_soon)
    assert exc.value.iterations_done >= 0


# --------------------------------------------------------------------------- #
# 9. 观测校验：坏数据挡在迭代之前
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "raw,why",
    [
        ([{"t": 0.1, "F": 0.2}, {"t": 0.2, "F": 0.3}], "too_few"),
        ([{"t": 0.0, "F": 0.0}, {"t": 0.2, "F": 0.1}, {"t": 0.2, "F": 0.3}],
         "non_increasing"),
        ([{"t": 0.0, "F": 0.0}, {"t": 0.2, "F": -0.1}, {"t": 0.3, "F": 0.2}],
         "negative_F"),
        ([{"t": 0.0, "F": 0.5}, {"t": 0.2, "F": 0.6}, {"t": 0.3, "F": 0.7}],
         "nonzero_origin"),
        ([{"t": -0.1, "F": 0.0}, {"t": 0.2, "F": 0.1}, {"t": 0.3, "F": 0.2}],
         "negative_t"),
        ([{"t": 0.0, "F": 0.0}, {"t": 0.2, "F": 0.0}, {"t": 0.3, "F": 0.0}],
         "all_zero"),
    ],
)
def test_bad_observations_rejected_before_iteration(raw, why):
    with pytest.raises(ValidationError) as exc:
        validate_observations(raw)
    assert exc.value.code == "invalid_observation"


def test_parallel_array_form_accepted():
    Ks_t, A_t = SAND
    ts = lin(0.05, 6.0, 10)
    Fs = [ponded_F(Ks_t, A_t, t) for t in ts]
    obs = validate_observations({"t": ts, "F": Fs})
    assert len(obs) == 10
    res = calibrate(obs)
    assert rel(res.Ks, Ks_t) < 1e-7


def test_fixed_param_validation():
    ts = lin(0.05, 2.0, 8)
    obs = make_obs(ts, [ponded_F(1.0, 5.0, t) for t in ts])
    with pytest.raises(ValidationError):
        calibrate(obs, fixed={"psi": 5.0})
    with pytest.raises(ValidationError):
        calibrate(obs, fixed={"Ks": 0.0})


def rel(a, b):
    return abs(a / b - 1.0)
