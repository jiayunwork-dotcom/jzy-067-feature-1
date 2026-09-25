"""反演标定的数值层测试。

钉死需求里的几条因果：

1. 无噪声观测正向生成、反向标定，可辨识前提下高精度还原已知参数；
2. 只给积水段：判成「组合可辨、单参不可辨」，绝不虚报精确单参；
3. 含自由入渗段的降雨观测：简并打破，Ks 与 A 分别还原；
4. 掺小扰动：参数随扰动连续、SSE 单调抬升但仍收敛；
5. 形态崩坏的观测（非单调乱跳）：要么诚实给出大残差，要么明判失败；
6. 病态输入（迭代被故意限死）：落报错分支，绝不吐最后一步参数；
7. 固定其一可打破简并；纯自由段观测无约束必须报错。
"""

from __future__ import annotations

import math
import random

import pytest

from model.calibration import (
    CalibrationCancelled,
    CalibrationError,
    Observation,
    model_cumulative,
    run_calibration,
    validate_observations,
)
from model.errors import ValidationError
from model.infiltration import analyze_ponding, state_at_time

TRUE = dict(Ks=1.09, psi=11.01, delta_theta=0.434)
A_TRUE = TRUE["psi"] * TRUE["delta_theta"]
I = 5.0


def _ponded_obs(ts):
    return [{"t": t,
             "F": state_at_time(TRUE["Ks"], TRUE["psi"], TRUE["delta_theta"],
                                t, already_ponded=True).F}
            for t in ts]


def _rain_obs(ts, i=I):
    return [{"t": t,
             "F": state_at_time(TRUE["Ks"], TRUE["psi"], TRUE["delta_theta"],
                                t, i=i).F}
            for t in ts]


# --------------------------------------------------------------------------- #
# 入箱校验：挡在迭代之前
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw", [
    [],
    [{"t": 0.0, "F": 0.0}],
    [{"t": 0.0, "F": 0.0}, {"t": 1.0, "F": 1.0}],
])
def test_too_few_observations_rejected(raw):
    with pytest.raises(ValidationError) as exc:
        validate_observations(raw)
    assert exc.value.code == "too_few_observations"


def test_times_must_be_strictly_increasing():
    with pytest.raises(ValidationError) as exc:
        validate_observations([
            {"t": 1.0, "F": 1.0}, {"t": 1.0, "F": 2.0}, {"t": 2.0, "F": 3.0}])
    assert exc.value.code == "times_not_increasing"
    with pytest.raises(ValidationError) as exc:
        validate_observations([
            {"t": 2.0, "F": 1.0}, {"t": 1.0, "F": 2.0}, {"t": 0.0, "F": 3.0}])
    assert exc.value.code == "times_not_increasing"


def test_negative_t_or_F_rejected():
    with pytest.raises(ValidationError):
        validate_observations([{"t": -1.0, "F": 1.0},
                               {"t": 1.0, "F": 2.0}, {"t": 2.0, "F": 3.0}])
    with pytest.raises(ValidationError) as exc:
        validate_observations([{"t": 0.0, "F": 0.0},
                               {"t": 1.0, "F": -0.5}, {"t": 2.0, "F": 3.0}])
    assert exc.value.code == "negative_infiltration"


def test_bad_shapes_and_nonfinite_rejected():
    with pytest.raises(ValidationError):
        validate_observations([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        validate_observations([{"t": 0.0, "F": float("nan")},
                               {"t": 1.0, "F": 2.0}, {"t": 2.0, "F": 3.0}])
    with pytest.raises(ValidationError):
        validate_observations([{"t": True, "F": 1.0},
                               {"t": 1.0, "F": 2.0}, {"t": 2.0, "F": 3.0}])
    with pytest.raises(ValidationError):
        run_calibration([{"t": 0.0, "F": 0.0},
                         {"t": 1.0, "F": 0.0}, {"t": 2.0, "F": 0.0}])


# --------------------------------------------------------------------------- #
# 1) 无噪声积水段 → 组合可辨、单参不可辨
# --------------------------------------------------------------------------- #

def test_ponded_only_identifies_combination_not_individuals():
    obs = _ponded_obs([0.02 * k for k in range(1, 16)])
    res = run_calibration(obs)
    assert res.converged is True
    ident = res.identifiability
    assert ident["status"] == "combination_only"
    # 顶层答案绝不吐虚假单参值
    assert res.fitted["Ks"] is None
    assert res.fitted["A"] is None
    # 脊线上的点只作诊断
    assert res.ridge_point is not None and res.ridge_point["Ks"] > 0
    # 可辨识组合与弱方向必须回报
    combo = ident["identifiable_combination"]
    assert combo["form"] == "Ks^a * A^b"
    assert combo["value"] > 0.0
    weak = ident["weak_direction"]
    assert abs(math.hypot(weak["log_Ks"], weak["log_A"]) - 1.0) < 1e-12
    # 无噪声数据本身拟合极好
    assert res.nrmse < 1e-8
    assert res.sse >= 0.0
    # 残差逐点可复核，且重算模型值与观测一致
    for row in res.residuals:
        assert abs(row["F_model"] - row["F_observed"] - row["residual"]) < 1e-12
        assert abs(row["residual"]) < 1e-8 * max(1.0, row["F_observed"])


def test_ponded_only_early_window_weak_direction_holds_KsA_product():
    # 极早段 F≈√(2 A Ks t)：弱方向应近似 (1,-1)/√2
    obs = _ponded_obs([1e-6 + k * 6e-7 for k in range(16)])
    res = run_calibration(obs)
    assert res.identifiability["status"] == "combination_only"
    weak = res.identifiability["weak_direction"]
    assert abs(weak["log_Ks"] - (-weak["log_A"])) < 5e-2
    # 回报的 Ks·A 必须贴近真值
    assert res.identifiability["Ks_times_A"] == pytest.approx(
        TRUE["Ks"] * A_TRUE, rel=1e-4)


# --------------------------------------------------------------------------- #
# 2) 固定其一 → 另一项高精度还原
# --------------------------------------------------------------------------- #

def test_fixed_A_recovers_Ks_at_high_precision():
    obs = _ponded_obs([0.02 * k for k in range(1, 16)])
    res = run_calibration(obs, fixed={"A": A_TRUE})
    assert res.identifiability["status"] == "individual"
    assert res.fitted["Ks"] == pytest.approx(TRUE["Ks"], rel=1e-9)


def test_fixed_Ks_recovers_A_at_high_precision():
    obs = _ponded_obs([0.02 * k for k in range(1, 16)])
    res = run_calibration(obs, fixed={"Ks": TRUE["Ks"]})
    assert res.identifiability["status"] == "individual"
    assert res.fitted["A"] == pytest.approx(A_TRUE, rel=1e-9)


# --------------------------------------------------------------------------- #
# 3) 含自由入渗段的降雨观测 → 两个量分别还原
# --------------------------------------------------------------------------- #

def test_rainfall_with_free_phase_recovers_both():
    info = analyze_ponding(TRUE["Ks"], TRUE["psi"], TRUE["delta_theta"], I)
    ts = [0.02 * k for k in range(1, 41)]
    assert any(t < info.tp for t in ts) and any(t > info.tp for t in ts)
    res = run_calibration(_rain_obs(ts), i=I)
    assert res.identifiability["status"] == "individual"
    assert res.fitted["Ks"] == pytest.approx(TRUE["Ks"], rel=1e-8)
    assert res.fitted["A"] == pytest.approx(A_TRUE, rel=1e-8)
    assert res.mode == "rainfall"
    cov = res.identifiability["phase_coverage"]
    assert cov["n_free"] >= 1 and cov["n_ponded"] >= 2


def test_rainfall_given_but_only_ponded_points_still_degenerate():
    # 给了 i，但所有点都在积水后：tp 没被夹住，仍只能定组合
    info = analyze_ponding(TRUE["Ks"], TRUE["psi"], TRUE["delta_theta"], I)
    ts = [info.tp * (1.5 + 0.2 * k) for k in range(12)]
    res = run_calibration(_rain_obs(ts), i=I)
    assert res.identifiability["status"] == "combination_only"
    assert res.fitted["Ks"] is None and res.fitted["A"] is None


def test_all_free_phase_observations_rejected_as_uninformative():
    obs = [{"t": 0.001 * k, "F": I * 0.001 * k} for k in range(1, 8)]
    with pytest.raises(CalibrationError) as exc:
        run_calibration(obs, i=I)
    assert exc.value.code == "uninformative_observations"


def test_fixed_Ks_ge_rainfall_rejected():
    obs = _rain_obs([0.02 * k for k in range(1, 10)])
    with pytest.raises(ValidationError):
        run_calibration(obs, i=I, fixed={"Ks": I})


# --------------------------------------------------------------------------- #
# 4) 掺噪连续：参数随扰动走、SSE 抬升但仍收敛
# --------------------------------------------------------------------------- #

def test_parameters_move_continuously_under_noise_and_sse_rises():
    ts = [0.02 * k for k in range(1, 41)]
    clean = _rain_obs(ts)
    r0 = run_calibration(clean, i=I)
    prev_Ks, prev_A, prev_sse = r0.fitted["Ks"], r0.fitted["A"], r0.sse
    for seed, sigma in enumerate((0.0005, 0.002, 0.004), start=1):
        rng = random.Random(seed)
        noisy = [{"t": d["t"], "F": d["F"] + rng.gauss(0, sigma)} for d in clean]
        res = run_calibration(noisy, i=I)
        assert res.converged is True
        assert res.identifiability["status"] == "individual"
        # 小扰动下参数不跳变（相对漂移远小于量级本身）
        assert abs(res.fitted["Ks"] - TRUE["Ks"]) / TRUE["Ks"] < 0.1
        assert abs(res.fitted["A"] - A_TRUE) / A_TRUE < 0.1
        # 扰动越大，SSE 不应比无噪声小（允许数值噪声），整体抬升
        assert res.sse >= prev_sse * (1.0 - 1e-6)
        prev_Ks, prev_A, prev_sse = res.fitted["Ks"], res.fitted["A"], res.sse
    assert prev_sse > r0.sse


# --------------------------------------------------------------------------- #
# 5) 形态崩坏的观测：诚实的大残差或明判失败，绝不假装精确
# --------------------------------------------------------------------------- #

def test_nonmonotonic_junk_either_poor_fit_or_honest_failure():
    rng = random.Random(42)
    junk = [{"t": 0.1 * k, "F": (k % 3) * 0.7 + rng.random() * 0.3}
            for k in range(1, 14)]
    try:
        res = run_calibration(junk)
    except CalibrationError as exc:
        # 明判失败也接受，但错误结构必须说清原因，且不带标定值
        assert exc.code in ("not_converged", "uninformative_observations")
        assert not hasattr(exc, "fitted")
        return
    # 若收敛了：必须诚实暴露拟合很差，绝不能宣称 good
    assert res.quality == "poor"
    assert res.nrmse > 0.1
    assert res.sse > 0.0


def test_constant_F_data_cannot_be_fit_as_good():
    obs = [{"t": 0.1 * k, "F": 1.0} for k in range(1, 12)]
    try:
        res = run_calibration(obs)
    except CalibrationError:
        return
    assert res.quality in ("fair", "poor")
    assert res.nrmse > 0.05


# --------------------------------------------------------------------------- #
# 6) 病态输入：迭代压不进判据 → 报错，绝不吐半成品
# --------------------------------------------------------------------------- #

def test_iteration_budget_exhausted_raises_and_withholds_parameters():
    ts = [0.02 * k for k in range(1, 41)]
    with pytest.raises(CalibrationError) as exc:
        run_calibration(_rain_obs(ts), i=I, max_iter=1)
    err = exc.value
    assert err.code == "not_converged"
    assert err.status_code == 422
    assert err.details["best_sse_reached"] > 0.0
    # 错误细节里可以带诊断量，但调用方拿不到“结果对象”
    assert err.details["iterations"] >= 1


def test_cancel_during_iteration_never_returns_half_parameters():
    obs = _ponded_obs([0.02 * k for k in range(1, 16)])

    def cancel_now():
        return True

    with pytest.raises(CalibrationCancelled):
        run_calibration(obs, should_cancel=cancel_now)


# --------------------------------------------------------------------------- #
# 7) 迭代轨迹、可复核量
# --------------------------------------------------------------------------- #

def test_result_carries_auditable_iteration_trace_and_residuals():
    ts = [0.02 * k for k in range(1, 41)]
    res = run_calibration(_rain_obs(ts), i=I)
    assert res.iterations >= 1
    assert res.total_iterations >= res.iterations
    assert res.function_evaluations >= 1
    assert len(res.residuals) == len(ts)
    # SSE 与逐点残差自洽
    assert res.sse == pytest.approx(
        sum(r["residual"] ** 2 for r in res.residuals), rel=1e-10)
    # 轨迹中 SSE 单调不增（只收接受步）
    accepted = [e for e in res.trace if e["accepted"]]
    sses = [e["sse"] for e in accepted]
    assert all(b <= a + 1e-18 for a, b in zip(sses, sses[1:]))
    # 残差点重算正向：F_model 与 model_cumulative 一致
    for row in res.residuals:
        Fm = model_cumulative(res.fitted["Ks"], res.fitted["A"],
                              row["t"], I)
        assert Fm == pytest.approx(row["F_model"], rel=1e-10)


def test_explicit_initial_profile_values_used(tmp_path):
    # 显式初值远离真值也应收敛到同一个最小点（多起点 + 稳健性）
    obs = _rain_obs([0.02 * k for k in range(1, 41)])
    res = run_calibration(obs, i=I, initial={"Ks": 0.3, "A": 1.0})
    assert res.fitted["Ks"] == pytest.approx(TRUE["Ks"], rel=1e-7)
    assert res.fitted["A"] == pytest.approx(A_TRUE, rel=1e-7)


def test_observation_dataclass_input_accepted():
    obs = [Observation(t=0.02 * k,
                       F=state_at_time(TRUE["Ks"], TRUE["psi"], TRUE["delta_theta"],
                                       0.02 * k, already_ponded=True).F)
           for k in range(1, 16)]
    res = run_calibration(obs, fixed={"A": A_TRUE})
    assert res.fitted["Ks"] == pytest.approx(TRUE["Ks"], rel=1e-9)
