"""历时点列的分段推进（可取消）。

给定时间网格与土壤/供水工况，逐点推进并给出 (t, F, f, phase) 点列。
积水工况下每个积水点都要解一次隐式方程，采用上一步的 F 做牛顿初值提示
（F 随时间单调增，上一步的解对新的右端恰好在根左侧）。

这是可取消的作业：每个点推进前检查 ``should_cancel``；一旦被取消，
立即抛 :class:`HydrographCancelled`，由作业层决定如何收尾——
**绝不能把没算完的点列当完整结果交出**。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Any

from .infiltration import (
    InfiltrationState,
    analyze_ponding,
    state_at_time,
)
from .solver import (
    DEFAULT_ABS_TOL,
    DEFAULT_MAX_ITER,
    DEFAULT_REL_TOL,
)

CancelCheck = Callable[[], bool]


class HydrographCancelled(Exception):
    """点列推进被外部取消。带已算部分仅供诊断，绝不当完整结果。"""

    def __init__(self, partial_points: list[dict[str, Any]], last_index: int) -> None:
        super().__init__("入渗点列作业已取消")
        self.partial_points = partial_points
        self.last_index = last_index


@dataclass(frozen=True)
class Hydrograph:
    """完整点列结果（只有正常跑完才会产出）。"""

    points: list[dict[str, Any]]
    n_points: int
    duration: float
    mode: str
    i: float | None
    ponding: dict[str, Any]
    phase_switches: list[dict[str, float | str]] = field(default_factory=list)
    cancelled: bool = False
    complete: bool = True


def build_time_grid(duration: float, n_points: int) -> list[float]:
    """在 [0, duration] 上均匀取 n_points 个点。"""
    if n_points < 2:
        raise ValueError("n_points 至少为 2")
    step = duration / (n_points - 1)
    return [step * k for k in range(n_points)]


def _state_to_point(st: InfiltrationState) -> dict[str, Any]:
    return {
        "t": st.t,
        "F": st.F,
        "f": st.f,
        "f_capacity": st.f_capacity,
        "phase": st.phase,
        "residual": st.residual,
        "tolerance": st.tolerance,
        "iterations": st.iterations,
    }


def run_hydrograph(
    Ks: float,
    psi: float,
    delta_theta: float,
    duration: float,
    *,
    n_points: int = 101,
    i: float | None = None,
    already_ponded: bool = False,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    max_iter: int = DEFAULT_MAX_ITER,
    should_cancel: CancelCheck | None = None,
    progress: Callable[[int, int], None] | None = None,
    time_grid: list[float] | None = None,
) -> Hydrograph:
    """分段推进完整点列。

    Args:
        duration: 总历时（时间），非负。
        n_points: 点列点数（含 t=0 与 t=duration）。
        i: 降雨强度；为 None 时必须 already_ponded。
        already_ponded: 自始积水声明。
        should_cancel: 返回 True 时作业即刻取消。
        progress: 每完成一个点回调 (done, total)。
        time_grid: 显式给定的非负递增时间网格（测试/高级用法）。
    """
    if i is None and not already_ponded:
        raise ValueError("必须给出降雨强度 i 或声明 already_ponded")

    if time_grid is None:
        if duration < 0.0:
            raise ValueError("duration 不能为负")
        grid = build_time_grid(duration, n_points)
    else:
        grid = list(time_grid)
        if any(grid[k] < grid[k - 1] for k in range(1, len(grid))):
            raise ValueError("时间网格必须单调不减")

    info = None
    if not already_ponded:
        info = analyze_ponding(Ks, psi, delta_theta, i)  # type: ignore[arg-type]

    points: list[dict[str, Any]] = []
    switches: list[dict[str, float | str]] = []
    F_hint: float | None = None
    prev_phase: str | None = None

    total = len(grid)
    for idx, t in enumerate(grid):
        if should_cancel is not None and should_cancel():
            raise HydrographCancelled(points, idx - 1)

        st = state_at_time(
            Ks, psi, delta_theta, t,
            i=None if already_ponded else i,
            already_ponded=already_ponded,
            abs_tol=abs_tol, rel_tol=rel_tol, max_iter=max_iter,
            F_hint=F_hint,
        )
        if prev_phase is not None and st.phase != prev_phase:
            switches.append({"t": t, "from": prev_phase, "to": st.phase})
        prev_phase = st.phase
        points.append(_state_to_point(st))
        F_hint = st.F  # 下一点的牛顿初值提示

        if progress is not None:
            progress(idx + 1, total)

    ponding_dict: dict[str, Any]
    if info is None:
        ponding_dict = {"will_pond": True, "declared": True}
    else:
        ponding_dict = {
            "will_pond": info.will_pond,
            "i": info.i,
            "tp": info.tp,
            "Fp": info.Fp,
            "equivalent_time": info.equivalent_time,
            "explanation": info.explanation,
        }

    return Hydrograph(
        points=points,
        n_points=len(points),
        duration=grid[-1] if grid else 0.0,
        mode="ponded_from_zero" if already_ponded else "rainfall",
        i=None if already_ponded else i,
        ponding=ponding_dict,
        phase_switches=switches,
        complete=True,
    )
