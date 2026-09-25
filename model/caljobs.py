"""标定后台作业的生命周期（与点列作业同构，但账完全分开）。

作业在有界线程池里跑 :func:`model.calibration.calibrate`，状态机

    queued -> running -> completed
                       \\-> cancelled   （中途取消，绝不交半成品参数）
                       \\-> failed      （不收敛 / 无信号 / 病态输入）

每份作业各自持有自己的观测副本、迭代轨迹与 LM 临时量——这些全部活在
:func:`model.calibration.calibrate` 的栈帧里，多份标定并发时互不串账。
标定成功且调用方要求建档时，由本层把结果写入工况仓库；只有参数可辨识
（both_identifiable / identifiable_via_constraint）的结果才允许固化，
组合可辨的结果建档等于把虚假精度存盘，一律拒绝并在结果里说明。
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Deque

from .calibration import CalibrationCancelled, CalibrationResult, calibrate
from .errors import ServiceError
from .profiles import ProfileStore


def _json_safe(obj: Any) -> Any:
    """把 inf/NaN 换成 None，保证吐出去的是严格合法的 JSON。

    极端简并（κ=∞、单参标准误无穷）是数学上真实的结果，但 Infinity 不是
    合法 JSON token，对外统一用 null 表示“无穷大/未定量”。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


@dataclass
class _CalibrationJob:
    id: str
    spec: dict[str, Any]
    state: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    progress_attempt: int = 0
    progress_iteration: int = 0
    progress_ssr: float | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "job_id": self.id,
            "state": self.state,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": {
                "attempt": self.progress_attempt,
                "iteration": self.progress_iteration,
                "current_ssr": self.progress_ssr,
                "max_iter": self.spec["options"].get("max_iter"),
            },
            "request": self.spec["request_echo"],
        }
        if self.state == "completed":
            body["complete"] = True
            body["result"] = self.result
        elif self.state == "cancelled":
            # 取消的标定永远不交半成品参数
            body["complete"] = False
            body["result"] = None
            body["iterations_before_cancel"] = self.progress_iteration
        elif self.state == "failed":
            body["complete"] = False
            body["result"] = None
            body["error"] = self.error
        return body


class CalibrationManager:
    """进程内标定作业管理器。"""

    def __init__(self, store: ProfileStore, workers: int = 2,
                 retention: int = 500) -> None:
        self._store = store
        self._pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ga-calibration"
        )
        self._jobs: dict[str, _CalibrationJob] = {}
        self._recent: Deque[str] = deque(maxlen=retention)
        self._lock = threading.Lock()
        # 默认标定器；测试可替换为确定性替身（如立刻抛 CalibrationCancelled）
        self.runner: Callable[..., CalibrationResult] = calibrate

    def set_runner(self, runner: Callable[..., CalibrationResult]) -> None:
        self.runner = runner

    def submit(self, spec: dict[str, Any], *, job_id: str | None = None,
               runner: Callable[..., CalibrationResult] | None = None) -> str:
        job = _CalibrationJob(id=job_id or uuid.uuid4().hex, spec=spec)
        self._pool.submit(self._run, job, runner or self.runner)
        with self._lock:
            self._jobs[job.id] = job
            self._recent.append(job.id)
        return job.id

    def _run(self, job: _CalibrationJob,
             runner: Callable[..., CalibrationResult]) -> None:
        job.started_at = time.time()
        job.state = "running"

        def on_progress(attempt: int, iteration: int, ssr: float) -> None:
            job.progress_attempt = attempt
            job.progress_iteration = iteration
            job.progress_ssr = ssr

        opts = job.spec["options"]
        try:
            result = runner(
                job.spec["observations"],
                i=job.spec.get("i"),
                fixed=job.spec.get("fixed"),
                initial=job.spec.get("initial"),
                max_iter=opts.get("max_iter", 100),
                gtol=opts.get("gtol", 1e-10),
                xtol=opts.get("xtol", 1e-12),
                ftol=opts.get("ftol", 1e-14),
                should_cancel=job.cancel_event.is_set,
                progress=on_progress,
            )
        except CalibrationCancelled as exc:
            job.state = "cancelled"
            job.finished_at = time.time()
            job.progress_iteration = exc.iterations_done
            job.result = None
            return
        except ServiceError as exc:
            job.state = "failed"
            job.finished_at = time.time()
            job.error = {"code": exc.code, "reason": exc.reason,
                         "details": exc.details}
            return
        except Exception as exc:  # pragma: no cover - 防御性
            job.state = "failed"
            job.finished_at = time.time()
            job.error = {"code": "internal_error", "reason": str(exc)}
            return

        body = _json_safe(result.to_dict())
        body["profile_save"] = self._maybe_save_profile(job, result)
        job.result = body
        job.state = "completed"
        job.finished_at = time.time()

    def _maybe_save_profile(self, job: _CalibrationJob,
                            result: CalibrationResult) -> dict[str, Any]:
        """按要求把可辨识的标定结果建档；不可建档时说明原因，不硬存。"""
        save_as = job.spec.get("save_as")
        if not save_as:
            return {"requested": False, "saved": False}
        if not result.identifiable:
            return {
                "requested": True,
                "saved": False,
                "reason": (
                    "本次标定为组合可辨：Ks 与 A 各自定不准，"
                    "把解谷上的任意一点固化成工况会存下虚假精度，故不建档。"
                    "请按 identifiability.resolution_hint 补信息后重标。"
                ),
            }
        split = job.spec["split"]  # 提交时已校验恰有一个键
        if "delta_theta" in split:
            delta_theta = split["delta_theta"]
            psi = result.A / delta_theta
        else:
            psi = split["psi"]
            delta_theta = result.A / psi
            if not (0.0 < delta_theta <= 1.0):
                return {
                    "requested": True,
                    "saved": False,
                    "reason": (
                        f"由 A/psi 反推的 delta_theta={delta_theta:g} 不在 (0,1]，"
                        "给定的 psi 与标定出的 A 不自洽，未建档。"
                    ),
                }
        profile = self._store.create(
            save_as, result.Ks, psi, delta_theta,
            description=(
                f"反演标定建档（job {job.id}）：SSR={result.ssr:.6g}，"
                f"{result.n_observations} 个观测，模式 {result.mode}，"
                f"可辨识性 {result.identifiability['status']}。"
            ),
            overwrite=True,
        )
        return {"requested": True, "saved": True, "profile": profile.to_dict()}

    def get(self, job_id: str) -> _CalibrationJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ServiceError("job_not_found", f"标定作业 {job_id!r} 不存在",
                               status_code=404)
        return job

    def status(self, job_id: str) -> dict[str, Any]:
        return self.get(job_id).snapshot()

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        job.cancel_event.set()
        return {"job_id": job_id, "cancel_requested": True, "state": job.state}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
